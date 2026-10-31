"""SeasonWise Ride & Travel Guide - chatbot backend.

How answers are produced
- AI mode (ANTHROPIC_API_KEY set): Claude answers any question in full. It calls tools for
  live data: weather (Open-Meteo), nearby petrol bunks / mechanics / hospitals etc. (OpenStreetMap),
  and SeasonWise's own destination data.
- Basic mode (no key): the chat still answers weather and nearby-service questions using the same
  tools, and clearly explains that general questions need the AI key.

Entry point used by app.py: chat(messages, location) -> reply text. It never raises.
"""
import json
import math
import os
import re
from datetime import date
from pathlib import Path

import requests
from weather_service import (
    AmbiguousLocationError,
    InvalidWeatherRequest,
    LocationNotFoundError,
    WeatherServiceError,
    geocode as geocode_location,
    get_weather as fetch_weather,
    validate_coordinates,
)

HERE = Path(__file__).parent
DESTINATIONS = json.loads((HERE / "data" / "destinations.json").read_text(encoding="utf-8"))

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-5-5").strip()
USER_AGENT = "SeasonWise-student-project/1.0"
MAX_MESSAGES = 20
MAX_CHARS = 2000


def ai_enabled():
    return bool(ANTHROPIC_KEY and CHAT_MODEL)


def ai_status():
    if not ANTHROPIC_KEY:
        return {"ai": False, "model": CHAT_MODEL, "reason": "ANTHROPIC_API_KEY is not configured."}
    if not CHAT_MODEL:
        return {"ai": False, "model": "", "reason": "CHAT_MODEL is empty."}
    return {"ai": True, "model": CHAT_MODEL, "reason": None}


class ChatError(Exception):
    """An error with a message that is safe to show the user."""


# ------------------------------------------------------------------ small helpers

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       66: "freezing rain", 67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
       77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers", 85: "snow showers",
       86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm"}

def geocode(name):
    """Resolve a city or town using the shared Open-Meteo geocoder."""
    place = geocode_location(name)
    return {**place, "lat": place["latitude"], "lng": place["longitude"]}


# ------------------------------------------------------------------ weather

def get_weather(lat, lng, days=7):
    return fetch_weather(latitude=lat, longitude=lng, days=days)


# ------------------------------------------------------------------ nearby services (OpenStreetMap)

SERVICES = {
    "fuel": ['"amenity"="fuel"'],
    "mechanic": ['"shop"="motorcycle_repair"', '"shop"="car_repair"', '"amenity"="car_repair"'],
    "tyre": ['"shop"="tyres"'],
    "ev_charging": ['"amenity"="charging_station"'],
    "hospital": ['"amenity"="hospital"', '"amenity"="clinic"'],
    "pharmacy": ['"amenity"="pharmacy"'],
    "atm": ['"amenity"="atm"', '"amenity"="bank"'],
    "food": ['"amenity"="restaurant"', '"amenity"="cafe"', '"amenity"="fast_food"'],
    "stay": ['"tourism"="hotel"', '"tourism"="guest_house"', '"tourism"="hostel"'],
    "toilet": ['"amenity"="toilets"'],
}
SERVICE_LABEL = {"fuel": "petrol/diesel bunk", "mechanic": "bike or car mechanic", "tyre": "tyre shop",
                 "ev_charging": "EV charging point", "hospital": "hospital or clinic", "pharmacy": "pharmacy",
                 "atm": "ATM or bank", "food": "restaurant or cafe", "stay": "hotel or guesthouse",
                 "toilet": "toilet"}

_nearby_cache = {}


def find_nearby(service, lat, lng, radius_km=15):
    if service not in SERVICES:
        raise ChatError(f"Unknown service '{service}'.")
    lat, lng = validate_coordinates(lat, lng)
    try:
        radius_km = float(radius_km or 15)
    except (TypeError, ValueError) as exc:
        raise ChatError("Search radius must be a number from 1 to 50 km.") from exc
    if not math.isfinite(radius_km) or not 1 <= radius_km <= 50:
        raise ChatError("Search radius must be from 1 to 50 km.")
    lat_r, lng_r, rad = round(lat, 3), round(lng, 3), int(radius_km * 1000)
    cache_key = (service, lat_r, lng_r, rad)
    if cache_key not in _nearby_cache:
        parts = "".join(f"nwr[{f}](around:{rad},{lat_r},{lng_r});" for f in SERVICES[service])
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": f"[out:json][timeout:25];({parts});out center tags 80;"},
                          headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
            raise ChatError("The nearby-places service returned invalid data.")
        _nearby_cache[cache_key] = payload["elements"]
    places = []
    for el in _nearby_cache[cache_key]:
        tags = el.get("tags", {})
        center = el.get("center") or {}
        plat = el.get("lat", center.get("lat"))
        plng = el.get("lon", center.get("lon"))
        if plat is None or plng is None:
            continue
        item = {
            "name": tags.get("name") or tags.get("brand") or f"Unnamed {SERVICE_LABEL[service]}",
            "distance_km": round(haversine_km(lat, lng, plat, plng), 1),
            "opening_hours": tags.get("opening_hours"),
            "phone": tags.get("phone") or tags.get("contact:phone"),
        }
        if service == "fuel":
            fuels = sorted(k.split(":", 1)[1].replace("_", " ") for k, v in tags.items()
                           if k.startswith("fuel:") and v == "yes")
            item["fuel_types"] = fuels or None
        places.append(item)
    places.sort(key=lambda x: x["distance_km"])
    return {"service": SERVICE_LABEL[service], "radius_km": radius_km, "count": len(places),
            "places": places[:8],
            "note": "Data comes from OpenStreetMap volunteers and can be incomplete. Call ahead to confirm."}


# ------------------------------------------------------------------ SeasonWise destination data

def destination_info(name):
    from app import TRAVEL_NOTES, travel_info  # imported here so app.py has finished loading
    key = name.strip().lower()
    for d in DESTINATIONS:
        if d["name"].lower() == key:
            return {"name": d["name"], "state": d["state"], "region": d["region"], "kind": d["kind"],
                    "best_months": d["best"], "rainy_months": d["rainy_months"],
                    "description": d["description"], "rating": d["rating"],
                    "lat": d["lat"], "lng": d["lon"], "travel": travel_info(d)}
    place = geocode(name)
    return {
        "name": place["name"],
        "state": place.get("admin1"),
        "country": place.get("country"),
        "lat": place["lat"],
        "lng": place["lng"],
        "note": "This place is not in SeasonWise's curated destination database. No rating or seasonal score is available.",
    }


# ------------------------------------------------------------------ tools

TOOLS = [
    {"name": "get_weather",
     "description": "Current modelled weather and a date-specific forecast of up to 16 days for any city or town. "
                    "Use a place name, or coordinates if the user shared their location. Never describe seasonal "
                    "estimates as a forecast.",
     "input_schema": {"type": "object", "properties": {
         "place": {"type": "string", "description": "City or town name, e.g. 'Leh, India'"},
         "start_date": {"type": "string", "description": "Optional trip start date, YYYY-MM-DD"},
         "end_date": {"type": "string", "description": "Optional trip end date, YYYY-MM-DD"},
         "days": {"type": "integer", "description": "Forecast duration, 1 to 16 days"}}}},
    {"name": "find_nearby",
     "description": "Find real nearby places from OpenStreetMap. service: fuel (petrol/diesel bunks), "
                    "mechanic (bike/car repair), tyre, ev_charging, hospital, pharmacy, atm, food, stay, toilet. "
                    "Search around a place name, or around the user's shared location if no place is given.",
     "input_schema": {"type": "object", "properties": {
         "service": {"type": "string", "enum": list(SERVICES)},
         "place": {"type": "string"},
         "radius_km": {"type": "number", "description": "default 15, max 50"}},
         "required": ["service"]}},
    {"name": "destination_info",
     "description": "SeasonWise details for a destination: best months, rainy months, description, how to "
                    "get there (flight/train/car/bike) and where to stay.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
]


def _location_for(args, user_loc):
    """Work out the point to search around: a named place, else the user's shared location."""
    place_name = args.get("place") or args.get("location")
    if place_name:
        g = geocode(place_name)
        return g["lat"], g["lng"], g["name"], g
    if user_loc:
        if user_loc.get("place"):
            g = geocode(user_loc["place"])
            return g["lat"], g["lng"], g["name"], g
        return user_loc["lat"], user_loc["lng"], user_loc.get("name", "your location"), None
    raise ChatError("NO_LOCATION")


def run_tool(name, args, user_loc):
    """Run one tool. Errors come back as a dict so the model can explain them to the user."""
    try:
        if name == "get_weather":
            start_date = args.get("start_date")
            end_date = args.get("end_date")
            if start_date is not None and not isinstance(start_date, str):
                raise ChatError("Weather dates must use YYYY-MM-DD format.")
            if end_date is not None and not isinstance(end_date, str):
                raise ChatError("Weather dates must use YYYY-MM-DD format.")
            place_name = args.get("place") or args.get("location")
            if place_name:
                return fetch_weather(
                    location_name=place_name,
                    start_date=start_date,
                    end_date=end_date,
                    days=args.get("days", 7),
                )
            lat, lng, label, _ = _location_for(args, user_loc)
            return fetch_weather(
                latitude=lat,
                longitude=lng,
                start_date=start_date,
                end_date=end_date,
                days=args.get("days", 7),
                location={"name": label, "latitude": lat, "longitude": lng},
            )
        if name == "find_nearby":
            lat, lng, label, _ = _location_for(args, user_loc)
            out = find_nearby(args["service"], lat, lng, args.get("radius_km", 15))
            out["around"] = label
            return out
        if name == "destination_info":
            return destination_info(args["name"])
        return {"error": f"Unknown tool {name}"}
    except ChatError as e:
        if str(e) == "NO_LOCATION":
            return {"error": "The user has not shared their location and did not name a place. Ask them for a town name or to tap the location button."}
        return {"error": str(e)}
    except requests.RequestException:
        return {"error": "The live data service is not responding right now. Ask the user to try again in a minute."}
    except WeatherServiceError as e:
        return {"error": str(e)}


# ------------------------------------------------------------------ Claude call

SYSTEM_PROMPT = """You are SeasonWise Ride & Travel Guide, a friendly, knowledgeable assistant for travellers and motorbike riders, mainly in India.

- Answer the user's question directly and fully, the way a knowledgeable friend would. You can answer general questions too (packing, route planning, riding tips, culture, food, safety, money, anything). Keep answers clear; use short numbered steps for directions or plans.
- Use the tools for live or location-based facts: weather, nearby petrol bunks, mechanics, tyre shops, hospitals, hotels and so on, and SeasonWise destination details. Never invent bunks, mechanics, phone numbers, hours, prices, or forecasts. If a tool returns nothing or an error, say so and suggest what the user can do.
- Preserve relevant information from earlier messages when answering follow-up questions. Ask a concise follow-up only when an essential detail is missing.
- For weather, distinguish current modelled conditions, numerical forecasts, and seasonal estimates. Give the key day-by-day points and rule-based advisories; do not call them official warnings.
- Never invent routes, distances, travel times, prices, availability, ratings, or live crowd counts. State when routing, cost, or seasonal details are estimates.
- For riding: suggest topping up fuel before long empty stretches, carrying a helmet and rain gear, checking tyres and brakes, and avoiding heavy rain or strong winds on mountain roads.
- For permits (Tawang, Lakshadweep, some border areas) and winter road closures, say rules change and suggest checking official sources.
- In an emergency or accident, first tell the user to call 112 (national emergency) or 108 (ambulance), then help with next steps.
- If you need the user's location and do not have it, ask them to tap the location button or name their town.

Today's date is {today}."""


def _call_claude(messages, system):
    if not ANTHROPIC_KEY:
        raise ChatError("Full AI chat is unavailable because ANTHROPIC_API_KEY is not configured.")
    if not CHAT_MODEL:
        raise ChatError("Full AI chat is unavailable because CHAT_MODEL is empty.")
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": CHAT_MODEL, "max_tokens": 1500, "system": system,
                  "tools": TOOLS, "messages": messages},
            timeout=90)
    except requests.Timeout as exc:
        raise ChatError("The AI service took too long to respond. Please try again.") from exc
    except requests.RequestException as exc:
        raise ChatError("I could not reach the AI service. Check the internet connection and try again.") from exc
    if r.status_code == 402:
        raise ChatError("The AI account has insufficient credit. Add credit to the Anthropic account or try again later.")
    if r.status_code == 401:
        raise ChatError("The AI key was rejected. Check that ANTHROPIC_API_KEY is correct, then restart the app.")
    if r.status_code == 429:
        raise ChatError("The AI service is busy or the key has run out of credit. Try again shortly.")
    if r.status_code == 529:
        raise ChatError("The AI service is temporarily overloaded. Please try again shortly.")
    if r.status_code == 400 and any(word in r.text.casefold() for word in ("credit", "billing", "balance")):
        raise ChatError("The AI account has insufficient credit. Add credit to the Anthropic account or try again later.")
    if r.status_code == 404 or (r.status_code == 400 and "model" in r.text.lower()):
        raise ChatError(f"The model name '{CHAT_MODEL}' was not accepted. Check the CHAT_MODEL setting.")
    if r.status_code >= 500:
        raise ChatError("The AI service is temporarily unavailable. Try again in a minute.")
    if r.status_code >= 400:
        raise ChatError(f"The AI service returned an error ({r.status_code}).")
    try:
        payload = r.json()
    except ValueError as exc:
        raise ChatError("The AI service returned an unreadable response. Please try again.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
        raise ChatError("The AI service returned an invalid response. Please try again.")
    return payload


def chat_with_claude(history, user_loc):
    system = SYSTEM_PROMPT.format(today=date.today().strftime("%A, %d %B %Y"))
    messages = list(history)
    for _ in range(6):  # allow a few rounds of tool use
        resp = _call_claude(messages, system)
        content = resp.get("content", [])
        text = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if resp.get("stop_reason") != "tool_use":
            if resp.get("stop_reason") == "max_tokens":
                raise ChatError("The AI response exceeded its output limit. Please ask a shorter question.")
            if resp.get("stop_reason") not in {"end_turn", None}:
                raise ChatError("The AI could not complete its response. Please try again.")
            if not text:
                raise ChatError("The AI service returned no answer. Please try asking again.")
            return text or "I did not get an answer. Please try asking again."
        messages.append({"role": "assistant", "content": content})
        results = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if not isinstance(block.get("id"), str) or not isinstance(block.get("name"), str):
                    raise ChatError("The AI requested a tool with invalid arguments. Please try again.")
                arguments = block.get("input", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                out = run_tool(block["name"], arguments, user_loc)
                results.append({"type": "tool_result", "tool_use_id": block["id"],
                                "content": json.dumps(out, ensure_ascii=False)})
        if not results:
            raise ChatError("The AI returned an incomplete tool request. Please try again.")
        messages.append({"role": "user", "content": results})
    raise ChatError("I could not finish looking that up after several steps. Please try again with a more focused question.")


# ------------------------------------------------------------------ basic mode (no AI key)

BASIC_NOTE = ("I can answer weather and nearby-service questions right now (petrol bunks, mechanics, hospitals, "
              "hotels and more). To ask me anything else in full, the site owner needs to add an AI key: set "
              "ANTHROPIC_API_KEY and restart the app.")

TOPICS = [
    ("fuel", ["petrol", "diesel", "fuel", "bunk", "pump"]),
    ("mechanic", ["mechanic", "repair", "workshop", "service centre", "breakdown", "bike service"]),
    ("tyre", ["tyre", "tire", "puncture"]),
    ("hospital", ["hospital", "doctor", "clinic", "ambulance"]),
    ("pharmacy", ["medicine", "pharmacy", "chemist"]),
    ("stay", ["hotel", "stay", "guesthouse", "homestay", "room"]),
    ("food", ["food", "restaurant", "eat", "dhaba"]),
    ("atm", ["atm", "cash"]),
    ("toilet", ["toilet", "washroom", "restroom"]),
    ("ev_charging", ["ev charg", "charging station"]),
]


MONTH_NUMBERS = {
    name.casefold(): index for index, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"), 1
    )
}


def _mentioned_place(texts):
    combined = " ".join(texts)
    for destination in sorted(DESTINATIONS, key=lambda item: len(item["name"]), reverse=True):
        if destination["name"].casefold() in combined.casefold():
            return destination["name"]
    match = re.search(
        r"\b(?:weather|forecast|rain|temperature|wind|nearby|near|in|around)\s+"
        r"(?:in|for|at|around|near)?\s*([A-Z][A-Za-z .'-]{1,60}(?:,\s*[A-Z][A-Za-z .'-]+)?)",
        combined,
    )
    if match:
        candidate = match.group(1).strip(" .,-")
        candidate = re.split(
            r"\b(?:tomorrow|today|this week|next week|next weekend|for \d+ days)\b",
            candidate,
            flags=re.IGNORECASE,
        )[0].strip(" .,-")
        return candidate or None
    return None


def _basic_weather_text(report):
    location = report.get("location") or {}
    label = ", ".join(filter(None, (location.get("name"), location.get("admin1"), location.get("country"))))
    current = report.get("current") or {}
    lines = [f"Weather for {label or 'your selected location'}:"]
    if current.get("temperature_c") is not None:
        lines.append(
            f"Current modelled conditions: {current['temperature_c']}°C"
            + (f", feels like {current['apparent_temperature_c']}°C" if current.get("apparent_temperature_c") is not None else "")
            + (f", {current['conditions']}" if current.get("conditions") else "")
            + "."
        )
    if report.get("days"):
        for day in report["days"]:
            temperature = "temperature unavailable"
            if day.get("temperature_min_c") is not None and day.get("temperature_max_c") is not None:
                temperature = f"{day['temperature_min_c']}–{day['temperature_max_c']}°C"
            details = [temperature]
            if day.get("conditions"):
                details.append(day["conditions"])
            if day.get("precipitation_probability_max") is not None:
                details.append(f"rain chance {day['precipitation_probability_max']}%")
            if day.get("precipitation_sum_mm") is not None:
                details.append(f"{day['precipitation_sum_mm']} mm precipitation")
            lines.append(f"{day['date']}: " + ", ".join(details) + ".")
    else:
        lines.append(report.get("message", "No date-specific forecast is available."))
    advisories = report.get("advisories") or []
    if advisories:
        lines.append("Travel advisories (rule-based, not official warnings):")
        lines.extend(f"- {item}" for item in advisories)
    lines.append(report.get("accuracy_note", "Forecasts can change; verify before departure."))
    return "\n".join(lines)


def basic_answer(message, user_loc, history=None):
    text = message.lower()
    prior_texts = [str(item.get("content", "")) for item in (history or []) if item.get("role") in ("user", "assistant")]
    place = _mentioned_place([*prior_texts[-6:], message])
    args = {"place": place} if place else {}
    if any(w in text for w in ["weather", "rain", "forecast", "temperature", "wind"]):
        date_match = re.search(r"\b(?:in|after)\s+(\d{1,2})\s+days?\b", text)
        start_date = None
        if "tomorrow" in text:
            from datetime import date, timedelta
            start_date = (date.today() + timedelta(days=1)).isoformat()
        weather_args = {**args, "days": 5}
        if start_date:
            weather_args.update({"start_date": start_date, "days": 1})
        elif date_match:
            from datetime import date, timedelta
            weather_args.update({"start_date": (date.today() + timedelta(days=int(date_match.group(1)))).isoformat(), "days": 1})
        w = run_tool("get_weather", weather_args, user_loc)
        if "error" in w:
            return w["error"]
        return _basic_weather_text(w)
    for service, words in TOPICS:
        if any(w in text for w in words):
            n = run_tool("find_nearby", {"service": service, **args}, user_loc)
            if "error" in n:
                return n["error"]
            if not n["places"]:
                return f"I could not find any {n['service']} within {n['radius_km']} km of {n['around']}."
            lines = [f"Nearest {n['service']} options around {n['around']}:"]
            for i, p in enumerate(n["places"], 1):
                hours = f", open {p['opening_hours']}" if p["opening_hours"] else ""
                lines.append(f"{i}. {p['name']}, {p['distance_km']} km{hours}")
            lines.append(n["note"])
            return "\n".join(lines)
    month = next((number for name, number in MONTH_NUMBERS.items() if name in text), None)
    if month and any(phrase in text for phrase in ("visit", "travel", "destination", "where", "places")):
        recommendations = sorted(
            DESTINATIONS,
            key=lambda item: (
                0 if month in item["months"] else 1,
                1 if month in item["rainy_months"] else 0,
                -item.get("rating", 0),
            ),
        )[:5]
        lines = [f"SeasonWise's seasonal guide suggests these places for {list(MONTH_NUMBERS)[month - 1].title()}:"]
        for item in recommendations:
            note = "usual preferred season" if month in item["months"] else "outside usual preferred months"
            if month in item["rainy_months"]:
                note += "; rain may affect plans"
            lines.append(f"- {item['name']} ({item['state']}): {note}.")
        lines.append("These are seasonal estimates, not live forecasts or measured crowd data.")
        if not ai_enabled():
            lines.append("Add a valid Anthropic API key for tailored itineraries and natural-language trip planning.")
        return "\n".join(lines)
    if place:
        d = run_tool("destination_info", {"name": place}, user_loc)
        if "error" in d:
            return d["error"]
        if d.get("description"):
            return (f"{d['name']} ({d.get('state') or d.get('country') or 'location'}): "
                    f"{d['description']} Best months: {d.get('best_months', 'not available')}. "
                    "You can ask about its weather or nearby services.")
        return (
            f"{d['name']} is located in {d.get('state') or d.get('country') or 'the selected area'}. "
            "It is not in SeasonWise's curated destination data, so I cannot verify attractions or seasonal advice. "
            "Ask about its weather or nearby mapped services."
        )
    return BASIC_NOTE


# ------------------------------------------------------------------ entry point

def chat(messages, location=None):
    """Used by app.py. messages: [{"role": "user"|"assistant", "content": str}] ending with a user message.
    location: {"lat": float, "lon": float} or {"place": str}. AI service failures raise ChatError."""
    user_loc = None
    if isinstance(location, dict) and isinstance(location.get("place"), str) and location["place"].strip():
        user_loc = {"place": location["place"].strip()[:120]}
    elif isinstance(location, dict) and "lat" in location and "lon" in location:
        try:
            lat, lon = validate_coordinates(location["lat"], location["lon"])
            user_loc = {"lat": lat, "lng": lon, "name": "your shared location"}
        except (TypeError, ValueError, InvalidWeatherRequest):
            user_loc = None
    history = [
        {"role": item["role"], "content": item["content"][:MAX_CHARS]}
        for item in messages
        if isinstance(item, dict)
        and item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]
    history = history[-MAX_MESSAGES:]
    if not history or history[-1]["role"] != "user":
        return "Ask me anything about your trip, the weather, or nearby petrol bunks and mechanics."
    question = history[-1]["content"]
    if ANTHROPIC_KEY and not CHAT_MODEL:
        raise ChatError("Full AI chat is unavailable because CHAT_MODEL is empty.")
    if ai_enabled():
        return chat_with_claude(history, user_loc)
    return basic_answer(question, user_loc, history)
