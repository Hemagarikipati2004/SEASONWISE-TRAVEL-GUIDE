from flask import Flask, render_template, request, jsonify
from datetime import date, datetime, timedelta
import requests
from urllib.parse import quote_plus
from functools import lru_cache
from weather_service import (
    AmbiguousLocationError,
    HEAVY_RAIN_MM_THRESHOLD,
    InvalidWeatherRequest,
    LocationNotFoundError,
    WIND_GUST_KMH_THRESHOLD,
    WIND_SPEED_KMH_THRESHOLD,
    WeatherServiceError,
    get_weather as fetch_weather,
    validate_coordinates,
)

app = Flask(__name__)

import json
from pathlib import Path

# All destinations live in data/destinations.json so places can be added without touching code.
DESTINATIONS = json.loads((Path(__file__).parent / "data" / "destinations.json").read_text(encoding="utf-8"))

REGIONS = ["North", "South", "East", "West", "Central", "Northeast"]

# Fallback photo per destination type, used when a place has no image of its own.
IMAGE_BY_KIND = {
    "beach": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=1000&q=80",
    "nature": "https://images.unsplash.com/photo-1593693411515-c20261bcad6e?auto=format&fit=crop&w=1000&q=80",
    "culture": "https://images.unsplash.com/photo-1599661046827-dacff0c0f09a?auto=format&fit=crop&w=1000&q=80",
    "mountains": "https://images.unsplash.com/photo-1626621341517-bbf3d9990a23?auto=format&fit=crop&w=1000&q=80",
    "adventure": "https://images.unsplash.com/photo-1566837497312-7be4a6e9c7c9?auto=format&fit=crop&w=1000&q=80",
}

def normalise_region(value):
    """Accept 'North', 'North India', or 'India' (= all regions)."""
    value = (value or "India").replace(" India", "").strip()
    return value if value in REGIONS else "India"


from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Wikipedia article to use for each place's photo, when the place name alone is ambiguous.
WIKI_TITLE = {
    "Mumbai": "Gateway of India",
    "Kolkata": "Victoria Memorial, Kolkata",
    "Mathura and Vrindavan": "Vrindavan",
    "Lonavala and Khandala": "Lonavala",
    "Ajanta and Ellora Caves": "Ajanta Caves",
    "Chitrakote Falls, Bastar": "Chitrakote Falls",
    "Mahabalipuram": "Shore Temple",
    "Amaravati": "Amaravati Stupa",
    "Goa": "Baga Beach",
    "Andaman Islands": "Andaman Islands",
}

@lru_cache(maxsize=512)
def _wiki_photo(title):
    """Return a photo URL for a Wikipedia article, or None."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(title.replace(" ", "_")),
            headers={"User-Agent": "SeasonWise-student-project/1.0"}, timeout=2)
        if r.status_code != 200:
            return None
        data = r.json()
        return (data.get("originalimage") or data.get("thumbnail") or {}).get("source")
    except Exception:
        return None

def attach_photos(places):
    """Give each place its own photo. Falls back to a type photo only if no photo can be found."""
    titles = [WIKI_TITLE.get(p["name"], p["name"]) for p in places]
    with ThreadPoolExecutor(max_workers=8) as pool:
        urls = list(pool.map(_wiki_photo, titles))
    for p, url in zip(places, urls):
        p["image"] = url or IMAGE_BY_KIND.get(p["kind"])
    return places


TRAVEL_NOTES = json.loads((Path(__file__).parent / "data" / "travel_notes.json").read_text(encoding="utf-8"))

# Default advice by destination type. Specific places override these in data/travel_notes.json.
DEFAULT_TRANSPORT = {
    "beach":     {"flight": ["Good", "Fly to the nearest airport, then take a taxi."],
                  "train":  ["Good", "Coastal trains are often comfortable and scenic."],
                  "car":    ["Good", "Useful for exploring several beaches."],
                  "bike":   ["Good", "Popular for short coastal rides; wear a helmet."]},
    "mountains": {"flight": ["Good", "Fly to the nearest airport, then drive up."],
                  "train":  ["Possible", "Trains often stop at the foothills, so a road leg follows."],
                  "car":    ["Best", "Hill roads are easiest by private car or cab."],
                  "bike":   ["Possible", "Scenic, but for experienced riders; check road conditions."]},
    "nature":    {"flight": ["Good", "Fly to the nearest airport for longer trips."],
                  "train":  ["Possible", "Check connections to the nearest station."],
                  "car":    ["Best", "Best way to reach forests, hills, and scenic spots."],
                  "bike":   ["Possible", "Good on quiet roads, but not in heavy rain."]},
    "culture":   {"flight": ["Good", "Good for long-distance travel."],
                  "train":  ["Best", "Trains are often the easiest way to reach historic cities."],
                  "car":    ["Good", "Useful for visiting several sites in one trip."],
                  "bike":   ["Possible", "Good for getting around the city; traffic can be heavy."]},
    "adventure": {"flight": ["Good", "Fly to the nearest base town, then continue by road."],
                  "train":  ["Possible", "Usually a road leg is needed from the station."],
                  "car":    ["Good", "A sturdy vehicle or hired cab with a driver is recommended."],
                  "bike":   ["Best", "Popular with riders; check fitness, permits, and weather."]},
}

DEFAULT_STAYS = {
    "beach":     ["Beach resorts", "Beachside guesthouses", "Homestays near the coast"],
    "mountains": ["Hill cottages", "Homestays with valley views", "Boutique hill resorts"],
    "nature":    ["Eco-lodges", "Forest resorts", "Homestays in nearby villages"],
    "culture":   ["Heritage hotels", "Guesthouses in the old town", "Business hotels near the centre"],
    "adventure": ["Base camps and lodges", "Guesthouses in the town", "Homestays"],
}

def travel_info(place):
    """Combine default transport and stay advice with any place-specific notes."""
    notes = TRAVEL_NOTES.get(place["name"], {})
    transport = {m: {"level": v[0], "note": v[1]} for m, v in DEFAULT_TRANSPORT[place["kind"]].items()}
    for m, v in notes.get("transport", {}).items():
        transport[m] = {"level": v[0], "note": v[1]}
    stays = notes.get("stays") or DEFAULT_STAYS[place["kind"]]
    return {"transport": transport, "stays": stays}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def season_score(place, month):
    score = 88 if month in place["months"] else 52
    if month in place["rainy_months"]:
        score -= 18
    return max(20, min(98, score))

def crowd_estimate(month, place):
    # Simple illustrative estimate; replace with actual visitor/event data for production.
    peak = month in [12, 1, 4, 5]
    if month in place["months"] and peak:
        return "High", "Popular travel period; book transport and stays early."
    if month in place["months"]:
        return "Moderate", "Usually a reasonable balance of conditions and crowds."
    return "Low–moderate", "Off-season may mean fewer visitors, but some activities may be affected."

def weather_for(place, start_date, end_date):
    try:
        report = fetch_weather(
            latitude=place["lat"],
            longitude=place["lon"],
            start_date=start_date,
            end_date=end_date,
            location={
                "name": place["name"],
                "admin1": place.get("state"),
                "country": "India",
                "latitude": place["lat"],
                "longitude": place["lon"],
                "timezone": None,
            },
        )
    except WeatherServiceError as exc:
        return {
            "type": "unavailable",
            "message": str(exc),
            "days": [],
            "advisories": [],
        }
    report = {**report, "days": [
        {
            **day,
            "date": date.fromisoformat(day["date"]).strftime("%a, %d %b"),
            "min": day.get("temperature_min_c"),
            "max": day.get("temperature_max_c"),
            "rain": day.get("precipitation_probability_max"),
            "code": day.get("weather_code"),
        }
        for day in report["days"]
    ]}
    return report

@app.route("/")
def home():
    return render_template("index.html", months=MONTHS)

@app.route("/api/recommend", methods=["POST"])
def recommend():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Please send trip details as a JSON object."}), 400

    start_value = payload.get("start_date")
    end_value = payload.get("end_date")
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        return jsonify({"error": "Please choose valid start and end dates."}), 400
    try:
        start_date = date.fromisoformat(start_value)
        end_date = date.fromisoformat(end_value)
    except ValueError:
        return jsonify({"error": "Please choose valid start and end dates."}), 400
    if end_date < start_date:
        return jsonify({"error": "End date must be on or after the start date."}), 400
    if (end_date - start_date).days > 30:
        return jsonify({"error": "Please select a trip of 31 days or fewer."}), 400

    kind = payload.get("kind", "any")
    region = normalise_region(payload.get("region", "India"))
    preference = payload.get("weather", "any")
    month = start_date.month
    results = []
    for p in DESTINATIONS:
        if region != "India" and p["region"] != region:
            continue
        if kind != "any" and p["kind"] != kind:
            continue
        s_score = season_score(p, month)
        weather_score = 82 if month not in p["rainy_months"] else 58
        if preference == "cold":
            weather_score += 8 if p["kind"] == "mountains" else -5
        elif preference == "warm":
            weather_score += 8 if p["kind"] == "beach" else -2
        elif preference == "pleasant":
            weather_score += 5 if p["kind"] in ["nature", "culture"] else 0
        weather_score = max(25, min(95, weather_score))
        crowd, crowd_note = crowd_estimate(month, p)
        crowd_score = 58 if crowd == "High" else 82 if crowd == "Moderate" else 88
        rating_score = p["rating"] / 5 * 100
        overall = round(s_score * .40 + weather_score * .30 + crowd_score * .15 + rating_score * .15)
        if s_score < 40:
            explanation = "This is outside the usual preferred season, so check access, local conditions, and activity availability before booking."
        elif month in p["rainy_months"]:
            explanation = "Seasonally possible, but rainfall may affect outdoor plans. Keep a flexible itinerary."
        else:
            explanation = "Your selected month generally aligns with this destination's preferred travel period."
        results.append({**p, **travel_info(p), "season_score": s_score, "weather_score": weather_score,
                        "weather_score_basis": "seasonal estimate",
                        "crowd": crowd, "crowd_note": crowd_note, "crowd_score": crowd_score,
                        "rating_score": round(rating_score),
                        "score": overall, "explanation": explanation})
    results.sort(key=lambda x: x["score"], reverse=True)

    # Limit forecast requests to the leading candidates; the rest retain labelled seasonal estimates.
    forecast_candidates = results[:5]
    with ThreadPoolExecutor(max_workers=5) as pool:
        forecasts = list(pool.map(lambda p: weather_for(p, start_date, end_date), forecast_candidates))
    for place, forecast in zip(forecast_candidates, forecasts):
        place["weather_report"] = forecast
        forecast_days = forecast.get("days", []) if forecast.get("type") == "forecast" else []
        has_forecast_metrics = any(
            day.get("rain") is not None
            or day.get("precipitation_sum_mm") is not None
            or day.get("wind_speed_max_kmh") is not None
            or (day.get("min") is not None and day.get("max") is not None)
            for day in forecast_days
        )
        if not forecast_days or forecast.get("forecast_complete", True) is False or not has_forecast_metrics:
            continue
        rain_chances = [day["rain"] for day in forecast_days if day.get("rain") is not None]
        rain_amounts = [
            day["precipitation_sum_mm"] for day in forecast_days
            if day.get("precipitation_sum_mm") is not None
        ]
        wind_speeds = [
            day["wind_speed_max_kmh"] for day in forecast_days
            if day.get("wind_speed_max_kmh") is not None
        ]
        wind_gusts = [
            day["wind_gusts_max_kmh"] for day in forecast_days
            if day.get("wind_gusts_max_kmh") is not None
        ]
        temperatures = [
            (day["min"] + day["max"]) / 2
            for day in forecast_days if day.get("min") is not None and day.get("max") is not None
        ]
        score = 82
        if rain_chances:
            average_rain = sum(rain_chances) / len(rain_chances)
            score -= 20 if average_rain >= 65 else 10 if average_rain >= 40 else 0
        if any(amount >= HEAVY_RAIN_MM_THRESHOLD for amount in rain_amounts):
            score -= 8
        if any(speed >= WIND_SPEED_KMH_THRESHOLD for speed in wind_speeds) or any(
            gust >= WIND_GUST_KMH_THRESHOLD for gust in wind_gusts
        ):
            score -= 10
        if temperatures:
            average_temperature = sum(temperatures) / len(temperatures)
            if preference == "cold":
                score += 8 if average_temperature < 18 else -8 if average_temperature > 26 else 0
            elif preference == "warm":
                score += 8 if average_temperature > 27 else -8 if average_temperature < 20 else 0
            elif preference == "pleasant":
                score += 5 if 18 <= average_temperature <= 30 else -5
        place["weather_score"] = max(25, min(95, score))
        place["weather_score_basis"] = "Open-Meteo forecast for the selected dates"
        place["weather_disadvantages"] = forecast.get("advisories", [])
        place["score"] = round(
            place["season_score"] * .40
            + place["weather_score"] * .30
            + place["crowd_score"] * .15
            + place["rating_score"] * .15
        )
        place["explanation"] += " The leading-candidate weather score also uses the available date-specific forecast."
    results.sort(key=lambda x: x["score"], reverse=True)
    for place in results:
        place["image"] = IMAGE_BY_KIND.get(place["kind"])
    attach_photos(results[:8])
    weather = results[0].get("weather_report") if results else None
    if results and weather is None:
        weather = weather_for(results[0], start_date, end_date)
    if not results:
        weather = {"type": "seasonal", "message": "No matching destinations.", "days": []}
    return jsonify({"destinations": results, "count": len(results), "region": region,
                    "weather": weather, "month": MONTHS[month-1],
                    "start_date": start_date.isoformat(), "end_date": end_date.isoformat()})


@app.route("/api/weather")
def api_weather():
    location_name = request.args.get("location", "").strip()
    start_value = request.args.get("start_date")
    end_value = request.args.get("end_date")
    try:
        if location_name:
            report = fetch_weather(
                location_name=location_name,
                start_date=start_value,
                end_date=end_value,
                days=request.args.get("days", 7),
            )
        else:
            report = fetch_weather(
                latitude=request.args.get("lat"),
                longitude=request.args.get("lon"),
                start_date=start_value,
                end_date=end_value,
                days=request.args.get("days", 7),
            )
    except AmbiguousLocationError as exc:
        return jsonify({"error": str(exc), "code": "ambiguous_location"}), 422
    except LocationNotFoundError as exc:
        return jsonify({"error": str(exc), "code": "location_not_found"}), 404
    except InvalidWeatherRequest as exc:
        return jsonify({"error": str(exc), "code": "invalid_request"}), 400
    except WeatherServiceError as exc:
        return jsonify({"error": str(exc), "code": "weather_unavailable"}), 503
    return jsonify(report)

@app.route("/api/destination/<int:destination_id>")
def destination_detail(destination_id):
    place = next((p for p in DESTINATIONS if p["id"] == destination_id), None)
    if not place:
        return jsonify({"error": "Destination not found."}), 404
    result = {**place, **travel_info(place)}
    result["hotel_search_url"] = "https://www.google.com/maps/search/ hotels near/" + quote_plus(place["name"] + ", " + place["state"] + ", India")
    result["google_hotels_url"] = "https://www.google.com/maps/search/?api=1&query=" + quote_plus("hotels near " + place["name"] + ", " + place["state"] + ", India")
    result["google_food_url"] = "https://www.google.com/maps/search/?api=1&query=" + quote_plus("restaurants near " + place["name"] + ", " + place["state"] + ", India")
    result["google_directions_url"] = "https://www.google.com/maps/dir/?api=1&destination=" + quote_plus(place["name"] + ", " + place["state"] + ", India")
    return jsonify(result)

# Live discovery of real, mapped attractions. The local JSON remains the fast curated baseline;
# this endpoint supplements it with OpenStreetMap tourism / historic / natural places.
DISCOVERY_BBOX = {
    "North": "23,68,37,79",
    "South": "8,74,20,85",
    "East": "18,84,28,97",
    "West": "15,68,25,78",
    "Central": "18,76,27,85",
    "Northeast": "21,89,30,98",
    "India": "6,68,38,98",
}
_DISCOVERY_CACHE = {}

def discover_osm_places(region="India", limit=1000):
    region = region if region in DISCOVERY_BBOX else "India"
    limit = max(50, min(int(limit), 1000))
    key = (region, limit)
    cached = _DISCOVERY_CACHE.get(key)
    if cached and (datetime.now() - cached[0]).total_seconds() < 3600:
        return cached[1]
    south, west, north, east = [float(x.strip()) for x in DISCOVERY_BBOX[region].split(",")]
    # Keep the query bounded and return named places only. Overpass results are real OSM records.
    query = f"""[out:json][timeout:35];(
      nwr["tourism"~"attraction|museum|viewpoint|zoo|theme_park|aquarium|gallery|artwork"]({south},{west},{north},{east});
      nwr["historic"~"monument|castle|archaeological_site|fort|memorial|ruins"]({south},{west},{north},{east});
      nwr["natural"~"waterfall|beach|hot_spring|cave_entrance"]({south},{west},{north},{east});
    );out center tags {limit};"""
    response = requests.post("https://overpass-api.de/api/interpreter", data={"data": query},
                             headers={"User-Agent": "SeasonWise/1.0 (travel discovery project)"}, timeout=45)
    response.raise_for_status()
    items = []
    seen = set()
    for el in response.json().get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("name:en")
        center = el.get("center", {})
        lat = el.get("lat", center.get("lat")); lon = el.get("lon", center.get("lon"))
        if not name or lat is None or lon is None:
            continue
        identity = (name.strip().casefold(), round(float(lat), 3), round(float(lon), 3))
        if identity in seen: continue
        seen.add(identity)
        state = tags.get("addr:state") or tags.get("is_in:state") or "India"
        raw_kind = tags.get("tourism", tags.get("historic", tags.get("natural", "attraction")))
        kind = "culture" if raw_kind in {"museum", "gallery", "monument", "castle", "archaeological_site", "fort", "memorial", "ruins", "artwork"} else "nature"
        items.append({"id": "osm-" + str(el.get("type")) + "-" + str(el.get("id")), "name": name,
                      "state": state, "region": region, "kind": kind, "lat": float(lat), "lon": float(lon),
                      "months": [10,11,12,1,2,3], "rainy_months": [6,7,8,9], "best": "Often October–March; check local conditions",
                      "rating": None, "description": tags.get("description") or tags.get("tourism") or "Mapped attraction. Verify local access, opening hours and conditions before visiting.",
                      "tags": [raw_kind], "source": "OpenStreetMap", "details_url": "https://www.openstreetmap.org/" + str(el.get("type")) + "/" + str(el.get("id"))})
        if len(items) >= limit: break
    _DISCOVERY_CACHE[key] = (datetime.now(), items)
    return items

@app.route("/api/places/discover")
def api_discover_places():
    region = normalise_region(request.args.get("region", "India"))
    try:
        limit = int(request.args.get("limit", "500"))
        places = discover_osm_places(region, limit)
    except (requests.RequestException, ValueError):
        return jsonify({"error": "Live place discovery is temporarily unavailable. Try again later; curated destinations are still available."}), 503
    existing = {(p["name"].casefold(), round(p["lat"], 3), round(p["lon"], 3)) for p in DESTINATIONS}
    places = [p for p in places if (p["name"].casefold(), round(p["lat"], 3), round(p["lon"], 3)) not in existing]
    return jsonify({"places": places, "count": len(places), "region": region,
                    "source": "OpenStreetMap", "note": "Live mapped places; data completeness varies by area. Verify access and opening hours."})


from chatbot import ChatError, ai_status, chat as trip_chat

@app.route("/api/chat/status")
def api_chat_status():
    return jsonify(ai_status())

@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Please send the conversation as a JSON object."}), 400

    messages = payload.get("messages")
    if (not isinstance(messages, list) or not messages
            or len(messages) > 40
            or any(not isinstance(message, dict) for message in messages)
            or messages[-1].get("role") != "user"
            or any(
                message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                or len(message["content"]) > 2000
                for message in messages
            )):
        return jsonify({"error": "Send the conversation with a final user message."}), 400
    messages = messages[-20:]  # keep the request small
    location = payload.get("location") or {}
    if not isinstance(location, dict):
        return jsonify({"error": "Location must be a JSON object."}), 400
    place_name = location.get("place")
    if place_name is not None and (
        not isinstance(place_name, str) or not place_name.strip() or len(place_name) > 120
    ):
        return jsonify({"error": "Enter a valid location name up to 120 characters long."}), 400
    try:
        if place_name:
            location = {"place": place_name.strip()}
        elif "lat" in location and "lon" in location:
            latitude, longitude = validate_coordinates(location["lat"], location["lon"])
            location = {"lat": latitude, "lon": longitude}
        else:
            location = {}
    except (TypeError, ValueError, InvalidWeatherRequest) as exc:
        return jsonify({"error": str(exc) or "Location coordinates must be valid numbers."}), 400
    try:
        reply = trip_chat(messages, location)
    except ChatError as e:
        return jsonify({"error": str(e)}), 503
    except Exception:
        app.logger.exception("Trip helper request failed")
        return jsonify({"error": "The assistant is unavailable right now. Please try again."}), 502
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
