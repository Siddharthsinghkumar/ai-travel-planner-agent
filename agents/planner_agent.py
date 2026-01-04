"""
Planner Agent (Brain Layer)

Responsibilities:
- Parse user intent
- Retrieve and validate flight & weather data
- Apply preference-aware scoring
- Generate LLM explanations
- Persist full audit trail to PostgreSQL

UI-agnostic, FastAPI-ready.
"""

import os
import re
import json
# ❌ REMOVED: from openai import OpenAI
from tools.airline_api import search_flights
from tools.weather_api import check_weather
from dotenv import load_dotenv
from datetime import datetime, timedelta
from airportsdata import load
from difflib import get_close_matches
import dateutil.parser
from pydantic import BaseModel, ValidationError, field_validator
from typing import Dict, Any, Optional, Union

load_dotenv()

# ✅ ADD: Import from llm_router
from agents.llm_router import generate

# ✅ ADD: Environment variable for cloud fallback
USE_CLOUD_FALLBACK = os.getenv("USE_CLOUD_LLM", "1") == "1"
print(f"🔧 LLM Configuration: USE_CLOUD_FALLBACK = {USE_CLOUD_FALLBACK}")

# Type alias for clarity
FlightDict = Dict[str, Union[str, int, None]]

# 🔧 Cache manager with monotonic TTL
class CacheManager:
    """Simple cache manager for non-UI environments with monotonic TTL"""
    _cache = {}
    
    @classmethod
    def cache_data(cls, show_spinner=False, ttl=None):
        """Decorator for caching function results with monotonic TTL"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                cache_key = f"{func.__name__}:{args}:{kwargs}"
                
                # Check if cached and not expired
                if cache_key in cls._cache:
                    cached_time, cached_value = cls._cache[cache_key]
                    # 🔧 Fix 1: Use total_seconds() to avoid rollover bugs
                    if ttl and (datetime.now() - cached_time).total_seconds() < ttl:
                        return cached_value
                
                # Execute and cache
                result = func(*args, **kwargs)
                cls._cache[cache_key] = (datetime.now(), result)
                return result
            
            return wrapper
        return decorator

# Use our custom cache manager
cache_data = CacheManager.cache_data

@cache_data(ttl=900)
def cached_search(origin, destination, date):
    return search_flights(origin, destination, date)

@cache_data(ttl=3600)
def cached_weather(city, date):
    return check_weather(city, date)

# ❌ REMOVED: get_llm_client() function

# Database imports for session logging
try:
    from agents.database import SessionLocal, SessionHistory
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("⚠️ Database module not available. Session logging disabled.")

# Pydantic model for flight validation
class Flight(BaseModel):
    """Validated flight data model"""
    airline: str
    flight_no: str
    departure_time: str
    arrival_time: str
    duration_min: int
    price_inr: Union[str, int]
    stops: str = "N/A"
    baggage: str = "N/A"
    layover_time: Union[str, int] = "0"
    date: Optional[str] = None
    
    @field_validator('price_inr', mode='before')
    @classmethod
    def validate_price(cls, v):
        """Convert price to consistent format"""
        if isinstance(v, int):
            return f"₹{v:,}"
        if isinstance(v, str):
            # Ensure it has ₹ symbol
            if not v.startswith('₹'):
                try:
                    price_int = int(str(v).replace(',', '').replace('₹', '').strip())
                    return f"₹{price_int:,}"
                except:
                    return "₹999,999"
        return v
    
    @field_validator('departure_time', 'arrival_time', mode='before')
    @classmethod
    def validate_time_format(cls, v):
        """Ensure time is in HH:MM format"""
        if isinstance(v, str):
            # Try to extract HH:MM format
            match = re.search(r'(\d{1,2}):(\d{2})', v)
            if match:
                hour, minute = match.groups()
                return f"{int(hour):02d}:{minute}"
        return "00:00"
    
    @field_validator('duration_min', mode='before')
    @classmethod
    def validate_duration(cls, v):
        """Ensure duration is integer"""
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                # Extract numbers from string
                numbers = re.findall(r'\d+', v)
                if numbers:
                    return int(numbers[0])
            except:
                pass
        return 999

# 🔧 Fix 2: Use type alias for clarity
def validate_flight_data(flight_dict: FlightDict) -> Optional[FlightDict]:
    """Validate flight data using Pydantic model"""
    try:
        validated = Flight(**flight_dict)
        return validated.model_dump()
    except ValidationError as e:
        print(f"❌ Flight validation failed: {e.errors()}")
        print(f"   Problematic data: {flight_dict}")
        return None

# Session logging function
def save_session(user_query, agent_reasoning, tool_output, final_response, user_id=None):
    """Save session history to database"""
    if not DB_AVAILABLE:
        print("📝 Session logging skipped (database not available)")
        return
    
    db = SessionLocal()
    try:
        # Pass dicts directly (no json.dumps) - DB columns are already JSON
        sh = SessionHistory(
            user_id=user_id,
            user_query=user_query,
            agent_reasoning=agent_reasoning,   # dict directly
            tool_output=tool_output,          # dict directly
            final_response=final_response
        )
        db.add(sh)
        db.commit()
        print("📝 Session saved to database")
    except Exception as e:
        print(f"❌ Failed to save session: {e}")
        db.rollback()
    finally:
        db.close()

# Build a lowercase city → IATA code map
def build_city_to_iata_map():
    airports = load('IATA')
    city_map = {}
    for code, data in airports.items():
        city = data.get('city') or data.get('city_en') or data.get('location')
        if city:
            city = city.strip().lower()
            if city not in city_map:
                city_map[city] = code
    return city_map

def resolve_city_to_iata(city_name: str) -> Optional[str]:
    name = city_name.strip().lower()
    for city_key, code in CITY_TO_IATA.items():
        # Allow substring match, e.g., "delhi" in "new delhi"
        if name in city_key or city_key in name:
            return code
    return None


CITY_TO_IATA = build_city_to_iata_map()
print("🗺️ Loaded city map:", list(CITY_TO_IATA.items())[:10])

def extract_cities_from_query(query):
    """
    Returns: (from_iata, to_iata) if both found, else (None, None)
    """
    pattern = re.search(r'\bfrom ([a-zA-Z\s]+?) to ([a-zA-Z\s]+?)(?:\s|$)', query.lower())
    if pattern:
        from_city = pattern.group(1).strip().lower()
        to_city = pattern.group(2).strip().lower()
        return resolve_city_to_iata(from_city), resolve_city_to_iata(to_city)
    return None, None

# Preferences and filter extractors
TIME_WINDOWS = {
    "morning": ("04:00", "11:59"),
    "afternoon": ("12:00", "17:59"),
    "evening": ("18:00", "23:59"),
    "night": ("00:00", "03:59")
}
AIRLINES = ["indigo", "air india", "vistara", "goair", "spicejet", "akasa", "airasia"]

# ——— SMART OPTIMIZATION INTENT DETECTION ———
def detect_flight_preference(query: str) -> str:
    q = query.lower()
    if "cheapest" in q or "lowest price" in q or "budget" in q:
        return "cheapest"
    if "shortest" in q or "fastest" in q or "least time" in q or "quickest" in q:
        return "shortest"
    if "balanced" in q or ("price" in q and "duration" in q):
        return "balanced"
    return "default"

# ——— DYNAMIC PRICE/DURATION WEIGHTS ———
def get_weights(preference: str) -> tuple[float, float]:
    if preference == "cheapest":
        return 0.8, 0.2
    elif preference == "shortest":
        return 0.2, 0.8
    elif preference == "balanced":
        return 0.5, 0.5
    return 0.6, 0.4  # default: lean toward price

# ——— HYBRID SCORING FUNCTION ———
def score_flight(flight: FlightDict, preference: str = "default") -> float:
    # parse price using validated format
    try:
        price_str = flight.get("price_inr", "₹999,999")
        price = int(str(price_str).replace('₹', '').replace(',', ''))
    except:
        price = 999_999
    
    # parse duration in minutes
    try:
        duration = int(flight.get("duration_min", 9999))
    except:
        duration = 9999

    # normalize to [0…100]
    price_score    = max(0, 100_000 - price) / 1_000   # ₹100 000 → 0; ₹0 → 100
    duration_score = max(0, 1_000 - duration)          # 1 000 min → 0; 0 min → 100

    wp, wd = get_weights(preference)
    return price_score * wp + duration_score * wd


def detect_time_preference(query):
    query = query.lower()
    for key in TIME_WINDOWS:
        if key in query:
            return key
    return None

def extract_price_limit(query):
    match = re.search(r'under\s*[₹]?\s*(\d+)', query.lower())
    return int(match.group(1)) if match else None

def detect_direct_flight(query):
    return "direct" in query.lower()

def extract_airline_preference(query):
    query = query.lower()
    return [airline for airline in AIRLINES if airline in query]

def extract_layover_limit(query):
    match = re.search(r'layover.*?(\d{1,2})\s*hours?', query.lower())
    return int(match.group(1)) * 60 if match else None

def extract_trip_duration(query):
    match = re.search(r'(\d+)[-\s]*(day|night)', query.lower())
    return int(match.group(1)) if match else None

def extract_return_date(query):
    match = re.search(r'return(?:ing)?(?: on)? (\d{1,2}[\-/]\d{1,2}(?:[\-/]\d{2,4})?)', query.lower())
    if match:
        try:
            return dateutil.parser.parse(match.group(1), dayfirst=True).strftime("%Y-%m-%d")
        except:
            return None
    return None

def extract_stopover_city(query):
    match = re.search(r'via ([a-zA-Z ]+)', query.lower())
    return match.group(1).strip().title() if match else None

def extract_baggage_pref(query):
    query = query.lower()
    if "hand baggage" in query or "cabin only" in query:
        return "hand"
    elif "check-in" in query:
        return "checked"
    return None

def detect_trip_type(query):
    q = query.lower()
    if "flexible" in q or "any day" in q or "around" in q:
        return "Flexible"
    if "business" in q:
        return "Business"
    if "holiday" in q or "vacation" in q:
        return "Holiday"
    if "urgent" in q or "emergency" in q:
        return "Urgent"
    return "Business"  # default: fast & single-day

def extract_cities_from_llm_response(text):
    # Remove markdown bold (**Delhi** → Delhi)
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # Match flexible sentence like: from Delhi to Mumbai ...
    match = re.search(r'from\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+?)(\s|$)', clean_text, re.IGNORECASE)
    if match:
        from_city = match.group(1).strip().lower()
        to_city = match.group(2).strip().lower()
        print(f"🔍 Fallback matched: from_city='{from_city}', to_city='{to_city}'")
        print("🔎 Trying to map:", from_city, "→", CITY_TO_IATA.get(from_city))
        print("🔎 Trying to map:", to_city, "→", CITY_TO_IATA.get(to_city))
        return resolve_city_to_iata(from_city), resolve_city_to_iata(to_city)
    return None, None

def extract_date_from_query(query):
    import re
    import dateutil.parser

    # Try to extract parts like "20 June" or "20th June"
    match = re.search(r'\b(\d{1,2})(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)', query.lower())
    if match:
        try:
            date_str = f"{match.group(1)} {match.group(3)} 2025"  # Assume this year for now
            dt = dateutil.parser.parse(date_str, dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"[DEBUG] Failed to parse extracted date: {e}")

    # Fallback: try dateutil directly on the whole sentence
    try:
        dt = dateutil.parser.parse(query, fuzzy=True, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"[DEBUG] Fuzzy parsing also failed: {e}")
        return None

def plan_trip(*, origin, destination, date, user_query, trip_type, flights=None):
    """
    Plan a trip with comprehensive flight search and analysis.
    
    Args:
        origin: Departure airport IATA code
        destination: Arrival airport IATA code
        date: Departure date (YYYY-MM-DD)
        user_query: Natural language query from user
        trip_type: Type of trip (Business, Holiday, Flexible, Urgent)
        flights: Optional pre-fetched flight data (for testing)
    
    Returns:
        Dict containing flight recommendations, weather, and LLM response
    """
    print("🔍 Initial origin:", origin, "| destination:", destination)

    parsed_date = extract_date_from_query(user_query)
    if parsed_date:
        date = parsed_date  # ✅ override with date from query
    print("🕵️ Extracted date from query:", parsed_date)    
    search_base = parsed_date if parsed_date else date
    base_date = datetime.strptime(search_base, "%Y-%m-%d")
    date = base_date.strftime("%Y-%m-%d")  # 🔁 ensures `date` is correct everywhere

    # Detect trip type if default or empty
    if not trip_type or trip_type.lower() == "flexible":
        trip_type = detect_trip_type(user_query)

    # Try to auto-detect city names if user typed: "from delhi to mumbai"
    parsed_origin, parsed_dest = extract_cities_from_query(user_query)
    if parsed_origin:
        origin = parsed_origin
    if parsed_dest:
        destination = parsed_dest

    # 🧠 FIX: Try GPT-based city correction early (before searching flights)
    fallback_note = ""
    if not origin or not destination:
        print("❗ Origin or destination missing. Trying GPT correction...")
        
        # ✅ UPDATED: Use llm_router instead of direct OpenAI
        mini_prompt = f"Where is this person flying from and to?\n\nQuery: {user_query}"
        
        # ✅ Pass USE_CLOUD_FALLBACK to the generate function if supported
        try:
            fixed_text = generate(
                prompt=mini_prompt,
                system="You're a travel assistant. Rephrase the route from the user query using correct city names.",
                model="gpt-4o-mini",
                use_cloud_fallback=USE_CLOUD_FALLBACK
            )
        except TypeError:
            # If generate doesn't accept use_cloud_fallback parameter
            fixed_text = generate(
                prompt=mini_prompt,
                system="You're a travel assistant. Rephrase the route from the user query using correct city names.",
                model="gpt-4o-mini"
            )
        
        print("🧠 LLM router fallback city fix response:\n", fixed_text)
        corrected_origin, corrected_dest = extract_cities_from_llm_response(fixed_text)
        if corrected_origin:
            origin = corrected_origin
        if corrected_dest:
            destination = corrected_dest

        if not origin or not destination:
            print("🚫 Still missing origin or destination after LLM correction.")
            return {"error": "❌ Could not determine origin or destination airport after AI correction."}


        if corrected_origin or corrected_dest:
            fallback_note = "✈️ Interpreted your city names using AI. Corrected spelling."
            print(f"🛠️ Early LLM correction: {origin=} {destination=}")

    all_flights = []

    print("trip_type received:", trip_type, type(trip_type))

    # Extract preferences
    time_pref = detect_time_preference(user_query)
    price_limit = extract_price_limit(user_query)
    wants_direct = detect_direct_flight(user_query)
    preferred_airlines = extract_airline_preference(user_query)
    layover_limit = extract_layover_limit(user_query)
    baggage_pref = extract_baggage_pref(user_query)
    duration = extract_trip_duration(user_query)
    return_date = extract_return_date(user_query)
    stopover_city = extract_stopover_city(user_query)


    if not return_date and duration:
        search_base = parsed_date if parsed_date else date
        base_date = datetime.strptime(search_base, "%Y-%m-%d")
        print("🔁 Searching flights around base date:", base_date.strftime("%Y-%m-%d"))

        return_date = (base_date + timedelta(days=duration)).strftime("%Y-%m-%d")

    if stopover_city:
        leg1 = plan_trip(
            origin=origin, 
            destination=stopover_city, 
            date=date, 
            user_query=user_query, 
            trip_type=trip_type
        )
        leg2 = plan_trip(
            origin=stopover_city, 
            destination=destination, 
            date=date, 
            user_query=user_query, 
            trip_type=trip_type
        )
        return {"multicity": True, "legs": [leg1, leg2]}

    is_round_trip = return_date is not None

    if flights is not None:
        print("Using provided flights")
        all_flights = flights
        for f in all_flights:
            if 'date' not in f:
                f['date'] = date
    else:
        # 🛑 Prevent .upper() crash if correction failed
        if not origin or not destination:
            return {"error": "❌ Could not determine origin or destination airport after AI correction."}

        # 🧠 Use parsed date (if found) instead of UI date for searching
        search_base = parsed_date if parsed_date else date
        base_date = datetime.strptime(search_base, "%Y-%m-%d")

        if trip_type.lower() == "flexible":
            search_dates = [(base_date + timedelta(days=delta)).strftime("%Y-%m-%d") for delta in range(-2, 3)]
            for d in search_dates:
                flights_for_day = cached_search(origin, destination, d)
                # Validate flight data
                validated_flights = []
                for f in flights_for_day:
                    validated = validate_flight_data(f)
                    if validated:
                        validated['date'] = d
                        validated_flights.append(validated)
                all_flights.extend(validated_flights)
        else:
            flights_for_day = cached_search(origin, destination, base_date.strftime("%Y-%m-%d"))
            print("📅 Searching for flights on:", base_date.strftime("%Y-%m-%d"))
            # Validate flight data
            validated_flights = []
            for f in flights_for_day:
                validated = validate_flight_data(f)
                if validated:
                    validated['date'] = base_date.strftime("%Y-%m-%d")
                    validated_flights.append(validated)
            all_flights.extend(validated_flights)

        print("🔁 Flights fetched around base date:", base_date.strftime("%Y-%m-%d"))
        print(f"✅ Valid flights found: {len(all_flights)}")
        
        # Explicit fallback when zero flights returned
        if not all_flights:
            print("⚠️ No flights found from API. Returning fallback data.")
            return {
                "warning": "No live flights found. This could be due to: (1) No flights on selected date, (2) API temporarily unavailable, or (3) Route not serviced.",
                "fallback": True,
                "suggestions": [
                    "Try a different date",
                    "Check nearby airports",
                    "Contact airline directly"
                ],
                "search_date": base_date.strftime("%Y-%m-%d")
            }

    if time_pref:
        start_time, end_time = TIME_WINDOWS[time_pref]
    else:
        start_time, end_time = "00:00", "23:59"

    def price_to_int(price):
        try:
            if isinstance(price, int):
                return price
            if isinstance(price, str):
                return int(str(price).replace('₹', '').replace(',', '').strip())
            return 10**9
        except:
            return 10**9


    def passes_filters(f):
        reasons = []

        dep_time = f.get("departure_time", "")[-5:]
        if not (start_time <= dep_time <= end_time):
            reasons.append("time")

        price = price_to_int(f.get("price_inr", "₹999,999"))
        if price_limit and price > price_limit:
            print(f"🔻 Rejected due to price: ₹{price} > ₹{price_limit}")
            reasons.append("price")

        if wants_direct:
            stops = f.get("stops", "").lower()
            if stops and not any(s in stops for s in ["non", "0", "direct"]):
                reasons.append("not direct")

        if preferred_airlines and f.get("airline", "").lower() not in preferred_airlines:
            reasons.append("airline")

        layover_val = f.get("layover_time", "0")
        try:
            layover_min = int(layover_val) if isinstance(layover_val, (int, str)) else 0
        except:
            layover_min = 0
            
        if layover_limit and layover_min > layover_limit:
            reasons.append("layover")

        if baggage_pref:
            baggage = f.get("baggage", "").lower()
            if baggage_pref == "hand" and "hand" not in baggage:
                reasons.append("baggage")
            if baggage_pref == "checked" and "check" not in baggage:
                reasons.append("baggage")

        if reasons:
            print(f"❌ Rejected {f.get('flight_no', 'N/A')} due to: {', '.join(reasons)}")
            return False
        return True


    filtered_flights = [f for f in all_flights if passes_filters(f)]

    # If none survive:
    if not filtered_flights:
        return {"error": "⚠️ Sorry, I couldn't find any flights matching your preferences."}

    print(f"✅ Final origin: {origin} | destination: {destination}")
    
    # Detect intent (cheapest/shortest/balanced)
    flight_pref = detect_flight_preference(user_query)

    # Score & sort flights by weighted price/duration score
    scored_flights = sorted(
        filtered_flights,
        key=lambda f: score_flight(f, flight_pref),
        reverse=True  # highest score first
    )
    best_flight = scored_flights[0]

    # Get weather data
    weather = cached_weather(destination, date)

    flights_str = "\n".join([
        f"- {f['airline']} {f['flight_no']} on {f['date']} | "
        f"{f['departure_time']} → {f['arrival_time']} | "
        f"{f['duration_min']} min | {f['price_inr']}"
        for f in all_flights[:10]
    ])

    # Build filters note for LLM
    filter_note = []
    if time_pref: filter_note.append(f"{time_pref} flights")
    if price_limit: filter_note.append(f"under ₹{price_limit}")
    if wants_direct: filter_note.append("direct flights only")
    if preferred_airlines: filter_note.append(f"preferred airlines: {', '.join(preferred_airlines)}")
    if layover_limit: filter_note.append(f"max layover: {layover_limit // 60}h")
    if baggage_pref: filter_note.append(f"{baggage_pref} baggage only")
    filters_applied = "; ".join(filter_note) if filter_note else "no specific filters"

    trip_description = f"a {trip_type} trip"
    if stopover_city:
        trip_description += f" via {stopover_city}"
    if return_date:
        trip_description += f", returning on {return_date}"

    prompt = f"""
You are a helpful travel assistant helping a user plan {trip_description}.

User preferences:
- {filters_applied}

Flight options from {origin} to {destination} around {date}:
{flights_str}

Best matching flight:
- {best_flight['airline']} {best_flight['flight_no']} on {best_flight['date']} |
  {best_flight['departure_time']} → {best_flight['arrival_time']} |
  Duration: {best_flight['duration_min']} minutes |
  Price: {best_flight['price_inr']} |
  Stops: {best_flight.get('stops', 'N/A')} |
  Baggage: {best_flight.get('baggage', 'N/A')}

Weather forecast for {destination} on {date}:
{weather}

User's question: {user_query}

Please recommend the best flight, explain why it matches their preferences, mention the weather suitability, and answer the user's query helpfully.
"""

    print("📨 Prompt to LLM:\n", prompt)

    try:
        # ✅ UPDATED: Use llm_router.generate() instead of direct OpenAI
        llm_text = generate(
            prompt=prompt,
            system="You are a professional travel planning assistant.",
            model="gpt-4o-mini",
            use_cloud_fallback=USE_CLOUD_FALLBACK
        )
        
        # Prepare result object
        result = {
            "llm_response": llm_text,
            "best_flight": best_flight,
            "weather": weather,
            "fallback_note": fallback_note,
            "all_flights": all_flights,
            "filtered_flights": filtered_flights,
            "search_date": base_date.strftime("%Y-%m-%d")
        }

        if is_round_trip:
            result["return_trip"] = plan_trip(
                origin=destination,
                destination=origin,
                date=return_date,
                user_query=user_query,
                trip_type=trip_type
            )

        print("✅ Returning search date:", result["search_date"])
        
        # 🧾 LOG SESSION TO DATABASE
        # 🔧 Fix 3: Add agent_version to DB logs (great for audits)
        save_session(
            user_query=user_query,
            agent_reasoning={
                "version": "planner-v1.1",
                "prompt": prompt,
                "scored_flights_count": len(scored_flights),
                "flight_preference": flight_pref,
                "trip_type": trip_type,
                "use_cloud_llm": USE_CLOUD_FALLBACK
            },
            tool_output={
                "all_flights_count": len(all_flights),
                "weather": weather,
                "origin": origin,
                "destination": destination,
                "search_date": base_date.strftime("%Y-%m-%d")
            },
            final_response=llm_text
        )
        
        return result

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        
        # 🧾 LOG ERROR SESSION TOO
        save_session(
            user_query=user_query,
            agent_reasoning={
                "version": "planner-v1.1",
                "prompt": prompt,
                "scored_flights_count": len(scored_flights),
                "flight_preference": flight_pref,
                "trip_type": trip_type,
                "use_cloud_llm": USE_CLOUD_FALLBACK,
                "error": str(e)
            },
            tool_output={
                "all_flights_count": len(all_flights),
                "weather": weather,
                "origin": origin,
                "destination": destination,
                "search_date": base_date.strftime("%Y-%m-%d")
            },
            final_response=f"LLM call failed: {e}"
        )
        
        return {"error": f"LLM call failed: {e}"}