"""
Planner Agent (Brain Layer)

Responsibilities:
- Parse user intent
- Retrieve and validate flight & weather data
- Apply preference-aware scoring
- Generate LLM explanations
- Persist full audit trail to PostgreSQL

UI-agnostic, FastAPI-ready.
Fully async with production-grade improvements.
Supports both blocking (full result) and streaming (token-by-token) responses.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from difflib import get_close_matches
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import dateutil.parser
from airportsdata import load
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

# Optional: better date parsing
try:
    import dateparser
    HAS_DATEPARSER = True
except ImportError:
    HAS_DATEPARSER = False

# Use TTLCache for bounded cache
try:
    from cachetools import TTLCache
except ImportError:
    TTLCache = None

# Local imports (will be used as defaults for injection)
from tools.airline_api import search_flights as default_flight_tool
from tools.weather_api import check_weather as default_weather_tool
from agents.llm_router import generate

# Metrics instrumentation
import core.metrics as metrics

load_dotenv()

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logger = logging.getLogger("planner_agent")

# ----------------------------------------------------------------------
# Environment flags & configurable timeouts
# ----------------------------------------------------------------------
USE_CLOUD_FALLBACK = os.getenv("USE_CLOUD_LLM", "1") == "1"
PLANNER_LLM_MODEL = os.getenv("PLANNER_LLM_MODEL", "gpt-4o-mini")
PLANNER_LLM_TIMEOUT = float(os.getenv("PLANNER_LLM_TIMEOUT", "29"))
STREAM_INIT_TIMEOUT = float(os.getenv("PLANNER_STREAM_INIT_TIMEOUT", "5"))

logger.info(
    f"LLM Configuration: USE_CLOUD_FALLBACK={USE_CLOUD_FALLBACK}, "
    f"MODEL={PLANNER_LLM_MODEL}, TIMEOUT={PLANNER_LLM_TIMEOUT}s"
)

# ----------------------------------------------------------------------
# Database session logging
# ----------------------------------------------------------------------
try:
    from agents.database import SessionLocal, SessionHistory
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("Database module not available. Session logging disabled.")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MAX_RECURSION_DEPTH = 3

# ----------------------------------------------------------------------
# LLM circuit breaker with auto-recovery
# ----------------------------------------------------------------------
_llm_failures = 0
_llm_failure_lock = asyncio.Lock()
LLM_FAILURE_THRESHOLD = 5
LLM_CIRCUIT_OPEN = False
LLM_CIRCUIT_RESET_TIMEOUT = 120  # seconds
_llm_circuit_reset_time: Optional[float] = None

async def check_llm_circuit() -> bool:
    """Return True if circuit is open (skip LLM). Handles auto-recovery."""
    global _llm_failures, LLM_CIRCUIT_OPEN, _llm_circuit_reset_time
    async with _llm_failure_lock:
        now = time.monotonic()
        # Auto-recover if timeout elapsed
        if LLM_CIRCUIT_OPEN and _llm_circuit_reset_time and now > _llm_circuit_reset_time:
            logger.info("LLM circuit breaker reset after timeout")
            LLM_CIRCUIT_OPEN = False
            _llm_failures = 0
            _llm_circuit_reset_time = None

        if _llm_failures >= LLM_FAILURE_THRESHOLD:
            if not LLM_CIRCUIT_OPEN:
                logger.warning("LLM circuit breaker OPEN")
                LLM_CIRCUIT_OPEN = True
                _llm_circuit_reset_time = now + LLM_CIRCUIT_RESET_TIMEOUT
        return LLM_CIRCUIT_OPEN

async def record_llm_success():
    """Reset failure count on success."""
    global _llm_failures, LLM_CIRCUIT_OPEN, _llm_circuit_reset_time
    async with _llm_failure_lock:
        _llm_failures = 0
        LLM_CIRCUIT_OPEN = False
        _llm_circuit_reset_time = None

async def record_llm_failure():
    """Increment failure count."""
    global _llm_failures
    async with _llm_failure_lock:
        _llm_failures += 1
        logger.warning(f"LLM failure count: {_llm_failures}")

# ----------------------------------------------------------------------
# Async-safe cache decorator with per-key locks and bounded TTL cache
# ----------------------------------------------------------------------
_cache_locks: Dict[Any, asyncio.Lock] = {}

def _get_cache_lock(key):
    """Lazily create a lock for a cache key to avoid event loop binding issues."""
    # Prevent unbounded growth of lock dictionary
    if len(_cache_locks) > 5000:
        _cache_locks.clear()
        logger.warning("Cache lock dictionary cleared due to size limit")

    if key not in _cache_locks:
        _cache_locks[key] = asyncio.Lock()
    return _cache_locks[key]

def async_cache(ttl: int, maxsize: int = 1000):
    """
    Decorator that caches the result of an async function for `ttl` seconds.
    Uses per-key locks to prevent cache stampede and bounded cache to limit memory.
    """
    def decorator(func):
        # Use TTLCache if available, else simple dict (unbounded)
        if TTLCache:
            cache = TTLCache(maxsize=maxsize, ttl=ttl)
        else:
            cache = {}
            logger.warning("cachetools not installed, using unbounded cache (memory may grow)")

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            key = (func.__name__, args, frozenset(kwargs.items()))
            lock = _get_cache_lock(key)

            async with lock:
                if TTLCache:
                    if key in cache:
                        logger.debug(f"Cache hit for {func.__name__}")
                        return cache[key]
                else:
                    now = time.monotonic()
                    if key in cache:
                        result, timestamp = cache[key]
                        if now - timestamp < ttl:
                            logger.debug(f"Cache hit for {func.__name__}")
                            return result

                logger.debug(f"Cache miss for {func.__name__}")
                result = await func(*args, **kwargs)

                if TTLCache:
                    cache[key] = result
                else:
                    cache[key] = (result, now)

                return result

        return wrapper
    return decorator

def create_cached_fetcher(ttl: int, maxsize: int, fetch_func: Callable):
    """
    Create an async cached version of a fetch function.
    The fetch function must accept the same arguments each time.
    """
    @async_cache(ttl=ttl, maxsize=maxsize)
    async def cached(*args, **kwargs):
        return await fetch_func(*args, **kwargs)
    return cached

# ----------------------------------------------------------------------
# Shared module-level cached tool instances (system-wide cache)
# ----------------------------------------------------------------------
_shared_cached_search = create_cached_fetcher(
    ttl=900,
    maxsize=500,
    fetch_func=default_flight_tool
)

_shared_cached_weather = create_cached_fetcher(
    ttl=3600,
    maxsize=500,
    fetch_func=default_weather_tool
)

# ----------------------------------------------------------------------
# Helper to normalize raw flight data into Flight objects
# ----------------------------------------------------------------------
def normalize_flights(raw_flights: List[Any], default_date: str) -> List['Flight']:
    normalized = []
    for f in raw_flights:

        # Case 1: dict
        if isinstance(f, dict):
            flight_data = dict(f)

        # Case 2: tool Flight or any object with __dict__
        elif hasattr(f, "__dict__"):
            flight_data = dict(vars(f))

        else:
            logger.warning(f"Skipping unknown flight type: {type(f)}")
            continue

        # Ensure date
        if 'date' not in flight_data or not flight_data.get('date'):
            flight_data['date'] = default_date

        try:
            normalized.append(Flight(**flight_data))
        except ValidationError as e:
            logger.warning(f"Skipping invalid flight after conversion: {e}")

    return normalized

# ----------------------------------------------------------------------
# Pydantic models for validation and structured output
# ----------------------------------------------------------------------
class Flight(BaseModel):
    """Validated flight data model."""
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
        if isinstance(v, int):
            return f"₹{v:,}"
        if isinstance(v, str):
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
        if isinstance(v, str):
            match = re.search(r'(\d{1,2}):(\d{2})', v)
            if match:
                hour, minute = match.groups()
                return f"{int(hour):02d}:{minute}"
        return "00:00"

    @field_validator('duration_min', mode='before')
    @classmethod
    def validate_duration(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                numbers = re.findall(r'\d+', v)
                if numbers:
                    return int(numbers[0])
            except:
                pass
        return 999

class PlanResult(BaseModel):
    """Structured output of the planning process."""
    llm_response: Optional[str]  # None when skip_llm=True
    best_flight: Dict[str, Any]
    weather: Dict[str, Any]
    search_date: str
    warnings: Optional[List[str]] = None
    debug_info: Optional[Dict[str, Any]] = None  # Internal metrics + extra data for streaming
    return_trip: Optional['PlanResult'] = None
    fallback_note: str = ""

class MultiCityResult(BaseModel):
    """Structured output for multi-city trips."""
    multicity: bool = True
    legs: List[PlanResult]

PlanResult.model_rebuild()

# ----------------------------------------------------------------------
# City → IATA mapping with aliases and fuzzy matching
# ----------------------------------------------------------------------
def build_city_to_iata_map() -> Dict[str, str]:
    airports = load('IATA')
    city_map = {}
    for code, data in airports.items():
        city = data.get('city') or data.get('city_en') or data.get('location')
        if city:
            city = city.strip().lower()
            if city not in city_map:
                city_map[city] = code
    return city_map

# Add common aliases
CITY_ALIASES = {
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "bangalore": "bengaluru",
    "pune": "pune",
    "delhi": "delhi",
    "new delhi": "delhi",
}

def resolve_city_to_iata(city_name: str) -> Optional[str]:
    name = city_name.strip().lower()
    # Apply alias mapping
    name = CITY_ALIASES.get(name, name)
    # Try exact match first
    if name in CITY_TO_IATA:
        return CITY_TO_IATA[name]
    # Fallback: substring match
    for city_key, code in CITY_TO_IATA.items():
        if name in city_key or city_key in name:
            return code
    # Fallback: fuzzy matching
    close = get_close_matches(name, CITY_TO_IATA.keys(), n=1, cutoff=0.8)
    if close:
        return CITY_TO_IATA[close[0]]
    return None

def normalize_airport(input_value: Optional[str]) -> Optional[str]:
    """
    If user already provided a valid 3-letter IATA code,
    return it directly without re-resolving.

    Otherwise fall back to city → IATA resolver.
    """
    if not input_value:
        return None

    value = input_value.strip()

    # If already 3-letter airport code, trust it
    if len(value) == 3 and value.isalpha():
        return value.upper()

    # Otherwise resolve city name
    return resolve_city_to_iata(value)

CITY_TO_IATA = build_city_to_iata_map()
logger.info(f"Loaded city map: {list(CITY_TO_IATA.items())[:10]}")

# ----------------------------------------------------------------------
# Preference extraction constants
# ----------------------------------------------------------------------
TIME_WINDOWS = {
    "morning": ("04:00", "11:59"),
    "afternoon": ("12:00", "17:59"),
    "evening": ("18:00", "23:59"),
    "night": ("00:00", "03:59")
}
AIRLINES = ["indigo", "air india", "vistara", "goair", "spicejet", "akasa", "airasia"]

# ----------------------------------------------------------------------
# Intent parsing (pure logic, no IO)
# ----------------------------------------------------------------------
class ParsedIntent(BaseModel):
    """All extracted information from the user query."""
    origin_iata: Optional[str] = None
    destination_iata: Optional[str] = None
    date: Optional[str] = None
    return_date: Optional[str] = None
    time_pref: Optional[str] = None
    price_limit: Optional[int] = None
    wants_direct: bool = False
    preferred_airlines: List[str] = Field(default_factory=list)
    layover_limit_minutes: Optional[int] = None
    baggage_pref: Optional[str] = None
    trip_duration_days: Optional[int] = None
    stopover_city: Optional[str] = None
    flight_pref: str = "default"
    trip_type: str = "Business"

def parse_intent(user_query: str) -> ParsedIntent:
    """Extract all structured data from the natural language query."""
    q = user_query.lower()
    intent = ParsedIntent()

    # --- Cities (with improved regex and alias support) ---
    # Match "from X to Y"
    pattern = re.search(r'\bfrom\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\.|,)', q)
    if pattern:
        from_city = pattern.group(1).strip().lower()
        to_city = pattern.group(2).strip().lower()
        # Use normalize_airport so that 3-letter codes are accepted directly
        intent.origin_iata = normalize_airport(from_city)
        intent.destination_iata = normalize_airport(to_city)

    # If not found, try matching "X to Y" (without "from")
    if not intent.origin_iata or not intent.destination_iata:
        pattern2 = re.search(r'\b([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\.|,)', q)
        if pattern2:
            from_city = pattern2.group(1).strip().lower()
            to_city = pattern2.group(2).strip().lower()
            # Guard against overly long captures (e.g., "travel to mumbai")
            # Also require that both cities actually resolve to IATA codes
            origin_iata = normalize_airport(from_city)
            dest_iata = normalize_airport(to_city)
            if origin_iata and dest_iata:
                intent.origin_iata = origin_iata
                intent.destination_iata = dest_iata

    # --- Date parsing ---
    today = datetime.now().date()
    parsed_date = None

    if HAS_DATEPARSER:
        settings = {'PREFER_DATES_FROM': 'future', 'DATE_ORDER': 'DMY'}
        parsed_date = dateparser.parse(q, settings=settings)
    else:
        # Fallback regex
        date_match = re.search(r'\b(\d{1,2})(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b', q)
        if date_match:
            day, _, month = date_match.groups()
            year = today.year
            try:
                parsed_date = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
            except:
                pass

    if not parsed_date:
        try:
            parsed_date = dateutil.parser.parse(q, fuzzy=True, dayfirst=True, default=datetime.now().replace(month=1, day=1))
        except:
            pass

    if parsed_date:
        if parsed_date.year == today.year and parsed_date.date() < today:
            parsed_date = parsed_date.replace(year=today.year + 1)
        intent.date = parsed_date.strftime("%Y-%m-%d")

    # --- Time preference ---
    for key in TIME_WINDOWS:
        if key in q:
            intent.time_pref = key
            break

    # --- Price limit ---
    price_match = re.search(r'under\s*[₹]?\s*(\d+)', q)
    if price_match:
        intent.price_limit = int(price_match.group(1))

    # --- Direct flights ---
    intent.wants_direct = "direct" in q

    # --- Airline preference ---
    intent.preferred_airlines = [a for a in AIRLINES if a in q]

    # --- Layover limit ---
    layover_match = re.search(r'layover.*?(\d{1,2})\s*hours?', q)
    if layover_match:
        intent.layover_limit_minutes = int(layover_match.group(1)) * 60

    # --- Baggage preference ---
    if "hand baggage" in q or "cabin only" in q:
        intent.baggage_pref = "hand"
    elif "check-in" in q:
        intent.baggage_pref = "checked"

    # --- Trip duration (for return) ---
    duration_match = re.search(r'(\d+)[-\s]*(day|night)', q)
    if duration_match:
        intent.trip_duration_days = int(duration_match.group(1))

    # --- Return date explicit ---
    return_match = re.search(r'return(?:ing)?(?: on)? (\d{1,2}[\-/]\d{1,2}(?:[\-/]\d{2,4})?)', q)
    if return_match:
        try:
            dt = dateutil.parser.parse(return_match.group(1), dayfirst=True)
            intent.return_date = dt.strftime("%Y-%m-%d")
        except:
            pass

    # --- Stopover city ---
    via_match = re.search(r'via ([a-zA-Z ]+)', q)
    if via_match:
        intent.stopover_city = via_match.group(1).strip().title()

    # --- Flight preference ---
    if "cheapest" in q or "lowest price" in q or "budget" in q:
        intent.flight_pref = "cheapest"
    elif "shortest" in q or "fastest" in q or "least time" in q or "quickest" in q:
        intent.flight_pref = "shortest"
    elif "balanced" in q or ("price" in q and "duration" in q):
        intent.flight_pref = "balanced"

    # --- Trip type ---
    if "flexible" in q or "any day" in q or "around" in q:
        intent.trip_type = "Flexible"
    elif "business" in q:
        intent.trip_type = "Business"
    elif "holiday" in q or "vacation" in q:
        intent.trip_type = "Holiday"
    elif "urgent" in q or "emergency" in q:
        intent.trip_type = "Urgent"

    return intent

# ----------------------------------------------------------------------
# Filtering and ranking (pure logic, using Flight objects)
# ----------------------------------------------------------------------
def price_to_int(price: Union[str, int]) -> int:
    if isinstance(price, int):
        return price
    try:
        return int(str(price).replace('₹', '').replace(',', '').strip())
    except:
        return 10**9

def filter_flights(flights: List[Flight], intent: ParsedIntent) -> List[Flight]:
    """Apply all user filters to the flight list."""
    filtered = []
    for f in flights:
        reasons = []
        if intent.time_pref:
            start, end = TIME_WINDOWS[intent.time_pref]
            dep = f.departure_time[-5:]
            if not (start <= dep <= end):
                reasons.append("time")
        if intent.price_limit:
            price = price_to_int(f.price_inr)
            if price > intent.price_limit:
                reasons.append("price")
        if intent.wants_direct:
            stops = f.stops.lower()
            if stops and not any(s in stops for s in ["non", "0", "direct"]):
                reasons.append("not direct")
        if intent.preferred_airlines:
            airline = f.airline.lower()
            if not any(pref in airline for pref in intent.preferred_airlines):
                reasons.append("airline")
        if intent.layover_limit_minutes:
            layover_val = f.layover_time
            try:
                layover_min = int(layover_val)
            except:
                layover_min = 0
            if layover_min > intent.layover_limit_minutes:
                reasons.append("layover")
        if intent.baggage_pref:
            baggage = f.baggage.lower()
            if intent.baggage_pref == "hand" and "hand" not in baggage:
                reasons.append("baggage")
            if intent.baggage_pref == "checked" and "check" not in baggage:
                reasons.append("baggage")

        if not reasons:
            filtered.append(f)
        else:
            logger.debug(f"Flight {f.flight_no} rejected: {reasons}")
    return filtered

def get_weights(pref: str) -> Tuple[float, float]:
    if pref == "cheapest":
        return 0.8, 0.2
    elif pref == "shortest":
        return 0.2, 0.8
    elif pref == "balanced":
        return 0.5, 0.5
    return 0.6, 0.4

def rank_flights(flights: List[Flight], intent: ParsedIntent) -> List[Flight]:
    """
    Rank flights by normalized price and duration.
    Normalization values computed once to avoid O(n²).
    """
    if not flights:
        return []
    prices = [price_to_int(f.price_inr) for f in flights]
    durations = [f.duration_min for f in flights]
    min_price, max_price = min(prices), max(prices)
    min_dur, max_dur = min(durations), max(durations)

    wp, wd = get_weights(intent.flight_pref)

    def score(f: Flight) -> float:
        price = price_to_int(f.price_inr)
        duration = f.duration_min
        price_norm = (max_price - price) / (max_price - min_price) if max_price > min_price else 1.0
        dur_norm = (max_dur - duration) / (max_dur - min_dur) if max_dur > min_dur else 1.0
        return price_norm * wp + dur_norm * wd

    scored = [(f, score(f)) for f in flights]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scored]

# ----------------------------------------------------------------------
# LLM explanation generation with timeout and circuit breaker (non-streaming)
# ----------------------------------------------------------------------
async def generate_explanation(
    user_query: str,
    intent: ParsedIntent,
    best_flight: Flight,
    weather: Dict,
    all_flights: List[Flight],
    filters_applied: str,
    trip_description: str,
) -> str:
    """Call LLM to produce a natural language response, with timeout and circuit breaker."""
    if await check_llm_circuit():
        logger.warning("LLM circuit breaker open, returning deterministic summary")
        return generate_deterministic_summary(best_flight, weather, filters_applied)

    flights_str = "\n".join([
        f"- {f.airline} {f.flight_no} on {f.date or 'N/A'} | "
        f"{f.departure_time} → {f.arrival_time} | "
        f"{f.duration_min} min | {f.price_inr}"
        for f in all_flights[:10]
    ])

    prompt = f"""
You are a helpful travel assistant helping a user plan {trip_description}.

User preferences:
- {filters_applied}

Flight options from {intent.origin_iata} to {intent.destination_iata} around {intent.date}:
{flights_str}

Best matching flight:
- {best_flight.airline} {best_flight.flight_no} on {best_flight.date or 'N/A'} |
  {best_flight.departure_time} → {best_flight.arrival_time} |
  Duration: {best_flight.duration_min} minutes |
  Price: {best_flight.price_inr} |
  Stops: {best_flight.stops} |
  Baggage: {best_flight.baggage}

Weather forecast for {intent.destination_iata}:
{weather}

User's question: {user_query}

Please recommend the best flight, explain why it matches their preferences, mention the weather suitability, and answer the user's query helpfully.
"""

    logger.info("Sending prompt to LLM")
    try:
        llm_text = await asyncio.wait_for(
            generate(
                prompt=prompt,
                system="You are a professional travel planning assistant.",
                model=PLANNER_LLM_MODEL,
                stream=False  # explicitly non-streaming
            ),
            timeout=PLANNER_LLM_TIMEOUT
        )
        await record_llm_success()
        return llm_text
    except asyncio.TimeoutError:
        logger.error(f"LLM call timed out after {PLANNER_LLM_TIMEOUT}s")
        await record_llm_failure()
        return generate_deterministic_summary(best_flight, weather, filters_applied, error="timed out")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        await record_llm_failure()
        return generate_deterministic_summary(best_flight, weather, filters_applied, error=str(e))

def generate_deterministic_summary(best_flight: Flight, weather: Dict, filters: str, error: str = "") -> str:
    """Fallback summary when LLM is unavailable."""
    if isinstance(weather, dict):
        condition = weather.get("condition", weather.get("description", "unknown"))
        temp = weather.get("temperature_c", weather.get("temp", "unknown"))
    else:
        # Weather object
        condition = getattr(weather, "condition", "unknown")
        temp = getattr(weather, "temperature_c", getattr(weather, "temp", "unknown"))

    weather_str = f"{condition}, {temp}°C" if temp != "unknown" else condition

    base = (f"I recommend {best_flight.airline} {best_flight.flight_no} at "
            f"{best_flight.departure_time} arriving {best_flight.arrival_time}. "
            f"Duration: {best_flight.duration_min} minutes, Price: {best_flight.price_inr}. "
            f"Weather at destination: {weather_str}. ")
    if error:
        base += f"(Note: Enhanced explanation unavailable due to {error}.)"
    return base

# ----------------------------------------------------------------------
# City correction using LLM (with circuit breaker)
# ----------------------------------------------------------------------
async def correct_cities_with_llm(user_query: str) -> Tuple[Optional[str], Optional[str]]:
    """Attempt to extract origin/destination IATA codes using LLM, respecting circuit breaker."""
    if await check_llm_circuit():
        logger.warning("LLM circuit breaker open, skipping city correction")
        return None, None

    mini_prompt = f"""
Extract the route from this query.

Return EXACTLY in this format:
from: <origin city>
to: <destination city>

Query: {user_query}
"""
    try:
        fixed_text = await generate(
            prompt=mini_prompt,
            system="You are a precise travel assistant. Always output the requested format.",
            model=PLANNER_LLM_MODEL,
            stream=False
        )
        from_match = re.search(r'from:\s*(.+)', fixed_text, re.IGNORECASE)
        to_match = re.search(r'to:\s*(.+)', fixed_text, re.IGNORECASE)
        if from_match and to_match:
            from_city = from_match.group(1).strip().lower()
            to_city = to_match.group(1).strip().lower()
            return resolve_city_to_iata(from_city), resolve_city_to_iata(to_city)
    except Exception as e:
        logger.warning(f"LLM city correction failed: {e}")
    return None, None

# ----------------------------------------------------------------------
# Main internal planning function (async, layered, with dependency injection)
# ----------------------------------------------------------------------
async def _plan_trip_internal(
    *,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    date: Optional[str] = None,
    user_query: str,
    trip_type: Optional[str] = None,
    flights: Optional[List[Union[Dict, Flight]]] = None,
    depth: int = 0,
    flight_tool: Callable = default_flight_tool,
    weather_tool: Callable = default_weather_tool,
    skip_llm: bool = False   # NEW: if True, return data without LLM explanation
) -> Union[PlanResult, MultiCityResult, Dict]:
    """Internal implementation without top-level timeout. Used for non‑streaming mode."""
    if depth > MAX_RECURSION_DEPTH:
        logger.error("Max recursion depth reached")
        return {"error": "Too deep recursion in trip planning"}

    # Use shared module-level cache unless custom tools are injected
    cached_search = (
        _shared_cached_search
        if flight_tool is default_flight_tool
        else create_cached_fetcher(900, 500, flight_tool)
    )

    cached_weather = (
        _shared_cached_weather
        if weather_tool is default_weather_tool
        else create_cached_fetcher(3600, 500, weather_tool)
    )

    # Phase timing
    phases = {}

    # ------------------------------------------------------------------
    # 1. Parse intent (overrides explicit params)
    # ------------------------------------------------------------------
    start = time.monotonic()
    intent = parse_intent(user_query)

    # Override with explicit parameters if provided, using normalize_airport for codes
    if origin:
        intent.origin_iata = normalize_airport(origin)
    if destination:
        intent.destination_iata = normalize_airport(destination)
    if date:
        intent.date = date
    if trip_type:
        intent.trip_type = trip_type

    phases['intent_parsing'] = time.monotonic() - start

    # If we still lack origin/destination, try LLM correction
    if not intent.origin_iata or not intent.destination_iata:
        logger.info("Missing origin/destination, attempting LLM correction")
        start = time.monotonic()
        corrected_origin, corrected_dest = await correct_cities_with_llm(user_query)
        if corrected_origin:
            intent.origin_iata = corrected_origin
        if corrected_dest:
            intent.destination_iata = corrected_dest
        phases['city_correction'] = time.monotonic() - start

    if not intent.origin_iata or not intent.destination_iata:
        return {"error": "Could not determine origin or destination airport after AI correction."}

    # ------------------------------------------------------------------
    # 2. Determine search date and return date
    # ------------------------------------------------------------------
    if intent.date:
        try:
            base_date = datetime.strptime(intent.date, "%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"Invalid date format '{intent.date}'; expected YYYY-MM-DD") from e
    else:
        base_date = datetime.now()
    search_date = base_date.strftime("%Y-%m-%d")

    if not intent.return_date and intent.trip_duration_days:
        intent.return_date = (base_date + timedelta(days=intent.trip_duration_days)).strftime("%Y-%m-%d")

    # Handle stopover (multicity) recursively
    if intent.stopover_city:
        stopover_iata = resolve_city_to_iata(intent.stopover_city)
        if not stopover_iata:
            return {"error": f"Could not resolve stopover city: {intent.stopover_city}"}
        leg1 = await _plan_trip_internal(
            origin=intent.origin_iata,
            destination=stopover_iata,
            date=search_date,
            user_query=user_query,
            trip_type=intent.trip_type,
            depth=depth+1,
            flight_tool=flight_tool,
            weather_tool=weather_tool,
            skip_llm=skip_llm
        )
        leg2 = await _plan_trip_internal(
            origin=stopover_iata,
            destination=intent.destination_iata,
            date=search_date,
            user_query=user_query,
            trip_type=intent.trip_type,
            depth=depth+1,
            flight_tool=flight_tool,
            weather_tool=weather_tool,
            skip_llm=skip_llm
        )
        # Type safety: ensure legs are PlanResult
        if not isinstance(leg1, PlanResult):
            return leg1
        if not isinstance(leg2, PlanResult):
            return leg2
        return MultiCityResult(legs=[leg1, leg2])

    # ------------------------------------------------------------------
    # 3. Fetch flights and weather in parallel
    # ------------------------------------------------------------------
    weather_task: Optional[asyncio.Task] = None
    all_flights: List[Flight] = []

    if flights is not None:
        # Normalize provided flights (could be dicts or Flight objects)
        all_flights = normalize_flights(flights, search_date)
        weather_task = asyncio.create_task(cached_weather(intent.destination_iata))
    else:
        if intent.trip_type.lower() == "flexible":
            dates = [(base_date + timedelta(days=delta)).strftime("%Y-%m-%d") for delta in range(-2, 3)]
            flight_tasks = [asyncio.create_task(cached_search(intent.origin_iata, intent.destination_iata, d)) for d in dates]
            weather_task = asyncio.create_task(cached_weather(intent.destination_iata))
            start = time.monotonic()
            results = await asyncio.gather(*flight_tasks)
            phases['flight_search'] = time.monotonic() - start
            # Normalize each day's flights
            all_flights = []
            for day_flights in results:
                all_flights.extend(normalize_flights(day_flights, search_date))
        else:
            flight_task = asyncio.create_task(cached_search(intent.origin_iata, intent.destination_iata, search_date))
            weather_task = asyncio.create_task(cached_weather(intent.destination_iata))
            start = time.monotonic()
            raw_flights = await flight_task
            phases['flight_search'] = time.monotonic() - start
            all_flights = normalize_flights(raw_flights, search_date)

    logger.info(
        "Flights fetched",
        extra={
            "origin": intent.origin_iata,
            "destination": intent.destination_iata,
            "count": len(all_flights)
        }
    )

    if not all_flights:
        if weather_task:
            start = time.monotonic()
            weather = await weather_task
            phases['weather_fetch'] = time.monotonic() - start
        else:
            weather = {}
        return {
            "warning": "No live flights found.",
            "fallback": True,
            "suggestions": ["Try a different date", "Check nearby airports"],
            "search_date": search_date,
            "weather": weather
        }

    # Await weather (only once)
    if weather_task:
        start = time.monotonic()
        weather = await weather_task
        phases['weather_fetch'] = time.monotonic() - start
    else:
        weather = {}

    # ------------------------------------------------------------------
    # 4. Apply filters and rank
    # ------------------------------------------------------------------
    start = time.monotonic()
    filtered = filter_flights(all_flights, intent)
    filtered_count = len(filtered)
    if not filtered:
        return {"error": "Sorry, I couldn't find any flights matching your preferences."}
    ranked = rank_flights(filtered, intent)
    best_flight = ranked[0]
    ranked_count = len(ranked)
    phases['filter_rank'] = time.monotonic() - start

    # ------------------------------------------------------------------
    # 5. Build description for LLM
    # ------------------------------------------------------------------
    filter_parts = []
    if intent.time_pref: filter_parts.append(f"{intent.time_pref} flights")
    if intent.price_limit: filter_parts.append(f"under ₹{intent.price_limit}")
    if intent.wants_direct: filter_parts.append("direct flights only")
    if intent.preferred_airlines: filter_parts.append(f"preferred airlines: {', '.join(intent.preferred_airlines)}")
    if intent.layover_limit_minutes: filter_parts.append(f"max layover: {intent.layover_limit_minutes//60}h")
    if intent.baggage_pref: filter_parts.append(f"{intent.baggage_pref} baggage only")
    filters_applied = "; ".join(filter_parts) if filter_parts else "no specific filters"

    trip_description = f"a {intent.trip_type} trip"
    if intent.stopover_city:
        trip_description += f" via {intent.stopover_city}"
    if intent.return_date:
        trip_description += f", returning on {intent.return_date}"

    # ------------------------------------------------------------------
    # 6. Convert weather to dict for consistent serialization (safe version)
    # ------------------------------------------------------------------
    if hasattr(weather, "model_dump"):
        weather_dict = weather.model_dump()
    elif hasattr(weather, "__dict__"):
        weather_dict = dict(vars(weather))
    elif isinstance(weather, dict):
        weather_dict = weather
    else:
        # Last-resort safe serialization
        weather_dict = json.loads(json.dumps(weather, default=str))

    # ------------------------------------------------------------------
    # 7. Prepare debug_info with extra data needed for streaming (if skip_llm)
    # ------------------------------------------------------------------
    debug_info = {"phases": phases.copy()}
    if skip_llm:
        # Include all data needed to reconstruct the prompt later
        debug_info["intent"] = intent.model_dump()
        debug_info["filters_applied"] = filters_applied
        debug_info["trip_description"] = trip_description
        debug_info["all_flights"] = [f.model_dump() for f in all_flights]  # for prompt building
        debug_info["filtered_count"] = filtered_count   # for accurate logging
        debug_info["ranked_count"] = ranked_count
        # phases are already under "phases"

    # ------------------------------------------------------------------
    # 8. Generate LLM explanation (if not skipped)
    # ------------------------------------------------------------------
    if skip_llm:
        llm_text = None
    else:
        start = time.monotonic()
        llm_text = await generate_explanation(
            user_query=user_query,
            intent=intent,
            best_flight=best_flight,
            weather=weather_dict,
            all_flights=all_flights,
            filters_applied=filters_applied,
            trip_description=trip_description
        )
        phases['llm_generation'] = time.monotonic() - start
        # Update debug_info with phases after LLM
        debug_info["phases"] = phases.copy()   # refresh after LLM

    # ------------------------------------------------------------------
    # 9. Prepare result and handle round-trip
    # ------------------------------------------------------------------
    result = PlanResult(
        llm_response=llm_text,
        best_flight=best_flight.model_dump(),
        weather=weather_dict,
        search_date=search_date,
        fallback_note="",
        debug_info=debug_info
    )

    if intent.return_date:
        start = time.monotonic()
        return_trip_result = await _plan_trip_internal(
            origin=intent.destination_iata,
            destination=intent.origin_iata,
            date=intent.return_date,
            user_query=user_query,
            trip_type=intent.trip_type,
            depth=depth+1,
            flight_tool=flight_tool,
            weather_tool=weather_tool,
            skip_llm=skip_llm
        )
        phases['return_trip'] = time.monotonic() - start
        if isinstance(return_trip_result, PlanResult):
            result.return_trip = return_trip_result

    # ------------------------------------------------------------------
    # 10. Log session (only if not skipping LLM)
    # ------------------------------------------------------------------
    if DB_AVAILABLE and not skip_llm:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    save_session,
                    user_query=user_query,
                    agent_reasoning={
                        "version": "planner-v7-normalized",
                        "intent": intent.model_dump(),
                        "filters_applied": filters_applied,
                        "ranked_count": len(ranked),
                        "flight_pref": intent.flight_pref,
                        "trip_type": intent.trip_type,
                        "use_cloud_llm": USE_CLOUD_FALLBACK,
                        "phases": phases
                    },
                    tool_output={
                        "all_flights_count": len(all_flights),
                        "filtered_count": filtered_count,
                        "weather": weather_dict,
                        "origin": intent.origin_iata,
                        "destination": intent.destination_iata,
                        "search_date": search_date
                    },
                    final_response=llm_text or ""
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.error("Database write timed out")

    return result

# ----------------------------------------------------------------------
# Public entry point with streaming support
# ----------------------------------------------------------------------
async def plan_trip(
    *,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    date: Optional[str] = None,
    user_query: str,
    trip_type: Optional[str] = None,
    stream: bool = False,
    flights: Optional[List[Union[Dict, Flight]]] = None,
    depth: int = 0,
    flight_tool: Callable = default_flight_tool,
    weather_tool: Callable = default_weather_tool
) -> Union[PlanResult, MultiCityResult, Dict, AsyncGenerator[str, None]]:
    """
    Public entry point for planning a trip.

    If stream=False (default): returns a PlanResult (or MultiCityResult/error dict) with the full
    response, as in previous versions.

    If stream=True: returns an asynchronous generator that yields tokens as they are produced
    by the LLM. The generator will also produce a final JSON payload prefixed with "[DONE_JSON]"
    containing the structured result (best flight, weather, etc.). This mode is intended for
    SSE (Server‑Sent Events) or similar incremental delivery.

    Args:
        origin: IATA code (optional, overrides parsing)
        destination: IATA code (optional, overrides parsing)
        date: YYYY-MM-DD (optional, overrides parsing)
        user_query: Natural language query
        trip_type: Business/Holiday/Flexible/Urgent (optional, overrides parsing)
        stream: If True, return a token generator instead of a full PlanResult.
        flights: Pre-fetched flight list (for testing, can be dicts or Flight objects)
        depth: Recursion depth (internal use)
        flight_tool: Async function to search flights (for injection)
        weather_tool: Async function to fetch weather (for injection)

    Returns:
        - If stream=False: PlanResult, MultiCityResult, or error dict
        - If stream=True: AsyncGenerator[str, None]
    """
    # Non‑streaming branch – existing behaviour
    if not stream:
        return await _plan_trip_internal(
            origin=origin,
            destination=destination,
            date=date,
            user_query=user_query,
            trip_type=trip_type,
            flights=flights,
            depth=depth,
            flight_tool=flight_tool,
            weather_tool=weather_tool,
            skip_llm=False
        )

    # --- Streaming branch ---
    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. Get all data without LLM explanation (skip_llm=True)
            data_result = await _plan_trip_internal(
                origin=origin,
                destination=destination,
                date=date,
                user_query=user_query,
                trip_type=trip_type,
                flights=flights,
                depth=depth,
                flight_tool=flight_tool,
                weather_tool=weather_tool,
                skip_llm=True
            )

            # Handle error responses
            if isinstance(data_result, dict) and "error" in data_result:
                yield json.dumps({"error": data_result["error"]})
                yield "[DONE_JSON]" + json.dumps({"error": data_result["error"]})
                return

            # 2. Check circuit breaker before calling LLM
            if await check_llm_circuit():
                yield "[ERROR] LLM temporarily unavailable"
                yield "[DONE_JSON]" + json.dumps({"error": "LLM temporarily unavailable"})
                return

            # 3. Extract data needed for prompt
            if isinstance(data_result, MultiCityResult):
                # For multi-city trips, we currently only stream a simple message and final JSON.
                # (Could be extended to stream per leg, but omitted for brevity.)
                yield "This is a multi-city trip. "
                final_json = data_result.model_dump()
                # Ensure llm_response is None for all legs
                for leg in final_json.get("legs", []):
                    leg["llm_response"] = None
                yield "[DONE_JSON]" + json.dumps(final_json)
                return

            # Single leg
            best_flight = Flight(**data_result.best_flight)
            weather = data_result.weather
            debug_info = data_result.debug_info or {}
            intent_dict = debug_info.get("intent", {})
            all_flights_dicts = debug_info.get("all_flights", [])
            filters_applied = debug_info.get("filters_applied", "")
            trip_description = debug_info.get("trip_description", "")

            # Build prompt (identical to generate_explanation)
            flights_str = "\n".join([
                f"- {f['airline']} {f['flight_no']} on {f.get('date','N/A')} | "
                f"{f['departure_time']} → {f['arrival_time']} | "
                f"{f['duration_min']} min | {f['price_inr']}"
                for f in all_flights_dicts[:10]
            ])

            prompt = f"""
You are a helpful travel assistant helping a user plan {trip_description}.

User preferences:
- {filters_applied}

Flight options from {intent_dict.get('origin_iata')} to {intent_dict.get('destination_iata')} around {intent_dict.get('date')}:
{flights_str}

Best matching flight:
- {best_flight.airline} {best_flight.flight_no} on {best_flight.date or 'N/A'} |
  {best_flight.departure_time} → {best_flight.arrival_time} |
  Duration: {best_flight.duration_min} minutes |
  Price: {best_flight.price_inr} |
  Stops: {best_flight.stops} |
  Baggage: {best_flight.baggage}

Weather forecast for {intent_dict.get('destination_iata')}:
{weather}

User's question: {user_query}

Please recommend the best flight, explain why it matches their preferences, mention the weather suitability, and answer the user's query helpfully.
"""

            # 4. Call LLM in streaming mode with handshake timeout + metrics
            llm_start = time.monotonic()
            try:
                token_stream = await asyncio.wait_for(
                    generate(
                        prompt=prompt,
                        system="You are a professional travel planning assistant.",
                        model=PLANNER_LLM_MODEL,
                        stream=True
                    ),
                    timeout=STREAM_INIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                await record_llm_failure()
                metrics.record_stream_failure("unknown")  # provider unknown at this point
                yield "[ERROR] LLM stream initialization timed out"
                yield "[DONE_JSON]" + json.dumps({"error": "LLM stream initialization timed out"})
                return

            # Try to extract provider from token_stream (if available)
            provider = getattr(token_stream, "provider", "unknown")
            metrics.record_stream_start(provider)

            got_first_token = False
            first_token_time = None
            full_response = ""

            # 5. Consume stream with total timeout (PLANNER_LLM_TIMEOUT seconds)
            try:
                async with asyncio.timeout(PLANNER_LLM_TIMEOUT):  # Python 3.11+
                    async for token in token_stream:
                        if not isinstance(token, str):
                            token = str(token)
                        if not got_first_token:
                            first_token_time = time.monotonic() - llm_start
                            try:
                                metrics.LLM_LATENCY.labels(provider=provider).observe(first_token_time)
                            except Exception:
                                # safe guard: if labels don't exist or provider unknown, ignore
                                pass
                            got_first_token = True

                        yield token
                        full_response += token
            except asyncio.TimeoutError:
                await record_llm_failure()
                metrics.record_stream_failure(provider)
                yield f"[ERROR] LLM streaming timed out after {PLANNER_LLM_TIMEOUT}s"
                yield "[DONE_JSON]" + json.dumps({"error": f"LLM streaming timed out after {PLANNER_LLM_TIMEOUT}s"})
                return

            # 6. Success – record metrics, success, build final JSON, and log
            total_time = time.monotonic() - llm_start
            try:
                metrics.LLM_LATENCY.labels(provider=provider).observe(total_time)
            except Exception:
                pass
            metrics.record_stream_success(provider, total_time)
            await record_llm_success()

            final_result = data_result.model_dump()
            final_result["llm_response"] = full_response
            # Remove debug_info if you don't want to expose internal data to the client
            # final_result.pop("debug_info", None)
            yield "[DONE_JSON]" + json.dumps(final_result)

            # 7. Log session asynchronously (non‑blocking) using stored filtered_count
            if DB_AVAILABLE:
                try:
                    filtered_count = debug_info.get("filtered_count", len(all_flights_dicts))
                    phases = debug_info.get("phases", {})
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            save_session,
                            user_query=user_query,
                            agent_reasoning={
                                "version": "planner-v7-streaming",
                                "intent": intent_dict,
                                "filters_applied": filters_applied,
                                "ranked_count": debug_info.get("ranked_count", 0),
                                "flight_pref": intent_dict.get("flight_pref"),
                                "trip_type": intent_dict.get("trip_type"),
                                "use_cloud_llm": USE_CLOUD_FALLBACK,
                                "phases": phases
                            },
                            tool_output={
                                "all_flights_count": len(all_flights_dicts),
                                "filtered_count": filtered_count,
                                "weather": weather,
                                "origin": intent_dict.get("origin_iata"),
                                "destination": intent_dict.get("destination_iata"),
                                "search_date": data_result.search_date
                            },
                            final_response=full_response
                        ),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.error("Database write timed out")

        except Exception as e:
            await record_llm_failure()
            logger.exception("Error in streaming plan_trip")
            yield f"[ERROR]{str(e)}"
            yield "[DONE_JSON]" + json.dumps({"error": str(e)})

    return stream_generator()

# ----------------------------------------------------------------------
# Session logging (sync)
# ----------------------------------------------------------------------
def save_session(user_query: str, agent_reasoning: dict, tool_output: dict, final_response: str, user_id: Optional[str] = None):
    if not DB_AVAILABLE:
        logger.info("Session logging skipped (database not available)")
        return

    db = SessionLocal()
    try:
        sh = SessionHistory(
            user_id=user_id,
            user_query=user_query,
            agent_reasoning=agent_reasoning,
            tool_output=tool_output,
            final_response=final_response
        )
        db.add(sh)
        db.commit()
        logger.info("Session saved to database")
    except Exception as e:
        logger.error(f"Failed to save session: {e}")
        db.rollback()
    finally:
        db.close()