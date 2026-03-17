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
import weakref
import time
import random
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import dateutil.parser
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

# Local imports – use wrappers for testability
from tools.airline_api import search_flights as _search_flights_impl, AirlineAPIError
from tools.weather_api import check_weather as _check_weather_impl, get_forecast_for_date as _get_forecast_for_date_impl
from agents.llm_router import generate

# New centralised location resolver
from core.iata_resolver import is_iata_token, resolve_location
from core.api_key_manager import key_manager as api_key_manager

# Wrappers (unit‑test seam)
def search_flights(*args, **kwargs):
    return _search_flights_impl(*args, **kwargs)

def check_weather(*args, **kwargs):
    return _check_weather_impl(*args, **kwargs)

def get_forecast_for_date(*args, **kwargs):
    return _get_forecast_for_date_impl(*args, **kwargs)

default_flight_tool = search_flights
default_weather_tool = get_forecast_for_date   # Use forecast-for-date as the default weather tool

# Metrics instrumentation
import core.metrics as metrics

load_dotenv()

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logger = logging.getLogger("planner_agent")
_FLIGHT_GET_WARNING_LOGGED = False

# ----------------------------------------------------------------------
# Environment flags & configurable timeouts
# ----------------------------------------------------------------------
USE_CLOUD_FALLBACK = os.getenv("USE_CLOUD_LLM", "1") == "1"
PLANNER_LLM_MODEL = os.getenv("PLANNER_LLM_MODEL", "gpt-4o-mini")
PLANNER_LLM_TIMEOUT = float(os.getenv("PLANNER_LLM_TIMEOUT", "45"))  # Increased from 29 to 45
STREAM_INIT_TIMEOUT = float(os.getenv("PLANNER_STREAM_INIT_TIMEOUT", "5"))

# Per‑call timeouts
FLIGHT_TOOL_TIMEOUT = float(os.getenv("FLIGHT_TOOL_TIMEOUT", "8"))
WEATHER_TOOL_TIMEOUT = float(os.getenv("WEATHER_TOOL_TIMEOUT", "5"))
LLM_CORRECTION_TIMEOUT = float(os.getenv("LLM_CORRECTION_TIMEOUT", "5"))

# Return trip timeout (covers flight + weather + LLM)
RETURN_TRIP_TIMEOUT = float(os.getenv("RETURN_TRIP_TIMEOUT", "40"))

# Retry configuration for flight tool
FLIGHT_RETRY_ATTEMPTS = int(os.getenv("FLIGHT_RETRY_ATTEMPTS", "3"))
FLIGHT_RETRY_BASE = float(os.getenv("FLIGHT_RETRY_BASE", "0.5"))      # seconds
FLIGHT_RETRY_MAX_BACKOFF = float(os.getenv("FLIGHT_RETRY_MAX_BACKOFF", "5.0"))
FLIGHT_RETRY_JITTER = float(os.getenv("FLIGHT_RETRY_JITTER", "0.25"))   # fraction

# Cache control
DISABLE_CACHE     = os.getenv("DISABLE_CACHE", "0") == "1"
CACHE_FLIGHT_TTL  = int(os.getenv("CACHE_FLIGHT_TTL", "900"))    # default 15 min
CACHE_WEATHER_TTL = int(os.getenv("CACHE_WEATHER_TTL", "3600"))  # default 1 hour

# NEW: Weather forecast max days limit (default to 5 for free OpenWeatherMap)
WEATHER_FORECAST_MAX_DAYS = int(os.getenv("WEATHER_FORECAST_MAX_DAYS", "5"))

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
# City name to IATA override for deterministic test behavior
# ----------------------------------------------------------------------
CITY_IATA_OVERRIDES = {
    "delhi": "DEL",
    "new delhi": "DEL",
    "mumbai": "BOM",
    "bombay": "BOM",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "kolkata": "CCU",
    "calcutta": "CCU",
    "chennai": "MAA",
}

def _record_price_snapshot_safe(*args, **kwargs):
    """
    Lazy wrapper to avoid eager DB initialization during module import.
    """
    try:
        from tools.price_tracker import record_price_snapshot
        return record_price_snapshot(*args, **kwargs)
    except Exception:
        return None


def _flight_value_safe(flight_obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely read flight fields from dict-like payloads or Flight/Pydantic objects.
    Logs a warning once when dict-style access is attempted on non-dict objects.
    """
    global _FLIGHT_GET_WARNING_LOGGED
    if isinstance(flight_obj, dict):
        return flight_obj.get(key, default)
    if hasattr(flight_obj, key):
        if not _FLIGHT_GET_WARNING_LOGGED:
            logger.warning("Non-dict flight object encountered; using attribute access fallback")
            _FLIGHT_GET_WARNING_LOGGED = True
        return getattr(flight_obj, key, default)
    return default


async def _build_booking_handoff_url_safe(*args, **kwargs):
    try:
        from tools.booking_handoff import build_booking_handoff_url
        return await build_booking_handoff_url(*args, **kwargs)
    except Exception:
        return None


async def _hold_booking_safe(*args, **kwargs):
    from tools.booking_handoff import hold_booking
    return await hold_booking(*args, **kwargs)


def _confirm_booking_safe(*args, **kwargs):
    from tools.booking_handoff import confirm_booking
    return confirm_booking(*args, **kwargs)


def _cancel_booking_safe(*args, **kwargs):
    from tools.booking_handoff import cancel_booking
    return cancel_booking(*args, **kwargs)


def _parse_price_insights_safe(*args, **kwargs):
    try:
        from tools.price_tracker import parse_price_insights
        return parse_price_insights(*args, **kwargs)
    except Exception:
        return None


def _format_price_insights_for_llm_safe(*args, **kwargs):
    try:
        from tools.price_tracker import format_price_insights_for_llm
        return format_price_insights_for_llm(*args, **kwargs)
    except Exception:
        return ""


def _analyze_price_trend_safe(*args, **kwargs):
    try:
        from tools.price_tracker import analyze_price_trend
        return analyze_price_trend(*args, **kwargs)
    except Exception:
        return None


def _predict_future_price_safe(*args, **kwargs):
    try:
        from tools.price_tracker import predict_future_price
        return predict_future_price(*args, **kwargs)
    except Exception:
        return None


def _detect_booking_or_tracking_action(user_query: str) -> Optional[str]:
    """
    Detect explicit booking lifecycle / price tracking intents.
    Returns one of: confirm_booking, cancel_booking, hold_booking, track_price, or None.
    """
    q = (user_query or "").lower()
    if not q:
        return None

    if ("confirm" in q and ("booking" in q or "reservation" in q)):
        return "confirm_booking"
    if any(x in q for x in ("cancel booking", "cancel my reservation", "cancel reservation", "cancel my booking")):
        return "cancel_booking"
    if any(x in q for x in ("notify me if", "track this flight price", "track price", "price drops", "price decreases", "alert me if price")):
        return "track_price"
    if any(x in q for x in ("hold this flight", "hold booking", "hold my booking", "reserve this flight", "reserve my booking")):
        return "hold_booking"
    return None


def _extract_booking_id(user_query: str) -> Optional[int]:
    """
    Extract booking id from phrases like:
      - booking 123
      - booking id: 123
      - reservation #123
      - id 123
    """
    q = user_query or ""
    patterns = [
        r"\b(?:booking|reservation)\s*(?:id)?\s*[:#]?\s*(\d+)\b",
        r"\b(?:id|#)\s*(\d+)\b",
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None

def _resolve_city_to_iata(city_text: str) -> Optional[str]:
    """
    Resolve a free-form city string to an IATA code. Priorities:
      1) exact override mapping (lowercased)
      2) word-boundary match against overrides (to catch 'new delhi')
      3) resolve_location(...) fallback
      4) if token looks like a 3-letter IATA, return upper()
      5) return None
    """
    if not city_text:
        return None
    token = city_text.strip().lower()

    # 1) exact override
    if token in CITY_IATA_OVERRIDES:
        return CITY_IATA_OVERRIDES[token]

    # 2) word-boundary check for multi-word overrides or aliases
    for k, v in CITY_IATA_OVERRIDES.items():
        # treat keys as already lowercased
        if re.search(r'\b' + re.escape(k) + r'\b', token):
            return v

    # 3) fallback to resolver (external module)
    try:
        resolved = resolve_location(token)  # keep existing resolver usage
        if resolved:
            return resolved
    except Exception as e:
        logger.debug("resolve_location failed", exc_info=True)

    # 4) treat 3-letter tokens as raw IATA
    if len(token) == 3 and token.isalpha():
        return token.upper()

    return None


def _sanitize_iata_code(value: Optional[str]) -> Optional[str]:
    """
    Convert any resolver output into a strict valid 3-letter IATA code.
    Returns None if the value cannot be validated.
    """
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    if len(raw) == 3 and raw.isalpha() and is_iata_token(raw.upper()):
        return raw.upper()

    resolved = resolve_location(raw)
    if resolved and len(resolved) == 3 and is_iata_token(resolved):
        return resolved

    return None


def _infer_route_pair_from_query(user_query: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Deterministically infer origin/destination from free-form text when route regex misses.
    Priority:
      1) explicit IATA token pair (e.g., "DEL BOM")
      2) known city alias pair in text order (e.g., "Delhi Mumbai")
    """
    trace: Dict[str, Any] = {
        "source": None,
        "iata_candidates": [],
        "city_candidates": [],
    }
    if not user_query:
        return None, None, trace

    # 1) explicit IATA tokens
    iata_tokens: List[str] = []
    for tok in re.findall(r"\b([A-Za-z]{3})\b", user_query):
        up = tok.upper()
        if is_iata_token(up) and up not in iata_tokens:
            iata_tokens.append(up)
    trace["iata_candidates"] = iata_tokens
    if len(iata_tokens) >= 2:
        trace["source"] = "iata_pair"
        return iata_tokens[0], iata_tokens[1], trace

    # 2) known city aliases in appearance order
    hits: List[Tuple[int, str, str]] = []
    q_lower = user_query.lower()
    # longer aliases first to avoid "new delhi" being swallowed by "delhi"
    for city_alias, code in sorted(CITY_IATA_OVERRIDES.items(), key=lambda kv: len(kv[0]), reverse=True):
        for m in re.finditer(r"\b" + re.escape(city_alias) + r"\b", q_lower):
            hits.append((m.start(), city_alias, code))
    hits.sort(key=lambda x: x[0])
    trace["city_candidates"] = [{"city": city, "code": code, "pos": pos} for pos, city, code in hits]

    ordered_codes: List[str] = []
    for _, _, code in hits:
        if code not in ordered_codes:
            ordered_codes.append(code)

    if len(ordered_codes) >= 2:
        trace["source"] = "city_pair"
        return ordered_codes[0], ordered_codes[1], trace

    return None, None, trace

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
class CacheLock(asyncio.Lock):
    """Simple subclass so Lock objects support weak references."""
    pass

_cache_locks = weakref.WeakValueDictionary()

def _get_cache_lock(key):
    """Lazily create/return a per-key asyncio.Lock stored as a weak ref.

    Locks are kept alive by strong references in tasks that hold/wait on them.
    When no task holds or waits on a lock any more, GC removes it from the dict.
    This removes the need for manual clear() and avoids orphaning in-flight locks.
    """
    lock = _cache_locks.get(key)
    if lock is None:
        lock = CacheLock()
        _cache_locks[key] = lock
    return lock

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
                if not DISABLE_CACHE:                    # skip entire cache block in dev
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

                if not DISABLE_CACHE:                    # don't store if cache disabled
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
    ttl=CACHE_FLIGHT_TTL,
    maxsize=500,
    fetch_func=default_flight_tool
)

_shared_cached_weather = create_cached_fetcher(
    ttl=CACHE_WEATHER_TTL,
    maxsize=500,
    fetch_func=default_weather_tool
)

# ----------------------------------------------------------------------
# Retry helper for flight searches (with exponential backoff + jitter)
# ----------------------------------------------------------------------
async def _call_with_retries(fn: Callable, attempts: int = FLIGHT_RETRY_ATTEMPTS) -> Any:
    """
    Generic async retry with exponential backoff + jitter.
    Calls a zero-argument coroutine function `fn` repeatedly until success or max attempts.
    Raises the last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except AirlineAPIError as e:
            # Upstream advertised 'unavailable' — try retrying a couple of times
            last_exc = e
            logger.warning("AirlineAPIError on attempt %d/%d: %s", attempt, attempts, str(e))
        except asyncio.TimeoutError as e:
            last_exc = e
            logger.warning("Timeout on airline API attempt %d/%d", attempt, attempts)
        except Exception as e:
            # transient network or parser error — treat as retryable for first N attempts
            last_exc = e
            logger.exception("Unexpected exception during airline fetch attempt %d/%d", attempt, attempts)

        if attempt == attempts:
            break

        # exponential backoff with jitter
        backoff = min(FLIGHT_RETRY_BASE * (2 ** (attempt - 1)), FLIGHT_RETRY_MAX_BACKOFF)
        jitter = backoff * (random.uniform(-FLIGHT_RETRY_JITTER, FLIGHT_RETRY_JITTER))
        sleep_for = max(0.05, backoff + jitter)
        await asyncio.sleep(sleep_for)

    # all attempts failed — re-raise last exception (caller will convert to fallback)
    raise last_exc

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
    stops: Union[str, int] = 0           # int from API; "N/A" when unknown
    layover_info: str = ""               # e.g. "1h 30m at BOM"
    baggage: str = "Check airline"       # Extracted from SerpAPI extensions
    booking_token: Optional[str] = None  # For booking handoff
    carbon_emissions_g: Optional[int] = None  # CO2 in grams
    date: Optional[str] = None
    handoff_url: Optional[str] = None    # Resolved booking deep-link (set after ranking)
    # New fields for multi-leg/layover tracking (added by airline_api)
    layover_durations_min: Optional[List[int]] = Field(default_factory=list)
    layover_airports: Optional[List[str]] = Field(default_factory=list)

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
    # New fields for weather presence tracking
    weather_present: bool = True
    weather_reason: Optional[str] = None

class MultiCityResult(BaseModel):
    """Structured output for multi-city trips."""
    multicity: bool = True
    legs: List[PlanResult]

PlanResult.model_rebuild()

# ----------------------------------------------------------------------
# Route extraction from natural language (pure regex, no IATA logic)
# ----------------------------------------------------------------------
def extract_stopover(query_text: str) -> Dict[str, Optional[str]]:
    """
    Robust extraction for 'via' stopover phrases.
    Returns dict with keys: origin_text, destination_text, via_text
    Accepts multiword city names and many phrasings.
    """
    q = query_text.strip()

    # Try: from <A> to <B> via <C>  (non-greedy multiword)
    m = re.search(r'from\s+(.+?)\s+to\s+(.+?)\s+(?:via|through|stopover|connecting via)\s+(.+)', q, re.IGNORECASE)
    if m:
        return {"origin_text": m.group(1).strip(), "destination_text": m.group(2).strip(), "via_text": m.group(3).strip()}

    # Try: <A> to <B> via <C>
    m = re.search(r'\b(.+?)\s+to\s+(.+?)\s+(?:via|through|stopover)\s+(.+)', q, re.IGNORECASE)
    if m:
        return {"origin_text": m.group(1).strip(), "destination_text": m.group(2).strip(), "via_text": m.group(3).strip()}

    # Try: via <C> alone
    m = re.search(r'\bvia\s+(.+)', q, re.IGNORECASE)
    if m:
        return {"origin_text": None, "destination_text": None, "via_text": m.group(1).strip()}

    # Fallback: simple "<A> to <B>"
    m = re.search(r'\b(.+?)\s+to\s+(.+?)\b', q, re.IGNORECASE)
    if m:
        return {"origin_text": m.group(1).strip(), "destination_text": m.group(2).strip(), "via_text": None}

    return {"origin_text": None, "destination_text": None, "via_text": None}

def normalize_trip(user_query: str) -> Dict[str, Any]:
    """
    Build a canonical trip object from raw user_query using regex and centralised resolver.
    Returns dict with keys: origin_iata, destination_iata, via_iata.
    """
    parts = extract_stopover(user_query)

    origin_iata = None
    dest_iata = None
    via_iata = None

    if parts["origin_text"]:
        origin_iata = _resolve_city_to_iata(parts["origin_text"])
        logger.debug(f"normalize_trip: origin_text='{parts['origin_text']}' -> origin_iata={origin_iata}")

    if parts["destination_text"]:
        dest_iata = _resolve_city_to_iata(parts["destination_text"])
        logger.debug(f"normalize_trip: dest_text='{parts['destination_text']}' -> dest_iata={dest_iata}")

    if parts["via_text"]:
        via_iata = _resolve_city_to_iata(parts["via_text"])
        logger.debug(f"normalize_trip: via_text='{parts['via_text']}' -> via_iata={via_iata}")

    return {
        "origin_iata": origin_iata,
        "destination_iata": dest_iata,
        "via_iata": via_iata,
    }

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
    wants_eco: bool = False           # True when user asks for green/eco/low-carbon flights
    trip_type: Optional[str] = None   # No default – fallback applied in business logic
    deep_search: bool = False         # NEW: user wants absolute cheapest, exhaustive search

def parse_intent(user_query: str) -> ParsedIntent:
    """Extract all structured data from the natural language query."""
    intent = ParsedIntent()

    # --- First, use the robust normalize_trip to get IATA codes ---
    trip = normalize_trip(user_query)
    if trip["origin_iata"]:
        intent.origin_iata = trip["origin_iata"]
    if trip["destination_iata"]:
        intent.destination_iata = trip["destination_iata"]

    # Deterministic fallback for compact route formats like "DEL BOM ..." or "Delhi Mumbai ..."
    if not intent.origin_iata or not intent.destination_iata:
        inferred_origin, inferred_dest, _ = _infer_route_pair_from_query(user_query)
        if not intent.origin_iata and inferred_origin:
            intent.origin_iata = inferred_origin
        if not intent.destination_iata and inferred_dest:
            intent.destination_iata = inferred_dest

    # Always sanitize to strict valid IATA codes.
    intent.origin_iata = _sanitize_iata_code(intent.origin_iata)
    intent.destination_iata = _sanitize_iata_code(intent.destination_iata)
    # via_iata is not stored directly in intent; we only use city name for stopover if needed
    # (we could keep via_city for potential later use, but for now we ignore)

    q = user_query.lower()
    today = datetime.now().date()
    _relative_date_set = False   # track if we set date via relative rules

    # --- "starting DATE" match (e.g., "starting March 20") ---
    starting_match = re.search(
        r'\bstarting\s+(?:on\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+\w+|\w+\s+\d{1,2}(?:,\s*\d{4})?|\d{4}-\d{2}-\d{2})',
        q, re.IGNORECASE
    )
    if starting_match and not _relative_date_set:
        try:
            parsed_start = dateutil.parser.parse(starting_match.group(1), dayfirst=True)
            if parsed_start.date() >= today:
                intent.date = parsed_start.strftime("%Y-%m-%d")
                _relative_date_set = True
        except Exception:
            pass

    # --- Enhanced relative date parsing (with word numbers + weeks) ---
    if not _relative_date_set:
        _WORD_TO_NUM = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
            'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
            'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'twenty': 20, 'thirty': 30, 'fortnight': 14,
        }
        q_rel = q
        for word, num in _WORD_TO_NUM.items():
            q_rel = re.sub(rf'\b{word}\b', str(num), q_rel)

        # "14 days after today" / "14 days from now"
        rel_days = re.search(
            r'(\d+)\s+days?\s+(?:after|from)\s+today|(\d+)\s+days?\s+from\s+now',
            q_rel
        )
        if rel_days:
            n = int(rel_days.group(1) or rel_days.group(2))
            intent.date = (today + timedelta(days=n)).strftime("%Y-%m-%d")
            _relative_date_set = True

        # "2 weeks from now / after today"
        elif re.search(r'(\d+)\s+weeks?\s+(?:from\s+(?:now|today)|after\s+today)', q_rel):
            m = re.search(r'(\d+)\s+weeks?', q_rel)
            intent.date = (today + timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")
            _relative_date_set = True

    # --- If relative date not set, fall back to other date parsers ---
    if not _relative_date_set:
        # Strip price expressions so "₹3000" / "under 5000 INR" don't pollute dateutil
        q_clean = re.sub(
            r'under\s*[₹$€£]?\s*\d+|[₹$€£]\s*\d+|\d[\d,]*\s*(?:rupees?|inr|usd|eur)\b',
            '', q, flags=re.IGNORECASE
        )

        parsed_date = None
        if HAS_DATEPARSER:
            settings = {'PREFER_DATES_FROM': 'future', 'DATE_ORDER': 'DMY'}
            parsed_date = dateparser.parse(q_clean, settings=settings)
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
                parsed_date = dateutil.parser.parse(
                    q_clean,
                    fuzzy=True, dayfirst=True,
                    default=datetime.now().replace(month=1, day=1)
                )
            except:
                pass

        if parsed_date:
            # Sanity check: reject absurd years and any past date
            if parsed_date.year > today.year + 2 or parsed_date.year < 2000:
                parsed_date = None
            elif parsed_date.date() < today:
                # Past date: bump forward one year as best guess
                bumped = parsed_date.replace(year=parsed_date.year + 1)
                # If still in the past (e.g., parsed year was far behind), reject entirely
                parsed_date = bumped if bumped.date() >= today else None
                if parsed_date:
                    intent.date = parsed_date.strftime("%Y-%m-%d")
            else:
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

    # --- Eco/green preference ---
    intent.wants_eco = any(kw in q for kw in ("eco", "green", "low carbon", "low-carbon", "sustainable", "environment"))

    # --- Airline preference ---
    intent.preferred_airlines = [a for a in AIRLINES if a in q]

    # --- Layover limit (explicit number) ---
    layover_match = re.search(r'layover.*?(\d{1,2})\s*hours?', q)
    if layover_match:
        intent.layover_limit_minutes = int(layover_match.group(1)) * 60

    # --- NEW: detect "short layover" or "quick connection" and set a default limit if not already set ---
    if not intent.layover_limit_minutes:
        if re.search(r'\b(short layover|quick connection)\b', q):
            intent.layover_limit_minutes = 180  # 3 hours

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

    # --- Stopover city (using improved regex) ---
    via_match = re.search(r'\bvia\s+([A-Za-z][a-z]+(?:\s+[A-Z][a-z]+)*)', q)
    if via_match:
        intent.stopover_city = via_match.group(1).strip().title()

    # --- Flight preference ---
    if "cheapest" in q or "cheap" in q or "lowest price" in q or "budget" in q:
        intent.flight_pref = "cheapest"
    elif "shortest" in q or "fastest" in q or "least time" in q or "quickest" in q:
        intent.flight_pref = "shortest"
    elif "balanced" in q or ("price" in q and "duration" in q):
        intent.flight_pref = "balanced"
    elif intent.wants_eco:
        intent.flight_pref = "eco"

    # --- NEW: deep search (absolute cheapest) ---
    if any(x in q for x in ["absolute cheapest", "cheapest possible", "lowest price ever"]):
        intent.deep_search = True

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

def normalize_flight_field(value: Any) -> str:
    """Convert flight field to a normalized string for matching."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return str(value).strip().lower()

def filter_flights(flights: List[Flight], intent: ParsedIntent) -> List[Flight]:
    """Apply all user filters to the flight list with tolerant matching."""
    filtered = []
    # Local warnings list (passed to LLM later)
    filter_warnings = []
    for f in flights:
        reasons = []
        # Normalize fields
        stops = normalize_flight_field(getattr(f, "stops", ""))
        baggage = normalize_flight_field(getattr(f, "baggage", ""))
        airline = normalize_flight_field(getattr(f, "airline", ""))

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
            # Tolerant direct detection
            stops_clean = stops.replace("stops", "").strip()
            direct = False
            # Unknown stops → assume it might be direct (benefit of doubt)
            if stops in ("n/a", "", "unknown"):
                direct = True
                filter_warnings.append("Stop data unavailable for recommended flight; directness cannot be confirmed.")
            elif any(x in stops for x in ("non-stop", "nonstop", "direct")):
                direct = True
            else:
                try:
                    if int(stops_clean) == 0:
                        direct = True
                except:
                    pass
            if not direct:
                reasons.append("not direct")
        if intent.preferred_airlines:
            if not any(pref in airline for pref in intent.preferred_airlines):
                reasons.append("airline")
        # --- Layover limit filter (using true layover durations) ---
        if intent.layover_limit_minutes:
            layover_durs = f.layover_durations_min or []   # assume flight has this field
            if layover_durs and any(d > intent.layover_limit_minutes for d in layover_durs):
                reasons.append("layover")
        # --- Stopover filter: handled separately in main flow, not here ---
        if intent.baggage_pref:
            if intent.baggage_pref == "hand":
                if not any(x in baggage for x in (
                    "hand", "cabin", "carry", "7kg", "8kg", "10kg"
                )):
                    reasons.append("baggage")
            if intent.baggage_pref == "checked":
                if not any(x in baggage for x in (
                    "checked", "check", "hold", "1pc", "2pc", "15kg", "20kg"
                )):
                    reasons.append("baggage")

        if not reasons:
            filtered.append(f)
        else:
            logger.debug(f"Flight {f.flight_no} rejected: {reasons}")
    return filtered, filter_warnings

def get_weights(pref: str) -> Tuple[float, float, float]:
    """Return (price_weight, duration_weight, carbon_weight)."""
    if pref == "cheapest":
        return 0.8, 0.2, 0.0
    elif pref == "shortest":
        return 0.2, 0.8, 0.0
    elif pref == "balanced":
        return 0.5, 0.5, 0.0
    elif pref == "eco":
        return 0.3, 0.2, 0.5   # heavily weight low carbon
    return 0.6, 0.4, 0.0

def rank_flights(flights: List[Flight], intent: ParsedIntent) -> List[Flight]:
    """
    Rank flights by normalized price, duration, and (when eco mode) carbon emissions.
    Normalization values computed once to avoid O(n²).
    """
    if not flights:
        return []
    prices    = [price_to_int(f.price_inr) for f in flights]
    durations = [f.duration_min for f in flights]
    carbons   = [f.carbon_emissions_g for f in flights if f.carbon_emissions_g is not None]

    min_price, max_price = min(prices), max(prices)
    min_dur,   max_dur   = min(durations), max(durations)
    min_co2,   max_co2   = (min(carbons), max(carbons)) if carbons else (0, 1)

    wp, wd, wc = get_weights(intent.flight_pref)

    # If deep_search, we can increase price weight even further (but already 0.8 for cheapest)
    # This could be used to adjust search parameters earlier, but for ranking we keep as is.

    def score(f: Flight) -> float:
        price    = price_to_int(f.price_inr)
        duration = f.duration_min
        price_norm = (max_price - price)    / (max_price - min_price) if max_price > min_price else 1.0
        dur_norm   = (max_dur   - duration) / (max_dur   - min_dur)   if max_dur   > min_dur   else 1.0
        # Carbon norm: lower CO2 scores higher; flights without data score 0.5 (neutral)
        if wc > 0 and f.carbon_emissions_g is not None and max_co2 > min_co2:
            co2_norm = (max_co2 - f.carbon_emissions_g) / (max_co2 - min_co2)
        else:
            co2_norm = 0.5
        return price_norm * wp + dur_norm * wd + co2_norm * wc

    scored = [(f, score(f)) for f in flights]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scored]

# ----------------------------------------------------------------------
# LLM explanation generation with timeout and circuit breaker (non-streaming)
# ----------------------------------------------------------------------
def _enforce_narrative_consistency(
    llm_text: str,
    best_flight: Flight,
    weather: Dict[str, Any],
) -> str:
    """
    Post-process LLM text to keep narrative aligned with structured data.
    Targeted fixes only:
    1) prevent layover wording for non-stop flights
    2) prevent min/max temperature inversion wording
    """
    if not llm_text:
        return llm_text

    text = llm_text

    # 1) Non-stop flights should not be described as having layovers.
    if best_flight.stops == 0 or str(best_flight.stops).strip() == "0":
        text = re.sub(
            r"\bwith a layover\s+(?:less than|under|below)\s+\d+\s*(?:hours?|hrs?|h|minutes?|mins?)\b",
            "with no layover",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bhas a layover\s+(?:less than|under|below)\s+\d+\s*(?:hours?|hrs?|h|minutes?|mins?)\b",
            "has no layover",
            text,
            flags=re.IGNORECASE,
        )
        replacements = (
            (r"\bwith a layover\b", "with no layover"),
            (r"\bhas a layover\b", "has no layover"),
            (r"\blayover of\b", "max-layover preference of"),
            (r"\bconnecting time\b", "connection preference"),
        )
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 2) Keep weather low/high labels consistent with structured min/max.
    if isinstance(weather, dict):
        temp_min = weather.get("temp_min_c")
        temp_max = weather.get("temp_max_c")
        try:
            if temp_min is not None and temp_max is not None:
                temp_min_f = float(temp_min)
                temp_max_f = float(temp_max)
                if temp_max_f > temp_min_f:
                    # If text describes max as low/min, rewrite that numeric mention to min.
                    max_int = str(int(temp_max_f))
                    min_str = str(temp_min)
                    low_inversion_re = re.compile(
                        rf'(?i)\b(low|minimum|min)(?P<middle>(?:(?!(?:high|max)).){{0,30}}?)(?P<value>{re.escape(max_int)}(?:\.0+)?)'
                    )
                    text = low_inversion_re.sub(
                        lambda m: f"{m.group(1)}{m.group('middle')}{min_str}",
                        text,
                    )
        except Exception:
            # Never fail planner response due to sanitization.
            pass

    return text


async def generate_explanation(
    user_query: str,
    intent: ParsedIntent,
    best_flight: Flight,
    weather: Dict,
    all_flights: List[Flight],
    filters_applied: str,
    trip_description: str,
    warnings: Optional[List[str]] = None,
    price_insights_str: str = "",   # pre-formatted price intelligence sentence
    price_analysis_str: str = "",   # NEW: structured price trend info
    price_prediction_str: str = "", # NEW: price prediction advice
    booking_url: Optional[str] = None,  # NEW: optional booking link
) -> str:
    """Call LLM to produce a natural language response, with timeout and circuit breaker."""
    if await check_llm_circuit():
        logger.warning("LLM circuit breaker open, returning deterministic summary")
        return generate_deterministic_summary(
            best_flight, weather, filters_applied,
            error="circuit breaker open", location=intent.destination_iata
        )

    # Cap flights shown in prompt to reduce token usage (5 for round trips, 10 otherwise)
    max_flights_in_prompt = 5 if intent.return_date else 10
    flights_str = "\n".join([
        f"- {f.airline} {f.flight_no} on {f.date or 'N/A'} | "
        f"{f.departure_time} → {f.arrival_time} | "
        f"{f.duration_min} min | {f.price_inr} | "
        f"Stops: {f.stops} | Baggage: {f.baggage}"
        for f in all_flights[:max_flights_in_prompt]
    ])

    # Format warnings if present, escalating "relaxed" warnings to mandatory instructions
    if warnings:
        processed = []
        for w in warnings:
            if "relaxed" in w.lower():
                processed.append(
                    "MANDATORY: Your baggage or airline preference could not be matched. "
                    "You MUST explicitly tell the user their preference was not available and "
                    "you are showing the closest alternative. Do NOT claim the flight meets their baggage requirement."
                )
            else:
                processed.append(w)
        warnings_str = "\nSystem Notes/Warnings:\n- " + "\n- ".join(processed)
    else:
        warnings_str = ""

    # Ensure weather values are plain (not enum objects) and build readable string
    weather_display = {}
    for k, v in weather.items():
        weather_display[k] = v.value if hasattr(v, 'value') else v

    # Determine forecast proximity to travel date
    forecast_date_str = weather_display.get("forecast_date")
    travel_date_str = best_flight.date or intent.date or "your travel date"
    forecast_is_approximate = False
    approx_note = ""
    if forecast_date_str and travel_date_str and travel_date_str != "your travel date":
        try:
            from datetime import datetime as _dt
            t_dt = _dt.strptime(travel_date_str, "%Y-%m-%d")
            f_dt = _dt.strptime(forecast_date_str, "%Y-%m-%d")
            delta = abs((t_dt - f_dt).days)
            if delta > 3:
                forecast_is_approximate = True
                approx_note = f" (closest available forecast; actual forecast for {forecast_date_str})"
        except Exception:
            pass

    # Build weather string — clearly labelled as FORECAST (not current conditions)
    forecast_label = travel_date_str
    weather_str = (
        f"Condition: {weather_display.get('condition', 'N/A')}, "
        f"Temperature: {weather_display.get('temperature_c', 'N/A')}°C "
        f"(feels like {weather_display.get('feels_like_c', 'N/A')}°C)"
    )
    if weather_display.get("temp_min_c") is not None and weather_display.get("temp_max_c") is not None:
        weather_str += (
            f", Daily low: {weather_display['temp_min_c']}°C, Daily high: {weather_display['temp_max_c']}°C"
        )
    weather_str += (
        f", Humidity: {weather_display.get('humidity', 'N/A')}%, "
        f"Wind: {weather_display.get('wind_kph', 'N/A')} kph, "
        f"AQI: {weather_display.get('air_quality_index', 'N/A')}"
    )
    if weather_display.get("precipitation_chance") is not None:
        weather_str += f", Precipitation chance: {weather_display['precipitation_chance']}%"
    if weather_display.get("has_rain"):
        weather_str += " ⚠️ Rain expected"
    if weather_display.get("has_snow"):
        weather_str += " ⚠️ Snow expected"

    # Handle unknown stops gracefully
    stops_val = best_flight.stops
    if str(stops_val) in ("N/A", "n/a", "", "unknown"):
        stops_display = "unknown (no data available)"
    elif isinstance(stops_val, int):
        stops_display = "non-stop" if stops_val == 0 else f"{stops_val} stop(s)"
    else:
        stops_display = str(stops_val)

    # Carbon display
    carbon_display = "N/A"
    if best_flight.carbon_emissions_g is not None:
        carbon_kg = round(best_flight.carbon_emissions_g / 1000, 1)
        carbon_display = f"{carbon_kg} kg CO₂"

    # CRITICAL CONSTRAINT for unknown stops
    constraint_note = ""
    if "unknown" in stops_display:
        constraint_note = "You MUST NOT state this is a non-stop or direct flight. If stops are unknown, say 'stop availability unknown'."

    # --- Baggage constraint for hand baggage mismatch ---
    baggage_constraint = ""
    if intent.baggage_pref == "hand":
        bf_baggage_low = best_flight.baggage.lower()
        if any(x in bf_baggage_low for x in ("checked", "check", "free bag", "hold")):
            baggage_constraint = (
                f"\nCRITICAL BAGGAGE CONSTRAINT: User requested hand/cabin-baggage only, "
                f"but this flight's baggage is '{best_flight.baggage}' (checked baggage). "
                "You MUST NOT open with or imply this flight meets the hand baggage requirement. "
                "Your first sentence MUST be something like: 'No hand-baggage-only flight was "
                "available; here is the closest alternative.' Then disclose the baggage mismatch clearly."
            )

    # --- Layover constraint for non-stop flights when user has a layover limit ---
    layover_constraint = ""
    if intent.layover_limit_minutes and best_flight.stops == 0:
        layover_constraint = (
            f"\nLAYOVER NOTE: This is a non-stop flight — it has NO layover at all, which "
            f"automatically satisfies the user's max layover requirement of {intent.layover_limit_minutes} minutes. "
            "CRITICAL PHRASING RULES — apply to every sentence including the opening and summary:\n"
            "  1. NEVER write 'with a layover less than X hours' or 'layover of less than X' "
            "anywhere in your response — those phrases imply the flight HAS a layover.\n"
            "  2. In your opening sentence, do NOT echo the user's constraint with 'layover of less than'. "
            "Instead write: 'Based on your max-layover preference, the best option is...'\n"
            f"  3. In your summary sentence, write something like: "
            f"'{best_flight.airline} {best_flight.flight_no} is the best non-stop option — it has no layover "
            "whatsoever, which exceeds your requirement.' Do not echo the user's constraint phrase."
        )

    # --- Temperature constraint for correct ordering ---
    temp_constraint = ""
    if weather_display.get("temp_min_c") is not None and weather_display.get("temp_max_c") is not None:
        temp_constraint = (
            f"\nTEMP RULE: Daily low is {weather_display['temp_min_c']}°C, daily high is "
            f"{weather_display['temp_max_c']}°C. Never swap these — do not describe "
            f"{weather_display['temp_max_c']}°C as the 'low' temperature."
        )

    # Construct system prompt with airline rule only if the best flight does NOT match the preference
    airline_rule = ""
    if intent.preferred_airlines:
        preferred_lower = [a.lower() for a in intent.preferred_airlines]
        best_airline_lower = best_flight.airline.lower()
        airline_matched = any(p in best_airline_lower or best_airline_lower in p for p in preferred_lower)
        if not airline_matched:
            airline_rule = (
                f"AIRLINE RULE: The user's preferred airline ({', '.join(intent.preferred_airlines)}) is not available. "
                "Your FIRST sentence MUST disclose this — something like: "
                "'No flights were found for [airline]; here is the closest available alternative.' "
                "Do NOT open with flight details and then mention the airline mismatch later. "
                "Do NOT invent or fabricate alternative flights — only present flights from the data above."
            )

    # Prepend a facts block to ensure origin/destination appear explicitly
    facts_block = (
        f"Origin: {intent.origin_iata or 'unknown'}\n"
        f"Destination: {intent.destination_iata or 'unknown'}\n"
        f"Departure date: {intent.date or 'not specified'}\n"
    )
    if intent.return_date:
        facts_block += f"Return date: {intent.return_date}\n"

    system = (
        "You are a professional travel planning assistant. "
        "CRITICAL: NEVER invent, fabricate, or suggest flight numbers, airline codes, prices, "
        "departure times, or any flight details that are not explicitly present in the flight "
        "data provided to you. If no matching airline is found, present only the available flight "
        "with a note that it differs from the preference — do NOT create fictional alternatives. "
        "RULE: If a flight data field is 'unknown (no data available)', you are PROHIBITED from "
        "stating or implying its value. "
        "IATA RULE: Whenever you mention a city's weather, always include its IATA code in "
        "parentheses, e.g. 'Weather for Bangalore (BLR)' or 'Mumbai (BOM)'. "
        "CITY NAME RULE: When writing about the flight destination, use ONLY the correct city name "
        "for the destination IATA code. Examples of correct mappings: MAA = Chennai (NOT Mumbai), "
        "BLR = Bangalore, BOM = Mumbai, DEL = Delhi. Never call MAA 'Mumbai' or BOM 'Chennai'. "
        + airline_rule
    )

    prompt = f"""
CRITICAL CONSTRAINT: The stops field for this flight is '{stops_display}'.
{constraint_note}{baggage_constraint}{layover_constraint}{temp_constraint}

You are a helpful travel assistant helping a user plan {trip_description}.

User preferences:
- {filters_applied}
{warnings_str}

Flight options from {intent.origin_iata} to {intent.destination_iata} around {intent.date}:
{flights_str}

Best matching flight:
- {best_flight.airline} {best_flight.flight_no} on {best_flight.date or 'N/A'} |
  {best_flight.departure_time} → {best_flight.arrival_time} |
  Duration: {best_flight.duration_min} minutes |
  Price: {best_flight.price_inr} |
  Stops: {stops_display}{f" ({best_flight.layover_info})" if best_flight.layover_info else ""} |
  Baggage: {best_flight.baggage} |
  Carbon emissions: {carbon_display}
{f"{chr(10)}{price_insights_str}" if price_insights_str else ""}
{f"{chr(10)}{price_analysis_str}" if price_analysis_str else ""}
{f"{chr(10)}{price_prediction_str}" if price_prediction_str else ""}
Weather FORECAST for {intent.destination_iata} on {forecast_label}{approx_note}:
{weather_str}

IMPORTANT: Only reference the exact flights listed above. Do not create or suggest any other flights, codes, or prices.
User's question: {user_query}

Please recommend the best flight, explain why it matches their preferences, mention the weather forecast suitability (including packing advice based on min/max temperature and any rain or snow alerts), and answer the user's query helpfully.
"""

    # Combine facts block and prompt with an instruction to echo the facts
    full_prompt = facts_block + "\nPlease include the above origin and destination clearly at the start of your summary.\n\n" + prompt

    logger.info("Sending prompt to LLM")
    # ADD DEBUG LOGGING FOR PROMPT AND TRIP_DESCRIPTION
    logger.debug(
        "LLM prompt (first 500 chars): %s",
        full_prompt[:500].replace('\n', ' ')
    )
    logger.debug(
        "LLM trip_description (first 500 chars): %s",
        trip_description[:500].replace('\n', ' ')
    )

    LLMROUTER_FALLBACK_MARKER = "All LLM backends failed"
    try:
        llm_text = await asyncio.wait_for(
            generate(
                prompt=full_prompt,
                system=system,
                model=PLANNER_LLM_MODEL,
                stream=False
            ),
            timeout=PLANNER_LLM_TIMEOUT
        )
        # Detect when llm_router itself returned an internal fallback string instead of raising
        if LLMROUTER_FALLBACK_MARKER in (llm_text or ""):
            logger.warning("generate() returned internal backend-failure fallback; retrying once")
            await asyncio.sleep(1)
            try:
                llm_text = await asyncio.wait_for(
                    generate(
                        prompt=full_prompt,
                        system=system,
                        model=PLANNER_LLM_MODEL,
                        stream=False
                    ),
                    timeout=min(PLANNER_LLM_TIMEOUT, 30)
                )
            except Exception:
                llm_text = ""
            if not llm_text or LLMROUTER_FALLBACK_MARKER in llm_text:
                await record_llm_failure()
                return generate_deterministic_summary(
                    best_flight, weather, filters_applied,
                    error="All LLM backends failed", location=intent.destination_iata
                )
        llm_text = _enforce_narrative_consistency(llm_text, best_flight, weather)
        await record_llm_success()
        return llm_text
    except asyncio.TimeoutError:
        logger.error(f"LLM call timed out after {PLANNER_LLM_TIMEOUT}s")
        await record_llm_failure()
        return generate_deterministic_summary(
            best_flight, weather, filters_applied,
            error="timed out", location=intent.destination_iata
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        await record_llm_failure()
        return generate_deterministic_summary(
            best_flight, weather, filters_applied,
            error=str(e), location=intent.destination_iata
        )

def generate_deterministic_summary(
    best_flight: Flight, weather: Dict, filters: str,
    error: str = "", location: str = ""
) -> str:
    """Fallback summary when LLM is unavailable."""
    if isinstance(weather, dict):
        condition = weather.get("condition", weather.get("description", "unknown"))
        temp = weather.get("temperature_c", weather.get("temp", "unknown"))
    else:
        # Weather object
        condition = getattr(weather, "condition", "unknown")
        temp = getattr(weather, "temperature_c", getattr(weather, "temp", "unknown"))

    weather_str = f"{condition}, {temp}°C" if temp != "unknown" else condition
    loc_display = f" ({location})" if location else ""

    base = (f"I recommend {best_flight.airline} {best_flight.flight_no} at "
            f"{best_flight.departure_time} arriving {best_flight.arrival_time}. "
            f"Duration: {best_flight.duration_min} minutes, Price: {best_flight.price_inr}. "
            f"Weather at destination{loc_display}: {weather_str}. ")
    if error:
        base += f"(Note: Enhanced explanation unavailable due to {error}.)"
    return base

# ----------------------------------------------------------------------
# City correction using LLM (with circuit breaker + tolerant extraction)
# ----------------------------------------------------------------------
async def correct_cities_with_llm(user_query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempt to extract origin/destination IATA codes using LLM with a structured JSON prompt.
    Returns (origin_iata, destination_iata, explanation) – explanation may be None if not needed.
    """
    if await check_llm_circuit():
        logger.warning("LLM circuit breaker open, skipping city correction")
        return None, None, None

    mini_prompt = f"""
Extract the route from this travel query.

Return ONLY valid JSON with the following fields:
- "origin": the origin city or IATA code (or null if unknown)
- "destination": the destination city or IATA code (or null if unknown)
- "via": any stopover city or IATA code (or null if none)
- "explanation": a one-sentence human summary of what you understood

Do not add any other text.

Query: {user_query}
"""
    try:
        fixed_text = await asyncio.wait_for(
            generate(
                prompt=mini_prompt,
                system="You are a precise travel assistant. Always output JSON only.",
                model=PLANNER_LLM_MODEL,
                stream=False
            ),
            timeout=LLM_CORRECTION_TIMEOUT
        )

        # Attempt to find a JSON object in the response (allows the LLM to add commentary)
        parsed_json = None
        try:
            # Try direct JSON load first
            parsed_json = json.loads(fixed_text)
        except Exception:
            # Search for first { ... } block
            m = re.search(r'\{.*\}', fixed_text, flags=re.DOTALL)
            if m:
                try:
                    parsed_json = json.loads(m.group(0))
                except Exception:
                    parsed_json = None

        if parsed_json:
            origin_val = parsed_json.get("origin")
            dest_val = parsed_json.get("destination")
            via_val = parsed_json.get("via")
            explanation = parsed_json.get("explanation", fixed_text)
            # If the model gave IATA codes, normalize them; else keep raw for normalize_trip
            return (
                resolve_location(origin_val) if origin_val else None,
                resolve_location(dest_val) if dest_val else None,
                explanation
            )

        # --- tolerant regex extraction for common LLM free-text outputs ---
        def extract_field(txt: str, keys):
            for key in keys:
                # e.g., "origin: Delhi" or "origin - Delhi" or "origin = Delhi"
                m = re.search(rf'{key}\s*[:=\-]\s*([A-Za-z ]{{2,40}})', txt, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            # try "from X to Y" pattern:
            m2 = re.search(r'from\s+([A-Za-z ]{{2,40}})\s+to\s+([A-Za-z ]{{2,40}})', txt, flags=re.IGNORECASE)
            if m2:
                return m2.group(1).strip()
            return None

        origin_val = extract_field(fixed_text, ["origin", "from"])
        dest_val = extract_field(fixed_text, ["destination", "to", "dest"])
        # Also check "from ... to ..." for destination
        m_to = re.search(r'from\s+([A-Za-z ]{{2,40}})\s+to\s+([A-Za-z ]{{2,40}})', fixed_text, flags=re.IGNORECASE)
        if m_to:
            origin_val = origin_val or m_to.group(1).strip()
            dest_val = dest_val or m_to.group(2).strip()

        via_val = None
        m_via = re.search(r'(via|stopover)\s+([A-Za-z ]{{2,40}})', fixed_text, flags=re.IGNORECASE)
        if m_via:
            via_val = m_via.group(2).strip()

        if origin_val or dest_val or via_val:
            # Normalize if they look like IATA codes
            return (
                resolve_location(origin_val) if origin_val else None,
                resolve_location(dest_val) if dest_val else None,
                fixed_text  # explanation
            )

        # If no JSON found and no regex extracted, fall back to older behavior
        parsed = normalize_trip(user_query)  # note: we ignore llm_correction_text here
        return parsed["origin_iata"], parsed["destination_iata"], parsed.get("via_iata")  # explanation? maybe fixed_text

    except asyncio.TimeoutError:
        logger.warning("LLM city correction timed out")
    except Exception as e:
        logger.warning(f"LLM city correction failed: {e}")
    return None, None, None

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
    skip_llm: bool = False   # if True, return data without LLM explanation
) -> Union[PlanResult, MultiCityResult, Dict]:
    """Internal implementation without top-level timeout. Used for non‑streaming mode."""
    # Prevent excessive recursion
    if depth >= MAX_RECURSION_DEPTH:
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
    warnings = []
    # Always provide a debug_info dict so later code paths can safely augment it.
    debug_info: Dict[str, Any] = {}
    # Ensure filtered_count always exists to avoid UnboundLocalError later
    filtered_count: int = 0
    normalization_debug: Dict[str, Any] = {
        "input": {
            "origin": origin,
            "destination": destination,
            "date": date,
            "user_query": user_query,
        }
    }
    relaxation_attempts: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Parse intent (overrides explicit params)
    # ------------------------------------------------------------------
    start = time.monotonic()
    # Only parse if there is meaningful user input; otherwise start with empty intent.
    if user_query:
        intent = parse_intent(user_query)
    else:
        intent = ParsedIntent()

    # Override with explicit parameters if provided, using central resolver
    if origin:
        intent.origin_iata = _resolve_city_to_iata(origin)
    if destination:
        intent.destination_iata = _resolve_city_to_iata(destination)
    if date:
        intent.date = date

    # Never propagate malformed airport values downstream.
    intent.origin_iata = _sanitize_iata_code(intent.origin_iata)
    intent.destination_iata = _sanitize_iata_code(intent.destination_iata)
    normalization_debug["after_initial_parse"] = {
        "origin_iata": intent.origin_iata,
        "destination_iata": intent.destination_iata,
    }

    # Trip type resolution: explicit parameter > parsed intent > fallback "Business"
    resolved_trip_type = trip_type or intent.trip_type or "Business"
    intent.trip_type = resolved_trip_type

    # Sanity check: stopover city cannot be same as origin or destination
    if intent.stopover_city:
        stopover_lower = intent.stopover_city.lower()
        if (intent.origin_iata and stopover_lower == intent.origin_iata.lower()) or \
           (intent.destination_iata and stopover_lower == intent.destination_iata.lower()):
            logger.warning(f"Stopover city '{intent.stopover_city}' same as origin/destination; ignoring.")
            intent.stopover_city = None

    phases['intent_parsing'] = time.monotonic() - start

    # If we still lack origin/destination, try LLM correction only if user_query exists
    llm_correction_explanation = None
    if user_query and (not intent.origin_iata or not intent.destination_iata):
        logger.info("Missing origin/destination, attempting LLM correction")
        start = time.monotonic()
        corrected_origin, corrected_dest, explanation = await correct_cities_with_llm(user_query)
        normalized_corrected_origin = _sanitize_iata_code(corrected_origin)
        normalized_corrected_dest = _sanitize_iata_code(corrected_dest)
        if corrected_origin:
            intent.origin_iata = normalized_corrected_origin
        if corrected_dest:
            intent.destination_iata = normalized_corrected_dest
        if explanation:
            llm_correction_explanation = explanation
        normalization_debug["llm_correction"] = {
            "attempted": True,
            "raw_origin": corrected_origin,
            "raw_destination": corrected_dest,
            "normalized_origin_iata": normalized_corrected_origin,
            "normalized_destination_iata": normalized_corrected_dest,
            "explanation": llm_correction_explanation,
        }
        phases['city_correction'] = time.monotonic() - start

    # Deterministic non-LLM fallback for compact route forms (e.g., "DEL BOM ...", "Delhi Mumbai ...").
    if user_query and (not intent.origin_iata or not intent.destination_iata):
        inferred_origin, inferred_dest, infer_trace = _infer_route_pair_from_query(user_query)
        if not intent.origin_iata and inferred_origin:
            intent.origin_iata = _sanitize_iata_code(inferred_origin)
        if not intent.destination_iata and inferred_dest:
            intent.destination_iata = _sanitize_iata_code(inferred_dest)
        normalization_debug["route_inference"] = infer_trace

    normalization_debug["final"] = {
        "origin_iata": intent.origin_iata,
        "destination_iata": intent.destination_iata,
    }

    if not intent.origin_iata or not intent.destination_iata:
        logger.warning(
            "Route normalization failed after correction",
            extra={
                "origin_iata": intent.origin_iata,
                "destination_iata": intent.destination_iata,
                "user_query": user_query,
            },
        )
        return {
            "error": "Could not determine origin or destination airport after AI correction.",
            "debug_info": {
                "phases": phases.copy(),
                "intent": intent.model_dump(),
                "normalization": normalization_debug,
                "relaxation_attempts": relaxation_attempts,
            },
        }

    # ------------------------------------------------------------------
    # 2. Determine search date and return date
    # ------------------------------------------------------------------
    if intent.date:
        try:
            base_date = datetime.strptime(intent.date, "%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"Invalid date format '{intent.date}'; expected YYYY-MM-DD") from e
    else:
        # Default to tomorrow if no date provided
        base_date = datetime.now() + timedelta(days=1)
    search_date = base_date.strftime("%Y-%m-%d")

    if not intent.return_date and intent.trip_duration_days:
        intent.return_date = (base_date + timedelta(days=intent.trip_duration_days)).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Handle stopover (multicity) – sequential legs, but note: this is for multi-segment, not true stopover
    # For true stopover, we should filter flights after search.
    # ------------------------------------------------------------------
    if intent.stopover_city:
        via_iata = resolve_location(intent.stopover_city) if intent.stopover_city else None
        if via_iata and via_iata not in (intent.origin_iata, intent.destination_iata):
            # Fetch leg1
            leg1 = await _plan_trip_internal(
                origin=intent.origin_iata,
                destination=via_iata,
                date=search_date,
                user_query="",
                trip_type=intent.trip_type,
                depth=depth+1,
                flight_tool=flight_tool,
                weather_tool=weather_tool,
                skip_llm=True
            )
            # Fetch leg2
            leg2 = await _plan_trip_internal(
                origin=via_iata,
                destination=intent.destination_iata,
                date=search_date,
                user_query="",
                trip_type=intent.trip_type,
                depth=depth+1,
                flight_tool=flight_tool,
                weather_tool=weather_tool,
                skip_llm=True
            )

            if isinstance(leg1, PlanResult) and isinstance(leg2, PlanResult):
                # Single LLM call covering both legs at once
                l1_flight = leg1.best_flight if isinstance(leg1.best_flight, Flight) else Flight(**leg1.best_flight)
                l2_flight = leg2.best_flight if isinstance(leg2.best_flight, Flight) else Flight(**leg2.best_flight)
                l1_weather = leg1.weather or {}
                l2_weather = leg2.weather or {}

                def _fmt_weather(w):
                    return f"{w.get('condition','?')}, {w.get('temperature_c','?')}°C"

                combined_trip_description = (
                    f"a multi-city trip via {intent.stopover_city}\n\n"
                    f"LEG 1: {intent.origin_iata} → {via_iata} on {search_date}\n"
                    f"  Flight: {l1_flight.airline} {l1_flight.flight_no} | "
                    f"{l1_flight.departure_time} → {l1_flight.arrival_time} | "
                    f"{l1_flight.duration_min}min | {l1_flight.price_inr} | Stops: {l1_flight.stops}\n"
                    f"  Weather at {via_iata} (stopover city): {_fmt_weather(l1_weather)}\n"
                    f"  NOTE: The {_fmt_weather(l1_weather)} temperature is FOR {via_iata} ONLY.\n\n"
                    f"LEG 2: {via_iata} → {intent.destination_iata} on {search_date}\n"
                    f"  Flight: {l2_flight.airline} {l2_flight.flight_no} | "
                    f"{l2_flight.departure_time} → {l2_flight.arrival_time} | "
                    f"{l2_flight.duration_min}min | {l2_flight.price_inr} | Stops: {l2_flight.stops}\n"
                    f"  Weather at {intent.destination_iata} (final destination): {_fmt_weather(l2_weather)}\n"
                    f"  NOTE: The {_fmt_weather(l2_weather)} temperature is FOR {intent.destination_iata} ONLY.\n\n"
                )

                # Add per-leg temperature ordering constraints
                def _temp_rule(w, loc):
                    t_min = w.get("temp_min_c")
                    t_max = w.get("temp_max_c")
                    if t_min is not None and t_max is not None and t_max != t_min:
                        return (
                            f"TEMP RULE FOR {loc}: Daily low={t_min}°C, daily high={t_max}°C. "
                            f"NEVER describe {t_max}°C as the 'low' temperature.\n"
                        )
                    return ""
                combined_trip_description += _temp_rule(l1_weather, via_iata)
                combined_trip_description += _temp_rule(l2_weather, intent.destination_iata)
                combined_trip_description += (
                    "Label your response clearly as 'Leg 1:' and 'Leg 2:' sections. "
                    "Do NOT mix temperatures between legs."
                )

                # --- Enrich with stopover filter info ---
                # Count how many itineraries actually have this stopover (simple: at least one leg has layover_airports containing via_iata)
                matched_leg1 = 1 if via_iata in l1_flight.layover_airports else 0
                matched_leg2 = 1 if via_iata in l2_flight.layover_airports else 0
                matched_count = matched_leg1 + matched_leg2
                debug_info_stopover = {
                    "requested": intent.stopover_city,
                    "resolved_iata": via_iata,
                    "matched_count": matched_count
                }
                logger.info("Stopover filter applied", extra=debug_info_stopover)

                if matched_count > 0:
                    # Use first matching flight's layover_info
                    top = l1_flight if matched_leg1 else l2_flight
                    layover_text = top.layover_info or "No layover details available"
                    leg_lines = []
                    # We don't have full leg list in Flight; but we have layover_info string. Use that.
                    combined_trip_description += f"\nTop matched itinerary has layovers: {layover_text}\nPlease explain the layovers and whether this matches the user's 'via {intent.stopover_city}' request in plain language.\n"
                else:
                    combined_trip_description += f"\nNo itineraries were found that stop via {intent.stopover_city}. Please explain that no flights matched the stopover requirement and suggest nearby alternatives or changing the stopover city.\n"

                # ADD DEBUG LOGGING FOR COMBINED TRIP DESCRIPTION
                logger.debug(
                    "Multi-city combined_trip_description (first 500 chars): %s",
                    combined_trip_description[:500].replace('\n', ' ')
                )

                # FIX: Use final destination weather (l2_weather) for the standard weather block,
                # while the combined_trip_description already contains per-leg weather.
                combined_llm = await generate_explanation(
                    user_query=user_query,
                    intent=intent,
                    best_flight=l1_flight,
                    weather=l2_weather,          # final destination weather for the standard block
                    all_flights=[l1_flight, l2_flight],
                    filters_applied="multi-city stopover trip",
                    trip_description=combined_trip_description,
                    warnings=warnings,
                    price_insights_str="",       # Not used in stopover for now
                    price_analysis_str="",
                    price_prediction_str="",
                )

                # Split the combined response on the "Leg 2" boundary so each leg gets its own slice.
                split_match = re.search(r'\bLeg\s*2\b', combined_llm, re.IGNORECASE)
                if split_match and split_match.start() > 30:
                    leg1.llm_response = combined_llm[:split_match.start()].strip()
                    leg2.llm_response = combined_llm[split_match.start():].strip()
                    # If either slice is still too short, augment with template rather than fully replacing
                    if len(leg1.llm_response) < 300:
                        leg1.llm_response += (
                            f"\n\nWeather details for {via_iata}: "
                            f"Daily low: {l1_weather.get('temp_min_c', 'N/A')}°C, "
                            f"daily high: {l1_weather.get('temp_max_c', 'N/A')}°C. "
                            f"Pack suitable clothing for {str(l1_weather.get('condition', 'the forecast')).lower()} "
                            f"weather with temperatures around {l1_weather.get('temperature_c', 'N/A')}°C at the stopover. "
                            f"Precipitation chance: {l1_weather.get('precipitation_chance', 'N/A')}%."
                        )
                    if len(leg2.llm_response) < 300:
                        leg2.llm_response += (
                            f"\n\nWeather details for {intent.destination_iata}: "
                            f"Daily low: {l2_weather.get('temp_min_c', 'N/A')}°C, "
                            f"daily high: {l2_weather.get('temp_max_c', 'N/A')}°C. "
                            f"Pack suitable clothing for {str(l2_weather.get('condition', 'the forecast')).lower()} "
                            f"weather with temperatures around {l2_weather.get('temperature_c', 'N/A')}°C at your destination. "
                            f"Precipitation chance: {l2_weather.get('precipitation_chance', 'N/A')}%."
                        )
                else:
                    # Combined call failed or was too short. Use deterministic summaries for BOTH legs.
                    l1_wx = l1_weather
                    leg1.llm_response = (
                        f"Leg 1: {intent.origin_iata} to {via_iata} on {search_date}\n\n"
                        f"Flight: {l1_flight.airline} {l1_flight.flight_no} | "
                        f"{l1_flight.departure_time} → {l1_flight.arrival_time} | "
                        f"{l1_flight.duration_min} min | {l1_flight.price_inr} | Stops: {l1_flight.stops}\n\n"
                        f"Weather at {via_iata} (stopover city): {_fmt_weather(l1_wx)}. "
                        f"Daily low: {l1_wx.get('temp_min_c', 'N/A')}°C, "
                        f"daily high: {l1_wx.get('temp_max_c', 'N/A')}°C. "
                        f"Pack suitable clothing for {str(l1_wx.get('condition', 'the forecast')).lower()} "
                        f"weather with temperatures around {l1_wx.get('temperature_c', 'N/A')}°C "
                        f"at the stopover. "
                        f"Precipitation chance: {l1_wx.get('precipitation_chance', 'N/A')}%."
                    )
                    l2_wx = l2_weather
                    leg2.llm_response = (
                        f"Leg 2: {via_iata} to {intent.destination_iata} on {search_date}\n\n"
                        f"Flight: {l2_flight.airline} {l2_flight.flight_no} | "
                        f"{l2_flight.departure_time} → {l2_flight.arrival_time} | "
                        f"{l2_flight.duration_min} min | {l2_flight.price_inr} | Stops: {l2_flight.stops}\n\n"
                        f"Weather at {intent.destination_iata} (final destination): {_fmt_weather(l2_wx)}. "
                        f"Daily low: {l2_wx.get('temp_min_c', 'N/A')}°C, "
                        f"daily high: {l2_wx.get('temp_max_c', 'N/A')}°C. "
                        f"Pack suitable clothing for {str(l2_wx.get('condition', 'the forecast')).lower()} "
                        f"weather with temperatures around {l2_wx.get('temperature_c', 'N/A')}°C "
                        f"at your final destination. "
                        f"Precipitation chance: {l2_wx.get('precipitation_chance', 'N/A')}%."
                    )

                # Add API trace and stopover filter to each leg's debug_info
                leg1.debug_info = leg1.debug_info or {}
                leg1.debug_info["stopover_filter"] = debug_info_stopover
                leg1.debug_info["api_trace"] = {
                    "flight": {
                        "request": {
                            "departure": intent.origin_iata,
                            "arrival": via_iata,
                            "date": search_date,
                            "intent_date": intent.date,
                            "return_date": intent.return_date if intent.return_date else None,
                        },
                        "raw_count": 1,
                        "filtered_count": 1,
                        "best_flight_no": l1_flight.flight_no,
                        "raw_response": [dict(l1_flight)],
                    },
                    "weather": {
                        "request": {
                            "location": via_iata,
                            "date": search_date,
                        },
                        "forecast_date": l1_weather.get("forecast_date"),
                        "condition": l1_weather.get("condition"),
                        "temperature_c": l1_weather.get("temperature_c"),
                        "raw_response": l1_weather,
                    }
                }
                leg2.debug_info = leg2.debug_info or {}
                leg2.debug_info["stopover_filter"] = debug_info_stopover
                leg2.debug_info["api_trace"] = {
                    "flight": {
                        "request": {
                            "departure": via_iata,
                            "arrival": intent.destination_iata,
                            "date": search_date,
                            "intent_date": intent.date,
                            "return_date": intent.return_date if intent.return_date else None,
                        },
                        "raw_count": 1,
                        "filtered_count": 1,
                        "best_flight_no": l2_flight.flight_no,
                        "raw_response": [dict(l2_flight)],
                    },
                    "weather": {
                        "request": {
                            "location": intent.destination_iata,
                            "date": search_date,
                        },
                        "forecast_date": l2_weather.get("forecast_date"),
                        "condition": l2_weather.get("condition"),
                        "temperature_c": l2_weather.get("temperature_c"),
                        "raw_response": l2_weather,
                    }
                }
                return MultiCityResult(legs=[leg1, leg2])
        # fallback: if via_iata not resolved or is same as origin/destination, treat as normal trip

    # ------------------------------------------------------------------
    # 2 & 3. Fetch ALL API data — flights and weather, with smart reuse and parallelism
    # ------------------------------------------------------------------
    start_apis = time.monotonic()

    # Outbound flight task
    flight_task = asyncio.create_task(
        flight_tool(
            departure=intent.origin_iata,
            arrival=intent.destination_iata,
            date=search_date,
            max_results=10,
            return_date=intent.return_date if intent.return_date else None,
        )
    )

    # We'll gather flight and outbound weather first (only outbound weather for now)
    gather_tasks = [flight_task]
    # We may add weather tasks later depending on key availability and date limits
    # For now, we just await the flight result.
    try:
        gather_results = await asyncio.gather(*gather_tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Parallel data fetch failed: {e}")
        return {"error": str(e), "search_date": search_date, "debug_info": {"phases": phases.copy()}}

    result_iter = iter(gather_results)
    flight_result = next(result_iter)

    # --- Attempt to reuse return flights from the initial API response ---
    return_flight_result = None
    if intent.return_date and not intent.stopover_city:
        try:
            # flight_result may be tuple (flights, price_insights) or just flights
            raw_flights = flight_result[0] if isinstance(flight_result, tuple) else flight_result
            if isinstance(raw_flights, list) and raw_flights:
                # Try to extract flights that match return date
                return_hits = []
                outbound_hits = []
                for f in raw_flights:
                    # Some flight dicts have 'date' field; some might have 'search_date'
                    flight_date = f.get('date') or f.get('search_date')
                    if flight_date == intent.return_date:
                        return_hits.append(f)
                    elif flight_date == search_date:
                        outbound_hits.append(f)
                if return_hits:
                    logger.info(f"Reusing {len(return_hits)} return flights from initial API response")
                    return_flight_result = return_hits
                else:
                    logger.debug("No return flights in initial response; will perform separate search")
            else:
                logger.debug("Flight result not a list, cannot extract return legs")
        except Exception as e:
            logger.warning(f"Error while inspecting flight_result for return leg: {e}; will fall back to separate call")
            return_flight_result = None

    # Now, after flight result, we can decide on weather fetching.
    today = datetime.now().date()
    return_date_str = intent.return_date if intent.return_date else None

    # Determine which weather dates we need (within forecast limit)
    dates_to_fetch = []
    # Outbound weather
    try:
        outbound_dt = datetime.strptime(search_date, "%Y-%m-%d").date()
        if (outbound_dt - today).days <= WEATHER_FORECAST_MAX_DAYS:
            dates_to_fetch.append(("outbound", intent.destination_iata, search_date))
    except Exception:
        # if date invalid, skip weather
        pass

    # Return weather (if we have return date and not stopover)
    if intent.return_date and not intent.stopover_city:
        try:
            return_dt = datetime.strptime(intent.return_date, "%Y-%m-%d").date()
            if (return_dt - today).days <= WEATHER_FORECAST_MAX_DAYS:
                dates_to_fetch.append(("return", intent.origin_iata, intent.return_date))
        except Exception:
            pass

    # Fetch weather intelligently: parallel if enough keys, else sequential
    weather_out = None
    weather_ret = None
    weather_present_out = False
    weather_present_ret = False
    weather_reason_out = None
    weather_reason_ret = None

    if dates_to_fetch:
        # Check how many weather keys are active
        status = await api_key_manager.get_status()
        weather_keys_active = sum(1 for e in status.get("weather", []) if e["active"])

        if len(dates_to_fetch) == 2 and weather_keys_active >= 2:
            # Run both weather fetches in parallel
            logger.info("Running parallel weather fetches (2 keys available)")
            tasks = []
            for _, loc, dt in dates_to_fetch:
                tasks.append(asyncio.create_task(
                    cached_weather(location=loc, travel_date=dt, units="metric")  # FIXED: use cached_weather
                ))
            weather_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Map back: first task is outbound (if both), second return
            if weather_results[0] is not None and not isinstance(weather_results[0], Exception):
                weather_out = weather_results[0]
                weather_present_out = True
            else:
                weather_out = {}
                weather_present_out = False
                weather_reason_out = "api_failure"
            if len(weather_results) > 1:
                if weather_results[1] is not None and not isinstance(weather_results[1], Exception):
                    weather_ret = weather_results[1]
                    weather_present_ret = True
                else:
                    weather_ret = {}
                    weather_present_ret = False
                    weather_reason_ret = "api_failure"
        else:
            # Sequential (safe)
            logger.info("Running sequential weather fetches (<=1 key available or only one date)")
            for leg, loc, dt in dates_to_fetch:
                try:
                    res = await cached_weather(location=loc, travel_date=dt, units="metric")  # FIXED: use cached_weather
                    if leg == "outbound":
                        weather_out = res
                        weather_present_out = True
                    else:
                        weather_ret = res
                        weather_present_ret = True
                except Exception as e:
                    logger.warning(f"Weather fetch for {leg} failed: {e}")
                    if leg == "outbound":
                        weather_out = {}
                        weather_present_out = False
                        weather_reason_out = "api_failure"
                    else:
                        weather_ret = {}
                        weather_present_ret = False
                        weather_reason_ret = "api_failure"
    else:
        # No weather within forecast window, use placeholders and set reason
        if not weather_out and (not dates_to_fetch or dates_to_fetch[0][0] != "outbound"):
            weather_out = {"condition": "Unpredictable this far in the future", "temperature_c": "N/A"}
            weather_present_out = False
            weather_reason_out = "forecast_horizon_exceeded"
        if intent.return_date and not intent.stopover_city and not weather_ret:
            weather_ret = {"condition": "Unpredictable this far in the future", "temperature_c": "N/A"}
            weather_present_ret = False
            weather_reason_ret = "forecast_horizon_exceeded"

    # PATCH: If return date exists but was beyond forecast window (not added to dates_to_fetch),
    # set the placeholder now. The else branch above only runs when dates_to_fetch is entirely empty.
    if intent.return_date and not intent.stopover_city and weather_ret is None:
        weather_ret = {"condition": "Unpredictable this far in the future", "temperature_c": "N/A"}
        weather_present_ret = False
        weather_reason_ret = "forecast_horizon_exceeded"

    # If we still don't have return_flight_result and we need it, fetch it now sequentially
    if intent.return_date and not intent.stopover_city and return_flight_result is None:
        logger.info("Performing separate return flight search")
        try:
            return_flight_result = await flight_tool(
                departure=intent.destination_iata,
                arrival=intent.origin_iata,
                date=return_date_str,
                max_results=10,
            )
        except Exception as e:
            logger.warning(f"Return flight search failed: {e}")
            return_flight_result = e

    phases['api_parallel'] = time.monotonic() - start_apis

    # --- Process flight_result and extract outbound flights and price_insights ---
    if isinstance(flight_result, Exception):
        logger.error(f"Flight search failed: {flight_result}")
        return {"error": str(flight_result), "search_date": search_date, "debug_info": {"phases": phases.copy()}}
    if isinstance(flight_result, tuple):
        all_flights, price_insights_raw = flight_result
    else:
        all_flights = flight_result
        price_insights_raw = None
    if not all_flights:
        return {"warning": "No live flights found.", "fallback": True, "search_date": search_date, "weather": {}, "debug_info": {"phases": phases.copy()}}

    # If we extracted return flights earlier, we need to separate them from all_flights.
    if return_flight_result and isinstance(return_flight_result, list) and not isinstance(return_flight_result, Exception):
        # Remove any flights in all_flights that have return date
        original_len = len(all_flights)
        all_flights = [f for f in all_flights if f.get('date') != intent.return_date]
        if len(all_flights) != original_len:
            logger.info(f"Removed {original_len - len(all_flights)} return-leg flights from outbound list")

    weather_data = weather_out if weather_out and not isinstance(weather_out, Exception) else {}

    # Capture raw flights before normalization (all_flights now only outbound)
    _raw_flights_before_normalize = [
        dict(f) if isinstance(f, dict) else vars(f)
        for f in all_flights
    ]

    # At this point, all_flights is a list of raw flight dicts; normalize them
    all_flights = normalize_flights(all_flights, search_date)

    # ------------------------------------------------------------------
    # 4. Apply filters and rank (with fallback) – update effective intent
    # ------------------------------------------------------------------
    start = time.monotonic()
    effective_intent = intent.model_copy(deep=True)  # start with original intent
    ranking_intent = intent

    # Apply standard filters first
    filtered, filter_warnings = filter_flights(all_flights, effective_intent)
    warnings.extend(filter_warnings)
    filtered_count = len(filtered) if filtered is not None else 0
    relaxation_attempts.append({
        "step": "strict_filters",
        "matched_count": filtered_count,
    })

    # If user requested a stopover, we need to filter flights that actually have that stopover
    # and enrich the prompt.
    stopover_matched_itins = []
    stopover_iata = None
    if intent.stopover_city:
        stopover_text = intent.stopover_city
        try:
            stopover_iata = resolve_location(stopover_text)
        except Exception:
            stopover_iata = None

        # Filter flights by layover_airports
        if stopover_iata:
            stopover_matched_itins = [f for f in all_flights if stopover_iata in (f.layover_airports or [])]
        else:
            # fallback to substring match in layover_info
            lower_v = stopover_text.lower()
            stopover_matched_itins = [f for f in all_flights if lower_v in (f.layover_info or "").lower()]

        # Update debug info
        debug_info_stopover = {
            "requested": stopover_text,
            "resolved_iata": stopover_iata,
            "matched_count": len(stopover_matched_itins)
        }
        debug_info = debug_info or {}
        debug_info["stopover"] = debug_info_stopover
        logger.info("Stopover filter applied", extra=debug_info_stopover)

        # If we have matches, we may want to prioritize them in ranking, but we'll handle in prompt enrichment
        # For now, we keep filtered list unchanged but note the matches.
        # We'll later use stopover_matched_itins to enrich prompt.

    # If strict filtering yields nothing, apply staged relaxation in a safe priority order.
    if not filtered:
        logger.warning("No flights after strict filtering; applying staged relaxation fallback")
        relaxed_intent = intent.model_copy(deep=True)

        def _snapshot_relaxed_state() -> Dict[str, Any]:
            return {
                "preferred_airlines": list(relaxed_intent.preferred_airlines or []),
                "layover_limit_minutes": relaxed_intent.layover_limit_minutes,
                "price_limit": relaxed_intent.price_limit,
                "time_pref": relaxed_intent.time_pref,
                "wants_direct": relaxed_intent.wants_direct,
                "baggage_pref": relaxed_intent.baggage_pref,
            }

        def _try_relaxed(step: str) -> bool:
            nonlocal filtered, effective_intent, filtered_count
            candidate, candidate_warnings = filter_flights(all_flights, relaxed_intent)
            warnings.extend(candidate_warnings)
            matched_count = len(candidate) if candidate else 0
            relaxation_attempts.append({
                "step": step,
                "matched_count": matched_count,
                "state": _snapshot_relaxed_state(),
            })
            if candidate:
                filtered = candidate
                effective_intent = relaxed_intent.model_copy(deep=True)
                filtered_count = len(filtered)
                return True
            return False

        # 1) Remove preferred airline constraint
        if not filtered and relaxed_intent.preferred_airlines:
            preferred = ", ".join(relaxed_intent.preferred_airlines)
            relaxed_intent.preferred_airlines = []
            warnings.append(
                f"No exact flights were available for preferred airline(s) {preferred}; "
                "relaxed airline preference."
            )
            _try_relaxed("remove_preferred_airline")

        # 2) Remove layover duration constraint
        if not filtered and relaxed_intent.layover_limit_minutes:
            prev_layover = relaxed_intent.layover_limit_minutes
            relaxed_intent.layover_limit_minutes = None
            warnings.append(
                f"No flights met the max layover limit ({prev_layover} minutes); "
                "relaxed layover-duration constraint."
            )
            _try_relaxed("remove_layover_limit")

        # 3) Increase price cap by 25–50%
        if not filtered and relaxed_intent.price_limit:
            original_cap = relaxed_intent.price_limit
            for factor in (1.25, 1.50):
                new_cap = max(int(original_cap * factor), original_cap + 1)
                if new_cap <= relaxed_intent.price_limit:
                    continue
                relaxed_intent.price_limit = new_cap
                warnings.append(
                    f"No flights were found under ₹{original_cap}. "
                    f"Increased budget cap to ₹{new_cap} and retried."
                )
                if _try_relaxed(f"increase_price_cap_{int(factor*100)}pct"):
                    break

        # 4) Increase allowed duration (broaden departure time window if set)
        if not filtered and relaxed_intent.time_pref:
            prev_time_pref = relaxed_intent.time_pref
            relaxed_intent.time_pref = None
            warnings.append(
                f"No flights matched the {prev_time_pref} departure window; "
                "expanded the allowed time window."
            )
            _try_relaxed("remove_time_window")

        # 5) Remove direct-only requirement
        if not filtered and relaxed_intent.wants_direct:
            relaxed_intent.wants_direct = False
            warnings.append("No direct flights matched all constraints; allowing connecting options.")
            _try_relaxed("remove_direct_only")

        # Existing behavior retained: relax baggage filter if still empty.
        if not filtered and relaxed_intent.baggage_pref:
            prev_baggage = relaxed_intent.baggage_pref
            relaxed_intent.baggage_pref = None
            warnings.append(
                f"No flights matched the {prev_baggage} baggage preference; "
                "relaxed baggage constraint."
            )
            _try_relaxed("remove_baggage_filter")

        # 6) Final fallback: choose cheapest available flights.
        if not filtered:
            effective_intent = relaxed_intent.model_copy(deep=True)
            effective_intent.flight_pref = "cheapest"
            ranking_intent = effective_intent.model_copy(deep=True)

            filtered = sorted(all_flights, key=lambda f: price_to_int(f.price_inr))
            filtered_count = len(filtered)
            relaxation_attempts.append({
                "step": "fallback_cheapest_available",
                "matched_count": filtered_count,
                "state": _snapshot_relaxed_state(),
            })

            if filtered:
                cheapest_price = price_to_int(filtered[0].price_inr)
                if intent.price_limit and cheapest_price > intent.price_limit:
                    warnings.append(
                        f"No flights were found under ₹{intent.price_limit}. "
                        f"The cheapest available option is ₹{cheapest_price}."
                    )
                else:
                    warnings.append(
                        "No exact flights were found with current constraints; "
                        "showing the cheapest available option."
                    )

    if not filtered:
        # Even all_flights was empty, but we already handled that earlier
        return {"error": "Sorry, I couldn't find any flights matching your preferences."}

    ranked = rank_flights(filtered, ranking_intent)
    best_flight = ranked[0]
    ranked_count = len(ranked)
    phases['filter_rank'] = time.monotonic() - start

    # ------------------------------------------------------------------
    # Resolve booking handoff URL for best flight (non-blocking; falls back gracefully)
    # ------------------------------------------------------------------
    # Spawn booking task but don't block LLM indefinitely
    booking_task = asyncio.create_task(
        _build_booking_handoff_url_safe(
            flight=best_flight.model_dump(),
            origin=intent.origin_iata,
            destination=intent.destination_iata,
            depart_date=search_date,
            return_date=intent.return_date,
        )
    )
    try:
        # Wait up to 0.8 seconds for booking URL; if it times out, continue with placeholder
        booking_url = await asyncio.wait_for(booking_task, timeout=0.8)
    except asyncio.TimeoutError:
        logger.warning("Booking handoff timed out, using placeholder")
        booking_url = None  # placeholder, will be updated later if needed
    except Exception as _e:
        logger.warning(f"build_booking_handoff_url failed: {_e}")
        booking_url = None

    if booking_url:
        best_flight = best_flight.model_copy(update={"handoff_url": booking_url})

    # ------------------------------------------------------------------
    # 5. Build description for LLM using effective_intent
    # ------------------------------------------------------------------
    filter_parts = []
    if effective_intent.time_pref: filter_parts.append(f"{effective_intent.time_pref} flights")
    if effective_intent.price_limit: filter_parts.append(f"under ₹{effective_intent.price_limit}")
    if effective_intent.wants_direct: filter_parts.append("direct flights only")
    # Always include original airline preference so LLM knows what was asked, even if relaxed
    if intent.preferred_airlines:
        filter_parts.append(f"preferred airlines: {', '.join(intent.preferred_airlines)}")
    elif effective_intent.preferred_airlines:
        filter_parts.append(f"preferred airlines: {', '.join(effective_intent.preferred_airlines)}")
    if effective_intent.layover_limit_minutes: filter_parts.append(f"max layover: {effective_intent.layover_limit_minutes//60}h")
    if effective_intent.baggage_pref: filter_parts.append(f"{effective_intent.baggage_pref} baggage only")
    if effective_intent.wants_eco:
        filter_parts.append("eco-friendly / lowest carbon emissions")
    # Include flight preference (cheapest/shortest/balanced) in filters
    if effective_intent.flight_pref == "cheapest":
        filter_parts.append("cheapest / lowest price")
    elif effective_intent.flight_pref == "shortest":
        filter_parts.append("shortest flight duration")
    elif effective_intent.flight_pref == "balanced":
        filter_parts.append("balanced price and duration")
    # NEW: deep search flag
    if intent.deep_search:
        filter_parts.append("deep search for absolute cheapest price")
    filters_applied = "; ".join(filter_parts) if filter_parts else "no specific filters"

    trip_description = f"a {intent.trip_type} trip"
    if intent.stopover_city:
        trip_description += f" via {intent.stopover_city}"
    if intent.return_date:
        trip_description += f", returning on {intent.return_date}"

    # ------------------------------------------------------------------
    # 5a. Enrich trip_description with stopover filter details (if any)
    # ------------------------------------------------------------------
    if intent.stopover_city and stopover_matched_itins is not None:
        if stopover_matched_itins:
            # Use top matched flight (first in filtered list? or from stopover_matched_itins)
            top = stopover_matched_itins[0]
            layover_text = top.layover_info or "No layover details available"
            # Build a small leg table if possible (but Flight may not have legs; use layover_info)
            leg_lines = []
            # Since we don't have full legs in Flight, we'll rely on layover_info.
            trip_description += (
                f"\n\nSTOPOVER REQUEST: {intent.stopover_city} (resolved: {stopover_iata})\n"
                f"Top matched itinerary layover(s): {layover_text}\n"
                "Please explain – in clear, helpful detail (about 80–120 words) – whether this itinerary satisfies the user's 'via' request; "
                "if not, explain why and suggest concrete alternatives (e.g., change stopover to X, allow 1 stop, or search nearby airports)."
            )
        else:
            # No matches: give the LLM the top few candidate flights and instruct it to explain failure and alternatives
            sample_lines = []
            for idx, f in enumerate(filtered[:3]):
                sample_lines.append(
                    f"{idx+1}. {f.airline} {f.flight_no} {f.departure_time}->{f.arrival_time} duration {f.duration_min}m layovers: {f.layover_airports or '[]'}"
                )
            trip_description += (
                f"\n\nUser asked for flights 'via {intent.stopover_city}', but no flights in the returned results stop via {stopover_text}.\n"
                "Here are the top 3 candidate flights returned by the search:\n" + "\n".join(sample_lines) + "\n"
                "Please write a clear, helpful explanation (about 80–120 words) that says no flights matched the requested stopover, why that may be (e.g., no itineraries with that layover in results), and suggest concrete alternatives the user can try (e.g., search for flights with 1 stop, allow a different via city, or change date)."
            )

    # ------------------------------------------------------------------
    # 6. Convert weather to dict for consistent serialization (safe version)
    # ------------------------------------------------------------------
    if hasattr(weather_data, "model_dump"):
        weather_dict = weather_data.model_dump()
    elif hasattr(weather_data, "__dict__"):
        weather_dict = dict(vars(weather_data))
    elif isinstance(weather_data, dict):
        weather_dict = weather_data
    else:
        # Last-resort safe serialization
        weather_dict = json.loads(json.dumps(weather_data, default=str))

    # ------------------------------------------------------------------
    # 7. Prepare debug_info with extra data needed for streaming (ALWAYS populated)
    # ------------------------------------------------------------------
    # Parse price_insights once here so both non-streaming and streaming paths can use it
    _price_insights_obj = None
    price_insights_str = ""
    price_analysis_str = ""
    price_prediction_str = ""
    if price_insights_raw:
        try:
            # Existing formatting
            _price_insights_obj = _parse_price_insights_safe(
                {"price_insights": price_insights_raw},
                current_price_inr=price_to_int(best_flight.price_inr),
            )
            if _price_insights_obj:
                price_insights_str = _format_price_insights_for_llm_safe(_price_insights_obj)

            # NEW: price analysis and prediction
            price_history = price_insights_raw.get("price_history", [])
            if price_history:
                analysis = _analyze_price_trend_safe(price_history)
                if analysis:
                    price_analysis_str = (
                        f"Price trend: {analysis.get('trend', 'stable')}. "
                        f"Average price for this route: ₹{analysis.get('average_price', 'N/A')}. "
                        f"Current price is {analysis.get('price_level', 'average')}."
                    )
                prediction = _predict_future_price_safe(price_history)
                if prediction:
                    price_prediction_str = (
                        f"Price prediction: {prediction.get('trend', 'prices may change')}. "
                        f"{prediction.get('advice', '')}"
                    )
        except Exception as _pi_err:
            logger.warning("price_insights formatting failed", extra={"error": str(_pi_err)})

    # ------------------------------------------------------------------
    # Add API trace to debug_info (now defined)
    # ------------------------------------------------------------------
    debug_info = debug_info or {}
    debug_info.update({
        "phases": phases.copy(),
        "intent": intent.model_dump(),
        "effective_intent": effective_intent.model_dump(),
        "filters_applied": filters_applied,
        "trip_description": trip_description,
        "all_flights": [f.model_dump() for f in all_flights],
        "filtered_count": filtered_count,
        "ranked_count": ranked_count,
        "price_insights_str": price_insights_str,
        "price_analysis_str": price_analysis_str,
        "price_prediction_str": price_prediction_str,
        "normalization": normalization_debug,
        "relaxation_attempts": relaxation_attempts,
        # NEW — full API traceability including raw responses
        "api_trace": {
            "flight": {
                "request": {
                    "departure": intent.origin_iata,
                    "arrival": intent.destination_iata,
                    "date": search_date,
                    "intent_date": intent.date,
                    "return_date": intent.return_date if intent.return_date else None,
                },
                "raw_count": len(all_flights),
                "filtered_count": filtered_count,
                "best_flight_no": best_flight.flight_no,
                "raw_response": _raw_flights_before_normalize,
            },
            "weather": {
                "request": {
                    "location": intent.destination_iata,
                    "date": search_date,
                },
                "forecast_date": weather_dict.get("forecast_date"),
                "condition": weather_dict.get("condition"),
                "temperature_c": weather_dict.get("temperature_c"),
                "raw_response": weather_dict,
            },
        },
    })

    # Log a stripped api_trace (no raw_response, no booking tokens) for clean log diffs
    _log_trace = {
        "flight": {
            "request": debug_info["api_trace"]["flight"]["request"],
            "raw_count": debug_info["api_trace"]["flight"]["raw_count"],
            "filtered_count": debug_info["api_trace"]["flight"]["filtered_count"],
            "best_flight_no": debug_info["api_trace"]["flight"]["best_flight_no"],
        },
        "weather": {
            "request": debug_info["api_trace"]["weather"]["request"],
            "forecast_date": debug_info["api_trace"]["weather"]["forecast_date"],
            "condition": debug_info["api_trace"]["weather"]["condition"],
            "temperature_c": debug_info["api_trace"]["weather"]["temperature_c"],
        }
    }
    logger.debug("api_trace (stripped): %s", _log_trace)

    # Build return trip data for LLM prompt injection (no recursive call needed)
    return_trip_result = None
    return_flight_data = None
    return_weather_data = None
    rw_dict = {}  # ensure rw_dict is always defined for later use
    if return_flight_result and not isinstance(return_flight_result, Exception):
        raw_rt_flights = return_flight_result[0] if isinstance(return_flight_result, tuple) else return_flight_result
        rt_flights_norm = normalize_flights(raw_rt_flights, return_date_str)
        if rt_flights_norm:
            # Rank return flights using the same intent (original preferences)
            rt_ranked = rank_flights(rt_flights_norm, intent)
            if rt_ranked:
                return_flight_data = rt_ranked[0]
                return_weather_data = weather_ret if weather_ret and not isinstance(weather_ret, Exception) else {}
                # Convert return weather to dict safely
                if isinstance(return_weather_data, dict):
                    rw_dict = return_weather_data
                elif hasattr(return_weather_data, "model_dump"):
                    rw_dict = return_weather_data.model_dump()
                elif hasattr(return_weather_data, "__dict__"):
                    rw_dict = dict(vars(return_weather_data))
                else:
                    rw_dict = {}
                # Build return trip PlanResult for the response payload (no LLM yet — will be filled later)
                return_trip_result = PlanResult(
                    llm_response=None,   # filled in by combined LLM below
                    best_flight=return_flight_data.model_dump(),
                    weather=rw_dict,
                    search_date=return_date_str,
                    fallback_note="",
                    debug_info={"phases": {}},
                    warnings=None,
                    weather_present=weather_present_ret,
                    weather_reason=weather_reason_ret,
                )
                # Enhanced return leg injection with MANDATORY instruction and brevity request
                trip_description += (
                    f"\n\nRETURN FLIGHT (on {return_date_str}):\n"
                    f"- Flight: {return_flight_data.airline} {return_flight_data.flight_no} | "
                    f"{return_flight_data.departure_time} → {return_flight_data.arrival_time} | "
                    f"Price: {return_flight_data.price_inr} | Stops: {return_flight_data.stops}\n"
                    f"- Weather at {intent.origin_iata} on {return_date_str}: "
                    f"{rw_dict.get('condition')}, {rw_dict.get('temperature_c')}°C\n\n"
                    f"MANDATORY: Your response MUST cover BOTH legs but you must be EXTREMELY BRIEF.\n"
                    f"CRITICAL RULE: Your ENTIRE response must be 4 sentences maximum. Do not write long paragraphs."
                )
                # Add TEMP RULE for return leg to prevent temperature inversion
                rw_t_min = rw_dict.get("temp_min_c")
                rw_t_max = rw_dict.get("temp_max_c")
                if rw_t_min is not None and rw_t_max is not None and rw_t_max > rw_t_min:
                    trip_description += (
                        f"\nTEMP RULE FOR RETURN ({intent.origin_iata}): "
                        f"Daily low={rw_t_min}°C, daily high={rw_t_max}°C. "
                        f"NEVER describe {rw_t_max}°C as the 'low' temperature in the return section."
                    )

    if skip_llm:
        # Provide a clean summary when LLM is skipped
        llm_text = f"Flight: {best_flight.airline} {best_flight.flight_no} ({best_flight.departure_time} - {best_flight.arrival_time}). Price: {best_flight.price_inr}. Weather: {weather_dict.get('condition')}, {weather_dict.get('temperature_c')}°C."
    else:
        start = time.monotonic()
        llm_text = await generate_explanation(
            user_query=user_query,
            intent=intent,
            best_flight=best_flight,
            weather=weather_dict,
            all_flights=all_flights,
            filters_applied=filters_applied,
            trip_description=trip_description,
            warnings=warnings,
            price_insights_str=price_insights_str,
            price_analysis_str=price_analysis_str,
            price_prediction_str=price_prediction_str,
            booking_url=booking_url,  # NEW: pass booking URL if available
        )
        phases['llm_generation'] = time.monotonic() - start
        debug_info["phases"] = phases.copy()

        # No second LLM call — the combined trip_description already covers both legs.
        # Split llm_text at the RETURN section to give return_trip its own slice.
        if return_trip_result and return_flight_data:
            rt_split = re.search(r'\bRETURN\b', llm_text, re.IGNORECASE)
            if rt_split and rt_split.start() > 50:
                return_trip_result.llm_response = llm_text[rt_split.start():].strip()
            else:
                # LLM ignored the return section — build a deterministic return summary
                # so return_trip.llm_response is always meaningful and validator-compliant
                rw_cond = rw_dict.get('condition', 'N/A')
                rw_temp = rw_dict.get('temperature_c', 'N/A')
                return_trip_result.llm_response = (
                    f"Return flight on {return_date_str}: "
                    f"{return_flight_data.airline} {return_flight_data.flight_no} | "
                    f"{return_flight_data.departure_time} → {return_flight_data.arrival_time} | "
                    f"Price: {return_flight_data.price_inr} | Stops: {return_flight_data.stops}. "
                    f"Weather at {intent.origin_iata} on {return_date_str}: {rw_cond}, {rw_temp}°C. "
                    f"Pack accordingly for your return journey."
                )

    # ------------------------------------------------------------------
    # 9. Prepare result and handle round-trip
    # ------------------------------------------------------------------
    # Ensure debug_info always contains price_insights_str (Fix #5)
    result_debug = debug_info or {}
    result_debug["price_insights_str"] = price_insights_str or ""
    result_debug["price_analysis_str"] = price_analysis_str or ""
    result_debug["price_prediction_str"] = price_prediction_str or ""

    result = PlanResult(
        llm_response=llm_text,
        best_flight=best_flight.model_dump(),
        weather=weather_dict,
        search_date=search_date,
        fallback_note="",
        debug_info=result_debug,
        warnings=warnings if warnings else None,
        return_trip=return_trip_result if isinstance(return_trip_result, PlanResult) else None,
        weather_present=weather_present_out,
        weather_reason=weather_reason_out,
    )

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
                        "effective_intent": effective_intent.model_dump(),
                        "filters_applied": filters_applied,
                        "ranked_count": len(ranked),
                        "flight_pref": intent.flight_pref,
                        "trip_type": intent.trip_type,
                        "use_cloud_llm": USE_CLOUD_FALLBACK,
                        "phases": phases,
                        "warnings": warnings
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

    # ------------------------------------------------------------------
    # 11. Final safety: ensure llm_response exists when not skipped
    # ------------------------------------------------------------------
    if not skip_llm and not result.llm_response:
        result.llm_response = "I found a flight matching your criteria, but the detailed explanation is currently unavailable."

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
    action = _detect_booking_or_tracking_action(user_query)

    async def _handle_action_intent() -> Dict[str, Any]:
        # Confirm / cancel are direct booking lifecycle actions and do not require route parsing.
        if action in ("confirm_booking", "cancel_booking"):
            booking_id = _extract_booking_id(user_query)
            if booking_id is None:
                return {
                    "error": "Please provide a numeric booking id to confirm/cancel.",
                    "action": action,
                }
            if action == "confirm_booking":
                try:
                    ok = await asyncio.to_thread(_confirm_booking_safe, booking_id)
                except Exception as e:
                    return {
                        "action": action,
                        "booking_id": booking_id,
                        "success": False,
                        "error": str(e),
                    }
                return {
                    "action": action,
                    "booking_id": booking_id,
                    "success": ok,
                    "message": "Booking confirmed." if ok else "Booking could not be confirmed.",
                }
            try:
                ok = await asyncio.to_thread(_cancel_booking_safe, booking_id)
            except Exception as e:
                return {
                    "action": action,
                    "booking_id": booking_id,
                    "success": False,
                    "error": str(e),
                }
            return {
                "action": action,
                "booking_id": booking_id,
                "success": ok,
                "message": "Booking cancelled." if ok else "Booking could not be cancelled.",
            }

        # Hold / track intents require selecting a flight first.
        planned = await _plan_trip_internal(
            origin=origin,
            destination=destination,
            date=date,
            user_query=user_query,
            trip_type=trip_type,
            flights=flights,
            depth=depth,
            flight_tool=flight_tool,
            weather_tool=weather_tool,
            skip_llm=True,
        )
        if isinstance(planned, dict):
            return planned
        if isinstance(planned, MultiCityResult):
            return {
                "error": "Booking actions currently support single-leg selections only.",
                "action": action,
            }

        best_flight = planned.best_flight or {}
        dbg = planned.debug_info or {}
        parsed_intent = dbg.get("intent", {}) if isinstance(dbg, dict) else {}
        resolved_origin = parsed_intent.get("origin_iata") or origin
        resolved_destination = parsed_intent.get("destination_iata") or destination
        depart_date = planned.search_date or date

        if not resolved_origin or not resolved_destination or not depart_date:
            return {
                "error": "Could not resolve route/date required for booking action.",
                "action": action,
            }

        default_hold = int(os.getenv("BOOKING_HOLD_MINUTES", "15"))
        track_hold = int(os.getenv("PRICE_TRACK_HOLD_MINUTES", "43200"))  # 30 days
        hold_minutes = track_hold if action == "track_price" else default_hold

        try:
            held = await _hold_booking_safe(
                flight=best_flight,
                origin=resolved_origin,
                destination=resolved_destination,
                depart_date=depart_date,
                hold_minutes=hold_minutes,
            )
        except Exception as e:
            return {
                "action": action,
                "success": False,
                "error": str(e),
            }

        if action == "track_price":
            # Seed a baseline snapshot so the monitor has an initial reference point.
            try:
                _record_price_snapshot_safe(
                    origin=resolved_origin,
                    destination=resolved_destination,
                    travel_date=depart_date,
                    price_inr=float(price_to_int(_flight_value_safe(best_flight, "price_inr", 0))),
                )
            except Exception as e:
                logger.warning("record_price_snapshot failed for tracking setup", extra={"error": str(e)})

            return {
                "action": action,
                "success": True,
                "message": "We will notify you if the price of this flight drops.",
                "booking": held,
                "best_flight": best_flight,
                "monitoring_active": True,
            }

        return {
            "action": action,
            "success": True,
            "message": "Flight held successfully.",
            "booking": held,
            "best_flight": best_flight,
        }

    if action:
        if stream:
            async def action_stream() -> AsyncGenerator[str, None]:
                payload = await _handle_action_intent()
                msg = payload.get("message") or payload.get("error") or "Action processed."
                yield msg
                yield "[DONE_JSON]" + json.dumps(payload)
            return action_stream()
        return await _handle_action_intent()

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
            # Warnings are at the top level of PlanResult
            warnings = data_result.warnings or []
            debug_info = data_result.debug_info or {}
            intent_dict = debug_info.get("intent", {})
            # Use effective_intent if available for filters description
            effective_intent_dict = debug_info.get("effective_intent", intent_dict)
            all_flights_dicts = debug_info.get("all_flights", [])
            filters_applied = debug_info.get("filters_applied", "")
            trip_description = debug_info.get("trip_description", "")
            price_insights_str = debug_info.get("price_insights_str", "")
            price_analysis_str = debug_info.get("price_analysis_str", "")
            price_prediction_str = debug_info.get("price_prediction_str", "")

            # Inject return leg details if present
            if data_result.return_trip:
                rt = data_result.return_trip
                rt_flight = rt.best_flight
                rt_weather = rt.weather
                trip_description += (
                    f"\n\nRETURN LEG (on {rt.search_date}):\n"
                    f"- Flight: {rt_flight.get('airline')} {rt_flight.get('flight_no')} | {rt_flight.get('departure_time')} → {rt_flight.get('arrival_time')} | Price: {rt_flight.get('price_inr')}\n"
                    f"- Weather: {rt_weather.get('condition')}, {rt_weather.get('temperature_c')}°C\n"
                    "Please summarize the round trip, mentioning both outbound and return flights and their respective weather."
                )

            # Build prompt — kept in sync with generate_explanation (non-streaming path)
            flights_str = "\n".join([
                f"- {f['airline']} {f['flight_no']} on {f.get('date','N/A')} | "
                f"{f['departure_time']} → {f['arrival_time']} | "
                f"{f['duration_min']} min | {f['price_inr']} | "
                f"Stops: {f.get('stops', 'N/A')} | Baggage: {f.get('baggage', 'N/A')}"
                for f in all_flights_dicts[:10]
            ])

            # Format warnings if present, escalating "relaxed" warnings to mandatory instructions
            if warnings:
                processed = []
                for w in warnings:
                    if "relaxed" in w.lower():
                        processed.append(
                            "MANDATORY: Your baggage or airline preference could not be matched. "
                            "You MUST explicitly tell the user their preference was not available and "
                            "you are showing the closest alternative. Do NOT claim the flight meets their baggage requirement."
                        )
                    else:
                        processed.append(w)
                warnings_str = "\nSystem Notes/Warnings:\n- " + "\n- ".join(processed)
            else:
                warnings_str = ""

            # Ensure weather values are plain
            weather_display = {}
            for k, v in weather.items():
                weather_display[k] = v.value if hasattr(v, 'value') else v

            # Determine forecast proximity to travel date
            forecast_date_str = weather_display.get("forecast_date")
            travel_date_str = best_flight.date or intent_dict.get("date") or "your travel date"
            forecast_is_approximate = False
            approx_note = ""
            if forecast_date_str and travel_date_str and travel_date_str != "your travel date":
                try:
                    from datetime import datetime as _dt
                    t_dt = _dt.strptime(travel_date_str, "%Y-%m-%d")
                    f_dt = _dt.strptime(forecast_date_str, "%Y-%m-%d")
                    delta = abs((t_dt - f_dt).days)
                    if delta > 3:
                        forecast_is_approximate = True
                        approx_note = f" (closest available forecast; actual forecast for {forecast_date_str})"
                except Exception:
                    pass

            # Full weather string with min/max, rain/snow alerts — same as non-streaming
            forecast_label = travel_date_str
            weather_str = (
                f"Condition: {weather_display.get('condition', 'N/A')}, "
                f"Temperature: {weather_display.get('temperature_c', 'N/A')}°C "
                f"(feels like {weather_display.get('feels_like_c', 'N/A')}°C)"
            )
            if weather_display.get("temp_min_c") is not None and weather_display.get("temp_max_c") is not None:
                weather_str += f", Daily low: {weather_display['temp_min_c']}°C, Daily high: {weather_display['temp_max_c']}°C"
            weather_str += (
                f", Humidity: {weather_display.get('humidity', 'N/A')}%, "
                f"Wind: {weather_display.get('wind_kph', 'N/A')} kph, "
                f"AQI: {weather_display.get('air_quality_index', 'N/A')}"
            )
            if weather_display.get("precipitation_chance") is not None:
                weather_str += f", Precipitation chance: {weather_display['precipitation_chance']}%"
            if weather_display.get("has_rain"):
                weather_str += " ⚠️ Rain expected"
            if weather_display.get("has_snow"):
                weather_str += " ⚠️ Snow expected"

            # Human-readable stops
            stops_val = best_flight.stops
            if str(stops_val) in ("N/A", "n/a", "", "unknown"):
                stops_display = "unknown (no data available)"
            elif isinstance(stops_val, int):
                stops_display = "non-stop" if stops_val == 0 else f"{stops_val} stop(s)"
            else:
                stops_display = str(stops_val)

            # Carbon display
            carbon_display = "N/A"
            if best_flight.carbon_emissions_g is not None:
                carbon_kg = round(best_flight.carbon_emissions_g / 1000, 1)
                carbon_display = f"{carbon_kg} kg CO₂"

            # CRITICAL CONSTRAINT for unknown stops
            constraint_note = ""
            if "unknown" in stops_display:
                constraint_note = "You MUST NOT state this is a non-stop or direct flight. If stops are unknown, say 'stop availability unknown'."

            # --- Layover constraint for non-stop flights when user has a layover limit ---
            layover_constraint = ""
            if intent_dict.get("layover_limit_minutes") and best_flight.stops == 0:
                layover_constraint = (
                    f"\nLAYOVER NOTE: This is a non-stop flight — it has NO layover at all, which "
                    f"automatically satisfies the user's max layover requirement of {intent_dict['layover_limit_minutes']} minutes. "
                    "CRITICAL PHRASING RULES — apply to every sentence including the opening and summary:\n"
                    "  1. NEVER write 'with a layover less than X hours' or 'layover of less than X' "
                    "anywhere in your response — those phrases imply the flight HAS a layover.\n"
                    "  2. In your opening sentence, do NOT echo the user's constraint with 'layover of less than'. "
                    "Instead write: 'Based on your max-layover preference, the best option is...'\n"
                    f"  3. In your summary sentence, write something like: "
                    f"'{best_flight.airline} {best_flight.flight_no} is the best non-stop option — it has no layover "
                    "whatsoever, which exceeds your requirement.' Do not echo the user's constraint phrase."
                )

            # --- Temperature constraint for correct ordering ---
            temp_constraint = ""
            if weather_display.get("temp_min_c") is not None and weather_display.get("temp_max_c") is not None:
                temp_constraint = (
                    f"\nTEMP RULE: Daily low is {weather_display['temp_min_c']}°C, daily high is "
                    f"{weather_display['temp_max_c']}°C. Never swap these — do not describe "
                    f"{weather_display['temp_max_c']}°C as the 'low' temperature."
                )

            # Construct system prompt with airline rule only if preferences exist
            airline_rule = ""
            if intent_dict.get("preferred_airlines"):
                preferred_lower = [a.lower() for a in intent_dict.get("preferred_airlines", [])]
                best_airline_lower = best_flight.airline.lower()
                airline_matched = any(p in best_airline_lower or best_airline_lower in p for p in preferred_lower)
                if not airline_matched:
                    airline_rule = (
                        "AIRLINE RULE: The user's preferred airline is not available. "
                        "Your FIRST sentence MUST disclose this — something like: "
                        "'No flights were found for [airline]; here is the closest available alternative.' "
                        "Do NOT open with flight details and then mention the airline mismatch later. "
                        "Do NOT invent or fabricate alternative flights — only present flights from the data above."
                    )

            # Facts block for streaming branch
            facts_block = (
                f"Origin: {intent_dict.get('origin_iata') or 'unknown'}\n"
                f"Destination: {intent_dict.get('destination_iata') or 'unknown'}\n"
                f"Departure date: {intent_dict.get('date') or 'not specified'}\n"
            )
            if intent_dict.get("return_date"):
                facts_block += f"Return date: {intent_dict['return_date']}\n"

            system = (
                "You are a professional travel planning assistant. "
                "CRITICAL: NEVER invent, fabricate, or suggest flight numbers, airline codes, prices, "
                "departure times, or any flight details that are not explicitly present in the flight "
                "data provided to you. If no matching airline is found, present only the available flight "
                "with a note that it differs from the preference — do NOT create fictional alternatives. "
                "RULE: If a flight data field is 'unknown (no data available)', you are PROHIBITED from "
                "stating or implying its value. "
                "IATA RULE: Whenever you mention a city's weather, always include its IATA code in "
                "parentheses, e.g. 'Weather for Bangalore (BLR)' or 'Mumbai (BOM)'. "
                "CITY NAME RULE: When writing about the flight destination, use ONLY the correct city name "
                "for the destination IATA code. Examples of correct mappings: MAA = Chennai (NOT Mumbai), "
                "BLR = Bangalore, BOM = Mumbai, DEL = Delhi. Never call MAA 'Mumbai' or BOM 'Chennai'. "
                + airline_rule
            )

            prompt = f"""
CRITICAL CONSTRAINT: The stops field for this flight is '{stops_display}'.
{constraint_note}{layover_constraint}{temp_constraint}

You are a helpful travel assistant helping a user plan {trip_description}.

User preferences:
- {filters_applied}
{warnings_str}

Flight options from {intent_dict.get('origin_iata')} to {intent_dict.get('destination_iata')} around {intent_dict.get('date')}:
{flights_str}

Best matching flight:
- {best_flight.airline} {best_flight.flight_no} on {best_flight.date or 'N/A'} |
  {best_flight.departure_time} → {best_flight.arrival_time} |
  Duration: {best_flight.duration_min} minutes |
  Price: {best_flight.price_inr} |
  Stops: {stops_display}{f" ({best_flight.layover_info})" if best_flight.layover_info else ""} |
  Baggage: {best_flight.baggage} |
  Carbon emissions: {carbon_display}
{f"{chr(10)}{price_insights_str}" if price_insights_str else ""}
{f"{chr(10)}{price_analysis_str}" if price_analysis_str else ""}
{f"{chr(10)}{price_prediction_str}" if price_prediction_str else ""}
Weather FORECAST for {intent_dict.get('destination_iata')} on {forecast_label}{approx_note}:
{weather_str}

IMPORTANT: Only reference the exact flights listed above. Do not create or suggest any other flights, codes, or prices.
User's question: {user_query}

Please recommend the best flight, explain why it matches their preferences, mention the weather forecast suitability (including packing advice based on min/max temperature and any rain or snow alerts), and answer the user's query helpfully.
"""

            full_prompt = facts_block + "\nPlease include the above origin and destination clearly at the start of your summary.\n\n" + prompt

            # ADD DEBUG LOGGING FOR STREAMING PROMPT
            logger.debug(
                "Streaming LLM prompt (first 500 chars): %s",
                full_prompt[:500].replace('\n', ' ')
            )

            # 4. Call LLM in streaming mode with handshake timeout + metrics
            llm_start = time.monotonic()
            try:
                token_stream = await asyncio.wait_for(
                    generate(
                        prompt=full_prompt,
                        system=system,
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
                async with asyncio.timeout(PLANNER_LLM_TIMEOUT):
                    # Handle async stream, sync iterable, or single string gracefully
                    if hasattr(token_stream, "__aiter__"):
                        async for token in token_stream:
                            if not isinstance(token, str):
                                token = str(token)
                            if not got_first_token:
                                first_token_time = time.monotonic() - llm_start
                                try:
                                    metrics.LLM_LATENCY.labels(provider=provider).observe(first_token_time)
                                except Exception:
                                    pass
                                got_first_token = True
                            yield token
                            full_response += token
                    elif hasattr(token_stream, "__iter__") and not isinstance(token_stream, (str, bytes)):
                        for token in token_stream:
                            if not isinstance(token, str):
                                token = str(token)
                            if not got_first_token:
                                first_token_time = time.monotonic() - llm_start
                                try:
                                    metrics.LLM_LATENCY.labels(provider=provider).observe(first_token_time)
                                except Exception:
                                    pass
                                got_first_token = True
                            yield token
                            full_response += token
                    else:
                        # single non-iterable result (string) — yield it as one chunk
                        token = token_stream
                        if not isinstance(token, str):
                            token = str(token)
                        if not got_first_token:
                            first_token_time = time.monotonic() - llm_start
                            try:
                                metrics.LLM_LATENCY.labels(provider=provider).observe(first_token_time)
                            except Exception:
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
                                "effective_intent": effective_intent_dict,
                                "filters_applied": filters_applied,
                                "ranked_count": debug_info.get("ranked_count", 0),
                                "flight_pref": intent_dict.get("flight_pref"),
                                "trip_type": intent_dict.get("trip_type"),
                                "use_cloud_llm": USE_CLOUD_FALLBACK,
                                "phases": phases,
                                "warnings": warnings
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

# ----------------------------------------------------------------------
# Helper: normalize airport for tests (simple deterministic fallback)
# ----------------------------------------------------------------------
def normalize_airport(text: Optional[str]) -> Optional[str]:
    """
    Normalize a free-text airport/city token into an IATA code or return None.
    Tries (in order):
      1. Direct resolver (city name, airport name, phrase) -> resolve_location(...)
      2. If token is a 3-letter alpha code, return uppercased token (best-effort)
      3. Otherwise None

    This is intentionally simple and deterministic for tests.
    """
    if not text:
        return None

    # First try the central resolver which already supports phrases/tokens/n-grams
    try:
        iata = resolve_location(text)
    except Exception:
        iata = None

    if iata:
        return iata

    # Fallback: if user passed a 3-letter token (e.g., "DEL" or "del"), accept it
    tok = text.strip()
    if len(tok) == 3 and tok.isalpha():
        return tok.upper()

    return None
