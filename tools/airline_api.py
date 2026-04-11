# NOTE:
# request_id is NOT manually injected into logger extra fields here.
# It is automatically added by the global JSON logging formatter
# via core.request_context.get_request_id().
# This keeps tool code clean while preserving full request correlation.
import os
import time
import asyncio
import logging
import re
import contextlib
import hashlib
import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from dateutil import parser  # for flexible datetime parsing
from cachetools import TTLCache  # bounded cache with TTL

# Import shared HTTP client, circuit breaker, metrics, and API key manager
from core.http_client import get_client
from core.circuit_breaker import get_circuit_breaker
from core.metrics import TOOL_REQUESTS, TOOL_LATENCY, AIRLINE_RETRIES, AIRLINE_ATTEMPTS
import core.metrics as app_metrics
from core.config import TESTING
from core.api_key_manager import key_manager as api_key_manager
from core.env_config import get_env_bool, get_env_float, get_env_int
from core.iata_resolver import city_for_iata, resolve_location

load_dotenv()

_GENERIC_LOCATION_TOKENS = {
    "city",
    "airport",
    "airports",
    "international",
    "terminal",
}


def _key_fingerprint(key: str) -> str:
    if not key:
        return "none"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def _tokens_for_match(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z]{3,}", (text or "").lower())
        if t not in _GENERIC_LOCATION_TOKENS
    }


def _resolution_matches_input(raw_text: str, resolved_iata: str) -> bool:
    """
    Guard against overly aggressive fuzzy matches for multi-word free-text input.
    For phrases with 2+ significant tokens, require token overlap with the
    resolved city's canonical name.
    """
    raw_tokens = _tokens_for_match(raw_text)
    if len(raw_tokens) <= 1:
        return True

    resolved_city = city_for_iata(resolved_iata) or ""
    city_tokens = _tokens_for_match(resolved_city)
    if not city_tokens:
        return False
    return bool(raw_tokens & city_tokens)


def expand_airports(city: str) -> str:
    """
    Normalize a user-provided location token for SerpAPI.
    Resolution order:
      1) If it's a comma-separated list, normalize each token recursively.
      2) Resolve via central iata_resolver.
      3) Fallback to uppercase raw token when it already looks like an explicit code.
      4) Return empty string for unresolved multi-word text (caller should fail clearly).
    """
    raw = (city or "").strip()
    if not raw:
        return ""

    if "," in raw:
        normalized_parts: List[str] = []
        seen: set[str] = set()
        for part in (p.strip() for p in raw.split(",")):
            normalized = expand_airports(part)
            if not normalized:
                continue
            for token in (t.strip() for t in normalized.split(",")):
                if token and token not in seen:
                    seen.add(token)
                    normalized_parts.append(token)
        return ",".join(normalized_parts)

    resolved = resolve_location(raw)
    if resolved and _resolution_matches_input(raw, resolved):
        return resolved

    upper = raw.upper()
    if len(upper) == 3 and upper.isalpha():
        return upper

    # Unresolved long/free-form location should fail fast in caller.
    return ""

# ----------------------------------------------------------------------
# Structured logging (configuration left to application)
# ----------------------------------------------------------------------
logger = logging.getLogger("airline_api")
_TESTING_LOGGED = False
SERPAPI_HTTP_TIMEOUT = get_env_float("SERPAPI_HTTP_TIMEOUT", 10.0)
SERPAPI_MAX_RETRIES = max(1, get_env_int("SERPAPI_MAX_RETRIES", 3))
SERPAPI_RETRY_BASE_DELAY = max(0.1, get_env_float("SERPAPI_RETRY_BASE_DELAY", 1.0))
SERPAPI_TOTAL_ATTEMPT_BUDGET = max(
    SERPAPI_MAX_RETRIES,
    get_env_int("SERPAPI_TOTAL_ATTEMPT_BUDGET", SERPAPI_MAX_RETRIES * 2),
)


def _health_check_non_destructive_mode() -> bool:
    """
    Non-destructive health mode for validation runs.
    When enabled, health checks must not mutate key exhaustion/quarantine state.
    """
    return bool(
        get_env_bool("HEALTHCHECK_NON_DESTRUCTIVE", default=False)
        or get_env_bool("VALIDATION_NON_DESTRUCTIVE_HEALTH", default=False)
    )


def _redact_request_params(params: dict) -> dict:
    """Return a copy safe for logs by masking credential-like fields."""
    redacted = {}
    for key, value in (params or {}).items():
        key_l = str(key).lower()
        if key_l in {"api_key", "authorization", "x-api-key", "appid"}:
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted

# ----------------------------------------------------------------------
# Custom exceptions
# ----------------------------------------------------------------------
class AirlineAPIError(Exception):
    """Raised when the airline search API fails after retries."""
    pass


def _extract_route_from_booking_token(token: str) -> Tuple[str, str, str, Optional[str], Optional[str]]:
    """
    Best-effort decode of SerpAPI/Google Flights booking token payload.
    Returns (departure_iata, arrival_iata, date_yyyy_mm_dd, airline_code, flight_number).

    The payload structure is not guaranteed, so this function is intentionally tolerant.
    Raises AirlineAPIError if decoding or extraction fails.
    """
    if not token or not isinstance(token, str):
        raise AirlineAPIError("Invalid booking token")

    # SerpAPI tokens are typically URL-safe base64 without padding.
    padded = token + ("=" * (-len(token) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="ignore")
    except Exception as e:
        raise AirlineAPIError("Could not decode booking token") from e

    try:
        payload = json.loads(decoded)
    except Exception as e:
        raise AirlineAPIError("Booking token payload is not valid JSON") from e

    def _is_iata(v: object) -> bool:
        return isinstance(v, str) and len(v) == 3 and v.isalpha()

    def _is_date(v: object) -> bool:
        if not isinstance(v, str):
            return False
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v))

    # Search recursively for a list pattern like [DEP, DATE, ARR, ..., AIRLINE, FLIGHT_NO]
    queue = [payload]
    while queue:
        node = queue.pop(0)
        if isinstance(node, list):
            if len(node) >= 3 and _is_iata(node[0]) and _is_date(node[1]) and _is_iata(node[2]):
                departure = str(node[0]).upper()
                date = str(node[1])
                arrival = str(node[2]).upper()
                airline_code = str(node[4]).upper() if len(node) > 4 and isinstance(node[4], str) else None
                flight_no = str(node[5]).strip() if len(node) > 5 and node[5] is not None else None
                return departure, arrival, date, airline_code, flight_no
            for child in node:
                if isinstance(child, (list, dict)):
                    queue.append(child)
        elif isinstance(node, dict):
            for child in node.values():
                if isinstance(child, (list, dict)):
                    queue.append(child)

    raise AirlineAPIError("Could not extract route details from booking token")


def _is_http_booking_link(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def _coerce_price_candidate(raw_price: Any) -> Optional[int]:
    """
    Best-effort numeric INR extraction from provider price fields.
    Accepts ints/floats, currency strings, and common nested dict price containers.
    """
    candidates: List[Any] = [raw_price]
    if isinstance(raw_price, dict):
        for key in (
            "price",
            "amount",
            "value",
            "raw",
            "display_price",
            "displayed_price",
            "extracted_price",
            "total",
            "total_price",
            "final_price",
        ):
            if key in raw_price:
                candidates.append(raw_price.get(key))

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, (int, float)):
            try:
                value = int(float(candidate))
                if value > 0:
                    return value
            except Exception:
                continue
        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                continue
            # Preserve only numeric segments from common currency representations.
            match = re.search(r"(\d[\d,]{1,})", text)
            if match:
                with contextlib.suppress(Exception):
                    value = int(match.group(1).replace(",", ""))
                    if value > 0:
                        return value
            continue
    return None


def _normalize_booking_request_candidate(candidate: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None

    raw_url = (
        candidate.get("url")
        or candidate.get("endpoint")
        or candidate.get("booking_url")
    )
    url = _is_http_booking_link(raw_url)
    if not url:
        return None

    post_data = candidate.get("post_data")
    method = str(candidate.get("method") or "").strip().upper()
    if not method:
        method = "POST" if post_data not in (None, "", {}, []) else "GET"
    if method not in {"GET", "POST"}:
        method = "GET"

    safe_headers: Dict[str, str] = {}
    headers = candidate.get("headers")
    if isinstance(headers, dict):
        for key, value in headers.items():
            if not isinstance(key, str) or value is None:
                continue
            key_l = key.lower().strip()
            if key_l in {"authorization", "cookie"}:
                continue
            safe_headers[key] = str(value)

    return {
        "url": url,
        "method": method,
        "post_data": post_data,
        "headers": safe_headers,
    }


def _iter_nested_nodes(root: Any, *, max_depth: int = 4, max_list_items: int = 12):
    queue: List[Tuple[Any, int]] = [(root, 0)]
    while queue:
        node, depth = queue.pop(0)
        yield node
        if depth >= max_depth:
            continue
        if isinstance(node, dict):
            for value in node.values():
                if isinstance(value, (dict, list)):
                    queue.append((value, depth + 1))
        elif isinstance(node, list):
            for item in node[:max_list_items]:
                if isinstance(item, (dict, list)):
                    queue.append((item, depth + 1))


def _extract_nested_booking_artifacts(raw_itinerary: Dict[str, Any]) -> Dict[str, Any]:
    booking_token: Optional[str] = None
    shareable_link: Optional[str] = None
    provider_link: Optional[str] = None
    partner_booking_link: Optional[str] = None
    booking_url: Optional[str] = None
    booking_request: Optional[Dict[str, Any]] = None
    booking_options: List[Dict[str, Any]] = []

    for node in _iter_nested_nodes(raw_itinerary):
        if not isinstance(node, dict):
            continue

        if not booking_token:
            token_candidate = node.get("booking_token") or node.get("departure_token")
            if isinstance(token_candidate, str) and token_candidate.strip():
                booking_token = token_candidate.strip()

        if not shareable_link:
            shareable_link = _is_http_booking_link(node.get("shareable_link"))

        if not provider_link:
            for key in (
                "provider_link",
                "partner_booking_link",
                "booking_url",
                "booking_link",
                "deeplink",
                "redirect_link",
            ):
                provider_link = _is_http_booking_link(node.get(key))
                if provider_link:
                    break

        if not partner_booking_link:
            partner_booking_link = _is_http_booking_link(node.get("partner_booking_link"))

        if not booking_url:
            booking_url = _is_http_booking_link(node.get("booking_url") or node.get("booking_link"))

        if booking_request is None:
            direct_candidate = node.get("booking_request")
            normalized_request = _normalize_booking_request_candidate(direct_candidate)
            if not normalized_request and (
                ("post_data" in node or "method" in node)
                and ("url" in node or "endpoint" in node or "booking_url" in node)
            ):
                normalized_request = _normalize_booking_request_candidate(node)
            if normalized_request:
                booking_request = normalized_request

        if not booking_options:
            options_candidate = node.get("booking_options")
            if isinstance(options_candidate, list):
                normalized_options = [dict(item) for item in options_candidate if isinstance(item, dict)]
                if normalized_options:
                    booking_options = normalized_options

    return {
        "booking_token": booking_token,
        "shareable_link": shareable_link,
        "provider_link": provider_link,
        "partner_booking_link": partner_booking_link,
        "booking_url": booking_url,
        "booking_request": booking_request,
        "booking_options": booking_options,
    }

# ----------------------------------------------------------------------
# Domain model
# ----------------------------------------------------------------------
@dataclass
class Flight:
    airline: str
    flight_no: str
    departure_time: str          # True first-leg departure (HH:MM)
    arrival_time: str            # True last-leg arrival   (HH:MM)
    duration_min: int            # Total trip duration from root total_duration field
    price_inr: Union[int, str]
    price_unavailable: bool = False
    stops: int = 0               # Number of stops = len(flights) - 1
    layover_info: str = ""       # Human-readable layover summary, e.g. "1h 30m at BOM"
    layover_airports: List[str] = field(default_factory=list)  # IATA codes of layover airports
    layover_durations_min: List[int] = field(default_factory=list)  # True layover durations in minutes
    baggage: str = "Check airline"  # Extracted from SerpAPI extensions/amenities
    booking_token: Optional[str] = None   # SerpAPI booking_token for handoff
    shareable_link: Optional[str] = None  # SerpAPI shareable Google Flights link (display/diagnostic only)
    provider_link: Optional[str] = None
    partner_booking_link: Optional[str] = None
    booking_url: Optional[str] = None
    booking_request: Optional[Dict[str, Any]] = None
    booking_options: Optional[List[dict]] = field(default_factory=list)
    carbon_emissions_g: Optional[int] = None  # CO2 in grams from carbon_emissions.this_flight
    legs: List[dict] = field(default_factory=list)  # Raw leg data from SerpAPI (each leg contains airline, flight_number, departure_airport, arrival_airport, etc.)
    airline_logo: Optional[str] = None
    itinerary_type: Optional[str] = None
    travel_class: Optional[str] = None
    legroom: Optional[str] = None
    marketed_as: List[str] = field(default_factory=list)
    extensions: List[str] = field(default_factory=list)
    separate_tickets: Optional[bool] = None
    local_prices: Optional[Any] = None
    baggage_prices: Optional[Any] = None
    booking_sellers: List[str] = field(default_factory=list)

# ----------------------------------------------------------------------
# In‑memory bounded cache to reduce duplicate calls
# ----------------------------------------------------------------------
_flight_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
FLIGHT_CACHE_SCHEMA_VERSION = 2


def _build_flight_cache_key(
    departure_ids: str,
    arrival_ids: str,
    date: str,
    return_date: Optional[str],
    serpapi_type: str,
    eco_mode: bool,
    min_layover: Optional[int],
    max_layover: Optional[int],
    deep_search: bool,
) -> Tuple[Any, ...]:
    return (
        FLIGHT_CACHE_SCHEMA_VERSION,
        departure_ids,
        arrival_ids,
        date,
        return_date,
        serpapi_type,
        eco_mode,
        min_layover,
        max_layover,
        deep_search,
    )

# ----------------------------------------------------------------------
# Per‑key rate limiting (async semaphores with delayed release)
# ----------------------------------------------------------------------
RATE_LIMIT_SECONDS = 1.0                # at most 1 call per second per key
_key_semaphores = {}                     # key_idx -> asyncio.Semaphore(1)
_key_semaphores_lock = asyncio.Lock()    # protects _key_semaphores

async def _rate_limit(key_idx: int):
    """
    Acquire the semaphore for the given key index, then spawn a background
    task to release it after RATE_LIMIT_SECONDS. This ensures that requests
    using the same key are spaced at least 1 second apart, while requests
    using different keys can run concurrently.
    """
    async with _key_semaphores_lock:
        if key_idx not in _key_semaphores:
            _key_semaphores[key_idx] = asyncio.Semaphore(1)
        sem = _key_semaphores[key_idx]

    await sem.acquire()

    async def _release_after(delay):
        await asyncio.sleep(delay)
        sem.release()

    asyncio.create_task(_release_after(RATE_LIMIT_SECONDS))

# ----------------------------------------------------------------------
# Duration parser (handles both int and string formats)
# ----------------------------------------------------------------------
def _parse_duration(duration) -> Optional[int]:
    """Convert duration from SerpAPI to minutes. Returns None if parsing fails."""
    if isinstance(duration, int):
        return duration
    if isinstance(duration, str):
        duration = duration.strip()
        
        # 1. Try colon format like "2:15" FIRST
        if ":" in duration:
            parts = duration.split(":")
            if len(parts) == 2:
                try:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    return hours * 60 + minutes
                except ValueError:
                    pass

        # 2. Try to parse formats like "2h 15m", "2 h 15 min", etc.
        pattern = r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m(?:in)?)?"
        match = re.match(pattern, duration, re.IGNORECASE)
        
        # Make sure the regex ACTUALLY found a number before returning
        if match and (match.group(1) or match.group(2)):
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2)) if match.group(2) else 0
            return hours * 60 + minutes
            
    return None

# ----------------------------------------------------------------------
# Helper: estimate reset time from SerpApi account info
# ----------------------------------------------------------------------
def _estimate_reset_from_account(account_data: dict) -> Optional[datetime]:
    """
    Try to determine when the quota will reset based on account info.
    Returns a datetime object or None if unknown.
    """
    # If there's an explicit reset field (not documented, but future‑proof)
    if "next_reset" in account_data:
        try:
            return datetime.fromisoformat(account_data["next_reset"])
        except (ValueError, TypeError):
            pass

    # Heuristic: monthly plans reset on 1st of next month
    plan_name = account_data.get("plan_name", "").lower()
    if "month" in plan_name:
        now = datetime.now()
        # first day of next month
        if now.month == 12:
            reset = datetime(now.year + 1, 1, 1)
        else:
            reset = datetime(now.year, now.month + 1, 1)
        return reset

    # Default fallback: 24 hours
    return datetime.now() + timedelta(days=1)

# ----------------------------------------------------------------------
# Track last account check per key to avoid spamming
# ----------------------------------------------------------------------
_last_account_check: dict = {}  # key index -> monotonic timestamp
SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_ENABLED = get_env_bool(
    "SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_ENABLED",
    default=False,
)
SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_INTERVAL_SECONDS = max(
    300,
    get_env_int("SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_INTERVAL_SECONDS", 1800),
)

async def _maybe_check_account(key: str, key_idx: int) -> None:
    """
    Periodically check the SerpApi account endpoint to update quota status.
    If quota exhausted, mark the key exhausted with appropriate reset time.
    """
    if not SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_ENABLED:
        return

    now = time.monotonic()
    last = _last_account_check.get(key_idx, 0)
    # Check on a long throttle to avoid adding pressure to hot successful search paths.
    if now - last < SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_INTERVAL_SECONDS:
        return

    _last_account_check[key_idx] = now
    try:
        client = get_client()
        resp = await client.get(
            "https://serpapi.com/account.json",
            params={"api_key": key}
        )
        if resp.status_code != 200:
            logger.debug(
                "Post-success account check skipped (non-blocking)",
                extra={"status": resp.status_code, "key_idx": key_idx},
            )
            return
        data = resp.json()
        searches_left = data.get("plan_searches_left")
        if searches_left is None:
            return

        if searches_left <= 0:
            reset_at = _estimate_reset_from_account(data)
            reset_timestamp = reset_at.timestamp() if reset_at else None
            details = f"plan_searches_left=0, account_data={str(data)[:200]}"
            await api_key_manager.mark_exhausted(
                "serpapi",
                key_idx,
                until=reset_timestamp,
                reason=f"quota | {details}"
            )
            # Listener will handle cache eviction – no direct call needed
            logger.warning("Key exhausted via account check", extra={
                "key_idx": key_idx,
                "reset_at": reset_at.isoformat() if reset_at else None
            })
    except Exception as e:
        logger.debug(
            "Post-success account check exception (non-blocking)",
            extra={"error": str(e), "key_idx": key_idx},
        )


def _schedule_post_success_account_check(key: str, key_idx: int) -> None:
    """
    Run post-success account checks off the hot request path.
    """
    async def _runner() -> None:
        try:
            await _maybe_check_account(key, key_idx)
        except Exception:
            # _maybe_check_account already swallows expected failures, but keep
            # a final guard here so scheduling never interrupts the success path.
            logger.debug("post_success_account_check_runner_failed", exc_info=True)

    try:
        asyncio.create_task(_runner())
    except Exception:
        # Extremely defensive: if task scheduling fails, keep request success intact.
        logger.debug("post_success_account_check_schedule_failed", exc_info=True)

# ----------------------------------------------------------------------
# Main search function
# ----------------------------------------------------------------------
async def search_flights(
    departure: str,
    arrival: str,
    date: str,
    max_results: int = 5,
    return_date: Optional[str] = None,
    layover_city: Optional[str] = None,
    # New optional parameters for enhanced control
    eco_mode: bool = False,
    min_layover: Optional[int] = None,
    max_layover: Optional[int] = None,
    deep_search: bool = False,
    use_cache: bool = True,
    restrict_fields: bool = True,  # if True, adds json_restrictor to limit payload
) -> Tuple[List[Flight], Optional[dict]]:
    """
    Asynchronously fetches flights from Google Flights via SerpApi.

    Uses multiple API keys with automatic rotation on quota exhaustion or rate limiting.
    Each key is reserved exclusively during the request.

    Args:
        departure (str): IATA code or city name (e.g., 'DEL' or 'delhi')
        arrival (str): IATA code or city name (e.g., 'BOM' or 'mumbai')
        date (str): Outbound date in YYYY-MM-DD format
        max_results (int): Limit number of results returned
        return_date (str, optional): Return date for round trips
        layover_city (str, optional): City name or IATA code to filter flights that stop at that city
        eco_mode (bool): If True, request flights with lower carbon emissions
        min_layover (int, optional): Minimum layover duration in minutes
        max_layover (int, optional): Maximum layover duration in minutes
        deep_search (bool): If True, enable SerpAPI deep search (more results, slower) and include price_insights
        use_cache (bool): If True, use in‑memory cache to avoid duplicate calls
        restrict_fields (bool): If True, set json_restrictor to limit response to essential fields

    Returns:
        Tuple[List[Flight], Optional[dict]]: Flights with rich details, plus route-level price insights if available.

    Raises:
        AirlineAPIError: If the request fails after retries, or if no API keys are available.
    """
    # --- TESTING bypass: return deterministic fake results and avoid HTTP calls ---
    _env_testing = os.getenv("TESTING", "").lower() in ("1", "true", "yes")
    global _TESTING_LOGGED
    if TESTING or _env_testing:
        if not _TESTING_LOGGED:
            logger.debug("TESTING mode enabled (env or config) — returning fake flight results")
            _TESTING_LOGGED = True
        return [
            Flight(
                airline="TestAir",
                flight_no="TA123",
                departure_time="06:00",
                arrival_time="08:00",
                duration_min=120,
                price_inr=1000,
                stops=0,
                layover_info="",
                layover_airports=[],
                layover_durations_min=[],
                baggage="1 free checked bag",
                booking_token=None,
                shareable_link=None,
                carbon_emissions_g=45000,
                legs=[],  # empty legs for fake flight
            )
        ], None

    # --- Expand city names to comma-separated IATA codes ---
    departure_ids = expand_airports(departure)
    arrival_ids = expand_airports(arrival)
    if not departure_ids:
        raise AirlineAPIError(f"Could not resolve departure location '{departure}' to a valid airport code.")
    if not arrival_ids:
        raise AirlineAPIError(f"Could not resolve arrival location '{arrival}' to a valid airport code.")

    # Start latency measurement
    start = time.monotonic()

    try:
        # Get the circuit breaker for this service
        breaker = await get_circuit_breaker("airline_api")

        # Determine trip type from return_date presence
        serpapi_type = "1" if return_date else "2"  # 1 = round trip, 2 = one way

        # Base parameters (without API key)
        base_params = {
            "engine": "google_flights",
            "departure_id": departure_ids,
            "arrival_id": arrival_ids,
            "outbound_date": date,
            "type": serpapi_type,
            "travel_class": "1",  # Economy
            "adults": "1",
            "hl": "en",
            "gl": "in",
            "currency": "INR",
        }

        # Add optional parameters
        if return_date:
            base_params["return_date"] = return_date

        if eco_mode:
            base_params["emissions"] = "1"

        if min_layover is not None and max_layover is not None:
            base_params["layover_duration"] = f"{min_layover},{max_layover}"

        if deep_search:
            base_params["deep_search"] = "true"

        # Cache control: no_cache = false means use SerpAPI's cache
        base_params["no_cache"] = "false" if use_cache else "true"

        # Field restriction to reduce payload
        if restrict_fields:
            fields = ["best_flights", "other_flights"]
            if deep_search:
                # Include price_insights only when deep search is explicitly requested
                fields.append("price_insights")
            base_params["json_restrictor"] = ",".join(fields)

        url = "https://serpapi.com/search"
        MAX_RETRIES = SERPAPI_MAX_RETRIES

        # Build cache key from all parameters that affect the result
        cache_key = _build_flight_cache_key(
            departure_ids=departure_ids,
            arrival_ids=arrival_ids,
            date=date,
            return_date=return_date,
            serpapi_type=serpapi_type,
            eco_mode=eco_mode,
            min_layover=min_layover,
            max_layover=max_layover,
            deep_search=deep_search,
        )

        # Check cache
        if use_cache and cache_key in _flight_cache:
            logger.debug("Returning cached flight results", extra={"cache_key": cache_key})
            cached = _flight_cache[cache_key]
            # cached is a tuple (flights, price_insights)
            # Ensure we don't return more than max_results
            return cached[0][:max_results], cached[1]

        # Structured log: request start
        logger.debug("SerpAPI request started", extra={
            "departure": departure,
            "arrival": arrival,
            "date": date,
            "return_date": return_date,
            "layover_city": layover_city,
            "eco_mode": eco_mode,
            "deep_search": deep_search,
        })

        # ------------------------------------------------------------------
        # Inner function that handles key rotation and per‑key retries.
        # Returns a tuple (parsed_results, attempts_used) on success.
        # ------------------------------------------------------------------
        async def _request_with_key_rotation() -> Tuple[List[Flight], int, Optional[dict]]:
            """Attempt request with key rotation and per‑key retries."""
            key_attempts = 0
            max_total_attempts = max(1, SERPAPI_TOTAL_ATTEMPT_BUDGET)
            max_key_attempts = min(10, max_total_attempts)  # safety bound
            total_attempts = 0

            while key_attempts < max_key_attempts and total_attempts < max_total_attempts:
                try:
                    # Reserve a key for the duration of this attempt.
                    # reserve_key returns (index, key) as per design.
                    async with api_key_manager.reserve_key("serpapi") as (idx, key):
                        key_fp = _key_fingerprint(key)
                        # For this key, try up to MAX_RETRIES times (network/5xx retries)
                        for attempt in range(MAX_RETRIES):
                            if total_attempts >= max_total_attempts:
                                app_metrics.record_retry_budget_exhausted("airline_search_flights")
                                raise AirlineAPIError(
                                    f"SerpAPI retry budget exhausted ({max_total_attempts} attempts)"
                                )
                            attempt_start = time.monotonic()
                            try:
                                # Wait for the per‑key rate limiter
                                await _rate_limit(idx)

                                client = get_client()
                                # Inject the current key
                                params = dict(base_params)
                                params["api_key"] = key

                                # --- Log the outgoing request (with API key redacted) ---
                                logger.debug("SerpAPI HTTP request", extra={
                                    "params": {k: v for k, v in params.items() if k != "api_key"},
                                    "key_source": "api_key_manager.reserve_key:serpapi",
                                    "key_idx": idx,
                                    "key_fp": key_fp,
                                    "client_mode": "shared_get_client",
                                })

                                total_attempts += 1
                                response = await client.get(
                                    url,
                                    params=params,
                                    timeout=SERPAPI_HTTP_TIMEOUT,
                                )

                                # ----- Handle specific status codes BEFORE raise_for_status -----
                                status = response.status_code

                                # 1) Unauthorized/forbidden key – mark temporarily exhausted.
                                # Treat 403 the same as 401 to avoid repeatedly selecting a denied key.
                                if status in (401, 403):
                                    until = (datetime.now(timezone.utc) + timedelta(days=30)).timestamp()
                                    details = (response.text or "")[:1000]
                                    await api_key_manager.mark_exhausted(
                                        "serpapi",
                                        idx,
                                        until=until,
                                        reason=f"unauthorized_http_{status} | {details}"
                                    )
                                    logger.warning(
                                        "Key unauthorized, marked exhausted",
                                        extra={"key_idx": idx, "status_code": status},
                                    )
                                    break  # try next key

                                # 2) Rate limit or payment required – attempt to get reset time
                                if status in (429, 402):
                                    try:
                                        acct_resp = await client.get(
                                            "https://serpapi.com/account.json",
                                            params={"api_key": key}
                                        )
                                        if acct_resp.status_code == 200:
                                            acct_data = acct_resp.json()
                                            searches_left = acct_data.get("plan_searches_left")
                                            if searches_left == 0:
                                                reset_at = _estimate_reset_from_account(acct_data)
                                                reset_timestamp = reset_at.timestamp() if reset_at else None
                                                details = f"plan_searches_left=0, account_data={str(acct_data)[:200]}"
                                                await api_key_manager.mark_exhausted(
                                                    "serpapi",
                                                    idx,
                                                    until=reset_timestamp,
                                                    reason=f"quota | {details}"
                                                )
                                                logger.warning("Key quota exhausted", extra={
                                                    "key_idx": idx,
                                                    "reset_at": reset_at.isoformat() if reset_at else None
                                                })
                                                break
                                    except Exception as e:
                                        logger.debug(
                                            "Account check skipped during 429/402 handling",
                                            extra={"error": str(e)},
                                        )
                                    # Fallback: mark exhausted with default reset (24h)
                                    until = (datetime.now(timezone.utc) + timedelta(days=1)).timestamp()
                                    details = (response.text or "")[:1000]
                                    await api_key_manager.mark_exhausted(
                                        "serpapi",
                                        idx,
                                        until=until,
                                        reason=f"rate_limit | {details}"
                                    )
                                    break

                                # 3) Now safe to raise for other HTTP errors
                                response.raise_for_status()

                                # ----- Textual quota detection (HTML or plain) -----
                                text = response.text or ""
                                lower_text = text.lower()
                                quota_patterns = [
                                    r'exhaust(ed|ion)?',
                                    r'used.*search',
                                    r'no more searches',
                                    r'your searches (are )?exhausted',
                                    r'quota (exceeded|limit|reached)',
                                    r'out of searches',
                                ]
                                if any(re.search(p, lower_text) for p in quota_patterns):
                                    # Assume quota exhausted, mark for 24h
                                    until = (datetime.now(timezone.utc) + timedelta(days=1)).timestamp()
                                    details = text[:1000]
                                    await api_key_manager.mark_exhausted(
                                        "serpapi",
                                        idx,
                                        until=until,
                                        reason=f"quota_text | {details}"
                                    )
                                    break

                                # 4) Try to parse JSON
                                try:
                                    data = response.json()

                                    # --- Log the response after successful parsing ---
                                    best = data.get("best_flights", [])
                                    other = data.get("other_flights", [])
                                    logger.debug("SerpAPI HTTP response", extra={
                                        "status_code": status,
                                        "raw_keys": list(data.keys()),
                                        "flight_count": len(best) + len(other),
                                    })

                                    logger.debug(
                                        "AIRLINE API TRACE",
                                        extra={
                                            "request_url": url,
                                            "request_params": _redact_request_params(params),
                                            "response_keys": sorted(list(data.keys()))[:50] if isinstance(data, dict) else None,
                                        },
                                    )
                                except ValueError:
                                    # Non-JSON can be transient upstream HTML/gateway responses.
                                    logger.warning(
                                        "SerpAPI returned non-JSON payload",
                                        extra={
                                            "status_code": status,
                                            "attempt": attempt + 1,
                                            "key_idx": idx,
                                            "text_preview": text[:200],
                                        },
                                    )
                                    AIRLINE_RETRIES.labels(reason="non_json").inc()
                                    if attempt == MAX_RETRIES - 1:
                                        break
                                    sleep_time = SERPAPI_RETRY_BASE_DELAY * (2 ** attempt)
                                    logger.debug("Retry scheduled", extra={"sleep_sec": sleep_time, "reason": "non_json"})
                                    await asyncio.sleep(sleep_time)
                                    continue

                                # 5) Check for explicit error in JSON
                                if "error" in data:
                                    error_msg = data["error"]
                                    # Robust: convert to string before lowercasing
                                    error_lower = str(error_msg).lower()
                                    logger.warning("SerpAPI returned error payload", extra={"error": error_msg})
                                    if any(re.search(p, error_lower) for p in quota_patterns):
                                        until = (datetime.now(timezone.utc) + timedelta(days=1)).timestamp()
                                        details = error_msg[:1000]
                                        await api_key_manager.mark_exhausted(
                                            "serpapi",
                                            idx,
                                            until=until,
                                            reason=f"quota | {details}"
                                        )
                                        break
                                    else:
                                        # Other errors are fatal
                                        raise AirlineAPIError(f"SerpAPI error: {error_msg}")

                                # ----- Success path: parse flights -----
                                # SerpAPI returns best_flights (curated) and other_flights (all others).
                                # For weaker/complex paths (round-trip/deep-search/high-breadth),
                                # interleave buckets so early parsing is less biased toward one bucket.
                                # Deduplicate on flight_number + departure_time to prevent double entries.
                                best_bucket = data.get("best_flights") or []
                                other_bucket = data.get("other_flights") or []
                                interleave_complex_route = bool(return_date or deep_search or max_results > 10)
                                merge_mode = "best_then_other"
                                ordered_candidates: List[Dict[str, Any]] = []
                                if interleave_complex_route:
                                    merge_mode = "interleave_best_other"
                                    max_bucket_len = max(len(best_bucket), len(other_bucket))
                                    for i in range(max_bucket_len):
                                        if i < len(best_bucket):
                                            ordered_candidates.append(best_bucket[i])
                                        if i < len(other_bucket):
                                            ordered_candidates.append(other_bucket[i])
                                else:
                                    ordered_candidates.extend(best_bucket)
                                    ordered_candidates.extend(other_bucket)
                                _seen_keys: set = set()
                                flights_combined: list = []
                                for _item in ordered_candidates:
                                    _legs = _item.get("flights") or []
                                    _key = (
                                        _legs[0].get("flight_number", "") if _legs else "",
                                        _legs[0].get("departure_airport", {}).get("time", "") if _legs else "",
                                    )
                                    if _key not in _seen_keys:
                                        _seen_keys.add(_key)
                                        flights_combined.append(_item)
                                flights = flights_combined
                                parsed_results = []
                                raw_candidate_count = len(flights)
                                parse_window = max_results * (3 if return_date else 2)
                                missing_price_count = 0
                                non_numeric_price_count = 0
                                kept_unavailable_price_count = 0
                                for raw in flights[:parse_window]:  # fetch extra for possible layover filtering
                                    try:
                                        legs = raw.get("flights")
                                        if not legs or not isinstance(legs, list):
                                            logger.warning("Missing 'flights' key or empty list", extra={
                                                "departure": departure,
                                                "arrival": arrival,
                                                "date": date,
                                            })
                                            continue

                                        # ---- Primary airline / flight number (first leg) ----
                                        first_leg = legs[0]
                                        last_leg  = legs[-1]

                                        required = ["airline", "flight_number", "departure_airport", "arrival_airport"]
                                        if not all(k in first_leg for k in required):
                                            logger.warning("Flight missing required fields", extra={
                                                "missing": [k for k in required if k not in first_leg],
                                            })
                                            continue

                                        # ---- Price ----
                                        raw_price_candidates: List[Any] = [
                                            raw.get("price"),
                                            raw.get("price_raw"),
                                            raw.get("displayed_price"),
                                            raw.get("total_price"),
                                            raw.get("extracted_price"),
                                        ]
                                        booking_options_for_price = raw.get("booking_options")
                                        if isinstance(booking_options_for_price, list):
                                            for option in booking_options_for_price[:4]:
                                                if not isinstance(option, dict):
                                                    continue
                                                raw_price_candidates.extend(
                                                    [
                                                        option.get("price"),
                                                        option.get("total_price"),
                                                        option.get("extracted_price"),
                                                    ]
                                                )
                                                together = option.get("together")
                                                if isinstance(together, dict):
                                                    raw_price_candidates.extend(
                                                        [
                                                            together.get("price"),
                                                            together.get("total_price"),
                                                            together.get("displayed_price"),
                                                            together.get("extracted_price"),
                                                        ]
                                                    )
                                        price_int = None
                                        for candidate_price in raw_price_candidates:
                                            price_int = _coerce_price_candidate(candidate_price)
                                            if price_int is not None:
                                                break

                                        price_unavailable = False
                                        if price_int is None:
                                            if raw.get("price") is None:
                                                missing_price_count += 1
                                            else:
                                                non_numeric_price_count += 1
                                            price_unavailable = True
                                            price_value: Union[int, str] = "Price unavailable"
                                        else:
                                            price_value = int(price_int)

                                        if price_unavailable:
                                            kept_unavailable_price_count += 1

                                        # Keep flights with unavailable price in candidate set.
                                        # Ranking logic will treat these as high-cost/unknown instead of cheapest.
                                        try:
                                            if not price_unavailable:
                                                price_value = int(price_value)
                                        except (ValueError, TypeError):
                                            price_value = "Price unavailable"
                                            price_unavailable = True

                                        # ---- True total duration (root field, not first-leg only) ----
                                        raw_duration = raw.get("total_duration") or first_leg.get("duration")
                                        duration_min = _parse_duration(raw_duration)
                                        if duration_min is None:
                                            logger.warning("Could not parse duration", extra={
                                                "duration_raw": raw_duration,
                                            })
                                            continue

                                        # ---- True departure / arrival times across all legs ----
                                        departure_time = first_leg["departure_airport"].get("time", "00:00")
                                        arrival_time   = last_leg["arrival_airport"].get("time", "00:00")

                                        # ---- Stops and layover details ----
                                        stops = len(legs) - 1

                                        # --- BEGIN LAYOVER COMPUTATION FROM LEG TIMES (true durations) ---
                                        layover_airports = []
                                        layover_info = ""
                                        layover_durations_min = []  # will be stored in Flight
                                        # Parse timestamps to datetime for each leg
                                        leg_datetimes = []
                                        for leg in legs:
                                            dep_airport = leg.get("departure_airport", {})
                                            arr_airport = leg.get("arrival_airport", {})
                                            dep_time_str = dep_airport.get("time")
                                            arr_time_str = arr_airport.get("time")
                                            dep_dt = None
                                            arr_dt = None
                                            if dep_time_str:
                                                try:
                                                    dep_dt = parser.parse(dep_time_str)
                                                except Exception:
                                                    pass
                                            if arr_time_str:
                                                try:
                                                    arr_dt = parser.parse(arr_time_str)
                                                except Exception:
                                                    pass
                                            leg_datetimes.append((dep_dt, arr_dt))

                                        # Compute layovers from differences
                                        layover_segments = []
                                        for i in range(len(legs) - 1):
                                            this_arr = leg_datetimes[i][1]   # arrival of current leg
                                            next_dep = leg_datetimes[i+1][0] # departure of next leg
                                            if this_arr and next_dep:
                                                gap_min = int((next_dep - this_arr).total_seconds() / 60)
                                                if gap_min > 0:
                                                    hours, mins = divmod(gap_min, 60)
                                                    next_airport_id = legs[i+1]["departure_airport"].get("id", "?")
                                                    human = f"{hours}h {mins}m at {next_airport_id}" if hours else f"{mins}m at {next_airport_id}"
                                                    layover_segments.append(human)
                                                    layover_airports.append(next_airport_id)
                                                    layover_durations_min.append(gap_min)
                                        if layover_segments:
                                            layover_info = "; ".join(layover_segments)
                                        else:
                                            # Fallback to raw layovers if any
                                            layover_list = raw.get("layovers", [])
                                            if layover_list:
                                                layover_parts = []
                                                for lv in layover_list:
                                                    lv_dur  = lv.get("duration")
                                                    lv_name = lv.get("name") or lv.get("id") or "unknown airport"
                                                    lv_id   = lv.get("id")
                                                    if lv_id:
                                                        layover_airports.append(lv_id)
                                                    if lv_dur is not None:
                                                        h, m = divmod(int(lv_dur), 60)
                                                        lv_str = f"{h}h {m}m at {lv_name}" if h else f"{m}m at {lv_name}"
                                                    else:
                                                        lv_str = f"layover at {lv_name}"
                                                    layover_parts.append(lv_str)
                                                layover_info = ", ".join(layover_parts)
                                            elif stops > 0:
                                                # Still no layover info, just list airports
                                                for leg in legs[:-1]:
                                                    arr_airport = leg.get("arrival_airport", {})
                                                    airport_id = arr_airport.get("id")
                                                    if airport_id:
                                                        layover_airports.append(airport_id)
                                                if layover_airports:
                                                    layover_info = "Layover at " + ", ".join(layover_airports)
                                        # --- END LAYOVER COMPUTATION ---

                                        # ---- Baggage (from extensions list, e.g. "1 free checked bag") ----
                                        extensions = raw.get("extensions", []) or []
                                        baggage_str = "Check airline"
                                        for ext in extensions:
                                            ext_lower = str(ext).lower()
                                            if "bag" in ext_lower or "luggage" in ext_lower or "carry" in ext_lower:
                                                baggage_str = str(ext).strip()
                                                break

                                        # ---- Booking artifacts for handoff recovery ----
                                        nested_artifacts = _extract_nested_booking_artifacts(raw)
                                        booking_token = (
                                            str(raw.get("booking_token") or nested_artifacts.get("booking_token") or "").strip()
                                            or None
                                        )
                                        shareable_link = _is_http_booking_link(
                                            raw.get("shareable_link")
                                        ) or nested_artifacts.get("shareable_link")
                                        provider_link = _is_http_booking_link(
                                            raw.get("provider_link")
                                        ) or nested_artifacts.get("provider_link")
                                        partner_booking_link = _is_http_booking_link(
                                            raw.get("partner_booking_link")
                                        ) or nested_artifacts.get("partner_booking_link")
                                        booking_url = _is_http_booking_link(
                                            raw.get("booking_url") or raw.get("url")
                                        ) or nested_artifacts.get("booking_url")

                                        booking_request = _normalize_booking_request_candidate(raw.get("booking_request"))
                                        if booking_request is None:
                                            booking_request = nested_artifacts.get("booking_request")

                                        booking_options = raw.get("booking_options")
                                        if isinstance(booking_options, list):
                                            booking_options = [dict(item) for item in booking_options if isinstance(item, dict)]
                                        else:
                                            booking_options = []
                                        if not booking_options:
                                            booking_options = list(nested_artifacts.get("booking_options") or [])

                                        # ---- Carbon emissions (grams) ----
                                        carbon_data = raw.get("carbon_emissions") or {}
                                        carbon_g: Optional[int] = None
                                        if carbon_data:
                                            raw_carbon = carbon_data.get("this_flight")
                                            if raw_carbon is not None:
                                                try:
                                                    carbon_g = int(raw_carbon)
                                                except (ValueError, TypeError):
                                                    pass

                                        airline_logo = _is_http_booking_link(
                                            first_leg.get("airline_logo") or raw.get("airline_logo")
                                        )
                                        itinerary_type = str(raw.get("type") or "").strip() or None
                                        travel_class = str(first_leg.get("travel_class") or "").strip() or None
                                        legroom = str(first_leg.get("legroom") or "").strip() or None
                                        extension_values = [
                                            str(ext).strip()
                                            for ext in (extensions or [])
                                            if str(ext or "").strip()
                                        ][:8]

                                        marketed_as: List[str] = []
                                        for key in ("flight_number", "operating_flight_number"):
                                            value = str(first_leg.get(key) or "").strip()
                                            if value and value not in marketed_as:
                                                marketed_as.append(value)

                                        separate_tickets = None
                                        local_prices: Optional[Any] = None
                                        baggage_prices: Optional[Any] = None
                                        booking_sellers: List[str] = []
                                        for option in booking_options[:8]:
                                            if not isinstance(option, dict):
                                                continue
                                            together = option.get("together") if isinstance(option.get("together"), dict) else {}
                                            seller = (
                                                together.get("book_with")
                                                or together.get("name")
                                                or option.get("book_with")
                                                or option.get("name")
                                                or option.get("provider")
                                            )
                                            seller_text = str(seller or "").strip()
                                            if seller_text and seller_text not in booking_sellers:
                                                booking_sellers.append(seller_text)
                                            if separate_tickets is None and option.get("separate_tickets") is not None:
                                                separate_tickets = bool(option.get("separate_tickets"))
                                            if local_prices is None:
                                                maybe_local = together.get("local_prices")
                                                if maybe_local not in (None, "", [], {}):
                                                    local_prices = maybe_local
                                            if baggage_prices is None:
                                                maybe_baggage = together.get("baggage_prices")
                                                if maybe_baggage not in (None, "", [], {}):
                                                    baggage_prices = maybe_baggage

                                        flight = Flight(
                                            airline=first_leg["airline"],
                                            flight_no=first_leg["flight_number"],
                                            departure_time=departure_time,
                                            arrival_time=arrival_time,
                                            duration_min=duration_min,
                                            price_inr=price_value,
                                            price_unavailable=price_unavailable,
                                            stops=stops,
                                            layover_info=layover_info,
                                            layover_airports=layover_airports,
                                            layover_durations_min=layover_durations_min,
                                            baggage=baggage_str,
                                            booking_token=booking_token,
                                            shareable_link=shareable_link,
                                            provider_link=provider_link,
                                            partner_booking_link=partner_booking_link,
                                            booking_url=booking_url,
                                            booking_request=booking_request,
                                            booking_options=booking_options,
                                            carbon_emissions_g=carbon_g,
                                            # Store raw leg data for downstream use
                                            legs=legs,
                                            airline_logo=airline_logo,
                                            itinerary_type=itinerary_type,
                                            travel_class=travel_class,
                                            legroom=legroom,
                                            marketed_as=marketed_as,
                                            extensions=extension_values,
                                            separate_tickets=separate_tickets,
                                            local_prices=local_prices,
                                            baggage_prices=baggage_prices,
                                            booking_sellers=booking_sellers,
                                        )
                                        parsed_results.append(flight)

                                    except Exception as e:
                                        logger.warning("Flight parsing skipped", extra={
                                            "error": str(e),
                                            "departure": departure,
                                            "arrival": arrival,
                                            "date": date,
                                        })
                                        continue

                                if missing_price_count or non_numeric_price_count:
                                    logger.warning(
                                        "Flights contained unusable pricing fields; retained with price_unavailable markers",
                                        extra={
                                            "raw_candidate_count": raw_candidate_count,
                                            "parsed_candidate_count": len(parsed_results),
                                            "merge_mode": merge_mode,
                                            "parse_window": parse_window,
                                            "missing_price_count": missing_price_count,
                                            "non_numeric_price_count": non_numeric_price_count,
                                            "kept_unavailable_price_count": kept_unavailable_price_count,
                                            "route": f"{departure}->{arrival}",
                                            "date": date,
                                        },
                                    )

                                # ---- Apply layover filtering if requested ----
                                if layover_city:
                                    layover_iata = resolve_location(layover_city)
                                    if layover_iata:
                                        parsed_results = [f for f in parsed_results if layover_iata in f.layover_airports]
                                        logger.debug("Applied layover filter", extra={
                                            "layover_city": layover_city,
                                            "layover_iata": layover_iata,
                                            "filtered_count": len(parsed_results)
                                        })
                                    else:
                                        logger.warning("Could not resolve layover city", extra={"city": layover_city})

                                # ---- price_insights (route-level, one per response) ----
                                raw_price_insights = data.get("price_insights") or None
                                price_insights_raw: Dict[str, Any] = {
                                    "price_insights": raw_price_insights,
                                    "_search_meta": {
                                        "raw_candidate_count": raw_candidate_count,
                                        "parsed_candidate_count": len(parsed_results),
                                        "merge_mode": merge_mode,
                                        "parse_window": parse_window,
                                        "missing_price_count": missing_price_count,
                                        "non_numeric_price_count": non_numeric_price_count,
                                        "kept_unavailable_price_count": kept_unavailable_price_count,
                                    },
                                }

                                latency = time.monotonic() - attempt_start
                                logger.debug("SerpAPI attempt succeeded", extra={
                                    "latency_sec": round(latency, 2),
                                    "attempt": attempt + 1,
                                    "key_source": "api_key_manager.reserve_key:serpapi",
                                    "key_idx": idx,
                                    "key_fp": key_fp,
                                    "client_mode": "shared_get_client",
                                })

                                # ---- After success, optionally check account usage ----
                                # Keep this out of the critical first-response path.
                                _schedule_post_success_account_check(key, idx)

                                # Record usage and return
                                await api_key_manager.record_usage("serpapi", idx)
                                return parsed_results, total_attempts, price_insights_raw

                            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                                latency = time.monotonic() - attempt_start
                                logger.warning("SerpAPI network error", extra={
                                    "error_type": type(e).__name__,
                                    "attempt": attempt + 1,
                                    "latency_sec": round(latency, 2),
                                    "key_source": "api_key_manager.reserve_key:serpapi",
                                    "key_idx": idx,
                                    "key_fp": key_fp,
                                    "client_mode": "shared_get_client",
                                })
                                AIRLINE_RETRIES.labels(reason="network").inc()
                                if attempt == MAX_RETRIES - 1:
                                    # Exhausted retries for this key → try next key
                                    break
                                sleep_time = SERPAPI_RETRY_BASE_DELAY * (2 ** attempt)
                                logger.debug("Retry scheduled", extra={"sleep_sec": sleep_time})
                                await asyncio.sleep(sleep_time)

                            except httpx.HTTPStatusError as e:
                                latency = time.monotonic() - attempt_start
                                status = e.response.status_code
                                logger.warning("SerpAPI HTTP error", extra={
                                    "status_code": status,
                                    "attempt": attempt + 1,
                                    "latency_sec": round(latency, 2),
                                    "key_source": "api_key_manager.reserve_key:serpapi",
                                    "key_idx": idx,
                                    "key_fp": key_fp,
                                    "client_mode": "shared_get_client",
                                })

                                # Already handled 401,429,402 above, but catch others
                                if status == 429:
                                    AIRLINE_RETRIES.labels(reason="http_429").inc()
                                    until = (datetime.now(timezone.utc) + timedelta(days=1)).timestamp()
                                    details = (e.response.text or "")[:1000]
                                    await api_key_manager.mark_exhausted(
                                        "serpapi",
                                        idx,
                                        until=until,
                                        reason=f"rate_limit | {details}"
                                    )
                                    break
                                elif 500 <= status < 600:
                                    AIRLINE_RETRIES.labels(reason="http_5xx").inc()
                                    if attempt == MAX_RETRIES - 1:
                                        break
                                    sleep_time = SERPAPI_RETRY_BASE_DELAY * (2 ** attempt)
                                    logger.debug("Server error, retry scheduled", extra={"sleep_sec": sleep_time})
                                    await asyncio.sleep(sleep_time)
                                else:
                                    # Client error (4xx except 401/429/402) – fatal
                                    raise AirlineAPIError(f"HTTP {status} for {departure}->{arrival} on {date}") from e

                            except Exception as e:
                                # Unexpected error – log and re-raise
                                logger.error("Unexpected error in airline API", extra={"error": str(e)})
                                raise

                        # If we exit the per‑key loop without success, continue to next key
                        key_attempts += 1
                        continue  # while loop will try next key

                except RuntimeError as e:
                    # No keys available from manager (all exhausted)
                    logger.error("No SerpAPI keys available")
                    raise AirlineAPIError("All SerpAPI keys exhausted or failed") from e

            # If we exit the while loop, all keys exhausted or failed
            if total_attempts >= max_total_attempts:
                app_metrics.record_retry_budget_exhausted("airline_search_flights")
                raise AirlineAPIError(f"SerpAPI retry budget exhausted ({max_total_attempts} attempts)")
            raise AirlineAPIError("All SerpAPI keys exhausted or failed")

        # Execute the whole operation under circuit breaker protection
        parsed_results, attempts_used, price_insights_raw = await breaker.call(_request_with_key_rotation)

        # Log final success with actual attempts count
        logger.debug("SerpAPI final success", extra={
            "results_count": len(parsed_results),
            "attempts": attempts_used,
        })

        # Record the number of attempts taken for this successful request
        AIRLINE_ATTEMPTS.observe(attempts_used)

        # Increment success counter
        TOOL_REQUESTS.labels(tool="airline", status="success").inc()

        # Store in cache
        if use_cache:
            _flight_cache[cache_key] = (parsed_results, price_insights_raw)

        # Return flights list + route-level price_insights dict (may be None)
        return parsed_results[:max_results], price_insights_raw

    except Exception:
        # Increment error counter for any exception (including AirlineAPIError)
        TOOL_REQUESTS.labels(tool="airline", status="error").inc()
        raise
    finally:
        # Record latency regardless of success/failure
        TOOL_LATENCY.labels(tool="airline").observe(time.monotonic() - start)


async def search_with_booking_token(token: str) -> List[Flight]:
    """
    Re-query SerpAPI for an itinerary referenced by booking token.

    This function intentionally reuses `search_flights` so all existing parsing,
    normalization, retries, caching behavior, and field extraction stay centralized.

    Args:
        token: SerpAPI/Google Flights booking token.

    Returns:
        List[Flight]: normalized flights compatible with `search_flights` output.

    Raises:
        AirlineAPIError: when token is invalid or no matching itinerary can be resolved.
    """
    departure, arrival, date, airline_code, flight_no = _extract_route_from_booking_token(token)

    # Re-query the route/date and filter to token-referenced carrier/flight when possible.
    flights, _ = await search_flights(
        departure=departure,
        arrival=arrival,
        date=date,
        max_results=20,
        use_cache=False,
        deep_search=True,
    )

    if not flights:
        raise AirlineAPIError("No flights found for booking token route")

    # Try to narrow to the same itinerary if token included airline/flight number.
    if airline_code or flight_no:
        def _norm(s: Optional[str]) -> str:
            return re.sub(r"[^A-Za-z0-9]", "", (s or "")).upper()

        wanted_airline = _norm(airline_code)
        wanted_fno = _norm(flight_no)
        matched = []
        for f in flights:
            no = _norm(f.flight_no)
            airline = _norm(f.airline)
            airline_ok = (not wanted_airline) or (wanted_airline in no) or (wanted_airline in airline)
            number_ok = (not wanted_fno) or (wanted_fno in no)
            if airline_ok and number_ok:
                matched.append(f)

        if matched:
            return matched

    # Graceful fallback: return route-level candidates if exact itinerary was not found.
    return flights

# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------
async def health_check() -> str:
    """
    Performs a minimal test request to verify the airline API is functioning.

    Uses a fixed route (DEL → BOM) for tomorrow's date.
    Return buckets:
      - ok: provider responded with valid data
      - degraded: transient/recoverable upstream issue
      - unavailable: no active SerpAPI key is currently usable
      - fail: hard auth/configuration failure
    """
    health_timeout = max(2.0, get_env_float("AIRLINE_HEALTH_TIMEOUT", 6.0))
    non_destructive = _health_check_non_destructive_mode()
    # For health checks we use the key manager as well, but without rotation
    try:
        async with api_key_manager.reserve_key("serpapi") as (idx, key):
            # Use tomorrow's date to avoid caching issues
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

            # Apply per‑key rate limiting (health check uses its reserved key)
            await _rate_limit(idx)

            client = get_client()
            params = {
                "engine": "google_flights",
                "departure_id": "DEL",
                "arrival_id": "BOM",
                "outbound_date": tomorrow,
                "type": "2",  # one way
                "travel_class": "1",
                "adults": "1",
                "hl": "en",
                "gl": "in",
                "currency": "INR",
                "deep_search": "true",
                "api_key": key,
            }
            response = await client.get(
                "https://serpapi.com/search",
                params=params,
                timeout=health_timeout,
            )

            # Handle 401 specially – log but do NOT auto-exhaust; let real requests handle it.
            if response.status_code == 401:
                logger.error("Health check failed: invalid API key (401); not auto-exhausting here")
                return "fail"

            if response.status_code in (402, 429):
                reset_timestamp = None
                try:
                    acct_resp = await client.get(
                        "https://serpapi.com/account.json",
                        params={"api_key": key},
                    )
                    if acct_resp.status_code == 200:
                        acct_data = acct_resp.json()
                        if acct_data.get("plan_searches_left") == 0:
                            reset_at = _estimate_reset_from_account(acct_data)
                            reset_timestamp = reset_at.timestamp() if reset_at else None
                except Exception as e:
                    logger.debug(
                        "Health account check skipped during quota handling",
                        extra={"error": str(e)},
                    )

                until = reset_timestamp or (datetime.now(timezone.utc) + timedelta(days=1)).timestamp()
                details = (response.text or "")[:400]
                if non_destructive:
                    logger.warning(
                        "Health check detected exhausted SerpAPI key but skipped quarantine (non-destructive mode)",
                        extra={"key_idx": idx, "status_code": response.status_code},
                    )
                else:
                    await api_key_manager.mark_exhausted(
                        "serpapi",
                        idx,
                        until=until,
                        reason=f"health_quota_http_{response.status_code} | {details}",
                    )
                    logger.warning(
                        "Health check quarantined exhausted SerpAPI key",
                        extra={"key_idx": idx, "status_code": response.status_code},
                    )
                return "degraded"

            if 500 <= response.status_code < 600:
                logger.warning(
                    "Health check degraded by upstream server error",
                    extra={"status_code": response.status_code},
                )
                return "degraded"

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                error_text = str(data.get("error") or "")
                error_lower = error_text.lower()
                quota_patterns = [
                    r'exhaust(ed|ion)?',
                    r'no more searches',
                    r'quota (exceeded|limit|reached)',
                    r'out of searches',
                ]
                if any(re.search(p, error_lower) for p in quota_patterns):
                    until = (datetime.now(timezone.utc) + timedelta(days=1)).timestamp()
                    if non_destructive:
                        logger.warning(
                            "Health check saw quota error payload but skipped key quarantine (non-destructive mode)",
                            extra={"key_idx": idx},
                        )
                    else:
                        await api_key_manager.mark_exhausted(
                            "serpapi",
                            idx,
                            until=until,
                            reason=f"health_quota_error | {error_text[:400]}",
                        )
                        logger.warning(
                            "Health check quarantined SerpAPI key based on quota error payload",
                            extra={"key_idx": idx},
                        )
                    return "degraded"
                transient_patterns = [
                    "temporarily unavailable",
                    "timeout",
                    "timed out",
                    "internal error",
                    "try again",
                    "later",
                ]
                if any(tok in error_lower for tok in transient_patterns):
                    logger.warning(
                        "Health check degraded by transient provider error payload",
                        extra={"error": error_text[:200]},
                    )
                    return "degraded"
                logger.error("Health check failed: API returned error", extra={"error": data["error"]})
                return "fail"

            logger.debug("Health check passed")
            return "ok"
    except RuntimeError:
        logger.warning("Health check unavailable: No SerpAPI keys available")
        return "unavailable"
    except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
        logger.warning(
            "Health check degraded by transient network issue",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return "degraded"
    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response is not None else None
        if status is not None and status >= 500:
            logger.warning("Health check degraded by HTTP 5xx", extra={"status_code": status})
            return "degraded"
        logger.error("Health check failed by HTTP error", extra={"status_code": status, "error": str(e)})
        return "fail"
    except Exception as e:
        logger.warning(
            "Health check degraded by unexpected exception",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return "degraded"
