"""
Weather API tool with async HTTP, circuit breaker, rate limiting,
structured logging, forecast support, and all requested improvements.
"""

import asyncio
import os
import time
import logging
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import IntEnum
from datetime import datetime, timedelta

import httpx

# Shared HTTP client
from core.http_client import get_client
# Centralized circuit breaker
from core.circuit_breaker import get_circuit_breaker
# Prometheus metrics
from core.metrics import (
    TOOL_REQUESTS,
    TOOL_LATENCY,
    WEATHER_RETRIES,
    WEATHER_ATTEMPTS,
)
# Request context (for correlation ID, but logging injects it automatically)
# Testing flag
from core.config import TESTING
# API key manager for rotation and exhaustion handling
from core.api_key_manager import key_manager

# ----------------------------------------------------------------------
# Module‑level configuration and state
# ----------------------------------------------------------------------

# API key is no longer defined at module level – retrieved via key manager

logger = logging.getLogger(__name__)
_TESTING_LOGGED = False

# API endpoints (all HTTPS)
BASE_URL = "https://api.openweathermap.org"
GEO_URL = f"{BASE_URL}/geo/1.0/direct"
CURRENT_URL = f"{BASE_URL}/data/2.5/weather"
FORECAST_URL = f"{BASE_URL}/data/2.5/forecast"
AIR_URL = f"{BASE_URL}/data/2.5/air_pollution"

# OpenWeather One Call 3.0 endpoints
ONECALL_URL = f"{BASE_URL}/data/3.0/onecall"
ONECALL_DAY_SUMMARY = f"{BASE_URL}/data/3.0/onecall/day_summary"
ONECALL_TIMEMACHINE = f"{BASE_URL}/data/3.0/onecall/timemachine"
ONECALL_OVERVIEW = f"{BASE_URL}/data/3.0/onecall/overview"

# Rate limiting (local, not yet centralized)
_last_call = 0.0
_rate_lock = asyncio.Lock()
RATE_LIMIT_SECONDS = float(os.getenv("WEATHER_RATE_LIMIT_SECONDS", "1.0"))

# Retry settings (local)
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _empty_weather_response(location: Optional[str] = None) -> Dict[str, Any]:
    """Graceful fallback payload for weather-unavailable scenarios."""
    return {
        "location": location,
        "forecast_date": None,
        "condition": None,
        "temperature_c": None,
        "feels_like_c": None,
        "humidity": None,
        "wind_kph": None,
        "air_quality_index": None,
        "timestamp": None,
        "temp_min_c": None,
        "temp_max_c": None,
        "has_rain": None,
        "has_snow": None,
        "precipitation_chance": None,
    }

# Geocoding cache with TTL (city.lower() -> (lat, lon, timestamp))
# Using a dict that maintains insertion order (Python 3.7+)
_geocode_cache: Dict[str, Tuple[float, float, float]] = {}
_geocode_lock = asyncio.Lock()
_geocode_inflight: Dict[str, asyncio.Future] = {}
GEOCODE_CACHE_TTL = 86400  # 24 hours
GEOCODE_MAX_SIZE = 10000   # Upper bound to prevent memory leak

def _cleanup_geocode_cache():
    """
    Remove oldest entries if cache exceeds max size.
    Simple LRU‑style: remove 20% of the oldest keys.
    """
    if len(_geocode_cache) <= GEOCODE_MAX_SIZE:
        return
    # Determine how many to remove (e.g., 20% of max size)
    to_remove = max(1, int(GEOCODE_MAX_SIZE * 0.2))
    # Since dict maintains insertion order, the first keys are oldest.
    keys = list(_geocode_cache.keys())[:to_remove]
    for k in keys:
        del _geocode_cache[k]
    logger.debug("Geocode cache cleaned", extra={"removed": to_remove, "remaining": len(_geocode_cache)})

# ----------------------------------------------------------------------
# Custom exceptions
# ----------------------------------------------------------------------

class WeatherAPIError(Exception):
    pass

# ----------------------------------------------------------------------
# Domain model
# ----------------------------------------------------------------------

class AQI(IntEnum):
    GOOD = 1
    FAIR = 2
    MODERATE = 3
    POOR = 4
    VERY_POOR = 5

@dataclass
class Weather:
    location: str
    condition: str
    temperature_c: float
    feels_like_c: float
    humidity: int
    wind_kph: float
    air_quality_index: Optional[AQI] = None
    timestamp: Optional[int] = None
    # ---- Forecast-grade fields (populated by get_forecast / get_forecast_for_date) ----
    temp_min_c: Optional[float] = None        # Daily low temperature
    temp_max_c: Optional[float] = None        # Daily high temperature
    has_rain: Optional[bool] = None           # True if rain expected
    has_snow: Optional[bool] = None           # True if snow expected
    precipitation_chance: Optional[int] = None  # 0-100 probability-of-precipitation (%)
    forecast_date: Optional[str] = None       # YYYY-MM-DD the forecast is for

    def __post_init__(self):
        # Clamp unreasonable values
        if self.temperature_c < -100 or self.temperature_c > 100:
            logger.warning("Unreasonable temperature clamped", extra={"value": self.temperature_c})
            self.temperature_c = max(-100, min(100, self.temperature_c))
        if self.feels_like_c < -100 or self.feels_like_c > 100:
            logger.warning("Unreasonable feels_like clamped", extra={"value": self.feels_like_c})
            self.feels_like_c = max(-100, min(100, self.feels_like_c))
        if self.humidity < 0 or self.humidity > 100:
            logger.warning("Unreasonable humidity clamped", extra={"value": self.humidity})
            self.humidity = max(0, min(100, self.humidity))
        if self.wind_kph < 0 or self.wind_kph > 500:
            logger.warning("Unreasonable wind speed clamped", extra={"value": self.wind_kph})
            self.wind_kph = max(0, min(500, self.wind_kph))

    def to_dict(self) -> dict:
        """Convert Weather object to a dictionary for serialization."""
        return {
            "location": self.location,
            "condition": self.condition,
            "temperature_c": self.temperature_c,
            "feels_like_c": self.feels_like_c,
            "humidity": self.humidity,
            "wind_kph": self.wind_kph,
            "air_quality_index": self.air_quality_index.value if self.air_quality_index else None,
            "timestamp": self.timestamp,
            "temp_min_c": self.temp_min_c,
            "temp_max_c": self.temp_max_c,
            "has_rain": self.has_rain,
            "has_snow": self.has_snow,
            "precipitation_chance": self.precipitation_chance,
            "forecast_date": self.forecast_date,
        }

# ----------------------------------------------------------------------
# Rate limiting helper (fixed: lock not held during sleep)
# ----------------------------------------------------------------------

async def _rate_limit():
    global _last_call
    async with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_call
        if elapsed >= RATE_LIMIT_SECONDS:
            # Good to go immediately, update last_call and return
            _last_call = now
            return
        # Reserve a slot in the future for spacing
        sleep_time = RATE_LIMIT_SECONDS - elapsed
        _last_call = now + sleep_time

    # IMPORTANT: release the lock before awaiting sleep
    if sleep_time > 0:
        await asyncio.sleep(sleep_time)
    # After sleeping, caller proceeds to make the request

# ----------------------------------------------------------------------
# Core request function (without circuit breaker wrapper)
# ----------------------------------------------------------------------

async def _make_request_raw(
    method: str,
    url: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Low‑level HTTP request with rate limiting, retries, and logging.
    Does NOT include circuit breaker – that is applied by the public wrapper.
    Uses key manager for API key rotation and exhaustion handling.
    """
    await _rate_limit()

    last_exception = None
    overall_start = time.monotonic()
    # request_id is automatically added to logs by logging configuration

    unauthorized_seen = 0

    for attempt in range(MAX_RETRIES):
        # Reserve a key for the duration of this attempt so it isn't rotated out.
        try:
            async with key_manager.reserve_key("weather") as (idx, key):
                if not key:
                    raise WeatherAPIError("Weather provider keys exhausted (service unavailable)")

                logger.info("[WEATHER] using key index: %s", idx)

                # Add API key to params
                params_with_key = dict(params)
                params_with_key["appid"] = key

                attempt_start = time.monotonic()
                try:
                    client = get_client()
                    resp = await client.request(method, url, params=params_with_key)
                    latency = time.monotonic() - attempt_start

                    logger.info(
                        "Weather API request attempt",
                        extra={
                            "url": url,
                            "status_code": resp.status_code,
                            "attempt": attempt + 1,
                            "latency_sec": round(latency, 3),
                            "api": "openweathermap",
                            "key_idx": idx,
                        },
                    )
                    logger.info("[WEATHER] response status: %s (key index: %s)", resp.status_code, idx)

                    # SUCCESS
                    if resp.status_code == 200:
                        data = resp.json()
                        # --- ADD THIS RAW LOGGING BLOCK ---
                        logger.debug(
                            "RAW WEATHER API TRACE",
                            extra={
                                "request_url": url,
                                "request_params": params_with_key,
                                "response_payload": data
                            }
                        )
                        # ----------------------------------
                        # Handle OpenWeather payload-level errors that may still arrive with HTTP 200.
                        if isinstance(data, dict):
                            cod = str(data.get("cod", "")).strip().lower()
                            msg = (data.get("message") or "").lower()
                            if cod in {"401", "403"} or "invalid api key" in msg or "unauthorized" in msg:
                                unauthorized_seen += 1
                                logger.warning(
                                    "Weather key unauthorized (likely not activated or invalid) — skipping key",
                                    extra={"key_idx": idx},
                                )
                                continue
                            if cod == "429" or "limit" in msg or "quota" in msg:
                                reset_ts = int((datetime.now() + timedelta(days=1)).timestamp())
                                details = (data.get("message") or "")[:1000]
                                await key_manager.mark_exhausted("weather", idx, until=reset_ts, reason=f"quota | {details}")
                                logger.warning("Weather key quota exceeded (text) — marked exhausted", extra={"key_idx": idx})
                                continue

                        # Normal success – record usage and return
                        await key_manager.record_usage("weather", idx)

                        total_time = time.monotonic() - overall_start
                        logger.info(
                            "Weather API request successful (total)",
                            extra={
                                "total_latency_sec": round(total_time, 3),
                                "attempts": attempt + 1,
                                "api": "openweathermap",
                            },
                        )
                        WEATHER_ATTEMPTS.observe(attempt + 1)
                        return data

                    # UNAUTHORIZED / FORBIDDEN -> do NOT exhaust; skip key and try another key
                    if resp.status_code in (401, 403):
                        unauthorized_seen += 1
                        logger.warning(
                            "Weather key unauthorized (likely not activated or invalid) — skipping key",
                            extra={"key_idx": idx},
                        )
                        continue

                    # RATE LIMIT / PAYMENT / QUOTA -> attempt to detect reset, else mark with default
                    if resp.status_code == 429:
                        # 429 is usually temporary; apply short cooldown and retry.
                        cooldown_seconds = 60
                        try:
                            ra = resp.headers.get("Retry-After")
                            if ra:
                                # some servers give seconds, some give HTTP-date — try integer first
                                try:
                                    cooldown_seconds = max(1, int(ra))
                                except ValueError:
                                    # fallback: parse HTTP-date — skip here for brevity, use default
                                    cooldown_seconds = 60
                        except Exception:
                            cooldown_seconds = 60

                        reset_ts = int(datetime.now().timestamp()) + cooldown_seconds

                        details = (resp.text or "")[:1000]
                        await key_manager.mark_exhausted("weather", idx, until=reset_ts, reason=f"http_429 | {details}")
                        WEATHER_RETRIES.labels(reason="http_429").inc()
                        logger.warning("Weather key rate-limited — marked exhausted", extra={"key_idx": idx})
                        await asyncio.sleep(random.uniform(0.1, 0.3))
                        continue

                    # RETRYABLE 5xx -> don't mark exhausted, allow retry/rotation
                    if resp.status_code in (500, 502, 503, 504):
                        WEATHER_RETRIES.labels(reason="http_5xx").inc()
                        wait_time = (2 ** attempt) + random.uniform(0, 0.3)
                        logger.warning("Server error from weather API, retrying", extra={"status_code": resp.status_code})
                        await asyncio.sleep(wait_time)
                        continue

                    # Other client errors -> fatal for this request
                    error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    raise WeatherAPIError(error_msg)

                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    # Network issue — do NOT mark key exhausted. Allow retry (may use next key).
                    last_exception = e
                    latency = time.monotonic() - attempt_start
                    WEATHER_RETRIES.labels(reason="network").inc()
                    logger.warning(
                        "Request attempt failed (network)",
                        extra={
                            "error_type": type(e).__name__,
                            "attempt": attempt + 1,
                            "latency_sec": round(latency, 3),
                            "api": "openweathermap",
                            "key_idx": idx,
                        },
                    )
                    # If not last attempt, sleep and continue to next attempt (which will reserve a fresh key)
                    if attempt < MAX_RETRIES - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 0.3)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise WeatherAPIError(f"Max retries exceeded: {e}") from e

                except Exception as e:
                    # Any unexpected error inside reserved key usage -> re-raise wrapped
                    raise WeatherAPIError(str(e)) from e

        except RuntimeError as e:
            # reserve_key may raise RuntimeError if no keys available — match airline API message
            raise WeatherAPIError("All weather keys exhausted or failed") from e

    # If we exit the attempts loop without success
    if unauthorized_seen:
        raise WeatherAPIError("All weather keys unauthorized or pending activation")
    raise WeatherAPIError(f"Max retries exceeded, last error: {last_exception}")

# ----------------------------------------------------------------------
# Public request wrapper with circuit breaker
# ----------------------------------------------------------------------

async def _make_request(
    method: str,
    url: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Make an HTTP request protected by the circuit breaker.
    """
    # Get the circuit breaker instance for this API
    breaker = await get_circuit_breaker("weather_api")

    # Use the breaker's call() method – it automatically handles open state
    return await breaker.call(
        lambda: _make_request_raw(method, url, params)
    )

# ----------------------------------------------------------------------
# OneCall API helper
# ----------------------------------------------------------------------

async def _get_onecall(
    lat: float,
    lon: float,
    *,
    units: str = "metric",
    exclude: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Wrapper around OpenWeather OneCall API.

    Allows excluding unused sections to reduce payload size.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "units": units,
    }

    if exclude:
        params["exclude"] = exclude

    if extra:
        params.update(extra)

    return await _make_request("GET", ONECALL_URL, params)

# ----------------------------------------------------------------------
# Geocoding with TTL cache (async‑safe)
# ----------------------------------------------------------------------

async def _get_coordinates(city: str) -> tuple[float, float]:
    """Convert city name to latitude/longitude using OpenWeatherMap Geocoding API."""
    if not city or not isinstance(city, str):
        raise ValueError("Location must be a non‑empty string")

    city_lower = city.lower()
    now = time.time()

    # Fast-path cache check
    if city_lower in _geocode_cache:
        lat, lon, ts = _geocode_cache[city_lower]
        if now - ts < GEOCODE_CACHE_TTL:
            logger.debug("Geocoding cache hit", extra={"city": city})
            return lat, lon
        logger.debug("Geocoding cache expired", extra={"city": city})

    # Dedupe concurrent requests for the same location
    created_future = False
    async with _geocode_lock:
        # Re-check cache while lock held
        cached = _geocode_cache.get(city_lower)
        if cached:
            lat, lon, ts = cached
            if time.time() - ts < GEOCODE_CACHE_TTL:
                return lat, lon

        inflight = _geocode_inflight.get(city_lower)
        if inflight is None:
            inflight = asyncio.get_running_loop().create_future()
            _geocode_inflight[city_lower] = inflight
            created_future = True

    if not created_future:
        lat, lon = await inflight
        return lat, lon

    url = GEO_URL
    params = {
        "q": city,
        "limit": 1,
        # API key will be added inside _make_request_raw
    }
    try:
        data = await _make_request("GET", url, params)

        if not isinstance(data, list) or len(data) == 0:
            raise WeatherAPIError(f"Location '{city}' not found")

        try:
            lat = data[0]["lat"]
            lon = data[0]["lon"]
        except (KeyError, IndexError) as e:
            raise WeatherAPIError(f"Unexpected geocoding response: {e}") from e

        # Write to cache and resolve waiting callers
        async with _geocode_lock:
            _geocode_cache[city_lower] = (lat, lon, time.time())
            _cleanup_geocode_cache()
            fut = _geocode_inflight.pop(city_lower, None)
            if fut and not fut.done():
                fut.set_result((lat, lon))
        return lat, lon
    except Exception as e:
        async with _geocode_lock:
            fut = _geocode_inflight.pop(city_lower, None)
            if fut and not fut.done():
                fut.set_exception(e)
        raise

# ----------------------------------------------------------------------
# Current weather (instrumented)
# ----------------------------------------------------------------------

async def get_current_weather(
    location: str,
    units: str = "metric",
) -> Weather:
    """Fetch current weather conditions using free-tier /data/2.5/weather."""
    start = time.monotonic()

    # TESTING bypass — return deterministic Weather object (no network)
    global _TESTING_LOGGED
    if TESTING:
        if not _TESTING_LOGGED:
            logger.info("TESTING mode enabled — returning fake weather results")
            _TESTING_LOGGED = True
        return Weather(
            location=location,
            condition="Clear",
            temperature_c=25.0,
            feels_like_c=25.0,
            humidity=50,
            wind_kph=5.0,
            air_quality_index=AQI.GOOD,
            timestamp=None,
        )

    try:
        if units not in ("metric", "imperial"):
            raise ValueError("units must be 'metric' or 'imperial'")

        lat, lon = await _get_coordinates(location)

        data = await _make_request(
            "GET",
            CURRENT_URL,
            {"lat": lat, "lon": lon, "units": units},
        )

        # Convert wind speed to kph if needed (/weather returns m/s in metric, mph in imperial)
        wind_speed = data.get("wind", {}).get("speed", 0.0)
        if units == "metric":
            wind_kph = wind_speed * 3.6
        else:
            wind_kph = wind_speed * 1.60934

        # Determine weather condition
        weather_list = data.get("weather", [])
        if not weather_list:
            raise WeatherAPIError("Missing 'weather' in current response")
        condition = weather_list[0].get("description", "Unknown").capitalize()

        # Rain/snow detection (based on condition codes)
        condition_id = weather_list[0].get("id", 0)
        condition_main = weather_list[0].get("main", "").lower()
        has_rain = condition_id < 700 and condition_id >= 200 and "snow" not in condition_main
        has_snow = "snow" in condition_main or (600 <= condition_id < 700)

        main = data.get("main", {})
        temp = main.get("temp")
        if temp is None:
            raise WeatherAPIError("Missing 'main.temp' in current response")

        temp_min = main.get("temp_min", temp)
        temp_max = main.get("temp_max", temp)

        result = Weather(
            location=location,
            condition=condition,
            temperature_c=temp,
            feels_like_c=main.get("feels_like", temp),
            humidity=main.get("humidity", 0),
            wind_kph=round(wind_kph, 1),
            timestamp=data.get("dt"),
            temp_min_c=temp_min,
            temp_max_c=temp_max,
            has_rain=has_rain,
            has_snow=has_snow,
            forecast_date=datetime.utcfromtimestamp(data.get("dt", int(time.time()))).strftime("%Y-%m-%d"),
        )

        # Fetch air quality separately (OneCall doesn't include it)
        try:
            air_params = {"lat": lat, "lon": lon}
            air_data = await _make_request("GET", AIR_URL, air_params)
            air_list = air_data.get("list")
            if air_list and isinstance(air_list, list) and len(air_list) > 0:
                aqi_value = air_list[0].get("main", {}).get("aqi")
                if aqi_value is not None:
                    result.air_quality_index = AQI(aqi_value)
        except Exception as e:
            logger.warning("Could not fetch AQI for current weather", extra={"error": str(e)})

        TOOL_REQUESTS.labels(tool="weather", status="success").inc()
        return result

    except Exception:
        TOOL_REQUESTS.labels(tool="weather", status="error").inc()
        raise

    finally:
        TOOL_LATENCY.labels(tool="weather").observe(
            time.monotonic() - start
        )

# ----------------------------------------------------------------------
# Forecast (instrumented)
# ----------------------------------------------------------------------

async def get_forecast(
    location: str,
    days: int = 3,
    units: str = "metric",
) -> List[Weather]:
    """
    Return weather forecast using free-tier /data/2.5/forecast.
    Returns one Weather object per day.
    """
    start = time.monotonic()

    if TESTING:
        base = datetime.now().date()
        out: List[Weather] = []
        for i in range(days):
            d = base + timedelta(days=i)
            out.append(
                Weather(
                    location=location,
                    condition="Clear",
                    temperature_c=25.0 + i,
                    feels_like_c=25.0 + i,
                    humidity=50,
                    wind_kph=5.0,
                    air_quality_index=AQI.GOOD,
                    timestamp=None,
                    temp_min_c=22.0 + i,
                    temp_max_c=28.0 + i,
                    has_rain=False,
                    has_snow=False,
                    precipitation_chance=10,
                    forecast_date=d.strftime("%Y-%m-%d"),
                )
            )
        return out

    try:
        if not 1 <= days <= 5:
            raise ValueError("days must be between 1 and 5")
        if units not in ("metric", "imperial"):
            raise ValueError("units must be 'metric' or 'imperial'")

        lat, lon = await _get_coordinates(location)

        data = await _make_request(
            "GET",
            FORECAST_URL,
            {"lat": lat, "lon": lon, "units": units},
        )

        entries = data.get("list", [])
        if not entries:
            raise WeatherAPIError("Missing 'list' in forecast response")

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in entries:
            ts = item.get("dt")
            if ts:
                day_key = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            else:
                dt_txt = item.get("dt_txt")
                if not dt_txt or len(dt_txt) < 10:
                    continue
                day_key = dt_txt[:10]
            grouped.setdefault(day_key, []).append(item)

        daily_forecasts = []
        for forecast_date in sorted(grouped.keys())[:days]:
            day_items = grouped[forecast_date]
            if not day_items:
                continue

            # Choose entry closest to local midday for condition/timestamp.
            def _hour_dist(it: Dict[str, Any]) -> int:
                dt_txt = it.get("dt_txt", "")
                if len(dt_txt) >= 13:
                    try:
                        hour = int(dt_txt[11:13])
                        return abs(hour - 12)
                    except Exception:
                        return 99
                return 99

            representative = min(day_items, key=_hour_dist)

            temps = [i.get("main", {}).get("temp") for i in day_items if i.get("main", {}).get("temp") is not None]
            temp_mins = [i.get("main", {}).get("temp_min") for i in day_items if i.get("main", {}).get("temp_min") is not None]
            temp_maxs = [i.get("main", {}).get("temp_max") for i in day_items if i.get("main", {}).get("temp_max") is not None]
            humidities = [i.get("main", {}).get("humidity") for i in day_items if i.get("main", {}).get("humidity") is not None]
            feels = [i.get("main", {}).get("feels_like") for i in day_items if i.get("main", {}).get("feels_like") is not None]
            winds = [i.get("wind", {}).get("speed") for i in day_items if i.get("wind", {}).get("speed") is not None]

            weather_items = []
            for i in day_items:
                weather_items.extend(i.get("weather", []))
            weather_entry = weather_items[0] if weather_items else {}
            condition = weather_entry.get("description", "Unknown").capitalize()
            condition_id = weather_entry.get("id", 0)
            condition_main = weather_entry.get("main", "").lower()

            has_snow = any(
                ("snow" in str(w.get("main", "")).lower()) or (600 <= int(w.get("id", 0)) < 700)
                for w in weather_items
            )
            has_rain = any(
                (200 <= int(w.get("id", 0)) < 700) and ("snow" not in str(w.get("main", "")).lower())
                for w in weather_items
            ) or any((i.get("pop", 0) or 0) > 0 for i in day_items)

            pop_values = [float(i.get("pop", 0) or 0) for i in day_items]
            precipitation_chance = int(round(max(pop_values) * 100)) if pop_values else 0

            wind_speed = (sum(winds) / len(winds)) if winds else 0.0
            if units == "metric":
                wind_kph = wind_speed * 3.6
            else:
                wind_kph = wind_speed * 1.60934

            base_temp = (sum(temps) / len(temps)) if temps else 0.0
            temp_min = min(temp_mins) if temp_mins else base_temp
            temp_max = max(temp_maxs) if temp_maxs else base_temp
            feels_like = (sum(feels) / len(feels)) if feels else base_temp
            humidity = int(round(sum(humidities) / len(humidities))) if humidities else 0

            # Keep fallback condition if weather list was sparse.
            if condition == "Unknown" and condition_id:
                condition = str(condition_main or "Unknown").capitalize()

            w = Weather(
                location=location,
                condition=condition,
                temperature_c=base_temp,
                feels_like_c=feels_like,
                humidity=humidity,
                wind_kph=round(wind_kph, 1),
                timestamp=representative.get("dt"),
                temp_min_c=temp_min,
                temp_max_c=temp_max,
                has_rain=has_rain,
                has_snow=has_snow,
                precipitation_chance=precipitation_chance,
                forecast_date=forecast_date,
            )
            daily_forecasts.append(w)

        result = daily_forecasts

        TOOL_REQUESTS.labels(tool="weather", status="success").inc()
        return result

    except Exception:
        TOOL_REQUESTS.labels(tool="weather", status="error").inc()
        raise

    finally:
        TOOL_LATENCY.labels(tool="weather").observe(
            time.monotonic() - start
        )

# ----------------------------------------------------------------------
# Forecast for a specific travel date (used by planner_agent)
# ----------------------------------------------------------------------

async def get_forecast_for_date(
    location: str,
    travel_date: str,
    units: str = "metric",
) -> Any:
    """
    Return the forecast Weather for a specific travel date (YYYY-MM-DD).

    Falls back to get_forecast(days=5) and picks the entry whose forecast_date
    is closest to travel_date. If the travel date is beyond the 5-day window,
    falls back to current weather so the planner always gets something useful.

    Args:
        location: City name or IATA code (geocoding is handled internally)
        travel_date: Target date in YYYY-MM-DD format
        units: "metric" (default) or "imperial"

    Returns:
        Weather object enriched with temp_min_c, temp_max_c, has_rain, has_snow,
        precipitation_chance, and forecast_date fields.
        If weather is unavailable across all keys, returns a structured empty dict.
    """
    try:
        forecasts = await get_forecast(location=location, days=5, units=units)
    except Exception as e:
        # Catch any unexpected error (e.g., all keys unauthorized, geocoding failure)
        logger.warning(
            "get_forecast_for_date: forecast unavailable, falling back to current weather",
            extra={"error": str(e), "location": location}
        )
        try:
            return await get_current_weather(location, units)
        except Exception as current_e:
            logger.warning(
                "get_forecast_for_date: current weather unavailable, returning empty weather response",
                extra={"error": str(current_e), "location": location},
            )
            return _empty_weather_response(location)

    # AQI enrichment is optional to avoid duplicate API calls on every request.
    include_aqi = os.getenv("WEATHER_INCLUDE_AQI_IN_FORECAST", "0").lower() in ("1", "true", "yes", "on")
    aqi = None
    if include_aqi:
        try:
            current = await get_current_weather(location=location, units=units)
            aqi = current.air_quality_index
        except Exception as current_err:
            logger.warning("Could not fetch AQI for forecast", extra={"error": str(current_err)})

    if not forecasts:
        logger.warning("get_forecast_for_date: no forecast entries returned, using current weather")
        try:
            return await get_current_weather(location, units)
        except Exception as current_e:
            logger.warning(
                "get_forecast_for_date: no forecast and current weather unavailable, returning empty weather response",
                extra={"error": str(current_e), "location": location},
            )
            return _empty_weather_response(location)

    try:
        target = datetime.strptime(travel_date, "%Y-%m-%d").date()
    except ValueError:
        logger.warning("get_forecast_for_date: invalid travel_date %s, using first forecast", travel_date)
        return forecasts[0]

    # Find the forecast entry with the date closest to the target
    best: Optional[Weather] = None
    best_delta: Optional[int] = None
    for fw in forecasts:
        if fw.forecast_date:
            try:
                fd = datetime.strptime(fw.forecast_date, "%Y-%m-%d").date()
                delta = abs((fd - target).days)
                if best_delta is None or delta < best_delta:
                    best = fw
                    best_delta = delta
            except ValueError:
                continue

    if best is not None:
        logger.info(
            "get_forecast_for_date: matched forecast",
            extra={"target": travel_date, "matched": best.forecast_date, "delta_days": best_delta}
        )

        # Attach the fetched AQI (if any)
        if aqi is not None:
            best.air_quality_index = aqi

        return best

    # Fallback: return first entry
    logger.warning("get_forecast_for_date: no dated entry found, returning first forecast")
    return forecasts[0]

async def get_weather(
    location: str,
    units: str = "metric",
) -> Weather:
    """Convenience function that returns only current weather."""
    return await get_current_weather(location, units)

# ----------------------------------------------------------------------
# Backward compatibility alias for planner_agent
# ----------------------------------------------------------------------

async def check_weather(location: str, units: str = "metric") -> Weather:
    """
    Alias for get_weather — kept for backward compatibility with planner_agent.
    Returns current weather for the given location.
    """
    return await get_weather(location, units)

# ----------------------------------------------------------------------
# Health check (real API ping)
# ----------------------------------------------------------------------

async def health_check() -> str:
    """
    Lightweight health check for weather API.
    Returns "ok" or "fail".
    """
    try:
        # Reserve a key for the health check to avoid rotation mid-check
        async with key_manager.reserve_key("weather") as (idx, key):
            if not key:
                logger.error("Weather health check failed: no keys available")
                return "fail"

            await _rate_limit()
            client = get_client()
            params = {"q": "Delhi", "limit": 1, "appid": key}

            # Try twice on network errors; do not mark key exhausted on timeout/connect failure here
            for attempt in range(2):
                try:
                    response = await client.get(GEO_URL, params=params, timeout=5.0)
                    # Distinguish unauthorized explicitly
                    if response.status_code in (401, 403):
                        logger.error("Weather health check failed: invalid API key (401/403); not auto-exhausting here")
                        return "fail"

                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, list):
                        logger.error("Weather health check failed: unexpected response structure")
                        return "fail"
                    logger.debug("Weather health check passed")
                    return "ok"
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    logger.warning("Weather health check network error", extra={"error": str(e), "attempt": attempt+1})
                    if attempt == 1:
                        # final attempt failed
                        raise
                    await asyncio.sleep(0.5)
    except Exception:
        logger.exception("Weather health check failed")
        return "fail"
