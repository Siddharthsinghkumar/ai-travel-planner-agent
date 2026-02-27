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

# ----------------------------------------------------------------------
# Module‑level configuration and state
# ----------------------------------------------------------------------

# API key is no longer defined at module level – retrieved inside functions

logger = logging.getLogger(__name__)

# API endpoints (all HTTPS)
BASE_URL = "https://api.openweathermap.org"
GEO_URL = f"{BASE_URL}/geo/1.0/direct"
CURRENT_URL = f"{BASE_URL}/data/2.5/weather"
FORECAST_URL = f"{BASE_URL}/data/2.5/forecast"
AIR_URL = f"{BASE_URL}/data/2.5/air_pollution"

# Rate limiting (local, not yet centralized)
_last_call = 0.0
_rate_lock = asyncio.Lock()
RATE_LIMIT_SECONDS = 0.5

# Retry settings (local)
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Geocoding cache with TTL (city.lower() -> (lat, lon, timestamp))
_geocode_cache: Dict[str, Tuple[float, float, float]] = {}
_geocode_lock = asyncio.Lock()
GEOCODE_CACHE_TTL = 86400  # 24 hours

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
        }

# ----------------------------------------------------------------------
# Rate limiting helper
# ----------------------------------------------------------------------

async def _rate_limit():
    global _last_call
    async with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_call
        if elapsed < RATE_LIMIT_SECONDS:
            await asyncio.sleep(RATE_LIMIT_SECONDS - elapsed)
        _last_call = time.monotonic()

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
    """
    # Retrieve and validate API key at call time
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        raise WeatherAPIError("WEATHER_API_KEY is not set")

    # Add API key to params (it may already be there from caller, but ensure it's present)
    params_with_key = dict(params)
    params_with_key["appid"] = api_key

    await _rate_limit()

    last_exception = None
    overall_start = time.monotonic()
    # request_id is automatically added to logs by logging configuration

    for attempt in range(MAX_RETRIES):
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
                },
            )

            if resp.status_code == 200:
                data = resp.json()

                total_time = time.monotonic() - overall_start
                logger.info(
                    "Weather API request successful (total)",
                    extra={
                        "total_latency_sec": round(total_time, 3),
                        "attempts": attempt + 1,
                        "api": "openweathermap",
                    },
                )

                # Record how many attempts were needed
                WEATHER_ATTEMPTS.observe(attempt + 1)

                return data

            if resp.status_code in RETRYABLE_STATUS_CODES:
                # Increment retry counter with appropriate reason
                if resp.status_code == 429:
                    WEATHER_RETRIES.labels(reason="http_429").inc()
                else:
                    WEATHER_RETRIES.labels(reason="http_5xx").inc()

                wait_time = (2 ** attempt) + random.uniform(0, 0.3)
                logger.warning(
                    "Retryable status, retrying",
                    extra={
                        "status_code": resp.status_code,
                        "wait_time": round(wait_time, 3),
                    },
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                raise WeatherAPIError(error_msg)

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_exception = e
            latency = time.monotonic() - attempt_start

            # Increment retry counter for network errors
            WEATHER_RETRIES.labels(reason="network").inc()

            logger.warning(
                "Request attempt failed",
                extra={
                    "error_type": type(e).__name__,
                    "attempt": attempt + 1,
                    "latency_sec": round(latency, 3),
                    "api": "openweathermap",
                },
            )
            if attempt < MAX_RETRIES - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 0.3)
                await asyncio.sleep(wait_time)
            else:
                raise WeatherAPIError(f"Max retries exceeded: {e}") from e

        except Exception as e:
            raise WeatherAPIError(str(e)) from e

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
# Geocoding with TTL cache (async‑safe)
# ----------------------------------------------------------------------

async def _get_coordinates(city: str) -> tuple[float, float]:
    """Convert city name to latitude/longitude using OpenWeatherMap Geocoding API."""
    if not city or not isinstance(city, str):
        raise ValueError("Location must be a non‑empty string")

    city_lower = city.lower()
    now = time.time()

    # Check cache with TTL (read is safe without lock)
    if city_lower in _geocode_cache:
        lat, lon, ts = _geocode_cache[city_lower]
        if now - ts < GEOCODE_CACHE_TTL:
            logger.debug("Geocoding cache hit", extra={"city": city})
            return lat, lon
        else:
            logger.debug("Geocoding cache expired", extra={"city": city})

    url = GEO_URL
    params = {
        "q": city,
        "limit": 1,
        # API key will be added inside _make_request_raw
    }
    data = await _make_request("GET", url, params)

    if not isinstance(data, list) or len(data) == 0:
        raise WeatherAPIError(f"Location '{city}' not found")

    try:
        lat = data[0]["lat"]
        lon = data[0]["lon"]
        # Write to cache with lock
        async with _geocode_lock:
            _geocode_cache[city_lower] = (lat, lon, time.time())
        return lat, lon
    except (KeyError, IndexError) as e:
        raise WeatherAPIError(f"Unexpected geocoding response: {e}") from e

# ----------------------------------------------------------------------
# Current weather (instrumented)
# ----------------------------------------------------------------------

async def get_current_weather(
    location: str,
    units: str = "metric",
) -> Weather:
    """Fetch current weather for a location."""
    start = time.monotonic()

    try:
        if units not in ("metric", "imperial"):
            raise ValueError("units must be 'metric' or 'imperial'")

        lat, lon = await _get_coordinates(location)

        # Prepare parameters for both requests (API key will be added inside _make_request_raw)
        weather_params = {
            "lat": lat,
            "lon": lon,
            "units": units,
        }
        air_params = {
            "lat": lat,
            "lon": lon,
        }

        # Run both requests concurrently to reduce total latency
        weather_task = asyncio.create_task(_make_request("GET", CURRENT_URL, weather_params))
        air_task = asyncio.create_task(_make_request("GET", AIR_URL, air_params))

        weather_data, air_data = await asyncio.gather(weather_task, air_task)

        # Defensive parsing
        try:
            main = weather_data.get("main")
            if not main:
                raise WeatherAPIError("Missing 'main' in weather response")
            temp = main.get("temp")
            feels_like = main.get("feels_like")
            humidity = main.get("humidity")
            if temp is None or feels_like is None or humidity is None:
                raise WeatherAPIError("Missing temperature/feels_like/humidity in weather response")

            weather_list = weather_data.get("weather")
            if not weather_list or not isinstance(weather_list, list) or len(weather_list) == 0:
                raise WeatherAPIError("Missing 'weather' array in response")
            condition = weather_list[0].get("description", "Unknown").capitalize()

            wind = weather_data.get("wind", {})
            wind_speed = wind.get("speed", 0.0)

            if units == "metric":
                wind_kph = wind_speed * 3.6
            else:
                wind_kph = wind_speed * 1.60934

            air_list = air_data.get("list")
            if not air_list or not isinstance(air_list, list) or len(air_list) == 0:
                raise WeatherAPIError("Missing 'list' in air pollution response")
            aqi_value = air_list[0].get("main", {}).get("aqi")
            if aqi_value is None:
                raise WeatherAPIError("Missing AQI in air pollution response")
            aqi = AQI(aqi_value)

        except (KeyError, TypeError, ValueError) as e:
            raise WeatherAPIError(f"Failed to parse weather data: {e}") from e

        result = Weather(
            location=location,
            condition=condition,
            temperature_c=temp,
            feels_like_c=feels_like,
            humidity=humidity,
            wind_kph=round(wind_kph, 1),
            air_quality_index=aqi,
        )

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
    Fetch weather forecast for the next `days` (max 5).
    Returns one Weather object per day (using the 12:00 UTC forecast).
    """
    start = time.monotonic()

    try:
        if not 1 <= days <= 5:
            raise ValueError("days must be between 1 and 5")
        if units not in ("metric", "imperial"):
            raise ValueError("units must be 'metric' or 'imperial'")

        lat, lon = await _get_coordinates(location)

        params = {
            "lat": lat,
            "lon": lon,
            "units": units,
            "cnt": days * 8,  # 8 three‑hour periods per day
        }
        data = await _make_request("GET", FORECAST_URL, params)

        try:
            forecast_list = data.get("list")
            if not forecast_list or not isinstance(forecast_list, list):
                raise WeatherAPIError("Missing 'list' in forecast response")

            daily_forecasts = []
            for entry in forecast_list:
                dt_txt = entry.get("dt_txt", "")
                if dt_txt.endswith("12:00:00"):
                    main = entry.get("main", {})
                    temp = main.get("temp")
                    feels_like = main.get("feels_like")
                    humidity = main.get("humidity")
                    if temp is None or feels_like is None or humidity is None:
                        raise WeatherAPIError("Missing temperature data in forecast entry")

                    weather_list = entry.get("weather")
                    if not weather_list or len(weather_list) == 0:
                        raise WeatherAPIError("Missing weather description in forecast")
                    condition = weather_list[0].get("description", "Unknown").capitalize()

                    wind = entry.get("wind", {})
                    wind_speed = wind.get("speed", 0.0)
                    if units == "metric":
                        wind_kph = wind_speed * 3.6
                    else:
                        wind_kph = wind_speed * 1.60934

                    timestamp = entry.get("dt")

                    daily_forecasts.append(Weather(
                        location=location,
                        condition=condition,
                        temperature_c=temp,
                        feels_like_c=feels_like,
                        humidity=humidity,
                        wind_kph=round(wind_kph, 1),
                        timestamp=timestamp,
                    ))

                    if len(daily_forecasts) >= days:
                        break

            if len(daily_forecasts) < days:
                logger.warning(
                    "Fewer midday forecasts than requested",
                    extra={"found": len(daily_forecasts), "requested": days}
                )

            result = daily_forecasts

            TOOL_REQUESTS.labels(tool="weather", status="success").inc()
            return result

        except (KeyError, TypeError, ValueError) as e:
            raise WeatherAPIError(f"Failed to parse forecast data: {e}") from e

    except Exception:
        TOOL_REQUESTS.labels(tool="weather", status="error").inc()
        raise

    finally:
        TOOL_LATENCY.labels(tool="weather").observe(
            time.monotonic() - start
        )

# ----------------------------------------------------------------------
# Unified get_weather (current only)
# ----------------------------------------------------------------------

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

    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        logger.error("Weather health check failed: WEATHER_API_KEY not set")
        return "fail"

    try:
        await _rate_limit()
        client = get_client()

        params = {
            "q": "Delhi",
            "limit": 1,
            "appid": api_key,
        }

        # --- retry block ---
        for attempt in range(2):
            try:
                response = await client.get(
                    GEO_URL,
                    params=params,
                    timeout=5.0,   # ✅ explicit timeout
                )
                response.raise_for_status()
                data = response.json()

                if not isinstance(data, list):
                    logger.error(
                        "Weather health check failed: unexpected response structure"
                    )
                    return "fail"

                logger.info("Weather health check passed")
                return "ok"

            except Exception:
                if attempt == 1:
                    raise
                await asyncio.sleep(0.5)  # small backoff

    except Exception:
        # ✅ full traceback logging
        logger.exception("Weather health check failed")
        return "fail"