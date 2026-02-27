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
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

import httpx
from dotenv import load_dotenv

# Import shared HTTP client, circuit breaker, and metrics
from core.http_client import get_client
from core.circuit_breaker import get_circuit_breaker
from core.metrics import TOOL_REQUESTS, TOOL_LATENCY, AIRLINE_RETRIES, AIRLINE_ATTEMPTS

load_dotenv()

# ----------------------------------------------------------------------
# Structured logging (configuration left to application)
# ----------------------------------------------------------------------
logger = logging.getLogger("airline_api")

# ----------------------------------------------------------------------
# Custom exceptions
# ----------------------------------------------------------------------
class AirlineAPIError(Exception):
    """Raised when the airline search API fails after retries."""
    pass

# ----------------------------------------------------------------------
# Domain model
# ----------------------------------------------------------------------
@dataclass
class Flight:
    airline: str
    flight_no: str
    departure_time: str
    arrival_time: str
    duration_min: int
    price_inr: int

# ----------------------------------------------------------------------
# API key loaded from environment (validation moved inside function)
# ----------------------------------------------------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # may be None; checked later

# ----------------------------------------------------------------------
# Rate limiting (async, with lock for concurrency safety)
# ----------------------------------------------------------------------
_last_call = 0.0
_rate_lock = asyncio.Lock()
RATE_LIMIT_SECONDS = 1.0  # at most 1 call per second

async def _rate_limit():
    global _last_call
    async with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_call
        if elapsed < RATE_LIMIT_SECONDS:
            sleep_time = RATE_LIMIT_SECONDS - elapsed
            logger.debug("Rate limiting sleep", extra={"sleep_sec": round(sleep_time, 2)})
            await asyncio.sleep(sleep_time)
        _last_call = time.monotonic()

# ----------------------------------------------------------------------
# Duration parser (handles both int and string formats)
# ----------------------------------------------------------------------
def _parse_duration(duration) -> Optional[int]:
    """Convert duration from SerpAPI to minutes. Returns None if parsing fails."""
    if isinstance(duration, int):
        return duration
    if isinstance(duration, str):
        # Try to parse formats like "2h 15m", "2 h 15 min", "2:15", etc.
        # Simple regex: capture hours and minutes
        pattern = r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m(?:in)?)?"
        match = re.match(pattern, duration.strip(), re.IGNORECASE)
        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2)) if match.group(2) else 0
            return hours * 60 + minutes
        # Try colon format like "2:15"
        if ":" in duration:
            parts = duration.split(":")
            if len(parts) == 2:
                try:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    return hours * 60 + minutes
                except ValueError:
                    pass
    return None

# ----------------------------------------------------------------------
# Main search function
# ----------------------------------------------------------------------
async def search_flights(
    departure: str,
    arrival: str,
    date: str,
    max_results: int = 5
) -> List[Flight]:
    """
    Asynchronously fetches flights from Google Flights via SerpApi.

    Args:
        departure (str): IATA code (e.g., 'DEL')
        arrival (str): IATA code (e.g., 'BOM')
        date (str): Date in YYYY-MM-DD format
        max_results (int): Limit number of results returned

    Returns:
        List[Flight]: Flights with airline, flight number, timing, duration, and price.

    Raises:
        AirlineAPIError: If the request fails after retries, or if API key is missing.
    """
    # Validate API key at call time (not at import)
    if not SERPAPI_KEY:
        raise AirlineAPIError("SERPAPI_KEY not configured in environment")

    # Start latency measurement
    start = time.monotonic()

    try:
        # Get the circuit breaker for this service
        breaker = await get_circuit_breaker("airline_api")

        params = {
            "engine": "google_flights",
            "departure_id": departure.upper(),
            "arrival_id": arrival.upper(),
            "outbound_date": date,
            "type": "2",  # One-way
            "travel_class": "1",  # Economy
            "adults": "1",
            "hl": "en",
            "gl": "in",
            "currency": "INR",
            "deep_search": "true",
            "api_key": SERPAPI_KEY,
        }

        url = "https://serpapi.com/search"
        MAX_RETRIES = 3

        # Structured log: request start
        logger.info("SerpAPI request started", extra={
            "departure": departure,
            "arrival": arrival,
            "date": date,
        })

        # ------------------------------------------------------------------
        # Inner function that contains the retry loop.
        # Returns a tuple (parsed_results, attempts_used) on success,
        # or raises an exception on final failure.
        # ------------------------------------------------------------------
        async def _make_request_with_retries() -> Tuple[List[Flight], int]:
            """Attempt the request up to MAX_RETRIES times with backoff."""
            last_exception = None

            for attempt in range(MAX_RETRIES):
                attempt_start = time.monotonic()
                try:
                    # Apply rate limiting before each HTTP call
                    await _rate_limit()

                    client = get_client()
                    # NOTE: The shared HTTP client from core.http_client is expected to have timeouts configured.
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()

                    # Check for explicit error in response
                    if "error" in data:
                        error_msg = data["error"]
                        logger.error("SerpAPI returned error", extra={"error": error_msg})
                        raise AirlineAPIError(f"SerpAPI error: {error_msg}")

                    # Parse response
                    flights = data.get("other_flights", [])
                    parsed_results = []
                    for raw in flights[:max_results]:
                        try:
                            # Defensive checks
                            if "flights" not in raw or not raw["flights"]:
                                logger.warning("Missing 'flights' key or empty list", extra={
                                    "departure": departure,
                                    "arrival": arrival,
                                    "date": date,
                                })
                                continue

                            flight_info = raw["flights"][0]

                            # Validate required fields
                            required = ["airline", "flight_number", "duration", "departure_airport", "arrival_airport"]
                            if not all(k in flight_info for k in required):
                                logger.warning("Flight missing required fields", extra={
                                    "missing": [k for k in required if k not in flight_info],
                                })
                                continue

                            # Price must exist
                            price = raw.get("price")
                            if price is None:
                                logger.warning("Missing price in flight result", extra={
                                    "flight": flight_info.get("flight_number"),
                                })
                                continue

                            # Convert price to int if possible
                            try:
                                price_int = int(price)
                            except (ValueError, TypeError):
                                logger.warning("Price is not an integer", extra={"price": price})
                                continue

                            # Parse duration (defensive)
                            duration_min = _parse_duration(flight_info["duration"])
                            if duration_min is None:
                                logger.warning("Could not parse duration", extra={
                                    "duration_raw": flight_info["duration"],
                                })
                                continue

                            flight = Flight(
                                airline=flight_info["airline"],
                                flight_no=flight_info["flight_number"],
                                departure_time=flight_info["departure_airport"]["time"],
                                arrival_time=flight_info["arrival_airport"]["time"],
                                duration_min=duration_min,
                                price_inr=price_int,
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

                    latency = time.monotonic() - attempt_start
                    logger.info("SerpAPI attempt succeeded", extra={
                        "latency_sec": round(latency, 2),
                        "attempt": attempt + 1,
                    })

                    # If we parsed at least one flight, return immediately (success)
                    if parsed_results:
                        return parsed_results, attempt + 1

                    # No results parsed – log and return empty list (this is not a failure)
                    logger.warning("No valid flights parsed", extra={
                        "departure": departure,
                        "arrival": arrival,
                        "date": date,
                    })
                    return parsed_results, attempt + 1

                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    latency = time.monotonic() - attempt_start
                    logger.error("SerpAPI network error", extra={
                        "error_type": type(e).__name__,
                        "attempt": attempt + 1,
                        "latency_sec": round(latency, 2),
                    })
                    # Instrument retry for network errors
                    AIRLINE_RETRIES.labels(reason="network").inc()
                    last_exception = e
                    if attempt == MAX_RETRIES - 1:
                        # Final attempt failed
                        AIRLINE_RETRIES.labels(reason="exhausted").inc()
                        raise AirlineAPIError(f"Retries exhausted for {departure}->{arrival} on {date}") from e
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info("Retry scheduled", extra={"sleep_sec": sleep_time})
                    await asyncio.sleep(sleep_time)

                except httpx.HTTPStatusError as e:
                    latency = time.monotonic() - attempt_start
                    status = e.response.status_code
                    logger.error("SerpAPI HTTP error", extra={
                        "status_code": status,
                        "attempt": attempt + 1,
                        "latency_sec": round(latency, 2),
                    })

                    if status == 429:
                        # Instrument retry for 429
                        AIRLINE_RETRIES.labels(reason="http_429").inc()
                        # Rate limit from SerpAPI – longer backoff
                        if attempt == MAX_RETRIES - 1:
                            AIRLINE_RETRIES.labels(reason="exhausted").inc()
                            raise AirlineAPIError(f"Retries exhausted after 429 for {departure}->{arrival} on {date}") from e
                        sleep_time = 5 * (attempt + 1)  # 5, 10, 15 seconds
                        logger.info("Rate limited (429), retry scheduled", extra={"sleep_sec": sleep_time})
                        await asyncio.sleep(sleep_time)
                    elif 500 <= status < 600:
                        # Instrument retry for 5xx
                        AIRLINE_RETRIES.labels(reason="http_5xx").inc()
                        # Server error – retry with normal backoff
                        if attempt == MAX_RETRIES - 1:
                            AIRLINE_RETRIES.labels(reason="exhausted").inc()
                            raise AirlineAPIError(f"Retries exhausted for {departure}->{arrival} on {date}") from e
                        sleep_time = 2 ** attempt
                        logger.info("Server error, retry scheduled", extra={"sleep_sec": sleep_time})
                        await asyncio.sleep(sleep_time)
                    else:
                        # Client error (4xx except 429) – do not retry
                        raise AirlineAPIError(f"HTTP {status} for {departure}->{arrival} on {date}") from e

            # Should never reach here, but for safety:
            raise AirlineAPIError(f"Unexpected exit from retry loop for {departure}->{arrival} on {date}")

        # Execute the whole retryable operation under circuit breaker protection
        parsed_results, attempts_used = await breaker.call(_make_request_with_retries)


        # Log final success with actual attempts count
        logger.info("SerpAPI final success", extra={
            "results_count": len(parsed_results),
            "attempts": attempts_used,
        })

        # Record the number of attempts taken for this successful request
        AIRLINE_ATTEMPTS.observe(attempts_used)

        # Increment success counter
        TOOL_REQUESTS.labels(tool="airline", status="success").inc()
        return parsed_results

    except Exception:
        # Increment error counter for any exception (including AirlineAPIError)
        TOOL_REQUESTS.labels(tool="airline", status="error").inc()
        raise
    finally:
        # Record latency regardless of success/failure
        TOOL_LATENCY.labels(tool="airline").observe(time.monotonic() - start)

# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------
async def health_check() -> str:
    """
    Performs a minimal test request to verify the airline API is functioning.

    Uses a fixed route (DEL → BOM) for tomorrow's date. Returns "ok" if a successful
    response is received and parsed; "fail" otherwise.
    """
    if not SERPAPI_KEY:
        logger.error("Health check failed: SERPAPI_KEY not configured")
        return "fail"

    # Use tomorrow's date to avoid caching issues
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        # We'll attempt a single request without retries or circuit breaker to keep it simple
        await _rate_limit()  # respect rate limiting even for health checks

        client = get_client()
        params = {
            "engine": "google_flights",
            "departure_id": "DEL",
            "arrival_id": "BOM",
            "outbound_date": tomorrow,
            "type": "2",
            "travel_class": "1",
            "adults": "1",
            "hl": "en",
            "gl": "in",
            "currency": "INR",
            "deep_search": "true",
            "api_key": SERPAPI_KEY,
        }
        response = await client.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            logger.error("Health check failed: API returned error", extra={"error": data["error"]})
            return "fail"

        # Even if no flights are returned, the API responded -> consider it healthy
        logger.info("Health check passed")
        return "ok"

    except Exception as e:
        logger.error("Health check failed", extra={"error": str(e)})
        return "fail"