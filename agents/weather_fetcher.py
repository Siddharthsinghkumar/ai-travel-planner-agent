"""Weather fetcher - handles weather data fetching and display normalization."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _normalize_weather_for_display(weather_value: Any, requested_location: Optional[str] = None) -> Any:
    """
    Keep weather payload shape intact and attach display labels without
    altering source numeric weather values.
    """
    from core.iata_resolver import city_for_iata, label_for_iata

    if weather_value is None or isinstance(weather_value, Exception):
        return weather_value

    payload: Any = weather_value
    if isinstance(weather_value, dict):
        payload = dict(weather_value)
    elif hasattr(weather_value, "model_dump"):
        payload = weather_value.model_dump()
    elif hasattr(weather_value, "to_dict"):
        payload = weather_value.to_dict()
    elif hasattr(weather_value, "__dict__"):
        payload = dict(vars(weather_value))

    if not isinstance(payload, dict):
        return weather_value

    if requested_location and not payload.get("location"):
        payload["location"] = requested_location

    from agents.intent_parser import _sanitize_iata_code
    normalized_location = _sanitize_iata_code(str(requested_location or payload.get("location") or ""))
    if normalized_location:
        payload["location"] = normalized_location
        location_city = city_for_iata(normalized_location)
        location_label = label_for_iata(normalized_location) or normalized_location
        if location_city:
            payload["location_city"] = location_city
        payload["location_label"] = location_label

    return payload


async def get_weather_once(
    location: str,
    travel_date: str,
    cached_weather: Any,
    weather_cache: dict,
) -> Any:
    """Fetch weather for a location with caching."""
    cache_key = f"{location}_{travel_date}"
    if cache_key in weather_cache:
        return weather_cache[cache_key]

    from core.iata_resolver import city_for_iata
    from agents.intent_parser import _sanitize_iata_code

    async def _fetch_weather() -> Any:
        weather_query_location = location
        normalized_loc = _sanitize_iata_code(location)
        if normalized_loc:
            weather_query_location = city_for_iata(normalized_loc) or location
        result = await cached_weather(location=weather_query_location, travel_date=travel_date, units="metric")
        normalized = _normalize_weather_for_display(result, requested_location=location)
        weather_cache[cache_key] = normalized
        return normalized

    return await _fetch_weather()