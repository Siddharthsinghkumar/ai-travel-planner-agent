"""Canonical tool gateway - single source of truth for tool-facing functions.

This module consolidates duplicate gateway wrappers from:
- agents/planner_tool_gateway.py
- tools/price_tracker_tool_gateway.py

It provides lazy imports to avoid circular dependencies while maintaining
a clean import surface for both planner and price tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TrackingBookingContext:
    """Context for tracking a booking in price monitoring."""
    booking_id: Optional[int]
    origin: str
    destination: str
    travel_date: str
    held_price: Optional[float]
    booking_token: str


# ----------------------------------------------------------------------
# Airline tool functions
# ----------------------------------------------------------------------

async def search_flights(*args, **kwargs):
    """Search for flights via SerpAPI."""
    from tools.airline_api import search_flights as impl
    return await impl(*args, **kwargs)


async def search_with_booking_token(*args, **kwargs):
    """Search for flights using a booking token."""
    from tools.airline_api import search_with_booking_token as impl
    return await impl(*args, **kwargs)


# ----------------------------------------------------------------------
# Booking handoff functions
# ----------------------------------------------------------------------

async def build_booking_handoff_url(*args, **kwargs):
    """Build booking handoff URL from a booking token."""
    from tools.booking_handoff import build_booking_handoff_url as impl
    return await impl(*args, **kwargs)


def get_active_held_bookings():
    """Get all currently held bookings."""
    from tools.booking_handoff import get_active_held_bookings as impl
    return impl()


def expire_held_booking_for_tracking_invalid_data(*args, **kwargs):
    """Expire a held booking due to invalid tracking data."""
    from tools.booking_handoff import expire_held_booking_for_tracking_invalid_data as impl
    return impl(*args, **kwargs)


def booking_resolution_cache_stats() -> dict:
    """Get booking resolution cache statistics."""
    from tools.booking_handoff import booking_resolution_cache_stats as impl
    return impl()


from agents.high_impact import high_impact


@high_impact("booking")
async def hold_booking(*args, **kwargs):
    """Hold a booking for later completion."""
    from tools.booking_handoff import hold_booking as impl
    return await impl(*args, **kwargs)


@high_impact("cancel")
def cancel_booking(*args, **kwargs):
    """Cancel a held booking."""
    from tools.booking_handoff import cancel_booking as impl
    return impl(*args, **kwargs)


# ----------------------------------------------------------------------
# Weather tool functions
# ----------------------------------------------------------------------

def check_weather(*args, **kwargs):
    """Check current weather for a location."""
    from tools.weather_api import check_weather as impl
    return impl(*args, **kwargs)


def get_forecast_for_date(*args, **kwargs):
    """Get weather forecast for a specific date."""
    from tools.weather_api import get_forecast_for_date as impl
    return impl(*args, **kwargs)


# ----------------------------------------------------------------------
# Price tracker functions
# ----------------------------------------------------------------------

def record_price_snapshot(*args, **kwargs) -> int:
    """Record a price snapshot for a route."""
    from tools.price_tracker import record_price_snapshot as impl
    return impl(*args, **kwargs)


def parse_price_insights(*args, **kwargs):
    """Parse price insights from SerpAPI response."""
    from tools.price_tracker import parse_price_insights as impl
    return impl(*args, **kwargs)


def format_price_insights_for_llm(*args, **kwargs) -> str:
    """Format price insights for LLM consumption."""
    from tools.price_tracker import format_price_insights_for_llm as impl
    return impl(*args, **kwargs)


def analyze_price_trend(*args, **kwargs):
    """Analyze price trend for a route."""
    from tools.price_tracker import analyze_price_trend as impl
    return impl(*args, **kwargs)


def predict_future_price(*args, **kwargs) -> Optional[float]:
    """Predict future price using trend analysis."""
    from tools.price_tracker import predict_future_price as impl
    return impl(*args, **kwargs)


async def record_flight_data(*args, **kwargs):
    """Record flight data for price tracking."""
    from tools.price_tracker import record_flight_data as impl
    return await impl(*args, **kwargs)


# ----------------------------------------------------------------------
# Helper functions for price tracker
# ----------------------------------------------------------------------

def tracking_context_from_booking(booking: dict) -> TrackingBookingContext:
    """Extract tracking context from a booking dict."""
    flight = booking.get("flight", {}) if isinstance(booking.get("flight"), dict) else {}

    booking_id: Optional[int] = None
    raw_booking_id = booking.get("id")
    try:
        if raw_booking_id is not None:
            booking_id = int(raw_booking_id)
    except Exception:
        booking_id = None

    origin = str(flight.get("origin") or flight.get("departure_iata") or "").strip().upper()
    destination = str(flight.get("destination") or flight.get("arrival_iata") or "").strip().upper()
    travel_date = str(flight.get("date") or "").strip()
    booking_token = str(flight.get("booking_token") or "").strip()

    held_price = None
    raw_price = flight.get("price_inr")
    if isinstance(raw_price, (int, float)):
        held_price = float(raw_price)
    elif isinstance(raw_price, str):
        cleaned = raw_price.replace("₹", "").replace(",", "").strip()
        try:
            held_price = float(cleaned)
        except Exception:
            held_price = None

    return TrackingBookingContext(
        booking_id=booking_id,
        origin=origin,
        destination=destination,
        travel_date=travel_date,
        held_price=held_price,
        booking_token=booking_token,
    )


def select_cheapest_flight(search_result: Any):
    """Select the cheapest flight from a search result."""
    if isinstance(search_result, list):
        flights = search_result
    elif isinstance(search_result, tuple) and search_result:
        candidate = search_result[0]
        flights = candidate if isinstance(candidate, list) else []
    else:
        flights = []
    if not flights:
        return None
    return min(flights, key=lambda flight: flight.price_inr)