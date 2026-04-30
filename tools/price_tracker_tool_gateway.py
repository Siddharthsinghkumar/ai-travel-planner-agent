"""Price-tracker-facing gateway (compatibility shim).

This module is kept for backward compatibility. All functionality has been
consolidated into tools/tool_gateway.py.

DEPRECATED: Import directly from tools.tool_gateway instead.
"""

from tools.tool_gateway import (
    TrackingBookingContext,
    get_active_held_bookings,
    expire_held_booking_for_tracking_invalid_data,
    search_with_booking_token,
    search_flights,
    build_booking_handoff_url,
    tracking_context_from_booking,
    select_cheapest_flight,
)

__all__ = [
    "TrackingBookingContext",
    "get_active_held_bookings",
    "expire_held_booking_for_tracking_invalid_data",
    "search_with_booking_token",
    "search_flights",
    "build_booking_handoff_url",
    "tracking_context_from_booking",
    "select_cheapest_flight",
]