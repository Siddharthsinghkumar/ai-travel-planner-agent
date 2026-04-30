"""Booking service layer.

Provides a clean boundary between API layer and booking_handoff tool.
Handles booking hold, cancel, and expiry with cache invalidation hooks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_booking_tool = None


def _get_booking_tool():
    """Lazy load booking tool."""
    global _booking_tool
    if _booking_tool is None:
        from tools.booking_handoff import hold_booking, cancel_booking, expire_held_booking_for_tracking_invalid_data
        _booking_tool = type("BookingTool", (), {
            "hold_booking": hold_booking,
            "cancel_booking": cancel_booking,
            "expire_held_booking": expire_held_booking_for_tracking_invalid_data,
        })()
    return _booking_tool


from agents.high_impact import high_impact


@high_impact("booking")
async def hold_booking_service(
    flight: Dict[str, Any],
    origin: str,
    destination: str,
    depart_date: str,
    return_date: Optional[str] = None,
    passengers: int = 1,
    held_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hold a booking for later completion.
    
    On hold, invalidate any related price tracker caches.
    """
    tool = _get_booking_tool()
    
    try:
        result = await tool.hold_booking(
            flight=flight,
            origin=origin,
            destination=destination,
            depart_date=depart_date,
            return_date=return_date,
            passengers=passengers,
            held_by=held_by,
        )
        
        # Invalidate related caches on successful hold
        await _invalidate_price_tracker_caches(origin, destination, depart_date)
        
        return result
    except Exception as e:
        logger.error(f"hold_booking_service failed: {e}")
        raise


@high_impact("cancel")
async def cancel_booking_service(booking_id: int) -> Dict[str, Any]:
    """
    Cancel a held booking.
    
    On cancel, invalidate any related price tracker caches.
    """
    tool = _get_booking_tool()
    
    try:
        result = await tool.cancel_booking(booking_id=booking_id)
        
        # Extract booking details for cache invalidation
        if result and "flight" in result:
            flight = result.get("flight", {})
            origin = flight.get("origin", "")
            destination = flight.get("destination", "")
            depart_date = flight.get("date", "")
            
            if origin and destination and depart_date:
                await _invalidate_price_tracker_caches(origin, destination, depart_date)
        
        return result
    except Exception as e:
        logger.error(f"cancel_booking_service failed: {e}")
        raise


async def _invalidate_price_tracker_caches(origin: str, destination: str, travel_date: str) -> None:
    """
    Invalidate price tracker caches for a route.
    
    This is the clean connection between booking lifecycle
    and price tracker caches.
    """
    try:
        from core.cache import notify_invalidation
        # Notify price tracker flight data cache
        cache_key = f"{origin.upper()}|{destination.upper()}|{travel_date}"
        notify_invalidation("price_tracker_flight_data", cache_key)
        logger.debug(f"Invalidated price tracker cache for {cache_key}")
    except Exception as e:
        logger.debug(f"Cache invalidation failed (non-blocking): {e}")


async def get_booking_resolution_service(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: Optional[str] = None,
    passengers: int = 1,
    flight: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get booking resolution through service layer."""
    tool = _get_booking_tool()
    return await tool.get_booking_resolution(
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        return_date=return_date,
        passenger_count=passengers,
        flight=flight,
    )