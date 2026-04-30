"""Flight service layer.

Provides a clean boundary between API layer and airline_api tool.
Handles flight search with unified caching and error handling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_airline_tool = None


def _get_airline_tool():
    """Lazy load airline tool."""
    global _airline_tool
    if _airline_tool is None:
        from tools.airline_api import search_flights, search_with_booking_token
        _airline_tool = type("AirlineTool", (), {
            "search_flights": search_flights,
            "search_with_booking_token": search_with_booking_token,
        })()
    return _airline_tool


async def search_flights_service(
    origin: str,
    destination: str,
    date: str,
    cabin: Optional[str] = None,
    adults: int = 1,
    infants: int = 0,
    children: int = 0,
    return_date: Optional[str] = None,
    stops: Optional[int] = None,
    max_price: Optional[int] = None,
    carriers: Optional[List[str]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Search for flights through service layer.
    
    Returns (flights, meta) tuple.
    """
    tool = _get_airline_tool()
    
    try:
        result = await tool.search_flights(
            origin=origin,
            destination=destination,
            date=date,
            cabin=cabin,
            adults=adults,
            infants=infants,
            children=children,
            return_date=return_date,
            stops=stops,
            max_price=max_price,
            carriers=carriers,
        )
        
        # Handle tuple return (flights, meta) or just flights
        if isinstance(result, tuple):
            flights, meta = result[0], result[1] if len(result) > 1 else {}
        else:
            flights, meta = result, {}
            
        return flights, meta
        
    except Exception as e:
        logger.error(f"search_flights_service failed: {e}")
        raise


async def search_with_booking_token_service(
    token: str,
    origin: str,
    destination: str,
    date: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Search flights using booking token through service layer."""
    tool = _get_airline_tool()
    
    try:
        result = await tool.search_with_booking_token(
            token=token,
            origin=origin,
            destination=destination,
            date=date,
        )
        
        if isinstance(result, tuple):
            flights, meta = result[0], result[1] if len(result) > 1 else {}
        else:
            flights, meta = result, {}
            
        return flights, meta
        
    except Exception as e:
        logger.error(f"search_with_booking_token_service failed: {e}")
        raise