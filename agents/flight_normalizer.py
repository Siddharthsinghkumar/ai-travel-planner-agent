"""Flight normalizer - normalizes flight data from various sources."""

import logging
from typing import Any, List, Optional, Union
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class Flight(BaseModel):
    """Validated flight data model."""
    airline: str
    flight_no: str
    departure_time: str
    arrival_time: str
    duration_min: int
    price_inr: Any
    price_unavailable: bool = False
    stops: Any = 0
    layover_info: str = ""
    layover_durations_min: Optional[List[int]] = None
    layover_airports: Optional[List[str]] = None
    baggage: str = "Check airline"
    booking_token: Optional[str] = None
    shareable_link: Optional[str] = None
    date: Optional[str] = None
    travel_class: Optional[str] = None


def normalize_flights(raw_flights: List[Any], default_date: str) -> List[Flight]:
    """Normalize a list of flights into Flight objects."""
    normalized = []
    for f in raw_flights:
        if isinstance(f, dict):
            flight_data = dict(f)
        elif hasattr(f, "__dict__"):
            flight_data = dict(vars(f))
        else:
            logger.debug(f"Skipping unknown flight type: {type(f)}")
            continue

        if 'date' not in flight_data or not flight_data.get('date'):
            flight_data['date'] = default_date

        try:
            normalized.append(Flight(**flight_data))
        except ValidationError as e:
            logger.debug(f"Skipping invalid flight after conversion: {e}")

    return normalized


def normalize_flight_field(value: Any) -> str:
    """Convert flight field to a normalized string for matching."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return str(value).strip().lower()


def normalize_airport(text: Optional[str]) -> Optional[str]:
    """Normalize a free-text airport/city token into an IATA code or return None."""
    if not text:
        return None
    from core.iata_resolver import resolve_location
    tok = text.strip()
    if len(tok) == 3 and tok.isalpha():
        from core.iata_resolver import is_iata_token
        if is_iata_token(tok.upper()):
            return tok.upper()
    try:
        iata = resolve_location(text)
    except Exception:
        iata = None
    if iata:
        return iata
    return None