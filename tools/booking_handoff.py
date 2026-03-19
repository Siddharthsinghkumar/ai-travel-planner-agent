# tools/booking_handoff.py
"""
Booking Handoff Tool

Responsibilities:
- Hold, confirm, cancel, and expire booking records in the database
- Resolve the best possible deep-link for a flight using (in priority order):
    1. SerpAPI booking_token  → calls /search?engine=google_flights_booking to get
                                airline-native checkout URL (exact itinerary, best UX)
    2. shareable_link         → direct Google Flights shareable link (still pre-filled)
    3. Google Flights fallback→ clean HTTPS search URL (last resort, no guessing)

NOTE: The old AIRLINE_BOOKING_URLS dict that guessed airline homepages is removed.
      Those URLs just dumped the user on a generic search page, which is worse than
      the Google Flights fallback. The SerpAPI token gives us the real checkout page.
"""

import asyncio
import logging
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import httpx
from sqlalchemy import Column, Integer, String, JSON, DateTime, Text
from agents.database import Base, SessionLocal, get_engine
from core.api_key_manager import key_manager as api_key_manager

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Database model
# ----------------------------------------------------------------------

class Booking(Base):
    """
    Represents a single flight booking record.

    Status lifecycle:
        HELD  →  CONFIRMED  (user confirmed within hold window)
        HELD  →  EXPIRED    (hold_minutes elapsed without confirmation)
        HELD  →  CANCELLED  (user explicitly cancelled before expiry)
    """
    __tablename__ = "bookings"

    id           = Column(Integer, primary_key=True, index=True)
    status       = Column(String,  nullable=False)           # HELD | CONFIRMED | CANCELLED | EXPIRED
    flight       = Column(JSON,    nullable=False)           # Full flight dict (includes booking_token, shareable_link)
    passenger    = Column(JSON,    nullable=True)            # Passenger info dict (name, DOB, passport…)
    booking_token = Column(Text,   nullable=True)            # SerpAPI booking_token (top-level, for quick access)
    shareable_link = Column(Text,  nullable=True)            # SerpAPI shareable_link (top-level, for quick access)
    handoff_url  = Column(Text,    nullable=True)            # Resolved deep-link written at hold time
    created_at   = Column(DateTime, default=datetime.utcnow)
    expires_at   = Column(DateTime, nullable=True)


def ensure_tables():
    """Create any missing tables (safe to call multiple times)."""
    Base.metadata.create_all(bind=get_engine())

ensure_tables()


# ----------------------------------------------------------------------
# SerpAPI booking resolution
# ----------------------------------------------------------------------

SERPAPI_BOOKING_ENDPOINT = "https://serpapi.com/search"

async def fetch_booking_options(booking_token: str) -> Optional[List[Dict[str, Any]]]:
    """
    Call SerpAPI's google_flights_booking engine with the given token.

    Returns a list of booking options, each containing:
        - provider (str): name of the booking site (e.g., "Expedia")
        - price (float): total price in the response currency
        - link (str): direct URL to complete the booking

    Returns None if the request fails or no options are found.
    Handles 401/429 (and any non-200) by returning None so the caller can fall back.
    """
    try:
        async with api_key_manager.reserve_key("serpapi") as (idx, key):
            params = {
                "engine": "google_flights_booking",
                "booking_token": booking_token,
                "api_key": key,
                "hl": "en",
                "gl": "in",
                "currency": "INR",
            }
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(SERPAPI_BOOKING_ENDPOINT, params=params)

            if resp.status_code != 200:
                logger.warning(
                    "SerpAPI booking options fetch failed",
                    extra={"status": resp.status_code, "token_preview": booking_token[:30]}
                )
                return None

            data = resp.json()

            if "error" in data:
                logger.warning("SerpAPI booking error", extra={"error": data["error"]})
                return None

            options = []
            for opt in data.get("booking_options", []):
                # Extract provider name, price, and link
                provider = opt.get("name")
                price = opt.get("price")
                link = opt.get("link") or opt.get("url")  # some responses use "link", some "url"
                if provider and price is not None and link:
                    # Convert price to float if it's a string; SerpAPI usually returns numeric
                    try:
                        price_float = float(price)
                    except (TypeError, ValueError):
                        logger.debug(f"Skipping option with non-numeric price: {price}")
                        continue
                    options.append({
                        "provider": provider,
                        "price": price_float,
                        "link": link,
                    })

            if not options:
                logger.warning("No valid booking options found in response")
                return None

            logger.info(f"Fetched {len(options)} booking options from SerpAPI")
            return options

    except Exception as e:
        logger.warning("fetch_booking_options exception", extra={"error": str(e)})
        return None


async def best_booking_option(booking_token: str) -> Optional[Dict[str, Any]]:
    """
    Fetch all booking options and return the cheapest one.

    Returns a dict with 'provider', 'price', 'link' keys, or None if no options.
    """
    options = await fetch_booking_options(booking_token)
    if not options:
        return None
    # Select option with minimum price
    best = min(options, key=lambda x: x["price"])
    logger.info(f"Best booking option: {best['provider']} at {best['price']}")
    return best


async def resolve_booking_token(booking_token: str) -> Optional[str]:
    """
    Return the direct booking URL from the cheapest option found via SerpAPI.
    Returns None if resolution fails (including 401/429 or no options).
    """
    best = await best_booking_option(booking_token)
    return best["link"] if best else None


def build_google_flights_fallback(
    *,
    origin: str,
    destination: str,
    depart_date: str,
    return_date: Optional[str] = None,
) -> str:
    """
    Build a clean, HTTPS Google Flights search URL as a last-resort fallback.

    Uses the /travel/flights?q= format which is stable and works without
    any special parameters.
    """
    # Human-readable query string that Google Flights understands well
    if return_date:
        query = f"Flights from {origin} to {destination} on {depart_date} returning {return_date}"
    else:
        query = f"Flights from {origin} to {destination} on {depart_date}"

    return f"https://www.google.com/travel/flights?q={urllib.parse.quote(query)}"


async def build_booking_handoff_url(
    *,
    flight: dict,
    origin: str,
    destination: str,
    depart_date: str,
    return_date: Optional[str] = None,
    passengers: int = 1,
) -> str:
    """
    Resolve the best possible booking deep-link for a flight, in priority order:

    1. SerpAPI booking_token  → calls the SerpAPI booking engine to get the
                                cheapest airline-native checkout URL.
    2. shareable_link         → the SerpAPI-provided shareable Google Flights link
                                which pre-fills the exact itinerary.
    3. Google Flights fallback→ a clean /travel/flights?q= search URL.

    Args:
        flight: Full flight dict from planner (must include 'booking_token'
                and/or 'shareable_link' keys populated by airline_api.py).
        origin, destination: IATA codes.
        depart_date: YYYY-MM-DD.
        return_date: YYYY-MM-DD or None for one-way.
        passengers: Number of adult passengers.

    Returns:
        str: The best available booking URL.
    """
    # ── Priority 1: SerpAPI booking_token ────────────────────────────────
    booking_token = flight.get("booking_token")
    if booking_token:
        resolved = await resolve_booking_token(booking_token)
        if resolved:
            logger.info("Handoff URL resolved via SerpAPI booking_token (cheapest option)")
            return resolved
        logger.info("booking_token resolution unavailable; falling through to shareable_link/google fallback")

    # ── Priority 2: shareable_link ───────────────────────────────────────
    shareable_link = flight.get("shareable_link")
    if shareable_link:
        logger.info("Handoff URL resolved via shareable_link")
        return shareable_link

    # ── Priority 3: Google Flights fallback ──────────────────────────────
    logger.info("Handoff URL falling back to Google Flights search URL")
    return build_google_flights_fallback(
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        return_date=return_date,
    )


# ----------------------------------------------------------------------
# Core booking CRUD
# ----------------------------------------------------------------------

async def hold_booking(
    *,
    flight: dict,
    origin: str,
    destination: str,
    depart_date: str,
    passenger: Optional[dict] = None,
    hold_minutes: int = 15,
    return_date: Optional[str] = None,
    passengers: int = 1,
) -> dict:
    """
    Create a HELD booking record and resolve + store the best handoff URL.

    Returns a dict with:
        id            – database booking id
        status        – "HELD"
        handoff_url   – resolved deep-link for the user to complete purchase
        expires_at    – ISO-8601 string of when the hold expires
    """
    # Resolve the URL before writing to DB so it's immediately available
    handoff_url = await build_booking_handoff_url(
        flight=flight,
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        return_date=return_date,
        passengers=passengers,
    )

    def _db_operations():
        db = SessionLocal()
        try:
            booking = Booking(
                status="HELD",
                flight=flight,
                passenger=passenger,
                booking_token=flight.get("booking_token"),
                shareable_link=flight.get("shareable_link"),
                handoff_url=handoff_url,
                expires_at=datetime.utcnow() + timedelta(minutes=hold_minutes),
            )
            db.add(booking)
            db.commit()
            db.refresh(booking)

            logger.info(
                "Booking held",
                extra={
                    "booking_id": booking.id,
                    "flight_no": flight.get("flight_no"),
                    "handoff_url_source": (
                        "serpapi_token" if flight.get("booking_token") else
                        "shareable_link" if flight.get("shareable_link") else
                        "google_fallback"
                    ),
                }
            )

            return booking.id, booking.status, booking.expires_at.isoformat()
        finally:
            db.close()

    b_id, b_status, b_expires = await asyncio.to_thread(_db_operations)

    return {
        "id": b_id,
        "status": b_status,
        "handoff_url": handoff_url,
        "expires_at": b_expires,
    }


def get_booking(booking_id: int) -> Optional[dict]:
    """Retrieve a booking by id. Returns None if not found."""
    db = SessionLocal()
    try:
        b = db.get(Booking, booking_id)
        if not b:
            return None
        return {
            "id":             b.id,
            "status":         b.status,
            "flight":         b.flight,
            "passenger":      b.passenger,
            "booking_token":  b.booking_token,
            "shareable_link": b.shareable_link,
            "handoff_url":    b.handoff_url,
            "created_at":     b.created_at.isoformat() if b.created_at else None,
            "expires_at":     b.expires_at.isoformat() if b.expires_at else None,
        }
    finally:
        db.close()


def confirm_booking(booking_id: int) -> bool:
    """
    Confirm a HELD booking. Returns False if the hold has expired or the
    booking is not in HELD status.
    """
    db = SessionLocal()
    try:
        b = db.get(Booking, booking_id)
        if not b:
            return False
        if b.expires_at and b.expires_at < datetime.utcnow():
            b.status = "EXPIRED"
            db.commit()
            logger.info("Booking auto-expired during confirm attempt", extra={"booking_id": booking_id})
            return False
        if b.status != "HELD":
            return False
        b.status = "CONFIRMED"
        db.commit()
        logger.info("Booking confirmed", extra={"booking_id": booking_id})
        return True
    finally:
        db.close()


def cancel_booking(booking_id: int) -> bool:
    """
    Cancel a HELD booking. Cannot cancel a CONFIRMED or EXPIRED booking.
    """
    db = SessionLocal()
    try:
        b = db.get(Booking, booking_id)
        if not b:
            return False
        if b.status in ("CONFIRMED", "EXPIRED"):
            return False
        b.status = "CANCELLED"
        db.commit()
        logger.info("Booking cancelled", extra={"booking_id": booking_id})
        return True
    finally:
        db.close()


def expire_bookings() -> int:
    """
    Bulk-expire all HELD bookings whose hold window has elapsed.
    Intended to be called by a periodic scheduler (e.g. every 5 minutes).

    Returns:
        int: Number of bookings transitioned to EXPIRED.
    """
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        stale = (
            db.query(Booking)
            .filter(Booking.status == "HELD", Booking.expires_at < now)
            .all()
        )
        count = len(stale)
        for b in stale:
            b.status = "EXPIRED"
        db.commit()
        if count:
            logger.info("Bulk-expired stale bookings", extra={"count": count})
        return count
    finally:
        db.close()


def get_active_held_bookings() -> list[dict]:
    """
    Return all HELD bookings that have not yet expired.
    Used by the price tracker to know which routes to monitor.
    """
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        rows = (
            db.query(Booking)
            .filter(Booking.status == "HELD", Booking.expires_at > now)
            .all()
        )
        return [
            {
                "id":            b.id,
                "flight":        b.flight,
                "booking_token": b.booking_token,
                "handoff_url":   b.handoff_url,
                "expires_at":    b.expires_at.isoformat() if b.expires_at else None,
            }
            for b in rows
        ]
    finally:
        db.close()
