# tools/price_tracker.py
"""
Price Tracker Tool

Two distinct capabilities:

1. Instant Price Intelligence  (parse_price_insights / format_price_insights_for_llm)
   ─────────────────────────────────────────────────────────────────────────────────
   SerpAPI's Google Flights response usually contains a `price_insights` block that
   compares the current price against the typical range for that route.  This module
   parses that block and formats it into a short, plain-English string that the
   planner agent can inject directly into the LLM prompt.

   Example LLM output enabled by this:
   "This IndiGo flight at ₹4,500 is currently BELOW the typical range of
    ₹5,800–₹7,200 for this route.  Prices have been rising — booking now is advised."

2. Background Price-Drop Monitoring  (record_price_snapshot / check_held_booking_prices)
   ──────────────────────────────────────────────────────────────────────────────────────
   Periodically re-queries SerpAPI for routes with active HELD bookings and logs
   price snapshots to the DB.  If the current price drops below the held price by
   more than a configurable threshold, an alert is written so the user can be notified.

   Intended to be scheduled via core/job_queue.py every N minutes.
"""

import asyncio
import logging
import os
import time
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Boolean, Text
from agents.database import Base, SessionLocal, get_engine

# Optional numpy for linear regression forecast
try:
    import numpy as np
except ImportError:
    np = None

# Local tools — imported lazily inside async functions to avoid circular imports
# from tools.airline_api import search_flights, search_with_booking_token, AirlineAPIError
# from tools.booking_handoff import get_active_held_bookings, fetch_booking_options

logger = logging.getLogger(__name__)

# ── Configurable thresholds ────────────────────────────────────────────────────
PRICE_DROP_ALERT_THRESHOLD_PCT = float(
    os.getenv("PRICE_DROP_ALERT_THRESHOLD_PCT", "5")
)  # alert when price drops ≥5% below held price
SNAPSHOT_RETENTION_DAYS = int(os.getenv("PRICE_SNAPSHOT_RETENTION_DAYS", "30"))
CHECK_COOLDOWN_SECONDS = int(os.getenv("PRICE_CHECK_COOLDOWN_SECONDS", "1800"))  # 30 minutes

# Bounded LRU cache for booking token throttling
THROTTLE_MAX_SIZE = 5000
_last_checked = OrderedDict()


def _mark_checked(token: str, now: float) -> None:
    """
    Record that a booking token was checked at time `now`.
    Maintains LRU order: moves token to end (most recent) and evicts oldest if size exceeded.
    """
    if token in _last_checked:
        _last_checked.move_to_end(token)
    _last_checked[token] = now
    while len(_last_checked) > THROTTLE_MAX_SIZE:
        _last_checked.popitem(last=False)  # remove least recently used


def _coerce_flights_result(result: Any) -> list:
    """
    Normalize airline search results to a list of flight-like objects.
    Supports:
    - list[Flight]
    - tuple/list payload where first item is list[Flight]
    """
    if isinstance(result, list):
        return result
    if isinstance(result, tuple) and result:
        flights = result[0]
        if isinstance(flights, list):
            return flights
    return []


# ----------------------------------------------------------------------
# Database model
# ----------------------------------------------------------------------

class PriceSnapshot(Base):
    """
    One price observation for a specific route on a specific date.

    Populated by the background job and used to:
    - Build a price history chart / trend sentence for the LLM
    - Detect price drops on active HELD bookings
    """
    __tablename__ = "price_snapshots"

    id              = Column(Integer, primary_key=True, index=True)
    origin          = Column(String(3),  nullable=False, index=True)
    destination     = Column(String(3),  nullable=False, index=True)
    travel_date     = Column(String(10), nullable=False, index=True)   # YYYY-MM-DD
    observed_at     = Column(DateTime,   default=datetime.utcnow, index=True)
    price_inr       = Column(Float,      nullable=False)               # cheapest found
    typical_low_inr = Column(Float,      nullable=True)                # from price_insights
    typical_high_inr= Column(Float,      nullable=True)                # from price_insights
    price_level     = Column(String(20), nullable=True)                # low | typical | high
    trend           = Column(String(20), nullable=True)                # rising | stable | falling
    insights_raw    = Column(JSON,       nullable=True)                # full price_insights blob
    alert_sent      = Column(Boolean,    default=False)                # True once drop alert fired


class PriceDropAlert(Base):
    """
    Records a detected price drop on an active HELD booking.
    The application layer can query this table to push notifications.
    """
    __tablename__ = "price_drop_alerts"

    id              = Column(Integer, primary_key=True, index=True)
    booking_id      = Column(Integer, nullable=False, index=True)      # FK → bookings.id
    origin          = Column(String(3),  nullable=False)
    destination     = Column(String(3),  nullable=False)
    travel_date     = Column(String(10), nullable=False)
    held_price_inr  = Column(Float,      nullable=False)               # price when hold was created
    new_price_inr   = Column(Float,      nullable=False)               # price detected now
    drop_pct        = Column(Float,      nullable=False)               # e.g. 8.3  (meaning 8.3% cheaper)
    new_handoff_url = Column(Text,       nullable=True)                # URL for the cheaper option
    created_at      = Column(DateTime,   default=datetime.utcnow)
    acknowledged    = Column(Boolean,    default=False)


def ensure_tables():
    Base.metadata.create_all(bind=get_engine())


ensure_tables()


# ----------------------------------------------------------------------
# 1. Instant Price Intelligence
# ----------------------------------------------------------------------

@dataclass
class PriceInsights:
    """Parsed representation of SerpAPI's price_insights block, enriched with historical analysis."""
    price_level: str          # "low" | "typical" | "high" — relative to historical average (computed)
    typical_low_inr: Optional[float]
    typical_high_inr: Optional[float]
    current_price_inr: Optional[float]
    trend: Optional[str]      # "rising" | "stable" | "falling" — computed from recent history
    avg_price: Optional[float]          # average price from available history
    forecast: Optional[str]             # simple prediction: "prices likely to increase/drop/stable"
    raw: dict                 # original block for debugging / storage


def _parse_price_insights_text(text: str) -> Optional[list]:
    """
    Attempt to extract price history from a textual summary like:
    "Prices for this route have ranged from ₹5000 to ₹7000 over the last 30 days."

    Returns a list of [timestamp, price] entries (with artificial timestamps) or None.
    """
    if not text:
        return None

    # Look for numbers with ₹ or INR prefixes
    numbers = re.findall(r'[₹]?\s*(\d{1,3}(?:,\d{3})*)\s*(?:-|to)\s*[₹]?\s*(\d{1,3}(?:,\d{3})*)', text)
    if numbers:
        # Take the first pair as low/high
        low_str, high_str = numbers[0]
        try:
            low = float(low_str.replace(',', ''))
            high = float(high_str.replace(',', ''))
        except ValueError:
            return None

        # Create a dummy history with two points: one 30 days ago at low, one now at high
        now = time.time() * 1000  # ms
        thirty_days_ago = now - 30 * 24 * 3600 * 1000
        return [
            [thirty_days_ago, low],
            [now, high]
        ]

    # If no range found, look for a single number (maybe current price) and return None
    return None


def _analyze_price_trend(price_history: list) -> dict:
    """
    Compute price level, trend, and average from a price_history list.

    Args:
        price_history: list of [timestamp, price] entries

    Returns:
        dict with keys: price_level (low/typical/high), trend (rising/stable/falling),
                        avg_price (float), current_price (float)
    """
    if not price_history:
        return {"price_level": "unknown", "trend": "unknown", "avg_price": None, "current_price": None}

    prices = [float(p[1]) for p in price_history if len(p) >= 2]
    if len(prices) < 2:
        return {"price_level": "unknown", "trend": "unknown", "avg_price": None, "current_price": prices[-1] if prices else None}

    avg_price = sum(prices) / len(prices)
    current_price = prices[-1]

    # Determine price level relative to average
    if current_price < avg_price * 0.9:
        price_level = "low"
    elif current_price > avg_price * 1.15:
        price_level = "high"
    else:
        price_level = "typical"

    # Determine trend using available history.
    trend = "stable"
    if len(prices) >= 2:
        baseline = prices[-3] if len(prices) >= 3 else prices[-2]
        if prices[-1] > baseline:
            trend = "rising"
        elif prices[-1] < baseline:
            trend = "falling"

    return {
        "price_level": price_level,
        "trend": trend,
        "avg_price": avg_price,
        "current_price": current_price,
    }


def _predict_future_price(price_history: list) -> str:
    """
    Use linear regression on price_history to forecast direction.

    Returns:
        string: "prices likely to increase" | "prices likely to drop" | "stable prices" | "unknown"
    """
    if np is None:
        return "unknown (numpy not installed)"

    if not price_history or len(price_history) < 2:
        return "unknown (insufficient data)"

    prices = [float(p[1]) for p in price_history if len(p) >= 2]
    if len(prices) < 2:
        return "unknown"

    x = np.arange(len(prices))
    y = np.array(prices)
    try:
        slope = np.polyfit(x, y, 1)[0]
    except Exception:
        return "unknown"

    if slope > 0.5:   # small positive slope still considered stable
        return "prices likely to increase"
    if slope < -0.5:
        return "prices likely to drop"
    return "stable prices"


def parse_price_insights(
    serpapi_response: dict,
    current_price_inr: Optional[int] = None,
) -> Optional[PriceInsights]:
    """
    Parse the `price_insights` block from a raw SerpAPI Google Flights response.

    SerpAPI returns something like:
    {
      "price_insights": {
        "lowest_price": 4200,
        "price_level": "low",            # "low" | "typical" | "high"
        "typical_price_range": [5500, 7200],
        "price_history": [[timestamp, price], …],
        "price_insights_text": "Prices for this route have ranged from ₹5000 to ₹7000 over the last 30 days."
      }
    }

    Args:
        serpapi_response: The full JSON dict returned by SerpAPI.
        current_price_inr: Override for the current price (e.g. from Flight.price_inr)
                           if not already in the insights block.

    Returns:
        PriceInsights dataclass, or None if the block is absent / malformed.
    """
    raw = serpapi_response.get("price_insights")
    if not raw or not isinstance(raw, dict):
        return None

    try:
        # Extract SerpAPI fields (fallbacks)
        price_level_serp = str(raw.get("price_level", "typical")).lower()
        if price_level_serp not in ("low", "typical", "high"):
            price_level_serp = "typical"

        typical_range = raw.get("typical_price_range")
        typical_low: Optional[float] = None
        typical_high: Optional[float] = None
        if isinstance(typical_range, (list, tuple)) and len(typical_range) >= 2:
            typical_low  = float(typical_range[0])
            typical_high = float(typical_range[1])

        current = current_price_inr
        if current is None:
            raw_current = raw.get("lowest_price") or raw.get("current_price")
            if raw_current is not None:
                try:
                    current = float(raw_current)
                except (ValueError, TypeError):
                    pass

        # Get price_history: either directly from API or from text fallback
        price_history = raw.get("price_history")
        if not price_history:
            text = raw.get("price_insights_text", "")
            price_history = _parse_price_insights_text(text)

        forecast = None
        avg_price = None
        price_level = price_level_serp   # default
        trend = None

        if price_history and isinstance(price_history, list) and len(price_history) >= 2:
            # Use our historical analysis for better trend and level
            analysis = _analyze_price_trend(price_history)
            price_level = analysis["price_level"]
            trend = analysis["trend"]
            avg_price = analysis["avg_price"]
            forecast = _predict_future_price(price_history)

        return PriceInsights(
            price_level=price_level,
            typical_low_inr=typical_low,
            typical_high_inr=typical_high,
            current_price_inr=current,
            trend=trend,
            avg_price=avg_price,
            forecast=forecast,
            raw=raw,
        )

    except Exception as e:
        logger.warning("parse_price_insights failed", extra={"error": str(e)})
        return None


def format_price_insights_for_llm(insights: PriceInsights) -> str:
    """
    Convert a PriceInsights object into a single plain-English sentence
    suitable for direct injection into the LLM planning prompt.

    Examples:
        "Price intelligence: ₹4,500 is LOW vs. typical ₹5,800–₹7,200 (prices rising — book soon)."
        "Price intelligence: ₹6,800 is TYPICAL for this route (₹5,500–₹7,200 range)."
        "Price intelligence: ₹8,200 is HIGH vs. typical ₹5,500–₹7,200 — consider flexible dates."
    """
    level_upper = insights.price_level.upper()

    # Price display
    price_str = (
        f"₹{int(insights.current_price_inr):,}"
        if insights.current_price_inr is not None
        else "this fare"
    )

    # Range display (from SerpAPI typical range)
    if insights.typical_low_inr is not None and insights.typical_high_inr is not None:
        range_str = f"typical ₹{int(insights.typical_low_inr):,}–₹{int(insights.typical_high_inr):,}"
    else:
        range_str = "the typical range"

    # Additional historical average info
    avg_info = ""
    if insights.avg_price is not None and insights.current_price_inr is not None:
        pct_diff = (insights.current_price_inr - insights.avg_price) / insights.avg_price * 100
        if abs(pct_diff) > 5:   # only mention if significant
            direction = "above" if pct_diff > 0 else "below"
            avg_info = f" ({abs(pct_diff):.0f}% {direction} the historical average of ₹{int(insights.avg_price):,})"

    # Trend suffix
    trend_suffix = ""
    if insights.trend == "rising":
        trend_suffix = " ⚠️ prices are rising — booking now is advisable"
    elif insights.trend == "falling":
        trend_suffix = " ✅ prices are currently falling"
    elif insights.trend == "stable":
        trend_suffix = " prices are stable"

    # Forecast
    forecast_suffix = ""
    if insights.forecast and insights.forecast != "unknown" and "unknown" not in insights.forecast:
        forecast_suffix = f" Forecast: {insights.forecast}."

    # Recommendation tail
    if insights.price_level == "low":
        rec = f"✅ This is a good deal. Recommend booking soon.{trend_suffix}{forecast_suffix}"
    elif insights.price_level == "high":
        rec = f"⚠️ Consider flexible dates or nearby airports.{trend_suffix}{forecast_suffix}"
    else:
        rec = f"Fair price for this route.{trend_suffix}{forecast_suffix}"

    return (
        f"Price intelligence: {price_str} is {level_upper} vs. {range_str}{avg_info}. {rec}"
    )


# ----------------------------------------------------------------------
# 2. Price Snapshot recording
# ----------------------------------------------------------------------

def record_price_snapshot(
    *,
    origin: str,
    destination: str,
    travel_date: str,
    price_inr: float,
    insights: Optional[PriceInsights] = None,
) -> int:
    """
    Persist a price observation to the database.

    Args:
        origin, destination: IATA codes.
        travel_date: YYYY-MM-DD.
        price_inr: Cheapest price found in this search.
        insights: Parsed PriceInsights (may be None if SerpAPI didn't return the block).

    Returns:
        int: The new PriceSnapshot row id.
    """
    db = SessionLocal()
    try:
        snap = PriceSnapshot(
            origin=origin.upper(),
            destination=destination.upper(),
            travel_date=travel_date,
            price_inr=price_inr,
            typical_low_inr=insights.typical_low_inr if insights else None,
            typical_high_inr=insights.typical_high_inr if insights else None,
            price_level=insights.price_level if insights else None,
            trend=insights.trend if insights else None,
            insights_raw=insights.raw if insights else None,
        )
        db.add(snap)
        db.commit()
        db.refresh(snap)
        logger.debug(
            "Price snapshot recorded",
            extra={
                "route": f"{origin}-{destination}",
                "travel_date": travel_date,
                "price_inr": price_inr,
                "level": insights.price_level if insights else "N/A",
            }
        )
        return snap.id
    finally:
        db.close()


def get_price_history(
    origin: str,
    destination: str,
    travel_date: str,
    limit: int = 30,
) -> list[dict]:
    """
    Retrieve recent price snapshots for a route, newest first.

    Useful for building a trend description or chart data for the UI.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(PriceSnapshot)
            .filter(
                PriceSnapshot.origin == origin.upper(),
                PriceSnapshot.destination == destination.upper(),
                PriceSnapshot.travel_date == travel_date,
            )
            .order_by(PriceSnapshot.observed_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "observed_at":     r.observed_at.isoformat(),
                "price_inr":       r.price_inr,
                "price_level":     r.price_level,
                "trend":           r.trend,
                "typical_low_inr": r.typical_low_inr,
                "typical_high_inr":r.typical_high_inr,
            }
            for r in rows
        ]
    finally:
        db.close()


# ----------------------------------------------------------------------
# 3. Background price-drop monitoring for HELD bookings
# ----------------------------------------------------------------------

async def check_held_booking_prices() -> list[dict]:
    """
    Background job: re-query flight prices for every active HELD booking and
    detect price drops.

    For each active hold:
    - If a booking token exists and the cooldown period has passed, call
      `airline_api.search_with_booking_token()` to get the precise current price.
    - Otherwise, fall back to a generic search via `airline_api.search_flights()`.
    - Compares the new cheapest price to the held price.
    - If the drop exceeds PRICE_DROP_ALERT_THRESHOLD_PCT, writes a
      PriceDropAlert row and resolves a fresh handoff URL for the cheaper option.

    Throttling: each booking token is checked at most once every
    CHECK_COOLDOWN_SECONDS seconds, using an LRU cache (_last_checked) with max size.

    Returns:
        list[dict]: All alerts fired in this run (may be empty).

    Scheduling recommendation:
        Schedule this via core/job_queue.py to run every 30–60 minutes.
    """
    # Lazy imports to avoid circular dependencies at module load time
    from tools.airline_api import search_flights, search_with_booking_token, AirlineAPIError
    from tools.booking_handoff import get_active_held_bookings, build_booking_handoff_url

    held = get_active_held_bookings()
    if not held:
        logger.debug("check_held_booking_prices: no active held bookings")
        return []

    alerts_fired = []
    now = time.time()

    for booking in held:
        flight_dict = booking.get("flight", {})
        origin      = flight_dict.get("origin") or flight_dict.get("departure_iata") or ""
        destination = flight_dict.get("destination") or flight_dict.get("arrival_iata") or ""
        travel_date = flight_dict.get("date") or ""
        held_price  = _extract_price_inr(flight_dict.get("price_inr"))
        booking_token = flight_dict.get("booking_token")  # may be None

        if not origin or not destination or not travel_date or held_price is None:
            logger.warning(
                "Skipping held booking: missing route data",
                extra={"booking_id": booking["id"]}
            )
            continue

        # Throttle check if booking_token exists
        if booking_token:
            last = _last_checked.get(booking_token, 0)
            if now - last < CHECK_COOLDOWN_SECONDS:
                logger.debug(
                    "Skipping booking token due to cooldown",
                    extra={
                        "booking_token_present": True,
                        "seconds_left": CHECK_COOLDOWN_SECONDS - (now - last),
                    }
                )
                continue

        new_flight = None
        try:
            if booking_token:
                # Use token to get precise current price
                # Assume search_with_booking_token returns a Flight object or list of flights
                result = await search_with_booking_token(booking_token)
                # If it returns a list, take the cheapest; if a single Flight, use it directly
                if isinstance(result, list):
                    if result:
                        new_flight = min(result, key=lambda f: f.price_inr)
                else:
                    new_flight = result
                _mark_checked(booking_token, now)   # update throttle cache
            else:
                # Fallback to generic search
                new_search_result = await search_flights(
                    departure=origin,
                    arrival=destination,
                    date=travel_date,
                    max_results=5,
                )
                new_flights = _coerce_flights_result(new_search_result)
                if new_flights:
                    new_flight = min(new_flights, key=lambda f: f.price_inr)
        except AirlineAPIError as e:
            logger.warning(
                "Price check failed",
                extra={"booking_id": booking["id"], "error": str(e)}
            )
            continue
        except Exception as e:
            logger.exception(
                "Unexpected error in price check",
                extra={"booking_id": booking["id"]}
            )
            continue

        if not new_flight:
            continue

        new_price = float(new_flight.price_inr)

        # Record snapshot regardless of whether it's a drop
        record_price_snapshot(
            origin=origin,
            destination=destination,
            travel_date=travel_date,
            price_inr=new_price,
        )

        # Check for meaningful drop
        drop_pct = (held_price - new_price) / held_price * 100
        if drop_pct >= PRICE_DROP_ALERT_THRESHOLD_PCT:
            # Resolve a fresh handoff URL for the cheaper option
            new_flight_dict = {
                "flight_no":     new_flight.flight_no,
                "airline":       new_flight.airline,
                "booking_token": new_flight.booking_token,
                "shareable_link": None,
                "price_inr":     new_flight.price_inr,
            }
            try:
                new_url = await build_booking_handoff_url(
                    flight=new_flight_dict,
                    origin=origin,
                    destination=destination,
                    depart_date=travel_date,
                )
            except Exception:
                new_url = None

            db = SessionLocal()
            try:
                alert = PriceDropAlert(
                    booking_id=booking["id"],
                    origin=origin,
                    destination=destination,
                    travel_date=travel_date,
                    held_price_inr=held_price,
                    new_price_inr=new_price,
                    drop_pct=round(drop_pct, 2),
                    new_handoff_url=new_url,
                )
                db.add(alert)
                db.commit()
                db.refresh(alert)
                alert_id = alert.id
            finally:
                db.close()

            alert_payload = {
                "alert_id":       alert_id,
                "booking_id":     booking["id"],
                "route":          f"{origin}→{destination}",
                "travel_date":    travel_date,
                "held_price_inr": held_price,
                "new_price_inr":  new_price,
                "drop_pct":       round(drop_pct, 2),
                "new_handoff_url": new_url,
                "new_flight":     f"{new_flight.airline} {new_flight.flight_no}",
            }
            alerts_fired.append(alert_payload)

            logger.info(
                "Price drop alert fired",
                extra={
                    "booking_id":   booking["id"],
                    "drop_pct":     round(drop_pct, 2),
                    "held_price":   held_price,
                    "new_price":    new_price,
                    "new_flight_no": new_flight.flight_no,
                }
            )

    return alerts_fired


def get_unacknowledged_alerts(booking_id: Optional[int] = None) -> list[dict]:
    """
    Fetch all unacknowledged price-drop alerts, optionally filtered by booking.

    The application layer should call this after check_held_booking_prices()
    and push the results to the user via whatever notification channel is available
    (WebSocket, email, push notification, etc.)
    """
    db = SessionLocal()
    try:
        q = db.query(PriceDropAlert).filter(PriceDropAlert.acknowledged == False)  # noqa: E712
        if booking_id is not None:
            q = q.filter(PriceDropAlert.booking_id == booking_id)
        rows = q.order_by(PriceDropAlert.created_at.desc()).all()
        return [
            {
                "alert_id":       r.id,
                "booking_id":     r.booking_id,
                "origin":         r.origin,
                "destination":    r.destination,
                "travel_date":    r.travel_date,
                "held_price_inr": r.held_price_inr,
                "new_price_inr":  r.new_price_inr,
                "drop_pct":       r.drop_pct,
                "new_handoff_url": r.new_handoff_url,
                "created_at":     r.created_at.isoformat(),
            }
            for r in rows
        ]
    finally:
        db.close()


def acknowledge_alert(alert_id: int) -> bool:
    """Mark a price-drop alert as acknowledged (read by user/system)."""
    db = SessionLocal()
    try:
        row = db.get(PriceDropAlert, alert_id)
        if not row:
            return False
        row.acknowledged = True
        db.commit()
        return True
    finally:
        db.close()


def purge_old_snapshots() -> int:
    """
    Delete price snapshots older than SNAPSHOT_RETENTION_DAYS.
    Call from a nightly cleanup job.

    Returns:
        int: Number of rows deleted.
    """
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=SNAPSHOT_RETENTION_DAYS)
        deleted = (
            db.query(PriceSnapshot)
            .filter(PriceSnapshot.observed_at < cutoff)
            .delete(synchronize_session=False)
        )
        db.commit()
        logger.info("Purged old price snapshots", extra={"count": deleted})
        return deleted
    finally:
        db.close()


# ----------------------------------------------------------------------
# Public aliases for legacy callers
# ----------------------------------------------------------------------

def analyze_price_trend(price_history):
    """Public alias for legacy callers (calls internal _analyze_price_trend)."""
    return _analyze_price_trend(price_history)


def predict_future_price(price_history):
    """Public alias for legacy callers (calls internal _predict_future_price)."""
    return _predict_future_price(price_history)


# Optionally make the public API explicit
try:
    __all__  # preserve if already defined
except NameError:
    __all__ = []

for _name in ("analyze_price_trend", "predict_future_price"):
    if _name not in __all__:
        __all__.append(_name)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _extract_price_inr(raw_price) -> Optional[float]:
    """Normalise a price value from the flight dict to a plain float."""
    if raw_price is None:
        return None
    if isinstance(raw_price, (int, float)):
        return float(raw_price)
    if isinstance(raw_price, str):
        cleaned = raw_price.replace("₹", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None
