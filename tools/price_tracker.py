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
import contextlib
import asyncio
import logging
import os
import time
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from cachetools import TTLCache
from core.env_config import get_env_int

from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Boolean, Text
from agents.database import Base, SessionLocal, get_engine
from tools import price_tracker_tool_gateway as tracker_gateway

# Import from canonical gateway for direct function calls
from tools.tool_gateway import (
    search_flights,
    search_with_booking_token,
    build_booking_handoff_url,
)

# Use wrappers that delegate to tracker_gateway at call time for test compatibility
# Tests patch tracker_gateway module attributes, we need to call through that reference
def get_active_held_bookings():
    return tracker_gateway.get_active_held_bookings()

def expire_held_booking_for_tracking_invalid_data(*args, **kwargs):
    return tracker_gateway.expire_held_booking_for_tracking_invalid_data(*args, **kwargs)

def tracking_context_from_booking(booking):
    return tracker_gateway.tracking_context_from_booking(booking)

# Optional numpy for linear regression forecast
try:
    import numpy as np
except ImportError:
    np = None

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
        price_level = price_level_serp   # default — SerpAPI's value is authoritative
        trend = None

        if price_history and isinstance(price_history, list) and len(price_history) >= 2:
            analysis = _analyze_price_trend(price_history)
            # Use SerpAPI's price_level as authoritative; only fall back to computed
            # value when SerpAPI didn't provide one.
            if price_level_serp == "typical" and analysis["price_level"] != "unknown":
                # "typical" is SerpAPI's default — computed value may be more precise
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
# 2. Flight data capture from successful SerpAPI calls
# ----------------------------------------------------------------------

# In-memory store: route+date → captured flight data
# This avoids making SerpAPI calls for price checks — we use data already captured.
# Using TTLCache with bounded maxsize to prevent unbounded growth.
FLIGHT_DATA_CACHE_TTL = max(300, get_env_int("FLIGHT_DATA_CACHE_TTL", 3600))  # default 1 hour
FLIGHT_DATA_CACHE_MAXSIZE = max(100, get_env_int("FLIGHT_DATA_CACHE_MAXSIZE", 500))
_flight_data_cache: TTLCache = TTLCache(maxsize=FLIGHT_DATA_CACHE_MAXSIZE, ttl=FLIGHT_DATA_CACHE_TTL)


def _flight_data_cache_key(origin: str, destination: str, travel_date: str) -> str:
    return f"{origin.upper()}|{destination.upper()}|{travel_date}"


async def record_flight_data(
    *,
    origin: str,
    destination: str,
    travel_date: str,
    flights: list,
    price_insights: Optional[dict] = None,
) -> None:
    """
    Capture flight data from a successful SerpAPI search response.
    This data is used by the price tracker to check held bookings WITHOUT
    making additional SerpAPI HTTP calls.

    Stores: cheapest price, all flight details (flight_no, airline, departure,
    arrival, price, stops, etc.), and price_insights if available.

    Called automatically after every successful search_flights() call.
    """
    if not flights:
        return

    key = _flight_data_cache_key(origin, destination, travel_date)
    captured = {
        "origin": origin.upper(),
        "destination": destination.upper(),
        "travel_date": travel_date,
        "captured_at": time.time(),
        "flight_count": len(flights),
        "cheapest_price": None,
        "cheapest_flight": None,
        "flights": [],
        "price_insights": price_insights,
    }

    cheapest = None
    cheapest_price = float("inf")

    for f in flights:
        flight_data = {}
        if hasattr(f, "model_dump"):
            flight_data = f.model_dump()
        elif isinstance(f, dict):
            flight_data = f
        elif hasattr(f, "__dict__"):
            flight_data = dict(vars(f))

        if not flight_data:
            continue

        # Extract key fields
        price = flight_data.get("price_inr")
        if price is not None:
            try:
                price = float(price)
                if price < cheapest_price:
                    cheapest_price = price
                    cheapest = flight_data
            except (ValueError, TypeError):
                pass

        # Store compact flight info
        captured["flights"].append({
            "flight_no": flight_data.get("flight_no"),
            "airline": flight_data.get("airline"),
            "departure_time": flight_data.get("departure_time"),
            "arrival_time": flight_data.get("arrival_time"),
            "duration_min": flight_data.get("duration_min"),
            "price_inr": flight_data.get("price_inr"),
            "stops": flight_data.get("stops"),
            "layover_info": flight_data.get("layover_info"),
            "baggage": flight_data.get("baggage"),
            "carbon_emissions_g": flight_data.get("carbon_emissions_g"),
            "booking_token": flight_data.get("booking_token"),
            "shareable_link": flight_data.get("shareable_link"),
        })

    captured["cheapest_price"] = cheapest_price if cheapest_price != float("inf") else None
    captured["cheapest_flight"] = cheapest

    # Store in cache (overwrite with latest data)
    _flight_data_cache[key] = captured

    # Also persist cheapest price to DB for historical tracking
    if cheapest_price != float("inf"):
        try:
            await record_price_snapshot(
                origin=origin,
                destination=destination,
                travel_date=travel_date,
                price_inr=cheapest_price,
            )
        except Exception:
            logger.debug("Failed to persist price snapshot from flight data capture")


def get_cached_flight_data(
    origin: str,
    destination: str,
    travel_date: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve captured flight data for a route+date.
    Returns None if no data has been captured yet.
    """
    key = _flight_data_cache_key(origin, destination, travel_date)
    return _flight_data_cache.get(key)


# ----------------------------------------------------------------------
# 3. Price Snapshot recording
# ----------------------------------------------------------------------

def _record_price_snapshot_sync(
    *,
    origin: str,
    destination: str,
    travel_date: str,
    price_inr: float,
    insights: Optional[PriceInsights] = None,
) -> int:
    """Internal sync implementation — do not call from async contexts."""
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


async def record_price_snapshot(
    *,
    origin: str,
    destination: str,
    travel_date: str,
    price_inr: float,
    insights: Optional[PriceInsights] = None,
) -> int:
    return await asyncio.to_thread(
        _record_price_snapshot_sync,
        origin=origin,
        destination=destination,
        travel_date=travel_date,
        price_inr=price_inr,
        insights=insights,
    )


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

def cleanup_invalid_held_tracking_rows() -> Dict[str, Any]:
    """
    One-shot cleanup for legacy HELD rows missing route/date tracking prerequisites.
    This is intended for startup hygiene so recurring tracker warning churn is avoided.
    """
    held_rows = get_active_held_bookings()
    summary: Dict[str, Any] = {
        "scanned": len(held_rows),
        "expired": 0,
        "expired_booking_ids": [],
    }
    if not held_rows:
        return summary

    for booking in held_rows:
        context = tracking_context_from_booking(booking)
        missing_fields = []
        if not context.origin:
            missing_fields.append("origin")
        if not context.destination:
            missing_fields.append("destination")
        if not context.travel_date:
            missing_fields.append("travel_date")
        if not missing_fields:
            continue
        if context.booking_id is None:
            continue
        with contextlib.suppress(Exception):
            expired = bool(
                expire_held_booking_for_tracking_invalid_data(
                    int(context.booking_id),
                    reason=f"startup_missing_tracking_fields:{','.join(missing_fields)}",
                    emit_warning=False,
                )
            )
            if expired:
                summary["expired"] += 1
                summary["expired_booking_ids"].append(int(context.booking_id))

    if summary["expired"]:
        logger.warning(
            "Expired legacy held bookings with invalid tracking prerequisites during startup cleanup",
            extra={
                "scanned": summary["scanned"],
                "expired": summary["expired"],
                "expired_booking_ids": summary["expired_booking_ids"][:20],
            },
        )
    return summary


async def check_held_booking_prices() -> list[dict]:
    """
    Background job: check held bookings for price drops.

    Uses flight data captured from successful SerpAPI searches — does NOT
    make any new SerpAPI HTTP calls. This avoids burning quota at scale.

    For each active hold:
    - Looks up cached flight data for the route+date
    - If data exists, compares cheapest cached price to held price
    - If the drop exceeds PRICE_DROP_ALERT_THRESHOLD_PCT, writes a PriceDropAlert

    Returns:
        list[dict]: All alerts fired in this run (may be empty).
    """
    held = get_active_held_bookings()
    if not held:
        logger.debug("check_held_booking_prices: no active held bookings")
        return []

    alerts_fired = []
    now = time.time()

    for booking in held:
        context = tracking_context_from_booking(booking)
        booking_id = context.booking_id or booking.get("id")
        origin = context.origin
        destination = context.destination
        travel_date = context.travel_date
        held_price = context.held_price
        booking_token = context.booking_token

        missing_fields = []
        if not origin:
            missing_fields.append("origin")
        if not destination:
            missing_fields.append("destination")
        if not travel_date:
            missing_fields.append("travel_date")
        if held_price is None:
            missing_fields.append("held_price")

        if missing_fields:
            invalidation_applied = False
            if any(field in {"origin", "destination", "travel_date"} for field in missing_fields):
                with contextlib.suppress(Exception):
                    invalidation_applied = bool(
                        expire_held_booking_for_tracking_invalid_data(
                            int(booking_id),
                            reason=f"missing_tracking_fields:{','.join(missing_fields)}",
                        )
                    )
            logger.warning(
                "Skipping held booking: missing tracking prerequisites",
                extra={
                    "booking_id": booking_id,
                    "missing_fields": ",".join(missing_fields),
                    "expired_legacy_invalid_row": invalidation_applied,
                },
            )
            continue

        # Use cached flight data from successful SerpAPI searches — NO new HTTP calls
        cached = get_cached_flight_data(origin, destination, travel_date)
        if not cached or cached.get("cheapest_price") is None:
            logger.debug(
                "No cached flight data for held booking — skipping (no SerpAPI call)",
                extra={
                    "booking_id": booking_id,
                    "origin": origin,
                    "destination": destination,
                    "travel_date": travel_date,
                },
            )
            continue

        cheapest_price = cached["cheapest_price"]
        cheapest_flight = cached.get("cheapest_flight")

        try:
            new_price = float(cheapest_price)
        except (ValueError, TypeError):
            logger.debug(
                "Invalid cached price for held booking",
                extra={"booking_id": booking_id, "price": cheapest_price},
            )
            continue

        if held_price <= 0 or new_price >= held_price:
            continue

        drop_pct = ((held_price - new_price) / held_price) * 100
        if drop_pct < PRICE_DROP_ALERT_THRESHOLD_PCT:
            continue

        # Record snapshot for this price observation
        try:
            await record_price_snapshot(
                origin=origin,
                destination=destination,
                travel_date=travel_date,
                price_inr=new_price,
            )
        except Exception:
            logger.debug("Failed to record price snapshot for alert")

        alert = {
            "booking_id": booking_id,
            "origin": origin,
            "destination": destination,
            "travel_date": travel_date,
            "held_price": held_price,
            "new_price": new_price,
            "drop_pct": round(drop_pct, 1),
            "flight_no": cheapest_flight.get("flight_no") if cheapest_flight else None,
            "airline": cheapest_flight.get("airline") if cheapest_flight else None,
        }
        alerts_fired.append(alert)
        logger.info(
            "Price drop alert fired (from cached data, no SerpAPI call)",
            extra={
                "booking_id": booking_id,
                "drop_pct": alert["drop_pct"],
                "held_price": held_price,
                "new_price": new_price,
            },
        )

    return alerts_fired


def get_unacknowledged_alerts(
    booking_id: Optional[int] = None,
    owner_principal_id: Optional[str] = None,
) -> list[dict]:
    """
    Fetch all unacknowledged price-drop alerts, optionally filtered by booking.

    The application layer should call this after check_held_booking_prices()
    and push the results to the user via whatever notification channel is available
    (WebSocket, email, push notification, etc.)
    """
    db = SessionLocal()
    try:
        q = db.query(PriceDropAlert).filter(PriceDropAlert.acknowledged == False)  # noqa: E712
        if owner_principal_id is not None:
            from tools.booking_handoff import Booking

            q = q.join(Booking, Booking.id == PriceDropAlert.booking_id).filter(
                Booking.owner_principal_id == str(owner_principal_id)
            )
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


def acknowledge_alert(alert_id: int, owner_principal_id: Optional[str] = None) -> bool:
    """Mark a price-drop alert as acknowledged (read by user/system)."""
    db = SessionLocal()
    try:
        if owner_principal_id is None:
            row = db.get(PriceDropAlert, alert_id)
        else:
            from tools.booking_handoff import Booking

            row = (
                db.query(PriceDropAlert)
                .join(Booking, Booking.id == PriceDropAlert.booking_id)
                .filter(
                    PriceDropAlert.id == alert_id,
                    Booking.owner_principal_id == str(owner_principal_id),
                )
                .first()
            )
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
