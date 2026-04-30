"""Planner-facing tool gateway (compatibility shim).

This module is kept for backward compatibility. All functionality has been
consolidated into tools/tool_gateway.py.

DEPRECATED: Import directly from tools.tool_gateway instead.
"""

from tools.tool_gateway import (
    search_flights,
    check_weather,
    get_forecast_for_date,
    build_booking_handoff_url,
    booking_resolution_cache_stats,
    hold_booking,
    cancel_booking,
    record_price_snapshot,
    parse_price_insights,
    format_price_insights_for_llm,
    analyze_price_trend,
    predict_future_price,
)

__all__ = [
    "search_flights",
    "check_weather",
    "get_forecast_for_date",
    "build_booking_handoff_url",
    "booking_resolution_cache_stats",
    "hold_booking",
    "cancel_booking",
    "record_price_snapshot",
    "parse_price_insights",
    "format_price_insights_for_llm",
    "analyze_price_trend",
    "predict_future_price",
]