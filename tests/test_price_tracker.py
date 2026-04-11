#tests/test_price_tracker.py
from types import SimpleNamespace

import pytest
import tools.price_tracker as price_tracker
from tools.price_tracker import parse_price_insights, format_price_insights_for_llm

def test_price_insights_parsing_and_formatting():
    # 1. Mock the SerpApi JSON response block
    raw_serpapi_response = {
        "price_insights": {
            "price_level": "low",
            "typical_price_range": [5500, 7200],
            "lowest_price": 4200,
            "price_history": [[1670000000, 6000], [1670086400, 4200]]  # Simulates a falling trend
        }
    }
    
    # 2. Test Parsing
    insights = parse_price_insights(raw_serpapi_response)
    
    assert insights is not None
    assert insights.price_level == "low"
    assert insights.typical_low_inr == 5500.0
    assert insights.typical_high_inr == 7200.0
    assert insights.current_price_inr == 4200.0
    assert insights.trend == "falling"
    
    # 3. Test Formatting for the LLM
    formatted_string = format_price_insights_for_llm(insights)
    
    # Assert the required keywords made it into the LLM prompt string
    assert "LOW vs. typical" in formatted_string
    assert "₹5,500–₹7,200" in formatted_string
    assert "falling" in formatted_string
    assert "Recommend booking soon" in formatted_string


@pytest.mark.asyncio
async def test_check_held_booking_prices_fallback_accepts_search_flights_tuple(monkeypatch):
    snapshot_calls = []

    async def fake_search_with_booking_token(_token):
        return []

    async def fake_search_flights(**_kwargs):
        return (
            [
                SimpleNamespace(
                    price_inr=4200,
                    flight_no="TA101",
                    airline="TestAir",
                    booking_token="tok_new",
                )
            ],
            {"price_insights": {"price_level": "low"}},
        )

    def fake_get_active_held_bookings():
        return [
            {
                "id": 123,
                "flight": {
                    "origin": "DEL",
                    "destination": "BOM",
                    "date": "2026-05-01",
                    "price_inr": 6000,
                    # No booking_token -> fallback branch uses search_flights()
                },
            }
        ]

    async def fake_build_booking_handoff_url(**_kwargs):
        return "https://example.com/booking"

    def fake_record_price_snapshot(**kwargs):
        snapshot_calls.append(kwargs)

    monkeypatch.setattr("tools.airline_api.search_with_booking_token", fake_search_with_booking_token)
    monkeypatch.setattr("tools.airline_api.search_flights", fake_search_flights)
    monkeypatch.setattr("tools.booking_handoff.get_active_held_bookings", fake_get_active_held_bookings)
    monkeypatch.setattr("tools.booking_handoff.build_booking_handoff_url", fake_build_booking_handoff_url)
    monkeypatch.setattr(price_tracker, "record_price_snapshot", fake_record_price_snapshot)
    monkeypatch.setattr(price_tracker, "PRICE_DROP_ALERT_THRESHOLD_PCT", 999.0)

    alerts = await price_tracker.check_held_booking_prices()

    assert alerts == []
    assert len(snapshot_calls) == 1
    assert snapshot_calls[0]["price_inr"] == 4200.0


@pytest.mark.asyncio
async def test_check_held_booking_prices_expires_legacy_rows_missing_route_fields(monkeypatch):
    async def should_not_run_search_with_booking_token(_token):
        raise AssertionError("search_with_booking_token should not run for invalid legacy rows")

    async def should_not_run_search_flights(**_kwargs):
        raise AssertionError("search_flights should not run for invalid legacy rows")

    expired_calls = []

    def fake_get_active_held_bookings():
        return [
            {
                "id": 999,
                "flight": {
                    # Missing origin/destination/date on purpose (legacy malformed row)
                    "price_inr": 6000,
                    "booking_token": "tok_legacy",
                },
            }
        ]

    def fake_expire_held_booking_for_tracking_invalid_data(booking_id: int, *, reason: str):
        expired_calls.append((booking_id, reason))
        return True

    async def fake_build_booking_handoff_url(**_kwargs):
        return None

    monkeypatch.setattr("tools.airline_api.search_with_booking_token", should_not_run_search_with_booking_token)
    monkeypatch.setattr("tools.airline_api.search_flights", should_not_run_search_flights)
    monkeypatch.setattr("tools.booking_handoff.get_active_held_bookings", fake_get_active_held_bookings)
    monkeypatch.setattr("tools.booking_handoff.expire_held_booking_for_tracking_invalid_data", fake_expire_held_booking_for_tracking_invalid_data)
    monkeypatch.setattr("tools.booking_handoff.build_booking_handoff_url", fake_build_booking_handoff_url)

    alerts = await price_tracker.check_held_booking_prices()

    assert alerts == []
    assert len(expired_calls) == 1
    booking_id, reason = expired_calls[0]
    assert booking_id == 999
    assert "missing_tracking_fields" in reason


def test_cleanup_invalid_held_tracking_rows_expires_startup_legacy_rows(monkeypatch):
    expired_calls = []

    def fake_get_active_held_bookings():
        return [
            {
                "id": 10,
                "flight": {
                    "origin": "DEL",
                    "destination": "BOM",
                    "date": "2026-05-01",
                    "price_inr": 6000,
                },
            },
            {
                "id": 11,
                "flight": {
                    "destination": "BLR",
                    "price_inr": 7000,
                },
            },
        ]

    def fake_expire_held_booking_for_tracking_invalid_data(booking_id: int, *, reason: str, emit_warning: bool = True):
        expired_calls.append((booking_id, reason, emit_warning))
        return True

    monkeypatch.setattr("tools.booking_handoff.get_active_held_bookings", fake_get_active_held_bookings)
    monkeypatch.setattr(
        "tools.booking_handoff.expire_held_booking_for_tracking_invalid_data",
        fake_expire_held_booking_for_tracking_invalid_data,
    )

    summary = price_tracker.cleanup_invalid_held_tracking_rows()

    assert summary["scanned"] == 2
    assert summary["expired"] == 1
    assert summary["expired_booking_ids"] == [11]
    assert expired_calls[0][0] == 11
    assert "startup_missing_tracking_fields" in expired_calls[0][1]
    assert expired_calls[0][2] is False
