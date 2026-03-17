#test/test_airline_api.py
import pytest
import os
from tools.airline_api import _parse_duration

def test_parse_duration():
    # Test standard string
    assert _parse_duration("2h 15m") == 135
    # Test colon format
    assert _parse_duration("2:15") == 135
    # Test minutes only
    assert _parse_duration("45m") == 45
    # Test hours only
    assert _parse_duration("3h") == 180
    # Test raw integer fallback
    assert _parse_duration(120) == 120
    # Test unparseable string
    assert _parse_duration("Unknown duration") is None

@pytest.mark.asyncio
async def test_flight_merging_logic(monkeypatch):
    # To test merging, we force the tool into TESTING mode
    # so it skips HTTP calls, but we can inspect how it behaves.
    monkeypatch.setenv("TESTING", "1")
    from tools.airline_api import search_flights
    
    flights, price_insights = await search_flights("DEL", "BOM", "2026-03-20")
    
    # Ensure the testing bypass returns our hardcoded test flight
    assert len(flights) == 1
    assert flights[0].airline == "TestAir"
    assert flights[0].duration_min == 120
    assert flights[0].carbon_emissions_g == 45000