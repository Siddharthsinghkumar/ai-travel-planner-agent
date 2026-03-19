# tests/test_iata_normalization.py
import pytest
import asyncio
from unittest.mock import AsyncMock

from agents.planner_agent import normalize_airport, parse_intent, _plan_trip_internal
from agents.planner_agent import Flight

def test_explicit_iata_uppercase():
    assert normalize_airport("DEL") == "DEL"

def test_explicit_iata_lowercase():
    assert normalize_airport("bom") == "BOM"

def test_non_iata_city():
    result = normalize_airport("delhi")
    assert result is not None
    assert len(result) == 3

def test_parse_intent_iata_codes():
    intent = parse_intent("DEL to BOM tomorrow")
    assert intent.origin_iata == "DEL"
    assert intent.destination_iata == "BOM"

@pytest.mark.asyncio
async def test_origin_override(monkeypatch):
    async def fake_handoff_url(*args, **kwargs):
        return "https://example.com/checkout"

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff_url)

    # Create a minimal fake flight result that matches the real Flight model.
    fake_flight = Flight(
        flight_no="XX123",
        airline="TestAir",
        departure_time="10:00",  # Fixed format for new Pydantic validator
        arrival_time="12:00",    # Fixed format for new Pydantic validator
        duration_min=120,
        price_inr=5000,
        stops=0,
        layover_info="",
        baggage="Check airline",
        booking_token=None,
        shareable_link="https://google.com",
        carbon_emissions_g=50000
    )
    fake_parsed_results = [fake_flight]
    fake_attempts = 1

    # Create a mock search function that returns the fake data.
    fake_search = AsyncMock(return_value=fake_parsed_results)

    # Inject the mock directly via the flight_tool parameter.
    result = await _plan_trip_internal(
        origin="DEL",
        destination="BOM",
        user_query="random text",
        skip_llm=True,
        flight_tool=fake_search
    )

    assert result.best_flight["flight_no"] == "XX123"
