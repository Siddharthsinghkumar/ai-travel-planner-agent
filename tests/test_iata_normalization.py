# tests/test_iata_normalization.py
import pytest
import asyncio
from unittest.mock import AsyncMock
from datetime import datetime, timedelta

import agents.planner_agent as planner_agent
from agents.planner_agent import normalize_airport, parse_intent, _plan_trip_internal
from agents.planner_agent import Flight, ParsedIntent
from core.iata_resolver import resolve_location_with_trace

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


def test_invalid_three_letter_plainword_is_not_treated_as_iata():
    assert normalize_airport("qqq") is None


def test_planner_does_not_own_city_iata_override_map():
    assert not hasattr(planner_agent, "CITY_IATA_OVERRIDES")


def test_resolver_trace_reports_alias_basis():
    code, trace = resolve_location_with_trace("Calcutta")
    assert code == "CCU"
    assert trace.get("match_basis") == "alias"
    assert trace.get("selected_iata") == "CCU"


def test_resolver_trace_reports_known_correction_basis():
    code, trace = resolve_location_with_trace("banglore")
    assert code == "BLR"
    assert trace.get("match_basis") in {"known_correction", "alias", "exact_city", "fuzzy_city"}
    assert trace.get("selected_iata") == "BLR"


def test_resolver_prefers_city_level_match_for_kochi_over_unrelated_airport_name():
    code, trace = resolve_location_with_trace("Kochi")
    assert code == "COK"
    assert trace.get("selected_iata") == "COK"
    assert trace.get("match_basis") in {"fuzzy_city", "fuzzy_city_preferred_tiebreak", "exact_city", "alias"}


def test_resolver_cochin_maps_to_cok():
    code, trace = resolve_location_with_trace("Cochin")
    assert code == "COK"
    assert trace.get("selected_iata") == "COK"


def test_resolver_multiword_city_remains_correct():
    code, trace = resolve_location_with_trace("New Delhi")
    assert code == "DEL"
    assert trace.get("selected_iata") == "DEL"


def test_resolver_noisy_city_phrase_still_resolves_to_city_not_airport_name_jump():
    code, trace = resolve_location_with_trace("flights to Kochi city")
    assert code == "COK"
    assert trace.get("selected_iata") == "COK"

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


@pytest.mark.asyncio
async def test_plan_trip_rejects_past_departure_date_without_tool_call(monkeypatch):
    fake_search = AsyncMock(return_value=[])

    result = await _plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date="2025-01-10",
        user_query="Flight from Delhi to Mumbai on 2025-01-10",
        skip_llm=True,
        flight_tool=fake_search,
    )

    assert isinstance(result, dict)
    assert "past" in str(result.get("error", "")).lower()
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_roundtrip_return_weather_date_mismatch_is_marked_unavailable(monkeypatch):
    async def fake_handoff_url(*args, **kwargs):
        return None

    today = datetime.now().date()
    outbound_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    return_date = (today + timedelta(days=3)).strftime("%Y-%m-%d")

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        if travel_date == return_date:
            # Deliberately mismatched provider date to validate alignment behavior.
            return {
                "condition": "Cloudy",
                "temperature_c": 24,
                "forecast_date": outbound_date,
                "temp_min_c": 20,
                "temp_max_c": 27,
            }
        return {
            "condition": "Clear",
            "temperature_c": 30,
            "forecast_date": travel_date,
            "temp_min_c": 26,
            "temp_max_c": 33,
        }

    async def fake_search(*args, **kwargs):
        departure = kwargs.get("departure") or "AAA"
        arrival = kwargs.get("arrival") or "BBB"
        date = kwargs.get("date") or outbound_date
        return [
            {
                "airline": "TestAir",
                "flight_no": "TA200",
                "departure_time": "09:00",
                "arrival_time": "11:00",
                "duration_min": 120,
                "price_inr": 6200,
                "stops": 0,
                "layover_info": f"via {arrival}",
                "layover_airports": [arrival],
                "layover_durations_min": [45],
                "baggage": "7kg cabin",
                "date": date,
            }
        ], {"_search_meta": {"raw_candidate_count": 1}}

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff_url)
    monkeypatch.setattr(
        "agents.planner_agent.parse_intent",
        lambda _query: ParsedIntent(
            origin_iata="DEL",
            destination_iata="BOM",
            date=outbound_date,
            return_date=return_date,
            trip_type="round-trip",
        ),
    )

    result = await _plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=outbound_date,
        user_query="Round-trip from Delhi to Mumbai",
        trip_type="round-trip",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.return_trip is not None
    assert result.return_trip.search_date == return_date
    assert result.return_trip.weather_present is False
    assert result.return_trip.weather_reason == "forecast_date_mismatch"
    assert result.return_trip.weather.get("forecast_date") == return_date
    assert result.return_trip.weather.get("requested_date") == return_date
    assert result.return_trip.weather.get("provider_forecast_date") == outbound_date
    assert result.return_trip.weather.get("forecast_exact_match") is False
