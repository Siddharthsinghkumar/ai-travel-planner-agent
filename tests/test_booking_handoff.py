#tests/test_booking_handoff.py
import asyncio
import threading
import pytest
import urllib.parse
from datetime import datetime, timedelta
import agents.planner_agent as planner_agent
import tools.booking_handoff as booking_handoff
from tools.booking_handoff import build_google_flights_fallback, build_booking_handoff_url

def test_build_google_flights_fallback():
    url = build_google_flights_fallback(
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20"
    )

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "www.google.com"
    assert parsed.path == "/travel/flights"
    assert query.get("hl", [None])[0] == "en"
    assert query.get("gl", [None])[0] == "in"
    assert query.get("curr", [None])[0] == "INR"
    q_text = query.get("q", [""])[0]
    assert "Flights from DEL airport to BOM airport on 2026-03-20" in q_text
    assert "IATA DEL BOM" in q_text


def test_planner_normalize_flights_retains_booking_artifacts():
    raw = [{
        "airline": "TestAir",
        "flight_no": "TA123",
        "departure_time": "08:00",
        "arrival_time": "10:00",
        "duration_min": 120,
        "price_inr": 5000,
        "stops": 0,
        "baggage": "7kg cabin",
        "booking_token": "tok_123",
        "shareable_link": "https://partner.example/share/abc",
        "booking_request": {"url": "https://partner.example/checkout", "method": "GET"},
        "booking_options": [{"booking_url": "https://partner.example/checkout/1"}],
    }]

    normalized = planner_agent.normalize_flights(raw, "2026-03-20")
    assert len(normalized) == 1
    assert normalized[0].shareable_link == "https://partner.example/share/abc"
    assert normalized[0].booking_request["url"] == "https://partner.example/checkout"
    assert normalized[0].booking_options[0]["booking_url"] == "https://partner.example/checkout/1"


def test_align_top_level_booking_handoff_with_rows_promotes_stronger_candidate():
    top_level = {
        "source": "google_flights_fallback",
        "reason": "no_booking_artifacts_google_search_fallback",
        "status": "ok",
        "handoff_mode": "search_fallback",
        "is_search_fallback": True,
        "is_provider_managed": False,
        "is_booking_quality_exit": False,
        "booking_exit_quality": "search_assist",
    }
    rows = [
        {
            "rank": 1,
            "handoff_url": "https://www.google.com/travel/flights?q=fallback",
            "booking_handoff": dict(top_level),
        },
        {
            "rank": 2,
            "handoff_url": "/booking/handoff/post/bridge-abc",
            "booking_handoff": {
                "source": "booking_token",
                "reason": "resolved_booking_request_post",
                "status": "ok",
                "handoff_mode": "provider_or_shareable",
                "requires_browser_post": True,
                "is_search_fallback": False,
                "is_provider_managed": True,
                "is_booking_quality_exit": True,
                "booking_exit_quality": "booking_ready",
            },
        },
    ]

    aligned, aligned_url, changed = planner_agent._align_top_level_booking_handoff_with_rows(top_level, rows)

    assert changed is True
    assert aligned["is_booking_quality_exit"] is True
    assert aligned["selected_flight_rank"] == 2
    assert aligned_url == "/booking/handoff/post/bridge-abc"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_priorities():
    # Test that shareable_link is prioritized over the fallback
    flight_mock = {
        "shareable_link": "https://google.com/flights/shareable_test"
    }
    
    url = await build_booking_handoff_url(
        flight=flight_mock,
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20"
    )
    
    assert url == "https://google.com/flights/shareable_test"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_return_details_for_resolved_booking_token(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": "https://airline.example/checkout",
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "status": "ok",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={"booking_token": "tok_123"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["url"] == "https://airline.example/checkout"
    assert details["source"] == "booking_token"
    assert details["reason"] == "resolved_booking_token"
    assert details["status"] == "ok"
    assert details["provider"] == "serpapi"
    assert details["is_exact_handoff"] is True
    assert details["is_search_fallback"] is False
    assert details["is_booking_quality_exit"] is True
    assert details["booking_exit_quality"] == "booking_ready"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_return_details_for_google_fallback(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={"booking_token": "tok_123"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "google_flights_fallback"
    assert details["reason"] == "booking_token_unresolved_google_search_fallback"
    assert details["status"] == "ok"
    assert details["provider"] == "google_flights"
    assert details["handoff_mode"] == "search_fallback"
    assert details["landing_guarantee"] == "best_effort_search"
    assert details["is_exact_handoff"] is False
    assert details["is_search_fallback"] is True
    assert details["is_booking_quality_exit"] is False
    assert details["booking_exit_quality"] == "search_assist"
    assert details["url"].startswith("https://www.google.com/travel/flights?")


@pytest.mark.asyncio
async def test_build_booking_handoff_url_uses_shareable_when_token_resolution_times_out(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_resolution_timeout",
            "status": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={"booking_token": "tok_123", "shareable_link": "https://google.com/flights/shareable_timeout"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "shareable_link"
    assert details["reason"] == "booking_token_resolution_timeout_fallback_shareable"
    assert details["status"] == "ok"
    assert details["provider"] == "serpapi"
    assert details["url"] == "https://google.com/flights/shareable_timeout"
    assert details["is_exact_handoff"] is False
    assert details["is_search_fallback"] is False
    assert details["is_booking_quality_exit"] is True
    assert details["booking_exit_quality"] == "booking_ready"


@pytest.mark.asyncio
async def test_resolve_booking_token_with_details_classifies_timeout(monkeypatch):
    async def slow_payload(**_kwargs):
        await asyncio.sleep(0.02)
        return {"booking_options": [{"link": "https://airline.example/late"}]}

    monkeypatch.setattr(booking_handoff, "_fetch_booking_options_payload", slow_payload)
    monkeypatch.setattr(booking_handoff, "BOOKING_TOKEN_RESOLVE_TIMEOUT", 0.001)

    result = await booking_handoff.resolve_booking_token_with_details("tok_123")

    assert result["url"] is None
    assert result["status"] == "unavailable"
    assert result["reason"] == "booking_token_resolution_timeout"
    assert result["source"] == "booking_token"
    assert result["is_exact_handoff"] is False
    assert result["is_search_fallback"] is False
    assert result["is_booking_quality_exit"] is False
    assert result["booking_exit_quality"] == "unavailable"


@pytest.mark.asyncio
async def test_planner_booking_handoff_safe_reports_unavailable_on_exception(monkeypatch):
    async def broken_build(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("tools.booking_handoff.build_booking_handoff_url", broken_build)

    result = await planner_agent._build_booking_handoff_url_safe(return_details=True)

    assert result["url"] is None
    assert result["source"] == "unavailable"
    assert result["reason"] == "booking_handoff_exception"
    assert result["status"] == "unavailable"
    assert result["is_exact_handoff"] is False
    assert result["is_search_fallback"] is False
    assert result["is_booking_quality_exit"] is False
    assert result["booking_exit_quality"] == "unavailable"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_invalid_shareable_link_falls_back_to_google():
    details = await build_booking_handoff_url(
        flight={"shareable_link": "javascript:alert(1)"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "google_flights_fallback"
    assert details["reason"] == "invalid_shareable_link_google_search_fallback"
    assert details["status"] == "ok"
    assert details["provider"] == "google_flights"
    assert details["handoff_mode"] == "search_fallback"
    assert details["landing_guarantee"] == "best_effort_search"
    assert details["is_exact_handoff"] is False
    assert details["is_search_fallback"] is True
    assert details["is_booking_quality_exit"] is False
    assert details["booking_exit_quality"] == "search_assist"
    assert details["url"].startswith("https://www.google.com/travel/flights?")


@pytest.mark.asyncio
async def test_build_booking_handoff_url_uses_shareable_when_booking_token_url_is_invalid(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": "javascript:bad",
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "status": "ok",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={"booking_token": "tok_123", "shareable_link": "https://google.com/flights/shareable_valid"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "shareable_link"
    assert details["reason"] == "booking_token_invalid_url_fallback_shareable"
    assert details["status"] == "ok"
    assert details["url"] == "https://google.com/flights/shareable_valid"
    assert details["is_exact_handoff"] is False
    assert details["is_search_fallback"] is False
    assert details["is_booking_quality_exit"] is True
    assert details["booking_exit_quality"] == "booking_ready"


@pytest.mark.asyncio
async def test_planner_booking_timeout_falls_back_to_google_handoff(monkeypatch):
    async def slow_handoff(*args, **kwargs):
        await asyncio.sleep(0.02)
        return {
            "url": "https://airline.example/late",
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "status": "ok",
            "provider": "serpapi",
        }

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        return [
            {
                "airline": "TestAir",
                "flight_no": "TA100",
                "departure_time": "09:00",
                "arrival_time": "11:00",
                "duration_min": 120,
                "price_inr": 5500,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            }
        ], {"_search_meta": {"raw_candidate_count": 1}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {
            "condition": "Clear",
            "temperature_c": 26,
            "forecast_date": travel_date,
        }

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", slow_handoff)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.001)
    monkeypatch.setattr("agents.planner_agent.ROUND_TRIP_HANDOFF_TIMEOUT_BONUS", 0.0)
    monkeypatch.setattr("agents.planner_agent.ROUND_TRIP_HANDOFF_TIMEOUT_BONUS", 0.0)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Find me a flight from Delhi to Mumbai",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.booking_handoff is not None
    assert result.booking_handoff["source"] == "google_flights_fallback"
    assert result.booking_handoff["reason"] == "booking_handoff_timeout_google_search_fallback"
    assert result.booking_handoff["status"] == "ok"
    assert result.booking_handoff["handoff_mode"] == "search_fallback"
    assert result.booking_handoff["landing_guarantee"] == "best_effort_search"
    assert result.booking_handoff["is_exact_handoff"] is False
    assert result.booking_handoff["is_search_fallback"] is True
    assert result.booking_handoff["is_booking_quality_exit"] is False
    assert result.booking_handoff["booking_exit_quality"] == "search_assist"
    assert result.booking_handoff["search_fallback_quality"] == "route_seeded_with_itinerary_hints"
    assert result.best_flight.get("handoff_url", "").startswith("https://www.google.com/travel/flights?")
    assert result.top_flights[0]["booking_handoff"]["search_fallback_quality"] == "route_seeded_with_itinerary_hints"


@pytest.mark.asyncio
async def test_planner_booking_exception_falls_back_to_google_handoff(monkeypatch):
    async def broken_handoff(*args, **kwargs):
        raise RuntimeError("handoff failure")

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        return [
            {
                "airline": "TestAir",
                "flight_no": "TA101",
                "departure_time": "09:00",
                "arrival_time": "11:00",
                "duration_min": 120,
                "price_inr": 5600,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            }
        ], {"_search_meta": {"raw_candidate_count": 1}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {
            "condition": "Clear",
            "temperature_c": 27,
            "forecast_date": travel_date,
        }

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", broken_handoff)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Find me a flight from Delhi to Mumbai",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.booking_handoff is not None
    assert result.booking_handoff["source"] == "google_flights_fallback"
    assert result.booking_handoff["reason"] == "booking_handoff_exception_google_search_fallback"
    assert result.booking_handoff["status"] == "ok"
    assert result.booking_handoff["handoff_mode"] == "search_fallback"
    assert result.booking_handoff["landing_guarantee"] == "best_effort_search"
    assert result.booking_handoff["is_exact_handoff"] is False
    assert result.booking_handoff["is_search_fallback"] is True
    assert result.booking_handoff["is_booking_quality_exit"] is False
    assert result.booking_handoff["booking_exit_quality"] == "search_assist"
    assert result.booking_handoff["search_fallback_quality"] == "route_seeded_with_itinerary_hints"
    assert result.best_flight.get("handoff_url", "").startswith("https://www.google.com/travel/flights?")
    assert result.top_flights[0]["booking_handoff"]["search_fallback_quality"] == "route_seeded_with_itinerary_hints"


@pytest.mark.asyncio
async def test_planner_round_trip_timeout_fallback_keeps_legacy_quality_and_exposes_tier_context(monkeypatch):
    async def slow_handoff(*args, **kwargs):
        await asyncio.sleep(0.02)
        return {
            "url": "https://airline.example/late",
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "status": "ok",
            "provider": "serpapi",
        }

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        return [
            {
                "airline": "TestAir",
                "flight_no": "TA100",
                "departure_time": "09:00",
                "arrival_time": "11:00",
                "duration_min": 120,
                "price_inr": 5500,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            }
        ], {"_search_meta": {"raw_candidate_count": 1}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {"condition": "Clear", "temperature_c": 26, "forecast_date": travel_date}

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", slow_handoff)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.001)
    monkeypatch.setattr("agents.planner_agent.ROUND_TRIP_HANDOFF_TIMEOUT_BONUS", 0.0)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    return_date = (datetime.now().date() + timedelta(days=4)).strftime("%Y-%m-%d")
    monkeypatch.setattr(
        "agents.planner_agent.parse_intent",
        lambda _query: planner_agent.ParsedIntent(
            origin_iata="DEL",
            destination_iata="BOM",
            date=date,
            return_date=return_date,
            trip_type="round-trip",
        ),
    )
    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Round-trip Delhi to Mumbai",
        trip_type="round-trip",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.booking_handoff["is_search_fallback"] is True
    assert result.booking_handoff["search_fallback_quality"] == "route_seeded_with_itinerary_hints"
    assert not result.booking_handoff["search_fallback_quality"].startswith("round_trip_")
    assert result.booking_handoff["search_fallback_context"]["route_type"] == "round_trip"
    assert result.booking_handoff["search_fallback_context"]["includes_return_leg_hint"] is True
    assert (
        result.booking_handoff["search_fallback_context"]["quality_tier"]
        == "round_trip_route_seeded_with_itinerary_hints"
    )


@pytest.mark.asyncio
async def test_planner_retains_per_flight_handoff_for_top_ranked_results(monkeypatch):
    async def fake_handoff(*args, **kwargs):
        flight = kwargs.get("flight") or {}
        flight_no = flight.get("flight_no")
        if flight_no == "TA001":
            return {
                "url": "https://airline.example/token-ta001",
                "source": "booking_token",
                "reason": "resolved_booking_token",
                "status": "ok",
                "provider": "serpapi",
            }
        if flight_no == "TA002":
            return {
                "url": "https://google.com/flights/share-ta002",
                "source": "shareable_link",
                "reason": "shareable_link_available",
                "status": "ok",
                "provider": "serpapi",
            }
        return {
            "url": "https://www.google.com/travel/flights?q=Flights%20from%20DEL%20to%20BOM%20on%202026-03-20",
            "source": "google_flights_fallback",
            "reason": "no_booking_artifacts_google_fallback",
            "status": "ok",
            "provider": "google_flights",
        }

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        return [
            {
                "airline": "TestAir",
                "flight_no": "TA001",
                "departure_time": "08:00",
                "arrival_time": "10:00",
                "duration_min": 120,
                "price_inr": 4500,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            },
            {
                "airline": "TestAir",
                "flight_no": "TA002",
                "departure_time": "09:00",
                "arrival_time": "11:05",
                "duration_min": 125,
                "price_inr": 5200,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            },
            {
                "airline": "TestAir",
                "flight_no": "TA003",
                "departure_time": "10:00",
                "arrival_time": "12:20",
                "duration_min": 140,
                "price_inr": 6100,
                "stops": 1,
                "layover_info": "45m at HYD",
                "baggage": "7kg cabin",
                "date": date,
            },
        ], {"_search_meta": {"raw_candidate_count": 3}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {
            "condition": "Clear",
            "temperature_c": 28,
            "forecast_date": travel_date,
        }

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_LIMIT", 3)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.2)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Find the cheapest flight from Delhi to Mumbai",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.top_flights is not None
    assert len(result.top_flights) == 3
    assert result.top_flights[0]["booking_handoff"]["source"] == "booking_token"
    assert result.top_flights[1]["booking_handoff"]["source"] == "shareable_link"
    assert result.top_flights[2]["booking_handoff"]["source"] == "google_flights_fallback"
    assert result.best_flight["flight_no"] == result.top_flights[0]["flight_no"]
    assert result.best_flight["handoff_url"] == result.top_flights[0]["handoff_url"]
    assert result.booking_handoff == result.top_flights[0]["booking_handoff"]


@pytest.mark.asyncio
async def test_planner_per_flight_timeout_classification_does_not_break_best_flight(monkeypatch):
    async def fake_handoff(*args, **kwargs):
        flight = kwargs.get("flight") or {}
        if flight.get("flight_no") == "TA002":
            await asyncio.sleep(0.03)
        return {
            "url": f"https://airline.example/{flight.get('flight_no', 'unknown').lower()}",
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "status": "ok",
            "provider": "serpapi",
        }

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        return [
            {
                "airline": "TestAir",
                "flight_no": "TA001",
                "departure_time": "07:30",
                "arrival_time": "09:30",
                "duration_min": 120,
                "price_inr": 4600,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            },
            {
                "airline": "TestAir",
                "flight_no": "TA002",
                "departure_time": "09:30",
                "arrival_time": "11:40",
                "duration_min": 130,
                "price_inr": 5000,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            },
        ], {"_search_meta": {"raw_candidate_count": 2}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {
            "condition": "Cloudy",
            "temperature_c": 27,
            "forecast_date": travel_date,
        }

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_LIMIT", 2)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.005)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Find the cheapest flight from Delhi to Mumbai",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.top_flights is not None
    assert len(result.top_flights) == 2
    by_flight_no = {f["flight_no"]: f for f in result.top_flights}

    assert by_flight_no["TA001"]["booking_handoff"]["source"] == "booking_token"
    assert by_flight_no["TA002"]["booking_handoff"]["reason"].startswith("booking_handoff_timeout")
    assert by_flight_no["TA002"]["booking_handoff"]["source"] in {"google_flights_fallback", "timeout"}
    assert result.booking_handoff == by_flight_no["TA001"]["booking_handoff"]
    assert result.best_flight["flight_no"] == "TA001"


@pytest.mark.asyncio
async def test_fetch_booking_options_reads_nested_rows_and_canonicalizes_links(monkeypatch):
    payload = {
        "best_flights": [
            {
                "booking_options": [
                    {
                        "provider": "ProviderA",
                        "price": "₹5,200",
                        "link": "https://www.google.com/url?q=https%3A%2F%2Fairline.example%2Fcheckout%3Fa%3D1",
                    }
                ]
            }
        ]
    }

    class _FakeResponse:
        status_code = 200

        def json(self):
            return payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _FakeResponse()

    class _ReserveCtx:
        async def __aenter__(self):
            return (0, "dummy")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())

    options = await booking_handoff.fetch_booking_options("tok_abc")

    assert options is not None
    assert len(options) == 1
    assert options[0]["provider"] == "ProviderA"
    assert options[0]["link"] == "https://airline.example/checkout?a=1"
    assert options[0]["price_available"] is True


@pytest.mark.asyncio
async def test_planner_promotes_booking_ready_handoff_over_fallback_without_reordering(monkeypatch):
    async def fake_handoff(*args, **kwargs):
        flight = kwargs.get("flight") or {}
        flight_no = flight.get("flight_no")
        if flight_no == "TA001":
            return {
                "url": "https://www.google.com/travel/flights?q=fallback",
                "source": "google_flights_fallback",
                "reason": "no_booking_artifacts_google_search_fallback",
                "status": "ok",
                "provider": "google_flights",
                "handoff_mode": "search_fallback",
                "is_search_fallback": True,
                "is_provider_managed": False,
                "is_booking_quality_exit": False,
                "booking_exit_quality": "search_assist",
                "search_fallback_quality": "route_seeded_with_itinerary_hints",
            }
        return {
            "url": "/booking/handoff/post/bridge-xyz",
            "source": "booking_token",
            "reason": "resolved_booking_request_post",
            "status": "ok",
            "provider": "serpapi",
            "handoff_mode": "provider_or_shareable",
            "artifact_field": "booking_request.post_followup_bridge",
            "requires_browser_post": True,
            "is_search_fallback": False,
            "is_provider_managed": True,
            "is_booking_quality_exit": True,
            "booking_exit_quality": "booking_ready",
        }

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        return [
            {
                "airline": "CheapAir",
                "flight_no": "TA001",
                "departure_time": "08:00",
                "arrival_time": "10:00",
                "duration_min": 120,
                "price_inr": 4000,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            },
            {
                "airline": "BookableAir",
                "flight_no": "TA002",
                "departure_time": "09:00",
                "arrival_time": "11:20",
                "duration_min": 140,
                "price_inr": 4100,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            },
        ], {"_search_meta": {"raw_candidate_count": 2}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {
            "condition": "Clear",
            "temperature_c": 28,
            "forecast_date": travel_date,
        }

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_LIMIT", 2)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.2)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Find the cheapest flight from Delhi to Mumbai",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.top_flights[0]["flight_no"] == "TA001"
    assert result.top_flights[1]["flight_no"] == "TA002"
    assert result.top_flights[0]["booking_handoff"]["is_booking_quality_exit"] is False
    assert result.top_flights[1]["booking_handoff"]["is_booking_quality_exit"] is True

    assert result.booking_handoff["is_booking_quality_exit"] is True
    assert result.booking_handoff["reason"] == "resolved_booking_request_post"
    assert result.booking_handoff["selected_flight_rank"] == 2


@pytest.mark.asyncio
async def test_planner_probe_candidates_can_promote_booking_ready_outside_top_payload(monkeypatch):
    async def fake_handoff(*args, **kwargs):
        flight = kwargs.get("flight") or {}
        if flight.get("flight_no") == "TA004":
            return {
                "url": "/booking/handoff/post/bridge-004",
                "source": "booking_token",
                "reason": "resolved_booking_request_post",
                "status": "ok",
                "provider": "serpapi",
                "handoff_mode": "provider_or_shareable",
                "artifact_field": "booking_request.post_followup_bridge",
                "requires_browser_post": True,
                "is_exact_handoff": False,
                "is_search_fallback": False,
                "is_provider_managed": True,
                "is_booking_quality_exit": True,
                "booking_exit_quality": "booking_ready",
            }
        return {
            "url": "https://www.google.com/travel/flights?q=fallback",
            "source": "google_flights_fallback",
            "reason": "no_booking_artifacts_google_search_fallback",
            "status": "ok",
            "provider": "google_flights",
            "handoff_mode": "search_fallback",
            "is_search_fallback": True,
            "is_provider_managed": False,
            "is_booking_quality_exit": False,
            "booking_exit_quality": "search_assist",
            "search_fallback_quality": "route_seeded_with_itinerary_hints",
        }

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        rows = []
        for idx, price in enumerate([4200, 4300, 4400, 4500, 4600], start=1):
            rows.append(
                {
                    "airline": "TestAir",
                    "flight_no": f"TA00{idx}",
                    "departure_time": f"{6 + idx:02d}:00",
                    "arrival_time": f"{8 + idx:02d}:10",
                    "duration_min": 130,
                    "price_inr": price,
                    "stops": 0,
                    "layover_info": "",
                    "baggage": "7kg cabin",
                    "date": date,
                }
            )
        return rows, {"_search_meta": {"raw_candidate_count": len(rows)}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {"condition": "Sunny", "temperature_c": 29, "forecast_date": travel_date}

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_LIMIT", 3)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_PROBE_LIMIT", 5)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.2)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Find cheapest flights from Delhi to Mumbai",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.top_flights is not None
    assert len(result.top_flights) == 3
    assert all(not f["booking_handoff"]["is_booking_quality_exit"] for f in result.top_flights)
    assert result.booking_handoff["is_booking_quality_exit"] is True
    assert result.booking_handoff["reason"] == "resolved_booking_request_post"
    assert result.booking_handoff["selected_flight_rank"] == 4


@pytest.mark.asyncio
async def test_planner_weak_route_probe_bonus_can_reach_deeper_booking_ready_candidate(monkeypatch):
    async def fake_handoff(*args, **kwargs):
        flight = kwargs.get("flight") or {}
        if flight.get("flight_no") == "TA003":
            return {
                "url": "/booking/handoff/post/bridge-003",
                "source": "booking_token",
                "reason": "resolved_booking_request_post",
                "status": "ok",
                "provider": "serpapi",
                "handoff_mode": "provider_or_shareable",
                "artifact_field": "booking_request.post_followup_bridge",
                "requires_browser_post": True,
                "is_exact_handoff": False,
                "is_search_fallback": False,
                "is_provider_managed": True,
                "is_booking_quality_exit": True,
                "booking_exit_quality": "booking_ready",
            }
        return {
            "url": "https://www.google.com/travel/flights?q=fallback",
            "source": "google_flights_fallback",
            "reason": "no_booking_artifacts_google_search_fallback",
            "status": "ok",
            "provider": "google_flights",
            "handoff_mode": "search_fallback",
            "is_search_fallback": True,
            "is_provider_managed": False,
            "is_booking_quality_exit": False,
            "booking_exit_quality": "search_assist",
            "search_fallback_quality": "route_seeded_with_itinerary_hints",
        }

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        rows = []
        for idx, price in enumerate([4200, 4300, 4400, 4500], start=1):
            rows.append(
                {
                    "airline": "WeakRouteAir",
                    "flight_no": f"TA00{idx}",
                    "departure_time": f"{6 + idx:02d}:00",
                    "arrival_time": f"{8 + idx:02d}:10",
                    "duration_min": 130,
                    "price_inr": price,
                    "stops": 0,
                    "layover_info": "",
                    "baggage": "7kg cabin",
                    "date": date,
                }
            )
        return rows, {"_search_meta": {"raw_candidate_count": len(rows)}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {"condition": "Sunny", "temperature_c": 29, "forecast_date": travel_date}

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_LIMIT", 1)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_PROBE_LIMIT", 1)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_PROBE_MAX", 4)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_SCAN_LIMIT", 4)
    monkeypatch.setattr("agents.planner_agent.WEAK_ROUTE_HANDOFF_PROBE_BONUS", 2)
    monkeypatch.setattr("agents.planner_agent.ROUND_TRIP_HANDOFF_PROBE_BONUS", 0)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.2)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Find cheapest flights from Delhi to Mumbai",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.top_flights is not None
    assert len(result.top_flights) == 1
    assert result.top_flights[0]["booking_handoff"]["is_booking_quality_exit"] is False
    assert result.booking_handoff["is_booking_quality_exit"] is True
    assert result.booking_handoff["selected_flight_rank"] == 3


@pytest.mark.asyncio
async def test_planner_weak_route_low_signal_nonzero_can_probe_deeper_round_trip_candidate(monkeypatch):
    async def fake_handoff(*args, **kwargs):
        flight = kwargs.get("flight") or {}
        if flight.get("flight_no") == "TA003":
            return {
                "url": "/booking/handoff/post/bridge-003",
                "source": "booking_token",
                "reason": "resolved_booking_request_post",
                "status": "ok",
                "provider": "serpapi",
                "handoff_mode": "provider_or_shareable",
                "artifact_field": "booking_request.post_followup_bridge",
                "requires_browser_post": True,
                "is_exact_handoff": False,
                "is_search_fallback": False,
                "is_provider_managed": True,
                "is_booking_quality_exit": True,
                "booking_exit_quality": "booking_ready",
            }
        return {
            "url": "https://www.google.com/travel/flights?q=fallback",
            "source": "google_flights_fallback",
            "reason": "no_booking_artifacts_google_search_fallback",
            "status": "ok",
            "provider": "google_flights",
            "handoff_mode": "search_fallback",
            "is_search_fallback": True,
            "is_provider_managed": False,
            "is_booking_quality_exit": False,
            "booking_exit_quality": "search_assist",
            "search_fallback_quality": "route_seeded_with_itinerary_hints",
        }

    def fake_parse_intent(_query: str):
        return planner_agent.ParsedIntent(
            origin_iata=None,
            destination_iata=None,
            date=(datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d"),
            trip_type="Business",
        )

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        rows = []
        for idx, price in enumerate([4200, 4300, 4400, 4500], start=1):
            row = {
                "airline": "WeakRouteAir",
                "flight_no": f"TA00{idx}",
                "departure_time": f"{6 + idx:02d}:00",
                "arrival_time": f"{8 + idx:02d}:10",
                "duration_min": 130,
                "price_inr": price,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            }
            if idx == 1:
                # Non-zero but weak handoff signal.
                row["shareable_link"] = "https://partner.example/share/ta001"
            if idx == 3:
                row["booking_request"] = {"method": "POST", "url": "https://partner.example/post/ta003"}
            rows.append(row)
        return rows, {"_search_meta": {"raw_candidate_count": len(rows)}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {"condition": "Sunny", "temperature_c": 29, "forecast_date": travel_date}

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff)
    monkeypatch.setattr("agents.planner_agent.parse_intent", fake_parse_intent)
    monkeypatch.setattr(
        "agents.planner_agent._infer_route_pair_from_query",
        lambda _q: ("DEL", "BOM", {"source": "resolver_phrase_pair"}),
    )
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_LIMIT", 1)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_PROBE_LIMIT", 1)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_PROBE_MAX", 4)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_SCAN_LIMIT", 4)
    monkeypatch.setattr("agents.planner_agent.WEAK_ROUTE_HANDOFF_PROBE_BONUS", 2)
    monkeypatch.setattr("agents.planner_agent.ROUND_TRIP_HANDOFF_PROBE_BONUS", 0)
    monkeypatch.setattr("agents.planner_agent.WEAK_ROUTE_HANDOFF_SIGNAL_THRESHOLD", 6)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.2)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = await planner_agent._plan_trip_internal(
        date=date,
        user_query="Round-trip Delhi Mumbai returning in 3 days",
        trip_type="round-trip",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.top_flights is not None
    assert len(result.top_flights) == 1
    assert result.top_flights[0]["booking_handoff"]["is_booking_quality_exit"] is False
    assert result.booking_handoff["is_booking_quality_exit"] is True
    assert result.booking_handoff["selected_flight_rank"] == 3


@pytest.mark.asyncio
async def test_planner_round_trip_low_signal_without_weak_route_confidence_can_probe_deeper_candidate(monkeypatch):
    async def fake_handoff(*args, **kwargs):
        flight = kwargs.get("flight") or {}
        if flight.get("flight_no") == "TA003":
            return {
                "url": "/booking/handoff/post/bridge-003",
                "source": "booking_token",
                "reason": "resolved_booking_request_post",
                "status": "ok",
                "provider": "serpapi",
                "handoff_mode": "provider_or_shareable",
                "artifact_field": "booking_request.post_followup_bridge",
                "requires_browser_post": True,
                "is_exact_handoff": False,
                "is_search_fallback": False,
                "is_provider_managed": True,
                "is_booking_quality_exit": True,
                "booking_exit_quality": "booking_ready",
            }
        return {
            "url": "https://www.google.com/travel/flights?q=fallback",
            "source": "google_flights_fallback",
            "reason": "no_booking_artifacts_google_search_fallback",
            "status": "ok",
            "provider": "google_flights",
            "handoff_mode": "search_fallback",
            "is_search_fallback": True,
            "is_provider_managed": False,
            "is_booking_quality_exit": False,
            "booking_exit_quality": "search_assist",
            "search_fallback_quality": "route_seeded_with_itinerary_hints",
        }

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    return_date = (datetime.now().date() + timedelta(days=4)).strftime("%Y-%m-%d")

    def fake_parse_intent(_query: str):
        return planner_agent.ParsedIntent(
            origin_iata="DEL",
            destination_iata="BOM",
            date=date,
            return_date=return_date,
            trip_type="round-trip",
        )

    async def fake_search(*args, **kwargs):
        search_date = kwargs.get("date")
        rows = []
        for idx, price in enumerate([4200, 4300, 4400, 4500], start=1):
            row = {
                "airline": "WeakRoundTripAir",
                "flight_no": f"TA00{idx}",
                "departure_time": f"{6 + idx:02d}:00",
                "arrival_time": f"{8 + idx:02d}:10",
                "duration_min": 130,
                "price_inr": price,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": search_date,
            }
            if idx == 1:
                row["shareable_link"] = "https://partner.example/share/ta001"
            if idx == 3:
                row["booking_request"] = {"method": "POST", "url": "https://partner.example/post/ta003"}
            rows.append(row)
        return rows, {"_search_meta": {"raw_candidate_count": len(rows)}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {"condition": "Sunny", "temperature_c": 29, "forecast_date": travel_date}

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff)
    monkeypatch.setattr("agents.planner_agent.parse_intent", fake_parse_intent)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_LIMIT", 1)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_PROBE_LIMIT", 1)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_PROBE_MAX", 4)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_SCAN_LIMIT", 4)
    monkeypatch.setattr("agents.planner_agent.WEAK_ROUTE_HANDOFF_PROBE_BONUS", 0)
    monkeypatch.setattr("agents.planner_agent.ROUND_TRIP_HANDOFF_PROBE_BONUS", 0)
    monkeypatch.setattr("agents.planner_agent.WEAK_ROUTE_HANDOFF_SIGNAL_THRESHOLD", 6)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.2)
    monkeypatch.setattr("agents.planner_agent.ROUND_TRIP_HANDOFF_TIMEOUT_BONUS", 0.0)

    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Round-trip Delhi Mumbai",
        trip_type="round-trip",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.top_flights is not None
    assert len(result.top_flights) == 1
    assert result.top_flights[0]["booking_handoff"]["is_booking_quality_exit"] is False
    assert result.booking_handoff["is_booking_quality_exit"] is True
    assert result.booking_handoff["selected_flight_rank"] == 3


@pytest.mark.asyncio
async def test_planner_search_fallback_pool_visibility_prefers_stronger_fallback(monkeypatch):
    async def fake_handoff(*args, **kwargs):
        flight = kwargs.get("flight") or {}
        if flight.get("flight_no") == "TA002":
            return {
                "url": "https://www.google.com/travel/flights?q=seeded_itinerary",
                "source": "google_flights_fallback",
                "reason": "no_booking_artifacts_google_search_fallback",
                "status": "ok",
                "provider": "google_flights",
                "handoff_mode": "search_fallback",
                "is_search_fallback": True,
                "is_provider_managed": False,
                "is_booking_quality_exit": False,
                "booking_exit_quality": "search_assist",
                "search_fallback_quality": "route_seeded_with_itinerary_hints",
            }
        return {
            "url": "https://www.google.com/travel/flights?q=seeded_basic",
            "source": "google_flights_fallback",
            "reason": "no_booking_artifacts_google_search_fallback",
            "status": "ok",
            "provider": "google_flights",
            "handoff_mode": "search_fallback",
            "is_search_fallback": True,
            "is_provider_managed": False,
            "is_booking_quality_exit": False,
            "booking_exit_quality": "search_assist",
            "search_fallback_quality": "route_seeded_basic",
        }

    async def fake_search(*args, **kwargs):
        date = kwargs.get("date")
        return [
            {
                "airline": "FallbackAir",
                "flight_no": "TA001",
                "departure_time": "08:00",
                "arrival_time": "10:00",
                "duration_min": 120,
                "price_inr": 4000,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            },
            {
                "airline": "FallbackAir",
                "flight_no": "TA002",
                "departure_time": "09:00",
                "arrival_time": "11:00",
                "duration_min": 120,
                "price_inr": 4200,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
            },
        ], {"_search_meta": {"raw_candidate_count": 2}}

    async def fake_weather(*args, **kwargs):
        travel_date = kwargs.get("travel_date") or kwargs.get("date")
        return {"condition": "Clear", "temperature_c": 27, "forecast_date": travel_date}

    monkeypatch.setattr("agents.planner_agent._build_booking_handoff_url_safe", fake_handoff)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_LIMIT", 2)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_PROBE_LIMIT", 2)
    monkeypatch.setattr("agents.planner_agent.PER_FLIGHT_HANDOFF_TIMEOUT", 0.2)

    date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=date,
        user_query="Find flights from Delhi to Mumbai",
        skip_llm=True,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.booking_handoff["is_search_fallback"] is True
    assert result.booking_handoff["search_fallback_quality"] == "route_seeded_with_itinerary_hints"
    assert result.booking_handoff["selected_flight_rank"] == 2
    assert result.booking_handoff["search_fallback_pool"]["fallback_candidates"] == 2
    assert result.booking_handoff["search_fallback_pool"]["best_fallback_rank"] == 2
    assert result.booking_handoff["search_fallback_pool"]["probed_candidates"] == 2
    assert result.booking_handoff["search_fallback_pool"]["positive_signal_candidates"] == 0
    assert result.booking_handoff["search_fallback_pool"]["artifact_signal_candidates"] == 0
    assert result.booking_handoff["search_fallback_pool"]["max_probe_signal"] == 0
    assert result.booking_handoff["search_fallback_pool"]["signal_threshold"] == 6
    assert (
        result.booking_handoff["search_fallback_pool"]["quality_counts"]["route_seeded_with_itinerary_hints"]
        == 1
    )
    assert result.booking_handoff["search_fallback_pool"]["quality_counts"]["route_seeded_basic"] == 1


@pytest.mark.asyncio
async def test_build_booking_handoff_url_canonicalizes_resolved_booking_token_link(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": "https://www.google.com/url?q=https%3A%2F%2Fairline.example%2Fcheckout%3Fit%3Dabc",
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "status": "ok",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={"booking_token": "tok_123"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "booking_token"
    assert details["status"] == "ok"
    assert details["url"] == "https://airline.example/checkout?it=abc"
    assert details["is_exact_handoff"] is True
    assert details["is_search_fallback"] is False
    assert details["is_booking_quality_exit"] is True
    assert details["booking_exit_quality"] == "booking_ready"


def test_google_flights_fallback_is_differentiated_when_flight_hints_differ():
    url_a = build_google_flights_fallback(
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        flight={"airline": "AirOne", "flight_no": "AO101", "departure_time": "08:10", "arrival_time": "10:15"},
    )
    url_b = build_google_flights_fallback(
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        flight={"airline": "AirTwo", "flight_no": "AT202", "departure_time": "09:00", "arrival_time": "11:00"},
    )
    assert url_a != url_b


def test_google_flights_fallback_can_be_identical_when_itinerary_hints_are_identical():
    flight_hint = {"airline": "AirOne", "flight_no": "AO101", "departure_time": "08:10", "arrival_time": "10:15"}
    url_a = build_google_flights_fallback(
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        flight=flight_hint,
    )
    url_b = build_google_flights_fallback(
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        flight=flight_hint,
    )
    assert url_a == url_b


def test_google_flights_fallback_includes_layover_iata_hints_when_available():
    url = build_google_flights_fallback(
        origin="PNQ",
        destination="COK",
        depart_date="2026-04-22",
        flight={
            "airline": "HintAir",
            "flight_no": "HA404",
            "departure_time": "07:15",
            "arrival_time": "11:40",
            "stops": 1,
            "layover_airports": ["BLR", "ignored", "MAA"],
        },
    )
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    q_text = query.get("q", [""])[0]
    assert "via BLR MAA" in q_text


def test_google_flights_fallback_round_trip_query_includes_round_trip_hint():
    url = build_google_flights_fallback(
        origin="BLR",
        destination="GOI",
        depart_date="2026-04-22",
        return_date="2026-04-25",
    )
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    q_text = query.get("q", [""])[0]
    assert "round trip" in q_text
    assert "returning 2026-04-25" in q_text
    assert "return flight GOI to BLR on 2026-04-25" in q_text


@pytest.mark.asyncio
async def test_fetch_booking_options_retries_transient_payload_error(monkeypatch):
    responses = [
        {"error": "temporarily unavailable, try again"},
        {"booking_options": [{"provider": "RetryProvider", "price": 6100, "link": "https://retry.example/checkout"}]},
    ]

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class _FakeClient:
        call_count = 0

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            payload = responses[min(_FakeClient.call_count, len(responses) - 1)]
            _FakeClient.call_count += 1
            return _FakeResponse(payload)

    class _ReserveCtx:
        async def __aenter__(self):
            return (0, "dummy")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRIES", 2)
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRY_BACKOFF", 0.0)

    options = await booking_handoff.fetch_booking_options("tok_retry")
    assert options is not None
    assert len(options) == 1
    assert options[0]["provider"] == "RetryProvider"
    assert _FakeClient.call_count == 2


@pytest.mark.asyncio
async def test_fetch_booking_options_honors_attempt_budget_under_transient_http_errors(monkeypatch):
    class _FakeResponse:
        status_code = 503

        def json(self):
            return {"error": "temporarily unavailable"}

    class _FakeClient:
        call_count = 0

        async def get(self, *args, **kwargs):
            _FakeClient.call_count += 1
            return _FakeResponse()

    class _ReserveCtx:
        async def __aenter__(self):
            return (0, "dummy")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRIES", 5)
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_ATTEMPTS_BUDGET", 2)
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRY_BACKOFF", 0.0)

    options = await booking_handoff.fetch_booking_options("tok_budget")
    assert options is None
    assert _FakeClient.call_count == 2


@pytest.mark.asyncio
async def test_build_booking_handoff_url_rejects_wrapped_invalid_token_link(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": "https://www.google.com/url?q=javascript%3Aalert%281%29",
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "status": "ok",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={"booking_token": "tok_123"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "google_flights_fallback"
    assert details["reason"] == "booking_token_invalid_url_google_search_fallback"
    assert details["status"] == "ok"
    assert details["handoff_mode"] == "search_fallback"
    assert details["landing_guarantee"] == "best_effort_search"
    assert details["is_exact_handoff"] is False
    assert details["is_search_fallback"] is True
    assert details["is_booking_quality_exit"] is False
    assert details["booking_exit_quality"] == "search_assist"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_prefers_partner_link_when_token_unresolved(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={"booking_token": "tok_123", "provider_link": "https://partner.example/checkout/abc"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "shareable_link"
    assert details["reason"] == "booking_token_unresolved_fallback_partner_link"
    assert details["status"] == "ok"
    assert details["artifact_field"] == "provider_link"
    assert details["handoff_mode"] == "provider_or_shareable"
    assert details["landing_guarantee"] == "provider_managed"
    assert details["url"] == "https://partner.example/checkout/abc"
    assert details["is_exact_handoff"] is False
    assert details["is_search_fallback"] is False
    assert details["is_booking_quality_exit"] is True
    assert details["booking_exit_quality"] == "booking_ready"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_does_not_promote_generic_google_search_partner_link(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={
            "booking_token": "tok_123",
            "provider_link": "https://www.google.com/travel/flights?q=Flights+from+DEL+to+BOM+on+2026-03-20",
        },
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "google_flights_fallback"
    assert details["reason"] == "booking_token_unresolved_google_search_fallback"
    assert details["handoff_mode"] == "search_fallback"
    assert details["is_exact_handoff"] is False
    assert details["is_search_fallback"] is True
    assert details["is_booking_quality_exit"] is False
    assert details["booking_exit_quality"] == "search_assist"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_extracts_nested_partner_booking_option(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={
            "booking_token": "tok_nested",
            "booking_options": [
                {
                    "seller": {
                        "name": "NestedPartner",
                        "booking_url": "https://nested.example/checkout/xyz",
                    }
                }
            ],
        },
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "shareable_link"
    assert details["reason"] == "booking_token_unresolved_fallback_partner_link"
    assert details["status"] == "ok"
    assert details["artifact_field"].startswith("booking_options")
    assert details["url"] == "https://nested.example/checkout/xyz"
    assert details["is_booking_quality_exit"] is True
    assert details["booking_exit_quality"] == "booking_ready"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_uses_booking_request_direct_get_when_token_unresolved(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={
            "booking_token": "tok_direct_req",
            "booking_request": {
                "method": "GET",
                "url": "https://partner.example/checkout/direct",
            },
        },
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "shareable_link"
    assert details["reason"] == "booking_token_unresolved_fallback_partner_link"
    assert details["status"] == "ok"
    assert details["artifact_field"] == "booking_request.url"
    assert details["url"] == "https://partner.example/checkout/direct"
    assert details["handoff_mode"] == "provider_or_shareable"
    assert details["is_booking_quality_exit"] is True
    assert details["booking_exit_quality"] == "booking_ready"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_uses_booking_request_post_followup(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "provider": "serpapi",
        }

    async def fake_followup(_flight):
        return "https://partner.example/checkout/from_post", "booking_request.post_followup"

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)
    monkeypatch.setattr(booking_handoff, "_resolve_booking_request_handoff", fake_followup)

    details = await build_booking_handoff_url(
        flight={
            "booking_token": "tok_post_req",
            "booking_request": {
                "method": "POST",
                "url": "https://partner.example/book",
                "post_data": {"offer_id": "abc"},
            },
        },
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "shareable_link"
    assert details["reason"] == "booking_token_unresolved_fallback_partner_link"
    assert details["status"] == "ok"
    assert details["artifact_field"] == "booking_request.post_followup"
    assert details["url"] == "https://partner.example/checkout/from_post"
    assert details["handoff_mode"] == "provider_or_shareable"
    assert details["is_booking_quality_exit"] is True
    assert details["booking_exit_quality"] == "booking_ready"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_marks_google_search_fallback_quality(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={
            "booking_token": "tok_no_links",
            "airline": "IndiGo",
            "flight_no": "6E123",
            "departure_time": "09:05",
            "arrival_time": "11:30",
        },
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "google_flights_fallback"
    assert details["handoff_mode"] == "search_fallback"
    assert details["is_booking_quality_exit"] is False
    assert details["search_fallback_quality"] == "route_seeded_with_itinerary_hints"
    assert details["search_fallback_context"]["route_type"] == "one_way"
    assert details["search_fallback_context"]["has_route_seed"] is True
    assert details["search_fallback_context"]["has_itinerary_hints"] is True
    assert details["search_fallback_context"]["includes_return_leg_hint"] is False
    assert details["search_fallback_context"]["quality_tier"] == "route_seeded_with_itinerary_hints"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_round_trip_fallback_exposes_context_tier(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={"booking_token": "tok_rt_no_hints"},
        origin="BLR",
        destination="GOI",
        depart_date="2026-04-22",
        return_date="2026-04-25",
        return_details=True,
    )

    # Backward-compatible outward quality string remains stable.
    assert details["search_fallback_quality"] == "route_seeded_basic"
    # Rich round-trip quality is now visible via compact context payload.
    assert details["search_fallback_context"]["route_type"] == "round_trip"
    assert details["search_fallback_context"]["has_route_seed"] is True
    assert details["search_fallback_context"]["has_itinerary_hints"] is False
    assert details["search_fallback_context"]["includes_return_leg_hint"] is True
    assert (
        details["search_fallback_context"]["quality_tier"]
        == "round_trip_route_seeded_with_return_leg"
    )


@pytest.mark.asyncio
async def test_build_booking_handoff_url_does_not_treat_google_booking_request_url_as_provider_exit(monkeypatch):
    async def fake_resolve_details(_booking_token: str):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={
            "booking_token": "tok_google_req",
            "booking_request": {
                "method": "GET",
                "url": "https://www.google.com/travel/flights?q=Flights+from+DEL+to+BOM+on+2026-03-20",
            },
        },
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["source"] == "google_flights_fallback"
    assert details["handoff_mode"] == "search_fallback"
    assert details["is_booking_quality_exit"] is False


@pytest.mark.asyncio
async def test_fetch_booking_options_includes_search_context_params(monkeypatch):
    captured_params = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "booking_options": [
                    {"provider": "PartnerA", "price": 5000, "link": "https://partner.example/checkout/abc"}
                ]
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url, params=None, **kwargs):
            captured_params.update(params or {})
            return _FakeResponse()

    class _ReserveCtx:
        async def __aenter__(self):
            return (0, "dummy")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())

    options = await booking_handoff.fetch_booking_options(
        "tok_ctx",
        departure_id="PNQ",
        arrival_id="COK",
        outbound_date="2026-04-22",
    )
    assert options and len(options) == 1
    assert captured_params["booking_token"] == "tok_ctx"
    assert captured_params["engine"] == "google_flights"
    assert captured_params["departure_id"] == "PNQ"
    assert captured_params["arrival_id"] == "COK"
    assert captured_params["outbound_date"] == "2026-04-22"
    assert captured_params["type"] == "2"
    assert "departure_token" not in captured_params


@pytest.mark.asyncio
async def test_resolve_booking_token_with_details_preserves_post_booking_request(monkeypatch):
    async def fake_payload(**_kwargs):
        return {
            "selected_flights": [
                {
                    "booking_request": {
                        "url": "https://www.google.com/travel/clk/f",
                        "post_data": {"token": "abc123", "k": "v"},
                    }
                }
            ]
        }

    monkeypatch.setattr(booking_handoff, "_fetch_booking_options_payload", fake_payload)

    details = await booking_handoff.resolve_booking_token_with_details(
        "tok_post",
        departure_id="PNQ",
        arrival_id="COK",
        outbound_date="2026-04-22",
    )

    assert details["source"] == "booking_token"
    assert details["reason"] == "resolved_booking_request_post"
    assert details["handoff_mode"] == "provider_or_shareable"
    assert details["is_provider_managed"] is True
    assert details["is_booking_quality_exit"] is True
    assert details["artifact_field"] == "booking_request.post_followup_bridge"
    assert details["requires_browser_post"] is True
    assert details["url"].startswith("/booking/handoff/post/")

    artifact_id = details["url"].rsplit("/", 1)[-1]
    artifact = booking_handoff.consume_post_handoff_artifact(artifact_id)
    assert artifact is not None
    assert artifact["url"] == "https://www.google.com/travel/clk/f"
    assert artifact["post_data"]["token"] == "abc123"


def test_post_handoff_artifact_rejects_invalid_and_is_one_time():
    assert booking_handoff.register_post_handoff_artifact(
        url="javascript:alert(1)",
        post_data={"a": "b"},
    ) is None

    bridge_url = booking_handoff.register_post_handoff_artifact(
        url="https://partner.example/checkout",
        post_data={"a": "b"},
    )
    assert bridge_url and bridge_url.startswith("/booking/handoff/post/")
    artifact_id = bridge_url.rsplit("/", 1)[-1]
    first = booking_handoff.consume_post_handoff_artifact(artifact_id)
    second = booking_handoff.consume_post_handoff_artifact(artifact_id)
    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_fetch_booking_options_logs_actionable_context_on_exception(monkeypatch, caplog):
    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url, params=None, **kwargs):
            raise RuntimeError("boom")

    class _ReserveCtx:
        async def __aenter__(self):
            return (0, "dummy")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRIES", 1)

    with caplog.at_level("INFO"):
        with pytest.raises(booking_handoff.BookingOptionsFetchError) as exc:
            await booking_handoff._fetch_booking_options_payload(
                booking_token="tok_ctx",
                departure_id="PNQ",
                arrival_id="COK",
                outbound_date="2026-04-22",
            )

    assert exc.value.reason == "booking_options_request_exception"
    record = next(
        r
        for r in caplog.records
        if "fetch_booking_options" in r.message and "fallback" in r.message
    )
    assert record.exception_type == "RuntimeError"
    assert record.has_booking_token is True
    assert record.has_departure_id is True
    assert record.has_arrival_id is True
    assert record.has_outbound_date is True
    assert record.has_departure_token is False
    assert record.exception_bucket == "unexpected"
    assert record.route_type == "one_way"
    assert record.response_has_booking_options is None
    assert record.response_has_booking_request_url is None
    assert record.response_has_booking_request_post_data is None


@pytest.mark.asyncio
async def test_fetch_booking_options_no_active_key_logs_as_debug_in_candidate_probe_context(monkeypatch, caplog):
    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url, params=None, **kwargs):
            raise RuntimeError("No available keys for service: serpapi")

    class _ReserveCtx:
        async def __aenter__(self):
            return (0, "dummy")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRIES", 1)

    with caplog.at_level("DEBUG"):
        with pytest.raises(booking_handoff.BookingOptionsFetchError):
            await booking_handoff._fetch_booking_options_payload(
                booking_token="tok_no_key",
                departure_id="DEL",
                arrival_id="BOM",
                outbound_date="2026-05-01",
            )

    record = next(
        r for r in caplog.records
        if "fetch_booking_options expected exception; candidate fallback" in r.message
    )
    assert record.exception_bucket == "no_active_key"
    assert record.log_severity == "debug"
    assert record.levelname == "DEBUG"


@pytest.mark.asyncio
async def test_resolve_booking_token_with_details_surfaces_fetch_failure_reason(monkeypatch):
    async def broken_payload(**_kwargs):
        raise booking_handoff.BookingOptionsFetchError(
            "booking_options_request_exception",
            context={"has_departure_id": False, "exception_bucket": "network"},
        )

    monkeypatch.setattr(booking_handoff, "_fetch_booking_options_payload", broken_payload)

    details = await booking_handoff.resolve_booking_token_with_details("tok_fail")

    assert details["status"] == "unavailable"
    assert details["reason"] == "booking_options_request_exception"
    assert details["source"] == "booking_token"
    assert details["failure_bucket"] == "network"


@pytest.mark.asyncio
async def test_fetch_booking_options_unexpected_candidate_exception_is_downgraded_to_info(monkeypatch, caplog):
    class _FailClient:
        async def get(self, *_args, **_kwargs):
            raise RuntimeError("unexpected parser branch failure")

    class _ReserveCtx:
        async def __aenter__(self):
            return (0, "dummy")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FailClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRIES", 1)

    with caplog.at_level("INFO"):
        with pytest.raises(booking_handoff.BookingOptionsFetchError):
            await booking_handoff._fetch_booking_options_payload(
                booking_token="tok_unexpected_candidate",
                departure_id="DEL",
                arrival_id="BOM",
                outbound_date="2026-05-01",
            )

    record = next(
        r for r in caplog.records
        if "unexpected exception; candidate fallback" in r.message
    )
    assert record.exception_bucket == "unexpected"
    assert record.log_severity == "info"
    assert record.levelname == "INFO"


@pytest.mark.asyncio
async def test_resolve_booking_token_with_details_sets_cache_hit_on_cached_success(monkeypatch):
    token = "tok_cache_hit_case"

    async def payload_once(**_kwargs):
        return {
            "booking_options": [
                {"provider": "PartnerA", "price": 4000, "link": "https://partner.example/checkout/cache-hit"}
            ]
        }

    monkeypatch.setattr(booking_handoff, "_fetch_booking_options_payload", payload_once)

    first = await booking_handoff.resolve_booking_token_with_details(token)
    second = await booking_handoff.resolve_booking_token_with_details(token)

    assert first["status"] == "ok"
    assert first["cache_hit"] is False
    assert second["status"] == "ok"
    assert second["cache_hit"] is True
    assert second["reason"].endswith("_cache")


@pytest.mark.asyncio
async def test_fetch_booking_options_parse_error_is_bucketed_with_route_type(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def json(self):
            raise ValueError("invalid json payload")

    class _FakeClient:
        async def get(self, _url, params=None, **kwargs):
            return _FakeResponse()

    class _ReserveCtx:
        async def __aenter__(self):
            return (0, "dummy")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRIES", 1)

    with pytest.raises(booking_handoff.BookingOptionsFetchError) as exc:
        await booking_handoff._fetch_booking_options_payload(
            booking_token="tok_parse",
            departure_id="BLR",
            arrival_id="GOI",
            outbound_date="2026-04-22",
            return_date="2026-04-25",
        )

    assert exc.value.reason == "booking_options_parse_error"
    assert exc.value.context.get("exception_bucket") == "response_parse"
    assert exc.value.context.get("route_type") == "round_trip"


@pytest.mark.asyncio
async def test_booking_options_success_logs_key_source_and_masked_fingerprint(monkeypatch, caplog):
    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "booking_options": [
                    {"provider": "PartnerA", "price": 5000, "link": "https://partner.example/checkout/abc"}
                ]
            }

    class _FakeClient:
        async def get(self, _url, params=None, **kwargs):
            return _FakeResponse()

    class _ReserveCtx:
        async def __aenter__(self):
            return (2, "dummy-serpapi-key")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

        async def mark_exhausted(self, *args, **kwargs):
            return None

        async def record_usage(self, *args, **kwargs):
            return None

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRIES", 1)

    with caplog.at_level("INFO"):
        options = await booking_handoff.fetch_booking_options(
            "tok_ctx",
            departure_id="PNQ",
            arrival_id="COK",
            outbound_date="2026-04-22",
        )

    assert options and len(options) == 1
    record = next(r for r in caplog.records if "booking_options fetch succeeded" in r.message)
    assert record.key_source == "api_key_manager.reserve_key:serpapi"
    assert record.client_mode == "shared_get_client"
    assert isinstance(record.key_fp, str)
    assert len(record.key_fp) == 10
    assert record.has_departure_id is True
    assert record.has_arrival_id is True
    assert record.has_outbound_date is True
    assert record.has_departure_token is False


@pytest.mark.asyncio
async def test_fetch_booking_options_round_trip_uses_timeout_bonus(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "booking_options": [
                    {"provider": "PartnerA", "price": 5000, "link": "https://partner.example/checkout/rt"}
                ]
            }

    class _FakeClient:
        async def get(self, _url, params=None, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return _FakeResponse()

    class _ReserveCtx:
        async def __aenter__(self):
            return (2, "dummy-serpapi-key")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

        async def mark_exhausted(self, *args, **kwargs):
            return None

        async def record_usage(self, *args, **kwargs):
            return None

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRIES", 1)
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_HTTP_TIMEOUT", 2.2)
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_ROUND_TRIP_TIMEOUT_BONUS", 0.5)

    data = await booking_handoff._fetch_booking_options_payload(
        booking_token="tok_ctx",
        departure_id="BLR",
        arrival_id="GOI",
        outbound_date="2026-04-22",
        return_date="2026-04-25",
    )

    assert data is not None
    assert captured["timeout"] == pytest.approx(2.7)


@pytest.mark.asyncio
async def test_fetch_booking_options_surfaces_provider_error_distinctly(monkeypatch, caplog):
    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"error": "Missing departure_id parameter."}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url, params=None, **kwargs):
            return _FakeResponse()

    class _ReserveCtx:
        async def __aenter__(self):
            return (1, "dummy")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeKeyManager:
        def reserve_key(self, service_name: str):
            assert service_name == "serpapi"
            return _ReserveCtx()

        async def mark_exhausted(self, *args, **kwargs):
            return None

        async def record_usage(self, *args, **kwargs):
            return None

    monkeypatch.setattr(booking_handoff, "_get_booking_http_client", lambda: _FakeClient())
    monkeypatch.setattr(booking_handoff, "api_key_manager", _FakeKeyManager())
    monkeypatch.setattr(booking_handoff, "BOOKING_OPTIONS_RETRIES", 1)

    with caplog.at_level("INFO"):
        with pytest.raises(booking_handoff.BookingOptionsFetchError) as exc:
            await booking_handoff._fetch_booking_options_payload(
                booking_token="tok_ctx",
                departure_id="PNQ",
                arrival_id="COK",
                outbound_date="2026-04-22",
            )

    assert exc.value.reason == "booking_options_provider_error"
    record = next(r for r in caplog.records if "booking_options provider error after shaped request" in r.message)
    assert record.has_booking_token is True
    assert record.has_departure_id is True
    assert record.has_arrival_id is True
    assert record.has_outbound_date is True
    assert record.response_has_booking_options is False
    assert record.response_has_selected_flights is False


@pytest.mark.asyncio
async def test_resolve_booking_token_with_details_accepts_nested_together_booking_request(monkeypatch):
    async def fake_payload(**_kwargs):
        return {
            "booking_options": [
                {
                    "together": {
                        "booking_request": {
                            "url": "https://www.google.com/travel/clk/f",
                            "post_data": {"token": "nested"},
                        }
                    }
                }
            ]
        }

    monkeypatch.setattr(booking_handoff, "_fetch_booking_options_payload", fake_payload)

    details = await booking_handoff.resolve_booking_token_with_details(
        "tok_nested",
        departure_id="PNQ",
        arrival_id="COK",
        outbound_date="2026-04-22",
    )

    assert details["status"] == "ok"
    assert details["reason"] == "resolved_booking_request_post"
    assert details["artifact_field"] == "booking_request.post_followup_bridge"
    assert details["requires_browser_post"] is True
    assert details["url"].startswith("/booking/handoff/post/")


@pytest.mark.asyncio
async def test_resolve_booking_token_with_details_reuses_cached_booking_request_post(monkeypatch):
    call_count = {"n": 0}

    async def fake_payload(**_kwargs):
        call_count["n"] += 1
        return {
            "selected_flights": [
                {
                    "booking_request": {
                        "url": "https://www.google.com/travel/clk/f",
                        "post_data": {"token": "stable"},
                    }
                }
            ]
        }

    monkeypatch.setattr(booking_handoff, "_fetch_booking_options_payload", fake_payload)

    first = await booking_handoff.resolve_booking_token_with_details(
        "tok_cache",
        departure_id="PNQ",
        arrival_id="COK",
        outbound_date="2026-04-22",
    )
    second = await booking_handoff.resolve_booking_token_with_details(
        "tok_cache",
        departure_id="PNQ",
        arrival_id="COK",
        outbound_date="2026-04-22",
    )

    assert call_count["n"] == 1
    assert first["reason"] == "resolved_booking_request_post"
    assert second["reason"] == "resolved_booking_request_post_cache"
    assert first["url"].startswith("/booking/handoff/post/")
    assert second["url"].startswith("/booking/handoff/post/")
    assert first["url"] != second["url"]


def test_booking_token_resolve_timeout_covers_retry_budget_floor():
    expected_floor = (
        booking_handoff.BOOKING_OPTIONS_HTTP_TIMEOUT * booking_handoff.BOOKING_OPTIONS_ATTEMPTS_BUDGET
        + booking_handoff.BOOKING_OPTIONS_RETRY_BACKOFF
        * max(0, booking_handoff.BOOKING_OPTIONS_ATTEMPTS_BUDGET - 1)
        + 0.5
    )
    assert booking_handoff.BOOKING_TOKEN_RESOLVE_TIMEOUT >= expected_floor


def test_post_handoff_ttl_floor_supports_manual_first_click_window():
    # Keep a practical minimum so fresh bridges remain usable for normal manual validation.
    assert booking_handoff.POST_HANDOFF_TTL_SECONDS >= 180


def test_post_handoff_persistent_consume_is_single_winner_under_concurrency(monkeypatch, tmp_path):
    # Keep this test environment-independent: force a deterministic, file-backed DB
    # so we exercise the persistent consume path regardless of local .env state.
    sqlite_path = tmp_path / "booking_handoff_concurrency.sqlite3"
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("TESTING_USE_PERSISTENT_DB", "1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{sqlite_path}")
    monkeypatch.setenv("POST_HANDOFF_REQUIRE_PERSISTENCE", "1")
    monkeypatch.setattr(booking_handoff, "POST_HANDOFF_REQUIRE_PERSISTENCE", True)

    import agents.database as database

    # agents.database caches engine/session globals; reset so our test env is honored.
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    booking_handoff.ensure_tables()

    # Clear in-memory cache so registration/consume behavior is isolated to this test.
    booking_handoff._post_handoff_artifacts.clear()
    bridge_url = booking_handoff.register_post_handoff_artifact(
        url="https://www.google.com/travel/clk/f",
        post_data={"token": "atomic-once"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert bridge_url is not None
    artifact_id = bridge_url.rsplit("/", 1)[-1]
    assert sqlite_path.exists()

    # Verify registration actually reached the persistent table.
    session = database.SessionLocal()
    try:
        row = session.query(booking_handoff.PostHandoffArtifact).filter(
            booking_handoff.PostHandoffArtifact.artifact_id == artifact_id
        ).first()
        assert row is not None
    finally:
        session.close()

    # Force both attempts through the persistent consume path (not memory fallback).
    booking_handoff._post_handoff_artifacts.clear()

    consume_results = []
    consume_errors = []
    consume_lock = threading.Lock()
    start_barrier = threading.Barrier(3)

    def _consume_once():
        try:
            start_barrier.wait(timeout=5)
            result = booking_handoff.consume_post_handoff_artifact_with_diagnostics(artifact_id)
            with consume_lock:
                consume_results.append(result)
        except Exception as exc:
            with consume_lock:
                consume_errors.append(exc)

    first_worker = threading.Thread(target=_consume_once, name="post-handoff-consume-1")
    second_worker = threading.Thread(target=_consume_once, name="post-handoff-consume-2")
    first_worker.start()
    second_worker.start()
    start_barrier.wait(timeout=5)
    first_worker.join(timeout=5)
    second_worker.join(timeout=5)

    assert first_worker.is_alive() is False
    assert second_worker.is_alive() is False
    assert consume_errors == []
    assert len(consume_results) == 2

    successes = [item for item in consume_results if item[0] is not None]
    failures = [item for item in consume_results if item[0] is None]

    assert len(successes) == 1
    assert successes[0][1]["lookup_result"] == "persistent_hit"
    assert len(failures) == 1
    assert failures[0][1]["lookup_result"] in {"already_consumed", "consume_race_lost"}
    assert failures[0][1]["lookup_result"] != "lookup_failed"


def test_register_post_handoff_artifact_rejects_non_persistent_when_required(monkeypatch):
    artifact_id = "fixed-artifact-id"

    class _FixedUUID:
        hex = artifact_id

    monkeypatch.setattr(booking_handoff.uuid, "uuid4", lambda: _FixedUUID())
    monkeypatch.setattr(booking_handoff, "POST_HANDOFF_REQUIRE_PERSISTENCE", True)
    monkeypatch.setattr(
        booking_handoff,
        "_store_post_handoff_artifact_persistent",
        lambda **_kwargs: False,
    )
    booking_handoff._post_handoff_artifacts.pop(artifact_id, None)

    bridge_url = booking_handoff.register_post_handoff_artifact(
        url="https://www.google.com/travel/clk/f",
        post_data={"token": "must-persist"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert bridge_url is None
    assert artifact_id not in booking_handoff._post_handoff_artifacts


def test_candidate_fallback_log_throttle_emits_first_and_periodic_summary(monkeypatch):
    booking_handoff._candidate_fallback_log_counts.clear()
    monkeypatch.setattr(booking_handoff, "get_request_id", lambda: "req-throttle")

    emits = []
    for _ in range(6):
        should_emit, occurrence = booking_handoff._should_emit_candidate_fallback_log(
            exception_bucket="provider_rate_limited",
            token_fp="tok123",
            route_type="round_trip",
            candidate_probe_context=True,
        )
        emits.append((should_emit, occurrence))

    # first event emits
    assert emits[0] == (True, 1)
    # middle repeats are suppressed
    assert all(not emits[i][0] for i in (1, 2, 3))
    # periodic summary emits on the 5th
    assert emits[4] == (True, 5)
    # then suppress again
    assert emits[5][0] is False
