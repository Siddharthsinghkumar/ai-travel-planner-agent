import asyncio
import contextlib
from datetime import datetime, timedelta

import pytest

import agents.planner_agent as planner_agent
import tools.booking_handoff as booking_handoff
from tools.booking_handoff import build_booking_handoff_url


class _FakeQuery:
    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def all(self):
        return []


class _FakeSession:
    def __init__(self, query_result=None):
        self._query_result = query_result or []

    def add(self, _obj):
        return None

    def commit(self):
        return None

    def refresh(self, booking_obj):
        booking_obj.id = 4242

    def close(self):
        return None

    def query(self, *_args, **_kwargs):
        fq = _FakeQuery()
        fq._result = self._query_result
        fq.all = lambda: self._query_result
        return fq


def _future_date(days: int = 1) -> str:
    return (datetime.now().date() + timedelta(days=days)).strftime("%Y-%m-%d")


@pytest.mark.asyncio
async def test_build_booking_handoff_url_requires_booking_token_no_fallback():
    details = await build_booking_handoff_url(
        flight={"shareable_link": "https://www.google.com/travel/flights?q=DEL+BOM"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details["status"] == "unavailable"
    assert details["reason"] == "booking_token_missing"
    assert details["url"] is None
    assert details["booking_exit_quality"] == "unavailable"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_returns_booking_ready_details(monkeypatch):
    async def fake_resolve_details(_booking_token: str, **_kwargs):
        return {
            "url": "https://airline.example/checkout/abc",
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "booking_exit_quality": "booking_ready",
            "provider": "serpapi",
            "cache_hit": True,
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    details = await build_booking_handoff_url(
        flight={"booking_token": "tok_123"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=True,
    )

    assert details == {
        "status": "booking_ready",
        "reason": "resolved_booking_token",
        "source": "booking_token",
        "url": "https://airline.example/checkout/abc",
        "booking_exit_quality": "booking_ready",
        "provider": "serpapi",
        "cache_hit": True,
    }


@pytest.mark.asyncio
async def test_build_booking_handoff_url_unavailable_when_token_unresolved(monkeypatch):
    async def fake_resolve_details(_booking_token: str, **_kwargs):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "booking_exit_quality": "unavailable",
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

    assert details["status"] == "unavailable"
    assert details["source"] == "booking_token"
    assert details["reason"] == "booking_token_unresolved"
    assert details["provider"] == "serpapi"
    assert details["url"] is None
    assert details["booking_exit_quality"] == "unavailable"


@pytest.mark.asyncio
async def test_build_booking_handoff_url_returns_none_without_details_when_unavailable(monkeypatch):
    async def fake_resolve_details(_booking_token: str, **_kwargs):
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "booking_exit_quality": "unavailable",
            "provider": "serpapi",
        }

    monkeypatch.setattr(booking_handoff, "resolve_booking_token_with_details", fake_resolve_details)

    url = await build_booking_handoff_url(
        flight={"booking_token": "tok_123"},
        origin="DEL",
        destination="BOM",
        depart_date="2026-03-20",
        return_details=False,
    )

    assert url is None


@pytest.mark.asyncio
async def test_hold_booking_resolves_when_booking_handoff_not_ready(monkeypatch):
    build_calls = {"count": 0}

    async def fake_build_booking_handoff_url(**_kwargs):
        build_calls["count"] += 1
        return None

    monkeypatch.setattr(booking_handoff, "build_booking_handoff_url", fake_build_booking_handoff_url)
    monkeypatch.setattr(booking_handoff, "SessionLocal", lambda: _FakeSession())

    flight = {
        "flight_no": "TA5252",
        "booking_handoff": {
            "booking_exit_quality": "deferred",
            "url": "https://partner.example/old",
        },
        "handoff_url": "https://partner.example/old",
    }

    held = await booking_handoff.hold_booking(
        flight=flight,
        origin="DEL",
        destination="BOM",
        depart_date="2026-06-12",
    )

    assert held["handoff_url"] is None
    assert held["checkout_ready"] is False
    assert held["checkout_status"] == "provider_handoff_unavailable"
    assert held["hold_outcome"] == "held_local_only"
    assert build_calls["count"] == 1


@pytest.mark.asyncio
async def test_hold_booking_persists_route_primitives_for_tracker(monkeypatch):
    captured = {}

    async def fake_build_booking_handoff_url(**_kwargs):
        return {
            "status": "unavailable",
            "reason": "booking_token_unresolved",
            "source": "booking_token",
            "url": None,
            "booking_exit_quality": "unavailable",
            "provider": "serpapi",
        }

    class _CapturingSession(_FakeSession):
        def add(self, obj):
            captured["booking_obj"] = obj
            return None

        def refresh(self, booking_obj):
            booking_obj.id = 6161

    monkeypatch.setattr(booking_handoff, "build_booking_handoff_url", fake_build_booking_handoff_url)
    monkeypatch.setattr(booking_handoff, "SessionLocal", lambda: _CapturingSession())

    held = await booking_handoff.hold_booking(
        flight={"flight_no": "TA6161", "price_inr": 5400},
        origin="DEL",
        destination="BOM",
        depart_date="2026-06-18",
    )

    persisted = captured["booking_obj"].flight
    assert held["checkout_ready"] is False
    assert held["hold_outcome"] == "held_local_only"
    assert persisted["origin"] == "DEL"
    assert persisted["destination"] == "BOM"
    assert persisted["date"] == "2026-06-18"
    assert persisted["departure_iata"] == "DEL"
    assert persisted["arrival_iata"] == "BOM"
    assert isinstance(persisted.get("booking_handoff"), dict)


def test_get_booking_normalizes_checkout_status_without_handoff_url(monkeypatch):
    class _BookingObj:
        id = 707
        status = "HELD"
        flight = {"booking_handoff": {"status": "unavailable"}}
        passenger = None
        booking_token = "tok_707"
        shareable_link = None
        handoff_url = None
        created_at = datetime.utcnow()
        expires_at = datetime.utcnow() + timedelta(minutes=10)

    class _GetSession(_FakeSession):
        def get(self, _model, booking_id):
            if booking_id == 707:
                return _BookingObj()
            return None

    monkeypatch.setattr(booking_handoff, "SessionLocal", lambda: _GetSession())

    payload = booking_handoff.get_booking(707)
    assert payload is not None
    assert payload["checkout_ready"] is False
    assert payload["checkout_status"] == "provider_handoff_unavailable"
    assert payload["hold_outcome"] == "held_local_only"


@pytest.mark.asyncio
async def test_planner_resolve_flight_booking_handoff_timeout_is_unavailable(monkeypatch):
    async def slow_handoff(*_args, **_kwargs):
        await asyncio.sleep(0.02)
        return {
            "url": "https://airline.example/late",
            "source": "booking_token",
            "reason": "resolved_booking_token",
            "status": "booking_ready",
            "booking_exit_quality": "booking_ready",
        }

    monkeypatch.setattr(planner_agent, "_build_booking_handoff_url_safe", slow_handoff)

    _, handoff_meta, handoff_url = await planner_agent._resolve_flight_booking_handoff(
        flight_obj={
            "airline": "TestAir",
            "flight_no": "TA100",
            "departure_time": "09:00",
            "arrival_time": "11:00",
            "duration_min": 120,
            "price_inr": 5500,
            "stops": 0,
            "baggage": "7kg",
            "booking_token": "tok_100",
        },
        origin="DEL",
        destination="BOM",
        depart_date=_future_date(2),
        return_date=None,
        timeout_sec=0.001,
    )

    assert handoff_url is None
    assert handoff_meta == {
        "url": None,
        "source": "booking_token",
        "reason": "booking_handoff_timeout",
        "status": "unavailable",
        "booking_exit_quality": "unavailable",
    }


@pytest.mark.asyncio
async def test_plan_trip_internal_search_only_deferred_handoff_contract_is_lean(monkeypatch):
    async def should_not_run_handoff(*_args, **_kwargs):
        raise AssertionError("booking handoff resolution should stay deferred in search-only mode")

    async def fake_search(**kwargs):
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
                "booking_token": "tok_ta001",
            }
        ], {"_search_meta": {"raw_candidate_count": 1}}

    async def fake_weather(*, location: str, travel_date: str, **kwargs):
        return {"condition": "Clear", "temperature_c": 28, "forecast_date": travel_date, "location": location}

    monkeypatch.setattr(planner_agent, "_build_booking_handoff_url_safe", should_not_run_handoff)

    result = await planner_agent._plan_trip_internal(
        origin="DEL",
        destination="BOM",
        date=_future_date(2),
        user_query="show me flights",
        skip_llm=True,
        resolve_booking_handoff=False,
        flight_tool=fake_search,
        weather_tool=fake_weather,
    )

    assert result.booking_handoff == {
        "url": None,
        "source": "deferred",
        "reason": "deferred_until_booking_intent",
        "status": "deferred",
        "booking_exit_quality": "deferred",
    }
    assert "selected_booking_handoff" not in result.best_flight
    assert "selected_booking_handoff_quality_context" not in result.best_flight
    assert "search_assist_url" not in result.best_flight
    assert "fallback_search_url" not in result.best_flight

    assert isinstance(result.top_flights, list) and len(result.top_flights) == 1
    assert isinstance(result.all_flights, list) and len(result.all_flights) == 1
    row_handoff = result.top_flights[0]["booking_handoff"]
    assert row_handoff == result.booking_handoff
    assert "handoff_mode" not in row_handoff
    assert "landing_guarantee" not in row_handoff
    assert "quality_summary" not in row_handoff
    assert "next_step_hint" not in row_handoff
    assert "failure_bucket" not in row_handoff


@pytest.mark.asyncio
async def test_plan_trip_internal_round_trip_uses_round_trip_block_not_legacy_contract(monkeypatch):
    outbound_date = _future_date(2)
    return_date = _future_date(5)

    monkeypatch.setattr(
        planner_agent,
        "parse_intent",
        lambda _q: planner_agent.ParsedIntent(
            origin_iata="DEL",
            destination_iata="BOM",
            date=outbound_date,
            return_date=return_date,
            trip_type="round-trip",
        ),
    )

    async def fake_handoff(*_args, **kwargs):
        flight = kwargs.get("flight") or {}
        if flight.get("flight_no") == "TA-OUT":
            return {
                "url": "https://partner.example/checkout/outbound",
                "source": "booking_token",
                "reason": "resolved_booking_token",
                "status": "booking_ready",
                "booking_exit_quality": "booking_ready",
            }
        return {
            "url": None,
            "source": "booking_token",
            "reason": "booking_token_unresolved",
            "status": "unavailable",
            "booking_exit_quality": "unavailable",
        }

    async def fake_flight_tool(**kwargs):
        departure = kwargs.get("departure")
        arrival = kwargs.get("arrival")
        date = kwargs.get("date")
        flight_no = "TA-OUT" if (departure, arrival) == ("DEL", "BOM") else "TA-RET"
        return [
            {
                "airline": "TestAir",
                "flight_no": flight_no,
                "departure_time": "09:00",
                "arrival_time": "11:00",
                "duration_min": 120,
                "price_inr": 4800,
                "stops": 0,
                "layover_info": "",
                "baggage": "7kg cabin",
                "date": date,
                "booking_token": f"tok-{flight_no}",
            }
        ], {"_search_meta": {"raw_candidate_count": 1}}

    async def fake_weather_tool(*, location: str, travel_date: str):
        return {
            "condition": "Clear",
            "temperature_c": 29,
            "forecast_date": travel_date,
            "location": location,
        }

    monkeypatch.setattr(planner_agent, "_build_booking_handoff_url_safe", fake_handoff)

    result = await planner_agent._plan_trip_internal(
        user_query="round trip del bom",
        trip_type="round-trip",
        skip_llm=True,
        flight_tool=fake_flight_tool,
        weather_tool=fake_weather_tool,
    )

    assert isinstance(result, planner_agent.PlanResult)
    assert "round_trip_contract" not in (result.booking_handoff or {})
    rt = (result.booking_handoff or {}).get("round_trip") or {}
    assert set(rt.keys()) == {
        "return_search_outcome",
        "return_search_reason",
        "return_handoff_status",
        "is_outbound_only_handoff",
    }
