import json

import pytest

import agents.planner_agent as planner_agent


@pytest.mark.asyncio
async def test_stream_error_emits_error_prefix_and_done_json(monkeypatch):
    async def fake_plan_trip_internal(**kwargs):
        return {"error": "Could not determine route"}

    monkeypatch.setattr(planner_agent, "_plan_trip_internal", fake_plan_trip_internal)

    agen = await planner_agent.plan_trip(
        origin="DEL",
        destination="BOM",
        date="2026-04-20",
        user_query="find me flights",
        stream=True,
    )

    chunks: list[str] = []
    async for chunk in agen:
        chunks.append(str(chunk))
        if "[DONE_JSON]" in str(chunk):
            break

    assert any(part.startswith("[ERROR] Could not determine route") for part in chunks)
    done_parts = [part for part in chunks if part.startswith("[DONE_JSON]")]
    assert done_parts
    done_payload = json.loads(done_parts[-1].replace("[DONE_JSON]", "", 1))
    assert done_payload.get("error") == "Could not determine route"


@pytest.mark.asyncio
async def test_stream_warning_dict_is_not_treated_as_success(monkeypatch):
    async def fake_plan_trip_internal(**kwargs):
        return {"warning": "No live flights found.", "fallback": True}

    monkeypatch.setattr(planner_agent, "_plan_trip_internal", fake_plan_trip_internal)

    agen = await planner_agent.plan_trip(
        origin="DEL",
        destination="BOM",
        date="2026-04-20",
        user_query="find me flights",
        stream=True,
    )

    chunks: list[str] = []
    async for chunk in agen:
        chunks.append(str(chunk))
        if "[DONE_JSON]" in str(chunk):
            break

    assert any(part.startswith("[ERROR] No live flights found.") for part in chunks)
    done_parts = [part for part in chunks if part.startswith("[DONE_JSON]")]
    assert done_parts
    done_payload = json.loads(done_parts[-1].replace("[DONE_JSON]", "", 1))
    assert done_payload.get("error") == "No live flights found."


@pytest.mark.asyncio
async def test_stream_llm_unavailable_returns_degraded_structured_done_json(monkeypatch):
    async def fake_plan_trip_internal(**kwargs):
        return planner_agent.PlanResult(
            llm_response="Deterministic summary",
            best_flight={
                "airline": "TestAir",
                "flight_no": "TA100",
                "departure_time": "10:00",
                "arrival_time": "12:00",
                "duration_min": 120,
                "price_inr": "₹5000",
                "stops": 0,
            },
            weather={"condition": "Cloudy", "temperature_c": 25},
            search_date="2026-04-20",
        )

    async def fake_check_llm_circuit():
        return True

    monkeypatch.setattr(planner_agent, "_plan_trip_internal", fake_plan_trip_internal)
    monkeypatch.setattr(planner_agent, "check_llm_circuit", fake_check_llm_circuit)

    agen = await planner_agent.plan_trip(
        origin="DEL",
        destination="BOM",
        date="2026-04-20",
        user_query="find me flights",
        stream=True,
    )

    chunks: list[str] = []
    async for chunk in agen:
        chunks.append(str(chunk))
        if "[DONE_JSON]" in str(chunk):
            break

    assert not any(part.startswith("[ERROR]") for part in chunks)
    done_parts = [part for part in chunks if part.startswith("[DONE_JSON]")]
    assert done_parts
    done_payload = json.loads(done_parts[-1].replace("[DONE_JSON]", "", 1))
    assert done_payload.get("result_status") == "degraded"
    assert done_payload.get("degradation", {}).get("reason") == "upstream_unavailable"
    assert "LLM explanation degraded" in str(done_payload.get("fallback_note") or "")
    assert done_payload.get("best_flight", {}).get("flight_no") == "TA100"


@pytest.mark.asyncio
async def test_stream_thinking_only_heartbeat_without_visible_tokens_degrades(monkeypatch):
    async def fake_plan_trip_internal(**kwargs):
        return planner_agent.PlanResult(
            llm_response=None,
            best_flight={
                "airline": "TestAir",
                "flight_no": "TA200",
                "departure_time": "09:00",
                "arrival_time": "11:00",
                "duration_min": 120,
                "price_inr": "₹6200",
                "stops": 0,
                "date": "2026-04-20",
                "baggage": "7kg cabin",
            },
            weather={"condition": "Sunny", "temperature_c": 30},
            search_date="2026-04-20",
            debug_info={
                "intent": {"origin_iata": "DEL", "destination_iata": "BOM", "date": "2026-04-20"},
                "all_flights": [
                    {
                        "airline": "TestAir",
                        "flight_no": "TA200",
                        "departure_time": "09:00",
                        "arrival_time": "11:00",
                        "duration_min": 120,
                        "price_inr": "₹6200",
                        "stops": 0,
                        "baggage": "7kg cabin",
                        "date": "2026-04-20",
                    }
                ],
                "filters_applied": "none",
                "trip_description": "a one-way trip from DEL to BOM",
            },
        )

    async def fake_check_llm_circuit(*args, **kwargs):
        return False

    async def fake_generate(*args, **kwargs):
        async def _stream():
            yield ""
            yield ""
        return _stream()

    monkeypatch.setattr(planner_agent, "_plan_trip_internal", fake_plan_trip_internal)
    monkeypatch.setattr(planner_agent, "check_llm_circuit", fake_check_llm_circuit)
    monkeypatch.setattr(planner_agent, "generate", fake_generate)

    agen = await planner_agent.plan_trip(
        origin="DEL",
        destination="BOM",
        date="2026-04-20",
        user_query="find me flights",
        stream=True,
    )

    chunks: list[str] = []
    async for chunk in agen:
        chunks.append(str(chunk))
        if "[DONE_JSON]" in str(chunk):
            break

    done_parts = [part for part in chunks if part.startswith("[DONE_JSON]")]
    assert done_parts
    done_payload = json.loads(done_parts[-1].replace("[DONE_JSON]", "", 1))
    assert done_payload.get("result_status") == "degraded"
    assert done_payload.get("degradation", {}).get("reason") == "upstream_stream_no_visible_tokens"
    assert "no visible answer text" in str(done_payload.get("fallback_note") or "")
