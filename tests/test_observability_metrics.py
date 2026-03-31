import re
import json

import httpx
import pytest
from prometheus_client import generate_latest

from api.app import app
import agents.planner_agent as planner_agent
import core.metrics as metrics


def test_observability_helpers_emit_metric_series():
    metrics.record_stream_init_timeout("unknown")
    metrics.record_stream_cancellation("ollama", "planner_consume")
    metrics.record_stream_fallback("stream_init_timeout", "unknown")
    metrics.record_llm_route_usage("ollama_first", "ollama_only", "none", stream=True)
    metrics.observe_llm_first_token("ollama", 0.15)
    metrics.observe_llm_full_response("ollama", stream=True, duration_sec=0.92)
    metrics.record_stream_done_json("success")

    text = generate_latest().decode()
    assert "stream_init_timeout_total" in text
    assert "stream_cancellations_total" in text
    assert "stream_fallback_total" in text
    assert "llm_route_usage_total" in text
    assert "llm_first_token_latency_seconds" in text
    assert "llm_full_response_latency_seconds" in text
    assert "stream_done_json_total" in text


@pytest.mark.asyncio
async def test_stream_done_json_metric_increments_for_streaming_action(monkeypatch):
    monkeypatch.setattr("agents.planner_agent._cancel_booking_safe", lambda *_args, **_kwargs: True)

    gen = await planner_agent.plan_trip(user_query="cancel booking 42", stream=True)
    combined = ""
    async for chunk in gen:
        combined += chunk
        if "[DONE_JSON]" in combined:
            break

    assert "[DONE_JSON]" in combined
    text = generate_latest().decode()
    assert 'stream_done_json_total{status="action"}' in text


@pytest.mark.asyncio
async def test_http_metrics_tracks_routes_but_excludes_metrics_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        version_resp = await client.get("/version")
        assert version_resp.status_code == 200
        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200

    text = metrics_resp.text
    assert re.search(r'http_requests_total\{[^}]*route="/version"', text)
    assert not re.search(r'http_requests_total\{[^}]*route="/metrics"', text)


@pytest.mark.asyncio
async def test_stream_mid_interruption_records_fallback_metric(monkeypatch):
    async def fake_plan_trip_internal(**_kwargs):
        return planner_agent.PlanResult(
            llm_response=None,
            best_flight={
                "airline": "TestAir",
                "flight_no": "TA777",
                "departure_time": "06:00",
                "arrival_time": "08:00",
                "duration_min": 120,
                "price_inr": 5000,
                "stops": 0,
                "baggage": "7kg cabin",
                "date": "2026-03-20",
            },
            weather={
                "condition": "Clear",
                "temperature_c": 28,
                "feels_like_c": 30,
                "humidity": 60,
                "wind_kph": 12,
                "air_quality_index": 2,
            },
            search_date="2026-03-20",
            warnings=[],
            debug_info={
                "intent": {
                    "origin_iata": "DEL",
                    "destination_iata": "BOM",
                    "date": "2026-03-20",
                },
                "route_labels": {
                    "origin_iata": "DEL",
                    "origin_city": "New Delhi",
                    "origin_label": "New Delhi (DEL)",
                    "destination_iata": "BOM",
                    "destination_city": "Mumbai",
                    "destination_label": "Mumbai (BOM)",
                },
                "all_flights": [
                    {
                        "airline": "TestAir",
                        "flight_no": "TA777",
                        "departure_time": "06:00",
                        "arrival_time": "08:00",
                        "duration_min": 120,
                        "price_inr": 5000,
                        "stops": 0,
                        "baggage": "7kg cabin",
                        "date": "2026-03-20",
                    }
                ],
                "filters_applied": "none",
                "trip_description": "a business trip",
                "price_insights_str": "",
                "price_analysis_str": "",
                "price_prediction_str": "",
            },
        )

    class _BrokenTokenStream:
        def __init__(self):
            self._sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._sent:
                self._sent = True
                return "First token "
            raise RuntimeError("simulated mid-stream failure")

    async def fake_generate(**_kwargs):
        return _BrokenTokenStream()

    async def fake_check_llm_circuit(*_args, **_kwargs):
        return False

    monkeypatch.setattr(planner_agent, "_plan_trip_internal", fake_plan_trip_internal)
    monkeypatch.setattr(planner_agent, "generate", fake_generate)
    monkeypatch.setattr(planner_agent, "check_llm_circuit", fake_check_llm_circuit)

    gen = await planner_agent.plan_trip(
        origin="DEL",
        destination="BOM",
        date="2026-03-20",
        user_query="delhi to mumbai",
        stream=True,
    )

    combined = ""
    async for chunk in gen:
        combined += str(chunk)
        if "[DONE_JSON]" in combined:
            break

    assert "[DONE_JSON]" in combined
    assert '"destination_label"' in combined and "Mumbai (BOM)" in combined
    done_chunks = [part for part in combined.split("[DONE_JSON]") if part.strip()]
    assert done_chunks
    parsed = json.loads(done_chunks[-1])
    assert parsed.get("result_status") == "degraded"
    assert parsed.get("degradation", {}).get("reason") in {"upstream_unavailable", "upstream_timeout"}

    metrics_text = generate_latest().decode()
    assert 'stream_fallback_total{provider="unknown",reason="mid_stream_interruption"}' in metrics_text
