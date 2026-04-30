import re
import json
import asyncio

import httpx
import pytest
from prometheus_client import generate_latest

from api.app import app
import api.app as api_app
import agents.planner_agent as planner_agent
import agents.cloud_llm as cloud_llm
import tools.booking_handoff as booking_handoff
import core.metrics as metrics
from core.api_key_manager import APIKeyManager, KeyEntry
from core.circuit_breaker import AsyncCircuitBreaker


def test_observability_helpers_emit_metric_series():
    metrics.record_stream_init_timeout("unknown")
    metrics.record_stream_cancellation("ollama", "planner_consume")
    metrics.record_stream_fallback("stream_init_timeout", "unknown")
    metrics.record_llm_route_usage("ollama_first", "ollama_only", "none", stream=True)
    metrics.observe_llm_first_token("ollama", 0.15)
    metrics.observe_llm_full_response("ollama", stream=True, duration_sec=0.92)
    metrics.record_stream_done_json("success")
    metrics.record_retry_budget_exhausted("unit_test")
    metrics.record_booking_handoff_consume("memory_hit", "hit")
    metrics.record_key_state_event("weather", "exhausted", "rate_limit")
    metrics.record_provider_health_failure("openai", "auth")
    metrics.record_provider_health_cooldown_skip("openai")
    metrics.record_circuit_transition("closed_or_half_open_to_open")

    text = generate_latest().decode()
    assert "stream_init_timeout_total" in text
    assert "stream_cancellations_total" in text
    assert "stream_fallback_total" in text
    assert "llm_route_usage_total" in text
    assert "llm_first_token_latency_seconds" in text
    assert "llm_full_response_latency_seconds" in text
    assert "stream_done_json_total" in text
    assert "retry_budget_exhausted_total" in text
    assert "booking_handoff_consume_total" in text
    assert "key_state_events_total" in text
    assert "provider_health_failures_total" in text
    assert "provider_health_cooldown_skips_total" in text
    assert "circuit_transitions_total" in text


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
        metrics_resp = await client.get("/metrics", headers={"X-Admin-Token": "admin-test-token"})
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


@pytest.mark.asyncio
async def test_hardening_admission_metrics_capture_duplicate_and_overload_contracts(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_plan_trip(**_kwargs):
        started.set()
        await release.wait()
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setenv("ASK_MAX_INFLIGHT", "1")
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}}

    payload_primary = {
        "origin": "DEL",
        "destination": "BOM",
        "date": "2026-05-20",
        "user_query": "primary inflight request",
    }
    payload_overload = {
        "origin": "DEL",
        "destination": "BLR",
        "date": "2026-05-20",
        "user_query": "secondary overload request",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        primary_task = asyncio.create_task(client.post("/ask", json=payload_primary))
        await asyncio.wait_for(started.wait(), timeout=1.0)

        duplicate = await client.post("/ask", json=payload_primary)
        overload = await client.post("/ask", json=payload_overload)
        release.set()
        primary = await primary_task

    assert primary.status_code == 200
    assert duplicate.status_code == 409
    assert overload.status_code == 429

    text = generate_latest().decode()
    assert 'ask_duplicates_total{outcome="in_progress",stream="false"}' in text
    assert 'ask_admission_total{outcome="rejected_duplicate",stream="false"}' in text
    assert 'ask_admission_total{outcome="rejected_overload",stream="false"}' in text


@pytest.mark.asyncio
async def test_hardening_admission_metrics_capture_recent_replay_contract(monkeypatch):
    calls = {"count": 0}

    async def fake_plan_trip(**_kwargs):
        calls["count"] += 1
        return {"result": "ok", "attempt": calls["count"]}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setenv("ASK_RECENT_COMPLETION_TTL_SECONDS", "5")
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}}

    payload = {
        "origin": "DEL",
        "destination": "BOM",
        "date": "2026-05-21",
        "user_query": "replay me",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first = await client.post("/ask", json=payload)
        second = await client.post("/ask", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers.get("X-Ask-Admission") == "replayed_recent"
    assert calls["count"] == 1

    text = generate_latest().decode()
    assert 'ask_duplicates_total{outcome="recent_replay",stream="false"}' in text
    assert 'ask_admission_total{outcome="replayed_recent",stream="false"}' in text


@pytest.mark.asyncio
async def test_hardening_metrics_cover_booking_consume_and_provider_health_classes(monkeypatch):
    monkeypatch.setenv("USE_CLOUD_LLM", "1")
    bridge = booking_handoff.register_post_handoff_artifact(
        url="https://provider.example/checkout",
        post_data={"token": "obs-metric"},
    )
    assert bridge is not None
    artifact_id = bridge.rsplit("/", 1)[-1]
    consumed, _diag_hit = booking_handoff.consume_post_handoff_artifact_with_diagnostics(artifact_id)
    assert consumed is not None
    _, diag_miss = booking_handoff.consume_post_handoff_artifact_with_diagnostics(artifact_id)
    assert diag_miss["consume_outcome"] == "miss"

    class _Adapter:
        async def ping(self, _model):
            raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(cloud_llm, "provider_chain", [("openai", _Adapter(), (Exception,))])
    monkeypatch.setattr(cloud_llm, "refresh_provider_chain_from_env", lambda force=False: None)

    async def _usable():
        return ["openai"]

    monkeypatch.setattr(cloud_llm, "get_usable_providers", _usable)
    cloud_llm._PROVIDER_FAIL_COOLDOWNS.clear()

    first = await cloud_llm.health_check()
    second = await cloud_llm.health_check()
    assert first == "fail"
    assert second == "fail"

    text = generate_latest().decode()
    assert re.search(
        r'booking_handoff_consume_total\{lookup_result="(memory_hit|persistent_hit)",outcome="hit"\}',
        text,
    )
    assert f'booking_handoff_consume_total{{lookup_result="{diag_miss["lookup_result"]}",outcome="miss"}}' in text
    assert 'provider_health_failures_total{provider="openai",reason_class="auth"}' in text
    assert 'provider_health_cooldown_skips_total{provider="openai"}' in text


@pytest.mark.asyncio
async def test_hardening_metrics_cover_key_state_and_circuit_transitions(monkeypatch):
    km = APIKeyManager()
    key = "obs-weather-key"
    km._keys = {"weather": [KeyEntry(value=key, fingerprint=km._fingerprint(key), exhausted_until=None)]}
    km._rr_index = {"weather": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    await km.mark_exhausted("weather", 0, reason="http_429")
    await km.clear_exhausted("weather", 0)

    breaker = AsyncCircuitBreaker(failure_threshold=1, recovery_timeout=0.0, half_open_max_calls=1)

    async def fail():
        raise RuntimeError("boom")

    async def ok():
        return "ok"

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    await asyncio.sleep(0)
    assert await breaker.call(ok) == "ok"

    text = generate_latest().decode()
    assert 'key_state_events_total{event="exhausted",reason_class="rate_limit",service="weather"}' in text
    assert 'key_state_events_total{event="recovered",reason_class="rate_limit",service="weather"}' in text
    assert 'circuit_transitions_total{transition="closed_or_half_open_to_open"}' in text
    assert 'circuit_transitions_total{transition="half_open_to_closed"}' in text
