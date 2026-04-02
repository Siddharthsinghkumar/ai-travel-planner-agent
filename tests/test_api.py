import asyncio
import json
import time
import httpx
import pytest
from datetime import datetime, timedelta
import api.app as api_app
from api.app import app
import tools.booking_handoff as booking_handoff

future_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")

@pytest.mark.asyncio
async def test_ask_endpoint(monkeypatch):

    async def fake_plan_trip(**kwargs):
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/ask", json={
            "origin": "DEL",
            "destination": "BOM",
            "date": future_date,
            "user_query": "Business trip"
        })

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_ask_duplicate_non_stream_returns_explicit_duplicate_in_progress(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"count": 0}

    async def fake_plan_trip(**_kwargs):
        calls["count"] += 1
        started.set()
        await release.wait()
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setenv("ASK_MAX_INFLIGHT", "8")
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}}

    payload = {
        "origin": "DEL",
        "destination": "BOM",
        "date": future_date,
        "user_query": "Business trip",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        primary_task = asyncio.create_task(client.post("/ask", json=payload))
        await asyncio.wait_for(started.wait(), timeout=1.0)

        duplicate = await client.post("/ask", json=payload)
        release.set()
        primary = await primary_task

    assert primary.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json()["error"] == "duplicate_request_in_progress"
    assert duplicate.headers.get("X-Ask-Admission") == "duplicate_in_progress"
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_ask_duplicate_stream_returns_explicit_duplicate_in_progress(monkeypatch):
    finish_stream = asyncio.Event()
    calls = {"count": 0}

    async def fake_plan_trip(**kwargs):
        calls["count"] += 1
        if kwargs.get("stream"):
            async def _gen():
                yield "partial chunk"
                await finish_stream.wait()
                yield "[DONE_JSON]" + '{"result":"ok"}'

            return _gen()
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setenv("ASK_MAX_INFLIGHT", "8")
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}}

    req = api_app.AskRequest(
        origin="DEL",
        destination="BOM",
        date=future_date,
        user_query="Business trip",
    )

    primary_stream = await api_app.ask(req=req, stream=True, async_job=False)
    assert primary_stream.status_code == 200
    stream_iter = primary_stream.body_iterator
    first_chunk = await stream_iter.__anext__()
    if isinstance(first_chunk, bytes):
        assert b"partial chunk" in first_chunk
    else:
        assert "partial chunk" in str(first_chunk)

    duplicate = await api_app.ask(req=req, stream=True, async_job=False)
    assert duplicate.status_code == 409
    duplicate_payload = json.loads(duplicate.body.decode("utf-8"))
    assert duplicate_payload["error"] == "duplicate_request_in_progress"
    assert duplicate.headers.get("X-Ask-Admission") == "duplicate_in_progress"

    finish_stream.set()
    async for _chunk in stream_iter:
        pass

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_ask_backpressure_returns_429_when_inflight_limit_reached(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"count": 0}

    async def fake_plan_trip(**_kwargs):
        calls["count"] += 1
        started.set()
        await release.wait()
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setenv("ASK_MAX_INFLIGHT", "1")
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}}

    primary_payload = {
        "origin": "DEL",
        "destination": "BOM",
        "date": future_date,
        "user_query": "Business trip",
    }
    secondary_payload = {
        "origin": "DEL",
        "destination": "BLR",
        "date": future_date,
        "user_query": "Holiday trip",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        primary_task = asyncio.create_task(client.post("/ask", json=primary_payload))
        await asyncio.wait_for(started.wait(), timeout=1.0)

        overloaded = await client.post("/ask", json=secondary_payload)
        release.set()
        primary = await primary_task

    assert primary.status_code == 200
    assert overloaded.status_code == 429
    assert overloaded.json()["error"] == "ask_overloaded"
    assert overloaded.headers.get("X-Ask-Admission") == "overloaded"
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_ask_stale_inflight_marker_is_pruned_and_does_not_block(monkeypatch):
    calls = {"count": 0}

    async def fake_plan_trip(**_kwargs):
        calls["count"] += 1
        return {"result": "fresh"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setenv("ASK_INFLIGHT_STALE_SECONDS", "1")
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}}

    payload = {
        "origin": "DEL",
        "destination": "BOM",
        "date": future_date,
        "user_query": "Business trip",
    }
    fingerprint = api_app._build_ask_request_fingerprint(
        origin=payload["origin"],
        destination=payload["destination"],
        date=payload["date"],
        user_query=payload["user_query"],
        trip_type=None,
        llm_mode=None,
        cloud_provider=None,
        stream=False,
    )
    app.state.ask_runtime_state["inflight"][fingerprint] = {
        "owner_request_id": "stale-request",
        "stream": False,
        "started_at": time.monotonic() - 500,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/ask", json=payload)

    assert response.status_code == 200
    assert response.json()["result"] == "fresh"
    assert calls["count"] == 1
    assert app.state.ask_runtime_state["inflight"] == {}


@pytest.mark.asyncio
async def test_async_job_rejected_when_multi_worker_topology_declared(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("ASYNC_JOB_REQUIRE_SINGLE_WORKER", "1")
    monkeypatch.delenv("ALLOW_UNSAFE_ASYNC_JOBS", raising=False)
    monkeypatch.setattr(app.state, "async_job_support", None, raising=False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ask?async_job=true",
            json={
                "date": future_date,
                "user_query": "delhi to mumbai",
                "trip_type": "Business",
            },
        )

    assert response.status_code == 503
    payload = response.json()
    detail = payload["detail"]
    assert detail["error"] == "async_job_topology_unsupported"
    assert detail["reason"] == "unsupported_multi_worker_topology"
    assert detail["declared_workers"] == 2


@pytest.mark.asyncio
async def test_async_job_allowed_with_explicit_unsafe_override(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("ASYNC_JOB_REQUIRE_SINGLE_WORKER", "1")
    monkeypatch.setenv("ALLOW_UNSAFE_ASYNC_JOBS", "1")
    monkeypatch.setattr(app.state, "async_job_support", None, raising=False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_resp = await client.post(
            "/ask?async_job=true",
            json={
                "date": future_date,
                "user_query": "delhi to mumbai",
                "trip_type": "Business",
            },
        )
        health_resp = await client.get("/health")

    assert create_resp.status_code == 202
    assert "job_id" in create_resp.json()
    assert health_resp.status_code == 200
    topology = health_resp.json()["runtime_topology"]
    assert topology["async_jobs_enabled"] is True
    assert topology["async_job_support"]["reason"] == "unsafe_override_enabled"
    assert topology["async_job_support"]["allow_unsafe_override"] is True
    assert topology["async_job_support"]["contract"] == "single_worker_required_process_local_queue"


@pytest.mark.asyncio
async def test_booking_post_handoff_bridge_rejects_unknown_artifact():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/booking/handoff/post/does-not-exist")

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["error"] == "booking_handoff_artifact_unavailable"
    assert payload["detail"]["lookup_result"] == "not_found"


@pytest.mark.asyncio
async def test_booking_post_handoff_bridge_serves_autosubmit_form():
    bridge_url = booking_handoff.register_post_handoff_artifact(
        url="https://provider.example/checkout",
        post_data={"token": "abc123", "fare": "X1"},
    )
    assert bridge_url is not None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(bridge_url)

    assert response.status_code == 200
    assert response.headers.get("X-Booking-Bridge-Consume-Result") in {"memory_hit", "persistent_hit"}
    body = response.text
    assert "form id='handoff'" in body
    assert "https://provider.example/checkout" in body
    assert "name=\"token\"" in body
    assert "name=\"fare\"" in body


@pytest.mark.asyncio
async def test_booking_post_handoff_bridge_persists_across_memory_cache_clear_and_is_one_time():
    bridge_url = booking_handoff.register_post_handoff_artifact(
        url="https://provider.example/checkout",
        post_data={"token": "abc123"},
    )
    assert bridge_url is not None
    artifact_id = bridge_url.rsplit("/", 1)[-1]

    # Simulate request landing on a different worker/process memory state.
    booking_handoff._post_handoff_artifacts.clear()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first = await client.get(f"/booking/handoff/post/{artifact_id}")
        second = await client.get(f"/booking/handoff/post/{artifact_id}")

    assert first.status_code == 200
    assert "form id='handoff'" in first.text
    assert second.status_code == 404
    second_payload = second.json()
    assert second_payload["detail"]["lookup_result"] in {"already_consumed", "consume_race_lost"}


@pytest.mark.asyncio
async def test_booking_post_handoff_bridge_concurrent_fresh_has_single_winner():
    bridge_url = booking_handoff.register_post_handoff_artifact(
        url="https://provider.example/checkout",
        post_data={"token": "abc123"},
    )
    assert bridge_url is not None
    artifact_id = bridge_url.rsplit("/", 1)[-1]

    # Simulate cross-worker behavior (persistent path only).
    booking_handoff._post_handoff_artifacts.clear()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first, second = await asyncio.gather(
            client.get(f"/booking/handoff/post/{artifact_id}"),
            client.get(f"/booking/handoff/post/{artifact_id}"),
        )

    assert sorted([first.status_code, second.status_code]) == [200, 404]


@pytest.mark.asyncio
async def test_lightweight_health_includes_external_dependency_note():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "external_dependency_checks" in body
    checks = body["external_dependency_checks"]
    assert checks["checked"] is False
    assert checks["deep_endpoint"] == "/health/deep"


@pytest.mark.asyncio
async def test_lightweight_health_runtime_topology_includes_role_clarity(monkeypatch):
    app.state.startup_complete = True
    app.state.key_manager_refresh_owner = False
    app.state.async_job_support = {
        "enabled": False,
        "reason": "unsupported_multi_worker_topology",
        "declared_workers": 2,
        "guard_active": True,
    }
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setattr("api.app.get_llm_mode_default", lambda: "ollama_only")
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: False)
    monkeypatch.setattr("api.app.key_manager.status", lambda: {"openai": [{"active": True}]})

    async def fake_ollama():
        return True

    monkeypatch.setattr("api.app._check_ollama_availability_for_options", fake_ollama)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    topology = body["runtime_topology"]
    assert topology["refresh_owner"] is False
    assert topology["worker_role"] == "follower"
    assert topology["async_jobs_enabled"] is False
    assert topology["async_job_support"]["reason"] == "unsupported_multi_worker_topology"


@pytest.mark.asyncio
async def test_health_deep_is_dependency_truth_while_health_stays_lightweight(monkeypatch):
    fake_deep = {
        "status": "degraded",
        "dependencies": {"airline": "fail"},
        "key_gate_issues": [],
        "key_status_assumptions": [{"provider": "airline", "assumption": "missing_key_status_assumed_active"}],
        "messages": {"airline": "down"},
    }

    async def fake_full_health_check():
        return fake_deep

    monkeypatch.setattr("api.app.full_health_check", fake_full_health_check)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        light = await client.get("/health")
        deep = await client.get("/health/deep")

    assert light.status_code == 200
    light_body = light.json()
    assert light_body["external_dependency_checks"]["checked"] is False
    assert light_body["external_dependency_checks"]["deep_endpoint"] == "/health/deep"

    assert deep.status_code == 200
    deep_body = deep.json()
    assert deep_body["status"] == "degraded"
    assert deep_body["dependencies"]["airline"] == "fail"
    assert deep_body["key_status_assumptions"]


@pytest.mark.asyncio
async def test_lightweight_health_cloud_only_mode_requires_cloud(monkeypatch):
    app.state.startup_complete = True
    monkeypatch.setattr("api.app.get_llm_mode_default", lambda: "cloud_only")
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: True)

    async def fake_usable():
        return []

    async def fake_ollama():
        return True

    monkeypatch.setattr("api.app.get_usable_providers", fake_usable)
    monkeypatch.setattr("api.app._check_ollama_availability_for_options", fake_ollama)
    monkeypatch.setattr("api.app.key_manager.status", lambda: {"openai": [{"active": True}]})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    body = response.json()
    assert response.status_code == 200
    assert body["llm_mode"] == "cloud_only"
    assert body["primary_llm_backend"] == "cloud"
    assert body["fallback_llm_backend"] is None
    assert body["dependencies"]["cloud"] == "unavailable"
    assert body["dependencies"]["ollama"] == "not_relevant"
    assert body["health_basis"]["required_unavailable"] == ["cloud"]
    assert body["status"] == "fail"


@pytest.mark.asyncio
async def test_lightweight_health_ollama_first_degrades_when_primary_unavailable_but_fallback_ok(monkeypatch):
    app.state.startup_complete = True
    monkeypatch.setattr("api.app.get_llm_mode_default", lambda: "ollama_first")
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: True)

    async def fake_usable():
        return ["gemini"]

    async def fake_ollama():
        return False

    monkeypatch.setattr("api.app.get_usable_providers", fake_usable)
    monkeypatch.setattr("api.app._check_ollama_availability_for_options", fake_ollama)
    monkeypatch.setattr("api.app.key_manager.status", lambda: {"openai": [{"active": True}]})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    body = response.json()
    assert response.status_code == 200
    assert body["llm_mode"] == "ollama_first"
    assert body["primary_llm_backend"] == "ollama"
    assert body["fallback_llm_backend"] == "cloud"
    assert body["dependencies"]["ollama"] == "unavailable"
    assert body["dependencies"]["cloud"] == "ok"
    assert body["health_basis"]["required_unavailable"] == ["ollama"]
    assert body["health_basis"]["fallback_unavailable"] == []
    assert body["status"] == "degraded"


@pytest.mark.asyncio
async def test_readiness_reports_warming_while_llm_prewarm_is_running():
    app.state.startup_complete = True
    app.state.llm_prewarm = {
        "enabled": True,
        "best_effort": True,
        "status": "running",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "warming"
    assert body["llm_prewarm"]["enabled"] is True
    assert body["llm_prewarm"]["status"] == "running"


@pytest.mark.asyncio
async def test_lightweight_health_marks_ollama_primary_prewarm_failure_as_degraded(monkeypatch):
    app.state.startup_complete = True
    app.state.llm_prewarm = {
        "enabled": True,
        "best_effort": True,
        "status": "failed",
        "attempts": 3,
        "last_error": "timeout",
        "last_updated": "2026-03-23T00:00:00Z",
        "model": "openhermes",
    }
    monkeypatch.setattr("api.app.get_llm_mode_default", lambda: "ollama_only")
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: False)
    monkeypatch.setattr("api.app.key_manager.status", lambda: {"openai": [{"active": True}]})

    async def fake_ollama():
        return True

    monkeypatch.setattr("api.app._check_ollama_availability_for_options", fake_ollama)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["health_basis"]["llm_prewarm_status"] == "failed"
    assert body["llm_prewarm"]["enabled"] is True
    assert body["llm_prewarm"]["status"] == "failed"


@pytest.mark.asyncio
async def test_lightweight_health_degrades_when_key_status_empty(monkeypatch):
    app.state.startup_complete = True
    monkeypatch.setattr("api.app.get_llm_mode_default", lambda: "ollama_only")
    monkeypatch.setattr("api.app.is_cloud_admin_enabled", lambda: False)
    monkeypatch.setattr("api.app.key_manager.status", lambda: {})

    async def fake_ollama():
        return True

    monkeypatch.setattr("api.app._check_ollama_availability_for_options", fake_ollama)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["dependencies"]["key_manager"] == "degraded"
    assert body["status"] == "degraded"
    assert body["health_basis"]["key_manager"]["reason"] == "empty_key_status"


@pytest.mark.asyncio
async def test_ask_unexpected_error_returns_generic_500(monkeypatch):
    async def fake_plan_trip(**kwargs):
        raise RuntimeError("sensitive backend details")

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ask",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "user_query": "Business trip",
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"


@pytest.mark.asyncio
async def test_ask_normalizes_route_trip_mode_override(monkeypatch):
    captured = {}

    async def fake_plan_trip(**kwargs):
        captured.update(kwargs)
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ask",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "user_query": "Cheapest flight",
                "trip_type": "round-trip",
            },
        )

    assert response.status_code == 200
    assert captured.get("trip_type") == "round-trip"


@pytest.mark.asyncio
async def test_ask_normalizes_via_stopover_trip_mode_override(monkeypatch):
    captured = {}

    async def fake_plan_trip(**kwargs):
        captured.update(kwargs)
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ask",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "user_query": "Cheapest flight",
                "trip_type": "via / stopover",
            },
        )

    assert response.status_code == 200
    assert captured.get("trip_type") == "via-stopover"


@pytest.mark.asyncio
async def test_ask_preserves_semantic_trip_type_override(monkeypatch):
    captured = {}

    async def fake_plan_trip(**kwargs):
        captured.update(kwargs)
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ask",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "user_query": "Business trip",
                "trip_type": "business",
            },
        )

    assert response.status_code == 200
    assert captured.get("trip_type") == "Business"


@pytest.mark.asyncio
async def test_ask_rejects_unknown_trip_type_value():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ask",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "user_query": "Cheapest flight",
                "trip_type": "roundtrip-typo",
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ask_non_stream_warning_fallback_is_not_success(monkeypatch):
    async def fake_plan_trip(**kwargs):
        return {
            "warning": "No live flights found.",
            "fallback": True,
            "failure_reason": "no_flights",
            "no_flights_reason": "no_inventory",
            "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
        }

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ask",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "user_query": "Find flights",
            },
        )

    assert response.status_code == 400
    assert response.json().get("detail") == "No live flights found."
    assert response.json().get("failure_reason") == "no_flights"
    assert response.json().get("failure_domain") == "search_outcome"
    assert response.json().get("no_flights_reason") == "no_inventory"
    assert response.json().get("result_status") == "error"
    assert response.json().get("flight_counts", {}).get("pre_filter") == 0


@pytest.mark.asyncio
async def test_ask_non_stream_empty_error_is_not_success(monkeypatch):
    async def fake_plan_trip(**kwargs):
        return {"error": "", "failure_reason": "planner_error"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ask",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "user_query": "Find flights",
            },
        )

    assert response.status_code == 400
    assert response.json().get("detail") == "Planner failed to produce a complete response."
    assert response.json().get("failure_reason") == "planner_error"
    assert response.json().get("failure_domain") == "internal_backend"
    assert response.json().get("result_status") == "error"
