import asyncio
import json
import time
import httpx
import pytest
from datetime import datetime, timedelta
import api.app as api_app
import agents.planner_agent as planner_agent
from api.app import app
import tools.booking_handoff as booking_handoff

future_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")


@pytest.fixture(autouse=True)
def _reset_ask_runtime_state_per_test():
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}, "recent_completed": {}}
    yield
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}, "recent_completed": {}}


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
async def test_booking_hold_and_list_endpoints(monkeypatch):
    async def fake_hold_booking(**_kwargs):
        return {
            "id": 101,
            "status": "HELD",
            "handoff_url": "https://example.com/booking",
            "checkout_ready": True,
            "checkout_status": "booking_ready",
            "hold_outcome": "held_with_checkout",
            "expires_at": "2030-01-01T00:00:00Z",
        }

    def fake_list_bookings(status=None, limit=100, owner_principal_id=None):
        return [
            {
                "id": 101,
                "status": "HELD",
                "handoff_url": "https://example.com/booking",
            }
        ]

    def fake_get_booking(booking_id: int, owner_principal_id=None):
        return {
            "id": booking_id,
            "status": "HELD",
            "handoff_url": "https://example.com/booking",
        }

    monkeypatch.setattr("tools.booking_handoff.hold_booking", fake_hold_booking)
    monkeypatch.setattr("tools.booking_handoff.list_bookings", fake_list_bookings)
    monkeypatch.setattr("tools.booking_handoff.get_booking", fake_get_booking)

    payload = {
        "flight": {"airline": "TestAir", "flight_no": "TA100", "price_inr": 5200},
        "origin": "DEL",
        "destination": "BOM",
        "depart_date": future_date,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/booking/hold", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "hold_booking"
        assert data["hold_created"] is True
        assert data["checkout_ready"] is True
        assert data["checkout_status"] == "booking_ready"
        assert data["hold_outcome"] == "held_with_checkout"
        assert data["booking"]["id"] == 101

        list_resp = await client.get("/bookings?limit=2")
        assert list_resp.status_code == 200
        assert list_resp.json()["items"][0]["id"] == 101

        get_resp = await client.get("/bookings/101")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == 101


@pytest.mark.asyncio
async def test_booking_hold_local_only_semantics_are_explicit(monkeypatch):
    async def fake_hold_booking(**_kwargs):
        return {
            "id": 102,
            "status": "HELD",
            "handoff_url": None,
            "checkout_ready": False,
            "checkout_status": "provider_handoff_unavailable",
            "hold_outcome": "held_local_only",
            "expires_at": "2030-01-01T00:00:00Z",
        }

    monkeypatch.setattr("tools.booking_handoff.hold_booking", fake_hold_booking)

    payload = {
        "flight": {"airline": "TestAir", "flight_no": "TA101", "price_inr": 5100},
        "origin": "DEL",
        "destination": "BOM",
        "depart_date": future_date,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/booking/hold", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["hold_created"] is True
    assert data["checkout_ready"] is False
    assert data["checkout_status"] == "provider_handoff_unavailable"
    assert data["hold_outcome"] == "held_local_only"
    assert data["booking"]["handoff_url"] is None


@pytest.mark.asyncio
async def test_ask_sync_contract_includes_non_null_all_flights(monkeypatch):
    async def fake_plan_trip(**_kwargs):
        return planner_agent.PlanResult(
            llm_response="ok",
            best_flight={
                "airline": "TestAir",
                "flight_no": "TA100",
                "departure_time": "09:00",
                "arrival_time": "11:00",
                "duration_min": 120,
                "price_inr": 5200,
                "stops": 0,
                "baggage": "7kg cabin",
            },
            weather={"condition": "Clear", "temperature_c": 29},
            search_date=future_date,
            top_flights=[
                {
                    "airline": "TestAir",
                    "flight_no": "TA100",
                    "departure_time": "09:00",
                    "arrival_time": "11:00",
                    "duration_min": 120,
                    "price_inr": "₹5,200",
                    "stops": 0,
                    "baggage": "7kg cabin",
                }
            ],
            all_flights=[
                {
                    "airline": "TestAir",
                    "flight_no": "TA100",
                    "departure_time": "09:00",
                    "arrival_time": "11:00",
                    "duration_min": 120,
                    "price_inr": "₹5,200",
                    "stops": 0,
                    "baggage": "7kg cabin",
                }
            ],
        )

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)

    payload = {
        "origin": "DEL",
        "destination": "BOM",
        "date": future_date,
        "user_query": "cheapest flight please",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/ask", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("all_flights"), list)
    assert len(data["all_flights"]) == 1
    assert data["all_flights"][0]["flight_no"] == "TA100"


@pytest.mark.asyncio
async def test_booking_confirm_endpoint_removed_and_cancel_remains_available(monkeypatch):
    def fake_cancel(_booking_id: int, owner_principal_id=None):
        return True

    monkeypatch.setattr("tools.booking_handoff.cancel_booking", fake_cancel)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/booking/confirm", json={"booking_id": 201})
        assert resp.status_code == 404

        resp = await client.post("/booking/cancel", json={"booking_id": 201})
        assert resp.status_code == 200
        assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_booking_track_price(monkeypatch):
    async def fake_hold_booking(**_kwargs):
        return {
            "id": 202,
            "status": "HELD",
            "handoff_url": "https://example.com/booking",
            "expires_at": "2030-01-01T00:00:00Z",
        }

    snapshot_calls = []

    async def fake_record_price_snapshot(**kwargs):
        snapshot_calls.append(kwargs)
        return 1

    def fake_get_booking(booking_id: int, owner_principal_id=None):
        return {
            "id": booking_id,
            "flight": {
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "price_inr": 5200,
            },
        }

    monkeypatch.setattr("tools.booking_handoff.hold_booking", fake_hold_booking)
    monkeypatch.setattr("tools.booking_handoff.get_booking", fake_get_booking)
    monkeypatch.setattr("tools.price_tracker.record_price_snapshot", fake_record_price_snapshot)
    monkeypatch.setattr(app.state, "price_tracker_enabled", True, raising=False)

    payload = {
        "flight": {"airline": "TestAir", "flight_no": "TA100", "price_inr": 5200},
        "origin": "DEL",
        "destination": "BOM",
        "depart_date": future_date,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/booking/track-price", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "track_price"
        assert data["monitoring_active"] is True
        assert data["tracking_state"]["route_tracking_ready"] is True
        assert data["tracking_state"]["checkout_dependency"] == "not_required"
        assert isinstance(data["tracking_state"]["baseline_snapshot_id"], int)
        assert snapshot_calls


@pytest.mark.asyncio
async def test_booking_track_price_rejects_non_numeric_selected_price(monkeypatch):
    async def should_not_hold(**_kwargs):
        raise AssertionError("hold_booking should not be called for unsupported tracking payload")

    monkeypatch.setattr("tools.booking_handoff.hold_booking", should_not_hold)
    monkeypatch.setattr(app.state, "price_tracker_enabled", True, raising=False)

    payload = {
        "flight": {"airline": "TestAir", "flight_no": "TA100", "price_inr": "Price unavailable"},
        "origin": "DEL",
        "destination": "BOM",
        "depart_date": future_date,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/booking/track-price", json=payload)

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "price_tracking_unsupported_selection"
    assert detail["reason"] == "selected_flight_price_unavailable"


@pytest.mark.asyncio
async def test_booking_track_price_snapshot_failure_returns_structured_error(monkeypatch):
    async def fake_hold_booking(**_kwargs):
        return {
            "id": 404,
            "status": "HELD",
            "handoff_url": "https://example.com/booking",
            "expires_at": "2030-01-01T00:00:00Z",
        }

    def fake_record_price_snapshot(**_kwargs):
        raise RuntimeError("db write failed")

    def fake_cancel(_booking_id: int, owner_principal_id=None):
        return True

    def fake_get_booking(booking_id: int, owner_principal_id=None):
        return {
            "id": booking_id,
            "flight": {
                "origin": "DEL",
                "destination": "BOM",
                "date": future_date,
                "price_inr": 5200,
            },
        }

    monkeypatch.setattr("tools.booking_handoff.hold_booking", fake_hold_booking)
    monkeypatch.setattr("tools.booking_handoff.get_booking", fake_get_booking)
    monkeypatch.setattr("tools.booking_handoff.cancel_booking", fake_cancel)
    monkeypatch.setattr("tools.price_tracker.record_price_snapshot", fake_record_price_snapshot)
    monkeypatch.setattr(app.state, "price_tracker_enabled", True, raising=False)

    payload = {
        "flight": {"airline": "TestAir", "flight_no": "TA100", "price_inr": 5200},
        "origin": "DEL",
        "destination": "BOM",
        "depart_date": future_date,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/booking/track-price", json=payload)

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["error"] == "price_tracking_setup_failed"
    assert detail["reason"] == "snapshot_persist_failed"
    assert detail["cleanup_cancelled"] is True


@pytest.mark.asyncio
async def test_booking_track_price_fails_when_held_tracking_prereqs_missing(monkeypatch):
    async def fake_hold_booking(**_kwargs):
        return {
            "id": 505,
            "status": "HELD",
            "handoff_url": None,
            "expires_at": "2030-01-01T00:00:00Z",
        }

    def fake_get_booking(_booking_id: int, owner_principal_id=None):
        return {
            "id": 505,
            "flight": {
                # Missing origin/destination/date to simulate malformed persisted hold.
                "price_inr": 5200,
            },
        }

    def fake_cancel(_booking_id: int, owner_principal_id=None):
        return True

    def should_not_snapshot(**_kwargs):
        raise AssertionError("record_price_snapshot should not run when held tracking prerequisites are invalid")

    monkeypatch.setattr("tools.booking_handoff.hold_booking", fake_hold_booking)
    monkeypatch.setattr("tools.booking_handoff.get_booking", fake_get_booking)
    monkeypatch.setattr("tools.booking_handoff.cancel_booking", fake_cancel)
    monkeypatch.setattr("tools.price_tracker.record_price_snapshot", should_not_snapshot)
    monkeypatch.setattr(app.state, "price_tracker_enabled", True, raising=False)

    payload = {
        "flight": {"airline": "TestAir", "flight_no": "TA100", "price_inr": 5200},
        "origin": "DEL",
        "destination": "BOM",
        "depart_date": future_date,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/booking/track-price", json=payload)

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["error"] == "price_tracking_setup_failed"
    assert detail["reason"] == "held_tracking_prerequisites_missing"
    assert detail["cleanup_cancelled"] is True
    assert "missing_fields" in detail


@pytest.mark.asyncio
async def test_price_tracking_alerts_endpoints(monkeypatch):
    fake_alerts = [
        {
            "alert_id": 7,
            "booking_id": 301,
            "origin": "DEL",
            "destination": "BOM",
            "travel_date": future_date,
            "held_price_inr": 6000,
            "new_price_inr": 5200,
            "drop_pct": 13.3,
            "created_at": "2026-01-01T00:00:00Z",
        }
    ]

    def fake_get_alerts(_booking_id=None, owner_principal_id=None):
        return fake_alerts

    def fake_ack(_alert_id: int, owner_principal_id=None):
        return True

    monkeypatch.setattr("tools.price_tracker.get_unacknowledged_alerts", fake_get_alerts)
    monkeypatch.setattr("tools.price_tracker.acknowledge_alert", fake_ack)
    monkeypatch.setattr(app.state, "price_tracker_enabled", True, raising=False)
    monkeypatch.setattr(
        app.state,
        "price_tracker_status",
        {"last_completed_at": "2026-01-01T00:00:00Z", "last_alert_count": 1},
        raising=False,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        status_resp = await client.get("/price-tracking/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["enabled"] is True

        list_resp = await client.get("/price-tracking/alerts")
        assert list_resp.status_code == 200
        assert list_resp.json()["items"][0]["alert_id"] == 7

        ack_resp = await client.post("/price-tracking/alerts/7/ack")
        assert ack_resp.status_code == 200
        assert ack_resp.json()["acknowledged"] is True


@pytest.mark.asyncio
async def test_debug_keys_sanitizes_sensitive_key_metadata(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "admin-test-token")

    def _fake_status():
        return {
            "serpapi": [
                {
                    "index": 0,
                    "active": False,
                    "in_use": 0,
                    "exhausted_until": "2030-01-01T00:00:00+00:00",
                    "pending_exhaust": False,
                    "pending_clear": False,
                    "searches_left": 0,
                    "last_checked_at": "2026-04-04T00:00:00+00:00",
                    "failure_classification": "monthly_quota",
                    "key_name": "SERPAPI_KEY_1",
                    "name_fingerprint": "name-fp",
                    "fingerprint": "value-fp",
                    "key_value_fingerprint": "value-fp",
                    "last_provider_error": "https://serpapi.com/account.json?api_key=SECRET",
                    "last_provider_reason": "account_reconcile_exception",
                }
            ]
        }

    monkeypatch.setattr(api_app.key_manager, "status", _fake_status)
    monkeypatch.setattr(
        api_app.key_manager,
        "serpapi_reconcile_status",
        lambda: {
            "running": True,
            "last_status": "degraded",
            "last_started_at": "2026-04-04T00:00:00+00:00",
            "last_completed_at": "2026-04-04T00:01:00+00:00",
            "last_error": "errors=1",
            "forced_key_count": 2,
        },
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/debug/keys", headers={"X-Admin-Token": "admin-test-token"})

    assert response.status_code == 200
    payload = response.json()
    row = payload["services"]["serpapi"][0]
    assert "key_name" not in row
    assert "name_fingerprint" not in row
    assert "fingerprint" not in row
    assert "key_value_fingerprint" not in row
    assert "last_provider_error" not in row
    assert "last_provider_reason" not in row
    reconcile = payload["serpapi_reconciliation"]
    assert "last_error" not in reconcile
    assert "forced_key_count" not in reconcile
    assert set(reconcile.keys()) == {"running", "last_status", "last_started_at", "last_completed_at"}


@pytest.mark.asyncio
async def test_debug_keys_missing_admin_token_returns_403(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLE", "false")
    monkeypatch.setenv("AUTH_DISABLE_ADMIN", "false")
    monkeypatch.setenv("ADMIN_TOKEN", "admin-test-token")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/debug/keys")
    assert response.status_code == 403


def test_build_allowed_origins_rejects_wildcard_and_invalid_entries(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "*,https://app.example.com,https://app.example.com/path,http://localhost:5173,not-a-url,https://APP.example.com",
    )
    origins = api_app._build_allowed_origins()
    assert origins == ["https://app.example.com", "http://localhost:5173"]


@pytest.mark.asyncio
async def test_health_keys_exposes_only_high_level_state(monkeypatch):
    async def _fake_get_status():
        return {
            "serpapi": [
                {
                    "index": 0,
                    "active": True,
                    "in_use": 1,
                    "exhausted_until": None,
                    "searches_left": 42,
                    "last_checked_at": "2026-04-04T00:00:00+00:00",
                    "failure_classification": "ok",
                    "key_name": "SERPAPI_KEY_1",
                    "fingerprint": "value-fp",
                    "name_fingerprint": "name-fp",
                    "last_provider_error": "transient provider text",
                }
            ],
            "openai": [
                {
                    "index": 0,
                    "active": False,
                    "in_use": 0,
                    "exhausted_until": "2030-01-01T00:00:00+00:00",
                    "key_name": "OPENAI_KEY_1",
                    "fingerprint": "openai-fp",
                }
            ],
        }

    monkeypatch.setattr(api_app.key_manager, "get_status", _fake_get_status)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health/keys")

    assert response.status_code == 200
    payload = response.json()
    serpapi_row = payload["serpapi"][0]
    openai_row = payload["openai"][0]
    assert set(serpapi_row.keys()) == {"index", "active", "state"}
    assert set(openai_row.keys()) == {"index", "active", "state"}
    assert serpapi_row["state"] == "active"
    assert openai_row["state"] == "exhausted"


@pytest.mark.asyncio
async def test_provider_state_override_admin_endpoints(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "admin-test-token")

    captured = {"set_calls": [], "disable_calls": []}

    async def _fake_set_provider_state_override(**kwargs):
        captured["set_calls"].append(kwargs)
        return {
            "id": 101,
            "provider": kwargs["provider"],
            "scope_type": kwargs["scope_type"],
            "scope_identifier": kwargs.get("scope_identifier"),
            "override_type": kwargs["override_type"],
            "override_until": kwargs.get("active_until"),
            "active_until": kwargs.get("active_until"),
            "override_until_semantics": "skips_reconcile_until",
            "note": kwargs.get("note"),
            "is_enabled": True,
        }

    async def _fake_list_provider_state_overrides(**kwargs):
        return [
            {
                "id": 101,
                "provider": "serpapi",
                "scope_type": "key",
                "scope_identifier": "abc123",
                "override_type": "skip_reconcile_until",
                "override_until": "2026-05-01T00:00:00+00:00",
                "active_until": "2026-05-01T00:00:00+00:00",
                "override_until_semantics": "skips_reconcile_until",
                "note": "maintenance window",
                "is_enabled": True,
                "is_currently_active": True,
            }
        ]

    async def _fake_disable_provider_state_override(override_id: int):
        captured["disable_calls"].append(override_id)
        return True

    monkeypatch.setattr(api_app.key_manager, "set_provider_state_override", _fake_set_provider_state_override)
    monkeypatch.setattr(api_app.key_manager, "list_provider_state_overrides", _fake_list_provider_state_overrides)
    monkeypatch.setattr(api_app.key_manager, "disable_provider_state_override", _fake_disable_provider_state_override)
    monkeypatch.setattr(api_app.key_manager, "key_scope_identifier", lambda provider, index: asyncio.sleep(0, result="abc123"))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_resp = await client.post(
            "/debug/provider-state/overrides",
            headers={"X-Admin-Token": "admin-test-token"},
            json={
                "provider": "serpapi",
                "scope_type": "key",
                "key_index": 0,
                "override_type": "skip_reconcile_until",
                "override_until": "2026-05-01T00:00:00+00:00",
                "note": "maintenance window",
            },
        )
        assert create_resp.status_code == 200
        list_resp = await client.get(
            "/debug/provider-state/overrides?provider=serpapi",
            headers={"X-Admin-Token": "admin-test-token"},
        )
        assert list_resp.status_code == 200
        disable_resp = await client.post(
            "/debug/provider-state/overrides/101/disable",
            headers={"X-Admin-Token": "admin-test-token"},
        )
        assert disable_resp.status_code == 200

    assert captured["set_calls"]
    assert captured["set_calls"][0]["provider"] == "serpapi"
    assert captured["set_calls"][0]["active_until"] == "2026-05-01T00:00:00+00:00"
    assert captured["disable_calls"] == [101]
    listed = list_resp.json()["overrides"]
    assert isinstance(listed, list)
    assert listed[0]["override_type"] == "skip_reconcile_until"


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
async def test_ask_recent_completion_replay_returns_cached_payload(monkeypatch):
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
        "date": future_date,
        "user_query": "repeat this quickly",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first = await client.post("/ask", json=payload)
        second = await client.post("/ask", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert second.headers.get("X-Ask-Admission") == "replayed_recent"
    assert second.headers.get("X-Ask-Contract") == "single-node-process-local"
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_ask_duplicate_burst_has_single_leader_and_deterministic_rejections(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"count": 0}

    async def fake_plan_trip(**_kwargs):
        calls["count"] += 1
        started.set()
        await release.wait()
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setenv("ASK_MAX_INFLIGHT", "16")
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}}

    payload = {
        "origin": "DEL",
        "destination": "BOM",
        "date": future_date,
        "user_query": "Business trip burst",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        tasks = [asyncio.create_task(client.post("/ask", json=payload)) for _ in range(6)]
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        release.set()
        responses = await asyncio.gather(*tasks)

    success = [resp for resp in responses if resp.status_code == 200]
    duplicates = [resp for resp in responses if resp.status_code == 409]

    assert len(success) == 1
    assert len(duplicates) == 5
    assert calls["count"] == 1
    for resp in duplicates:
        assert resp.json()["error"] == "duplicate_request_in_progress"
        assert resp.headers.get("X-Ask-Admission") == "duplicate_in_progress"


@pytest.mark.asyncio
async def test_ask_short_distinct_burst_is_bounded_by_inflight_limit(monkeypatch):
    started_two = asyncio.Event()
    release = asyncio.Event()
    calls = {"count": 0}

    async def fake_plan_trip(**_kwargs):
        calls["count"] += 1
        if calls["count"] >= 2:
            started_two.set()
        await release.wait()
        return {"result": "ok"}

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setenv("ASK_MAX_INFLIGHT", "2")
    app.state.ask_runtime_state = {"lock": asyncio.Lock(), "inflight": {}}

    destinations = ["BOM", "BLR", "MAA", "CCU", "GOI"]
    payloads = [
        {
            "origin": "DEL",
            "destination": destination,
            "date": future_date,
            "user_query": f"burst-{idx}",
        }
        for idx, destination in enumerate(destinations)
    ]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        tasks = [asyncio.create_task(client.post("/ask", json=payload)) for payload in payloads]
        await asyncio.wait_for(started_two.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        release.set()
        responses = await asyncio.gather(*tasks)

    accepted = [resp for resp in responses if resp.status_code == 200]
    overloaded = [resp for resp in responses if resp.status_code == 429]

    assert len(accepted) == 2
    assert len(overloaded) == 3
    assert calls["count"] == 2
    for resp in overloaded:
        assert resp.json()["error"] == "ask_overloaded"
        assert resp.headers.get("X-Ask-Admission") == "overloaded"


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


@pytest.mark.asyncio
async def test_booking_post_handoff_bridge_rejects_unknown_artifact():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/booking/handoff/post/does-not-exist")

    # GET endpoint returns 200 with HTML form regardless of artifact existence
    # Artifact validation happens on POST
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_booking_post_handoff_bridge_unknown_artifact_html_client_gets_clear_page():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/booking/handoff/post/does-not-exist",
            headers={"Accept": "text/html"},
        )

    # GET endpoint returns 200 with HTML form regardless of artifact existence
    assert response.status_code == 200
    assert "Continue to booking" in response.text


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
    body = response.text
    assert "form id='handoff-consume'" in body
    assert "Continue to booking" in body


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
        first = await client.post(f"/booking/handoff/post/{artifact_id}")
        second = await client.post(f"/booking/handoff/post/{artifact_id}")

    assert first.status_code == 200
    assert "form id='handoff'" in first.text
    assert second.status_code in {404, 410}


@pytest.mark.asyncio
async def test_booking_post_handoff_bridge_consumed_artifact_html_client_gets_gone_page():
    bridge_url = booking_handoff.register_post_handoff_artifact(
        url="https://provider.example/checkout",
        post_data={"token": "abc123"},
    )
    assert bridge_url is not None
    artifact_id = bridge_url.rsplit("/", 1)[-1]

    booking_handoff._post_handoff_artifacts.clear()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first = await client.post(f"/booking/handoff/post/{artifact_id}")
        second = await client.post(
            f"/booking/handoff/post/{artifact_id}",
            headers={"Accept": "text/html"},
        )

    assert first.status_code == 200
    assert second.status_code in {404, 410}
    assert "Booking Link Unavailable" in second.text or second.status_code in {404, 410}


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
            client.post(f"/booking/handoff/post/{artifact_id}"),
            client.post(f"/booking/handoff/post/{artifact_id}"),
        )

    # One should succeed (200), the other should fail (404 or 410)
    assert first.status_code == 200 or second.status_code == 200
    assert first.status_code in {200, 404, 410}
    assert second.status_code in {200, 404, 410}


@pytest.mark.asyncio
async def test_booking_post_handoff_bridge_repeated_concurrent_consumes_remain_single_winner():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for idx in range(5):
            bridge_url = booking_handoff.register_post_handoff_artifact(
                url=f"https://provider.example/checkout/{idx}",
                post_data={"token": f"tok-{idx}"},
            )
            assert bridge_url is not None
            artifact_id = bridge_url.rsplit("/", 1)[-1]
            booking_handoff._post_handoff_artifacts.clear()

            first, second = await asyncio.gather(
                client.post(f"/booking/handoff/post/{artifact_id}"),
                client.post(f"/booking/handoff/post/{artifact_id}"),
            )
            # One should succeed (200), the other should fail (404 or 410)
            assert first.status_code == 200 or second.status_code == 200
            assert first.status_code in {200, 404, 410}
            assert second.status_code in {200, 404, 410}


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
    assert body["status"] in {"ok", "degraded", "fail"}
    assert "dependencies" in body


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
    assert body["dependencies"]["cloud"] == "unavailable"
    assert body["dependencies"]["ollama"] == "not_relevant"
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
    assert body["dependencies"]["ollama"] == "unavailable"
    assert body["dependencies"]["cloud"] == "ok"
    assert body["status"] == "degraded"


@pytest.mark.asyncio
async def test_readiness_stays_ok_while_best_effort_llm_prewarm_is_running():
    app.state.startup_complete = True
    app.state.llm_prewarm = {
        "enabled": True,
        "best_effort": True,
        "status": "running",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
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
    assert response.json()["detail"] == "Internal server error."


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

    assert response.status_code == 200
    assert response.json().get("detail") == "No live flights found."
    assert response.json().get("failure_reason") == "no_flights"
    assert response.json().get("failure_domain") == "search_outcome"
    assert response.json().get("no_flights_reason") == "no_inventory"
    assert response.json().get("result_status") == "success"
    assert response.json().get("flight_counts", {}).get("pre_filter") == 0


@pytest.mark.asyncio
async def test_ask_non_stream_warning_fallback_preserves_handoff_contract_fields(monkeypatch):
    async def fake_plan_trip(**kwargs):
        return {
            "warning": "No live flights found.",
            "fallback": True,
            "failure_reason": "no_flights",
            "no_flights_reason": "no_inventory",
            "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
            "booking_handoff": {
                "status": "deferred",
                "source": "deferred",
                "reason": "deferred_until_booking_intent",
                "url": None,
                "booking_exit_quality": "deferred",
            },
            "top_flights": [],
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

    body = response.json()
    assert response.status_code == 200
    assert body.get("booking_handoff", {}).get("booking_exit_quality") == "deferred"
    assert isinstance(body.get("top_flights"), list)
    assert body.get("debug_info", {}).get("top_flights") == []
    assert "booking_handoff_quality_context" not in (body.get("debug_info") or {})


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

    assert response.status_code == 500
    assert response.json().get("detail") == "Planner failed to produce a complete response."
    assert response.json().get("failure_reason") == "planner_error"
    assert response.json().get("failure_domain") == "internal_backend"
    assert response.json().get("result_status") == "error"


@pytest.mark.asyncio
async def test_ask_non_stream_error_preserves_handoff_contract_fields(monkeypatch):
    async def fake_plan_trip(**kwargs):
        return {
            "error": "Could not determine origin or destination airport after AI correction.",
            "failure_reason": "invalid_route",
            "flight_counts": {"pre_filter": 0, "post_filter": 0, "filtered_out": 0},
            "booking_handoff": {
                "status": "unavailable",
                "source": "unavailable",
                "reason": "invalid_route",
                "url": None,
                "booking_exit_quality": "unavailable",
            },
            "top_flights": [],
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

    body = response.json()
    assert response.status_code == 500
    assert body.get("failure_reason") == "invalid_route"
    assert body.get("booking_handoff", {}).get("status") == "unavailable"
    assert isinstance(body.get("top_flights"), list)
    assert body.get("debug_info", {}).get("top_flights") == []
    assert "booking_handoff_quality_context" not in (body.get("debug_info") or {})
