#tests/test_full_pipeline.py
import httpx
import json
import pytest
from api.app import app

payload = {
    "date": "2026-03-15",
    "user_query": "delhi to mumbai",
    "trip_type": "Business"
}


@pytest.fixture(autouse=True)
def patch_external_tools(monkeypatch):
    fake_result = {
        "best_flight": {
            "airline": "TestAir",
            "flight_no": "TA123",
            "booking_token": "fake_token_123",
            "shareable_link": "https://example.com/flights",
        },
        "weather": {
            "location": "BOM",
            "condition": "Clear",
            "temperature_c": 28,
        },
        "llm_response": "Recommended option: TestAir TA123.",
        "debug_info": {
            "price_insights_str": "Stable fare range.",
            "all_flights": [],
        },
    }

    async def fake_plan_trip(*args, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                yield "Planning your trip...\n"
                yield "[DONE_JSON]" + json.dumps(fake_result)
            return _gen()
        return fake_result

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

@pytest.mark.asyncio
async def test_full_blocking_flow():
    """Phase 1 (non-stream) + Phase 3 timeout + Phase 4 metrics"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "best_flight" in data
    assert "weather" in data
    assert "llm_response" in data
    # NEW ASSERTIONS
    assert "booking_token" in data["best_flight"] or "shareable_link" in data["best_flight"]
    assert "debug_info" in data
    assert "price_insights_str" in data["debug_info"]


@pytest.mark.asyncio
async def test_streaming_flow():
    """Phase 1 streaming end-to-end"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream("POST", "/ask?stream=true", json=payload) as resp:
            assert resp.status_code == 200
            content = ""
            async for chunk in resp.aiter_text():
                if not chunk:
                    break
                content += chunk
                if "[DONE_JSON]" in content or "event: done" in content:
                    break
            assert "[DONE_JSON]" in content or "event: done" in content


@pytest.mark.asyncio
async def test_async_job_flow():
    """Phase 2 job queue lifecycle"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/ask?async_job=true", json=payload)
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        status_resp = await client.get(f"/jobs/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] in ["queued", "running", "done"]


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Phase 4 metrics exposure"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Prime HTTP metrics with non-/metrics traffic.
        warm_resp = await client.post("/ask", json=payload)
        assert warm_resp.status_code == 200
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "llm_requests_total" in text
    assert "stream_requests_total" in text
    assert "job_queue_size" in text
    assert "http_requests_total" in text
    assert "http_request_duration_seconds" in text
    assert "http_inflight_requests" in text
