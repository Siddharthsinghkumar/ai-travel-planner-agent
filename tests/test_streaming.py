# tests/test_streaming.py
import asyncio
import json
import httpx
import pytest

from api.app import app

@pytest.mark.asyncio
async def test_streaming_endpoint_returns_sse(monkeypatch):
    fake_result = {
        "best_flight": {"airline": "TestAir", "flight_no": "TA123"},
        "llm_response": "This is a streamed response.",
    }

    async def fake_plan_trip(*args, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                yield "Searching flights...\n"
                yield "Found options...\n"
                yield "This is a streamed response."
                yield "[DONE_JSON]" + json.dumps(fake_result)
            return _gen()
        return fake_result

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

    payload = {"date": "2026-03-15", "user_query": "test from delhi to mumbai", "trip_type": "Business"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream("POST", "/ask?stream=true", json=payload) as resp:
            assert resp.status_code == 200
            content = ""
            # read a few chunks (non-blocking) to ensure we got streaming data
            async for chunk in resp.aiter_text():
                if not chunk:
                    break
                content += chunk
                # stop early once we saw the done event or the done JSON prefix
                if "[DONE_JSON]" in content or "event: done" in content:
                    break

            text = content
            # check for at least one token and SSE framing
            assert "data:" in text or "Searching flights" in text
            # check we received model tokens
            assert "streamed response" in text.lower()
            # ensure final marker present
            assert "[DONE_JSON]" in text or "event: done" in text


@pytest.mark.asyncio
async def test_streaming_endpoint_preserves_multiline_chunks_as_valid_sse(monkeypatch):
    fake_result = {
        "best_flight": {"airline": "TestAir", "flight_no": "TA124"},
        "llm_response": "line1\nline2",
    }

    async def fake_plan_trip(*args, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                yield "line1\nline2"
                yield "[DONE_JSON]" + json.dumps(fake_result)
            return _gen()
        return fake_result

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

    payload = {"date": "2026-03-15", "user_query": "test", "trip_type": "Business"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream("POST", "/ask?stream=true", json=payload) as resp:
            assert resp.status_code == 200
            content = ""
            async for chunk in resp.aiter_text():
                content += chunk
                if "[DONE_JSON]" in content:
                    break

    assert "data: line1" in content
    assert "data: line2" in content
    assert "[DONE_JSON]" in content


@pytest.mark.asyncio
async def test_streaming_endpoint_emits_structured_events_before_done_json(monkeypatch):
    fake_result = {
        "best_flight": {"airline": "TestAir", "flight_no": "TA125"},
        "llm_response": "final explanation",
    }

    async def fake_plan_trip(*args, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                yield 'event: reasoning_step\ndata: {"step":"Gathering live options."}\n\n'
                yield 'event: flights\ndata: {"all_flights":[{"airline":"TestAir","flight_no":"TA125"}],"best_flight":{"airline":"TestAir","flight_no":"TA125"}}\n\n'
                yield 'event: weather\ndata: {"weather":{"condition":"Cloudy","temperature_c":26.2}}\n\n'
                yield "Token chunk"
                yield "[DONE_JSON]" + json.dumps(fake_result)
            return _gen()
        return fake_result

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

    payload = {"date": "2026-03-15", "user_query": "test", "trip_type": "Business"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream("POST", "/ask?stream=true", json=payload) as resp:
            assert resp.status_code == 200
            content = ""
            async for chunk in resp.aiter_text():
                content += chunk
                if "[DONE_JSON]" in content:
                    break

    reasoning_idx = content.find("event: reasoning_step")
    flights_idx = content.find("event: flights")
    weather_idx = content.find("event: weather")
    done_idx = content.find("[DONE_JSON]")
    assert reasoning_idx != -1
    assert flights_idx != -1
    assert weather_idx != -1
    assert done_idx != -1
    assert reasoning_idx < done_idx
    assert flights_idx < done_idx
    assert weather_idx < done_idx


@pytest.mark.asyncio
async def test_streaming_endpoint_emits_early_structured_activity_before_delayed_tokens(monkeypatch):
    fake_result = {
        "best_flight": {"airline": "TestAir", "flight_no": "TA126"},
        "llm_response": "final explanation",
    }

    async def fake_plan_trip(*args, **kwargs):
        if kwargs.get("stream"):
            async def _gen():
                yield 'event: reasoning_step\ndata: {"step":"Gathering live options."}\n\n'
                await asyncio.sleep(0.2)
                yield "Token chunk"
                yield "[DONE_JSON]" + json.dumps(fake_result)
            return _gen()
        return fake_result

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

    payload = {"date": "2026-03-15", "user_query": "test", "trip_type": "Business"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream("POST", "/ask?stream=true", json=payload) as resp:
            assert resp.status_code == 200
            chunks: list[str] = []
            async for chunk in resp.aiter_text():
                if not chunk:
                    continue
                chunks.append(chunk)
                if "[DONE_JSON]" in "".join(chunks):
                    break

    combined = "".join(chunks)
    assert "event: reasoning_step" in combined
    assert "Token chunk" in combined
    assert combined.find("event: reasoning_step") < combined.find("Token chunk")
