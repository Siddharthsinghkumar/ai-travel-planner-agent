import json

import httpx
import pytest

from api.app import app
from core.llm_mode import get_default_cloud_provider, get_llm_mode_and_priority, get_effective_cloud_provider


@pytest.mark.asyncio
async def test_ask_non_stream_applies_request_mode_and_provider(monkeypatch):
    provider = get_default_cloud_provider()

    async def fake_plan_trip(*args, **kwargs):
        mode, _ = await get_llm_mode_and_priority()
        return {
            "mode_seen": mode,
            "provider_seen": get_effective_cloud_provider(),
        }

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/ask",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-04-20",
                "user_query": "business trip",
                "llm_mode": "cloud_only",
                "cloud_provider": provider,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode_seen"] == "cloud_only"
    assert body["provider_seen"] == provider


@pytest.mark.asyncio
async def test_ask_stream_applies_request_mode_and_provider(monkeypatch):
    provider = get_default_cloud_provider()

    async def fake_plan_trip(*args, **kwargs):
        mode, _ = await get_llm_mode_and_priority()

        async def _gen():
            yield f"mode={mode};provider={get_effective_cloud_provider()}"
            yield "[DONE_JSON]" + json.dumps({"ok": True})

        return _gen()

    monkeypatch.setattr("api.app.planner_agent.plan_trip", fake_plan_trip)
    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream(
            "POST",
            "/ask?stream=true",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-04-20",
                "user_query": "business trip",
                "llm_mode": "ollama_first",
                "cloud_provider": provider,
            },
        ) as resp:
            assert resp.status_code == 200
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk
                if "event: done" in body or "[DONE_JSON]" in body:
                    break

    assert "mode=ollama_first" in body
    assert f"provider={provider}" in body


@pytest.mark.asyncio
async def test_ask_rejects_invalid_llm_mode():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/ask",
            json={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-04-20",
                "user_query": "business trip",
                "llm_mode": "not_a_mode",
            },
        )

    assert resp.status_code == 422
