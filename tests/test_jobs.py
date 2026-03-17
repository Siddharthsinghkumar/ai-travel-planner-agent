import asyncio
import pytest
from fastapi.testclient import TestClient
from api.app import app

@pytest.mark.asyncio
async def test_enqueue_and_poll(monkeypatch):
    async def fake_plan_trip(*args, **kwargs):
        await asyncio.sleep(0.01)
        return {"ok": True, "result": "done"}

    # patch BEFORE app startup so background worker uses fake
    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

    # start app AFTER monkeypatch to ensure lifespan sees patched function
    with TestClient(app) as client:
        r = client.post("/ask?async_job=true", json={
            "date": "2026-03-15",
            "user_query": "test",
            "trip_type": "Business"
        })
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        for _ in range(50):
            r2 = client.get(f"/jobs/{job_id}")
            assert r2.status_code == 200
            j = r2.json()
            if j["status"] == "done":
                assert j["result"]["ok"] is True
                return
            await asyncio.sleep(0.01)

    pytest.fail("job did not finish in time")