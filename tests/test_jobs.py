import asyncio
import pytest
import httpx
from api.app import app
from core import job_queue

@pytest.mark.asyncio
async def test_enqueue_and_poll(monkeypatch):
    async def fake_plan_trip(*args, **kwargs):
        await asyncio.sleep(0.01)
        return {"ok": True, "result": "done"}

    # patch BEFORE app startup so background worker uses fake
    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

    worker_task = asyncio.create_task(job_queue.worker_loop())
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post("/ask?async_job=true", json={
                "date": "2026-03-15",
                "user_query": "test",
                "trip_type": "Business"
            })
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            for _ in range(50):
                r2 = await client.get(f"/jobs/{job_id}")
                assert r2.status_code == 200
                j = r2.json()
                if j["status"] == "done":
                    assert j["result"]["ok"] is True
                    return
                await asyncio.sleep(0.01)
    finally:
        await job_queue.stop_worker()
        await worker_task

    pytest.fail("job did not finish in time")
