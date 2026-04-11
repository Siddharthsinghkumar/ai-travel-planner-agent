import asyncio
import json
import pytest
import httpx
from api.app import app
from core import job_queue

def _reset_job_queue_state():
    job_queue._jobs.clear()
    job_queue._job_event_queues.clear()
    job_queue._job_tasks.clear()
    job_queue._queue = asyncio.Queue()

@pytest.mark.asyncio
async def test_enqueue_and_poll(monkeypatch):
    _reset_job_queue_state()
    monkeypatch.setenv("UVICORN_WORKERS", "1")
    monkeypatch.setenv("ASYNC_JOB_REQUIRE_SINGLE_WORKER", "1")
    monkeypatch.delenv("ALLOW_UNSAFE_ASYNC_JOBS", raising=False)
    monkeypatch.setattr(app.state, "async_job_support", None, raising=False)

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
                assert "contract" in j
                if j["status"] == "done":
                    assert j["result"]["ok"] is True
                    return
                await asyncio.sleep(0.01)
    finally:
        await job_queue.stop_worker()
        await worker_task

    pytest.fail("job did not finish in time")


@pytest.mark.asyncio
async def test_cancel_job_endpoint(monkeypatch):
    _reset_job_queue_state()
    monkeypatch.setenv("UVICORN_WORKERS", "1")
    monkeypatch.setenv("ASYNC_JOB_REQUIRE_SINGLE_WORKER", "1")
    monkeypatch.delenv("ALLOW_UNSAFE_ASYNC_JOBS", raising=False)
    monkeypatch.setattr(app.state, "async_job_support", None, raising=False)
    async def fake_plan_trip(*args, **kwargs):
        await asyncio.sleep(0.2)
        return {"ok": True}

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

            cancel_resp = await client.post(f"/jobs/{job_id}/cancel", json={})
            assert cancel_resp.status_code == 200
            cancel_payload = cancel_resp.json()
            assert cancel_payload.get("status") in {"cancel_requested", "cancelled", "already_terminal"}
            if isinstance(cancel_payload.get("job"), dict):
                assert "event_seq" not in cancel_payload["job"]

            final_status = None
            for _ in range(60):
                status_resp = await client.get(f"/jobs/{job_id}")
                assert status_resp.status_code == 200
                job = status_resp.json()
                final_status = job.get("status")
                if final_status in {"cancelled", "done", "error"}:
                    break
                await asyncio.sleep(0.01)

            assert final_status in {"cancelled", "done", "error"}
    finally:
        await job_queue.stop_worker()
        await worker_task


def test_prune_jobs_removes_terminal(monkeypatch):
    _reset_job_queue_state()
    monkeypatch.setattr(job_queue, "JOB_RETENTION_SECONDS", 0)
    monkeypatch.setattr(job_queue, "JOB_PRUNE_INTERVAL_SECONDS", 0)

    job_id = "job-terminal"
    now_iso = job_queue._utc_now_iso()
    job_queue._jobs[job_id] = {
        "job_id": job_id,
        "status": "done",
        "result": {"ok": True},
        "error": None,
        "message": "done",
        "created_at": now_iso,
        "updated_at": now_iso,
        "completed_at": now_iso,
        "cancel_requested": False,
    }

    removed = job_queue.prune_jobs()
    assert removed == 1
    assert job_id not in job_queue._jobs


@pytest.mark.asyncio
async def test_job_events_stream_contract_is_structured_and_not_nested_sse(monkeypatch):
    _reset_job_queue_state()
    monkeypatch.setenv("UVICORN_WORKERS", "1")
    monkeypatch.setenv("ASYNC_JOB_REQUIRE_SINGLE_WORKER", "1")
    monkeypatch.delenv("ALLOW_UNSAFE_ASYNC_JOBS", raising=False)
    monkeypatch.setattr(app.state, "async_job_support", None, raising=False)

    async def fake_plan_trip(*_args, **kwargs):
        if kwargs.get("stream"):
            async def _agen():
                yield "event: reasoning_step\ndata: {\"step\":\"Collecting fares\"}\n\n"
                yield "token chunk "
                yield "[DONE_JSON]" + json.dumps({"ok": True, "source": "done_json"})
            return _agen()
        return {"ok": True, "source": "non_stream"}

    monkeypatch.setattr("agents.planner_agent.plan_trip", fake_plan_trip)

    worker_task = asyncio.create_task(job_queue.worker_loop())
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=5.0) as client:
            r = await client.post("/ask?async_job=true", json={
                "date": "2026-03-15",
                "user_query": "test",
                "trip_type": "Business",
            })
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            async with client.stream("GET", f"/jobs/{job_id}/events") as stream_resp:
                assert stream_resp.status_code == 200
                chunks = []
                async for chunk in stream_resp.aiter_text():
                    chunks.append(chunk)
                raw = "".join(chunks)

            assert "event: reasoning_step" in raw
            assert "\"event\": \"reasoning_step\"" in raw
            assert "\"data\": {\"step\": \"Collecting fares\"}" in raw
            assert "\"event\": \"token\"" in raw
            assert "\"data\": {\"chunk\": \"token chunk \"}" in raw
            assert "\"event\": \"done\"" in raw
            assert "\"result\": {\"ok\": true, \"source\": \"done_json\"}" in raw
            assert "\"message\": \"event: reasoning_step" not in raw

            job_resp = await client.get(f"/jobs/{job_id}")
            assert job_resp.status_code == 200
            payload = job_resp.json()
            assert payload["status"] == "done"
            assert payload["result"]["source"] == "done_json"
            assert "event_seq" not in payload
    finally:
        await job_queue.stop_worker()
        await worker_task
