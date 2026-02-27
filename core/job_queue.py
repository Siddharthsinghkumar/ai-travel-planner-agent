# core/job_queue.py
import asyncio
import json
import uuid
import traceback
from typing import Any, Dict

# Prometheus metrics
from core import metrics

_jobs: Dict[str, Dict[str, Any]] = {}
_job_event_queues: Dict[str, asyncio.Queue] = {}
_queue: asyncio.Queue = asyncio.Queue()
_worker_task: asyncio.Task | None = None

async def enqueue_job(payload: dict) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}
    _job_event_queues[job_id] = asyncio.Queue()
    # initial event
    await _job_event_queues[job_id].put({"type": "queued", "message": "job queued"})
    await _queue.put((job_id, payload))
    # update gauge for job queue size
    try:
        metrics.JOB_QUEUE_SIZE.set(_queue.qsize())
    except Exception:
        pass
    return job_id

async def get_job(job_id: str):
    return _jobs.get(job_id)

async def get_job_event_queue(job_id: str) -> asyncio.Queue | None:
    return _job_event_queues.get(job_id)

async def stop_worker():
    """Signal the worker loop to shut down gracefully."""
    await _queue.put(None)

async def _process_job(job_id: str, payload: dict):
    # Use `planner_agent` to process the payload. If streaming available, forward tokens to event queue.
    from agents import planner_agent
    q = _job_event_queues.get(job_id)
    try:
        _jobs[job_id]["status"] = "running"
        if q:
            await q.put({"type": "running", "message": "job started"})

        # Prefer streaming plan if available to forward progress events:
        agen_or_result = await planner_agent.plan_trip(
            origin=payload.get("origin"),
            destination=payload.get("destination"),
            date=payload.get("date"),
            user_query=payload.get("user_query"),
            trip_type=payload.get("trip_type"),
            stream=True,
        )

        # If we received an async generator, forward tokens; else, we got final result
        if hasattr(agen_or_result, "__aiter__"):
            final_found = False
            final_payload = None
            async for token in agen_or_result:
                if q:
                    await q.put({"type": "token", "message": token})

                # tokens from planner may contain the final JSON prefixed with [DONE_JSON]
                if isinstance(token, str) and token.startswith("[DONE_JSON]"):
                    json_part = token[len("[DONE_JSON]"):]
                    try:
                        parsed = json.loads(json_part)
                        final_payload = parsed
                        final_found = True
                        # push final 'done' event immediately
                        if q:
                            await q.put({"type": "done", "message": parsed})
                    except Exception as e:
                        if q:
                            await q.put({"type": "error", "message": f"failed to parse DONE_JSON: {e}"})
                    # usually generator ends after DONE_JSON; but continue just in case
            # If we found final payload from stream, store it and skip non-stream call
            if final_found and final_payload is not None:
                _jobs[job_id]["result"] = final_payload
                _jobs[job_id]["status"] = "done"
                return
        else:
            # non-stream return value (final) — push as final event
            if q:
                await q.put({"type": "result", "message": agen_or_result})

        # Finalize: if we didn't receive final dict yet, call non-streaming plan for final structured result
        if _jobs[job_id]["result"] is None:
            try:
                # ensure final structured result is available (non-stream)
                final = await planner_agent.plan_trip(
                    origin=payload.get("origin"),
                    destination=payload.get("destination"),
                    date=payload.get("date"),
                    user_query=payload.get("user_query"),
                    trip_type=payload.get("trip_type"),
                    stream=False,
                )
                _jobs[job_id]["result"] = final
                _jobs[job_id]["status"] = "done"
                if q:
                    await q.put({"type": "done", "message": final})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(e)
                if q:
                    await q.put({"type": "error", "message": str(e)})
        return

    except asyncio.CancelledError:
        raise
    except Exception:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = traceback.format_exc()
        if q:
            await q.put({"type": "error", "message": _jobs[job_id]["error"]})
    finally:
        # close event queue by putting a sentinel
        if q:
            await q.put({"type": "closed", "message": ""})

async def worker_loop():
    """Worker loop that runs forever in the app lifetime."""
    try:
        while True:
            item = await _queue.get()

            # Update gauge after popping the item
            try:
                metrics.JOB_QUEUE_SIZE.set(_queue.qsize())
            except Exception:
                pass

            # 🟢 Sentinel handling
            if item is None:
                break

            job_id, payload = item
            try:
                await _process_job(job_id, payload)
            except Exception:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = "worker exception"
            finally:
                # ensure we mark queue task done and update gauge
                try:
                    _queue.task_done()
                except Exception:
                    pass
                try:
                    metrics.JOB_QUEUE_SIZE.set(_queue.qsize())
                except Exception:
                    pass
    except asyncio.CancelledError:
        # 🔥 CRITICAL: swallow cancellation cleanly
        # Prevent cancellation from bubbling into TestClient shutdown
        pass