# core/job_queue.py
import asyncio
import json
import os
import time
import uuid
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

# Prometheus metrics
from core import metrics

_jobs: Dict[str, Dict[str, Any]] = {}
_job_event_queues: Dict[str, asyncio.Queue] = {}
_queue: asyncio.Queue = asyncio.Queue()
_worker_task: asyncio.Task | None = None
_job_tasks: Dict[str, asyncio.Task] = {}

JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "3600"))
JOB_PRUNE_INTERVAL_SECONDS = int(os.getenv("JOB_PRUNE_INTERVAL_SECONDS", "300"))
_last_prune_at: float = 0.0

TERMINAL_STATUSES = {"done", "error", "cancelled"}


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _next_event_seq(job_id: str) -> int:
    job = _jobs.get(job_id)
    if not isinstance(job, dict):
        return 0
    seq = int(job.get("event_seq", 0) or 0) + 1
    job["event_seq"] = seq
    return seq


def _build_job_event(
    job_id: str,
    *,
    event: str,
    status: str,
    message: Optional[str] = None,
    data: Optional[Any] = None,
    result: Optional[Any] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event": event,
        "job_id": job_id,
        "status": status,
        "timestamp": _utc_now_iso(),
        "sequence": _next_event_seq(job_id),
    }
    if message is not None:
        payload["message"] = str(message)
    if data is not None:
        payload["data"] = data
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    return payload


def _parse_sse_frame(frame_text: str) -> Optional[Dict[str, Any]]:
    text = str(frame_text or "")
    if not (text.startswith("event:") and text.endswith("\n\n")):
        return None

    event_name: Optional[str] = None
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip() or None
            continue
        if line.startswith("data:"):
            data_lines.append(line[6:] if line.startswith("data: ") else line[5:])

    if not event_name:
        return None
    data_text = "\n".join(data_lines)
    data_payload: Any = data_text
    if data_text.strip():
        try:
            data_payload = json.loads(data_text)
        except Exception:
            data_payload = data_text
    return {"event": event_name, "data": data_payload, "data_text": data_text}


def _should_prune(now: float) -> bool:
    if JOB_RETENTION_SECONDS <= 0:
        return True
    if JOB_PRUNE_INTERVAL_SECONDS <= 0:
        return True
    return (now - _last_prune_at) >= JOB_PRUNE_INTERVAL_SECONDS


def _set_job_status(
    job_id: str,
    status: str,
    *,
    error: Optional[str] = None,
    result: Optional[Any] = None,
    message: Optional[str] = None,
) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    job["status"] = status
    job["updated_at"] = _utc_now_iso()
    if status in TERMINAL_STATUSES:
        job["completed_at"] = job.get("completed_at") or _utc_now_iso()
    if error is not None:
        job["error"] = error
    if result is not None:
        job["result"] = result
    if message is not None:
        job["message"] = message


async def _emit_job_event(job_id: str, payload: Dict[str, Any]) -> None:
    queue = _job_event_queues.get(job_id)
    if queue is None:
        return
    await queue.put(payload)


def prune_jobs(now: Optional[float] = None) -> int:
    global _last_prune_at
    now = time.time() if now is None else now
    if not _should_prune(now):
        return 0
    _last_prune_at = now
    if JOB_RETENTION_SECONDS < 0:
        return 0
    cutoff = None
    if JOB_RETENTION_SECONDS > 0:
        cutoff = now - JOB_RETENTION_SECONDS

    removed = 0
    for job_id, job in list(_jobs.items()):
        status = str(job.get("status") or "")
        if status not in TERMINAL_STATUSES:
            continue
        if cutoff is None:
            should_remove = True
        else:
            completed_at = job.get("completed_at")
            should_remove = False
            if completed_at:
                try:
                    # best-effort parse of ISO string for pruning
                    completed_ts = datetime.fromisoformat(completed_at.replace("Z", "+00:00")).timestamp()
                    should_remove = completed_ts <= cutoff
                except Exception:
                    should_remove = False
        if should_remove:
            _jobs.pop(job_id, None)
            _job_event_queues.pop(job_id, None)
            _job_tasks.pop(job_id, None)
            removed += 1
    return removed

async def enqueue_job(payload: dict) -> str:
    prune_jobs()
    job_id = str(uuid.uuid4())
    now_iso = _utc_now_iso()
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "result": None,
        "error": None,
        "message": "job queued",
        "created_at": now_iso,
        "updated_at": now_iso,
        "completed_at": None,
        "cancel_requested": False,
        "event_seq": 0,
    }
    _job_event_queues[job_id] = asyncio.Queue()
    # initial event
    await _emit_job_event(
        job_id,
        _build_job_event(
            job_id,
            event="queued",
            status="queued",
            message="job queued",
        ),
    )
    await _queue.put((job_id, payload))
    # update gauge for job queue size
    try:
        metrics.JOB_QUEUE_SIZE.set(_queue.qsize())
    except Exception:
        pass
    return job_id

async def get_job(job_id: str):
    prune_jobs()
    return _jobs.get(job_id)

async def get_job_event_queue(job_id: str) -> asyncio.Queue | None:
    prune_jobs()
    return _job_event_queues.get(job_id)

async def stop_worker():
    """Signal the worker loop to shut down gracefully."""
    await _queue.put(None)

async def _process_job(job_id: str, payload: dict):
    # Use `planner_agent` to process the payload. If streaming is available,
    # forward clean structured progress events to the job event queue.
    from agents import planner_agent
    from core.llm_mode import llm_routing_context
    q = _job_event_queues.get(job_id)
    try:
        job = _jobs.get(job_id)
        if not job:
            return
        if job.get("cancel_requested"):
            _set_job_status(job_id, "cancelled", message="job cancelled before start")
            await _emit_job_event(
                job_id,
                _build_job_event(
                    job_id,
                    event="cancelled",
                    status="cancelled",
                    message="job cancelled before start",
                ),
            )
            return
        _set_job_status(job_id, "running", message="job started")
        if q:
            await _emit_job_event(
                job_id,
                _build_job_event(
                    job_id,
                    event="running",
                    status="running",
                    message="job started",
                ),
            )

        with llm_routing_context(
            llm_mode=payload.get("llm_mode"),
            cloud_provider=payload.get("cloud_provider"),
        ):
            # Prefer streaming plan if available to forward progress events:
            agen_or_result = await planner_agent.plan_trip(
                origin=payload.get("origin"),
                destination=payload.get("destination"),
                date=payload.get("date"),
                user_query=payload.get("user_query"),
                trip_type=payload.get("trip_type"),
                stream=True,
            )

            # If we received an async generator, forward progress events; else, we got final result.
            if hasattr(agen_or_result, "__aiter__"):
                async for chunk in agen_or_result:
                    chunk_text = str(chunk)
                    if chunk_text.startswith("[DONE_JSON]"):
                        json_part = chunk_text[len("[DONE_JSON]"):]
                        try:
                            parsed = json.loads(json_part)
                        except Exception as exc:
                            error_text = f"failed to parse DONE_JSON: {exc}"
                            _set_job_status(job_id, "error", error=error_text, message=error_text)
                            if q:
                                await _emit_job_event(
                                    job_id,
                                    _build_job_event(
                                        job_id,
                                        event="error",
                                        status="error",
                                        message=error_text,
                                        error=error_text,
                                    ),
                                )
                            return
                        _set_job_status(job_id, "done", result=parsed, message="job completed")
                        if q:
                            await _emit_job_event(
                                job_id,
                                _build_job_event(
                                    job_id,
                                    event="done",
                                    status="done",
                                    message="job completed",
                                    data=parsed,
                                    result=parsed,
                                ),
                            )
                        return

                    structured = _parse_sse_frame(chunk_text) if isinstance(chunk, str) else None
                    if structured is not None:
                        event_name = str(structured.get("event") or "job_event")
                        if event_name == "done":
                            # Streaming jobs use [DONE_JSON] for terminal result payload.
                            continue
                        if q:
                            await _emit_job_event(
                                job_id,
                                _build_job_event(
                                    job_id,
                                    event=event_name,
                                    status="running",
                                    data=structured.get("data"),
                                ),
                            )
                        continue

                    if q:
                        await _emit_job_event(
                            job_id,
                            _build_job_event(
                                job_id,
                                event="token",
                                status="running",
                                data={"chunk": chunk_text},
                            ),
                        )

                # Stream ended without DONE_JSON. Produce a final structured result
                # using one non-stream call so polling clients still receive /jobs result.
                if _jobs.get(job_id, {}).get("result") is None and _jobs.get(job_id, {}).get("status") not in TERMINAL_STATUSES:
                    final = await planner_agent.plan_trip(
                        origin=payload.get("origin"),
                        destination=payload.get("destination"),
                        date=payload.get("date"),
                        user_query=payload.get("user_query"),
                        trip_type=payload.get("trip_type"),
                        stream=False,
                    )
                    _set_job_status(job_id, "done", result=final, message="job completed")
                    if q:
                        await _emit_job_event(
                            job_id,
                            _build_job_event(
                                job_id,
                                event="done",
                                status="done",
                                message="job completed",
                                data=final,
                                result=final,
                            ),
                        )
                    return
            else:
                # Non-stream return value (final)
                _set_job_status(job_id, "done", result=agen_or_result, message="job completed")
                if q:
                    await _emit_job_event(
                        job_id,
                        _build_job_event(
                            job_id,
                            event="done",
                            status="done",
                            message="job completed",
                            data=agen_or_result,
                            result=agen_or_result,
                        ),
                    )
                return

        return

    except asyncio.CancelledError:
        _set_job_status(job_id, "cancelled", message="job cancelled")
        if q:
            await _emit_job_event(
                job_id,
                _build_job_event(
                    job_id,
                    event="cancelled",
                    status="cancelled",
                    message="job cancelled",
                ),
            )
        raise
    except Exception:
        error_text = traceback.format_exc()
        _set_job_status(job_id, "error", error=error_text, message="job failed")
        if q:
            await _emit_job_event(
                job_id,
                _build_job_event(
                    job_id,
                    event="error",
                    status="error",
                    message="job failed",
                    error=error_text,
                ),
            )
    finally:
        # close event queue by putting a sentinel
        if q:
            await _emit_job_event(
                job_id,
                _build_job_event(
                    job_id,
                    event="closed",
                    status=str(_jobs.get(job_id, {}).get("status") or "unknown"),
                    message="event stream closed",
                ),
            )

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
                task = asyncio.create_task(_process_job(job_id, payload))
                _job_tasks[job_id] = task
                await task
            except asyncio.CancelledError:
                # Job-level cancellation should not terminate the worker loop.
                pass
            except Exception:
                _set_job_status(job_id, "error", error="worker exception")
            finally:
                _job_tasks.pop(job_id, None)
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


async def request_cancel_job(job_id: str) -> Dict[str, Any]:
    job = _jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    status = str(job.get("status") or "")
    if status in TERMINAL_STATUSES:
        return {"status": "already_terminal", "job": job}
    job["cancel_requested"] = True
    job["updated_at"] = _utc_now_iso()
    if status == "queued":
        _set_job_status(job_id, "cancelled", message="job cancelled before start")
        await _emit_job_event(
            job_id,
            _build_job_event(
                job_id,
                event="cancelled",
                status="cancelled",
                message="job cancelled before start",
            ),
        )
        return {"status": "cancelled", "job": _jobs.get(job_id)}
    task = _job_tasks.get(job_id)
    if task:
        task.cancel()
    return {"status": "cancel_requested", "job": _jobs.get(job_id)}
