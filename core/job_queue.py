"""Process-local async job queue contract.

This queue is intentionally in-memory and single-process. It is the canonical
runtime contract for this repository's single-node deployment baseline.
Distributed async semantics (shared queue/state across workers or nodes) are
explicitly deferred and out of scope for this runtime path.

WARNING: Jobs are EPHEMERAL and will be lost on process restart or crash.
User-visible persistent async jobs/tracking currently requires external
persistence (e.g., database-backed job tracking) for true durability.
This module provides at-most-once semantics within a single process only.
"""

import asyncio
import copy
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Prometheus metrics
from core import metrics

logger = logging.getLogger(__name__)

_jobs: Dict[str, Dict[str, Any]] = {}
_job_event_queues: Dict[str, asyncio.Queue] = {}
_worker_task: asyncio.Task | None = None
_job_tasks: Dict[str, asyncio.Task] = {}

JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "3600"))
JOB_PRUNE_INTERVAL_SECONDS = int(os.getenv("JOB_PRUNE_INTERVAL_SECONDS", "300"))
JOB_QUEUE_MAXSIZE = max(1, int(os.getenv("JOB_QUEUE_MAXSIZE", "64")))
JOB_MAX_IN_MEMORY = max(JOB_QUEUE_MAXSIZE, int(os.getenv("JOB_MAX_IN_MEMORY", "512")))
JOB_TIMEOUT_SECONDS = max(10, int(os.getenv("JOB_TIMEOUT_SECONDS", "300")))
_last_prune_at: float = 0.0
_queue: asyncio.Queue = asyncio.Queue(maxsize=JOB_QUEUE_MAXSIZE)

TERMINAL_STATUSES = {"done", "error", "cancelled"}
_async_state_lock: asyncio.Lock | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_session():
    from agents.database import SessionLocal
    return SessionLocal()


async def _db_upsert_job(job_id: str, job: Dict[str, Any], payload: Optional[Dict[str, Any]] = None):
    def _upsert():
        session = _db_session()
        try:
            from agents.database import BackgroundJob
            existing = session.query(BackgroundJob).filter(BackgroundJob.job_id == job_id).first()
            if existing:
                existing.status = job['status']
                existing.result = job.get('result')
                existing.error = job.get('error')
                existing.message = job['message']
                existing.updated_at = job['updated_at']
                existing.completed_at = job.get('completed_at')
                existing.cancel_requested = job.get('cancel_requested', False)
                existing.event_seq = job['event_seq']
                if payload is not None:
                    existing.payload = payload
            else:
                new_job = BackgroundJob(
                    job_id=job_id,
                    owner_principal_id=job['owner_principal_id'],
                    status=job['status'],
                    result=job.get('result'),
                    error=job.get('error'),
                    message=job['message'],
                    created_at=job['created_at'],
                    updated_at=job['updated_at'],
                    completed_at=job.get('completed_at'),
                    cancel_requested=job.get('cancel_requested', False),
                    event_seq=job['event_seq'],
                    payload=payload,
                )
                session.add(new_job)
            session.commit()
        finally:
            session.close()

    await asyncio.to_thread(_upsert)


async def _db_delete_jobs(job_ids: list[str]):
    if not job_ids:
        return
    def _delete():
        session = _db_session()
        try:
            from agents.database import BackgroundJob
            session.query(BackgroundJob).filter(BackgroundJob.job_id.in_(job_ids)).delete(synchronize_session=False)
            session.commit()
        finally:
            session.close()

    await asyncio.to_thread(_delete)


async def _load_jobs_on_startup():
    def _load():
        session = _db_session()
        try:
            from agents.database import BackgroundJob
            rows = session.query(BackgroundJob).all()
            results = []
            for row in rows:
                job = {
                    "job_id": row.job_id,
                    "owner_principal_id": row.owner_principal_id,
                    "status": row.status,
                    "result": row.result,
                    "error": row.error,
                    "message": row.message,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "completed_at": row.completed_at,
                    "cancel_requested": bool(row.cancel_requested),
                    "event_seq": row.event_seq,
                }
                results.append((job, row.payload))
            return results
        finally:
            session.close()

    results = await asyncio.to_thread(_load)
    for job, pay_raw in results:
        job_id = job["job_id"]
        _jobs[job_id] = job

        if job["status"] in ("queued", "running"):
            payload = pay_raw if isinstance(pay_raw, dict) else {}
            if job["status"] == "running":
                job["status"] = "queued"
                job["message"] = "re-queued after server restart"
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                await _db_upsert_job(job_id, job)

            await _queue.put((job_id, payload))


async def initialize_job_queue():
    """Initialize the database and load pending jobs."""
    from agents.database import init_db
    init_db()
    await _load_jobs_on_startup()


JOB_RUNTIME_WARNING_MESSAGE = (
    "Async jobs are ephemeral and stored in-memory only. "
    "Jobs are lost on process restart — at-most-once semantics within a single process."
)


def job_runtime_warning_payload() -> Dict[str, Any]:
    return {
        "jobs_tracking_memory_only": True,
        "lost_on_restart": True,
        "durable_persistence": False,
        "warning": JOB_RUNTIME_WARNING_MESSAGE,
    }


def _get_async_lock() -> asyncio.Lock:

    global _async_state_lock
    if _async_state_lock is None:
        _async_state_lock = asyncio.Lock()
    return _async_state_lock


def _snapshot_job(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(job, dict):
        return None
    return copy.deepcopy(job)


class JobQueueBackpressureError(RuntimeError):
    def __init__(self, reason: str, *, retry_after_seconds: int = 1):
        super().__init__(reason)
        self.reason = str(reason or "queue_backpressure")
        self.retry_after_seconds = max(1, int(retry_after_seconds))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def _next_event_seq(job_id: str) -> int:
    async with _get_async_lock():
        job = _jobs.get(job_id)
        if not isinstance(job, dict):
            return 0
        seq = int(job.get("event_seq", 0) or 0) + 1
        job["event_seq"] = seq
        return seq


async def _build_job_event(
    job_id: str,
    *,
    event: str,
    status: str,
    message: Optional[str] = None,
    data: Optional[Any] = None,
    result: Optional[Any] = None,
    error: Optional[Any] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event": event,
        "job_id": job_id,
        "status": status,
        "timestamp": _utc_now_iso(),
        "sequence": await _next_event_seq(job_id),
        "runtime": job_runtime_warning_payload(),
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
        except (json.JSONDecodeError, ValueError):
            data_payload = data_text
    return {"event": event_name, "data": data_payload, "data_text": data_text}


def _public_job_error(*, code: str, message: str, job_id: str) -> Dict[str, Any]:
    return {
        "code": str(code or "job_error"),
        "message": str(message or "Job failed."),
        "job_id": str(job_id),
    }


def _should_prune(now: float) -> bool:
    if JOB_RETENTION_SECONDS <= 0:
        return True
    if JOB_PRUNE_INTERVAL_SECONDS <= 0:
        return True
    return (now - _last_prune_at) >= JOB_PRUNE_INTERVAL_SECONDS


async def _set_job_status(
    job_id: str,
    status: str,
    *,
    error: Optional[Any] = None,
    result: Optional[Any] = None,
    message: Optional[str] = None,
) -> None:
    async with _get_async_lock():
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
        
        # Persist to DB
        await _db_upsert_job(job_id, job)


async def _emit_job_event(job_id: str, payload: Dict[str, Any]) -> None:
    async with _get_async_lock():
        queue = _job_event_queues.get(job_id)
    if queue is None:
        return
    await queue.put(payload)


def _prune_jobs_unlocked(now: float) -> list[str]:
    global _last_prune_at
    if not _should_prune(now):
        return []
    _last_prune_at = now
    if JOB_RETENTION_SECONDS < 0:
        return []
    cutoff = None
    if JOB_RETENTION_SECONDS > 0:
        cutoff = now - JOB_RETENTION_SECONDS

    removed_ids = []
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
                except (ValueError, TypeError, OSError):
                    should_remove = False
        if should_remove:
            _jobs.pop(job_id, None)
            _job_event_queues.pop(job_id, None)
            _job_tasks.pop(job_id, None)
            removed_ids.append(job_id)
    return removed_ids


def prune_jobs(now: Optional[float] = None) -> int:
    """
    Backward-compatible synchronous prune helper (primarily for tests/tools).
    Async request/job hot paths should use `prune_jobs_async()`.
    """
    now_ts = time.time() if now is None else now
    return len(_prune_jobs_unlocked(now_ts))


async def prune_jobs_async(now: Optional[float] = None) -> int:
    now_ts = time.time() if now is None else now
    async with _get_async_lock():
        removed_ids = _prune_jobs_unlocked(now_ts)
    if removed_ids:
        await _db_delete_jobs(removed_ids)
    return len(removed_ids)

async def enqueue_job(payload: dict, *, owner_principal_id: str) -> str:
    await prune_jobs_async()
    owner_id = str(owner_principal_id or "").strip()
    if not owner_id:
        raise ValueError("owner_principal_id is required")
    job_id = str(uuid.uuid4())
    now_iso = _utc_now_iso()
    job_data = {
        "job_id": job_id,
        "owner_principal_id": owner_id,
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
    async with _get_async_lock():
        if _queue.full():
            raise JobQueueBackpressureError("queue_full", retry_after_seconds=1)
        if len(_jobs) >= JOB_MAX_IN_MEMORY:
            _prune_jobs_unlocked(time.time())
            if len(_jobs) >= JOB_MAX_IN_MEMORY:
                raise JobQueueBackpressureError("job_registry_full", retry_after_seconds=2)
        _jobs[job_id] = job_data
        _job_event_queues[job_id] = asyncio.Queue()
        # Persist to DB
        await _db_upsert_job(job_id, job_data, payload=payload)
    # initial event
    await _emit_job_event(
        job_id,
        await _build_job_event(
            job_id,
            event="queued",
            status="queued",
            message="job queued",
        ),
    )
    try:
        _queue.put_nowait((job_id, payload))
    except asyncio.QueueFull:
        async with _get_async_lock():
            _jobs.pop(job_id, None)
            _job_event_queues.pop(job_id, None)
        raise JobQueueBackpressureError("queue_full", retry_after_seconds=1)
    # update gauge for job queue size
    try:
        metrics.JOB_QUEUE_SIZE.set(_queue.qsize())
    except (AttributeError, TypeError):
        pass
    return job_id

async def get_job(job_id: str, *, owner_principal_id: Optional[str] = None):
    await prune_jobs_async()
    async with _get_async_lock():
        job = _jobs.get(job_id)
        if job is None:
            return None
        if owner_principal_id is not None and str(job.get("owner_principal_id") or "") != str(owner_principal_id):
            return None
        return _snapshot_job(job)

async def get_job_event_queue(job_id: str, *, owner_principal_id: Optional[str] = None) -> asyncio.Queue | None:
    await prune_jobs_async()
    async with _get_async_lock():
        if owner_principal_id is not None:
            job = _jobs.get(job_id)
            if job is None:
                return None
            if str(job.get("owner_principal_id") or "") != str(owner_principal_id):
                return None
        return _job_event_queues.get(job_id)


async def list_jobs(*, owner_principal_id: str, limit: int = 100) -> list[Dict[str, Any]]:
    await prune_jobs_async()
    owner_id = str(owner_principal_id or "").strip()
    max_items = max(1, min(int(limit or 100), 500))
    async with _get_async_lock():
        rows: list[Dict[str, Any]] = []
        for job in _jobs.values():
            if str(job.get("owner_principal_id") or "") != owner_id:
                continue
            rows.append(_snapshot_job(job) or {})
    rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return rows[:max_items]

async def stop_worker():
    """Signal the worker loop to shut down gracefully."""
    await _queue.put(None)

async def _process_job(job_id: str, payload: dict):
    # Use `planner_agent` to process the payload. If streaming is available,
    # forward clean structured progress events to the job event queue.
    from agents import planner_agent
    from core.llm_mode import llm_routing_context
    async with _get_async_lock():
        q = _job_event_queues.get(job_id)
    try:
        async with _get_async_lock():
            job = _jobs.get(job_id)
        if not job:
            return
        if job.get("cancel_requested"):
            await _set_job_status(job_id, "cancelled", message="job cancelled before start")
            await _emit_job_event(
                job_id,
                await _build_job_event(
                    job_id,
                    event="cancelled",
                    status="cancelled",
                    message="job cancelled before start",
                ),
            )
            return
        await _set_job_status(job_id, "running", message="job started")
        if q:
            await _emit_job_event(
                job_id,
                await _build_job_event(
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
                        except (json.JSONDecodeError, ValueError):
                            logger.exception(
                                "job_done_json_parse_failed",
                                extra={"job_id": job_id},
                            )
                            public_error = _public_job_error(
                                code="job_done_payload_invalid",
                                message="Job failed due to an internal response parsing error.",
                                job_id=job_id,
                            )
                            await _set_job_status(
                                job_id,
                                "error",
                                error=public_error,
                                message=public_error["message"],
                            )
                            if q:
                                await _emit_job_event(
                                    job_id,
                                    await _build_job_event(
                                        job_id,
                                        event="error",
                                        status="error",
                                        message=public_error["message"],
                                        error=public_error,
                                    ),
                                )
                            return
                        await _set_job_status(job_id, "done", result=parsed, message="job completed")
                        if q:
                            await _emit_job_event(
                                job_id,
                                await _build_job_event(
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
                                await _build_job_event(
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
                            await _build_job_event(
                                job_id,
                                event="token",
                                status="running",
                                data={"chunk": chunk_text},
                            ),
                        )

                # Stream ended without DONE_JSON. Produce a final structured result
                # using one non-stream call so polling clients still receive /jobs result.
                async with _get_async_lock():
                    current_job = _jobs.get(job_id, {})
                    needs_terminal_result = (
                        current_job.get("result") is None
                        and current_job.get("status") not in TERMINAL_STATUSES
                    )
                if needs_terminal_result:
                    final = await planner_agent.plan_trip(
                        origin=payload.get("origin"),
                        destination=payload.get("destination"),
                        date=payload.get("date"),
                        user_query=payload.get("user_query"),
                        trip_type=payload.get("trip_type"),
                        stream=False,
                    )
                    await _set_job_status(job_id, "done", result=final, message="job completed")
                    if q:
                        await _emit_job_event(
                            job_id,
                            await _build_job_event(
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
                await _set_job_status(job_id, "done", result=agen_or_result, message="job completed")
                if q:
                    await _emit_job_event(
                        job_id,
                        await _build_job_event(
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
        await _set_job_status(job_id, "cancelled", message="job cancelled")
        if q:
            await _emit_job_event(
                job_id,
                await _build_job_event(
                    job_id,
                    event="cancelled",
                    status="cancelled",
                    message="job cancelled",
                ),
            )
        raise
    except Exception:
        logger.exception("job_processing_failed", extra={"job_id": job_id})
        public_error = _public_job_error(
            code="job_execution_failed",
            message="Job failed due to an internal error.",
            job_id=job_id,
        )
        await _set_job_status(job_id, "error", error=public_error, message=public_error["message"])
        if q:
            await _emit_job_event(
                job_id,
                await _build_job_event(
                    job_id,
                    event="error",
                    status="error",
                    message=public_error["message"],
                    error=public_error,
                ),
            )
    finally:
        # close event queue by putting a sentinel
        if q:
            async with _get_async_lock():
                terminal_status = str(_jobs.get(job_id, {}).get("status") or "unknown")
            await _emit_job_event(
                job_id,
                await _build_job_event(
                    job_id,
                    event="closed",
                    status=terminal_status,
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
            except (AttributeError, TypeError):
                pass

            # 🟢 Sentinel handling
            if item is None:
                break

            job_id, payload = item
            try:
                task = asyncio.create_task(_process_job(job_id, payload))
                async with _get_async_lock():
                    _job_tasks[job_id] = task
                await asyncio.wait_for(task, timeout=JOB_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.error("job_timeout", extra={"job_id": job_id, "timeout_s": JOB_TIMEOUT_SECONDS})
                await _set_job_status(job_id, "error", error=f"job timed out after {JOB_TIMEOUT_SECONDS}s")
            except asyncio.CancelledError:
                # Job-level cancellation should not terminate the worker loop.
                pass
            except Exception as exc:
                logger.error("worker_job_exception", extra={"job_id": job_id, "error": str(exc)})
                await _set_job_status(job_id, "error", error=f"worker exception: {type(exc).__name__}")
            finally:
                async with _get_async_lock():
                    _job_tasks.pop(job_id, None)
                # ensure we mark queue task done and update gauge
                try:
                    _queue.task_done()
                except Exception:
                    pass
                try:
                    metrics.JOB_QUEUE_SIZE.set(_queue.qsize())
                except (AttributeError, TypeError):
                    pass
    except asyncio.CancelledError:
        # 🔥 CRITICAL: swallow cancellation cleanly
        # Prevent cancellation from bubbling into TestClient shutdown
        pass


async def request_cancel_job(job_id: str, *, owner_principal_id: Optional[str] = None) -> Dict[str, Any]:
    task = None
    async with _get_async_lock():
        job = _jobs.get(job_id)
        if not job:
            return {"status": "not_found"}
        if owner_principal_id is not None and str(job.get("owner_principal_id") or "") != str(owner_principal_id):
            return {"status": "not_found"}
        status = str(job.get("status") or "")
        if status in TERMINAL_STATUSES:
            return {"status": "already_terminal", "job": _snapshot_job(job)}
        job["cancel_requested"] = True
        job["updated_at"] = _utc_now_iso()
        await _db_upsert_job(job_id, job)
        if status != "queued":
            task = _job_tasks.get(job_id)

    if status == "queued":
        await _set_job_status(job_id, "cancelled", message="job cancelled before start")
        await _emit_job_event(
            job_id,
            await _build_job_event(
                job_id,
                event="cancelled",
                status="cancelled",
                message="job cancelled before start",
            ),
        )
        async with _get_async_lock():
            return {"status": "cancelled", "job": _snapshot_job(_jobs.get(job_id))}
    if task:
        task.cancel()
    async with _get_async_lock():
        return {"status": "cancel_requested", "job": _snapshot_job(_jobs.get(job_id))}
