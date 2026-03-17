# api/app.py
# NOTE:
# We intentionally rely on FastAPI's default exception handling for
# unexpected errors. The planner layer already converts operational
# failures (timeouts, tool errors, LLM failures) into structured
# JSON responses. Only truly unexpected exceptions propagate as 500,
# which is desirable for visibility and debugging.
# NOTE:
# The /ask endpoint now supports both non‑streaming (JSON) and streaming (SSE)
# responses. Streaming is enabled by passing ?stream=true in the query string.
# Background jobs are triggered by ?async_job=true; they return a 202 with a job_id
# that can be polled via GET /jobs/{job_id} or streamed via GET /jobs/{job_id}/events.

import uuid
import json
import logging
import os
import asyncio
import time
import fcntl                     # for process‑level locking
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Request, Response, HTTPException, Query, Header, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Use module import instead of direct function import for better testability
import agents.planner_agent as planner_agent

# Import specific tool exceptions for granular error handling
from tools.airline_api import AirlineAPIError
from core.http_client import close_client
from core.request_context import set_request_id
from core.logging_config import setup_logging
from core.health import full_health_check
from core.async_llm_client import init_llm_client, close_llm_client
from core import job_queue                     # background job worker
from core.api_key_manager import key_manager    # key rotation manager
from agents.cloud_llm import on_key_event       # callback for key changes

logger = logging.getLogger(__name__)

# --- Pluggable lock helpers (file-based default, redis optional) ---
try:
    import redis.asyncio as redis_async  # optional dependency for redis lock backend
except ImportError:
    redis_async = None

KEY_MANAGER_LOCK_BACKEND = os.getenv("KEY_MANAGER_LOCK_BACKEND", "file").lower()
KEY_MANAGER_REDIS_URL = os.getenv("KEY_MANAGER_REDIS_URL", "redis://localhost:6379/0")
KEY_MANAGER_LOCK_NAME = os.getenv("KEY_MANAGER_LOCK_NAME", "llm:key_refresh_lock")
KEY_MANAGER_LOCK_TTL = int(os.getenv("KEY_MANAGER_LOCK_TTL_SECONDS", "60"))  # lock TTL for redis
KEY_MANAGER_LOCK_PATH = os.getenv("KEY_MANAGER_LOCK_PATH", "/tmp/llm_key_refresh.lock")


def _acquire_process_lock(path: str) -> Optional[int]:
    """
    Try to acquire an exclusive lock on the given file.
    Returns a file descriptor if successful, None if another process holds the lock.
    """
    fd = None
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except (IOError, OSError, BlockingIOError):
        # Lock already held by another process
        if fd is not None:
            os.close(fd)
        return None
    except Exception:
        # Unexpected error; fall back to no lock
        if fd is not None:
            os.close(fd)
        return None


async def _acquire_redis_lock(redis_url: str, name: str, ttl: int):
    """Try to acquire a Redis-based distributed lock. Returns a tuple (client, lock) on success, or (None, None)."""
    if redis_async is None:
        return None, None
    client = None
    try:
        client = redis_async.from_url(redis_url)
        lock = client.lock(name, timeout=ttl)
        acquired = await lock.acquire(blocking=False)
        if acquired:
            return client, lock
        else:
            await client.close()
            return None, None
    except Exception:
        # Any redis error -> don't acquire
        if client:
            try:
                await client.close()
            except Exception:
                pass
        return None, None


async def _acquire_pluggable_lock():
    """Return a tuple (backend, handle) where backend is 'file' or 'redis' and handle is fd or (client, lock)."""
    if KEY_MANAGER_LOCK_BACKEND == "redis":
        client, lock = await _acquire_redis_lock(KEY_MANAGER_REDIS_URL, KEY_MANAGER_LOCK_NAME, KEY_MANAGER_LOCK_TTL)
        if lock:
            return "redis", (client, lock)
        return "redis", None
    # default: file lock
    fd = _acquire_process_lock(KEY_MANAGER_LOCK_PATH)
    if fd is not None:
        return "file", fd
    return "file", None


async def prewarm_llm():
    """
    Ollama prewarm with retries and exponential backoff.
    Will not crash startup if Ollama is slow or unavailable.
    """
    from agents import ollama_client
    from agents.ollama_client import OllamaError

    timeout = int(os.getenv("OLLAMA_PREWARM_TIMEOUT", "60"))
    max_retries = int(os.getenv("OLLAMA_PREWARM_RETRIES", "3"))
    backoff = 1

    for attempt in range(1, max_retries + 1):
        try:
            # Using a hardcoded model name; adjust if needed.
            await ollama_client.generate(
                prompt="hi",
                model="openhermes",
                timeout=timeout,
                stream=False
            )
            logger.info("Ollama prewarm OK")
            return
        except OllamaError as e:
            logger.warning("Ollama prewarm attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                logger.warning(
                    "Ollama prewarm failed after %d attempts — continuing without prewarm",
                    max_retries
                )
                return
        except Exception as e:
            # Catch any other unexpected errors (e.g., import issues) and treat as failure
            logger.warning("Unexpected error during prewarm attempt %d: %s", attempt, e)
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                logger.warning("Prewarm aborted after %d attempts", max_retries)
                return


def require_admin_token(x_admin_token: str = Header(...)):
    """Dependency to protect admin endpoints with a token from environment."""
    expected = os.getenv("ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_complete = False

    # Startup: configure structured JSON logging
    setup_logging()

    # Initialize shared LLM client
    await init_llm_client()

    # Load API keys from environment into the key manager
    try:
        await key_manager.load_env_keys()
    except Exception:
        logger.exception("key_manager_load_failed")

    # ---- Register key event listener early (idempotent) ----
    try:
        already_registered = False
        # best-effort detection to avoid duplicate registration in same process
        listeners = getattr(key_manager, "_key_event_listeners", None)
        if listeners is not None:
            try:
                if on_key_event in listeners:
                    already_registered = True
            except Exception:
                # fall back to identity scan
                for item in list(listeners):
                    if getattr(item, "__name__", None) == getattr(on_key_event, "__name__", None):
                        already_registered = True
                        break

        if not already_registered:
            key_manager.register_key_event_listener(on_key_event)
            app.state.cloud_llm_listener_registered = True
            logger.info("Registered cloud LLM key event listener")
        else:
            logger.info("Cloud LLM key event listener already registered in this process")
    except Exception:
        logger.exception("Failed to register cloud LLM key event listener")

    # ---- Pluggable lock to ensure only one process/replica runs the refresh loop ----
    lock_backend, lock_handle = await _acquire_pluggable_lock()
    should_run_refresh = lock_handle is not None

    # Fallback: env var override (useful for containers where you set one replica manually)
    if not should_run_refresh and os.getenv("RUN_KEY_REFRESH", "").lower() in ("1", "true"):
        logger.warning(
            "RUN_KEY_REFRESH=true but lock not acquired; starting refresh loop anyway. "
            "Ensure only one replica has this variable set."
        )
        should_run_refresh = True

    if should_run_refresh:
        logger.info("Starting key manager background refresh loop (lock_backend=%s).", lock_backend)
        # Save lock handle for shutdown cleanup
        app.state.key_manager_lock_backend = lock_backend
        app.state.key_manager_lock_handle = lock_handle

        # Start the key manager's background refresh loop (interval configurable)
        refresh_interval = int(os.getenv("KEY_ENV_MONITOR_TICK", "60"))
        # start_refresh_loop is synchronous; it creates an internal task.
        key_manager.start_refresh_loop(
            interval_seconds=refresh_interval,
            skip_lock_check=True      # we already acquired the lock ourselves
        )
        # Store the internal task so we can cancel it on shutdown
        app.state.key_manager_task = key_manager._refresh_task
    else:
        logger.info("Another process/replica holds the key manager lock; not starting refresh loop.")
        app.state.key_manager_lock_backend = None
        app.state.key_manager_lock_handle = None
        app.state.key_manager_task = None

    # Start the background job worker loop (always needed)
    app.state.job_worker = asyncio.create_task(job_queue.worker_loop())

    # Optional prewarm (non‑blocking)
    if os.getenv("PLANNER_PREWARM") == "1":
        async def background_prewarm():
            try:
                await prewarm_llm()
            except Exception:
                logger.exception("Background prewarm failed")
        asyncio.create_task(background_prewarm())

    app.state.startup_complete = True
    yield

    app.state.startup_complete = False

    # Shutdown: gracefully stop the job worker
    try:
        await job_queue.stop_worker()
        await app.state.job_worker
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

    # Stop the key manager's background refresh loop only if we started it
    if getattr(app.state, "key_manager_task", None):
        try:
            # stop_refresh_loop is synchronous; it cancels the internal task.
            key_manager.stop_refresh_loop()
        except Exception:
            logger.exception("key_manager_stop_refresh_loop_failed")

        # Cancel background task if still running (though stop_refresh_loop should have done it)
        task = app.state.key_manager_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("key_manager_task_cancel_failed")

    # Release the lock we acquired (backend-specific)
    backend = getattr(app.state, "key_manager_lock_backend", None)
    handle = getattr(app.state, "key_manager_lock_handle", None)
    if backend == "file" and handle is not None:
        try:
            os.close(handle)
            logger.info("Released file lock for key manager refresh.")
        except Exception:
            logger.exception("failed_to_release_file_lock")
    elif backend == "redis" and handle is not None:
        client, lock = handle
        try:
            # release the redis lock and close connection
            await lock.release()
        except Exception:
            logger.exception("failed_to_release_redis_lock")
        try:
            await client.close()
        except Exception:
            pass

    # Clean up clients
    await close_llm_client()
    await close_client()

    # Ensure cloud_llm provider adapters are closed (safe even if none initialised)
    try:
        import agents.cloud_llm as cloud_llm
        await cloud_llm.close_client()
    except Exception:
        logger.exception("cloud_llm_close_failed_during_lifespan_shutdown")


app = FastAPI(
    title="LLM Travel Agent",
    lifespan=lifespan
)

# Add CORS middleware – now configurable via environment variable
# Read production origins from environment, fallback to localhost for dev
env_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173")
allowed_origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to log raw request bodies for debugging 422 errors
@app.middleware("http")
async def log_request_body(request: Request, call_next):
    try:
        body_bytes = await request.body()
        logger.debug(
            "Raw request body",
            extra={
                "path": request.url.path,
                "method": request.method,
                "body": body_bytes.decode(errors="replace")
            }
        )
        # Reattach the body so FastAPI can still read it
        request._body = body_bytes
    except Exception:
        # Logging must never break the request
        pass
    response = await call_next(request)
    return response


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Generate a unique request ID and store it in the context."""
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


class AskRequest(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    date: Optional[str] = None
    user_query: Optional[str] = None
    trip_type: Optional[str] = None          # now optional, planner may default to "Business"

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        """If date is provided, ensure it's in YYYY-MM-DD format."""
        if v is None or v == "":
            return None
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be YYYY-MM-DD")

        return v

    @model_validator(mode="after")
    def normalize_and_validate(self):
        # Normalize: strip whitespace, convert empty strings to None
        self.origin = self.origin.strip() if self.origin else None
        self.destination = self.destination.strip() if self.destination else None
        self.date = self.date.strip() if self.date else None
        self.user_query = self.user_query.strip() if self.user_query else None
        self.trip_type = self.trip_type.strip() if self.trip_type else None

        origin = self.origin
        destination = self.destination
        date = self.date
        user_query = self.user_query

        # Rule 1 — reject completely empty
        if not origin and not destination and not date and not user_query:
            raise ValueError(
                "At least one of user_query or origin/destination must be provided."
            )

        # Rule 2 — structured must include both origin and destination together
        if (origin or destination) and not (origin and destination):
            raise ValueError(
                "Both origin and destination must be provided together."
            )

        return self


@app.post("/ask")
async def ask(
    req: AskRequest,
    stream: bool = False,
    async_job: bool = Query(False, description="Enqueue request as background job")
):
    """
    Plan a trip based on the user's request.
    - If `async_job=true`, the request is enqueued and returns a 202 with a job_id.
    - Otherwise:
        - If `stream=false` (default), returns a single JSON response.
        - If `stream=true`, returns a Server‑Sent Events (SSE) stream of tokens.
    """
    # Define global timeout early so it's available in exception handlers
    GLOBAL_TIMEOUT = int(os.getenv("PLANNER_GLOBAL_TIMEOUT", "60"))

    # Use the already normalized values from the model
    origin = req.origin
    destination = req.destination
    # Detect structured‑only mode (no user query, both origin/destination present, date missing)
    is_structured_only = (
        not req.user_query
        and origin
        and destination
        and not req.date
    )

    # Compute effective date (structured default rule applies to all branches)
    DEFAULT_STRUCTURED_OFFSET_DAYS = int(
        os.getenv("DEFAULT_STRUCTURED_OFFSET_DAYS", "15")
    )
    effective_date = req.date

    if is_structured_only:
        effective_date = (
            datetime.now() + timedelta(days=DEFAULT_STRUCTURED_OFFSET_DAYS)
        ).strftime("%Y-%m-%d")

    # Determine the user query to send to the planner
    if is_structured_only:
        # For pure structured requests, give a sensible default prompt
        planner_user_query = "Provide best available option."
    else:
        planner_user_query = req.user_query or ""

    try:
        # Background job branch
        if async_job:
            from core.job_queue import enqueue_job
            # Exclude None fields for a cleaner payload
            payload = req.model_dump(exclude_none=True)
            # Override with processed values
            payload["origin"] = origin
            payload["destination"] = destination
            payload["date"] = effective_date
            payload["user_query"] = planner_user_query
            job_id = await enqueue_job(payload)
            return Response(
                status_code=202,
                content=json.dumps({"job_id": job_id}),
                media_type="application/json"
            )

        if stream:
            # Streaming branch: call planner with stream=True
            # No outer timeout – planner handles streaming timeouts internally
            agen_or_result = await planner_agent.plan_trip(
                origin=origin,
                destination=destination,
                date=effective_date,
                user_query=planner_user_query,
                trip_type=req.trip_type,
                stream=True
            )

            async def event_stream():
                # If the planner returns an async generator, iterate and yield SSE frames
                if hasattr(agen_or_result, "__aiter__"):
                    async for chunk in agen_or_result:
                        # Basic SSE framing; newlines in chunk should be escaped if needed
                        yield f"data: {chunk}\n\n"
                    # Final done event
                    yield "event: done\ndata: \n\n"
                else:
                    # Fallback: if planner returned a dict (non‑streaming), send it as one event
                    yield f"data: {json.dumps(agen_or_result)}\n\n"
                    yield "event: done\ndata: \n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Non‑streaming branch: apply global timeout
        result = await asyncio.wait_for(
            planner_agent.plan_trip(
                origin=origin,
                destination=destination,
                date=effective_date,
                user_query=planner_user_query,
                trip_type=req.trip_type,
            ),
            timeout=GLOBAL_TIMEOUT
        )

        # If the planner returns a dict with an "error" key, treat it as a client error.
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {GLOBAL_TIMEOUT} seconds")
        raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        # Re-raise HTTPExceptions that we intentionally throw
        raise
    except AirlineAPIError as e:
        # Upstream tool failed: 502 Bad Gateway is appropriate
        logger.exception("Airline API failure")
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        # Defensive: bad data formatting inside planner
        logger.exception("Bad request data")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in /ask")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Retrieve the current status and result of a background job."""
    from core.job_queue import get_job
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str):
    """SSE stream of events for a background job."""
    queue = await job_queue.get_job_event_queue(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_stream():
        while True:
            # Stop if client disconnected
            if await request.is_disconnected():
                break
            try:
                evt = await queue.get()
            except asyncio.CancelledError:
                break
            if evt is None:
                break

            # Deep‑safe JSON serialization
            def to_serializable(obj):
                if hasattr(obj, "model_dump"):          # Pydantic v2
                    return obj.model_dump()
                if hasattr(obj, "dict"):                # Pydantic v1
                    return obj.dict()
                if isinstance(obj, dict):
                    return {k: to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [to_serializable(i) for i in obj]
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                return str(obj)                          # fallback

            evt = to_serializable(evt)

            # Send event as SSE data (client will parse JSON)
            yield f"data: {json.dumps(evt)}\n\n"

            # Close stream on terminal event
            if evt.get("type") in ("closed", "done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Admin‑protected debug endpoints
@app.get("/debug/keys", dependencies=[Depends(require_admin_token)])
async def debug_keys():
    """Return masked keys and their status (active/exhausted until). Requires admin token."""
    try:
        # key_manager.status() may be async or sync; handle both
        status = key_manager.status()
        if asyncio.iscoroutine(status):
            status = await status
        return status
    except Exception:
        logger.exception("debug_keys_failed")
        raise HTTPException(status_code=500, detail="key manager error")


@app.post("/debug/keys/reload", dependencies=[Depends(require_admin_token)])
async def reload_keys_endpoint():
    """Force a reload of API keys from environment variables. Requires admin token."""
    try:
        await key_manager.load_env_keys()
        return {"status": "reloaded"}
    except Exception:
        logger.exception("debug_keys_reload_failed")
        raise HTTPException(status_code=500, detail="reload failed")


@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe."""
    if not getattr(app.state, "startup_complete", False):
        health = {"status": "starting"}
        return Response(
            content=json.dumps(health),
            status_code=503,
            media_type="application/json"
        )
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Lightweight health check for container probes (no external API calls)."""
    logger.debug("lightweight health check")

    async def _check_key_manager() -> str:
        try:
            status = key_manager.status()
            if asyncio.iscoroutine(status):
                await status
            return "ok"
        except Exception:
            logger.exception("lightweight_health_key_manager_failed")
            return "fail"

    async def _check_database() -> str:
        try:
            from agents.database import SessionLocal  # local import keeps startup behavior unchanged
            from sqlalchemy import text
        except Exception:
            # Database layer not available in this runtime.
            return "unavailable"

        def _ping_db() -> None:
            db = SessionLocal()
            try:
                db.execute(text("SELECT 1"))
            finally:
                db.close()

        try:
            await asyncio.wait_for(asyncio.to_thread(_ping_db), timeout=0.2)
            return "ok"
        except Exception:
            logger.exception("lightweight_health_database_failed")
            return "fail"

    async def _check_ollama() -> str:
        # Optional lightweight check. Do not fail overall status if this fails.
        try:
            from agents.ollama_client import health_check as ollama_health_check
        except Exception:
            return "unavailable"

        try:
            res = await asyncio.wait_for(ollama_health_check(), timeout=0.05)
            return "ok" if res == "ok" else "fail"
        except Exception:
            logger.warning("lightweight_health_ollama_failed", exc_info=True)
            return "fail"

    dependencies = {
        "app": "ok" if getattr(app.state, "startup_complete", False) else "fail",
        "key_manager": "fail",
        "database": "unavailable",
        "ollama": "unavailable",
    }

    key_manager_status, database_status, ollama_status = await asyncio.gather(
        _check_key_manager(),
        _check_database(),
        _check_ollama(),
    )
    dependencies["key_manager"] = key_manager_status
    dependencies["database"] = database_status
    dependencies["ollama"] = ollama_status

    # /health should remain stable and avoid external-API-triggered failures.
    hard_fail_fields = ("app", "key_manager", "database")
    status = "fail" if any(dependencies[f] == "fail" for f in hard_fail_fields) else "ok"

    return {"status": status, "dependencies": dependencies}


@app.get("/health/deep")
async def health_deep():
    """Deep health check (includes external API checks)."""
    logger.debug("deep health check (external APIs)")
    start = time.monotonic()
    result = await full_health_check()
    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.debug(
        "deep health check complete",
        extra={"elapsed_ms": elapsed_ms, "status": result.get("status")},
    )
    return result


@app.get("/health/keys")
async def health_keys():
    """Return key manager metadata status (no secret values)."""
    status = await key_manager.get_status()

    out = {}
    for service, entries in (status or {}).items():
        rows = []
        if isinstance(entries, list):
            iterable = enumerate(entries)
        elif isinstance(entries, dict):
            # Backward compatibility if a dict shape is returned.
            iterable = []
            for k, v in entries.items():
                try:
                    idx = int(k)
                except Exception:
                    idx = len(iterable)
                iterable.append((idx, v))
        else:
            iterable = [(0, entries)]

        for idx, entry in iterable:
            if isinstance(entry, dict):
                rows.append(
                    {
                        "index": entry.get("index", idx),
                        "active": bool(entry.get("active", False)),
                        "in_use": int(entry.get("in_use", 0) or 0),
                        "exhausted_until": entry.get("exhausted_until"),
                    }
                )
            elif isinstance(entry, str):
                rows.append(
                    {
                        "index": idx,
                        "active": entry == "active",
                        "in_use": 0,
                        "exhausted_until": None,
                    }
                )
            else:
                rows.append(
                    {
                        "index": idx,
                        "active": False,
                        "in_use": 0,
                        "exhausted_until": None,
                    }
                )
        out[service] = rows

    return out


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/version")
async def version():
    """
    Return version information to help debug deployment consistency.
    - git_commit: set via environment variable GIT_COMMIT (optional)
    - timestamp: last modification time of this file
    """
    return {
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "file_mtime": os.path.getmtime(__file__)
    }
