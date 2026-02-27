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
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
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

logger = logging.getLogger(__name__)


async def prewarm_llm():
    """
    Ollama-only prewarm. Will NOT attempt cloud.
    Calls Ollama client generate() directly to guarantee local-only call.
    """
    try:
        # Call Ollama directly to avoid router fallback to cloud.
        from agents import ollama_client

        # Use the configured OLLAMA_MODEL (ollama_client module uses OLLAMA_MODEL)
        model = getattr(ollama_client, "OLLAMA_MODEL", None)

        # Small safe prompt; short timeout
        await ollama_client.generate(
            prompt="Hello (warmup).",
            system="warmup",
            model=model,
            stream=False,
            request_id="prewarm",
            timeout=10.0
        )

        logger.info("Ollama prewarm completed successfully")

    except Exception as e:
        # Prewarm must not crash startup — log and continue
        logger.warning("Ollama prewarm failed (ignored)", exc_info=e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: configure structured JSON logging
    setup_logging()

    # Initialize shared LLM client
    await init_llm_client()

    # Start the background job worker loop
    app.state.job_worker = asyncio.create_task(job_queue.worker_loop())

    # Optional prewarm (non‑blocking)
    if os.getenv("PLANNER_PREWARM") == "1":
        async def background_prewarm():
            try:
                await prewarm_llm()
            except Exception:
                logger.exception("Background prewarm failed")
        asyncio.create_task(background_prewarm())

    yield

    # Shutdown: gracefully stop the worker
    try:
        await job_queue.stop_worker()
        await app.state.job_worker
    except asyncio.CancelledError:
        pass
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


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Generate a unique request ID and store it in the context."""
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


class AskRequest(BaseModel):
    origin: str | None = None
    destination: str | None = None
    date: str = Field(..., description="YYYY-MM-DD")
    user_query: str
    trip_type: str = "Business"

    @field_validator("date")
    @classmethod
    def date_must_be_future(cls, v):
        """Validate that the provided date is not in the past."""
        dt = datetime.strptime(v, "%Y-%m-%d")
        if dt.date() < datetime.now().date():
            raise ValueError("Date cannot be in the past")
        return v


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
    try:
        # Background job branch
        if async_job:
            from core.job_queue import enqueue_job
            payload = req.model_dump()
            job_id = await enqueue_job(payload)
            return Response(
                status_code=202,
                content=json.dumps({"job_id": job_id}),
                media_type="application/json"
            )

        GLOBAL_TIMEOUT = int(os.getenv("PLANNER_GLOBAL_TIMEOUT", "60"))

        if stream:
            # Streaming branch: call planner with stream=True and yield SSE events
            async def event_stream():
                agen_or_result = await planner_agent.plan_trip(
                    origin=req.origin,
                    destination=req.destination,
                    date=req.date,
                    user_query=req.user_query,
                    trip_type=req.trip_type,
                    stream=True
                )
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
                origin=req.origin,
                destination=req.destination,
                date=req.date,
                user_query=req.user_query,
                trip_type=req.trip_type
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


@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe."""
    health = await full_health_check()
    if health["status"] != "ok":
        return Response(
            content=json.dumps(health),
            status_code=503,
            media_type="application/json"
        )
    return health


@app.get("/health")
async def health():
    """Comprehensive health check for monitoring."""
    return await full_health_check()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)