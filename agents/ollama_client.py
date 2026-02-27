# NOTE:
# request_id is passed explicitly from the router layer rather than
# pulled directly via get_request_id() from ContextVar.
# This is intentional:
#   - Ensures explicit correlation propagation across service boundaries
#   - Keeps the LLM client decoupled from web framework context
#   - Improves testability and reuse outside FastAPI
# Logging configuration still injects ContextVar request_id automatically
# if set, but manual propagation guarantees consistency.
import os
import asyncio
import logging
import json
import time
import weakref
from typing import Optional, Union, AsyncGenerator

import httpx

# Core shared modules (must provide the following):
#   - AsyncCircuitBreaker with:
#       * run_generator_protected(agen_factory) -> async generator that protects iteration
#         (must iterate and record success/failure, not just return the generator)
#       * record_failure() / record_success() public methods
#       * call() for non‑streaming calls that records success on normal completion
#         and failure on exception, **including when the wrapped task is cancelled**
#         (i.e., it must treat asyncio.CancelledError as a failure and record it).
#   - RetryConfig and async_retry decorator, with retry_async respecting 'retry_after'
#     on RateLimitError and **filtering httpx.HTTPStatusError to retry only 429 and 5xx**.
#   - LLMError base exception
#   - metrics functions increment_llm_success, increment_llm_failure, increment_llm_cancelled
from core.circuit_breaker import AsyncCircuitBreaker, CircuitBreakerOpenError
from core.retry import RetryConfig, async_retry
from core.exceptions import LLMError
from core.metrics import increment_llm_success, increment_llm_failure, increment_llm_cancelled

# ----------------------------------------------------------------------
# Environment and defaults
# ----------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "openhermes")

if not OLLAMA_MODEL:
    raise RuntimeError("OLLAMA_MODEL not configured")

# Validate base URL using httpx.URL (catches malformed URLs)
try:
    httpx.URL(OLLAMA_BASE_URL)
except Exception as e:
    raise RuntimeError(f"Invalid OLLAMA_BASE_URL: {OLLAMA_BASE_URL}") from e

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Shared HTTPX async client with connection pooling (per‑loop isolation using weakref)
# ----------------------------------------------------------------------
_client_map = weakref.WeakKeyDictionary()  # key: loop, value: AsyncClient
_client_lock_map = weakref.WeakKeyDictionary()  # key: loop, value: Lock

async def get_async_client() -> httpx.AsyncClient:
    """
    Return a shared async HTTPX client with connection pooling,
    isolated per event loop to avoid cross‑loop binding issues.
    """
    loop = asyncio.get_running_loop()

    if loop not in _client_map:
        # Ensure lock for this loop exists
        if loop not in _client_lock_map:
            _client_lock_map[loop] = asyncio.Lock()
        async with _client_lock_map[loop]:
            if loop not in _client_map:  # double‑check after acquiring lock
                _client_map[loop] = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=5.0,
                        read=120.0,
                        write=10.0,
                        pool=5.0
                    ),
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=20
                    )
                )
    return _client_map[loop]

async def close_client():
    """
    Close all HTTPX clients associated with any event loops.
    Should be called during application shutdown.
    """
    clients = list(_client_map.values())
    for client in clients:
        await client.aclose()
    _client_map.clear()
    _client_lock_map.clear()

# ----------------------------------------------------------------------
# Custom exceptions
# ----------------------------------------------------------------------
class OllamaError(LLMError):
    """Ollama-specific error."""
    pass

class RateLimitError(OllamaError):
    """Raised when Ollama returns 429 Too Many Requests."""
    def __init__(self, retry_after: Optional[str] = None):
        super().__init__("Rate limited (429)")
        self.retry_after = retry_after

# ----------------------------------------------------------------------
# Circuit breaker instance for Ollama
# ----------------------------------------------------------------------
ollama_breaker = AsyncCircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60
)

# ----------------------------------------------------------------------
# Retry callback for observability
# ----------------------------------------------------------------------
def _on_retry(attempt: int, delay: float, exc: Exception):
    logger.warning(
        "ollama_retry",
        extra={
            "attempt": attempt,
            "delay": delay,
            "error_type": type(exc).__name__,
        }
    )

# ----------------------------------------------------------------------
# Retry configuration for non‑streaming calls
# ----------------------------------------------------------------------
retry_cfg = RetryConfig(
    retries=3,
    base_delay=1.0,
    max_backoff=8.0,
    jitter=True,
    on_retry=_on_retry,
    # Retry only network issues and explicit rate limit exceptions.
    # HTTPStatusError is handled by the retry filter; we do NOT include it here.
    retry_on=(
        httpx.TimeoutException,
        httpx.ConnectError,
        RateLimitError,
    )
)

# ----------------------------------------------------------------------
# Internal unprotected streaming generator (actual HTTP logic)
# ----------------------------------------------------------------------
async def _streaming_call_internal(
    payload: dict,
    request_id: Optional[str] = None,
    timeout: float = 30.0
) -> AsyncGenerator[str, None]:
    """
    Core streaming logic – no circuit breaker, no retry.
    Yields tokens as they arrive.

    Note: Streaming requests are NOT retried to avoid duplicate partial responses.
    RateLimitError is propagated to the caller; the outer retry layer does not apply.
    """
    client = await get_async_client()
    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    # Granular timeouts
    timeout_obj = httpx.Timeout(connect=5.0, read=timeout, write=10.0, pool=5.0)

    # Counter to limit JSON decode error logging
    bad_json_count = 0
    max_bad_json_log = 5

    try:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            headers=headers,
            timeout=timeout_obj
        ) as response:
            # Special handling for 404 (model not found) before raise_for_status
            if response.status_code == 404:
                error_text = await response.aread()
                if "model" in error_text.decode().lower():
                    raise OllamaError("Model not found")
                raise OllamaError(f"HTTP 404: {error_text.decode()}")

            # Log rate limiting and server/client errors
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                logger.warning(
                    "Ollama rate limited",
                    extra={
                        "request_id": request_id,
                        "status": 429,
                        "retry_after": retry_after,
                        "model": payload.get("model")
                    }
                )
                # Raise dedicated exception so the caller knows the server asked to slow down.
                # Note: No retry here – streaming does not retry.
                raise RateLimitError(retry_after=retry_after)
            elif 500 <= response.status_code < 600:
                logger.error(
                    "Ollama server error",
                    extra={
                        "request_id": request_id,
                        "status": response.status_code,
                        "model": payload.get("model")
                    }
                )
            elif 400 <= response.status_code < 500 and response.status_code != 404:
                logger.warning(
                    "Ollama client error",
                    extra={
                        "request_id": request_id,
                        "status": response.status_code,
                        "model": payload.get("model")
                    }
                )

            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    bad_json_count += 1
                    if bad_json_count <= max_bad_json_log:
                        logger.debug(
                            "Invalid JSON chunk from Ollama",
                            extra={"line": line, "request_id": request_id}
                        )
                    continue
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
    except asyncio.CancelledError:
        # Do NOT increment metric here; let the outer `generate()` handle cancellation metrics.
        # Re-raise so that the breaker can record failure if it catches CancelledError.
        raise
    except httpx.HTTPStatusError as exc:
        # Log detailed HTTP error info, ensuring we don't double‑read the body
        status = exc.response.status_code
        # Use .content if available (synchronous), fall back to aread()
        body = getattr(exc.response, "content", None)
        if not body:
            try:
                body = await exc.response.aread()
            except Exception:
                body = b""
        # Truncate body to avoid huge log lines
        MAX_LOG_BODY = 2000
        body_text = body.decode(errors="replace")[:MAX_LOG_BODY]
        logger.error(
            "Ollama HTTP error during streaming",
            extra={
                "status": status,
                "body": body_text,
                "request_id": request_id,
                "model": payload.get("model")
            }
        )
        raise OllamaError(f"HTTP {status}: {body_text}") from exc
    except Exception as e:
        logger.exception(
            "Ollama streaming failed (non-HTTP)",
            extra={"request_id": request_id, "model": payload.get("model")}
        )
        raise OllamaError(f"Streaming failed: {str(e)}") from e

# ----------------------------------------------------------------------
# Protected streaming call – uses circuit breaker's generator helper
# ----------------------------------------------------------------------
async def _streaming_call(
    payload: dict,
    request_id: Optional[str] = None,
    timeout: float = 30.0
) -> AsyncGenerator[str, None]:
    """
    Protected version of streaming call — uses the breaker's run_generator_protected.
    Assumes AsyncCircuitBreaker has a method `run_generator_protected` that
    takes a generator factory and yields items while managing circuit state.
    """
    agen_factory = lambda: _streaming_call_internal(payload, request_id, timeout)
    try:
        async for token in ollama_breaker.run_generator_protected(agen_factory):
            yield token
    except CircuitBreakerOpenError:
        logger.warning(
            "Ollama circuit open, rejecting streaming request",
            extra={"request_id": request_id}
        )
        # Optional: increment a metric for circuit open events
        increment_llm_failure("ollama.circuit_open")
        raise

# ----------------------------------------------------------------------
# Protected non‑streaming call (with circuit breaker + retry)
# ----------------------------------------------------------------------
@async_retry(retry_cfg)
async def _non_streaming_call_impl(
    payload: dict,
    request_id: Optional[str] = None,
    timeout: float = 30.0
) -> str:
    """
    Internal non‑streaming call – actual HTTP request.
    Retries are applied via decorator. Circuit breaker is applied in the outer wrapper.

    The retry logic uses the RetryConfig above. HTTPStatusError is not included in retry_on,
    so the retry filter will decide: 429 and 5xx will be retried, other 4xx will not.
    """
    client = await get_async_client()
    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    # Granular timeouts
    timeout_obj = httpx.Timeout(connect=5.0, read=timeout, write=10.0, pool=5.0)

    try:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            headers=headers,
            timeout=timeout_obj
        )

        # Read body once to avoid multiple reads
        body_bytes = await response.aread()
        body_text = body_bytes.decode(errors="replace")

        # Special handling for 404 (model not found)
        if response.status_code == 404:
            if "model" in body_text.lower():
                raise OllamaError("Model not found")
            raise OllamaError(f"HTTP 404: {body_text}")

        # Log rate limiting and server/client errors
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            logger.warning(
                "Ollama rate limited",
                extra={
                    "request_id": request_id,
                    "status": 429,
                    "retry_after": retry_after,
                    "model": payload.get("model")
                }
            )
            # Raise dedicated exception so retry logic can respect Retry-After
            raise RateLimitError(retry_after=retry_after)
        elif 500 <= response.status_code < 600:
            logger.error(
                "Ollama server error",
                extra={
                    "request_id": request_id,
                    "status": response.status_code,
                    "model": payload.get("model")
                }
            )
        elif 400 <= response.status_code < 500 and response.status_code != 404:
            logger.warning(
                "Ollama client error",
                extra={
                    "request_id": request_id,
                    "status": response.status_code,
                    "model": payload.get("model")
                }
            )

        response.raise_for_status()

        # Parse JSON from the already-read body – handle malformed JSON gracefully
        try:
            data = json.loads(body_bytes)
        except json.JSONDecodeError as e:
            raise OllamaError(f"Invalid JSON from Ollama: {e}") from e

        if "message" not in data or "content" not in data["message"]:
            raise OllamaError("Malformed response: missing message.content")
        return data["message"]["content"]
    except httpx.HTTPStatusError as exc:
        # Use body_bytes from above if available, else read again carefully
        status = exc.response.status_code
        if 'body_bytes' in locals():
            body = body_bytes
        else:
            body = getattr(exc.response, "content", None)
            if not body:
                try:
                    body = await exc.response.aread()
                except Exception:
                    body = b""
        # Truncate body to avoid huge log lines
        MAX_LOG_BODY = 2000
        body_text = body.decode(errors="replace")[:MAX_LOG_BODY]
        logger.error(
            "Ollama HTTP error",
            extra={
                "status": status,
                "body": body_text,
                "request_id": request_id,
                "model": payload.get("model")
            }
        )
        raise  # re-raise for retry handling (HTTPStatusError is not in retry_on, but filter may act)
    except Exception as e:
        logger.exception(
            "Ollama request failed (non-HTTP)",
            extra={"request_id": request_id, "model": payload.get("model")}
        )
        raise OllamaError(str(e)) from e

async def _non_streaming_call(
    payload: dict,
    request_id: Optional[str] = None,
    timeout: float = 30.0
) -> str:
    """Wrapper that adds circuit breaker protection to the retried call.

    The breaker's call() method must record success on normal completion and failure on exception.
    This is critical for the circuit to close again after a recovery.
    IMPORTANT: The breaker's call() must also treat asyncio.CancelledError as a failure
    and record it; otherwise timeouts (which cancel the task) won't open the circuit.
    """
    return await ollama_breaker.call(lambda: _non_streaming_call_impl(payload, request_id, timeout))

# ----------------------------------------------------------------------
# Public generate function
# ----------------------------------------------------------------------
async def generate(
    prompt: str,
    system: Optional[str] = None,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.2,
    stream: bool = False,
    request_id: Optional[str] = None,
    timeout: float = 30.0
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Generate a completion from Ollama using the chat API.

    Args:
        prompt: User message (must not be empty).
        system: Optional system message.
        model: Model name (must not be empty; defaults to env var OLLAMA_MODEL).
        temperature: Sampling temperature (should be between 0 and 2).
        stream: If True, returns an async generator of tokens.
        request_id: Optional correlation ID (added to headers).
        timeout: Total timeout in seconds (for both streaming and non‑streaming).

    Returns:
        Full response string if stream=False, else async generator.
    """
    # Input validation
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    if not model:
        raise ValueError("Model must be provided")
    if temperature < 0 or temperature > 2:
        raise ValueError("Temperature must be between 0 and 2")

    # Prepare payload for Ollama chat API
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature
        },
        "stream": stream
    }

    logger.info(
        "Ollama request started",
        extra={
            "model": model,
            "stream": stream,
            "request_id": request_id,
            "temperature": temperature
        }
    )

    start_time = time.monotonic()

    if stream:
        # Streaming branch – return a generator that consumes the protected stream
        async def token_generator():
            success = False
            cancelled = False
            stream_iter = _streaming_call(payload, request_id, timeout)
            try:
                # Use asyncio.timeout for total timeout enforcement (Python 3.11+)
                async with asyncio.timeout(timeout):
                    async for token in stream_iter:
                        yield token
                success = True
            except asyncio.CancelledError:
                cancelled = True
                logger.info("ollama_stream_cancelled", extra={"request_id": request_id})
                increment_llm_cancelled("ollama")
                raise
            except TimeoutError:
                # Total timeout reached; ensure inner generator is closed & breaker records a failure
                try:
                    await stream_iter.aclose()
                except Exception:
                    logger.debug("failed to aclose stream_iter after timeout", exc_info=True)
                try:
                    await ollama_breaker.record_failure()
                except Exception:
                    logger.debug("failed to record breaker failure", exc_info=True)
                logger.error(
                    "Ollama streaming timed out",
                    extra={"request_id": request_id, "timeout": timeout}
                )
                # Convert to well-defined public exception
                raise OllamaError(f"Streaming timed out after {timeout}s")
            except Exception:
                raise
            finally:
                latency = time.monotonic() - start_time
                logger.info(
                    "Ollama streaming completed",
                    extra={"request_id": request_id, "latency_sec": latency}
                )
                if success:
                    increment_llm_success("ollama")
                elif not cancelled:
                    increment_llm_failure("ollama")

        return token_generator()
    else:
        # Non‑streaming branch – enforce overall SLA timeout and record failures on timeout
        try:
            result = await asyncio.wait_for(
                _non_streaming_call(payload, request_id, timeout),
                timeout=timeout
            )
        except asyncio.CancelledError:
            # Client cancelled – increment cancellation metric and re-raise
            logger.info("ollama_request_cancelled", extra={"request_id": request_id})
            increment_llm_cancelled("ollama")
            raise
        except asyncio.TimeoutError:
            logger.error(
                "Ollama request timed out",
                extra={"request_id": request_id, "timeout": timeout}
            )
            increment_llm_failure("ollama")
            raise OllamaError(f"Request timed out after {timeout}s") from None
        except Exception:
            # All other exceptions are already logged and counted by _non_streaming_call
            raise
        else:
            latency = time.monotonic() - start_time
            logger.info(
                "Ollama request succeeded",
                extra={"request_id": request_id, "latency_sec": latency}
            )
            increment_llm_success("ollama")
            return result

# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------
async def health_check() -> str:
    print("HEALTH CHECK FILE:", __file__)
    """
    Check if Ollama is reachable AND configured model exists.
    """
    try:
        client = await get_async_client()

        timeout_obj = httpx.Timeout(
            connect=2.0,
            read=5.0,
            write=2.0,
            pool=2.0
        )


        response = await client.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=timeout_obj
        )

        if response.status_code != 200:
            return "fail"

        data = response.json()
        models = data.get("models", [])

        model_names = {
            m.get("name") or m.get("model")
            for m in models
            if isinstance(m, dict)
        }

        if OLLAMA_MODEL in model_names:
            return "ok"

        # Also allow prefix match (openhermes vs openhermes:latest)
        prefix = OLLAMA_MODEL.split(":")[0]
        for name in model_names:
            if name and name.startswith(prefix):
                return "ok"

        logger.warning(
            "Ollama model not found",
            extra={"expected": OLLAMA_MODEL, "available": list(model_names)}
        )

        return "fail"

    except Exception as e:
        logger.warning("Ollama health check failed", extra={"error": str(e)})
        return "fail"


# ----------------------------------------------------------------------
# Graceful shutdown hook (call in your app's shutdown)
# ----------------------------------------------------------------------
async def shutdown():
    await close_client()

# ----------------------------------------------------------------------
# OOP Wrapper for Router Compatibility
# ----------------------------------------------------------------------

class OllamaClient:
    """
    Thin wrapper around module-level Ollama functions
    to provide class-based interface expected by LLMRouter.
    """

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
        request_id: Optional[str] = None,
    ):
        return await generate(
            prompt=prompt,
            system=system,
            model=model or OLLAMA_MODEL,
            stream=stream,
            request_id=request_id,
        )

    async def health_check(self) -> bool:
        return await health_check() == "ok"
