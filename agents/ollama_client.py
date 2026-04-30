# NOTE:
# request_id is passed explicitly from the router layer rather than
# pulled directly via get_request_id() from ContextVar.
# This is intentional:
#   - Ensures explicit correlation propagation across service boundaries
#   - Keeps the LLM client decoupled from web framework context
#   - Improves testability and reuse outside FastAPI
# Logging configuration still injects ContextVar request_id automatically
# if set, but manual propagation guarantees consistency.
import asyncio
import logging
import json
import time
import weakref
from pathlib import Path
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
from core.env_config import get_env_bool, get_env_float, get_env_int, get_env_str
from core.ollama_context import RUNTIME_NUM_CTX_DEFAULT, resolve_runtime_num_ctx

# ----------------------------------------------------------------------
# Environment and defaults
# ----------------------------------------------------------------------
OLLAMA_BASE_URL = get_env_str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = get_env_str("OLLAMA_MODEL", "openhermes")
OLLAMA_TIMEOUT = get_env_float("OLLAMA_TIMEOUT", 30.0)   # default timeout in seconds
OLLAMA_BREAKER_FAILURE_THRESHOLD = max(1, get_env_int("OLLAMA_BREAKER_FAILURE_THRESHOLD", 5))
OLLAMA_BREAKER_RECOVERY_TIMEOUT = max(5, get_env_int("OLLAMA_BREAKER_RECOVERY_TIMEOUT", 60))
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_DOTENV_CANDIDATES = (_PROJECT_ROOT / ".env",)

def _resolve_ollama_model() -> str:
    model = (get_env_str("OLLAMA_MODEL", OLLAMA_MODEL) or OLLAMA_MODEL).strip()
    if not model:
        raise RuntimeError("OLLAMA_MODEL not configured")
    return model


def _resolve_ollama_base_url() -> str:
    base_url = (get_env_str("OLLAMA_BASE_URL", OLLAMA_BASE_URL) or OLLAMA_BASE_URL).strip()
    try:
        parsed = httpx.URL(base_url)
    except Exception as e:
        raise RuntimeError(f"Invalid OLLAMA_BASE_URL: {base_url}") from e
    host = (parsed.host or "").strip().lower()
    if host in {"0.0.0.0", "::"}:
        # 0.0.0.0 is valid for server bind, but invalid as a client destination.
        # Normalize to loopback so planner/router can reach local Ollama reliably.
        host = "127.0.0.1"

    normalized = f"{parsed.scheme}://{host}"
    if parsed.port:
        normalized += f":{parsed.port}"

    # Preserve an explicit non-root path prefix, but never return a trailing slash
    # to avoid constructing URLs like "...//api/chat".
    path = parsed.path
    if isinstance(path, bytes):
        path = path.decode(errors="ignore")
    path = (path or "").strip()
    if path and path != "/":
        normalized += path.rstrip("/")

    query = parsed.query
    if isinstance(query, bytes):
        query = query.decode(errors="ignore")
    if query:
        normalized += f"?{query}"

    return normalized


def _resolve_ollama_timeout() -> float:
    return max(1.0, get_env_float("OLLAMA_TIMEOUT", OLLAMA_TIMEOUT))


def _resolve_ollama_thinking_mode() -> str:
    """
    Controls whether to explicitly request model-side reasoning/thinking output.
    Supported values:
      - auto: do not set model think option (default)
      - disable: send think=False when supported by model/runtime
      - force: send think=True when supported by model/runtime
    """
    raw = (get_env_str("OLLAMA_THINKING_MODE", "auto") or "auto").strip().lower()
    if raw in {"disable", "force", "auto"}:
        return raw
    return "auto"


def _resolve_ollama_num_ctx() -> Optional[int]:
    """
    Optional explicit context window for Ollama requests.
    Returns None when unset/invalid so runtime defaults apply.
    """
    resolution = resolve_runtime_num_ctx(
        process_env=None,
        dotenv_paths=_RUNTIME_DOTENV_CANDIDATES,
        minimum_value=1,
        fallback_default=RUNTIME_NUM_CTX_DEFAULT,
    )
    value = resolution.get("effective_num_ctx")
    if isinstance(value, int) and value > 0:
        return value
    return None


def _estimate_tokens_from_chars(chars: int) -> int:
    if chars <= 0:
        return 0
    return max(1, (int(chars) + 3) // 4)


def _message_char_stats(messages: object) -> dict:
    system_chars = 0
    user_chars = 0
    assistant_chars = 0
    total_chars = 0
    if not isinstance(messages, list):
        return {
            "system_chars": 0,
            "user_chars": 0,
            "assistant_chars": 0,
            "total_chars": 0,
            "prompt_est_tokens": 0,
        }
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        chars = len(content)
        total_chars += chars
        role = str(message.get("role") or "").strip().lower()
        if role == "system":
            system_chars += chars
        elif role == "assistant":
            assistant_chars += chars
        else:
            user_chars += chars
    return {
        "system_chars": system_chars,
        "user_chars": user_chars,
        "assistant_chars": assistant_chars,
        "total_chars": total_chars,
        "prompt_est_tokens": _estimate_tokens_from_chars(total_chars),
    }


def get_runtime_inference_config() -> dict:
    """
    Return resolved local-runtime config that materially affects memory/latency.
    """
    num_ctx_resolution = resolve_runtime_num_ctx(
        process_env=None,
        dotenv_paths=_RUNTIME_DOTENV_CANDIDATES,
        minimum_value=1,
        fallback_default=RUNTIME_NUM_CTX_DEFAULT,
    )
    num_ctx = num_ctx_resolution.get("effective_num_ctx")
    num_ctx_value = int(num_ctx) if isinstance(num_ctx, int) and num_ctx > 0 else None
    return {
        "backend": "ollama",
        "model": _resolve_ollama_model(),
        "num_ctx": num_ctx_value,
        "num_ctx_source": str(num_ctx_resolution.get("source") or "unset"),
        "num_ctx_process_raw": str(num_ctx_resolution.get("process_raw") or ""),
        "thinking_mode": _resolve_ollama_thinking_mode(),
        "timeout_sec": _resolve_ollama_timeout(),
    }

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
    failure_threshold=OLLAMA_BREAKER_FAILURE_THRESHOLD,
    recovery_timeout=OLLAMA_BREAKER_RECOVERY_TIMEOUT,
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


def _extract_stream_token_or_heartbeat(data: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Normalize stream chunk payloads across Ollama/model variants.

    Returns:
      (token, kind)
      - token: visible text token, heartbeat empty string, or None
      - kind: "visible", "thinking_heartbeat", or None
    """
    message = data.get("message")
    token: Optional[str] = None

    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            token = content

    if token is None:
        response_text = data.get("response")
        if isinstance(response_text, str):
            token = response_text

    if isinstance(token, str) and token:
        return token, "visible"

    thinking = data.get("thinking")
    if not isinstance(thinking, str) and isinstance(message, dict):
        maybe_nested_thinking = message.get("thinking")
        if isinstance(maybe_nested_thinking, str):
            thinking = maybe_nested_thinking
    if isinstance(thinking, str) and thinking.strip():
        # Heartbeat: upstream is actively producing reasoning tokens even before
        # user-visible answer text appears.
        return "", "thinking_heartbeat"

    return None, None

# ----------------------------------------------------------------------
# Internal unprotected streaming generator (actual HTTP logic)
# ----------------------------------------------------------------------
async def _streaming_call_internal(
    payload: dict,
    request_id: Optional[str] = None,
    timeout: float = 28.0
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

    msg_stats = _message_char_stats(payload.get("messages"))
    request_start_monotonic = time.monotonic()
    request_start_epoch_ms = int(time.time() * 1000)

    # Counter to limit JSON decode error logging
    bad_json_count = 0
    max_bad_json_log = 5
    visible_chunk_count = 0
    thinking_heartbeat_count = 0
    response_chars = 0
    first_chunk_latency_sec: Optional[float] = None
    first_visible_chunk_latency_sec: Optional[float] = None
    first_chunk_epoch_ms: Optional[int] = None
    first_visible_chunk_epoch_ms: Optional[int] = None

    try:
        async with client.stream(
            "POST",
            f"{_resolve_ollama_base_url()}/api/chat",
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
                token, token_kind = _extract_stream_token_or_heartbeat(data)
                if token_kind == "visible":
                    if first_chunk_latency_sec is None:
                        first_chunk_latency_sec = time.monotonic() - request_start_monotonic
                        first_chunk_epoch_ms = int(time.time() * 1000)
                    if first_visible_chunk_latency_sec is None:
                        first_visible_chunk_latency_sec = time.monotonic() - request_start_monotonic
                        first_visible_chunk_epoch_ms = int(time.time() * 1000)
                    visible_chunk_count += 1
                    response_chars += len(token or "")
                    yield token  # type: ignore[arg-type]
                elif token_kind == "thinking_heartbeat":
                    if first_chunk_latency_sec is None:
                        first_chunk_latency_sec = time.monotonic() - request_start_monotonic
                        first_chunk_epoch_ms = int(time.time() * 1000)
                    thinking_heartbeat_count += 1
                    # Yield an empty heartbeat so router/planner can mark the stream as alive
                    # without exposing internal reasoning text.
                    yield ""
                if data.get("done"):
                    completion_epoch_ms = int(time.time() * 1000)
                    completion_latency_sec = time.monotonic() - request_start_monotonic
                    logger.debug(
                        "ollama_stream_chunk_profile",
                        extra={
                            "request_id": request_id,
                            "visible_chunks": visible_chunk_count,
                            "thinking_heartbeat_chunks": thinking_heartbeat_count,
                            "model": payload.get("model"),
                            "request_start_epoch_ms": request_start_epoch_ms,
                            "first_chunk_epoch_ms": first_chunk_epoch_ms,
                            "first_visible_chunk_epoch_ms": first_visible_chunk_epoch_ms,
                            "completion_epoch_ms": completion_epoch_ms,
                            "first_chunk_latency_sec": (
                                round(first_chunk_latency_sec, 3)
                                if first_chunk_latency_sec is not None
                                else None
                            ),
                            "first_visible_chunk_latency_sec": (
                                round(first_visible_chunk_latency_sec, 3)
                                if first_visible_chunk_latency_sec is not None
                                else None
                            ),
                            "completion_latency_sec": round(completion_latency_sec, 3),
                            "prompt_chars": msg_stats.get("total_chars"),
                            "prompt_est_tokens": msg_stats.get("prompt_est_tokens"),
                            "response_chars": response_chars,
                            "response_est_tokens": _estimate_tokens_from_chars(response_chars),
                            "done": data.get("done"),
                            "done_reason": data.get("done_reason"),
                            "total_duration": data.get("total_duration"),
                            "load_duration": data.get("load_duration"),
                            "prompt_eval_count": data.get("prompt_eval_count"),
                            "prompt_eval_duration": data.get("prompt_eval_duration"),
                            "eval_count": data.get("eval_count"),
                            "eval_duration": data.get("eval_duration"),
                        },
                    )
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
    except CircuitBreakerOpenError as exc:
        logger.warning(
            "Ollama circuit open, rejecting streaming request",
            extra={"request_id": request_id}
        )
        # Optional: increment a metric for circuit open events
        increment_llm_failure("ollama.circuit_open")
        raise OllamaError("Circuit breaker is open for ollama backend") from exc

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
    request_start_monotonic = time.monotonic()
    request_start_epoch_ms = int(time.time() * 1000)
    msg_stats = _message_char_stats(payload.get("messages"))
    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    # Granular timeouts
    timeout_obj = httpx.Timeout(connect=5.0, read=timeout, write=10.0, pool=5.0)

    try:
        response = await client.post(
            f"{_resolve_ollama_base_url()}/api/chat",
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

        # Raise dedicated exception for rate limiting (so retry can respect Retry-After)
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
            raise RateLimitError(retry_after=retry_after)

        # FAIL FAST: for 5xx errors (server / model runner), raise an OllamaError
        if 500 <= response.status_code < 600:
            logger.error(
                "Ollama server error",
                extra={
                    "request_id": request_id,
                    "status": response.status_code,
                    "model": payload.get("model"),
                    "body": body_text[:1000]  # truncated for logs
                }
            )
            # Create a clear error message including the server body to help upstream decide
            raise OllamaError(f"Ollama server error {response.status_code}: {body_text}")

        # client error (4xx except 404) – log but do not raise; raise_for_status will handle
        if 400 <= response.status_code < 500 and response.status_code != 404:
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

        completion_stats = {
            "request_id": request_id,
            "model": payload.get("model"),
            "request_start_epoch_ms": request_start_epoch_ms,
            "completion_epoch_ms": int(time.time() * 1000),
            "completion_latency_sec": round(time.monotonic() - request_start_monotonic, 3),
            "prompt_chars": msg_stats.get("total_chars"),
            "prompt_est_tokens": msg_stats.get("prompt_est_tokens"),
            "done": data.get("done"),
            "done_reason": data.get("done_reason"),
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
        }
        if any(
            completion_stats.get(k) is not None
            for k in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
        ):
            logger.debug("ollama_non_stream_completion_stats", extra=completion_stats)

        content: Optional[str] = None
        message_obj = data.get("message")
        if isinstance(message_obj, dict):
            maybe_content = message_obj.get("content")
            if isinstance(maybe_content, str):
                content = maybe_content
        if content is None:
            maybe_response = data.get("response")
            if isinstance(maybe_response, str):
                content = maybe_response
        if content is None:
            raise OllamaError("Malformed response: missing message.content/response")
        completion_stats["response_chars"] = len(content)
        completion_stats["response_est_tokens"] = _estimate_tokens_from_chars(len(content))
        logger.debug("ollama_non_stream_runtime_profile", extra=completion_stats)
        return content
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
    try:
        count_cancelled_as_failure = get_env_bool(
            "OLLAMA_BREAKER_COUNT_CANCELLED_AS_FAILURE",
            default=False,
        )
        return await ollama_breaker.call(
            lambda: _non_streaming_call_impl(payload, request_id, timeout),
            treat_cancelled_as_failure=count_cancelled_as_failure,
        )
    except CircuitBreakerOpenError as exc:
        logger.warning(
            "Ollama circuit open, rejecting non-streaming request",
            extra={"request_id": request_id, "model": payload.get("model")},
        )
        increment_llm_failure("ollama.circuit_open")
        raise OllamaError("Circuit breaker is open for ollama backend") from exc


async def prewarm(
    *,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    request_id: Optional[str] = None,
) -> str:
    """
    Best-effort warmup call for model load.
    Uses the retried non-streaming transport directly (without planner/router breaker coupling)
    so startup prewarm failures do not poison runtime circuit state.
    """
    warm_model = (model or _resolve_ollama_model()).strip()
    resolved_timeout = _resolve_ollama_timeout() if timeout is None else max(1.0, float(timeout))
    options = {"temperature": 0.0}
    num_ctx = _resolve_ollama_num_ctx()
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload = {
        "model": warm_model,
        "messages": [
            {"role": "system", "content": "Respond with OK only."},
            {"role": "user", "content": "ok"},
        ],
        "options": options,
        "stream": False,
    }
    try:
        # Prewarm should exercise actual transport timeout behavior (no duplicate outer timeout).
        return await _non_streaming_call_impl(payload, request_id=request_id, timeout=resolved_timeout)
    except Exception as e:
        raise OllamaError(str(e)) from e

# ----------------------------------------------------------------------
# Public generate function
# ----------------------------------------------------------------------
async def generate(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    stream: bool = False,
    request_id: Optional[str] = None,
    timeout: Optional[float] = None,   # use env var default
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
        timeout: Total timeout in seconds (for both streaming and non‑streaming). Defaults to env OLLAMA_TIMEOUT.

    Returns:
        Full response string if stream=False, else async generator.
    """
    # Input validation
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    resolved_model = (model or _resolve_ollama_model()).strip()
    resolved_timeout = _resolve_ollama_timeout() if timeout is None else max(1.0, float(timeout))

    if not resolved_model:
        raise ValueError("Model must be provided")
    if temperature < 0 or temperature > 2:
        raise ValueError("Temperature must be between 0 and 2")

    # Prepare payload for Ollama chat API
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    msg_stats = _message_char_stats(messages)
    request_start_epoch_ms = int(time.time() * 1000)

    payload = {
        "model": resolved_model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": stream
    }
    num_ctx = _resolve_ollama_num_ctx()
    if num_ctx is not None:
        payload["options"]["num_ctx"] = num_ctx
    thinking_mode = _resolve_ollama_thinking_mode()
    if thinking_mode == "disable":
        payload["options"]["think"] = False
    elif thinking_mode == "force":
        payload["options"]["think"] = True

    logger.debug(
        "Ollama request started",
        extra={
            "model": resolved_model,
            "stream": stream,
            "request_id": request_id,
            "temperature": temperature,
            "thinking_mode": thinking_mode,
            "num_ctx": num_ctx,
            "request_start_epoch_ms": request_start_epoch_ms,
            "prompt_chars": msg_stats.get("total_chars"),
            "prompt_est_tokens": msg_stats.get("prompt_est_tokens"),
        }
    )

    start_time = time.monotonic()

    if stream:
        # Streaming branch – return a generator that consumes the protected stream
        async def token_generator():
            success = False
            cancelled = False
            first_chunk_latency_sec: Optional[float] = None
            first_chunk_epoch_ms: Optional[int] = None
            response_chars = 0
            chunk_count = 0
            stream_iter = _streaming_call(payload, request_id, resolved_timeout)
            try:
                # Do not enforce an additional total stream timeout here.
                # Router/planner layers already enforce stream-init/per-chunk/overall budgets.
                # A duplicate total timeout at this layer can cause avoidable degradation
                # even while chunks are flowing successfully.
                async for token in stream_iter:
                    chunk_count += 1
                    if first_chunk_latency_sec is None:
                        first_chunk_latency_sec = time.monotonic() - start_time
                        first_chunk_epoch_ms = int(time.time() * 1000)
                    response_chars += len(token or "")
                    yield token
                success = True
            except asyncio.CancelledError:
                cancelled = True
                logger.debug("ollama_stream_cancelled", extra={"request_id": request_id})
                increment_llm_cancelled("ollama")
                raise
            except Exception:
                raise
            finally:
                latency = time.monotonic() - start_time
                logger.debug(
                    "Ollama streaming completed",
                    extra={
                        "request_id": request_id,
                        "model": resolved_model,
                        "latency_sec": round(latency, 3),
                        "request_start_epoch_ms": request_start_epoch_ms,
                        "first_chunk_epoch_ms": first_chunk_epoch_ms,
                        "completion_epoch_ms": int(time.time() * 1000),
                        "first_chunk_latency_sec": (
                            round(first_chunk_latency_sec, 3)
                            if first_chunk_latency_sec is not None
                            else None
                        ),
                        "chunks": chunk_count,
                        "response_chars": response_chars,
                        "response_est_tokens": _estimate_tokens_from_chars(response_chars),
                        "prompt_chars": msg_stats.get("total_chars"),
                        "prompt_est_tokens": msg_stats.get("prompt_est_tokens"),
                    }
                )
                if success:
                    increment_llm_success("ollama")
                elif not cancelled:
                    increment_llm_failure("ollama")

        return token_generator()
    else:
        # Non‑streaming branch – enforce overall SLA timeout and record failures on timeout
        try:
            result = await _non_streaming_call(payload, request_id, resolved_timeout)
        except asyncio.CancelledError:
            # Client cancelled – increment cancellation metric and re-raise
            logger.debug("ollama_request_cancelled", extra={"request_id": request_id})
            increment_llm_cancelled("ollama")
            raise
        except Exception:
            # Record non-cancellation failure at the public API boundary.
            increment_llm_failure("ollama")
            raise
        else:
            latency = time.monotonic() - start_time
            logger.debug(
                "Ollama request succeeded",
                extra={
                    "request_id": request_id,
                    "model": resolved_model,
                    "latency_sec": round(latency, 3),
                    "request_start_epoch_ms": request_start_epoch_ms,
                    "completion_epoch_ms": int(time.time() * 1000),
                    "prompt_chars": msg_stats.get("total_chars"),
                    "prompt_est_tokens": msg_stats.get("prompt_est_tokens"),
                    "response_chars": len(result or ""),
                    "response_est_tokens": _estimate_tokens_from_chars(len(result or "")),
                }
            )
            increment_llm_success("ollama")
            return result

# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------
async def health_check() -> str:
    logger.debug("Ollama health check invoked")
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
            f"{_resolve_ollama_base_url()}/api/tags",
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

        expected_model = _resolve_ollama_model()

        if expected_model in model_names:
            return "ok"

        # Also allow prefix match (openhermes vs openhermes:latest)
        prefix = expected_model.split(":")[0]
        for name in model_names:
            if name and name.startswith(prefix):
                return "ok"

        logger.warning(
            "Ollama model not found",
            extra={"expected": expected_model, "available": list(model_names)}
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
        timeout: Optional[float] = None,
    ):
        return await generate(
            prompt=prompt,
            system=system,
            model=model or _resolve_ollama_model(),
            stream=stream,
            request_id=request_id,
            timeout=timeout,
        )

    async def health_check(self) -> bool:
        return await health_check() == "ok"
