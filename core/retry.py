# core/retry.py

import httpx
import asyncio
import random
import logging
import email.utils
import time
import inspect
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, Tuple, Type, Any, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """
    Configuration for retry behaviour.

    Attributes:
        retries: Maximum number of retry attempts.
        base_delay: Initial delay before first retry (seconds).
        max_backoff: Maximum delay between retries (seconds).
        jitter: If True, adds random full jitter to the delay (exponential backoff with
                random value between 0 and the capped exponential value). Recommended
                for production to prevent thundering herds.
        max_total_timeout: Optional global timeout for all retry attempts (seconds).
                           If the total elapsed time exceeds this value, no further retries are made.
        per_attempt_timeout: Optional timeout for each individual attempt (seconds).
                             If set, each attempt is bounded by this value. It also respects
                             the remaining global timeout if max_total_timeout is also set.
        retry_filter: Optional custom callable to decide if an exception is retryable.
                      If None, the default filter is used.
        retry_on: Additional exception types that should be retried.
        on_retry: Optional hook called before each retry sleep. Receives attempt (1-indexed),
                  delay (seconds), and the exception that triggered the retry.
    """
    retries: int = 3
    base_delay: float = 1.0
    max_backoff: float = 60.0
    jitter: bool = True
    max_total_timeout: Optional[float] = None
    per_attempt_timeout: Optional[float] = None
    retry_filter: Optional[Callable[[Exception], bool]] = None
    retry_on: Tuple[Type[Exception], ...] = (httpx.TimeoutException, httpx.ConnectError)
    on_retry: Optional[Callable[[int, float, Exception], None]] = None


def default_retry_filter(exc: Exception) -> bool:
    """
    Default decision logic for retryable errors.

    Retries on:
        - Network/connection errors (httpx.TimeoutException, ConnectError, NetworkError)
        - Any exception with a 'retry_after' attribute (e.g., custom rate limit errors)
        - Exceptions with a 'status_code' attribute that indicates 429 or 5xx (except 501)
        - HTTP 429 (Too Many Requests) and 5xx (except 501) from httpx.HTTPStatusError
    """
    # Network / connection problems
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
        return True

    # Custom rate limit exceptions (often have retry_after)
    if hasattr(exc, "retry_after"):
        return True

    # Exceptions that directly expose a status_code (e.g., custom API errors)
    if hasattr(exc, "status_code"):
        try:
            status = int(getattr(exc, "status_code"))
            if status == 429:
                return True
            if 500 <= status < 600 and status != 501:
                return True
            return False
        except Exception:
            pass

    # httpx HTTPStatusError
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        status = exc.response.status_code
        if status == 429:
            return True
        if 500 <= status < 600 and status != 501:
            return True
        return False

    return False


async def retry_async(
    func: Callable[[], Awaitable[Any]],
    *,
    config: Optional[RetryConfig] = None,
    request_id: Optional[str] = None,
    retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Any:
    """
    Execute an async function with retries.

    Important: This function should only be used on idempotent operations.
    Do not use on streaming generators, database writes, or other operations
    with side effects that cannot be safely repeated.

    Args:
        func: Async callable that takes no arguments and returns an awaitable.
        config: RetryConfig instance; if None, a default config is used.
        request_id: Optional identifier for logging correlation.
        retry_exceptions: Optional tuple of exception types that should always be retried,
                          taking highest priority over config.retry_on and retry_filter.

    Returns:
        Result of the function call.

    Raises:
        The last exception encountered if all retries fail or if a non‑retryable error occurs.
        If max_total_timeout is exceeded, the last exception is raised.
    """
    config = config or RetryConfig()
    retry_filter = config.retry_filter or default_retry_filter
    attempt = 0
    start_time = time.monotonic()

    while True:
        # Compute remaining global timeout (None or seconds remaining)
        remaining = None
        if config.max_total_timeout is not None:
            elapsed = time.monotonic() - start_time
            remaining = config.max_total_timeout - elapsed
            if remaining <= 0:
                # Global deadline reached — raise a timeout so it's handled uniformly below
                exc = httpx.TimeoutException("retry_async: global max_total_timeout exceeded")
                # Go directly to unified handling (no need to set should_retry here)
                # We'll treat it as a regular exception that should not be retried
                # because the while loop will break after handling.
                # But we'll let the unified handler decide based on retry_filter.
                # Since we're out of time, we should not retry.
                # So we set should_retry = False manually and skip to log/raise.
                # To keep code simple, we can raise immediately, but that would bypass logging.
                # Better: set exc and let the normal flow handle it.
                # However, we need to ensure that the loop exits. We'll set attempt to retries-1 to force exit.
                # Let's just raise immediately as it's a final condition.
                raise exc

        # Determine timeout for this attempt
        attempt_timeout = config.per_attempt_timeout

        # If global timeout exists, clamp attempt timeout to remaining time
        if remaining is not None:
            if attempt_timeout is None:
                attempt_timeout = remaining
            else:
                attempt_timeout = min(attempt_timeout, remaining)

        try:
            if attempt_timeout is not None:
                return await asyncio.wait_for(func(), timeout=attempt_timeout)
            else:
                # Backward compatible behavior
                return await func()

        except asyncio.TimeoutError as te:
            # Convert to httpx.TimeoutException so retry_filter recognizes it
            exc = httpx.TimeoutException("per-attempt timeout exceeded")
            exc.__cause__ = te

        except Exception as e:
            exc = e

        # ----- Unified retry handling -----
        # Never retry cancellation
        if isinstance(exc, asyncio.CancelledError):
            raise

        # Determine if this exception is retryable
        # Priority: explicit retry_exceptions (highest) -> config.retry_on -> retry_filter
        if retry_exceptions and isinstance(exc, retry_exceptions):
            should_retry = True
        elif config.retry_on and isinstance(exc, config.retry_on):
            should_retry = True
        else:
            should_retry = retry_filter(exc)

        # Recompute elapsed / timeout_exceeded (for logging and final decision)
        elapsed = time.monotonic() - start_time
        timeout_exceeded = (
            config.max_total_timeout is not None and
            elapsed > config.max_total_timeout
        )

        # If this was the last attempt, error is not retryable, or timeout exceeded, log and raise
        if not should_retry or attempt >= config.retries - 1 or timeout_exceeded:
            # Different log levels for HTTP errors based on status code
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                status = exc.response.status_code
                if 500 <= status < 600:
                    logger.error(
                        "Retry exhausted - server error",
                        extra={
                            "event": "retry_failed",
                            "status": status,
                            "attempt": attempt + 1,
                            "max_retries": config.retries,
                            "timeout_exceeded": timeout_exceeded,
                            "request_id": request_id,
                            "elapsed_seconds": round(elapsed, 3),
                        },
                    )
                else:
                    logger.warning(
                        "Retry exhausted - client error",
                        extra={
                            "event": "retry_failed",
                            "status": status,
                            "attempt": attempt + 1,
                            "max_retries": config.retries,
                            "timeout_exceeded": timeout_exceeded,
                            "request_id": request_id,
                            "elapsed_seconds": round(elapsed, 3),
                        },
                    )
            else:
                logger.error(
                    "Retry exhausted or non‑retryable error",
                    extra={
                        "event": "retry_failed",
                        "attempt": attempt + 1,
                        "max_retries": config.retries,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "timeout_exceeded": timeout_exceeded,
                        "request_id": request_id,
                        "elapsed_seconds": round(elapsed, 3),
                    },
                )
            raise exc

        # ----- Compute next delay -----
        # Try to honor Retry-After if present
        retry_after_seconds = None
        raw = None

        # 1) Exception has a 'retry_after' attribute (e.g., custom RateLimitError)
        if hasattr(exc, "retry_after") and getattr(exc, "retry_after") is not None:
            raw = getattr(exc, "retry_after")
        # 2) httpx HTTPStatusError with response headers
        elif isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            raw = exc.response.headers.get("Retry-After")

        if raw is not None:
            s = str(raw)
            # Numeric seconds?
            if s.isdigit():
                retry_after_seconds = int(s)
            else:
                # Try parsing as HTTP-date (RFC 1123)
                try:
                    parsed = email.utils.parsedate_to_datetime(s)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    # Calculate seconds from now
                    retry_after_seconds = max(0, int((parsed - datetime.now(timezone.utc)).total_seconds()))
                except Exception:
                    retry_after_seconds = None

        if retry_after_seconds is not None:
            delay = min(retry_after_seconds, config.max_backoff)
            retry_source = "Retry-After"
        else:
            # Exponential backoff (cap at max_backoff)
            exponential_cap = min(config.max_backoff, config.base_delay * (2 ** attempt))

            if config.jitter:
                # FULL jitter (recommended): pick uniformly from [0, exponential_cap].
                # This prevents synchronized retry storms much better than small additive jitter.
                delay = random.uniform(0, exponential_cap)
            else:
                # No jitter: deterministic exponential backoff (not recommended at scale)
                delay = exponential_cap
            retry_source = "exponential_full_jitter" if config.jitter else "exponential"

        # Optional hook for metrics (e.g., count rate‑limit events)
        if config.on_retry:
            try:
                config.on_retry(attempt + 1, delay, exc)
            except Exception:
                logger.debug("on_retry hook failed", exc_info=True)

        logger.warning(
            "Retrying after failure",
            extra={
                "event": "retry_attempt",
                "attempt": attempt + 1,
                "delay": round(delay, 3),
                "retry_source": retry_source,
                "error_type": type(exc).__name__,
                "request_id": request_id,
                "elapsed_seconds": round(elapsed, 3),
            },
        )

        # If global deadline would be exceeded by sleeping full delay, clamp sleep
        if config.max_total_timeout is not None:
            elapsed = time.monotonic() - start_time
            remaining = config.max_total_timeout - elapsed
            if remaining <= 0:
                raise httpx.TimeoutException("retry_async: global max_total_timeout exceeded before sleep")
            # sleep at most remaining
            sleep_for = min(delay, remaining)
        else:
            sleep_for = delay

        await asyncio.sleep(sleep_for)
        attempt += 1
        continue


def async_retry(config: Optional[RetryConfig] = None):
    """
    Decorator that wraps an async function with retry logic.

    Important: This decorator should only be applied to idempotent functions.
    Do not use on streaming generators, database writes, or other operations
    with side effects that cannot be safely repeated.

    Example:
        @async_retry(RetryConfig(retries=4, base_delay=1))
        async def fetch_data():
            ...
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        # Prevent accidentally decorating async generator functions (streams)
        if inspect.isasyncgenfunction(func):
            raise TypeError(
                "async_retry cannot be applied to async generator (streaming) functions. "
                "For streaming endpoints: retry only the stream-opening call, then iterate "
                "the returned async-iterator without retrying the generator itself."
            )

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If you need request_id, pass it via retry_async explicitly:
            # request_id = kwargs.pop("request_id", None)  # example
            return await retry_async(
                lambda: func(*args, **kwargs),
                config=config,
                # request_id=request_id,  # uncomment if you extract it
            )
        return wrapper
    return decorator


# --- Example usage patterns for ollama_client.py ---
#
# Non‑streaming call with retries:
#   async def _call_ollama(...):
#       async def do_request():
#           resp = await client.post(url, json=payload, timeout=timeout)
#           resp.raise_for_status()
#           return resp.json()
#
#       cfg = RetryConfig(
#           retries=5,
#           base_delay=1.0,
#           max_backoff=30.0,
#           max_total_timeout=60.0,   # total time for all retries
#           per_attempt_timeout=12.0,  # each attempt limited to 12s
#           on_retry=lambda a, d, e: metrics.inc("ollama.retry")
#       )
#       return await retry_async(do_request, config=cfg, request_id=request_id)
#
# Streaming call (retry only the opening of the stream, not the generator itself):
#   async def _start_stream(...):
#       async def open_stream():
#           resp = await client.post(stream_url, json=payload, timeout=timeout)
#           resp.raise_for_status()
#           return resp
#
#       response = await retry_async(open_stream, config=stream_cfg, request_id=request_id)
#       async for chunk in response.aiter_bytes():
#           yield chunk