"""
core/async_llm_client.py

Centralized LLM client with connection pooling, retries, circuit breaker, timeouts,
streaming support, and automatic fallback between cloud and Ollama.

Singleton instance is obtained via init_llm_client().

FastAPI integration example:

    from fastapi import FastAPI
    from core.async_llm_client import init_llm_client, close_llm_client

    app = FastAPI()

    @app.on_event("startup")
    async def startup():
        await init_llm_client()

    @app.on_event("shutdown")
    async def shutdown():
        await close_llm_client()
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict, Any, Tuple

import httpx

# Optional tokenizer for accurate token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Centralized retry utility
from core.retry import retry_async, RetryConfig
# new — use the AsyncCircuitBreaker already defined in core/circuit_breaker.py
from core.circuit_breaker import AsyncCircuitBreaker as CircuitBreaker
# Metrics – Prometheus-style counters and histograms
from core.metrics import LLM_REQUESTS, LLM_LATENCY, increment as _increment_metric

logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and prevents a cloud call."""
    pass

class LLMClientError(Exception):
    """Base class for LLM client errors."""
    pass


# ----------------------------------------------------------------------
# Unified interface contract
# ----------------------------------------------------------------------
class LLMClientInterface(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        *,
        prompt: str,
        system: str,
        model: str,
        provider: str = "auto",
        timeout: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Generate a complete response (non‑streaming)."""
        pass

    @abstractmethod
    async def stream(
        self,
        *,
        prompt: str,
        system: str,
        model: str,
        provider: str = "auto",
        first_chunk_timeout: float = 10.0,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a response token by token."""
        pass


# ----------------------------------------------------------------------
# Singleton instance management (thread/async safe)
# ----------------------------------------------------------------------
_client_instance: Optional["AsyncLLMClient"] = None
_init_lock = asyncio.Lock()

async def init_llm_client() -> "AsyncLLMClient":
    """
    Async‑safe singleton initializer. Returns the global LLM client instance.
    Reads configuration from environment variables:
        CLOUD_PROVIDER      - "openai", "anthropic", etc. (default "openai")
        OPENAI_API_KEY      - required if CLOUD_PROVIDER == "openai"
        ANTHROPIC_API_KEY   - required if CLOUD_PROVIDER == "anthropic"
        fucking add gemini from the job v2 project to support multi key rotation so no pay 
        CLOUD_BASE_URL      - optional base URL override for cloud API
        OLLAMA_BASE_URL     - required (local Ollama endpoint)
    Raises ValueError if required variables are missing.
    """
    global _client_instance
    async with _init_lock:
        if _client_instance is None:
            # Read provider and keys
            cloud_provider = os.getenv("CLOUD_PROVIDER", "openai").lower()
            ollama_base_url = os.getenv("OLLAMA_BASE_URL")
            cloud_base_url = os.getenv("CLOUD_BASE_URL")  # optional
            openai_key = os.getenv("OPENAI_API_KEY")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")

            # Ollama is always required (local fallback)
            if not ollama_base_url:
                raise ValueError("OLLAMA_BASE_URL environment variable must be set")

            # Validate cloud credentials based on chosen provider
            if cloud_provider == "openai" and not openai_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set for openai provider")
            if cloud_provider == "anthropic" and not anthropic_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable must be set for anthropic provider")
            # For other providers (e.g., "none", "gemini"), no key required here

            _client_instance = AsyncLLMClient(
                cloud_provider=cloud_provider,
                cloud_base_url=cloud_base_url,
                ollama_base_url=ollama_base_url,
                cloud_api_key=openai_key,
                anthropic_api_key=anthropic_key,
            )
    return _client_instance

def get_llm_client() -> "AsyncLLMClient":
    """
    Synchronous accessor (use only after init_llm_client has been called,
    or in contexts where you are sure the instance already exists).
    For async startup, prefer init_llm_client().
    """
    if _client_instance is None:
        raise RuntimeError("LLM client not initialized. Call init_llm_client() first.")
    return _client_instance

async def close_llm_client() -> None:
    """Close and clear the singleton client (idempotent)."""
    global _client_instance
    if _client_instance is None:
        return
    try:
        await _client_instance.close()
    except Exception as e:
        logger.exception("Error while closing LLM client", exc_info=e)
    finally:
        _client_instance = None


class AsyncLLMClient(LLMClientInterface):
    """
    Unified async LLM client that handles both cloud (OpenAI-compatible) and Ollama endpoints.
    Manages connection pools, retries, timeouts, circuit breaker, and streaming.

    Args:
        cloud_provider: Which cloud provider to use ("openai", "anthropic", etc.).
        cloud_base_url: Base URL for cloud API (optional, uses provider default if None).
        ollama_base_url: Base URL for Ollama.
        cloud_api_key: API key for OpenAI (used if provider is "openai").
        anthropic_api_key: API key for Anthropic (used if provider is "anthropic").
        ... (other parameters as before)
    """

    # Maximum allowed prompt length in characters (very coarse pre‑filter)
    MAX_PROMPT_CHARS = 200_000

    # Default token limits per model (can be extended)
    DEFAULT_TOKEN_LIMITS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
    }
    DEFAULT_TOKEN_LIMIT = 8192

    def __init__(
        self,
        cloud_provider: str = "openai",
        cloud_base_url: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        cloud_timeout: float = 15.0,
        ollama_timeout: float = 30.0,
        max_connections: int = 20,
        max_keepalive_connections: int = 10,
        breaker_failure_threshold: int = 5,
        breaker_recovery_timeout: int = 120,
        retries: int = 3,
        max_backoff: float = 8.0,
        enable_jitter: bool = True,
        cloud_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize HTTP clients, circuit breaker, and retry settings.
        (Should normally be called only once via init_llm_client().)
        """
        self.cloud_provider = cloud_provider.lower()
        self.anthropic_api_key = anthropic_api_key

        # Build headers for cloud client – currently only OpenAI style; extend as needed.
        headers = {"Content-Type": "application/json"}
        if self.cloud_provider == "openai":
            if cloud_api_key:
                headers["Authorization"] = f"Bearer {cloud_api_key}"
            else:
                logger.warning("OPENAI_API_KEY not set – cloud calls may fail authentication")
        elif self.cloud_provider == "anthropic":
            if anthropic_api_key:
                headers["x-api-key"] = anthropic_api_key
                # Anthropic may also require a version header; add if needed.
                # headers["anthropic-version"] = "2023-06-01"
            else:
                logger.warning("ANTHROPIC_API_KEY not set – cloud calls may fail authentication")
        else:
            # For other providers, assume no auth or handled elsewhere.
            pass

        # Determine default cloud base URL if not provided
        if cloud_base_url is None:
            if self.cloud_provider == "openai":
                cloud_base_url = "https://api.openai.com/v1"
            elif self.cloud_provider == "anthropic":
                cloud_base_url = "https://api.anthropic.com/v1"  # example, adjust as needed
            else:
                cloud_base_url = ""  # will cause errors if used

        # Connection pools with granular timeouts
        self.cloud_client = httpx.AsyncClient(
            base_url=cloud_base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=5.0,
                read=cloud_timeout,
                write=5.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )
        self.ollama_client = httpx.AsyncClient(
            base_url=ollama_base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=ollama_timeout,
                write=5.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )

        # Circuit breaker for cloud (ollama is considered fallback and not protected)
        self.breaker = CircuitBreaker(
            failure_threshold=breaker_failure_threshold,
            recovery_timeout=breaker_recovery_timeout,
        )

        self.retries = retries
        self.cloud_timeout = cloud_timeout
        self.ollama_timeout = ollama_timeout
        self.max_backoff = max_backoff
        self.enable_jitter = enable_jitter

    # ----------------------------------------------------------------------
    # Async context manager for graceful shutdown
    # ----------------------------------------------------------------------
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close both HTTP clients (idempotent)."""
        if not self.cloud_client.is_closed:
            await self.cloud_client.aclose()
        if not self.ollama_client.is_closed:
            await self.ollama_client.aclose()

    # ----------------------------------------------------------------------
    # Health check for observability
    # ----------------------------------------------------------------------
    async def health(self) -> dict:
        """Return health status including circuit breaker state."""
        return {
            "circuit_breaker_state": self.breaker.state,
            "cloud_client": "connected" if not self.cloud_client.is_closed else "closed",
            "ollama_client": "connected" if not self.ollama_client.is_closed else "closed",
        }

    # ----------------------------------------------------------------------
    # Internal helpers: retryable exception classification, token checking
    # ----------------------------------------------------------------------

    def _is_retryable_exception(self, exc: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # CircuitOpenError is not retryable in the sense of retrying the same call;
        # it indicates the breaker is open, so we should fall back immediately.
        if isinstance(exc, CircuitOpenError):
            return False

        # Network / timeout errors
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
            return True

        # HTTP status errors
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            # Retry on 429 (rate limit), 5xx (server errors) except 501 (not implemented)
            if status == 429 or (500 <= status < 600 and status != 501):
                return True
            # Do not retry 4xx except 429
            return False

        # asyncio.TimeoutError is retryable (our own first-chunk timeout)
        if isinstance(exc, asyncio.TimeoutError):
            return True

        # Other exceptions – not retryable by default
        return False

    def _approx_tokens_from_text(self, text: str, model: Optional[str] = None) -> int:
        """
        Approximate token count for a given text.
        Uses tiktoken if available, otherwise a simple heuristic (characters/4).
        """
        if tiktoken is not None:
            try:
                # Try to get encoding for the specified model
                if model:
                    enc = tiktoken.encoding_for_model(model)
                else:
                    enc = tiktoken.get_encoding("cl100k_base")  # default for GPT-4
                return len(enc.encode(text))
            except Exception as e:
                logger.debug("tiktoken failed, falling back to heuristic: %s", e)
        # Heuristic: ~4 characters per token (typical for English)
        return max(1, len(text) // 4)

    async def _check_prompt_size(self, system: str, prompt: str, model: str):
        """
        Validate that the combined system + prompt does not exceed the model's token limit.
        Raises ValueError if too large.
        """
        total_text = system + "\n" + prompt
        tokens = self._approx_tokens_from_text(total_text, model)
        limit = self.DEFAULT_TOKEN_LIMITS.get(model, self.DEFAULT_TOKEN_LIMIT)
        if tokens > limit:
            raise ValueError(
                f"Prompt too large: {tokens} tokens (limit {limit} for model {model})"
            )

    # ----------------------------------------------------------------------
    # Private transport methods (pure HTTP, no retry/breaker/timeout)
    # These accept an optional per‑request timeout and are responsible for
    # the actual HTTP calls.
    # ----------------------------------------------------------------------

    async def _cloud_generate(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> str:
        """
        Non-streaming call to cloud endpoint.
        Assumes OpenAI-compatible /v1/chat/completions.
        Returns the full response text.
        """
        effective_timeout = timeout or self.cloud_timeout
        timeout_obj = httpx.Timeout(connect=5.0, read=effective_timeout, write=5.0, pool=5.0)
        response = await self.cloud_client.post(
            "/v1/chat/completions", json=payload, timeout=timeout_obj
        )
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMClientError(f"Unexpected cloud response structure: {e}") from e

    async def _ollama_generate(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> str:
        """
        Non-streaming call to Ollama /api/chat or /api/generate.
        Uses /api/chat with messages format (compatible with OpenAI-style payload).
        Returns the full response text.
        """
        effective_timeout = timeout or self.ollama_timeout
        timeout_obj = httpx.Timeout(connect=5.0, read=effective_timeout, write=5.0, pool=5.0)
        response = await self.ollama_client.post("/api/chat", json=payload, timeout=timeout_obj)
        response.raise_for_status()
        data = response.json()
        try:
            return data["message"]["content"]
        except KeyError as e:
            raise LLMClientError(f"Unexpected ollama response structure: {e}") from e

    async def _cloud_stream(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> AsyncIterator[str]:
        """
        Streaming call to cloud endpoint.
        Yields content tokens as they arrive.
        """
        effective_timeout = timeout or self.cloud_timeout
        timeout_obj = httpx.Timeout(connect=5.0, read=effective_timeout, write=5.0, pool=5.0)
        async with self.cloud_client.stream(
            "POST", "/v1/chat/completions", json=payload, timeout=timeout_obj
        ) as response:
            response.raise_for_status()
            # OpenAI SSE format: each line starts with "data: "
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # strip "data: "
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.warning(
                            "Malformed SSE chunk from cloud",
                            extra={"event": "malformed_chunk", "error": str(e), "data": data}
                        )
                        continue

    async def _ollama_stream(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> AsyncIterator[str]:
        """
        Streaming call to Ollama endpoint.
        Yields content tokens as they arrive (Ollama returns JSON lines).
        """
        effective_timeout = timeout or self.ollama_timeout
        timeout_obj = httpx.Timeout(connect=5.0, read=effective_timeout, write=5.0, pool=5.0)
        async with self.ollama_client.stream(
            "POST", "/api/chat", json=payload, timeout=timeout_obj
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(
                            "Malformed JSON line from ollama",
                            extra={"event": "malformed_chunk", "error": str(e), "line": line}
                        )
                        continue

    # ----------------------------------------------------------------------
    # Streaming with timeout guard (first chunk)
    # ----------------------------------------------------------------------
    async def _stream_with_first_chunk_timeout(
        self,
        stream_gen: AsyncIterator[str],
        timeout: float,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Wrap an async generator and enforce a timeout for the first yielded chunk.
        Subsequent chunks rely on the HTTP client's per‑read timeouts.
        Closes the underlying generator on timeout or empty stream.
        """
        try:
            # Wait for first chunk with timeout
            first_chunk = await asyncio.wait_for(anext(stream_gen), timeout=timeout)
            yield first_chunk
        except asyncio.TimeoutError:
            # Close underlying generator to free resources
            try:
                await stream_gen.aclose()
            except Exception:
                pass
            logger.error(
                "Stream first chunk timeout",
                extra={"event": "stream_timeout", "timeout": timeout, "request_id": request_id}
            )
            raise asyncio.TimeoutError(f"First streaming chunk timed out after {timeout}s")
        except StopAsyncIteration:
            # Empty stream – close and exit
            try:
                await stream_gen.aclose()
            except Exception:
                pass
            return

        # Yield the rest normally
        async for chunk in stream_gen:
            yield chunk

    # ----------------------------------------------------------------------
    # Public API (LLMClientInterface implementation)
    # ----------------------------------------------------------------------

    async def generate(
        self,
        *,
        prompt: str,
        system: str,
        model: str,
        provider: str = "auto",
        timeout: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: User message.
            system: System prompt.
            model: Model name (e.g., "gpt-4", "llama2").
            provider: "cloud", "ollama", or "auto" (try cloud, fallback to ollama).
            timeout: Override default timeout for this call.
            request_id: Correlation ID for logging.

        Returns:
            Generated text.
        """
        # Coarse character‑based guard (cheap)
        if len(prompt) > self.MAX_PROMPT_CHARS:
            raise ValueError(f"Prompt exceeds maximum length of {self.MAX_PROMPT_CHARS} characters")

        # Token‑aware guard
        await self._check_prompt_size(system, prompt, model)

        # Prepare payload (OpenAI-compatible messages)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,  # ensure non-streaming
        }

        start = time.monotonic()
        success = False
        actual_provider = provider  # will be updated based on actual usage
        try:
            if provider == "cloud":
                actual_provider = "cloud"
                result = await self._call_cloud_with_protections(payload, timeout, request_id)
            elif provider == "ollama":
                actual_provider = "ollama"
                result = await self._call_ollama_with_retry(payload, timeout, request_id)
            else:  # "auto"
                result, actual_provider = await self._auto_fallback_with_provider(payload, timeout, request_id)
            success = True
            return result
        finally:
            latency = time.monotonic() - start
            logger.info(
                "LLM generate completed",
                extra={
                    "event": "llm_generate",
                    "provider": provider,          # input provider for context
                    "actual_provider": actual_provider,
                    "model": model,
                    "latency_sec": latency,
                    "breaker_state": self.breaker.state,
                    "success": success,
                    "request_id": request_id,
                },
            )
            # Prometheus metrics with actual provider
            LLM_REQUESTS.labels(provider=actual_provider, status="success" if success else "error").inc()
            LLM_LATENCY.labels(provider=actual_provider).observe(latency)

    async def stream(
        self,
        *,
        prompt: str,
        system: str,
        model: str,
        provider: str = "auto",
        first_chunk_timeout: float = 10.0,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response token by token.

        Args:
            prompt: User message.
            system: System prompt.
            model: Model name.
            provider: "cloud", "ollama", or "auto" (try cloud, fallback to ollama).
            first_chunk_timeout: Maximum seconds to wait for the first chunk.
            request_id: Correlation ID for logging.

        Yields:
            Tokens as they arrive.
        """
        # Coarse character‑based guard
        if len(prompt) > self.MAX_PROMPT_CHARS:
            raise ValueError(f"Prompt exceeds maximum length of {self.MAX_PROMPT_CHARS} characters")

        # Token‑aware guard
        await self._check_prompt_size(system, prompt, model)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,  # enable streaming
        }

        start = time.monotonic()
        success = False
        actual_provider = provider  # will be updated
        try:
            if provider == "cloud":
                actual_provider = "cloud"
                # Check circuit breaker before attempting
                if await self.breaker.is_open():
                    raise CircuitOpenError("Circuit breaker is open, cannot call cloud")

                # Single attempt – no retry wrapper
                stream_gen = self._cloud_stream(payload, timeout=self.cloud_timeout)
                wrapped = self._stream_with_first_chunk_timeout(
                    stream_gen, first_chunk_timeout, request_id
                )
                try:
                    first_token = await anext(wrapped)
                except Exception as e:
                    # Close wrapped generator on failure to avoid leaks
                    try:
                        await wrapped.aclose()
                    except Exception:
                        pass
                    # Record cloud failure so breaker learns
                    try:
                        await self.breaker.record_failure()
                    except Exception:
                        logger.exception("Error recording breaker failure during stream fallback")
                    # Only fallback if the exception is retryable
                    if self._is_retryable_exception(e):
                        logger.warning(
                            f"Cloud streaming failed with retryable error, falling back to Ollama: {e}",
                            extra={"event": "stream_fallback", "error": str(e), "request_id": request_id}
                        )
                        # Fallback to Ollama – actual provider becomes ollama
                        actual_provider = "ollama"
                        async for token in self._stream_ollama_fallback(
                            payload, first_chunk_timeout, request_id
                        ):
                            yield token
                        success = True
                        return
                    else:
                        # Non‑retryable – already recorded failure, now raise
                        raise

                # First token received – record success and yield
                await self.breaker.record_success()
                try:
                    yield first_token
                    async for token in wrapped:
                        yield token
                except asyncio.CancelledError:
                    logger.info(
                        "llm_stream_cancelled",
                        extra={"event": "stream_cancelled", "provider": "cloud", "request_id": request_id}
                    )
                    _increment_metric("llm.stream_cancelled", tags={"provider": "cloud"})
                    await self.breaker.record_failure()
                    raise
                except Exception:
                    await self.breaker.record_failure()
                    raise
                finally:
                    try:
                        await wrapped.aclose()
                    except Exception:
                        pass

                success = True
                return

            if provider == "ollama":
                actual_provider = "ollama"
                # No circuit breaker for Ollama
                stream_gen = self._ollama_stream(payload, timeout=self.ollama_timeout)
                wrapped = self._stream_with_first_chunk_timeout(
                    stream_gen, first_chunk_timeout, request_id
                )
                try:
                    first_token = await anext(wrapped)
                    yield first_token
                    async for token in wrapped:
                        yield token
                except asyncio.CancelledError:
                    logger.info(
                        "llm_stream_cancelled",
                        extra={"event": "stream_cancelled", "provider": "ollama", "request_id": request_id}
                    )
                    _increment_metric("llm.stream_cancelled", tags={"provider": "ollama"})
                    raise
                finally:
                    try:
                        await wrapped.aclose()
                    except Exception:
                        pass
                success = True
                return

            # Auto: try cloud, fallback to ollama on retryable failures
            actual_provider = "cloud"  # assume cloud first
            try:
                if await self.breaker.is_open():
                    raise CircuitOpenError("Circuit breaker open, skipping cloud")

                stream_gen = self._cloud_stream(payload, timeout=self.cloud_timeout)
                wrapped = self._stream_with_first_chunk_timeout(
                    stream_gen, first_chunk_timeout, request_id
                )
                try:
                    first_token = await anext(wrapped)
                except Exception as e:
                    try:
                        await wrapped.aclose()
                    except Exception:
                        pass
                    # Record cloud failure
                    try:
                        await self.breaker.record_failure()
                    except Exception:
                        logger.exception("Error recording breaker failure during stream fallback")
                    if self._is_retryable_exception(e):
                        logger.warning(
                            f"Cloud streaming failed with retryable error, falling back to Ollama: {e}",
                            extra={"event": "stream_fallback", "error": str(e), "request_id": request_id}
                        )
                        actual_provider = "ollama"
                        async for token in self._stream_ollama_fallback(
                            payload, first_chunk_timeout, request_id
                        ):
                            yield token
                        success = True
                        return
                    else:
                        raise

                # Success path with cloud
                await self.breaker.record_success()
                try:
                    yield first_token
                    async for token in wrapped:
                        yield token
                except asyncio.CancelledError:
                    logger.info(
                        "llm_stream_cancelled",
                        extra={"event": "stream_cancelled", "provider": "cloud", "request_id": request_id}
                    )
                    _increment_metric("llm.stream_cancelled", tags={"provider": "cloud"})
                    await self.breaker.record_failure()
                    raise
                except Exception:
                    await self.breaker.record_failure()
                    raise
                finally:
                    try:
                        await wrapped.aclose()
                    except Exception:
                        pass
                success = True
                return

            except CircuitOpenError:
                # Breaker open – go directly to Ollama fallback
                logger.info("Breaker open – using Ollama fallback", extra={"request_id": request_id})
                actual_provider = "ollama"
                async for token in self._stream_ollama_fallback(
                    payload, first_chunk_timeout, request_id
                ):
                    yield token
                success = True
                return

        finally:
            latency = time.monotonic() - start
            logger.info(
                "LLM stream completed",
                extra={
                    "event": "llm_stream",
                    "provider": provider,          # input provider for context
                    "actual_provider": actual_provider,
                    "model": model,
                    "latency_sec": latency,
                    "breaker_state": self.breaker.state,
                    "success": success,
                    "request_id": request_id,
                },
            )
            # Prometheus metrics with actual provider
            LLM_REQUESTS.labels(provider=actual_provider, status="success" if success else "error").inc()
            LLM_LATENCY.labels(provider=actual_provider).observe(latency)

    async def _stream_ollama_fallback(
        self,
        payload: Dict[str, Any],
        first_chunk_timeout: float,
        request_id: Optional[str],
    ) -> AsyncIterator[str]:
        """
        Fallback streaming using Ollama (no circuit breaker).
        """
        stream_gen = self._ollama_stream(payload, timeout=self.ollama_timeout)
        wrapped = self._stream_with_first_chunk_timeout(
            stream_gen, first_chunk_timeout, request_id
        )
        try:
            first_token = await anext(wrapped)
            yield first_token
            async for token in wrapped:
                yield token
        except asyncio.CancelledError:
            logger.info(
                "llm_stream_cancelled",
                extra={"event": "stream_cancelled", "provider": "ollama", "request_id": request_id}
            )
            _increment_metric("llm.stream_cancelled", tags={"provider": "ollama"})
            raise
        finally:
            try:
                await wrapped.aclose()
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # Internal orchestration methods (with protections)
    # ----------------------------------------------------------------------

    async def _call_cloud_with_protections(
        self, payload: Dict[str, Any], timeout: Optional[float], request_id: Optional[str]
    ) -> str:
        """Call cloud with circuit breaker, retry, and timeout."""
        effective_timeout = timeout or self.cloud_timeout

        cfg = RetryConfig(
            retries=self.retries,
            base_delay=1.0,
            max_backoff=self.max_backoff,
            jitter=self.enable_jitter,
            retry_filter=self._is_retryable_exception,
            on_retry=lambda attempt, delay, exc: _increment_metric(
                "llm.retry", tags={"provider": "cloud"}
            ),
        )

        async def protected_call():
            # The actual call inside breaker and retry – no extra _with_timeout,
            # because _cloud_generate already enforces its own timeout.
            return await retry_async(
                lambda: self._cloud_generate(payload, timeout=effective_timeout),
                config=cfg,
                request_id=request_id,
            )

        # Let breaker handle the call; it will raise CircuitOpenError if open
        return await self.breaker.call(protected_call)

    async def _call_ollama_with_retry(
        self, payload: Dict[str, Any], timeout: Optional[float], request_id: Optional[str]
    ) -> str:
        """Call Ollama with retry and timeout (no breaker)."""
        effective_timeout = timeout or self.ollama_timeout

        cfg = RetryConfig(
            retries=self.retries,
            base_delay=1.0,
            max_backoff=self.max_backoff,
            jitter=self.enable_jitter,
            retry_filter=self._is_retryable_exception,
            on_retry=lambda attempt, delay, exc: _increment_metric(
                "llm.retry", tags={"provider": "ollama"}
            ),
        )

        return await retry_async(
            lambda: self._ollama_generate(payload, timeout=effective_timeout),
            config=cfg,
            request_id=request_id,
        )

    async def _auto_fallback(
        self, payload: Dict[str, Any], timeout: Optional[float], request_id: Optional[str]
    ) -> str:
        """Legacy method for backward compatibility; use _auto_fallback_with_provider."""
        result, _ = await self._auto_fallback_with_provider(payload, timeout, request_id)
        return result

    async def _auto_fallback_with_provider(
        self, payload: Dict[str, Any], timeout: Optional[float], request_id: Optional[str]
    ) -> Tuple[str, str]:
        """Try cloud, fall back to Ollama on any exception. Returns (result, actual_provider)."""
        try:
            if await self.breaker.is_open():
                # Skip cloud if breaker is open
                raise CircuitOpenError("Circuit breaker open")
            result = await self._call_cloud_with_protections(payload, timeout, request_id)
            return result, "cloud"
        except Exception as e:
            # Only fallback if the exception is retryable and not CircuitOpenError
            if isinstance(e, CircuitOpenError) or not self._is_retryable_exception(e):
                logger.error(
                    f"Cloud generation failed with non-retryable error, not falling back: {e}",
                    extra={"event": "generate_no_fallback", "error": str(e), "request_id": request_id}
                )
                raise
            logger.warning(
                f"Cloud generation failed with retryable error, falling back to Ollama: {e}",
                extra={"event": "generate_fallback", "error": str(e), "request_id": request_id}
            )
            result = await self._call_ollama_with_retry(payload, timeout, request_id)
            return result, "ollama"