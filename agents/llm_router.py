import logging
import os
import time
import uuid
import asyncio
import inspect
from typing import Optional, Union, AsyncGenerator, Dict, Any, List

# Import async clients (assumed to be fully async, with health checks and streaming)
from agents.ollama_client import OLLAMA_TIMEOUT, OllamaClient, OllamaError
from agents.cloud_llm import CloudLLMClient, CloudLLMError
# Import dynamic mode/priority from orchestrator
from core.llm_mode import (
    get_effective_cloud_provider,
    get_llm_mode_and_priority,
    LLM_MODE_CLOUD_FIRST,
    LLM_MODE_CLOUD_ONLY,
    LLM_MODE_OLLAMA_FIRST,
    LLM_MODE_OLLAMA_ONLY,
)
# Import metrics system (if available; otherwise replace with no-op)
try:
    import core.metrics as metrics
except ImportError:
    # Fallback no-op metrics
    class NoOpMetrics:
        def increment(self, name, **tags):
            pass
    metrics = NoOpMetrics()


# --- Wrapper to guarantee .provider on async generators ---
class ProviderAsyncGen:
    """
    Wrapper that exposes `.provider` while delegating async iteration
    to the inner async generator `agen`. Also forwards aclose().
    """
    def __init__(self, agen, provider: str):
        self._agen = agen
        self.provider = provider

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._agen.__anext__()

    async def aclose(self):
        close = getattr(self._agen, "aclose", None)
        if close:
            return await close()
        return None
# ------------------------------------------------------------


# Environment configuration
LOCAL_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", str(OLLAMA_TIMEOUT)))  # seconds
CLOUD_TIMEOUT = float(os.getenv("CLOUD_LLM_TIMEOUT", "60"))  # seconds
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT", "90"))  # seconds; routing stage budget

# Module logger
logger = logging.getLogger(__name__)

class AllBackendsFailed(Exception):
    """Raised when no LLM backend can successfully generate a response."""
    def __init__(
        self,
        message: str = "All LLM backends failed",
        *,
        mode: Optional[str] = None,
        effective_mode: Optional[str] = None,
        cloud_provider: Optional[str] = None,
        failures: Optional[List[Dict[str, str]]] = None,
    ):
        super().__init__(message)
        self.mode = mode
        self.effective_mode = effective_mode
        self.cloud_provider = cloud_provider
        self.failures = failures or []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "effective_mode": self.effective_mode,
            "cloud_provider": self.cloud_provider,
            "failures": self.failures,
        }

class LLMRouter:
    """
    Async LLM router with priority-based routing, timeouts,
    and structured logging. Delegates retries and circuit breaking to clients.

    Note: ROUTER_TIMEOUT applies only to the routing phase (selection of backend and initial response).
          For streaming responses, the streaming phase is governed by per‑chunk timeouts
          (see _stream_with_timeout). Once a stream starts successfully (first chunk received),
          no further fallback occurs.
    """

    def __init__(self, ollama_client: OllamaClient, cloud_client: Optional[CloudLLMClient] = None):
        self.ollama = ollama_client
        self.cloud = cloud_client

    @staticmethod
    def derive_effective_mode(requested_mode: str, cloud_available: bool, ollama_available: bool) -> str:
        mode = (requested_mode or LLM_MODE_OLLAMA_FIRST).lower()
        if cloud_available and not ollama_available:
            return LLM_MODE_CLOUD_ONLY
        if ollama_available and not cloud_available:
            return LLM_MODE_OLLAMA_ONLY
        return mode

    async def _is_cloud_available(self) -> bool:
        if self.cloud is None:
            return False
        has_usable_provider = getattr(self.cloud, "has_usable_provider", None)
        if callable(has_usable_provider):
            try:
                probe = has_usable_provider()
                if inspect.isawaitable(probe):
                    return bool(await asyncio.wait_for(probe, timeout=1.0))
                return bool(probe)
            except Exception:
                logger.debug("Cloud provider usability probe failed", exc_info=True)
                return False
        return True

    async def _is_ollama_available(self) -> bool:
        if self.ollama is None:
            return False
        health_check = getattr(self.ollama, "health_check", None)
        if callable(health_check):
            try:
                probe = health_check()
                if inspect.isawaitable(probe):
                    return bool(await asyncio.wait_for(probe, timeout=1.0))
                return bool(probe)
            except Exception:
                logger.debug("Ollama availability probe failed", exc_info=True)
                return False
        return True

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
        request_id: Optional[str] = None,
        return_metadata: bool = False
    ) -> Union[str, AsyncGenerator[str, None], Dict[str, Any]]:
        """
        Route the request based on priority, health, and availability.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            model: Optional model name (overrides default per backend).
            stream: If True, return an async generator.
            request_id: Unique ID for tracing.
            return_metadata: If True, return a dict with response and metadata for non-streaming.

        Returns:
            Depending on stream and return_metadata.

        Raises:
            AllBackendsFailed: if no backend could generate a response.
            TimeoutError: if the global router timeout is exceeded.
            RuntimeError: if mode requires a backend that is not configured.

        Note: For streaming, once the generator starts yielding, no fallback is possible.
              Per‑chunk timeouts are enforced using the backend-specific timeout value.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Wrap the entire operation with a global timeout
        try:
            return await asyncio.wait_for(
                self._route(prompt, system, model, stream, request_id, return_metadata),
                timeout=ROUTER_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error("Router global timeout", extra={"request_id": request_id, "timeout": ROUTER_TIMEOUT})
            metrics.increment("llm.router.failure", tags={"reason": "global_timeout"})
            raise TimeoutError(f"Router timeout after {ROUTER_TIMEOUT}s")
        except AllBackendsFailed:
            metrics.increment("llm.router.failure", tags={"reason": "all_backends_failed"})
            raise
        except Exception:
            metrics.increment("llm.router.failure", tags={"reason": "unexpected"})
            raise

    async def _route(
        self,
        prompt: str,
        system: Optional[str],
        model: Optional[str],
        stream: bool,
        request_id: str,
        return_metadata: bool
    ) -> Union[str, AsyncGenerator[str, None], Dict[str, Any]]:
        """Internal routing logic."""
        # Get mode and priority once at the beginning
        mode, priority = await get_llm_mode_and_priority()
        requested_mode = (mode or LLM_MODE_OLLAMA_FIRST).lower()
        priority = (priority or "local-first").lower()
        selected_cloud_provider = get_effective_cloud_provider()

        cloud_available, ollama_available = await asyncio.gather(
            self._is_cloud_available(),
            self._is_ollama_available(),
        )
        mode = self.derive_effective_mode(requested_mode, cloud_available, ollama_available)

        # Define backend order based on current mode/priority
        backends = self._get_backend_order(mode, priority)
        allow_cloud_provider_fallback = self._allow_cloud_provider_fallback(mode)

        # Explicitly check if mode requires a backend that is not configured
        if mode == LLM_MODE_CLOUD_ONLY and self.cloud is None:
            raise RuntimeError("LLM_MODE=cloud but cloud client not configured")
        if mode == LLM_MODE_OLLAMA_ONLY and self.ollama is None:
            raise RuntimeError("LLM_MODE=local but local client not configured")

        # Attempt backends in order
        last_error = None
        failure_details: List[Dict[str, str]] = []
        for idx, backend_name in enumerate(backends):
            if backend_name == "local":
                client = self.ollama
                timeout = LOCAL_TIMEOUT
                backend_label = "ollama"
                # For local (Ollama), we pass None to let the client use its default model (OLLAMA_MODEL)
                backend_model = None
            else:  # cloud
                if self.cloud is None:
                    continue
                client = self.cloud
                timeout = CLOUD_TIMEOUT
                backend_label = "cloud"
                # For cloud, we pass the requested model (which may be None, letting client use its default)
                backend_model = model

            # No explicit health check; rely on client's circuit breaker and retries

                # Attempt the call with timeout
            try:
                start = time.monotonic()
                # The client's generate method should already be wrapped with circuit breaker and retries
                if stream:
                    # Get the async generator from client
                    if backend_name == "cloud":
                        gen_candidate = client.generate(
                            prompt,
                            system=system,
                            model=backend_model,
                            stream=True,
                            cloud_provider=selected_cloud_provider,
                            allow_provider_fallback=allow_cloud_provider_fallback,
                        )
                    else:
                        gen_candidate = client.generate(
                            prompt,
                            system=system,
                            model=backend_model,
                            stream=True,
                        )

                    # If the client returned a coroutine (most clients that `return agen`
                    # inside an `async def` will do this), await it to get the async generator.
                    if inspect.iscoroutine(gen_candidate):
                        try:
                            gen = await gen_candidate
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.warning(f"First chunk from {backend_label} failed (couldn't get generator)", extra={
                                "backend": backend_label,
                                "error": str(e),
                                "request_id": request_id
                            })
                            metrics.increment("llm.backend.first_chunk_failure", tags={"backend": backend_label})
                            metrics.increment("llm.router.stream_failure", tags={"backend": backend_label})
                            last_error = e
                            failure_details.append(
                                self._build_failure_detail(backend_label, e, "stream_init")
                            )
                            continue
                    else:
                        gen = gen_candidate

                    # gen should now be an async generator. Try to get the first chunk.
                    try:
                        first_chunk = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
                    except asyncio.CancelledError:
                        raise
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning(f"First chunk from {backend_label} failed", extra={
                            "backend": backend_label,
                            "error": str(e),
                            "request_id": request_id
                        })
                        metrics.increment("llm.backend.first_chunk_failure", tags={"backend": backend_label})
                        metrics.increment("llm.router.stream_failure", tags={"backend": backend_label})
                        last_error = e
                        failure_details.append(
                            self._build_failure_detail(backend_label, e, "stream_first_chunk")
                        )
                        # Close the generator to avoid leaks
                        try:
                            await gen.aclose()
                        except Exception:
                            pass
                        continue

                    # First chunk succeeded; now we can stream with per-chunk timeout
                    async def stream_with_first():
                        yield first_chunk
                        # Use the existing _stream_with_timeout for the rest
                        async for chunk in self._stream_with_timeout(gen, timeout, backend_label, request_id):
                            yield chunk

                    logger.info("LLM streaming started", extra={
                        "backend": backend_label,
                        "request_id": request_id
                    })
                    metrics.increment("llm.backend.stream_started", tags={"backend": backend_label})
                    # Router-level success for streaming
                    metrics.increment("llm.router.success", tags={"backend": backend_label})

                    # create actual async generator instance
                    agen_instance = stream_with_first()

                    # wrap in ProviderAsyncGen to guarantee provider attribute and aclose forwarding
                    return ProviderAsyncGen(agen_instance, backend_label)

                else:
                    # Non-streaming: wait for full response with timeout
                    if backend_name == "cloud":
                        generate_call = client.generate(
                            prompt,
                            system=system,
                            model=backend_model,
                            stream=False,
                            cloud_provider=selected_cloud_provider,
                            allow_provider_fallback=allow_cloud_provider_fallback,
                        )
                    else:
                        generate_call = client.generate(
                            prompt,
                            system=system,
                            model=backend_model,
                            stream=False,
                        )
                    result = await asyncio.wait_for(generate_call, timeout=timeout)
                    latency = time.monotonic() - start
                    logger.info("LLM backend success", extra={
                        "backend": backend_label,
                        "latency_sec": round(latency, 3),
                        "request_id": request_id
                    })
                    metrics.increment("llm.backend.success", tags={"backend": backend_label})
                    metrics.increment("llm.router.success", tags={"backend": backend_label})
                    if return_metadata:
                        return {
                            "response": result,
                            "backend": backend_label,
                            "mode": requested_mode,
                            "effective_mode": mode,
                            "cloud_provider": selected_cloud_provider if backend_label == "cloud" else None,
                            "escalated": (idx > 0),  # True if not first attempted backend
                            "request_id": request_id
                        }
                    return result

            except asyncio.CancelledError:
                logger.info(
                    "LLM request cancelled",
                    extra={"backend": backend_label, "request_id": request_id}
                )
                raise
            except asyncio.TimeoutError:
                logger.warning("LLM backend timeout", extra={
                    "backend": backend_label,
                    "timeout_sec": timeout,
                    "request_id": request_id
                })
                metrics.increment("llm.backend.timeout", tags={"backend": backend_label})
                last_error = TimeoutError(f"{backend_label} timeout after {timeout}s")
                failure_details.append(
                    self._build_failure_detail(backend_label, last_error, "timeout")
                )
                continue
            except (OllamaError, CloudLLMError) as e:
                # These are expected errors from the clients (already include circuit breaker failures)
                logger.warning("LLM backend error", extra={
                    "backend": backend_label,
                    "error": str(e),
                    "request_id": request_id
                })
                metrics.increment("llm.backend.error", tags={"backend": backend_label})
                last_error = e
                failure_details.append(self._build_failure_detail(backend_label, e, "backend_error"))
                continue
            except Exception as e:
                # Unexpected errors
                logger.error("Unexpected error from LLM backend", extra={
                    "backend": backend_label,
                    "error": str(e),
                    "request_id": request_id
                })
                metrics.increment("llm.backend.unexpected_error", tags={"backend": backend_label})
                last_error = e
                failure_details.append(self._build_failure_detail(backend_label, e, "unexpected"))
                continue

        if stream:
            metrics.increment("llm.router.stream_failure", tags={"reason": "all_backends_failed"})
        # If we get here, all backends failed
        raise AllBackendsFailed(
            "All LLM backends failed",
            mode=requested_mode,
            effective_mode=mode,
            cloud_provider=selected_cloud_provider,
            failures=failure_details,
        ) from last_error

    def _get_backend_order(self, mode: str, priority: str) -> list:
        """
        Return list of backend names in order of attempt based on current mode/priority.
        Args:
            mode: normalized mode ("local", "cloud", or "hybrid")
            priority: normalized priority ("local-first" or "cloud-first")
        """
        if mode == LLM_MODE_CLOUD_ONLY:
            return ["cloud"]
        if mode == LLM_MODE_OLLAMA_ONLY:
            return ["local"]
        if mode == LLM_MODE_CLOUD_FIRST or priority == "cloud-first":
            return ["cloud", "local"]
        return ["local", "cloud"]

    @staticmethod
    def _allow_cloud_provider_fallback(mode: str) -> bool:
        if mode == LLM_MODE_CLOUD_ONLY:
            return os.getenv("CLOUD_ONLY_ALLOW_PROVIDER_FALLBACK", "1") == "1"
        return True

    @staticmethod
    def _classify_error(error: Exception) -> str:
        if isinstance(error, TimeoutError) or isinstance(error, asyncio.TimeoutError):
            return "timeout"

        message = str(error).lower()

        if "no available keys for service" in message or "no usable keys for provider" in message:
            return "provider_no_active_key"

        if (
            "insufficient_quota" in message
            or "quota exceeded" in message
            or "quota has been exceeded" in message
            or "quota" in message
        ):
            return "provider_quota_exhausted"

        if (
            "billing" in message
            or "hard limit" in message
            or "credit" in message
            or "payment required" in message
            or "account not active" in message
        ):
            return "provider_billing_blocked"

        if "rate limit" in message or "too many requests" in message or "429" in message:
            return "provider_rate_limited"

        if (
            "authentication" in message
            or "unauthorized" in message
            or "invalid api key" in message
            or "permission denied" in message
            or "forbidden" in message
        ):
            return "provider_auth_failed"

        if (
            "connection" in message
            or "unreachable" in message
            or "name resolution" in message
            or "dns" in message
            or "network" in message
            or "temporarily unavailable" in message
        ):
            return "provider_unreachable"

        if "circuit breaker open" in message or "circuit breaker is open" in message:
            return "circuit_open"
        if "cancel" in message:
            return "cancelled"
        return "routing_failed"

    def _build_failure_detail(self, backend: str, error: Exception, stage: str) -> Dict[str, str]:
        return {
            "backend": backend,
            "stage": stage,
            "reason": self._classify_error(error),
            "error": str(error),
        }

    async def _stream_with_timeout(
        self,
        gen: AsyncGenerator[str, None],
        timeout: float,
        backend_label: str,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """
        Wrap an async generator to enforce a timeout for each chunk.
        If the next chunk is not received within `timeout` seconds,
        we close the generator, log, and stop iteration.
        """
        while True:
            try:
                # Wait for the next chunk with a timeout
                chunk = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
                yield chunk
            except StopAsyncIteration:
                # Normal completion
                break
            except asyncio.TimeoutError:
                logger.warning("Streaming timeout", extra={
                    "backend": backend_label,
                    "timeout_sec": timeout,
                    "request_id": request_id
                })
                metrics.increment("llm.backend.stream_timeout", tags={"backend": backend_label})
                # router-level metric for observability
                metrics.increment("llm.router.stream_failure", tags={"backend": backend_label})
                # Attempt to close the generator to avoid resource leaks
                try:
                    await gen.aclose()
                except Exception:
                    pass
                break
            except Exception as e:
                logger.error("Streaming error", extra={
                    "backend": backend_label,
                    "error": str(e),
                    "request_id": request_id
                })
                metrics.increment("llm.backend.stream_error", tags={"backend": backend_label})
                # router-level metric for observability
                metrics.increment("llm.router.stream_failure", tags={"backend": backend_label})
                # Ensure generator is closed before re-raising
                try:
                    await gen.aclose()
                except Exception:
                    pass
                raise

# Singleton instance (clients would be injected in production)
# For simplicity, we create instances here, but in a real app you'd use dependency injection.
ollama_client = OllamaClient()  # assumes async client
# Try to import cloud client; if not available, cloud_client stays None
try:
    cloud_client = CloudLLMClient()
except ImportError:
    cloud_client = None
router = LLMRouter(ollama_client, cloud_client)

# Convenience async function matching previous interface
async def generate(
    prompt: str,
    system: str = "",
    model: Optional[str] = None,
    stream: bool = False,
    request_id: Optional[str] = None,
    return_metadata: bool = False
) -> Union[str, AsyncGenerator[str, None], Dict[str, Any]]:
    """Convenience wrapper for router.generate."""
    return await router.generate(
        prompt=prompt,
        system=system,
        model=model,
        stream=stream,
        request_id=request_id,
        return_metadata=return_metadata
    )

# Additional convenience wrapper for streaming-only calls
async def generate_stream(
    *,
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    request_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Convenience wrapper that returns an async generator for streaming.
    Internally calls generate(..., stream=True).
    """
    return await generate(
        prompt=prompt,
        system=system,
        model=model,
        stream=True,
        request_id=request_id,
        return_metadata=False
    )
