import logging
import os
import time
import uuid
import asyncio
import inspect
from typing import Optional, Union, AsyncGenerator, Dict, Any

# Import async clients (assumed to be fully async, with health checks and streaming)
from agents.ollama_client import OllamaClient, OllamaError
from agents.cloud_llm import CloudLLMClient, CloudLLMError
# Import dynamic mode/priority from orchestrator
from core.llm_mode import get_llm_mode_and_priority
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
LOCAL_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "30"))
CLOUD_TIMEOUT = float(os.getenv("CLOUD_LLM_TIMEOUT", "60"))
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT", "90"))   # global timeout for entire generate call

# Module logger
logger = logging.getLogger(__name__)

class AllBackendsFailed(Exception):
    """Raised when no LLM backend can successfully generate a response."""
    pass

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
        # Normalize for internal use
        mode = (mode or "hybrid").lower()
        priority = (priority or "local-first").lower()

        # Define backend order based on current mode/priority
        backends = self._get_backend_order(mode, priority)

        # Explicitly check if mode requires a backend that is not configured
        if mode == "cloud" and self.cloud is None:
            raise RuntimeError("LLM_MODE=cloud but cloud client not configured")
        if mode == "local" and self.ollama is None:
            raise RuntimeError("LLM_MODE=local but local client not configured")

        # Attempt backends in order
        last_error = None
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
                    gen_candidate = client.generate(prompt, system=system, model=backend_model, stream=True)

                    # If the client returned a coroutine (most clients that `return agen`
                    # inside an `async def` will do this), await it to get the async generator.
                    if inspect.iscoroutine(gen_candidate):
                        try:
                            gen = await gen_candidate
                        except Exception as e:
                            logger.warning(f"First chunk from {backend_label} failed (couldn't get generator)", extra={
                                "backend": backend_label,
                                "error": str(e),
                                "request_id": request_id
                            })
                            metrics.increment("llm.backend.first_chunk_failure", tags={"backend": backend_label})
                            metrics.increment("llm.router.stream_failure", tags={"backend": backend_label})
                            last_error = e
                            continue
                    else:
                        gen = gen_candidate

                    # gen should now be an async generator. Try to get the first chunk.
                    try:
                        first_chunk = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning(f"First chunk from {backend_label} failed", extra={
                            "backend": backend_label,
                            "error": str(e),
                            "request_id": request_id
                        })
                        metrics.increment("llm.backend.first_chunk_failure", tags={"backend": backend_label})
                        metrics.increment("llm.router.stream_failure", tags={"backend": backend_label})
                        last_error = e
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
                    result = await asyncio.wait_for(
                        client.generate(prompt, system=system, model=backend_model, stream=False),
                        timeout=timeout
                    )
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
                            "mode": mode,
                            "escalated": (idx > 0),  # True if not first attempted backend
                            "request_id": request_id
                        }
                    return result

            except asyncio.TimeoutError:
                logger.warning("LLM backend timeout", extra={
                    "backend": backend_label,
                    "timeout_sec": timeout,
                    "request_id": request_id
                })
                metrics.increment("llm.backend.timeout", tags={"backend": backend_label})
                last_error = TimeoutError(f"{backend_label} timeout after {timeout}s")
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
                continue

        if stream:
            metrics.increment("llm.router.stream_failure", tags={"reason": "all_backends_failed"})
        # If we get here, all backends failed
        raise AllBackendsFailed("All LLM backends failed") from last_error

    def _get_backend_order(self, mode: str, priority: str) -> list:
        """
        Return list of backend names in order of attempt based on current mode/priority.
        Args:
            mode: normalized mode ("local", "cloud", or "hybrid")
            priority: normalized priority ("local-first" or "cloud-first")
        """
        if mode == "cloud":
            return ["cloud"]
        if mode == "local":
            return ["local"]
        # hybrid mode
        if priority == "cloud-first":
            return ["cloud", "local"]
        # default: local-first
        return ["local", "cloud"]

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