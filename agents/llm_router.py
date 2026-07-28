import logging
import time
import uuid
import asyncio
import inspect
from typing import Optional, Union, AsyncGenerator, Dict, Any, List

# Import async clients (assumed to be fully async, with health checks and streaming)
from agents.ollama_client import (
    OLLAMA_TIMEOUT,
    OllamaClient,
    OllamaError,
    get_runtime_inference_config,
)
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
from core.env_config import get_env_bool, get_env_float
# Import metrics system (if available; otherwise replace with no-op)
try:
    import core.metrics as metrics
except ImportError:
    # Fallback no-op metrics
    class NoOpMetrics:
        def increment(self, name, **tags):
            pass
        def __getattr__(self, _name):
            def _noop(*_args, **_kwargs):
                return None
            return _noop
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


def _resolve_local_timeout() -> float:
    ollama_timeout = max(1.0, get_env_float("OLLAMA_TIMEOUT", OLLAMA_TIMEOUT))
    planner_timeout_hint = get_env_float("PLANNER_LLM_TIMEOUT", OLLAMA_TIMEOUT)
    stream_init_hint = get_env_float("PLANNER_STREAM_INIT_TIMEOUT", planner_timeout_hint)
    local_timeout_default = max(ollama_timeout, planner_timeout_hint, stream_init_hint)
    return get_env_float("LOCAL_LLM_TIMEOUT", local_timeout_default)


def _resolve_cloud_timeout() -> float:
    return get_env_float("CLOUD_LLM_TIMEOUT", 60.0)


def _resolve_router_timeout() -> float:
    configured = get_env_float("ROUTER_TIMEOUT", 90.0)
    # Prevent a self-defeating timeout stack where router cancels before
    # local backend timeout ownership can complete.
    local_floor = _resolve_local_timeout() + 5.0
    return max(configured, local_floor)


def _resolve_probe_timeout(env_name: str, default: float) -> float:
    return max(0.2, get_env_float(env_name, default))


def _estimate_tokens_from_chars(chars: int) -> int:
    if chars <= 0:
        return 0
    return max(1, (int(chars) + 3) // 4)


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
        # Respect strict routing modes exactly as requested.
        if mode in {LLM_MODE_CLOUD_ONLY, LLM_MODE_OLLAMA_ONLY}:
            return mode
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
                probe_timeout = _resolve_probe_timeout("CLOUD_ROUTER_PROBE_TIMEOUT", 1.0)
                probe = has_usable_provider()
                if inspect.isawaitable(probe):
                    return bool(await asyncio.wait_for(probe, timeout=probe_timeout))
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
                probe_timeout = _resolve_probe_timeout("OLLAMA_ROUTER_PROBE_TIMEOUT", 1.5)
                probe = health_check()
                if inspect.isawaitable(probe):
                    return bool(await asyncio.wait_for(probe, timeout=probe_timeout))
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

        Note: For streaming, once the generator starts yielding in llm_router,
              no fallback is possible at the router level. However, the planner
              agent implements graceful degradation by yielding structured fallback
              data when streaming is interrupted mid-stream.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Wrap the entire operation with a global timeout
        router_timeout = _resolve_router_timeout()
        try:
            return await asyncio.wait_for(
                self._route(prompt, system, model, stream, request_id, return_metadata),
                timeout=router_timeout
            )
        except asyncio.TimeoutError:
            logger.error("Router global timeout", extra={"request_id": request_id, "timeout": router_timeout})
            metrics.increment("llm.router.failure", tags={"reason": "global_timeout"})
            raise TimeoutError(f"Router timeout after {router_timeout}s")
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
        local_timeout = _resolve_local_timeout()
        cloud_timeout = _resolve_cloud_timeout()
        router_timeout = _resolve_router_timeout()
        prompt_chars = len(prompt or "")
        system_chars = len(system or "")
        prompt_payload_chars = prompt_chars + system_chars
        prompt_est_tokens = _estimate_tokens_from_chars(prompt_payload_chars)

        cloud_available, ollama_available = await asyncio.gather(
            self._is_cloud_available(),
            self._is_ollama_available(),
        )
        mode = self.derive_effective_mode(requested_mode, cloud_available, ollama_available)
        strict_ollama_mode_probe_bypass = (
            mode == LLM_MODE_OLLAMA_ONLY
            and self.ollama is not None
            and not ollama_available
        )
        local_routable = ollama_available or strict_ollama_mode_probe_bypass

        # Define backend order based on current mode/priority
        backends = self._get_backend_order(mode, priority)
        allow_cloud_provider_fallback = self._allow_cloud_provider_fallback(mode)
        logger.debug(
            "LLM route context",
            extra={
                "request_id": request_id,
                "requested_mode": requested_mode,
                "effective_mode": mode,
                "priority": priority,
                "cloud_available": cloud_available,
                "ollama_available": ollama_available,
                "backend_order": backends,
                "local_timeout_sec": local_timeout,
                "cloud_timeout_sec": cloud_timeout,
                "router_timeout_sec": router_timeout,
                "ollama_probe_bypass_strict_mode": strict_ollama_mode_probe_bypass,
            },
        )
        if strict_ollama_mode_probe_bypass:
            logger.info(
                "Ollama availability probe failed in strict mode; attempting backend call anyway",
                extra={
                    "request_id": request_id,
                    "requested_mode": requested_mode,
                    "effective_mode": mode,
                },
            )

        # Explicitly check if mode requires a backend that is not configured
        if mode == LLM_MODE_CLOUD_ONLY and self.cloud is None:
            raise RuntimeError("LLM_MODE=cloud_only but cloud client not configured")
        if mode == LLM_MODE_OLLAMA_ONLY and self.ollama is None:
            raise RuntimeError("LLM_MODE=ollama_only but local client not configured")

        # Attempt backends in order
        last_error = None
        failure_details: List[Dict[str, str]] = []
        for idx, backend_name in enumerate(backends):
            attempt_is_last = idx == (len(backends) - 1)
            if backend_name == "cloud" and not cloud_available:
                unavailable_error = RuntimeError("Cloud backend unavailable")
                last_error = unavailable_error
                failure_details.append(
                    self._build_failure_detail("cloud", unavailable_error, "availability")
                )
                continue
            if backend_name == "local" and not local_routable:
                unavailable_error = RuntimeError("Ollama backend unavailable")
                last_error = unavailable_error
                failure_details.append(
                    self._build_failure_detail("ollama", unavailable_error, "availability")
                )
                continue

            if backend_name == "local":
                client = self.ollama
                timeout = local_timeout
                backend_label = "ollama"
                # For local (Ollama), we pass None to let the client use its default model (OLLAMA_MODEL)
                backend_model = None
                local_runtime_cfg = get_runtime_inference_config()
                backend_model_name = str(local_runtime_cfg.get("model") or "").strip() or None
                backend_num_ctx = local_runtime_cfg.get("num_ctx")
                backend_num_ctx_source = (
                    str(local_runtime_cfg.get("num_ctx_source") or "").strip() or None
                )
                backend_thinking_mode = (
                    str(local_runtime_cfg.get("thinking_mode") or "").strip().lower() or None
                )
            else:  # cloud
                if self.cloud is None:
                    continue
                client = self.cloud
                timeout = cloud_timeout
                backend_label = "cloud"
                # For cloud, we pass the requested model (which may be None, letting client use its default)
                backend_model = model
                backend_model_name = str(backend_model or "").strip() or None
                backend_num_ctx = None
                backend_num_ctx_source = None
                backend_thinking_mode = None

            # No explicit health check; rely on client's circuit breaker and retries

                # Attempt the call with timeout
            try:
                start = time.monotonic()
                request_start_epoch_ms = int(time.time() * 1000)
                # The client's generate method should already be wrapped with circuit breaker and retries
                if stream:
                    # Get the async generator from client
                    if backend_name == "cloud":
                        gen_candidate = client.generate(
                            prompt,
                            system=system,
                            model=backend_model,
                            stream=True,
                            request_id=request_id,
                            cloud_provider=selected_cloud_provider,
                            allow_provider_fallback=allow_cloud_provider_fallback,
                        )
                    else:
                        gen_candidate = client.generate(
                            prompt,
                            system=system,
                            model=backend_model,
                            stream=True,
                            request_id=request_id,
                            timeout=timeout,
                        )

                    # If the client returned a coroutine (most clients that `return agen`
                    # inside an `async def` will do this), await it to get the async generator.
                    if inspect.iscoroutine(gen_candidate):
                        try:
                            gen = await gen_candidate
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            log_fn = logger.warning if attempt_is_last else logger.info
                            log_fn(f"First chunk from {backend_label} failed (couldn't get generator)", extra={
                                "backend": backend_label,
                                "error": str(e),
                                "request_id": request_id,
                                "attempt_index": idx + 1,
                                "attempt_total": len(backends),
                                "requested_mode": requested_mode,
                                "effective_mode": mode,
                                "timeout_sec": timeout,
                            })
                            metrics.increment("llm.backend.first_chunk_failure", tags={"backend": backend_label})
                            metrics.record_router_stream_failure(f"{backend_label}:stream_init")
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
                        first_chunk_latency_sec = time.monotonic() - start
                        first_chunk_epoch_ms = int(time.time() * 1000)
                    except StopAsyncIteration:
                        empty_stream_error = RuntimeError(f"{backend_label} stream ended before first chunk")
                        log_fn = logger.warning if attempt_is_last else logger.info
                        log_fn("LLM backend produced empty stream", extra={
                            "backend": backend_label,
                            "request_id": request_id,
                            "attempt_index": idx + 1,
                            "attempt_total": len(backends),
                            "requested_mode": requested_mode,
                            "effective_mode": mode,
                            "timeout_sec": timeout,
                        })
                        metrics.increment("llm.backend.first_chunk_failure", tags={"backend": backend_label})
                        metrics.record_router_stream_failure(f"{backend_label}:stream_empty")
                        last_error = empty_stream_error
                        failure_details.append(
                            self._build_failure_detail(backend_label, empty_stream_error, "stream_empty")
                        )
                        continue
                    except asyncio.CancelledError:
                        raise
                    except (asyncio.TimeoutError, Exception) as e:
                        log_fn = logger.warning if attempt_is_last else logger.info
                        log_fn(f"First chunk from {backend_label} failed", extra={
                            "backend": backend_label,
                            "error": str(e),
                            "request_id": request_id,
                            "attempt_index": idx + 1,
                            "attempt_total": len(backends),
                            "requested_mode": requested_mode,
                            "effective_mode": mode,
                            "timeout_sec": timeout,
                        })
                        metrics.increment("llm.backend.first_chunk_failure", tags={"backend": backend_label})
                        metrics.record_router_stream_failure(f"{backend_label}:first_chunk")
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
                        response_chars = len(first_chunk or "")
                        yield first_chunk
                        # Use the existing _stream_with_timeout for the rest
                        async for chunk in self._stream_with_timeout(gen, timeout, backend_label, request_id):
                            response_chars += len(chunk or "")
                            yield chunk
                        
                        response_est_tokens = _estimate_tokens_from_chars(response_chars)
                        metrics.record_llm_tokens(backend_label, prompt_est_tokens, response_est_tokens)

                    logger.debug("LLM streaming started", extra={
                        "backend": backend_label,
                        "request_id": request_id,
                        "model": backend_model_name,
                        "num_ctx": backend_num_ctx,
                        "num_ctx_source": backend_num_ctx_source,
                        "thinking_mode": backend_thinking_mode,
                        "first_chunk_latency_sec": round(first_chunk_latency_sec, 3),
                        "prompt_chars": prompt_payload_chars,
                        "prompt_est_tokens": prompt_est_tokens,
                    })
                    # Route usage should reflect the backend that actually served this request,
                    # not the configured/requested cloud provider.
                    metrics.record_llm_route_usage(
                        mode=requested_mode,
                        effective_mode=mode,
                        provider=backend_label,
                        stream=stream,
                    )
                    metrics.increment("llm.backend.stream_started", tags={"backend": backend_label})
                    # Router-level success for streaming
                    metrics.increment("llm.router.success", tags={"backend": backend_label})
                    if idx > 0:
                        metrics.record_stream_fallback("router_backend_fallback", backend_label)

                    # create actual async generator instance
                    agen_instance = stream_with_first()

                    # wrap in ProviderAsyncGen to guarantee provider attribute and aclose forwarding
                    wrapped_stream = ProviderAsyncGen(agen_instance, backend_label)
                    wrapped_stream.llm_metadata = {
                        "backend": backend_label,
                        "model": backend_model_name,
                        "num_ctx": backend_num_ctx,
                        "num_ctx_source": backend_num_ctx_source,
                        "thinking_mode": backend_thinking_mode,
                        "mode": requested_mode,
                        "effective_mode": mode,
                        "cloud_provider": selected_cloud_provider if backend_label == "cloud" else None,
                        "escalated": (idx > 0),
                        "request_id": request_id,
                        "request_start_epoch_ms": request_start_epoch_ms,
                        "first_chunk_epoch_ms": first_chunk_epoch_ms,
                        "first_chunk_latency_sec": round(first_chunk_latency_sec, 3),
                        "prompt_chars": prompt_payload_chars,
                        "prompt_est_tokens": prompt_est_tokens,
                        "system_chars": system_chars,
                    }
                    return wrapped_stream

                else:
                    # Non-streaming: wait for full response with timeout
                    if backend_name == "cloud":
                        generate_call = client.generate(
                            prompt,
                            system=system,
                            model=backend_model,
                            stream=False,
                            request_id=request_id,
                            cloud_provider=selected_cloud_provider,
                            allow_provider_fallback=allow_cloud_provider_fallback,
                        )
                    else:
                        generate_call = client.generate(
                            prompt,
                            system=system,
                            model=backend_model,
                            stream=False,
                            request_id=request_id,
                            timeout=timeout,
                        )
                    result = await asyncio.wait_for(generate_call, timeout=timeout)
                    latency = time.monotonic() - start
                    completion_epoch_ms = int(time.time() * 1000)
                    response_chars = len(result or "") if isinstance(result, str) else len(str(result or ""))
                    logger.debug("LLM backend success", extra={
                        "backend": backend_label,
                        "model": backend_model_name,
                        "num_ctx": backend_num_ctx,
                        "num_ctx_source": backend_num_ctx_source,
                        "thinking_mode": backend_thinking_mode,
                        "latency_sec": round(latency, 3),
                        "prompt_chars": prompt_payload_chars,
                        "prompt_est_tokens": prompt_est_tokens,
                        "response_chars": response_chars,
                        "request_id": request_id
                    })
                    # Route usage should reflect the backend that actually served this request,
                    # not the configured/requested cloud provider.
                    response_est_tokens = _estimate_tokens_from_chars(response_chars)
                    metrics.record_llm_route_usage(
                        mode=requested_mode,
                        effective_mode=mode,
                        provider=backend_label,
                        stream=stream,
                    )
                    metrics.record_llm_tokens(backend_label, prompt_est_tokens, response_est_tokens)
                    metrics.increment("llm.backend.success", tags={"backend": backend_label})
                    metrics.increment("llm.router.success", tags={"backend": backend_label})
                    metrics.observe_llm_full_response(provider=backend_label, stream=False, duration_sec=latency)
                    if return_metadata:
                        return {
                            "response": result,
                            "backend": backend_label,
                            "model": backend_model_name,
                            "num_ctx": backend_num_ctx,
                            "num_ctx_source": backend_num_ctx_source,
                            "thinking_mode": backend_thinking_mode,
                            "mode": requested_mode,
                            "effective_mode": mode,
                            "cloud_provider": selected_cloud_provider if backend_label == "cloud" else None,
                            "escalated": (idx > 0),  # True if not first attempted backend
                            "request_id": request_id,
                            "request_start_epoch_ms": request_start_epoch_ms,
                            "completion_epoch_ms": completion_epoch_ms,
                            "latency_sec": round(latency, 3),
                            "prompt_chars": prompt_payload_chars,
                            "prompt_est_tokens": prompt_est_tokens,
                            "system_chars": system_chars,
                            "response_chars": response_chars,
                        }
                    return result

            except asyncio.CancelledError:
                logger.debug(
                    "LLM request cancelled",
                    extra={"backend": backend_label, "request_id": request_id}
                )
                if stream:
                    metrics.record_stream_cancellation(backend_label, "router")
                raise
            except asyncio.TimeoutError:
                log_fn = logger.warning if attempt_is_last else logger.info
                log_fn("LLM backend timeout", extra={
                    "backend": backend_label,
                    "timeout_sec": timeout,
                    "request_id": request_id,
                    "attempt_index": idx + 1,
                    "attempt_total": len(backends),
                    "requested_mode": requested_mode,
                    "effective_mode": mode,
                })
                metrics.increment("llm.backend.timeout", tags={"backend": backend_label})
                last_error = TimeoutError(f"{backend_label} timeout after {timeout}s")
                failure_details.append(
                    self._build_failure_detail(backend_label, last_error, "timeout")
                )
                continue
            except (OllamaError, CloudLLMError) as e:
                # These are expected errors from the clients (already include circuit breaker failures)
                failure_reason = self._classify_error(e)
                log_fn = logger.warning if attempt_is_last else logger.info
                log_fn("LLM backend error", extra={
                    "backend": backend_label,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "failure_reason": failure_reason,
                    "request_id": request_id,
                    "attempt_index": idx + 1,
                    "attempt_total": len(backends),
                    "requested_mode": requested_mode,
                    "effective_mode": mode,
                })
                metrics.increment("llm.backend.error", tags={"backend": backend_label})
                last_error = e
                failure_details.append(self._build_failure_detail(backend_label, e, "backend_error"))
                continue
            except Exception as e:
                failure_reason = self._classify_error(e)
                classified_unexpected = failure_reason == "routing_failed"
                if classified_unexpected:
                    logger.error("Unexpected error from LLM backend", extra={
                        "backend": backend_label,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "failure_reason": failure_reason,
                        "request_id": request_id,
                        "attempt_index": idx + 1,
                        "attempt_total": len(backends),
                        "requested_mode": requested_mode,
                        "effective_mode": mode,
                    })
                    stage = "unexpected"
                    metrics.increment("llm.backend.unexpected_error", tags={"backend": backend_label})
                else:
                    log_fn = logger.warning if attempt_is_last else logger.info
                    log_fn("LLM backend exception (classified)", extra={
                        "backend": backend_label,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "failure_reason": failure_reason,
                        "request_id": request_id,
                        "attempt_index": idx + 1,
                        "attempt_total": len(backends),
                        "requested_mode": requested_mode,
                        "effective_mode": mode,
                    })
                    stage = "backend_error"
                    metrics.increment("llm.backend.error", tags={"backend": backend_label})
                last_error = e
                failure_details.append(self._build_failure_detail(backend_label, e, stage))
                continue

        if stream:
            metrics.record_router_stream_failure("all_backends_failed")
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
            return get_env_bool("CLOUD_ONLY_ALLOW_PROVIDER_FALLBACK", default=True)
        return True

    @staticmethod
    def _classify_error(error: Exception) -> str:
        if isinstance(error, TimeoutError) or isinstance(error, asyncio.TimeoutError):
            return "timeout"
        if isinstance(error, TypeError):
            return "client_contract_error"
        if isinstance(error, (ValueError, KeyError)):
            return "malformed_response"

        message = str(error).lower()

        if "timed out" in message or "timeout" in message:
            return "timeout"

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
            "unexpected keyword argument" in message
            or "missing required positional argument" in message
        ):
            return "client_contract_error"

        if (
            "malformed response" in message
            or "missing message.content" in message
            or "missing message.content/response" in message
            or "invalid json" in message
            or "jsondecodeerror" in message
        ):
            return "malformed_response"

        if (
            "connection" in message
            or "unreachable" in message
            or "backend unavailable" in message
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
        if "stream ended before first chunk" in message or "empty stream" in message:
            return "stream_empty"
        if "no visible response text" in message or "no visible answer text" in message:
            return "stream_no_visible_tokens"
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
                metrics.record_router_stream_failure(f"{backend_label}:stream_timeout")
                # Attempt to close the generator to avoid resource leaks
                try:
                    await gen.aclose()
                except Exception:
                    pass
                raise TimeoutError(f"{backend_label} streaming chunk timeout after {timeout}s")
            except Exception as e:
                logger.error("Streaming error", extra={
                    "backend": backend_label,
                    "error": str(e),
                    "request_id": request_id
                })
                metrics.increment("llm.backend.stream_error", tags={"backend": backend_label})
                # router-level metric for observability
                metrics.record_router_stream_failure(f"{backend_label}:stream_error")
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
