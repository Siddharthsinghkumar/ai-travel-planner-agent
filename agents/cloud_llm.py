# NOTE: The application must call close_client() on shutdown, e.g.:
# @app.on_event("shutdown")
# async def shutdown():
#     await cloud_llm.close_client()


# NOTE:
# Prometheus metrics (LLM_REQUESTS and LLM_LATENCY) are intentionally
# instrumented at the public API layer (generate/stream methods).
# This ensures:
#   - One metric emission per logical LLM request
#   - Correct provider labeling (cloud vs ollama fallback)
#   - Accurate end-to-end latency measurement
# Metrics are NOT added to lower-level transport methods to avoid
# double-counting retries or internal fallback attempts.
import os
import time
import asyncio
import logging
import importlib
from typing import Optional, Any, List, Tuple

# Core infrastructure
from core.retry import retry_async, RetryConfig
from core.circuit_breaker import get_circuit_breaker, CircuitBreakerOpenError, is_open as circuit_is_open
from core.request_context import get_request_id

# Configure logging
logger = logging.getLogger(__name__)

# Environment configuration
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "openai").lower()
CLOUD_PROVIDER_CHAIN = os.getenv("CLOUD_PROVIDER_CHAIN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_MODEL = os.getenv("CLOUD_LLM_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT = float(os.getenv("CLOUD_LLM_TIMEOUT", "30"))
DEFAULT_TEMPERATURE = float(os.getenv("CLOUD_LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("CLOUD_LLM_MAX_TOKENS", "1024"))
COST_PER_1K_TOKENS = float(os.getenv("CLOUD_LLM_COST_PER_1K_TOKENS", "0.0"))
MAX_COST_PER_REQUEST = float(os.getenv("CLOUD_LLM_MAX_COST", "0"))  # 0 = disabled
FALLBACK_MODELS = [m.strip() for m in os.getenv("CLOUD_LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
STREAM_CHUNK_TIMEOUT = float(os.getenv("CLOUD_LLM_STREAM_CHUNK_TIMEOUT", "5.0"))

# ----------------------------------------------------------------------
# Helper: per‑chunk timeout wrapper for async iterators
# ----------------------------------------------------------------------
async def _aiter_with_timeout(aiter, per_item_timeout: float):
    """
    Wrap an async iterable so each __anext__ is awaited with a timeout.
    Yields items until StopAsyncIteration or a per-item timeout occurs.
    """
    ait = aiter.__aiter__()
    while True:
        try:
            item = await asyncio.wait_for(ait.__anext__(), timeout=per_item_timeout)
        except asyncio.TimeoutError:
            raise
        except StopAsyncIteration:
            break
        yield item


# ----------------------------------------------------------------------
# Provider Adapter – normalizes API responses to a consistent shape
# ----------------------------------------------------------------------
class ProviderAdapter:
    def __init__(self, client, provider: str):
        self.client = client
        self.provider = provider

    async def ping(self, model: str) -> None:
        """Perform a minimal health check call. Raises exception on failure."""
        if self.provider == "gemini":
            # Use the wrapper with minimal tokens
            await self.client["call"](
                prompt="ping",
                model=model,
                max_tokens=1,
                temperature=0.0
            )
        else:
            await self.create_completion(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=1,
                timeout=5.0
            )

    async def create_completion(self, *, model, messages, temperature, max_tokens, timeout):
        """Non‑streaming completion. Returns a normalized object with .choices[0].message.content and .usage."""
        if self.provider == "openai":
            raw = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout
            )
            return raw

        elif self.provider == "anthropic":
            # Convert messages to Anthropic prompt format (simplified; real apps need proper formatting)
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            prompt = "\n\n".join(user_msgs)

            raw = await asyncio.wait_for(
                self.client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=timeout
            )
            # Normalize to match OpenAI's response structure
            class FakeChoice:
                def __init__(self, text):
                    self.message = type("Message", (), {"content": text})
            class FakeResponse:
                def __init__(self, text, usage):
                    self.choices = [FakeChoice(text)]
                    self.usage = usage
            text = raw.completion if hasattr(raw, "completion") else raw.choices[0].text
            usage = getattr(raw, "usage", None)
            return FakeResponse(text, usage)

        elif self.provider == "gemini":
            # Build a prompt from messages
            system_part = ""
            user_part = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_part = msg["content"]
                elif msg["role"] == "user":
                    user_part = msg["content"]
            prompt = f"{system_part}\n\n{user_part}" if system_part else user_part

            # Call the Gemini wrapper (this runs in a thread)
            content = await self.client["call"](
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Build a fake response object with usage=None (Gemini helper may not return usage)
            class FakeChoice:
                def __init__(self, text):
                    self.message = type("Message", (), {"content": text})
            class FakeResponse:
                def __init__(self, text):
                    self.choices = [FakeChoice(text)]
                    self.usage = None
            return FakeResponse(content)

        else:
            raise RuntimeError(f"Unsupported provider: {self.provider}")

    async def open_stream(self, *, model, messages, temperature, max_tokens, timeout):
        """Open a streaming connection. Returns an async iterable yielding normalized chunks."""
        if self.provider == "openai":
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                ),
                timeout=timeout
            )
            return stream

        elif self.provider == "anthropic":
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            prompt = "\n\n".join(user_msgs)

            raw_stream = await asyncio.wait_for(
                self.client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                ),
                timeout=timeout
            )
            # Wrap to yield normalized chunks (with .choices[0].delta.content)
            async def _normalize_anthropic_stream():
                async for chunk in raw_stream:
                    delta_text = getattr(chunk, "completion", None) or chunk.choices[0].text
                    if delta_text is None:
                        continue
                    # Build fake chunk object
                    class FakeDelta:
                        def __init__(self, content): self.content = content
                    class FakeChoice:
                        def __init__(self, delta): self.delta = delta
                    class FakeChunk:
                        def __init__(self, delta): self.choices = [FakeChoice(FakeDelta(delta))]
                    yield FakeChunk(delta_text)
            return _normalize_anthropic_stream()

        elif self.provider == "gemini":
            # Gemini helper likely does not support streaming. Fallback: get full response
            # and yield it as a single chunk.
            system_part = ""
            user_part = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_part = msg["content"]
                elif msg["role"] == "user":
                    user_part = msg["content"]
            prompt = f"{system_part}\n\n{user_part}" if system_part else user_part

            content = await self.client["call"](
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Define an async generator that yields the whole content as one chunk
            async def _gemini_stream():
                # Build a fake chunk with the complete content
                class FakeDelta:
                    def __init__(self, content): self.content = content
                class FakeChoice:
                    def __init__(self, delta): self.delta = delta
                class FakeChunk:
                    def __init__(self, delta): self.choices = [FakeChoice(FakeDelta(delta))]
                yield FakeChunk(content)
            return _gemini_stream()

        else:
            raise RuntimeError(f"Unsupported provider: {self.provider}")

    async def close(self):
        """Close the underlying client if possible."""
        try:
            if self.provider in ("openai", "anthropic"):
                close_coro = getattr(self.client, "close", None)
                if callable(close_coro):
                    maybe = close_coro()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            # Gemini client (if any) may have its own close method
            elif self.provider == "gemini" and self.client.get("instance") is not None:
                close_method = getattr(self.client["instance"], "close", None)
                if callable(close_method):
                    maybe = close_method()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            logger.info("Cloud provider client closed", extra={"provider": self.provider})
        except Exception as e:
            logger.exception("cloud_client_close_failed", extra={"provider": self.provider, "error": str(e)})


# ----------------------------------------------------------------------
# Build the provider chain
# ----------------------------------------------------------------------
def _parse_provider_chain() -> List[str]:
    """Return list of provider names in order of fallback."""
    if CLOUD_PROVIDER_CHAIN:
        return [p.strip().lower() for p in CLOUD_PROVIDER_CHAIN.split(",") if p.strip()]
    else:
        return [CLOUD_PROVIDER]


def _init_provider(provider: str):
    """Initialise a single provider. Returns (adapter, retry_exceptions) or raises."""
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing")
        from openai import AsyncOpenAI, RateLimitError, APIConnectionError
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        return ProviderAdapter(client, provider), (RateLimitError, APIConnectionError, asyncio.TimeoutError)

    elif provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic, RateLimitError as AnthroRateLimitError, APIConnectionError as AnthroConnErr
        except ImportError:
            raise RuntimeError("Anthropic provider requested but `anthropic` package not installed")
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        return ProviderAdapter(client, provider), (AnthroRateLimitError, AnthroConnErr, asyncio.TimeoutError)

    elif provider == "gemini":
        try:
            gemini_mod = importlib.import_module("gemini_multikey_9_3_helper_script")
        except Exception as e:
            raise RuntimeError("Gemini helper module not importable") from e

        # Look for a GeminiClient class, otherwise use module-level generate functions
        GeminiClient = getattr(gemini_mod, "GeminiClient", None)
        gemini_instance = None
        if GeminiClient is not None:
            gemini_instance = GeminiClient()   # adjust constructor args if needed

        async def _gemini_generate(prompt: str, *, stream: bool = False, model: Optional[str] = None,
                                   max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                                   **kwargs) -> str:
            def sync_call() -> str:
                # Try client method first
                if gemini_instance is not None and hasattr(gemini_instance, "generate"):
                    return gemini_instance.generate(prompt, model=model, max_output_tokens=max_tokens, temperature=temperature)
                # Then module-level generate()
                if hasattr(gemini_mod, "generate"):
                    return gemini_mod.generate(prompt, max_output_tokens=max_tokens, temperature=temperature)
                # Finally generate_single()
                if hasattr(gemini_mod, "generate_single"):
                    return gemini_mod.generate_single(prompt, max_output_tokens=max_tokens, temperature=temperature)
                raise RuntimeError("No usable Gemini entrypoint found")
            return await asyncio.to_thread(sync_call)

        client = {
            "type": "gemini",
            "call": _gemini_generate,
            "instance": gemini_instance   # keep for potential close()
        }
        # Only retry on timeout (or a specific Gemini exception if defined)
        # If the helper defines a custom exception, add it here.
        gemini_retry_exc = (asyncio.TimeoutError,)
        # Optionally, if gemini_mod has a known exception class, include it:
        # gemini_exc = getattr(gemini_mod, "GeminiError", None)
        # if gemini_exc:
        #     gemini_retry_exc = (asyncio.TimeoutError, gemini_exc)
        return ProviderAdapter(client, provider), gemini_retry_exc

    else:
        raise RuntimeError(f"Unsupported provider: {provider}")


# Build the chain of available providers
provider_chain: List[Tuple[str, ProviderAdapter, Tuple]] = []
_adapters = {}  # for closing later

for prov in _parse_provider_chain():
    try:
        adapter, retry_exc = _init_provider(prov)
        provider_chain.append((prov, adapter, retry_exc))
        _adapters[prov] = adapter
        logger.info("Initialised cloud provider", extra={"provider": prov})
    except Exception as e:
        logger.warning("Skipping provider %s due to initialisation error: %s", prov, e, exc_info=True)

if not provider_chain:
    # Do NOT raise here — allow the application to start even if cloud providers
    # are not available (e.g. because the app is intended to use Ollama or is in test).
    logger.error("No cloud providers could be initialised. Check API keys and dependencies.")
    DEFAULT_PROVIDER = None
    _adapter = None
else:
    # Default to first provider in chain for single-provider operations
    DEFAULT_PROVIDER = provider_chain[0][0]
    _adapter = _adapters[DEFAULT_PROVIDER]   # for backward compatibility in health_check


class CloudLLMError(Exception):
    """Custom exception for cloud LLM errors."""
    pass


async def _check_circuit_breaker(provider: str):
    """Raise CloudLLMError if circuit breaker for this provider is open."""
    if await circuit_is_open(f"cloud_llm_{provider}"):
        raise CloudLLMError(f"Circuit breaker open for provider {provider}")


def _estimate_cost(usage) -> float | None:
    """Calculate estimated cost based on token usage."""
    if COST_PER_1K_TOKENS <= 0 or not usage:
        return None
    total_tokens = getattr(usage, "total_tokens", 0)
    return (total_tokens / 1000) * COST_PER_1K_TOKENS


async def _call_adapter_completion(
    adapter: ProviderAdapter,
    retry_exc: Tuple,
    messages: list,
    model: str,
    temperature: float,
    timeout: float,
    max_tokens: int | None = None,
) -> Any:
    """
    Internal function to call a specific provider with retry, circuit breaker, and timeout.
    Returns the normalized response object.
    """
    start = time.monotonic()
    request_id = get_request_id()
    provider = adapter.provider

    async def _do_call():
        return await adapter.create_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    try:
        cfg = RetryConfig(retries=3, base_delay=1.0)
        response = await retry_async(
            _do_call,
            config=cfg,
            request_id=request_id,
            retry_exceptions=retry_exc   # use provider-specific retry exceptions
        )
        latency = time.monotonic() - start

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        log_extra = {
            "request_id": request_id,
            "provider": provider,
            "model": model,
            "latency_sec": round(latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        cost = _estimate_cost(usage)
        if cost is not None:
            log_extra["estimated_cost_usd"] = round(cost, 6)

        logger.info("Cloud LLM success", extra=log_extra)
        return response

    except Exception as e:
        logger.error("Cloud LLM failure", extra={
            "request_id": request_id,
            "provider": provider,
            "error": str(e),
            "model": model,
        })
        raise


async def generate(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Generate a non‑streaming completion with provider‑level and model‑level fallback,
    and cost guardrail.
    """
    # Handle defaults
    primary_model = DEFAULT_MODEL if model is None else model
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # For each provider in chain
    last_exception = None
    for prov_name, adapter, retry_exc in provider_chain:
        # Check circuit breaker for this provider
        try:
            await _check_circuit_breaker(prov_name)
        except CloudLLMError as e:
            logger.warning("Skipping provider %s due to open circuit", prov_name)
            last_exception = e
            continue

        # Build list of models to try: primary + fallbacks
        models_to_try = [primary_model] + FALLBACK_MODELS
        for attempt_model in models_to_try:
            try:
                response = await _call_adapter_completion(
                    adapter=adapter,
                    retry_exc=retry_exc,
                    messages=messages,
                    model=attempt_model,
                    temperature=temperature,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )

                # Validate response structure
                if not hasattr(response, "choices") or not response.choices:
                    raise CloudLLMError("Malformed response: missing choices")
                if not hasattr(response.choices[0], "message") or not hasattr(response.choices[0].message, "content"):
                    raise CloudLLMError("Malformed response: missing message content")
                content = response.choices[0].message.content
                if content is None:
                    raise CloudLLMError("Empty completion (content is None)")

                # Cost guardrail
                if MAX_COST_PER_REQUEST > 0:
                    usage = getattr(response, "usage", None)
                    cost = _estimate_cost(usage)
                    if cost and cost > MAX_COST_PER_REQUEST:
                        raise CloudLLMError(f"Cost exceeded: ${cost:.6f} > ${MAX_COST_PER_REQUEST:.6f}")

                return content

            except Exception as e:
                last_exception = e
                logger.warning("Provider %s model %s failed, trying next fallback", prov_name, attempt_model, exc_info=True)
                continue

        # If all models for this provider failed, log and move to next provider
        logger.warning("All models for provider %s failed, trying next provider", prov_name)

    # If all providers and models failed
    raise last_exception or CloudLLMError("All providers and models failed")


async def generate_stream(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
):
    """
    Generate a streaming completion with provider‑level and model‑level fallback
    (before first token) and cost guardrail.
    """
    primary_model = DEFAULT_MODEL if model is None else model
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_exception = None
    start_time = time.monotonic()
    request_id = get_request_id()

    for prov_name, adapter, retry_exc in provider_chain:
        # Check circuit breaker for this provider
        try:
            await _check_circuit_breaker(prov_name)
        except CloudLLMError as e:
            logger.warning("Skipping provider %s due to open circuit", prov_name)
            last_exception = e
            continue

        models_to_try = [primary_model] + FALLBACK_MODELS
        tokens_emitted = False

        for attempt_model in models_to_try:
            try:
                # Get circuit breaker instance for this attempt (provider-specific)
                breaker = await get_circuit_breaker(f"cloud_llm_{prov_name}")

                # Configure retry for opening the stream
                cfg = RetryConfig(retries=3, base_delay=1.0)

                # Generator factory: opens the stream and yields chunks with timeout
                async def _stream_generator_factory():
                    # Open stream with retry (before first token)
                    stream = await retry_async(
                        lambda: adapter.open_stream(
                            model=attempt_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=timeout,
                        ),
                        config=cfg,
                        request_id=request_id,
                        retry_exceptions=retry_exc,
                    )
                    # Now consume stream with per‑chunk timeout
                    async for chunk in _aiter_with_timeout(stream, STREAM_CHUNK_TIMEOUT):
                        yield chunk

                # Run the whole generator under breaker protection
                estimated_tokens = 0
                first_token = True

                async for chunk in breaker.run_generator_protected(_stream_generator_factory):
                    # ----- defensive delta extraction -----
                    choice0 = getattr(chunk, "choices", [None])[0]
                    if not choice0:
                        continue

                    delta = None
                    delta_obj = getattr(choice0, "delta", None)
                    if delta_obj:
                        delta = getattr(delta_obj, "content", None)

                    if delta is None:
                        delta = (
                            getattr(choice0, "text", None)
                            or getattr(getattr(choice0, "message", None), "content", None)
                        )

                    if not delta:
                        continue

                    # Cost guardrail
                    estimated_tokens += max(1, len(delta) // 4)
                    if MAX_COST_PER_REQUEST > 0 and COST_PER_1K_TOKENS > 0:
                        estimated_cost = (estimated_tokens / 1000) * COST_PER_1K_TOKENS
                        if estimated_cost > MAX_COST_PER_REQUEST:
                            raise CloudLLMError("Streaming cost exceeded")

                    if first_token:
                        tokens_emitted = True
                        ttft = time.monotonic() - start_time
                        logger.info("Cloud LLM first token", extra={
                            "request_id": request_id,
                            "provider": prov_name,
                            "model": attempt_model,
                            "ttft_sec": round(ttft, 3),
                        })
                        first_token = False

                    yield delta

                # Generator finished normally – success (breaker already recorded success)
                total_latency = time.monotonic() - start_time
                logger.info("Cloud LLM stream completed", extra={
                    "request_id": request_id,
                    "provider": prov_name,
                    "model": attempt_model,
                    "total_latency_sec": round(total_latency, 3),
                    "estimated_tokens": estimated_tokens,
                })
                return

            except CircuitBreakerOpenError:
                logger.warning("cloud_llm circuit open", extra={
                    "request_id": request_id,
                    "provider": prov_name,
                    "model": attempt_model,
                })
                raise CloudLLMError("Circuit breaker open")

            except Exception as e:
                # If tokens were already emitted, we must escalate (no fallback mid‑stream)
                if tokens_emitted:
                    logger.error("Stream failed after first token", extra={
                        "request_id": request_id,
                        "provider": prov_name,
                        "model": attempt_model,
                        "error": str(e),
                    })
                    raise

                # Otherwise (failure before first token) – try next model in this provider
                last_exception = e
                logger.warning("Stream open failed for provider %s model %s, trying next model", prov_name, attempt_model, exc_info=True)
                continue

        # All models for this provider failed before first token – move to next provider
        logger.warning("All models for provider %s failed before first token, trying next provider", prov_name)

    # All providers and models failed without emitting a single token
    raise last_exception or CloudLLMError("All streaming providers and models failed")


async def health_check() -> str:
    """
    Lightweight health check for dependency injection / health endpoints.
    Returns "ok" if at least one cloud backend is reachable, otherwise "fail".
    """
    for prov_name, adapter, _ in provider_chain:
        try:
            await adapter.ping(DEFAULT_MODEL)
            logger.info("Health check ok", extra={"provider": prov_name})
            return "ok"
        except Exception:
            logger.warning("Health check failed for provider %s", prov_name, exc_info=True)
            continue
    return "fail"


async def close_client():
    """Close all underlying provider clients gracefully."""
    for adapter in _adapters.values():
        await adapter.close()
    logger.info("Cloud client shutdown complete")


class CloudLLMClient:
    """
    Wrapper around cloud LLM module-level functions
    to match router expectations.
    """

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
        request_id: Optional[str] = None,
    ):
        if stream:
            return generate_stream(
                prompt=prompt,
                system=system or "",
                model=model,
            )
        return await generate(
            prompt=prompt,
            system=system or "",
            model=model,
        )

    async def health_check(self) -> bool:
        return await health_check() == "ok"