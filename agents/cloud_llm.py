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
import time
import asyncio
import logging
import importlib
import hashlib
from contextlib import asynccontextmanager
from typing import Optional, Any, List, Tuple, Dict

# Core infrastructure
from core.retry import retry_async, RetryConfig
from core.circuit_breaker import get_circuit_breaker, CircuitBreakerOpenError, is_open as circuit_is_open
from core.request_context import get_request_id
from core.api_key_manager import key_manager
from core.env_config import get_env_bool, get_env_float, get_env_int, get_env_str, parse_csv_env

# Configure logging
logger = logging.getLogger(__name__)

# Environment configuration (no longer used directly for API keys)
CLOUD_PROVIDER = (get_env_str("CLOUD_PROVIDER", "gemini") or "gemini").lower()
CLOUD_PROVIDER_CHAIN = get_env_str("CLOUD_PROVIDER_CHAIN")
DEFAULT_MODEL = get_env_str("CLOUD_LLM_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT = get_env_float("CLOUD_LLM_TIMEOUT", 30.0)
DEFAULT_TEMPERATURE = get_env_float("CLOUD_LLM_TEMPERATURE", 0.7)
DEFAULT_MAX_TOKENS = get_env_int("CLOUD_LLM_MAX_TOKENS", 1024)
COST_PER_1K_TOKENS = get_env_float("CLOUD_LLM_COST_PER_1K_TOKENS", 0.0)
MAX_COST_PER_REQUEST = get_env_float("CLOUD_LLM_MAX_COST", 0.0)  # 0 = disabled
FALLBACK_MODELS = parse_csv_env("CLOUD_LLM_FALLBACK_MODELS")
STREAM_CHUNK_TIMEOUT = get_env_float("CLOUD_LLM_STREAM_CHUNK_TIMEOUT", 5.0)

# ---- Cooldown globals for health check ----
PROVIDER_FAIL_COOLDOWN = get_env_int("PROVIDER_FAIL_COOLDOWN", 300)  # seconds
_PROVIDER_FAIL_COOLDOWNS = {}  # provider_name -> unix timestamp until which we skip health checks
# --------------------------------------------------

# ---- Cloud enablement flag ----
def is_cloud_admin_enabled() -> bool:
    """
    Administrative cloud enablement switch.
    USE_CLOUD_LLM=0 disables cloud routing even when provider keys are usable.
    """
    return get_env_bool("USE_CLOUD_LLM", default=True)


# Backward-compatible snapshot retained for legacy log/debug surfaces.
USE_CLOUD_LLM = is_cloud_admin_enabled()
# --------------------------------------------------

# Lazy imports for optional provider SDKs
try:
    from openai import AsyncOpenAI, RateLimitError, APIConnectionError, AuthenticationError as OpenAIAuthError
except ImportError:
    AsyncOpenAI = None
    RateLimitError = None
    APIConnectionError = None
    OpenAIAuthError = None

try:
    from anthropic import AsyncAnthropic, RateLimitError as AnthroRateLimitError, APIConnectionError as AnthroConnErr, AuthenticationError as AnthroAuthError
except ImportError:
    AsyncAnthropic = None
    AnthroRateLimitError = None
    AnthroConnErr = None
    AnthroAuthError = None


# ----------------------------------------------------------------------
# Client cache with in‑use tracking, per‑entry locks, and fingerprints
# ----------------------------------------------------------------------
_clients: Dict[Tuple[str, int], dict] = {}       # (provider, idx) -> metadata dict
_client_lock = asyncio.Lock()

async def _create_client(provider: str, idx: int, key: str):
    """Create a new client for the given provider and key."""
    if provider == "openai":
        if AsyncOpenAI is None:
            raise CloudLLMError("OpenAI SDK not installed")
        return AsyncOpenAI(api_key=key)
    elif provider == "anthropic":
        if AsyncAnthropic is None:
            raise CloudLLMError("Anthropic SDK not installed")
        return AsyncAnthropic(api_key=key)
    else:
        raise CloudLLMError(f"Unsupported provider for client creation: {provider}")

async def _close_client(client):
    """Safely close a client if it has a close method."""
    try:
        if hasattr(client, "aclose"):
            await client.aclose()
        elif hasattr(client, "close"):
            maybe = client.close()
            if asyncio.iscoroutine(maybe):
                await maybe
    except Exception:
        logger.exception("Error closing cached client")

@asynccontextmanager
async def get_client(provider: str, idx: int, key: str):
    """
    Async context manager that yields a cached client for (provider, idx).
    Increments _in_use while the client is held; refuses if _pending_clear is set.
    Uses per‑entry locks for atomicity.
    """
    cache_key = (provider, idx)
    entry_obj = None
    created_local_client = None

    # Fast path: try to reuse an existing entry
    async with _client_lock:
        existing = _clients.get(cache_key)

    if existing:
        # Ensure the per-entry lock exists (create if missing) under global lock to avoid races
        if "_lock" not in existing:
            async with _client_lock:
                if "_lock" not in existing:
                    existing["_lock"] = asyncio.Lock()

        entry_lock = existing["_lock"]
        # Acquire the per-entry lock to check pending_clear and bump _in_use atomically
        async with entry_lock:
            # It's possible another coroutine cleared the entry concurrently; re-check presence
            async with _client_lock:
                if cache_key not in _clients:
                    existing = None
                else:
                    existing = _clients[cache_key]

            if existing is None:
                # fall through to creation path
                pass
            else:
                if existing.get("_pending_clear"):
                    raise CloudLLMError(f"Client for {provider}:{idx} is pending clear and cannot be used")
                existing["_in_use"] = existing.get("_in_use", 0) + 1
                entry_obj = existing

    # If absent, create client outside the lock (avoid blocking other coros)
    if entry_obj is None:
        created_local_client = await _create_client(provider, idx, key)
        # Compute a stable fingerprint for this key so env-change events can target exact instances
        key_fingerprint = hashlib.sha256(key.encode()).hexdigest()
        new_entry = {
            "client": created_local_client,
            "_in_use": 1,
            "_pending_clear": False,
            "created_at": time.time(),
            "_lock": asyncio.Lock(),
            "fingerprint": key_fingerprint,
        }

        async with _client_lock:
            # double-check whether another coroutine created it meanwhile
            existing = _clients.get(cache_key)
            if existing:
                # ensure per-entry lock exists
                if "_lock" not in existing:
                    existing["_lock"] = asyncio.Lock()
                entry_lock = existing["_lock"]

                async with entry_lock:
                    if existing.get("_pending_clear"):
                        # If theirs is pending_clear, drop our client and error
                        await _close_client(created_local_client)
                        raise CloudLLMError(f"Client for {provider}:{idx} is pending clear and cannot be used")
                    existing["_in_use"] = existing.get("_in_use", 0) + 1
                    entry_obj = existing
                    # close our unused created client to avoid leaks
                    await _close_client(created_local_client)
                    created_local_client = None
            else:
                _clients[cache_key] = new_entry
                entry_obj = new_entry

    try:
        yield entry_obj["client"]
    finally:
        # decrement the specific entry object we incremented earlier using its per-entry lock
        if entry_obj:
            entry_lock = entry_obj.get("_lock")
            if entry_lock:
                async with entry_lock:
                    entry_obj["_in_use"] = max(0, entry_obj.get("_in_use", 1) - 1)
            else:
                # fallback (should not happen)
                async with _client_lock:
                    entry_obj["_in_use"] = max(0, entry_obj.get("_in_use", 1) - 1)

async def clear_client_cache(provider: str, idx: int, timeout: float = 30.0, expected_fingerprint: str | None = None) -> bool:
    """
    Safely remove the cached client for (provider, idx).
    Marks it pending_clear, waits for _in_use to become zero (with timeout), then closes and removes it.
    If expected_fingerprint is provided, it must match the current entry's fingerprint to proceed.
    Returns True if removed, False if timed out or fingerprint mismatch.
    """
    cache_key = (provider, idx)
    start = time.monotonic()

    async with _client_lock:
        entry = _clients.get(cache_key)
        if not entry:
            return True  # already gone
        # If caller provided an expected_fingerprint, ensure it matches the current entry
        if expected_fingerprint is not None:
            actual_fp = entry.get("fingerprint")
            if actual_fp != expected_fingerprint:
                logger.info(
                    "clear_client_cache skipped for %s:%d due to fingerprint mismatch (expected=%s actual=%s)",
                    provider, idx, expected_fingerprint, actual_fp
                )
                return False
        # Now mark pending_clear while holding client global lock (we will also acquire per-entry lock below)
        entry["_pending_clear"] = True

    # Wait for in_use to drop to zero (exponential backoff)
    backoff = 0.1
    while True:
        async with _client_lock:
            current_entry = _clients.get(cache_key)
        if not current_entry:
            # already removed
            break

        entry_lock = current_entry.get("_lock")
        if entry_lock is None:
            # create a lock if somehow missing (under global lock)
            async with _client_lock:
                current_entry = _clients.get(cache_key)
                if current_entry and "_lock" not in current_entry:
                    current_entry["_lock"] = asyncio.Lock()
                entry_lock = current_entry.get("_lock")

        async with entry_lock:
            # Re-read in_use while holding per-entry lock
            in_use = current_entry.get("_in_use", 0)
            if in_use == 0:
                break

        if time.monotonic() - start > timeout:
            # Snapshot in_use under per-entry lock for the log
            async with entry_lock:
                final_in_use = current_entry.get("_in_use", 0)
            logger.warning(
                "clear_client_cache timed out for %s:%d (in_use=%d), force-closing",
                provider, idx, final_in_use
            )
            break

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 1.0)

    # Now close and remove – hold BOTH locks to ensure atomic pop
    async with _client_lock:
        entry = _clients.get(cache_key)
        if not entry:
            return True

        entry_lock = entry.get("_lock")
        if entry_lock:
            async with entry_lock:
                # verify fingerprint once more
                if expected_fingerprint is not None and entry.get("fingerprint") != expected_fingerprint:
                    logger.info("clear_client_cache aborted at removal: fingerprint changed for %s:%d", provider, idx)
                    return False
                popped = _clients.pop(cache_key, None)
        else:
            # fallback if no per-entry lock (should not happen)
            popped = _clients.pop(cache_key, None)

    # Close outside both locks to avoid blocking
    if popped:
        await _close_client(popped["client"])
        logger.info("Cleared cached client for %s:%d", provider, idx)
        return True
    return False

async def close_all_clients():
    """Close all cached clients (used during shutdown)."""
    # Mark all as pending clear to block new acquisitions
    async with _client_lock:
        for entry in _clients.values():
            entry["_pending_clear"] = True

    # Wait briefly for in_use to drop
    for cache_key, entry in list(_clients.items()):
        waited = 0.0
        while waited < 5.0:
            async with _client_lock:
                if entry["_in_use"] == 0:
                    break
            await asyncio.sleep(0.1)
            waited += 0.1

    # Now close all clients
    async with _client_lock:
        for cache_key, entry in list(_clients.items()):
            await _close_client(entry["client"])
            _clients.pop(cache_key, None)
    logger.info("All cached clients closed")


# ----------------------------------------------------------------------
# Key event listener (to be registered by application startup)
# ----------------------------------------------------------------------
async def on_key_event(event: str, payload: dict):
    """
    Handle key events from the key manager.
    Expected events: "key_exhausted", "env_changed".
    """
    try:
        if event == "key_exhausted":
            provider = payload.get("service")
            idx = payload.get("index")
            if provider and idx is not None:
                logger.info("Key exhausted for %s:%d – clearing client cache", provider, idx)
                asyncio.create_task(clear_client_cache(provider, idx))
        elif event == "env_changed":
            # payload may include new_fingerprint_maps / old_fingerprint_maps
            new_maps = payload.get("new_fingerprint_maps", {}) or {}
            old_maps = payload.get("old_fingerprint_maps", {}) or {}
            affected = payload.get("affected", [])
            for provider, idx in affected:
                # try to find the new fingerprint for this provider/idx (fall back to old)
                expected_fp = None
                try:
                    expected_fp = new_maps.get(provider, {}).get(idx)
                    if expected_fp is None:
                        expected_fp = old_maps.get(provider, {}).get(idx)
                except Exception:
                    # if maps use string keys or other shape, do a best-effort
                    pass

                logger.info("Environment changed for %s:%d – clearing client cache (expected_fp=%s)", provider, idx, expected_fp)
                asyncio.create_task(clear_client_cache(provider, idx, expected_fingerprint=expected_fp))
        else:
            logger.debug("Ignoring unknown key event: %s", event)
    except Exception:
        logger.exception("on_key_event raised unexpectedly (event=%s)", event)


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
    def __init__(self, provider: str, gemini_module=None, gemini_instance=None):
        self.provider = provider
        # For Gemini we keep the module and instance (if any) for reuse, but keys come from key_manager
        self.gemini_module = gemini_module
        self.gemini_instance = gemini_instance

    async def ping(self, model: str) -> None:
        """Perform a minimal health check call. Raises exception on failure."""
        # Use a minimal completion
        await self.create_completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=1,
            timeout=5.0
        )

    async def create_completion(self, *, model, messages, temperature, max_tokens, timeout):
        """Non‑streaming completion. Returns a normalized object with .choices[0].message.content and .usage."""
        provider = self.provider

        if provider == "openai":
            async with _reserve_provider_key("openai") as (idx, key):
                async with get_client(provider, idx, key) as client:
                    try:
                        raw = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            ),
                            timeout=timeout
                        )
                        await key_manager.record_usage("openai", idx)
                        return raw
                    except OpenAIAuthError as e:
                        await key_manager.mark_exhausted("openai", idx, reason="unauthorized")
                        raise
                    except RateLimitError as e:
                        await key_manager.mark_exhausted("openai", idx, reason="rate_limit")
                        raise
                    except (APIConnectionError, asyncio.TimeoutError) as e:
                        # Transient errors, don't mark exhausted
                        raise
                    except Exception as e:
                        raise

        elif provider == "anthropic":
            async with _reserve_provider_key("anthropic") as (idx, key):
                async with get_client(provider, idx, key) as client:
                    # Convert messages to Anthropic prompt format (simplified)
                    system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
                    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
                    prompt = "\n\n".join(user_msgs)

                    try:
                        raw = await asyncio.wait_for(
                            client.completions.create(
                                model=model,
                                prompt=prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            ),
                            timeout=timeout
                        )
                        await key_manager.record_usage("anthropic", idx)
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
                    except AnthroAuthError as e:
                        await key_manager.mark_exhausted("anthropic", idx, reason="unauthorized")
                        raise
                    except AnthroRateLimitError as e:
                        await key_manager.mark_exhausted("anthropic", idx, reason="rate_limit")
                        raise
                    except (AnthroConnErr, asyncio.TimeoutError) as e:
                        raise
                    except Exception as e:
                        raise

        elif provider == "gemini":
            # Use the Gemini helper module stored during init
            if self.gemini_module is None:
                raise CloudLLMError("Gemini helper not available")
            async with _reserve_provider_key("gemini") as (idx, key):
                # Build a prompt from messages
                system_part = ""
                user_part = ""
                for msg in messages:
                    if msg["role"] == "system":
                        system_part = msg["content"]
                    elif msg["role"] == "user":
                        user_part = msg["content"]
                prompt = f"{system_part}\n\n{user_part}" if system_part else user_part

                # Define a sync function that calls the helper with the key
                def sync_call() -> str:
                    # Try client method first if we have an instance that can accept a key
                    if self.gemini_instance is not None and hasattr(self.gemini_instance, "generate"):
                        return self.gemini_instance.generate(prompt, model=model, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                    # Then module-level generate() with api_key param
                    if hasattr(self.gemini_module, "generate"):
                        return self.gemini_module.generate(prompt, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                    # Finally generate_single()
                    if hasattr(self.gemini_module, "generate_single"):
                        return self.gemini_module.generate_single(prompt, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                    raise RuntimeError("No usable Gemini entrypoint found")

                try:
                    content = await asyncio.to_thread(sync_call)
                    await key_manager.record_usage("gemini", idx)
                    # Build a fake response
                    class FakeChoice:
                        def __init__(self, text):
                            self.message = type("Message", (), {"content": text})
                    class FakeResponse:
                        def __init__(self, text):
                            self.choices = [FakeChoice(text)]
                            self.usage = None
                    return FakeResponse(content)
                except Exception as e:
                    # If the error indicates quota exhaustion, mark key exhausted
                    if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                        await key_manager.mark_exhausted("gemini", idx, reason="rate_limit")
                    elif "auth" in str(e).lower() or "unauthorized" in str(e).lower() or "key" in str(e).lower():
                        await key_manager.mark_exhausted("gemini", idx, reason="unauthorized")
                    raise

        else:
            raise RuntimeError(f"Unsupported provider: {provider}")

    async def open_stream(self, *, model, messages, temperature, max_tokens, timeout):
        """Open a streaming connection. Returns an async iterable yielding normalized chunks."""
        provider = self.provider

        if provider == "openai":
            async def _stream_generator():
                async with _reserve_provider_key("openai") as (idx, key):
                    async with get_client(provider, idx, key) as client:
                        try:
                            stream = await asyncio.wait_for(
                                client.chat.completions.create(
                                    model=model,
                                    messages=messages,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    stream=True,
                                ),
                                timeout=timeout
                            )
                        except OpenAIAuthError as e:
                            await key_manager.mark_exhausted("openai", idx, reason="unauthorized")
                            raise
                        except RateLimitError as e:
                            await key_manager.mark_exhausted("openai", idx, reason="rate_limit")
                            raise
                        except (APIConnectionError, asyncio.TimeoutError) as e:
                            raise
                        except Exception as e:
                            raise

                        async for chunk in stream:
                            yield chunk

                        await key_manager.record_usage("openai", idx)
            return _stream_generator()

        elif provider == "anthropic":
            async def _stream_generator():
                async with _reserve_provider_key("anthropic") as (idx, key):
                    async with get_client(provider, idx, key) as client:
                        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
                        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
                        prompt = "\n\n".join(user_msgs)
                        try:
                            raw_stream = await asyncio.wait_for(
                                client.completions.create(
                                    model=model,
                                    prompt=prompt,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    stream=True,
                                ),
                                timeout=timeout
                            )
                        except AnthroAuthError as e:
                            await key_manager.mark_exhausted("anthropic", idx, reason="unauthorized")
                            raise
                        except AnthroRateLimitError as e:
                            await key_manager.mark_exhausted("anthropic", idx, reason="rate_limit")
                            raise
                        except (AnthroConnErr, asyncio.TimeoutError) as e:
                            raise
                        except Exception as e:
                            raise

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

                        await key_manager.record_usage("anthropic", idx)
            return _stream_generator()

        elif provider == "gemini":
            # Gemini streaming not natively supported; fallback to non-streaming
            async def _stream_generator():
                async with _reserve_provider_key("gemini") as (idx, key):
                    # Build prompt
                    system_part = ""
                    user_part = ""
                    for msg in messages:
                        if msg["role"] == "system":
                            system_part = msg["content"]
                        elif msg["role"] == "user":
                            user_part = msg["content"]
                    prompt = f"{system_part}\n\n{user_part}" if system_part else user_part

                    def sync_call() -> str:
                        if self.gemini_instance is not None and hasattr(self.gemini_instance, "generate"):
                            return self.gemini_instance.generate(prompt, model=model, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                        if hasattr(self.gemini_module, "generate"):
                            return self.gemini_module.generate(prompt, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                        if hasattr(self.gemini_module, "generate_single"):
                            return self.gemini_module.generate_single(prompt, max_output_tokens=max_tokens, temperature=temperature, api_key=key)
                        raise RuntimeError("No usable Gemini entrypoint found")

                    try:
                        content = await asyncio.to_thread(sync_call)
                    except Exception as e:
                        if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                            await key_manager.mark_exhausted("gemini", idx, reason="rate_limit")
                        elif "auth" in str(e).lower() or "unauthorized" in str(e).lower() or "key" in str(e).lower():
                            await key_manager.mark_exhausted("gemini", idx, reason="unauthorized")
                        raise

                    # Yield as one chunk
                    class FakeDelta:
                        def __init__(self, content): self.content = content
                    class FakeChoice:
                        def __init__(self, delta): self.delta = delta
                    class FakeChunk:
                        def __init__(self, delta): self.choices = [FakeChoice(FakeDelta(delta))]
                    yield FakeChunk(content)

                    await key_manager.record_usage("gemini", idx)
            return _stream_generator()

        else:
            raise RuntimeError(f"Unsupported provider: {provider}")

    async def close(self):
        """Close any persistent resources (Gemini client may have its own close)."""
        try:
            if self.provider == "gemini" and self.gemini_instance is not None:
                close_method = getattr(self.gemini_instance, "close", None)
                if callable(close_method):
                    maybe = close_method()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            logger.info("Cloud provider adapter closed", extra={"provider": self.provider})
        except Exception as e:
            logger.exception("cloud_adapter_close_failed", extra={"provider": self.provider, "error": str(e)})


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
    """Initialise a single provider. Returns (adapter, retry_exceptions) or None if skipped."""
    if provider == "openai":
        # No client created here; key_manager will provide keys at call time
        return ProviderAdapter(provider), (RateLimitError, APIConnectionError, asyncio.TimeoutError, OpenAIAuthError)

    elif provider == "anthropic":
        # Skip if SDK not installed (but we already have lazy imports)
        if AsyncAnthropic is None:
            logger.warning("Anthropic package not installed - skipping provider.")
            return None
        return ProviderAdapter(provider), (AnthroRateLimitError, AnthroConnErr, asyncio.TimeoutError, AnthroAuthError)

    elif provider == "gemini":
        # Try to import the helper; if it fails, skip Gemini.
        try:
            gemini_mod = importlib.import_module("gemini_multikey_9_3_helper_script")
        except Exception as e:
            logger.warning("Gemini helper module not importable - skipping provider: %s", e)
            return None

        # Look for a GeminiClient class, otherwise use module-level generate functions
        GeminiClient = getattr(gemini_mod, "GeminiClient", None)
        gemini_instance = None
        if GeminiClient is not None:
            # Instantiate without a key; we'll pass key per call
            try:
                gemini_instance = GeminiClient(api_key=None)
            except TypeError:
                try:
                    gemini_instance = GeminiClient()
                except Exception:
                    gemini_instance = None  # adjust if constructor requires key

        adapter = ProviderAdapter(provider, gemini_module=gemini_mod, gemini_instance=gemini_instance)
        # Only retry on timeout for Gemini (or other exceptions if we detect)
        gemini_retry_exc = (asyncio.TimeoutError,)
        return adapter, gemini_retry_exc

    else:
        logger.warning("Unsupported provider %s - skipping.", provider)
        return None


# Build the chain of available providers
provider_chain: List[Tuple[str, ProviderAdapter, Tuple]] = []
_adapters = {}  # for closing later

for prov in _parse_provider_chain():
    try:
        res = _init_provider(prov)
        if res is None:
            logger.info("Provider %s intentionally skipped (missing config).", prov)
            continue
        adapter, retry_exc = res
        provider_chain.append((prov, adapter, retry_exc))
        _adapters[prov] = adapter
        logger.info("Initialised cloud provider", extra={"provider": prov})
    except Exception as e:
        logger.warning("Skipping provider %s due to initialization error: %s", prov, e, exc_info=True)
        continue

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


class CloudProviderUnavailableError(CloudLLMError):
    """Raised when a cloud provider has no currently available keys."""
    pass


def _is_no_available_keys_error(exc: Exception) -> bool:
    return "no available keys for service" in str(exc).lower()


@asynccontextmanager
async def _reserve_provider_key(service: str):
    try:
        async with key_manager.reserve_key(service) as reservation:
            yield reservation
    except RuntimeError as exc:
        if _is_no_available_keys_error(exc):
            raise CloudProviderUnavailableError(str(exc)) from exc
        raise


def _resolve_provider_entries(
    selected_provider: Optional[str],
    allow_provider_fallback: bool,
) -> List[Tuple[str, "ProviderAdapter", Tuple]]:
    entries = list(provider_chain)
    if not entries:
        return []

    if not selected_provider:
        return entries

    normalized = selected_provider.strip().lower()
    prioritized = [entry for entry in entries if entry[0] == normalized]
    if not prioritized:
        available = ", ".join(name for name, _, _ in entries)
        raise CloudLLMError(
            f"Selected cloud provider '{selected_provider}' is not configured. "
            f"Available providers: {available or '(none)'}"
        )

    if not allow_provider_fallback:
        return prioritized

    remainder = [entry for entry in entries if entry[0] != normalized]
    return prioritized + remainder


def get_available_providers() -> List[str]:
    return [name for name, _, _ in provider_chain]


_PROVIDER_KEY_SERVICE_MAP = {
    "openai": "openai",
    "gemini": "gemini",
    "anthropic": "anthropic",
}


def _is_key_entry_usable(entry: Any, now_ts: float) -> bool:
    if isinstance(entry, str):
        return entry.strip().lower() == "active"

    if not isinstance(entry, dict):
        return False

    if bool(entry.get("pending_clear", False) or entry.get("_pending_clear", False)):
        return False

    if "active" in entry and not bool(entry.get("active")):
        return False

    exhausted_until = entry.get("exhausted_until")
    if exhausted_until in (None, ""):
        return True

    try:
        if isinstance(exhausted_until, (int, float)):
            return float(exhausted_until) <= now_ts
    except Exception:
        return False

    # When exhausted_until is already ISO text, key_manager.status() has typically
    # already folded this into `active`; if active==True we treat it as usable.
    return "active" in entry and bool(entry.get("active"))


async def get_provider_usability() -> Dict[str, bool]:
    providers = get_available_providers()
    if not providers:
        return {}

    try:
        status = await key_manager.get_status()
    except Exception:
        logger.exception("provider_usability_status_failed")
        return {provider: False for provider in providers}

    now_ts = time.time()
    usability: Dict[str, bool] = {}

    for provider in providers:
        service = _PROVIDER_KEY_SERVICE_MAP.get(provider)
        if not service:
            usability[provider] = False
            continue

        entries = (status or {}).get(service, [])
        if isinstance(entries, dict):
            iterable = list(entries.values())
        elif isinstance(entries, list):
            iterable = entries
        else:
            iterable = []

        usability[provider] = any(_is_key_entry_usable(entry, now_ts) for entry in iterable)

    return usability


async def get_usable_providers() -> List[str]:
    usability = await get_provider_usability()
    return [provider for provider in get_available_providers() if usability.get(provider, False)]


async def cloud_backend_is_usable(*, respect_admin_flag: bool = True) -> bool:
    if respect_admin_flag and not is_cloud_admin_enabled():
        return False
    return bool(await get_usable_providers())


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
    cloud_provider: str | None = None,
    allow_provider_fallback: bool = True,
) -> str:
    """
    Generate a non‑streaming completion with provider‑level and model‑level fallback,
    and cost guardrail.
    """
    if not is_cloud_admin_enabled():
        raise CloudLLMError("Cloud LLM is disabled by configuration (USE_CLOUD_LLM=0)")

    # Handle defaults
    primary_model = DEFAULT_MODEL if model is None else model
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    provider_entries = _resolve_provider_entries(cloud_provider, allow_provider_fallback)
    if not provider_entries:
        raise CloudLLMError("No cloud providers available")

    # For each provider in chain
    last_exception = None
    for prov_name, adapter, retry_exc in provider_entries:
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
                if _is_no_available_keys_error(e):
                    logger.info(
                        "Provider %s unavailable due to key exhaustion/unavailability",
                        prov_name,
                        extra={"provider": prov_name},
                    )
                    break
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
    cloud_provider: str | None = None,
    allow_provider_fallback: bool = True,
):
    """
    Generate a streaming completion with provider‑level and model‑level fallback
    (before first token) and cost guardrail.
    """
    if not is_cloud_admin_enabled():
        raise CloudLLMError("Cloud LLM is disabled by configuration (USE_CLOUD_LLM=0)")

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

    provider_entries = _resolve_provider_entries(cloud_provider, allow_provider_fallback)
    if not provider_entries:
        raise CloudLLMError("No cloud providers available")

    for prov_name, adapter, retry_exc in provider_entries:
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

                async for chunk in breaker.run_generator_protected(
                    _stream_generator_factory,
                    non_failure_exceptions=(CloudProviderUnavailableError,),
                ):
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
                if _is_no_available_keys_error(e):
                    logger.info(
                        "Stream skipped provider %s due to unavailable keys",
                        prov_name,
                        extra={"provider": prov_name},
                    )
                    break
                logger.warning("Stream open failed for provider %s model %s, trying next model", prov_name, attempt_model, exc_info=True)
                continue

        # All models for this provider failed before first token – move to next provider
        logger.warning("All models for provider %s failed before first token, trying next provider", prov_name)

    # All providers and models failed without emitting a single token
    raise last_exception or CloudLLMError("All streaming providers and models failed")


async def health_check() -> str:
    """
    Cloud health check.
    - If USE_CLOUD_LLM=0 → cloud is administratively disabled → return "disabled"
    - If USE_CLOUD_LLM=1 with no usable provider keys → return "unavailable"
    - If USE_CLOUD_LLM=1 with usable providers → require at least one provider ping healthy
    """
    if not is_cloud_admin_enabled():
        logger.info("Cloud LLM disabled via USE_CLOUD_LLM=0; skipping cloud health probe.")
        return "disabled"

    usable_providers = await get_usable_providers()
    if not usable_providers:
        logger.info("Cloud LLM enabled but no usable cloud providers available.")
        return "unavailable"

    now = time.time()

    for prov_name, adapter, _ in provider_chain:
        if prov_name not in usable_providers:
            continue
        if adapter is None:
            continue

        cooldown_until = _PROVIDER_FAIL_COOLDOWNS.get(prov_name, 0)
        if now < cooldown_until:
            continue

        try:
            await adapter.ping(DEFAULT_MODEL)
            logger.info("Health check ok", extra={"provider": prov_name})
            return "ok"
        except Exception as e:
            msg = str(e).lower()
            if "ratelimit" in msg or "insufficient_quota" in msg or "429" in msg:
                _PROVIDER_FAIL_COOLDOWNS[prov_name] = now + PROVIDER_FAIL_COOLDOWN
            continue

    return "fail"


async def close_client():
    """Close all underlying provider clients gracefully."""
    await close_all_clients()
    # Close adapters (Gemini instance if any)
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
        cloud_provider: Optional[str] = None,
        allow_provider_fallback: bool = True,
    ):
        if stream:
            return generate_stream(
                prompt=prompt,
                system=system or "",
                model=model,
                cloud_provider=cloud_provider,
                allow_provider_fallback=allow_provider_fallback,
            )
        return await generate(
            prompt=prompt,
            system=system or "",
            model=model,
            cloud_provider=cloud_provider,
            allow_provider_fallback=allow_provider_fallback,
        )

    async def health_check(self) -> bool:
        return await health_check() == "ok"

    async def get_provider_usability(self) -> Dict[str, bool]:
        return await get_provider_usability()

    async def get_usable_providers(self) -> List[str]:
        return await get_usable_providers()

    async def has_usable_provider(self) -> bool:
        return await cloud_backend_is_usable(respect_admin_flag=True)
