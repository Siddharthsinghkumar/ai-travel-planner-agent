import pytest
import asyncio

from agents.llm_router import AllBackendsFailed, LLMRouter
from core.llm_mode import get_default_cloud_provider, llm_routing_context


class _FakeOllamaClient:
    def __init__(self, *, should_fail: bool = False, trace: list[str] | None = None, available: bool = True):
        self.should_fail = should_fail
        self.calls = []
        self.trace = trace if trace is not None else []
        self.available = available

    async def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        self.trace.append("ollama")
        if self.should_fail:
            raise RuntimeError("ollama failure")
        return "ollama-ok"

    async def health_check(self):
        return self.available


class _FakeCloudClient:
    def __init__(self, *, should_fail: bool = False, trace: list[str] | None = None, available: bool = True):
        self.should_fail = should_fail
        self.calls = []
        self.trace = trace if trace is not None else []
        self.available = available

    async def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        self.trace.append("cloud")
        if self.should_fail:
            raise RuntimeError("cloud failure")
        return "cloud-ok"

    async def has_usable_provider(self):
        return self.available


@pytest.mark.asyncio
async def test_router_ollama_only_never_calls_cloud():
    trace = []
    ollama = _FakeOllamaClient(should_fail=True, trace=trace)
    cloud = _FakeCloudClient(should_fail=False, trace=trace)
    router = LLMRouter(ollama, cloud)

    with llm_routing_context(llm_mode="ollama_only", cloud_provider=get_default_cloud_provider()):
        with pytest.raises(AllBackendsFailed) as exc:
            await router.generate(prompt="hello", stream=False)

    assert trace == ["ollama"]
    assert len(cloud.calls) == 0
    assert exc.value.mode == "ollama_only"


@pytest.mark.asyncio
async def test_router_cloud_only_uses_selected_provider_and_skips_ollama():
    trace = []
    ollama = _FakeOllamaClient(should_fail=False, trace=trace)
    cloud = _FakeCloudClient(should_fail=False, trace=trace)
    router = LLMRouter(ollama, cloud)
    provider = get_default_cloud_provider()

    with llm_routing_context(llm_mode="cloud_only", cloud_provider=provider):
        result = await router.generate(prompt="hello", stream=False)

    assert result == "cloud-ok"
    assert trace == ["cloud"]
    assert len(ollama.calls) == 0
    assert cloud.calls[0]["cloud_provider"] == provider


@pytest.mark.asyncio
async def test_router_cloud_first_falls_back_to_ollama():
    trace = []
    ollama = _FakeOllamaClient(should_fail=False, trace=trace)
    cloud = _FakeCloudClient(should_fail=True, trace=trace)
    router = LLMRouter(ollama, cloud)

    with llm_routing_context(llm_mode="cloud_first", cloud_provider=get_default_cloud_provider()):
        result = await router.generate(prompt="hello", stream=False)

    assert result == "ollama-ok"
    assert trace == ["cloud", "ollama"]


@pytest.mark.asyncio
async def test_router_ollama_first_falls_back_to_cloud():
    trace = []
    ollama = _FakeOllamaClient(should_fail=True, trace=trace)
    cloud = _FakeCloudClient(should_fail=False, trace=trace)
    router = LLMRouter(ollama, cloud)

    with llm_routing_context(llm_mode="ollama_first", cloud_provider=get_default_cloud_provider()):
        result = await router.generate(prompt="hello", stream=False)

    assert result == "cloud-ok"
    assert trace == ["ollama", "cloud"]


@pytest.mark.asyncio
async def test_router_cancellation_propagates_without_fallback():
    trace = []

    class _CancellingOllama(_FakeOllamaClient):
        async def generate(self, *args, **kwargs):
            self.calls.append(kwargs)
            self.trace.append("ollama")
            raise asyncio.CancelledError()

    ollama = _CancellingOllama(trace=trace)
    cloud = _FakeCloudClient(should_fail=False, trace=trace)
    router = LLMRouter(ollama, cloud)

    with llm_routing_context(llm_mode="ollama_first", cloud_provider=get_default_cloud_provider()):
        with pytest.raises(asyncio.CancelledError):
            await router.generate(prompt="cancel-me", stream=False)

    assert trace == ["ollama"]


def test_router_error_classification_provider_no_active_key():
    reason = LLMRouter._classify_error(RuntimeError("No available keys for service: openai"))
    assert reason == "provider_no_active_key"


def test_router_error_classification_quota_and_billing():
    quota_reason = LLMRouter._classify_error(RuntimeError("insufficient_quota for this model"))
    billing_reason = LLMRouter._classify_error(RuntimeError("billing hard limit reached"))
    assert quota_reason == "provider_quota_exhausted"
    assert billing_reason == "provider_billing_blocked"


def test_router_error_classification_auth_and_default():
    auth_reason = LLMRouter._classify_error(RuntimeError("unauthorized: invalid api key"))
    default_reason = LLMRouter._classify_error(RuntimeError("unexpected cloud failure signature"))
    assert auth_reason == "provider_auth_failed"
    assert default_reason == "routing_failed"


def test_router_effective_mode_derivation():
    assert LLMRouter.derive_effective_mode("cloud_first", cloud_available=False, ollama_available=True) == "ollama_only"
    assert LLMRouter.derive_effective_mode("ollama_first", cloud_available=True, ollama_available=False) == "cloud_only"
    assert LLMRouter.derive_effective_mode("cloud_first", cloud_available=True, ollama_available=True) == "cloud_first"


@pytest.mark.asyncio
async def test_router_cloud_only_degrades_to_ollama_when_cloud_unavailable():
    trace = []
    ollama = _FakeOllamaClient(should_fail=False, trace=trace, available=True)
    cloud = _FakeCloudClient(should_fail=False, trace=trace, available=False)
    router = LLMRouter(ollama, cloud)

    with llm_routing_context(llm_mode="cloud_only", cloud_provider=get_default_cloud_provider()):
        result = await router.generate(prompt="hello", stream=False)

    assert result == "ollama-ok"
    assert trace == ["ollama"]


@pytest.mark.asyncio
async def test_router_ollama_only_degrades_to_cloud_when_ollama_unavailable():
    trace = []
    ollama = _FakeOllamaClient(should_fail=False, trace=trace, available=False)
    cloud = _FakeCloudClient(should_fail=False, trace=trace, available=True)
    router = LLMRouter(ollama, cloud)

    with llm_routing_context(llm_mode="ollama_only", cloud_provider=get_default_cloud_provider()):
        result = await router.generate(prompt="hello", stream=False)

    assert result == "cloud-ok"
    assert trace == ["cloud"]
