# tests/test_cloud_provider.py
import pytest
import types
import time

import agents.cloud_llm as cloud_llm


@pytest.fixture(autouse=True)
def _enable_cloud_by_default(monkeypatch):
    monkeypatch.setenv("USE_CLOUD_LLM", "1")


class FakeAdapter:
    def __init__(self, name, response=None, raise_on_call=False, ping_error=None):
        self.provider = name
        self._response = response
        self._raise = raise_on_call
        self._ping_error = ping_error

    async def create_completion(self, model, messages, temperature, max_tokens, timeout):
        if self._raise:
            raise RuntimeError(f"{self.provider} failure")
        class Choice:
            def __init__(self, text):
                self.message = types.SimpleNamespace(content=text)
        class Resp:
            def __init__(self, text):
                self.choices = [Choice(text)]
                self.usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return Resp(self._response)

    async def open_stream(self, model, messages, temperature, max_tokens, timeout):
        # return an async generator that yields one chunk then stops
        async def gen():
            if self._raise:
                raise RuntimeError("open_stream failed")
            class FakeDelta:
                def __init__(self, content): self.content = content
            class FakeChoice:
                def __init__(self, delta): self.delta = delta
            class FakeChunk:
                def __init__(self, delta): self.choices = [FakeChoice(FakeDelta(delta))]
            yield FakeChunk(self._response)
        return gen()

    async def ping(self, _model):
        if self._ping_error is not None:
            raise self._ping_error
        return True


@pytest.fixture(autouse=True)
def _clear_provider_cooldowns():
    cloud_llm._PROVIDER_FAIL_COOLDOWNS.clear()
    yield
    cloud_llm._PROVIDER_FAIL_COOLDOWNS.clear()

@pytest.mark.asyncio
async def test_generate_prefers_first_provider():
    # gemini first, openai second
    gem = FakeAdapter("gemini", response="gemini-ok")
    oai = FakeAdapter("openai", response="openai-ok")
    cloud_llm.provider_chain = [("gemini", gem, (Exception,)), ("openai", oai, (Exception,))]
    res = await cloud_llm.generate(prompt="hi")
    assert "gemini-ok" in res

@pytest.mark.asyncio
async def test_generate_falls_back_to_openai_on_gemini_error():
    gem = FakeAdapter("gemini", raise_on_call=True)
    oai = FakeAdapter("openai", response="openai-ok")
    cloud_llm.provider_chain = [("gemini", gem, (Exception,)), ("openai", oai, (Exception,))]
    res = await cloud_llm.generate(prompt="hi")
    assert "openai-ok" in res

@pytest.mark.asyncio
async def test_generate_stream_fallback_before_first_token():
    gem = FakeAdapter("gemini", raise_on_call=True)
    oai = FakeAdapter("openai", response="openai-stream")
    cloud_llm.provider_chain = [("gemini", gem, (Exception,)), ("openai", oai, (Exception,))]
    # generate_stream returns an async generator — collect text
    agen = cloud_llm.generate_stream(prompt="stream me")
    collected = []
    async for chunk in agen:
        collected.append(chunk)
    joined = "".join(collected)
    assert "openai-stream" in joined


@pytest.mark.asyncio
async def test_provider_usability_gemini_only(monkeypatch):
    gem = FakeAdapter("gemini", response="ok")
    oai = FakeAdapter("openai", response="ok")
    monkeypatch.setattr(cloud_llm, "provider_chain", [("gemini", gem, (Exception,)), ("openai", oai, (Exception,))])

    async def fake_status():
        return {
            "gemini": [{"active": True, "pending_clear": False, "exhausted_until": None}],
            "openai": [{"active": False, "pending_clear": False, "exhausted_until": None}],
        }

    monkeypatch.setattr(cloud_llm.key_manager, "get_status", fake_status)

    usability = await cloud_llm.get_provider_usability()
    usable = await cloud_llm.get_usable_providers()
    assert usability["gemini"] is True
    assert usability["openai"] is False
    assert usable == ["gemini"]


@pytest.mark.asyncio
async def test_provider_usability_none_available(monkeypatch):
    gem = FakeAdapter("gemini", response="ok")
    oai = FakeAdapter("openai", response="ok")
    monkeypatch.setattr(cloud_llm, "provider_chain", [("gemini", gem, (Exception,)), ("openai", oai, (Exception,))])

    async def fake_status():
        return {
            "gemini": [{"active": False, "pending_clear": False, "exhausted_until": None}],
            "openai": [{"active": False, "pending_clear": False, "exhausted_until": None}],
        }

    monkeypatch.setattr(cloud_llm.key_manager, "get_status", fake_status)

    assert await cloud_llm.get_usable_providers() == []
    assert await cloud_llm.cloud_backend_is_usable() is False


@pytest.mark.asyncio
async def test_cloud_backend_usability_respects_admin_flag(monkeypatch):
    gem = FakeAdapter("gemini", response="ok")
    monkeypatch.setattr(cloud_llm, "provider_chain", [("gemini", gem, (Exception,))])

    async def fake_status():
        return {
            "gemini": [{"active": True, "pending_clear": False, "exhausted_until": None}],
        }

    monkeypatch.setattr(cloud_llm.key_manager, "get_status", fake_status)
    monkeypatch.setenv("USE_CLOUD_LLM", "0")

    assert await cloud_llm.cloud_backend_is_usable() is False
    assert await cloud_llm.cloud_backend_is_usable(respect_admin_flag=False) is True


def test_cloud_admin_enablement_default_and_override(monkeypatch):
    monkeypatch.delenv("USE_CLOUD_LLM", raising=False)
    assert cloud_llm.is_cloud_admin_enabled() is True

    monkeypatch.setenv("USE_CLOUD_LLM", "0")
    assert cloud_llm.is_cloud_admin_enabled() is False

    monkeypatch.setenv("USE_CLOUD_LLM", "true")
    assert cloud_llm.is_cloud_admin_enabled() is True


def test_init_provider_gemini_skips_legacy_helper_when_disabled(monkeypatch):
    monkeypatch.setattr(cloud_llm, "ENABLE_GEMINI_HELPER", False)

    def _unexpected_import(_name):
        raise AssertionError("gemini helper import should not be attempted when disabled")

    monkeypatch.setattr(cloud_llm.importlib, "import_module", _unexpected_import)
    assert cloud_llm._init_provider("gemini") is None


def test_resolve_provider_entries_falls_back_when_selected_provider_missing():
    orig = list(cloud_llm.provider_chain)
    try:
        cloud_llm.provider_chain = [("openai", object(), (Exception,))]
        entries = cloud_llm._resolve_provider_entries("gemini", allow_provider_fallback=True)
        assert [name for name, _, _ in entries] == ["openai"]
    finally:
        cloud_llm.provider_chain = orig


def test_resolve_provider_entries_raises_when_selected_provider_missing_and_fallback_disabled():
    orig = list(cloud_llm.provider_chain)
    try:
        cloud_llm.provider_chain = [("openai", object(), (Exception,))]
        with pytest.raises(cloud_llm.CloudLLMError, match="Selected cloud provider 'gemini' is not configured"):
            cloud_llm._resolve_provider_entries("gemini", allow_provider_fallback=False)
    finally:
        cloud_llm.provider_chain = orig


def test_provider_runtime_status_reports_gemini_uninitialized_reason(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER_CHAIN", "gemini")
    monkeypatch.setenv("ENABLE_GEMINI_HELPER", "0")

    cloud_llm.refresh_provider_chain_from_env(force=True)
    status = cloud_llm.get_provider_runtime_status()
    gemini = (status.get("provider_init_status") or {}).get("gemini") or {}

    assert status.get("configured_provider_source") == "CLOUD_PROVIDER_CHAIN"
    assert gemini.get("configured") is True
    assert gemini.get("initialized") is False
    assert gemini.get("reason") == "gemini_helper_disabled"


def test_classify_provider_health_failure_reason_classes():
    assert cloud_llm._classify_provider_health_failure(RuntimeError("401 unauthorized")) == "auth"
    assert cloud_llm._classify_provider_health_failure(RuntimeError("insufficient_quota")) == "quota"
    assert cloud_llm._classify_provider_health_failure(RuntimeError("rate limit 429")) == "rate_limit"
    assert cloud_llm._classify_provider_health_failure(RuntimeError("network timeout")) == "transient"
    assert cloud_llm._classify_provider_health_failure(RuntimeError("Circuit breaker open")) == "circuit_open"


@pytest.mark.asyncio
async def test_health_check_auth_cooldown_and_recovery(monkeypatch):
    adapter = FakeAdapter("openai", ping_error=RuntimeError("401 unauthorized"))
    monkeypatch.setattr(cloud_llm, "provider_chain", [("openai", adapter, (Exception,))])
    monkeypatch.setattr(cloud_llm, "refresh_provider_chain_from_env", lambda force=False: None)
    monkeypatch.setattr(cloud_llm, "PROVIDER_AUTH_FAIL_COOLDOWN", 120)

    async def _usable():
        return ["openai"]

    monkeypatch.setattr(cloud_llm, "get_usable_providers", _usable)

    first = await cloud_llm.health_check()
    assert first == "fail"
    cooldown_until = cloud_llm._PROVIDER_FAIL_COOLDOWNS.get("openai", 0)
    assert cooldown_until > time.time() + 100

    # Provider would now succeed, but active cooldown should defer probing.
    adapter._ping_error = None
    second = await cloud_llm.health_check()
    assert second == "fail"

    # Once cooldown expires, probing resumes and provider can recover to healthy.
    cloud_llm._PROVIDER_FAIL_COOLDOWNS["openai"] = time.time() - 1
    third = await cloud_llm.health_check()
    assert third == "ok"
    assert "openai" not in cloud_llm._PROVIDER_FAIL_COOLDOWNS


@pytest.mark.asyncio
async def test_health_check_transient_cooldown_and_prunes_unconfigured_entries(monkeypatch):
    adapter = FakeAdapter("openai", ping_error=RuntimeError("temporary network timeout"))
    monkeypatch.setattr(cloud_llm, "provider_chain", [("openai", adapter, (Exception,))])
    monkeypatch.setattr(cloud_llm, "refresh_provider_chain_from_env", lambda force=False: None)
    monkeypatch.setattr(cloud_llm, "PROVIDER_TRANSIENT_FAIL_COOLDOWN", 7)

    async def _usable():
        return ["openai"]

    monkeypatch.setattr(cloud_llm, "get_usable_providers", _usable)

    cloud_llm._PROVIDER_FAIL_COOLDOWNS["orphan-provider"] = time.time() + 999
    started = time.time()

    status = await cloud_llm.health_check()

    assert status == "fail"
    assert "orphan-provider" not in cloud_llm._PROVIDER_FAIL_COOLDOWNS
    openai_until = cloud_llm._PROVIDER_FAIL_COOLDOWNS.get("openai", 0)
    assert openai_until >= started + 5
    assert openai_until <= started + 15
