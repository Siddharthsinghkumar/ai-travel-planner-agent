import asyncio

import pytest

from agents import cloud_llm


@pytest.mark.asyncio
async def test_on_key_event_skips_non_cloud_provider_exhaustion(monkeypatch):
    scheduled = []
    cleared = []

    async def fake_clear(provider, idx, **kwargs):
        cleared.append((provider, idx, kwargs))
        return True

    def fake_create_task(coro):
        scheduled.append(coro)
        return None

    monkeypatch.setattr(cloud_llm, "clear_client_cache", fake_clear)
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    await cloud_llm.on_key_event(
        "key_exhausted",
        {"service": "serpapi", "index": 3, "reason_class": "quota", "pending": False},
    )

    assert scheduled == []
    assert cleared == []


@pytest.mark.asyncio
async def test_on_key_event_clears_cloud_provider_cache_on_exhaustion(monkeypatch):
    scheduled = []
    cleared = []

    async def fake_clear(provider, idx, **kwargs):
        cleared.append((provider, idx, kwargs))
        return True

    def fake_create_task(coro):
        scheduled.append(coro)
        return None

    monkeypatch.setattr(cloud_llm, "clear_client_cache", fake_clear)
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    await cloud_llm.on_key_event(
        "key_exhausted",
        {"service": "openai", "index": 1, "reason_class": "quota", "pending": False},
    )

    assert len(scheduled) == 1
    await scheduled[0]
    assert cleared == [("openai", 1, {})]


@pytest.mark.asyncio
async def test_on_key_event_env_changed_skips_non_cloud_and_clears_cloud(monkeypatch):
    scheduled = []
    refreshed = []
    cleared = []

    async def fake_clear(provider, idx, **kwargs):
        cleared.append((provider, idx, kwargs))
        return True

    def fake_refresh_provider_chain_from_env(*, force=False):
        refreshed.append(force)

    def fake_create_task(coro):
        scheduled.append(coro)
        return None

    monkeypatch.setattr(cloud_llm, "clear_client_cache", fake_clear)
    monkeypatch.setattr(cloud_llm, "refresh_provider_chain_from_env", fake_refresh_provider_chain_from_env)
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    await cloud_llm.on_key_event(
        "env_changed",
        {
            "affected": [("serpapi", 3), ("openai", 0)],
            "new_fingerprint_maps": {"openai": {0: "new"}},
            "old_fingerprint_maps": {"openai": {0: "old"}},
        },
    )

    assert refreshed == [True]
    assert len(scheduled) == 1
    await scheduled[0]
    assert cleared == [("openai", 0, {"expected_fingerprint": "new"})]
