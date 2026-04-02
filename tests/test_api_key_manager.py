#tests/test_api_key_manager.py
import asyncio
import pytest
import time
import core.api_key_manager as key_manager_module
from core.api_key_manager import APIKeyManager, KeyEntry

@pytest.mark.asyncio
async def test_key_rotation_and_exhaustion():
    # 1. Create a fresh key manager and load fake keys using KeyEntry dataclass
    km = APIKeyManager()
    km._keys = {
        "fake_api": [
            KeyEntry(value="key1", fingerprint="hash1", exhausted_until=None),
            KeyEntry(value="key2", fingerprint="hash2", exhausted_until=None)
        ]
    }
    km._rr_index = {"fake_api": 0}

    # 2. Test Reservation (Should get key1)
    async with km.reserve_key("fake_api") as (idx1, key_val1):
        assert key_val1 == "key1"
        assert idx1 == 0

    # 3. Test Rotation (Should get key2 next)
    async with km.reserve_key("fake_api") as (idx2, key_val2):
        assert key_val2 == "key2"
        assert idx2 == 1

    # 4. Test Exhaustion 
    # Mark key2 as exhausted for 10 seconds
    future_time = time.time() + 10
    await km.mark_exhausted("fake_api", 1, until=future_time, reason="test_quota")
    
    # Verify key2 is skipped and it wraps around to key1
    async with km.reserve_key("fake_api") as (idx3, key_val3):
        assert key_val3 == "key1"
        assert idx3 == 0

    # 5. Test All Keys Exhausted
    await km.mark_exhausted("fake_api", 0, until=future_time, reason="test_quota")
    
    # Should raise RuntimeError because no keys are available
    with pytest.raises(RuntimeError, match="No available keys"):
        async with km.reserve_key("fake_api") as (idx4, key_val4):
            pass


def test_parse_env_keys_merges_runtime_env_over_dotenv(monkeypatch):
    km = APIKeyManager()

    monkeypatch.setattr(key_manager_module, "find_dotenv", lambda **_kwargs: "/tmp/fake.env")
    monkeypatch.setattr(
        key_manager_module,
        "dotenv_values",
        lambda _path: {
            "OPENAI_KEY_1": "dotenv-openai",
            "GEMINI_KEY_1": "dotenv-gemini",
        },
    )
    monkeypatch.setenv("OPENAI_KEY_1", "runtime-openai")

    parsed = km._parse_env_keys()

    assert "runtime-openai" in parsed["openai"]
    assert "dotenv-openai" not in parsed["openai"]
    assert "dotenv-gemini" in parsed["gemini"]


def test_parse_env_keys_reads_runtime_env_when_dotenv_missing(monkeypatch):
    km = APIKeyManager()

    monkeypatch.setattr(key_manager_module, "find_dotenv", lambda **_kwargs: "")
    monkeypatch.setenv("GEMINI_KEY_1", "runtime-gemini")

    parsed = km._parse_env_keys()

    assert parsed["gemini"] == ["runtime-gemini"]


@pytest.mark.asyncio
async def test_clear_exhausted_is_non_deadlocking_and_persists_snapshot(monkeypatch):
    km = APIKeyManager()
    fp = km._fingerprint("weather-key-1")
    km._keys = {
        "weather": [KeyEntry(value="weather-key-1", fingerprint=fp, exhausted_until=time.time() + 120)]
    }

    writes = []
    monkeypatch.setattr(km, "_write_state_file", lambda state: writes.append(state))

    await asyncio.wait_for(km.clear_exhausted("weather", 0), timeout=1.0)

    assert km._keys["weather"][0].exhausted_until is None
    assert writes == [{}]


@pytest.mark.asyncio
async def test_reload_env_keys_detects_index_reorder(monkeypatch):
    km = APIKeyManager()
    key_a = "openai-key-a"
    key_b = "openai-key-b"
    km._keys = {
        "openai": [
            KeyEntry(value=key_a, fingerprint=km._fingerprint(key_a)),
            KeyEntry(value=key_b, fingerprint=km._fingerprint(key_b)),
        ]
    }
    km._rr_index = {"openai": 0}
    monkeypatch.setattr(km, "_parse_env_keys", lambda: {"openai": [key_b, key_a]})
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    changed = await km._reload_env_keys_if_changed()

    assert changed is True
    assert [entry.fingerprint for entry in km._keys["openai"]] == [
        km._fingerprint(key_b),
        km._fingerprint(key_a),
    ]


@pytest.mark.asyncio
async def test_reload_env_keys_writes_empty_state_when_exhaustion_clears(monkeypatch):
    km = APIKeyManager()
    old_key = "openai-old"
    new_key = "openai-new"
    km._keys = {
        "openai": [
            KeyEntry(
                value=old_key,
                fingerprint=km._fingerprint(old_key),
                exhausted_until=time.time() + 3600,
            )
        ]
    }
    km._rr_index = {"openai": 0}
    monkeypatch.setattr(km, "_parse_env_keys", lambda: {"openai": [new_key]})
    writes = []
    monkeypatch.setattr(km, "_write_state_file", lambda state: writes.append(state))

    changed = await km._reload_env_keys_if_changed()

    assert changed is True
    assert writes
    assert writes[-1] == {}


@pytest.mark.asyncio
async def test_mark_exhausted_keeps_monotonic_exhaustion_window(monkeypatch):
    km = APIKeyManager()
    first_until = time.time() + 300
    km._keys = {
        "openai": [KeyEntry(value="openai-key", fingerprint=km._fingerprint("openai-key"), exhausted_until=first_until)]
    }
    km._rr_index = {"openai": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    await km.mark_exhausted("openai", 0, until=time.time() + 10, reason="transient timeout")

    assert km._keys["openai"][0].exhausted_until >= (first_until - 0.001)


@pytest.mark.asyncio
async def test_status_sweep_clears_expired_pending_exhaustion(monkeypatch):
    km = APIKeyManager()
    key = KeyEntry(value="gemini-key", fingerprint=km._fingerprint("gemini-key"))
    key.in_use = 1
    key._pending_exhaust = True
    key._pending_exhaust_until = time.time() - 5
    km._keys = {"gemini": [key]}
    km._rr_index = {"gemini": 0}
    writes = []
    monkeypatch.setattr(km, "_write_state_file", lambda state: writes.append(state))

    await km.status()

    assert key._pending_exhaust is False
    assert key._pending_exhaust_until is None
    assert writes
    assert writes[-1] == {}


@pytest.mark.asyncio
async def test_mark_key_pending_clear_does_not_stick_when_not_in_use():
    km = APIKeyManager()
    key = KeyEntry(value="anthropic-key", fingerprint=km._fingerprint("anthropic-key"))
    km._keys = {"anthropic": [key]}
    km._rr_index = {"anthropic": 0}

    await km.mark_key_pending_clear("anthropic", 0)

    assert key._pending_clear is False


def test_load_initial_state_discards_expired_persisted_exhaustion(monkeypatch):
    km = APIKeyManager()
    key = "openai-k1"
    fp = km._fingerprint(key)

    monkeypatch.setattr(
        km,
        "_load_state_file",
        lambda: {"openai": {fp: {"exhausted_until": time.time() - 30}}},
    )
    monkeypatch.setattr(km, "_parse_env_keys", lambda: {"openai": [key]})

    km._load_initial_state()

    assert km._keys["openai"][0].exhausted_until is None


def test_start_refresh_loop_releases_lockfile_when_event_loop_missing(monkeypatch):
    km = APIKeyManager()
    fake_fd = 123
    lock_calls = {"flock": 0, "close": 0}

    monkeypatch.setattr(key_manager_module, "_try_acquire_lockfile", lambda _path: fake_fd)

    def _raise_no_loop():
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(key_manager_module.asyncio, "get_running_loop", _raise_no_loop)

    def _fake_flock(fd, _op):
        assert fd == fake_fd
        lock_calls["flock"] += 1

    def _fake_close(fd):
        assert fd == fake_fd
        lock_calls["close"] += 1

    monkeypatch.setattr(key_manager_module.fcntl, "flock", _fake_flock)
    monkeypatch.setattr(key_manager_module.os, "close", _fake_close)

    km.start_refresh_loop(skip_lock_check=False)

    assert km._lockfile_fd is None
    assert lock_calls["flock"] >= 1
    assert lock_calls["close"] == 1
