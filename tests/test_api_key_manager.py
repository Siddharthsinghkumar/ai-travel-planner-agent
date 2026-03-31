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
