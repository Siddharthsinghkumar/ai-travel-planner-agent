#tests/test_api_key_manager.py
import asyncio
import os
import pytest
import time
from datetime import datetime, UTC, timezone, timedelta
import core.api_key_manager as key_manager_module
from core.api_key_manager import APIKeyManager, KeyEntry
from agents.database import SessionLocal, ProviderKeyState, ProviderStateOverride, init_db
import agents.database as database_module

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
    
    # Should raise RuntimeError because no keys are available within wait window.
    with pytest.raises(RuntimeError, match="No available keys"):
        async with km.reserve_key("fake_api", wait_timeout=0.05) as (idx4, key_val4):
            pass


@pytest.mark.asyncio
async def test_single_key_reservation_serializes_instead_of_fail_fast():
    km = APIKeyManager()
    key_name = "SERPAPI_KEY_1"
    key = KeyEntry(
        value="serpapi-only",
        fingerprint=km._fingerprint("serpapi-only"),
        key_name=key_name,
        name_fingerprint=km._fingerprint_name(key_name),
    )
    km._keys = {"serpapi": [key]}
    km._rr_index = {"serpapi": 0}

    order: list[str] = []

    async def _worker(label: str, hold: float):
        async with km.reserve_key("serpapi", wait_timeout=1.0) as (_idx, _key):
            order.append(f"start-{label}")
            await asyncio.sleep(hold)
            order.append(f"end-{label}")

    await asyncio.gather(
        _worker("a", 0.08),
        _worker("b", 0.01),
    )

    assert order == ["start-a", "end-a", "start-b", "end-b"]


@pytest.mark.asyncio
async def test_multi_key_reservation_allows_parallel_allocation():
    km = APIKeyManager()
    k1 = KeyEntry(
        value="serpapi-k1",
        fingerprint=km._fingerprint("serpapi-k1"),
        key_name="SERPAPI_KEY_1",
        name_fingerprint=km._fingerprint_name("SERPAPI_KEY_1"),
    )
    k2 = KeyEntry(
        value="serpapi-k2",
        fingerprint=km._fingerprint("serpapi-k2"),
        key_name="SERPAPI_KEY_2",
        name_fingerprint=km._fingerprint_name("SERPAPI_KEY_2"),
    )
    km._keys = {"serpapi": [k1, k2]}
    km._rr_index = {"serpapi": 0}

    picked: list[int] = []

    async def _worker():
        async with km.reserve_key("serpapi", wait_timeout=1.0) as (idx, _):
            picked.append(idx)
            await asyncio.sleep(0.05)

    await asyncio.gather(_worker(), _worker())
    assert sorted(picked) == [0, 1]


def test_parse_env_keys_merges_runtime_env_over_dotenv(monkeypatch):
    km = APIKeyManager()
    for name in list(os.environ.keys()):
        if name.startswith("OPENAI_KEY_") or name.startswith("GEMINI_KEY_"):
            monkeypatch.delenv(name, raising=False)

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
    for name in list(os.environ.keys()):
        if name.startswith("GEMINI_KEY_"):
            monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(key_manager_module, "find_dotenv", lambda **_kwargs: "")
    monkeypatch.setenv("GEMINI_KEY_1", "runtime-gemini")

    parsed = km._parse_env_keys()

    assert parsed["gemini"] == ["runtime-gemini"]


@pytest.mark.asyncio
async def test_clear_exhausted_is_non_deadlocking_and_persists_snapshot(monkeypatch):
    km = APIKeyManager()
    fp = km._fingerprint("weather-key-1")
    name = "WEATHER_KEY_1"
    km._keys = {
        "weather": [
            KeyEntry(
                value="weather-key-1",
                fingerprint=fp,
                exhausted_until=time.time() + 120,
                key_name=name,
                name_fingerprint=km._fingerprint_name(name),
            )
        ]
    }

    await asyncio.wait_for(km.clear_exhausted("weather", 0), timeout=1.0)

    assert km._keys["weather"][0].exhausted_until is None
    session = SessionLocal()
    try:
        row = session.query(ProviderKeyState).filter(
            ProviderKeyState.provider == "weather",
            ProviderKeyState.key_name_fingerprint == km._fingerprint_name(name),
        ).first()
        assert row is not None
        assert row.is_exhausted is False
    finally:
        session.close()


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
    monkeypatch.setattr(
        km,
        "_parse_env_key_records",
        lambda: {
            "openai": [
                {"index": 1, "value": key_b, "name": "OPENAI_KEY_1"},
                {"index": 2, "value": key_a, "name": "OPENAI_KEY_2"},
            ]
        },
    )
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    changed = await km._reload_env_keys_if_changed()

    assert changed is True
    assert [entry.fingerprint for entry in km._keys["openai"]] == [
        km._fingerprint(key_b),
        km._fingerprint(key_a),
    ]


@pytest.mark.asyncio
async def test_reload_env_keys_persists_rotated_key_state_to_db(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    old_key = "openai-old"
    new_key = "openai-new"
    key_name = "OPENAI_KEY_1"
    key_name_fp = km._fingerprint_name(key_name)
    km._keys = {
        "openai": [
            KeyEntry(
                value=old_key,
                fingerprint=km._fingerprint(old_key),
                exhausted_until=time.time() + 3600,
                key_name=key_name,
                name_fingerprint=key_name_fp,
            )
        ]
    }
    km._rr_index = {"openai": 0}
    monkeypatch.setattr(
        km,
        "_parse_env_key_records",
        lambda: {"openai": [{"index": 1, "value": new_key, "name": key_name}]},
    )

    changed = await km._reload_env_keys_if_changed()

    assert changed is True
    session = SessionLocal()
    try:
        row = session.query(ProviderKeyState).filter(
            ProviderKeyState.provider == "openai",
            ProviderKeyState.key_name_fingerprint == key_name_fp,
        ).first()
        assert row is not None
        assert row.key_value_fingerprint == km._fingerprint(new_key)
        assert row.is_exhausted is False
    finally:
        session.close()


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
    key_name = "GEMINI_KEY_1"
    key = KeyEntry(
        value="gemini-key",
        fingerprint=km._fingerprint("gemini-key"),
        key_name=key_name,
        name_fingerprint=km._fingerprint_name(key_name),
    )
    key.in_use = 1
    key._pending_exhaust = True
    key._pending_exhaust_until = time.time() - 5
    km._keys = {"gemini": [key]}
    km._rr_index = {"gemini": 0}

    await km.status()

    assert key._pending_exhaust is False
    assert key._pending_exhaust_until is None


@pytest.mark.asyncio
async def test_mark_key_pending_clear_does_not_stick_when_not_in_use():
    km = APIKeyManager()
    key = KeyEntry(value="anthropic-key", fingerprint=km._fingerprint("anthropic-key"))
    km._keys = {"anthropic": [key]}
    km._rr_index = {"anthropic": 0}

    await km.mark_key_pending_clear("anthropic", 0)

    assert key._pending_clear is False


def test_provider_state_session_bootstraps_schema_once(monkeypatch):
    km = APIKeyManager()
    km._provider_state_schema_ready = False
    calls = {"init_db": 0, "session": 0}

    class _FakeSession:
        def close(self):
            return None

    def _fake_init_db():
        calls["init_db"] += 1

    def _fake_session_local():
        calls["session"] += 1
        return _FakeSession()

    monkeypatch.setattr(database_module, "init_db", _fake_init_db)
    monkeypatch.setattr(database_module, "SessionLocal", _fake_session_local)

    with km._provider_state_session() as session1:
        assert session1 is not None
    with km._provider_state_session() as session2:
        assert session2 is not None

    assert calls["init_db"] == 1
    assert calls["session"] == 2


@pytest.mark.asyncio
async def test_status_does_not_expose_key_names_fingerprints_or_raw_provider_errors():
    km = APIKeyManager()
    key = KeyEntry(
        value="serpapi-sensitive",
        fingerprint=km._fingerprint("serpapi-sensitive"),
        key_name="SERPAPI_KEY_1",
        name_fingerprint=km._fingerprint_name("SERPAPI_KEY_1"),
    )
    key.last_provider_error = "https://serpapi.com/account.json?api_key=SECRET"
    key.last_provider_reason = "account_reconcile_exception"
    key.searches_left = 0
    key.failure_classification = "monthly_quota"
    key.last_checked_at = time.time()
    km._keys = {"serpapi": [key]}
    km._rr_index = {"serpapi": 0}

    status = await km.status()
    row = status["serpapi"][0]
    assert "key_name" not in row
    assert "name_fingerprint" not in row
    assert "fingerprint" not in row
    assert "last_provider_error" not in row
    assert "last_provider_reason" not in row
    assert row["searches_left"] == 0
    assert row["failure_classification"] == "monthly_quota"


def test_load_initial_state_discards_expired_persisted_exhaustion(monkeypatch):
    km = APIKeyManager()
    key = "openai-k1"
    fp = km._fingerprint(key)

    monkeypatch.setattr(
        km,
        "_load_state_file",
        lambda: {"openai": {fp: {"exhausted_until": time.time() - 30}}},
    )
    monkeypatch.setattr(
        km,
        "_parse_env_key_records",
        lambda: {"openai": [{"index": 1, "value": key, "name": "OPENAI_KEY_1"}]},
    )

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


@pytest.mark.asyncio
async def test_key_manager_repeated_failure_recovery_sequence_remains_stable(monkeypatch):
    km = APIKeyManager()
    key = "weather-key-seq"
    km._keys = {
        "weather": [KeyEntry(value=key, fingerprint=km._fingerprint(key), exhausted_until=None)]
    }
    km._rr_index = {"weather": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    for _ in range(6):
        await km.mark_exhausted("weather", 0, reason="http_429")
        exhausted = km._keys["weather"][0].exhausted_until
        assert exhausted is not None
        await km.clear_exhausted("weather", 0)
        assert km._keys["weather"][0].exhausted_until is None
        assert km._keys["weather"][0]._pending_exhaust is False
        assert km._keys["weather"][0].in_use == 0

    status = await km.status()
    assert status["weather"][0]["active"] is True


def _clear_provider_key_state_rows() -> None:
    init_db()
    session = SessionLocal()
    try:
        session.query(ProviderKeyState).delete()
        session.query(ProviderStateOverride).delete()
        session.commit()
    finally:
        session.close()


@pytest.mark.asyncio
async def test_serpapi_provider_state_persists_across_manager_restart(monkeypatch):
    _clear_provider_key_state_rows()
    km1 = APIKeyManager()
    key_value = "serpapi-key-live"
    key_name = "SERPAPI_KEY_1"
    ke1 = KeyEntry(
        value=key_value,
        fingerprint=km1._fingerprint(key_value),
        key_name=key_name,
        name_fingerprint=km1._fingerprint_name(key_name),
    )
    km1._keys = {"serpapi": [ke1]}
    km1._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km1, "_write_state_file", lambda _state: None)

    until = time.time() + 1800
    await km1.mark_exhausted("serpapi", 0, reason="quota | plan_searches_left=0", until=until)

    km2 = APIKeyManager()
    ke2 = KeyEntry(
        value=key_value,
        fingerprint=km2._fingerprint(key_value),
        key_name=key_name,
        name_fingerprint=km2._fingerprint_name(key_name),
        exhausted_until=None,
    )
    km2._keys = {"serpapi": [ke2]}
    km2._rr_index = {"serpapi": 0}
    await km2._hydrate_serpapi_state_from_db()

    assert km2._keys["serpapi"][0].exhausted_until is not None
    assert km2._keys["serpapi"][0].exhausted_until > time.time()


@pytest.mark.asyncio
async def test_hydrate_serpapi_monthly_quota_without_reset_does_not_force_reconcile(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key_name = "SERPAPI_KEY_11"
    name_fp = km._fingerprint_name(key_name)
    key_value = "serpapi-quota-no-reset"
    km._upsert_serpapi_provider_state(
        key_name_fingerprint=name_fp,
        key_value_fingerprint=km._fingerprint(key_value),
        is_exhausted=True,
        searches_left=0,
        last_checked_at=key_manager_module._now(),
        expected_reset_basis="account_reconcile_without_reset_timestamp",
        expected_reset_at=None,
        last_error="quota",
        last_reason="account_reconcile_quota_exhausted",
        failure_classification="monthly_quota",
    )
    km._keys = {
        "serpapi": [
            KeyEntry(
                value=key_value,
                fingerprint=km._fingerprint(key_value),
                key_name=key_name,
                name_fingerprint=name_fp,
            )
        ]
    }
    km._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    await km._hydrate_serpapi_state_from_db()

    assert km._keys["serpapi"][0].exhausted_until is not None
    assert name_fp not in km._serpapi_force_reconcile_name_fps


@pytest.mark.asyncio
async def test_serpapi_rotated_key_clears_stale_exhaustion_state(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key_name = "SERPAPI_KEY_2"
    name_fp = km._fingerprint_name(key_name)
    old_key = "serpapi-old"
    new_key = "serpapi-new"

    km._upsert_serpapi_provider_state(
        key_name_fingerprint=name_fp,
        key_value_fingerprint=km._fingerprint(old_key),
        is_exhausted=True,
        searches_left=0,
        last_checked_at=key_manager_module._now(),
        expected_reset_basis="policy_inferred_cycle_boundary",
        expected_reset_at=key_manager_module._now() + key_manager_module.timedelta(days=20),
        last_error="quota",
        last_reason="quota",
        failure_classification="monthly_quota",
    )

    ke = KeyEntry(
        value=new_key,
        fingerprint=km._fingerprint(new_key),
        key_name=key_name,
        name_fingerprint=name_fp,
        exhausted_until=None,
    )
    km._keys = {"serpapi": [ke]}
    km._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)
    await km._hydrate_serpapi_state_from_db()

    assert km._keys["serpapi"][0].exhausted_until is None
    assert km._keys["serpapi"][0].expected_reset_basis == "cleared_on_key_rotation"
    assert name_fp in km._serpapi_force_reconcile_name_fps

    session = SessionLocal()
    try:
        row = session.query(ProviderKeyState).filter(
            ProviderKeyState.provider == "serpapi",
            ProviderKeyState.key_name_fingerprint == name_fp,
        ).first()
        assert row is not None
        assert row.is_exhausted is False
        assert row.key_value_fingerprint == km._fingerprint(new_key)
    finally:
        session.close()


@pytest.mark.asyncio
async def test_startup_reconciliation_marks_quota_exhausted(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key_value = "serpapi-reconcile"
    key_name = "SERPAPI_KEY_3"
    km._keys = {
        "serpapi": [
            KeyEntry(
                value=key_value,
                fingerprint=km._fingerprint(key_value),
                key_name=key_name,
                name_fingerprint=km._fingerprint_name(key_name),
            )
        ]
    }
    km._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"plan_searches_left": 0, "plan_name": "Monthly plan"}

    class _Client:
        async def get(self, *_args, **_kwargs):
            return _Resp()

    monkeypatch.setattr("core.http_client.get_client", lambda: _Client())

    result = await km.reconcile_serpapi_account_state()

    assert result["checked"] == 1
    assert km._keys["serpapi"][0].exhausted_until is not None
    assert km._keys["serpapi"][0].searches_left == 0


@pytest.mark.asyncio
async def test_serpapi_quota_without_reset_uses_weekly_deferral():
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key_name = "SERPAPI_KEY_13"
    key = KeyEntry(
        value="serpapi-unknown-reset",
        fingerprint=km._fingerprint("serpapi-unknown-reset"),
        key_name=key_name,
        name_fingerprint=km._fingerprint_name(key_name),
    )
    km._keys = {"serpapi": [key]}
    km._rr_index = {"serpapi": 0}

    before = time.time()
    await km.mark_exhausted("serpapi", 0, reason="quota | plan_searches_left=0")
    after = time.time()

    entry = km._keys["serpapi"][0]
    assert entry.exhausted_until is not None
    # Weekly fallback window with modest tolerance for runtime.
    assert entry.exhausted_until >= before + (6 * 24 * 3600)
    assert entry.exhausted_until <= after + (8 * 24 * 3600)
    assert entry.expected_reset_basis == "weekly_unknown_reset_fallback"


@pytest.mark.asyncio
async def test_serpapi_quota_uses_default_reset_day_when_available():
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key_name = "SERPAPI_KEY_14"
    key = KeyEntry(
        value="serpapi-default-reset-day",
        fingerprint=km._fingerprint("serpapi-default-reset-day"),
        key_name=key_name,
        name_fingerprint=km._fingerprint_name(key_name),
        default_reset_day=15,
    )
    km._keys = {"serpapi": [key]}
    km._rr_index = {"serpapi": 0}

    before = time.time()
    await km.mark_exhausted("serpapi", 0, reason="quota | plan_searches_left=0")

    entry = km._keys["serpapi"][0]
    assert entry.exhausted_until is not None
    assert entry.exhausted_until > before
    assert entry.exhausted_until < (before + 40 * 24 * 3600)
    assert entry.expected_reset_basis == "default_key_monthly_reset_day"


@pytest.mark.asyncio
async def test_serpapi_reconcile_skips_future_exhausted_keys(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key_value = "serpapi-skip-recheck"
    key_name = "SERPAPI_KEY_9"
    future_until = time.time() + 86400
    km._keys = {
        "serpapi": [
            KeyEntry(
                value=key_value,
                fingerprint=km._fingerprint(key_value),
                key_name=key_name,
                name_fingerprint=km._fingerprint_name(key_name),
                exhausted_until=future_until,
                expected_reset_at=future_until,
                last_checked_at=time.time(),
                failure_classification="monthly_quota",
            )
        ]
    }
    km._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    calls = {"count": 0}

    class _Client:
        async def get(self, *_args, **_kwargs):
            calls["count"] += 1
            class _Resp:
                status_code = 200
                @staticmethod
                def json():
                    return {"plan_searches_left": 0, "plan_name": "Monthly"}
            return _Resp()

    monkeypatch.setattr("core.http_client.get_client", lambda: _Client())
    result = await km.reconcile_serpapi_account_state()

    assert result["checked"] == 0
    assert result["skipped"] == 1
    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_serpapi_rotation_force_reconcile_bypasses_skip(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key_value = "serpapi-force-reconcile"
    key_name = "SERPAPI_KEY_10"
    name_fp = km._fingerprint_name(key_name)
    future_until = time.time() + 86400
    km._keys = {
        "serpapi": [
            KeyEntry(
                value=key_value,
                fingerprint=km._fingerprint(key_value),
                key_name=key_name,
                name_fingerprint=name_fp,
                exhausted_until=future_until,
                expected_reset_at=future_until,
                last_checked_at=time.time(),
                failure_classification="monthly_quota",
            )
        ]
    }
    km._rr_index = {"serpapi": 0}
    km._serpapi_force_reconcile_name_fps.add(name_fp)
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    calls = {"count": 0}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"plan_searches_left": 20}

    class _Client:
        async def get(self, *_args, **_kwargs):
            calls["count"] += 1
            return _Resp()

    monkeypatch.setattr("core.http_client.get_client", lambda: _Client())
    result = await km.reconcile_serpapi_account_state()

    assert result["checked"] == 1
    assert result["forced"] >= 1
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_serpapi_failure_classification_distinguishes_quota_and_transient(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key_value = "serpapi-classify"
    key_name = "SERPAPI_KEY_4"
    name_fp = km._fingerprint_name(key_name)
    km._keys = {
        "serpapi": [
            KeyEntry(
                value=key_value,
                fingerprint=km._fingerprint(key_value),
                key_name=key_name,
                name_fingerprint=name_fp,
            )
        ]
    }
    km._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    await km.mark_exhausted("serpapi", 0, reason="quota | plan_searches_left=0", until=time.time() + 1800)
    session = SessionLocal()
    try:
        row = session.query(ProviderKeyState).filter(
            ProviderKeyState.provider == "serpapi",
            ProviderKeyState.key_name_fingerprint == name_fp,
        ).first()
        assert row is not None
        assert row.failure_classification == "monthly_quota"
    finally:
        session.close()

    await km.clear_exhausted("serpapi", 0)
    await km.mark_exhausted("serpapi", 0, reason="timeout while contacting provider", until=time.time() + 60)
    session = SessionLocal()
    try:
        row = session.query(ProviderKeyState).filter(
            ProviderKeyState.provider == "serpapi",
            ProviderKeyState.key_name_fingerprint == name_fp,
        ).first()
        assert row is not None
        assert row.failure_classification == "transient"
    finally:
        session.close()


@pytest.mark.asyncio
async def test_serpapi_known_reset_override_caps_specific_key_horizon(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key_a = "serpapi-a"
    key_b = "serpapi-b"
    name_a = "SERPAPI_KEY_1"
    name_b = "SERPAPI_KEY_2"
    ke_a = KeyEntry(
        value=key_a,
        fingerprint=km._fingerprint(key_a),
        key_name=name_a,
        name_fingerprint=km._fingerprint_name(name_a),
    )
    ke_b = KeyEntry(
        value=key_b,
        fingerprint=km._fingerprint(key_b),
        key_name=name_b,
        name_fingerprint=km._fingerprint_name(name_b),
    )
    km._keys = {"serpapi": [ke_a, ke_b]}
    km._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    far_future = time.time() + 86400 * 10
    await km.mark_exhausted("serpapi", 0, reason="quota | plan_searches_left=0", until=far_future)
    await km.mark_exhausted("serpapi", 1, reason="quota | plan_searches_left=0", until=far_future)

    known_reset = key_manager_module._now() + key_manager_module.timedelta(minutes=30)
    await km.set_provider_state_override(
        provider="serpapi",
        scope_type="key",
        scope_identifier=ke_b.name_fingerprint,
        override_type="force_exhausted_until",
        active_until=known_reset.isoformat(),
        note="known billing reset for slot 2",
    )

    # Non-target key keeps its original horizon.
    assert km._keys["serpapi"][0].exhausted_until is not None
    assert km._keys["serpapi"][0].exhausted_until >= far_future - 1
    # Target key is capped to the known reset horizon.
    target_until = km._keys["serpapi"][1].exhausted_until
    assert target_until is not None
    assert abs(target_until - known_reset.timestamp()) < 3

    session = SessionLocal()
    try:
        row = session.query(ProviderKeyState).filter(
            ProviderKeyState.provider == "serpapi",
            ProviderKeyState.key_name_fingerprint == ke_b.name_fingerprint,
        ).first()
        assert row is not None
        assert row.expected_reset_basis == "operator_known_reset_datetime"
        assert row.failure_classification == "manual_override"
        assert row.expected_reset_at is not None
    finally:
        session.close()

    # Simulate known reset reached: expire override row and key horizon, then key should be reusable.
    session = SessionLocal()
    try:
        override_row = session.query(ProviderStateOverride).filter(
            ProviderStateOverride.provider == "serpapi",
            ProviderStateOverride.scope_type == "key",
            ProviderStateOverride.scope_identifier == ke_b.name_fingerprint,
        ).order_by(ProviderStateOverride.id.desc()).first()
        assert override_row is not None
        override_row.active_until = key_manager_module._now() - key_manager_module.timedelta(seconds=1)
        session.commit()
    finally:
        session.close()
    km._invalidate_provider_override_cache("serpapi")
    km._keys["serpapi"][1].exhausted_until = time.time() - 1
    await km.status()
    async with km.reserve_key("serpapi") as (idx, _key):
        assert idx == 1


@pytest.mark.asyncio
async def test_provider_override_datetime_output_is_utc_normalized():
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    _active_until_ist = (datetime.now() + timedelta(days=60)).replace(
        hour=5, minute=30, second=0, microsecond=0
    ).astimezone(
        timezone(timedelta(hours=5, minutes=30))
    )
    _expected_utc = _active_until_ist.astimezone(UTC).isoformat()
    row = await km.set_provider_state_override(
        provider="openai",
        scope_type="provider_account",
        scope_identifier="acct-main",
        override_type="force_exhausted_until",
        active_until=_active_until_ist.isoformat(),
        note="tz normalization check",
    )
    assert row["active_until"] == _expected_utc
    assert row["override_until"] == _expected_utc
    assert row["override_until_semantics"] == "forces_exhaustion_until"

    listed = await km.list_provider_state_overrides(provider="openai", include_inactive=True)
    assert listed
    assert listed[0]["active_until"] == _expected_utc
    assert listed[0]["override_until"] == _expected_utc


@pytest.mark.asyncio
async def test_serpapi_key_override_inactive_when_value_rotates_same_slot(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key = KeyEntry(
        value="serpapi-binding-b",
        fingerprint=km._fingerprint("serpapi-binding-b"),
        key_name="SERPAPI_KEY_8",
        name_fingerprint=km._fingerprint_name("SERPAPI_KEY_8"),
    )
    km._keys = {"serpapi": [key]}
    km._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    await km.set_provider_state_override(
        provider="serpapi",
        scope_type="key",
        scope_identifier=key.name_fingerprint,
        override_type="force_exhausted_until",
        active_until=(key_manager_module._now() + key_manager_module.timedelta(minutes=20)).isoformat(),
        note="rotate value",
    )
    rotated_value = "serpapi-binding-b-rotated"
    monkeypatch.setattr(
        km,
        "_parse_env_key_records",
        lambda: {
            "serpapi": [
                {"index": 8, "value": rotated_value, "name": "SERPAPI_KEY_8"},
            ]
        },
    )
    await km._reload_env_keys_if_changed()

    async with km.reserve_key("serpapi") as (idx, _val):
        assert idx == 0

    listed = await km.list_provider_state_overrides(provider="serpapi", include_inactive=True)
    assert listed
    assert listed[0]["binding_matches_current_key"] is False
    assert listed[0]["is_currently_active"] is False


@pytest.mark.asyncio
async def test_serpapi_key_override_inactive_when_slot_name_changes_same_value(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    value = "serpapi-binding-c"
    key = KeyEntry(
        value=value,
        fingerprint=km._fingerprint(value),
        key_name="SERPAPI_KEY_9",
        name_fingerprint=km._fingerprint_name("SERPAPI_KEY_9"),
    )
    km._keys = {"serpapi": [key]}
    km._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    await km.set_provider_state_override(
        provider="serpapi",
        scope_type="key",
        scope_identifier=key.name_fingerprint,
        override_type="force_exhausted_until",
        active_until=(key_manager_module._now() + key_manager_module.timedelta(minutes=20)).isoformat(),
        note="rename slot",
    )
    monkeypatch.setattr(
        km,
        "_parse_env_key_records",
        lambda: {
            "serpapi": [
                {"index": 99, "value": value, "name": "SERPAPI_KEY_99"},
            ]
        },
    )
    await km._reload_env_keys_if_changed()

    async with km.reserve_key("serpapi") as (idx, _val):
        assert idx == 0


@pytest.mark.asyncio
async def test_serpapi_key_override_inactive_when_name_and_value_both_change(monkeypatch):
    _clear_provider_key_state_rows()
    km = APIKeyManager()
    key = KeyEntry(
        value="serpapi-binding-d",
        fingerprint=km._fingerprint("serpapi-binding-d"),
        key_name="SERPAPI_KEY_10",
        name_fingerprint=km._fingerprint_name("SERPAPI_KEY_10"),
    )
    km._keys = {"serpapi": [key]}
    km._rr_index = {"serpapi": 0}
    monkeypatch.setattr(km, "_write_state_file", lambda _state: None)

    await km.set_provider_state_override(
        provider="serpapi",
        scope_type="key",
        scope_identifier=key.name_fingerprint,
        override_type="force_exhausted_until",
        active_until=(key_manager_module._now() + key_manager_module.timedelta(minutes=20)).isoformat(),
        note="change both",
    )
    monkeypatch.setattr(
        km,
        "_parse_env_key_records",
        lambda: {
            "serpapi": [
                {"index": 11, "value": "serpapi-binding-d-rotated", "name": "SERPAPI_KEY_11"},
            ]
        },
    )
    await km._reload_env_keys_if_changed()

    async with km.reserve_key("serpapi") as (idx, _val):
        assert idx == 0
