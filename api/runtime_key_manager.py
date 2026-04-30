"""Key-manager lifecycle ownership for app startup/shutdown."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable

from fastapi import FastAPI

from core.api_key_manager import key_manager


async def startup_key_manager_runtime(
    app: FastAPI,
    *,
    logger,
    key_load_timeout: float,
    cloud_admin_enabled: bool,
    cloud_startup_relevant: bool,
    refresh_provider_chain_from_env_fn: Callable[..., Any],
    key_event_listener: Callable[..., Any],
    acquire_pluggable_lock_fn: Callable[[], Awaitable[tuple[str, Any]]],
    run_redis_lock_lease_keeper_fn: Callable[[FastAPI, Any, int], Awaitable[None]],
    key_manager_lock_ttl: int,
    run_key_refresh_override: bool,
    key_env_monitor_tick: int,
    serpapi_reconcile_interval: int,
) -> str:
    """
    Start key-manager hydration + ownership loops.

    Returns:
        lock backend selected by ownership probe (e.g. "file", "redis").
    """
    try:
        await asyncio.wait_for(key_manager.load_env_keys(), timeout=key_load_timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "key_manager_load_deferred_to_background",
            extra={"timeout_seconds": key_load_timeout},
        )

        async def _background_key_manager_load() -> None:
            try:
                await key_manager.load_env_keys()
                logger.info("key_manager_background_load_complete")
            except Exception:
                logger.exception("key_manager_background_load_failed")

        app.state.key_manager_hydration_task = asyncio.create_task(_background_key_manager_load())
    except Exception:
        logger.exception("key_manager_load_failed")

    if cloud_admin_enabled:
        try:
            if cloud_startup_relevant:
                refresh_provider_chain_from_env_fn(force=False)
            else:
                logger.debug("cloud_provider_refresh_skipped_non_routing_ollama_only_mode")
        except Exception:
            logger.exception("cloud_provider_refresh_failed")
    else:
        logger.debug("cloud_provider_refresh_skipped_cloud_disabled")

    try:
        already_registered = False
        listeners = getattr(key_manager, "_key_event_listeners", None)
        if listeners is not None:
            try:
                if key_event_listener in listeners:
                    already_registered = True
            except Exception:
                for item in list(listeners):
                    if getattr(item, "__name__", None) == getattr(key_event_listener, "__name__", None):
                        already_registered = True
                        break

        if not already_registered:
            key_manager.register_key_event_listener(key_event_listener)
            app.state.cloud_llm_listener_registered = True
            logger.debug("Registered cloud LLM key event listener")
        else:
            logger.debug("Cloud LLM key event listener already registered in this process")
    except Exception:
        logger.exception("Failed to register cloud LLM key event listener")

    lock_backend, lock_handle = await acquire_pluggable_lock_fn()
    should_run_refresh = lock_handle is not None

    if not should_run_refresh and run_key_refresh_override:
        if lock_backend == "redis":
            logger.error(
                "RUN_KEY_REFRESH=true ignored for redis backend when lock is not acquired; "
                "refusing unsafe refresh-loop ownership."
            )
        else:
            logger.warning(
                "RUN_KEY_REFRESH=true but lock not acquired; starting refresh loop anyway. "
                "Ensure only one replica has this variable set."
            )
            should_run_refresh = True

    if should_run_refresh:
        logger.info("Starting key manager background refresh loop (lock_backend=%s).", lock_backend)
        app.state.key_manager_lock_backend = lock_backend
        app.state.key_manager_lock_handle = lock_handle

        key_manager.start_refresh_loop(
            interval_seconds=key_env_monitor_tick,
            skip_lock_check=True,
        )
        # SerpAPI reconcile loop REMOVED — was burning 1 SerpAPI search per key
        # every 30 minutes (6 keys × 48 runs/day = 288 wasted searches/day).
        # With 250 searches/month/key this is unacceptable.
        app.state.serpapi_reconcile_task = None
        app.state.key_manager_task = key_manager._refresh_task
        app.state.key_manager_refresh_owner = True
        if lock_backend == "redis" and lock_handle is not None:
            _client, lock = lock_handle
            app.state.key_manager_lease_task = asyncio.create_task(
                run_redis_lock_lease_keeper_fn(app, lock, key_manager_lock_ttl)
            )
    else:
        logger.debug("Another process/replica holds the key manager lock; not starting refresh loop.")
        app.state.key_manager_lock_backend = None
        app.state.key_manager_lock_handle = None
        app.state.key_manager_task = None
        app.state.serpapi_reconcile_task = None
        app.state.key_manager_refresh_owner = False

    return lock_backend


async def shutdown_key_manager_runtime(app: FastAPI, *, logger) -> None:
    """Stop key-manager background loops and release ownership lock."""
    if getattr(app.state, "key_manager_task", None):
        try:
            key_manager.stop_refresh_loop()
        except Exception:
            logger.exception("key_manager_stop_refresh_loop_failed")

        task = app.state.key_manager_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("key_manager_task_cancel_failed")

    # SerpAPI reconcile loop removed — no shutdown needed

    lease_task = getattr(app.state, "key_manager_lease_task", None)
    if lease_task and not lease_task.done():
        lease_task.cancel()
        try:
            await lease_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("key_manager_lease_task_cancel_failed")

    hydration_task = getattr(app.state, "key_manager_hydration_task", None)
    if hydration_task and not hydration_task.done():
        hydration_task.cancel()
        try:
            await hydration_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("key_manager_hydration_task_cancel_failed")

    backend = getattr(app.state, "key_manager_lock_backend", None)
    handle = getattr(app.state, "key_manager_lock_handle", None)
    if backend == "file" and handle is not None:
        try:
            os.close(handle)
            logger.info("Released file lock for key manager refresh.")
        except Exception:
            logger.exception("failed_to_release_file_lock")
    elif backend == "redis" and handle is not None:
        client, lock = handle
        try:
            await lock.release()
        except Exception:
            logger.exception("failed_to_release_redis_lock")
        try:
            await client.close()
        except Exception:
            pass
