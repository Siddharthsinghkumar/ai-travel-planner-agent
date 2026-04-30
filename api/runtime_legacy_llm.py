"""Compatibility lifecycle for the legacy async LLM client path."""

from __future__ import annotations

from fastapi import FastAPI


async def startup_legacy_llm_client(
    app: FastAPI,
    *,
    enabled: bool,
    logger,
    init_client_fn,
) -> None:
    """
    Initialize the legacy async LLM client only when explicitly enabled.
    The modern path remains llm_router + provider adapters.
    """
    if enabled:
        try:
            await init_client_fn()
            app.state.legacy_llm_client_initialized = True
        except Exception as exc:
            logger.warning("legacy_llm_client_init_skipped: %s", str(exc))
    else:
        logger.debug("legacy_llm_client_disabled_by_config")


async def shutdown_legacy_llm_client(
    app: FastAPI,
    *,
    close_client_fn,
) -> None:
    """Shutdown helper for the legacy compatibility client."""
    if getattr(app.state, "legacy_llm_client_initialized", False):
        await close_client_fn()
