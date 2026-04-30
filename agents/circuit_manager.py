"""Circuit manager for LLM failure handling."""

import asyncio
import time
from typing import Optional

from core.env_config import get_env_int, get_env_float

_llm_failures = 0
_llm_failure_lock = asyncio.Lock()
LLM_FAILURE_THRESHOLD = max(1, get_env_int("LLM_FAILURE_THRESHOLD", 5))
LLM_CIRCUIT_RESET_TIMEOUT = get_env_float("LLM_CIRCUIT_RESET_TIMEOUT", 60.0)
LLM_CIRCUIT_OPEN = False
_llm_circuit_reset_time: Optional[float] = None
_llm_last_failure_at: Optional[float] = None
_llm_last_failure_stage: Optional[str] = None
_llm_last_failure_reason: Optional[str] = None


async def check_llm_circuit(*, llm_mode: Optional[str] = None, effective_mode: Optional[str] = None) -> bool:
    """Return True if circuit is open (skip LLM). Handles auto-recovery."""
    global _llm_failures, LLM_CIRCUIT_OPEN, _llm_circuit_reset_time
    global _llm_last_failure_at, _llm_last_failure_stage, _llm_last_failure_reason

    from core.llm_mode import get_llm_mode_and_priority, get_llm_mode_default

    mode_hint = (effective_mode or llm_mode or "").strip().lower()
    if not mode_hint:
        try:
            resolved_mode, _ = await get_llm_mode_and_priority()
            mode_hint = (resolved_mode or "").strip().lower()
        except Exception:
            try:
                mode_hint = (get_llm_mode_default() or "").strip().lower()
            except Exception:
                mode_hint = ""

    from agents.planner_agent import _is_ollama_only_mode

    if _is_ollama_only_mode(mode_hint, mode_hint):
        async with _llm_failure_lock:
            if LLM_CIRCUIT_OPEN or _llm_failures:
                import logging
                logging.getLogger(__name__).info(
                    "LLM planner circuit state reset in ollama_only mode",
                    extra={
                        "failure_scope": "planner_process_consecutive",
                        "previous_failure_count": _llm_failures,
                        "was_open": LLM_CIRCUIT_OPEN,
                    },
                )
            LLM_CIRCUIT_OPEN = False
            _llm_failures = 0
            _llm_circuit_reset_time = None
            _llm_last_failure_at = None
            _llm_last_failure_stage = None
            _llm_last_failure_reason = None
        return False

    async with _llm_failure_lock:
        now = time.monotonic()
        if LLM_CIRCUIT_OPEN and _llm_circuit_reset_time and now > _llm_circuit_reset_time:
            import logging
            logging.getLogger(__name__).info(
                "LLM circuit breaker reset after timeout",
                extra={
                    "failure_scope": "planner_process_consecutive",
                    "reset_timeout_sec": LLM_CIRCUIT_RESET_TIMEOUT,
                },
            )
            LLM_CIRCUIT_OPEN = False
            _llm_failures = 0
            _llm_circuit_reset_time = None
            _llm_last_failure_at = None
            _llm_last_failure_stage = None
            _llm_last_failure_reason = None

        if _llm_failures >= LLM_FAILURE_THRESHOLD:
            if not LLM_CIRCUIT_OPEN:
                import logging
                logging.getLogger(__name__).warning(
                    "LLM circuit breaker OPEN",
                    extra={
                        "failure_scope": "planner_process_consecutive",
                        "failure_count": _llm_failures,
                        "threshold": LLM_FAILURE_THRESHOLD,
                        "reset_timeout_sec": LLM_CIRCUIT_RESET_TIMEOUT,
                        "last_failure_stage": _llm_last_failure_stage,
                        "last_failure_reason": _llm_last_failure_reason,
                    },
                )
                LLM_CIRCUIT_OPEN = True
                _llm_circuit_reset_time = now + LLM_CIRCUIT_RESET_TIMEOUT
        return LLM_CIRCUIT_OPEN


async def record_llm_success():
    """Reset failure count on success."""
    global _llm_failures, LLM_CIRCUIT_OPEN, _llm_circuit_reset_time
    global _llm_last_failure_at, _llm_last_failure_stage, _llm_last_failure_reason

    async with _llm_failure_lock:
        _llm_failures = 0
        LLM_CIRCUIT_OPEN = False
        _llm_circuit_reset_time = None
        _llm_last_failure_at = None
        _llm_last_failure_stage = None
        _llm_last_failure_reason = None


async def record_llm_failure(stage: str, reason: str):
    """Record an LLM failure."""
    global _llm_failures, _llm_last_failure_at, _llm_last_failure_stage, _llm_last_failure_reason

    async with _llm_failure_lock:
        _llm_failures += 1
        _llm_last_failure_at = time.monotonic()
        _llm_last_failure_stage = stage
        _llm_last_failure_reason = reason