"""Shared OLLAMA_NUM_CTX resolution helpers.

This module keeps runtime and validation context-window resolution predictable
and explicit, with a machine-readable source attached to every decision.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

VALIDATION_NUM_CTX_BASELINE = 4096
RUNTIME_NUM_CTX_DEFAULT = 4096
VALIDATION_NUM_CTX_MODES = {"validated_default", "passthrough", "override"}


def _coerce_ctx_value(raw: Any, *, minimum_value: int) -> tuple[Optional[int], Optional[str]]:
    text = str(raw or "").strip()
    if not text:
        return None, None
    try:
        parsed = int(float(text))
    except Exception:
        return None, "invalid"
    if parsed <= 0:
        return None, "invalid"
    if parsed < minimum_value:
        return max(1, int(minimum_value)), "clamped"
    return parsed, None


def _read_env_value_from_file(path: Optional[Path], key: str) -> Optional[str]:
    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        return None
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            if k.strip() == key:
                return v.strip()
    except Exception:
        return None
    return None


def normalize_validation_num_ctx_mode(raw_mode: Optional[str]) -> str:
    mode = str(raw_mode or "validated_default").strip().lower()
    if mode not in VALIDATION_NUM_CTX_MODES:
        return "validated_default"
    return mode


def resolve_runtime_num_ctx(
    *,
    process_env: Optional[Mapping[str, str]] = None,
    dotenv_paths: Optional[Sequence[Path]] = None,
    minimum_value: int = 1,
    fallback_default: Optional[int] = RUNTIME_NUM_CTX_DEFAULT,
) -> dict[str, Any]:
    env = process_env or os.environ
    raw_process = str(env.get("OLLAMA_NUM_CTX") or "").strip()
    if raw_process:
        value, status = _coerce_ctx_value(raw_process, minimum_value=minimum_value)
        if value is not None:
            return {
                "mode": "runtime",
                "effective_num_ctx": value,
                "source": "process_env:OLLAMA_NUM_CTX",
                "process_raw": raw_process,
                "dotenv_raw": "",
                "note": "clamped_to_minimum" if status == "clamped" else "",
            }
        if fallback_default is not None:
            fallback_value = max(minimum_value, int(fallback_default))
            return {
                "mode": "runtime",
                "effective_num_ctx": fallback_value,
                "source": f"default ({fallback_value})",
                "process_raw": raw_process,
                "dotenv_raw": "",
                "note": "invalid_process_env_fell_back_to_default",
            }
        return {
            "mode": "runtime",
            "effective_num_ctx": None,
            "source": "unset",
            "process_raw": raw_process,
            "dotenv_raw": "",
            "note": "invalid_process_env_ignored",
        }

    dotenv_paths = dotenv_paths or ()
    for path in dotenv_paths:
        raw_dotenv = str(_read_env_value_from_file(path, "OLLAMA_NUM_CTX") or "").strip()
        if not raw_dotenv:
            continue
        value, status = _coerce_ctx_value(raw_dotenv, minimum_value=minimum_value)
        if value is not None:
            return {
                "mode": "runtime",
                "effective_num_ctx": value,
                "source": f"dotenv:{Path(path).name}:OLLAMA_NUM_CTX",
                "process_raw": "",
                "dotenv_raw": raw_dotenv,
                "note": "clamped_to_minimum" if status == "clamped" else "",
            }
        if fallback_default is not None:
            fallback_value = max(minimum_value, int(fallback_default))
            return {
                "mode": "runtime",
                "effective_num_ctx": fallback_value,
                "source": f"default ({fallback_value})",
                "process_raw": "",
                "dotenv_raw": raw_dotenv,
                "note": f"invalid_dotenv_value_in_{Path(path).name}_fell_back_to_default",
            }
        return {
            "mode": "runtime",
            "effective_num_ctx": None,
            "source": "unset",
            "process_raw": "",
            "dotenv_raw": raw_dotenv,
            "note": f"invalid_dotenv_value_in_{Path(path).name}_ignored",
        }

    if fallback_default is not None:
        fallback_value = max(minimum_value, int(fallback_default))
        return {
            "mode": "runtime",
            "effective_num_ctx": fallback_value,
            "source": f"default ({fallback_value})",
            "process_raw": "",
            "dotenv_raw": "",
            "note": "default_applied",
        }

    return {
        "mode": "runtime",
        "effective_num_ctx": None,
        "source": "unset",
        "process_raw": "",
        "dotenv_raw": "",
        "note": "unset",
    }


def resolve_validation_num_ctx(
    *,
    mode: Optional[str],
    validation_override_raw: Optional[str],
    process_env: Optional[Mapping[str, str]] = None,
    passthrough_env_paths: Optional[Sequence[Path]] = None,
    baseline_default: int = VALIDATION_NUM_CTX_BASELINE,
    minimum_value: int = 1024,
) -> dict[str, Any]:
    resolved_mode = normalize_validation_num_ctx_mode(mode)
    baseline_value = max(int(minimum_value), int(baseline_default))
    override_raw = str(validation_override_raw or "").strip()
    env = process_env or os.environ
    process_raw = str(env.get("OLLAMA_NUM_CTX") or "").strip()

    if resolved_mode == "override":
        if override_raw:
            override_value, status = _coerce_ctx_value(override_raw, minimum_value=minimum_value)
            if override_value is not None:
                return {
                    "mode": resolved_mode,
                    "effective_num_ctx": override_value,
                    "source": "validation_override:VALIDATION_OLLAMA_NUM_CTX",
                    "process_raw": process_raw,
                    "override_raw": override_raw,
                    "passthrough_source": "",
                    "note": "clamped_to_minimum" if status == "clamped" else "",
                }
            return {
                "mode": resolved_mode,
                "effective_num_ctx": baseline_value,
                "source": f"default ({baseline_value})",
                "process_raw": process_raw,
                "override_raw": override_raw,
                "passthrough_source": "",
                "note": "invalid_validation_override_fell_back_to_baseline",
            }
        return {
            "mode": resolved_mode,
            "effective_num_ctx": baseline_value,
            "source": f"default ({baseline_value})",
            "process_raw": process_raw,
            "override_raw": "",
            "passthrough_source": "",
            "note": "override_mode_without_validation_override_fell_back_to_baseline",
        }

    if resolved_mode == "passthrough":
        runtime = resolve_runtime_num_ctx(
            process_env=env,
            dotenv_paths=passthrough_env_paths,
            minimum_value=minimum_value,
            fallback_default=None,
        )
        runtime_effective = runtime.get("effective_num_ctx")
        if isinstance(runtime_effective, int) and runtime_effective > 0:
            return {
                "mode": resolved_mode,
                "effective_num_ctx": runtime_effective,
                "source": f"passthrough:{runtime.get('source')}",
                "process_raw": process_raw,
                "override_raw": override_raw,
                "passthrough_source": str(runtime.get("source") or ""),
                "note": str(runtime.get("note") or ""),
            }
        return {
            "mode": resolved_mode,
            "effective_num_ctx": baseline_value,
            "source": f"default ({baseline_value})",
            "process_raw": process_raw,
            "override_raw": override_raw,
            "passthrough_source": str(runtime.get("source") or ""),
            "note": "passthrough_unset_fell_back_to_baseline",
        }

    note = ""
    if override_raw:
        note = "validation_override_ignored_unless_mode_is_override"
    return {
        "mode": "validated_default",
        "effective_num_ctx": baseline_value,
        "source": f"default ({baseline_value})",
        "process_raw": process_raw,
        "override_raw": override_raw,
        "passthrough_source": "",
        "note": note,
    }
