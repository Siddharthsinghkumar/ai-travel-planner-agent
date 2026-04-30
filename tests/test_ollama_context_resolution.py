from pathlib import Path

from core.ollama_context import (
    RUNTIME_NUM_CTX_DEFAULT,
    normalize_validation_num_ctx_mode,
    resolve_runtime_num_ctx,
    resolve_validation_num_ctx,
)


def test_runtime_num_ctx_prefers_process_env_over_dotenv(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_NUM_CTX=8192\n", encoding="utf-8")
    resolution = resolve_runtime_num_ctx(
        process_env={"OLLAMA_NUM_CTX": "6144"},
        dotenv_paths=[env_file],
        minimum_value=1,
        fallback_default=None,
    )
    assert resolution["effective_num_ctx"] == 6144
    assert resolution["source"] == "process_env:OLLAMA_NUM_CTX"


def test_runtime_num_ctx_uses_dotenv_when_process_missing(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_NUM_CTX=5120\n", encoding="utf-8")
    resolution = resolve_runtime_num_ctx(
        process_env={},
        dotenv_paths=[env_file],
        minimum_value=1,
        fallback_default=None,
    )
    assert resolution["effective_num_ctx"] == 5120
    assert resolution["source"] == "dotenv:.env:OLLAMA_NUM_CTX"


def test_runtime_num_ctx_defaults_to_4096_when_unset():
    resolution = resolve_runtime_num_ctx(
        process_env={},
        dotenv_paths=[],
        minimum_value=1,
    )
    assert resolution["effective_num_ctx"] == RUNTIME_NUM_CTX_DEFAULT
    assert resolution["source"] == f"default ({RUNTIME_NUM_CTX_DEFAULT})"


def test_validation_default_mode_forces_baseline_even_with_override_present():
    resolution = resolve_validation_num_ctx(
        mode="validated_default",
        validation_override_raw="12288",
        process_env={"OLLAMA_NUM_CTX": "8192"},
        passthrough_env_paths=None,
        baseline_default=4096,
        minimum_value=1024,
    )
    assert resolution["mode"] == "validated_default"
    assert resolution["effective_num_ctx"] == 4096
    assert resolution["source"] == "default (4096)"
    assert "ignored" in str(resolution.get("note") or "")


def test_validation_passthrough_mode_uses_runtime_resolution():
    resolution = resolve_validation_num_ctx(
        mode="passthrough",
        validation_override_raw="",
        process_env={"OLLAMA_NUM_CTX": "8192"},
        passthrough_env_paths=None,
        baseline_default=4096,
        minimum_value=1024,
    )
    assert resolution["mode"] == "passthrough"
    assert resolution["effective_num_ctx"] == 8192
    assert str(resolution["source"]).startswith("passthrough:process_env")


def test_validation_override_mode_requires_explicit_override_value():
    resolution = resolve_validation_num_ctx(
        mode="override",
        validation_override_raw="",
        process_env={"OLLAMA_NUM_CTX": "8192"},
        passthrough_env_paths=None,
        baseline_default=4096,
        minimum_value=1024,
    )
    assert resolution["mode"] == "override"
    assert resolution["effective_num_ctx"] == 4096
    assert resolution["source"] == "default (4096)"


def test_normalize_validation_mode_defaults_to_validated_default():
    assert normalize_validation_num_ctx_mode("unexpected_mode") == "validated_default"
