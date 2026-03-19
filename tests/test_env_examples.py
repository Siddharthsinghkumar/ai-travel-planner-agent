from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def test_root_env_example_contains_canonical_backend_vars_only():
    env_example = REPO_ROOT / ".env.example"
    assert env_example.exists(), ".env.example must exist at repo root"

    keys = _parse_env_keys(env_example)
    assert "LLM_MODE" in keys
    assert "USE_CLOUD_LLM" in keys
    assert "CLOUD_PROVIDER_CHAIN" in keys
    assert "CLOUD_PROVIDER" in keys
    assert "OLLAMA_BASE_URL" in keys
    assert "OLLAMA_MODEL" in keys
    assert "OPENAI_KEY_1" in keys
    assert "GEMINI_KEY_1" in keys

    deprecated_or_dead = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLOUD_BASE_URL",
        "LLM_PRIORITY",
        "LLM_PREWARM",
        "PLANNER_STREAMING_ENABLED",
        "VITE_TOTAL_LLM_TIMEOUT_MS",
    }
    assert keys.isdisjoint(deprecated_or_dead)


def test_frontend_env_example_contains_only_vite_keys():
    env_example = REPO_ROOT / "frontend" / ".env.example"
    assert env_example.exists(), "frontend/.env.example must exist"

    keys = _parse_env_keys(env_example)
    assert "VITE_API_BASE_URL" in keys
    assert "VITE_STREAM_SOFT_DELAY_MS" in keys
    assert "VITE_STREAM_HARD_NO_ACTIVITY_MS" in keys
    assert "VITE_UI_MODE" in keys
    assert "VITE_DEBUG_MODE" in keys
    assert all(key.startswith("VITE_") for key in keys)
