from core.llm_mode import (
    get_cloud_provider_chain_resolution,
    get_llm_mode_default,
    get_llm_mode_resolution,
    get_mode_dependency_map,
)


def test_hybrid_mode_uses_cloud_first_when_priority_cloud_first(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "hybrid")
    monkeypatch.setenv("LLM_PRIORITY", "cloud-first")
    assert get_llm_mode_default() == "cloud_first"


def test_hybrid_mode_uses_ollama_first_when_priority_local_first(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "hybrid")
    monkeypatch.setenv("LLM_PRIORITY", "local-first")
    assert get_llm_mode_default() == "ollama_first"


def test_canonical_mode_ignores_legacy_priority(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "cloud_only")
    monkeypatch.setenv("LLM_PRIORITY", "local-first")
    assert get_llm_mode_default() == "cloud_only"


def test_priority_only_mode_resolution_remains_legacy_compatible(monkeypatch):
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.setenv("LLM_PRIORITY", "cloud-first")
    resolution = get_llm_mode_resolution()
    assert resolution["mode"] == "cloud_first"
    assert resolution["source"] == "LLM_PRIORITY"
    assert resolution["legacy_priority_used"] is True


def test_cloud_provider_chain_resolution_prefers_chain_over_single_provider(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER_CHAIN", "gemini,openai")
    monkeypatch.setenv("CLOUD_PROVIDER", "openai")
    resolution = get_cloud_provider_chain_resolution()
    assert resolution["providers"] == ["gemini", "openai"]
    assert resolution["source"] == "CLOUD_PROVIDER_CHAIN"


def test_mode_dependency_marks_cloud_knobs_ignored_in_ollama_only():
    mapping = get_mode_dependency_map("ollama_only")
    assert mapping["mode"] == "ollama_only"
    assert "CLOUD_PROVIDER_CHAIN" in mapping["ignored_for_routing"]
    assert "USE_CLOUD_LLM" in mapping["ignored_for_routing"]
