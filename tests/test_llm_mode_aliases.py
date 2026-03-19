from core.llm_mode import get_llm_mode_default


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
