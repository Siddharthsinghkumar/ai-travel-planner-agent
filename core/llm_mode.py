from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Dict, Optional, Tuple
from core.env_config import get_env_str, parse_csv_env, is_env_set

LLM_MODE_OLLAMA_ONLY = "ollama_only"
LLM_MODE_CLOUD_ONLY = "cloud_only"
LLM_MODE_CLOUD_FIRST = "cloud_first"
LLM_MODE_OLLAMA_FIRST = "ollama_first"

VALID_LLM_MODES = (
    LLM_MODE_OLLAMA_ONLY,
    LLM_MODE_CLOUD_ONLY,
    LLM_MODE_CLOUD_FIRST,
    LLM_MODE_OLLAMA_FIRST,
)

_LEGACY_MODE_ALIASES = {
    "local": LLM_MODE_OLLAMA_ONLY,
    "cloud": LLM_MODE_CLOUD_ONLY,
}

_llm_mode_override_ctx: ContextVar[Optional[str]] = ContextVar("llm_mode_override", default=None)
_cloud_provider_override_ctx: ContextVar[Optional[str]] = ContextVar("cloud_provider_override", default=None)


DEFAULT_CLOUD_PROVIDER = "gemini"


def _dedupe_preserve_order(items: list[str], *, lower: bool = False) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = (item or "").strip()
        normalized = cleaned.lower() if lower else cleaned
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized if lower else cleaned)
    return out


def get_cloud_provider_chain_resolution() -> Dict[str, Any]:
    """
    Resolve configured cloud-provider order with explicit source metadata.

    Precedence:
    1) CLOUD_PROVIDER_CHAIN (canonical ordered list)
    2) CLOUD_PROVIDER (single-provider fallback)
    3) default ("gemini")
    """
    raw_chain = parse_csv_env("CLOUD_PROVIDER_CHAIN")
    if raw_chain:
        providers = _dedupe_preserve_order(raw_chain, lower=True)
        return {
            "providers": providers or [DEFAULT_CLOUD_PROVIDER],
            "source": "CLOUD_PROVIDER_CHAIN",
            "raw_chain": raw_chain,
            "raw_default_provider": get_env_str("CLOUD_PROVIDER", None),
        }

    default_provider = (get_env_str("CLOUD_PROVIDER", DEFAULT_CLOUD_PROVIDER) or DEFAULT_CLOUD_PROVIDER).strip().lower()
    return {
        "providers": [default_provider] if default_provider else [DEFAULT_CLOUD_PROVIDER],
        "source": "CLOUD_PROVIDER" if is_env_set("CLOUD_PROVIDER") else "default",
        "raw_chain": [],
        "raw_default_provider": default_provider,
    }


def get_configured_cloud_providers() -> list[str]:
    """Return provider names configured via environment (order-preserving, deduplicated)."""
    resolution = get_cloud_provider_chain_resolution()
    return list(resolution["providers"])


def get_default_cloud_provider() -> str:
    providers = get_configured_cloud_providers()
    return providers[0] if providers else DEFAULT_CLOUD_PROVIDER


def normalize_llm_mode(mode: Optional[str], priority: Optional[str] = None) -> Optional[str]:
    if mode is None:
        return None

    normalized = mode.strip().lower()
    if not normalized:
        return None

    if normalized in VALID_LLM_MODES:
        return normalized

    if normalized == "hybrid":
        normalized_priority = (priority or get_env_str("LLM_PRIORITY", "local-first")).strip().lower()
        return LLM_MODE_CLOUD_FIRST if normalized_priority == "cloud-first" else LLM_MODE_OLLAMA_FIRST

    aliased = _LEGACY_MODE_ALIASES.get(normalized)
    if aliased:
        return aliased

    allowed = ", ".join(VALID_LLM_MODES)
    raise ValueError(f"Invalid llm_mode '{mode}'. Allowed values: {allowed}")


def normalize_cloud_provider(provider: Optional[str]) -> Optional[str]:
    if provider is None:
        return None

    normalized = provider.strip().lower()
    if not normalized:
        return None

    allowed = set(get_configured_cloud_providers())
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed)) if allowed else "(none configured)"
        raise ValueError(
            f"Invalid cloud_provider '{provider}'. Configured providers: {allowed_text}"
        )
    return normalized


def get_effective_cloud_provider() -> Optional[str]:
    override = _cloud_provider_override_ctx.get()
    if override:
        return override
    return get_default_cloud_provider()


def get_llm_mode_resolution() -> Dict[str, Any]:
    """
    Resolve effective mode with explicit authority/source metadata.

    Precedence:
    1) LLM_MODE (canonical)
    2) LLM_PRIORITY (legacy compatibility only when LLM_MODE is unset)
    3) default (ollama_first)
    """
    raw_mode = get_env_str("LLM_MODE", None)
    raw_priority = get_env_str("LLM_PRIORITY", None)
    source = "default"
    notes: list[str] = []
    legacy_priority_used = False

    if raw_mode and raw_mode.strip():
        source = "LLM_MODE"
        resolved = normalize_llm_mode(raw_mode, raw_priority)
        if raw_mode.strip().lower() == "hybrid":
            legacy_priority_used = True
            notes.append("legacy_hybrid_mode")
    elif raw_priority and raw_priority.strip():
        source = "LLM_PRIORITY"
        legacy_priority_used = True
        resolved = normalize_llm_mode("hybrid", raw_priority)
        notes.append("legacy_priority_without_llm_mode")
    else:
        resolved = LLM_MODE_OLLAMA_FIRST
        notes.append("default_ollama_first")

    return {
        "mode": resolved or LLM_MODE_OLLAMA_FIRST,
        "source": source,
        "raw_mode": raw_mode,
        "raw_priority": raw_priority,
        "legacy_priority_used": legacy_priority_used,
        "notes": notes,
    }


def get_llm_mode_default() -> str:
    return str(get_llm_mode_resolution()["mode"])


def get_mode_dependency_map(mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Report mode-scoped env authority to make routing behavior operator-friendly.
    This map is informational and does not enforce behavior by itself.
    """
    resolved_mode = normalize_llm_mode(mode) if mode else get_llm_mode_default()
    resolved_mode = resolved_mode or LLM_MODE_OLLAMA_FIRST

    common = [
        "LLM_MODE",
        "PLANNER_LLM_TIMEOUT",
        "ROUTER_TIMEOUT",
        "PLANNER_STREAM_INIT_TIMEOUT",
        "PLANNER_STREAM_TOTAL_TIMEOUT",
    ]
    cloud = [
        "USE_CLOUD_LLM",
        "CLOUD_PROVIDER_CHAIN",
        "CLOUD_PROVIDER",
        "CLOUD_LLM_MODEL",
        "CLOUD_LLM_TIMEOUT",
        "CLOUD_LLM_STREAM_CHUNK_TIMEOUT",
        "CLOUD_ONLY_ALLOW_PROVIDER_FALLBACK",
    ]
    ollama = [
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT",
        "OLLAMA_THINKING_MODE",
        "OLLAMA_BREAKER_FAILURE_THRESHOLD",
        "OLLAMA_BREAKER_RECOVERY_TIMEOUT",
        "LOCAL_LLM_TIMEOUT",
        "OLLAMA_ROUTER_PROBE_TIMEOUT",
    ]
    compatibility = [
        "LLM_PRIORITY",
        "ENABLE_LEGACY_ASYNC_LLM_CLIENT",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLOUD_BASE_URL",
    ]

    authoritative = list(common)
    ignored_for_routing: list[str] = []
    if resolved_mode == LLM_MODE_OLLAMA_ONLY:
        authoritative.extend(ollama)
        ignored_for_routing = list(cloud)
    elif resolved_mode == LLM_MODE_CLOUD_ONLY:
        authoritative.extend(cloud)
        ignored_for_routing = list(ollama)
    else:
        authoritative.extend(cloud)
        authoritative.extend(ollama)

    return {
        "mode": resolved_mode,
        "authoritative": _dedupe_preserve_order(authoritative),
        "ignored_for_routing": _dedupe_preserve_order(ignored_for_routing),
        "compatibility": _dedupe_preserve_order(compatibility),
    }


async def get_llm_mode_and_priority() -> Tuple[str, str]:
    """
    Returns canonical mode plus derived legacy priority for compatibility.

    Canonical mode: ollama_only | cloud_only | cloud_first | ollama_first
    Priority: cloud-first | local-first (derived from mode)
    """
    override_mode = _llm_mode_override_ctx.get()
    mode = normalize_llm_mode(override_mode) if override_mode else get_llm_mode_default()

    if mode == LLM_MODE_CLOUD_FIRST:
        priority = "cloud-first"
    elif mode == LLM_MODE_OLLAMA_FIRST:
        priority = "local-first"
    elif mode == LLM_MODE_CLOUD_ONLY:
        priority = "cloud-first"
    else:
        priority = "local-first"

    return mode, priority


def set_llm_mode_override(mode: Optional[str]) -> Optional[Token]:
    normalized = normalize_llm_mode(mode) if mode else None
    return _llm_mode_override_ctx.set(normalized)


def set_cloud_provider_override(provider: Optional[str]) -> Optional[Token]:
    normalized = normalize_cloud_provider(provider) if provider else None
    return _cloud_provider_override_ctx.set(normalized)


def reset_llm_mode_override(token: Optional[Token]) -> None:
    if token is not None:
        _llm_mode_override_ctx.reset(token)


def reset_cloud_provider_override(token: Optional[Token]) -> None:
    if token is not None:
        _cloud_provider_override_ctx.reset(token)


@contextmanager
def llm_routing_context(llm_mode: Optional[str] = None, cloud_provider: Optional[str] = None):
    mode_token: Optional[Token] = None
    provider_token: Optional[Token] = None
    try:
        if llm_mode is not None:
            mode_token = set_llm_mode_override(llm_mode)
        if cloud_provider is not None:
            provider_token = set_cloud_provider_override(cloud_provider)
        yield
    finally:
        reset_cloud_provider_override(provider_token)
        reset_llm_mode_override(mode_token)
