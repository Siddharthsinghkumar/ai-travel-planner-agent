from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Optional, Tuple
from core.env_config import get_env_str, parse_csv_env

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


def _configured_cloud_provider_chain_from_env() -> list[str]:
    chain = [part.lower() for part in parse_csv_env("CLOUD_PROVIDER_CHAIN")]
    if chain:
        return chain

    default_provider = (get_env_str("CLOUD_PROVIDER", DEFAULT_CLOUD_PROVIDER) or DEFAULT_CLOUD_PROVIDER).strip().lower()
    return [default_provider] if default_provider else [DEFAULT_CLOUD_PROVIDER]


def get_configured_cloud_providers() -> list[str]:
    """Return provider names configured via environment (order-preserving, deduplicated)."""
    seen: set[str] = set()
    providers: list[str] = []
    for provider in _configured_cloud_provider_chain_from_env():
        if provider not in seen:
            seen.add(provider)
            providers.append(provider)
    return providers


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


def get_llm_mode_default() -> str:
    raw_mode = get_env_str("LLM_MODE", "hybrid")
    raw_priority = get_env_str("LLM_PRIORITY", "local-first")
    normalized = normalize_llm_mode(raw_mode, raw_priority)
    return normalized or LLM_MODE_OLLAMA_FIRST


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
