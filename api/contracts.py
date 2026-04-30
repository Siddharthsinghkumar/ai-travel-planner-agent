"""Authoritative API contracts for frontend-consumed endpoints.

These models back FastAPI OpenAPI generation and are consumed by the
frontend contract sync pipeline in scripts/sync_frontend_contract.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict


LLMMode = Literal["ollama_only", "cloud_only", "cloud_first", "ollama_first"]


class LLMOptionsDefaults(BaseModel):
    llm_mode: LLMMode
    cloud_provider: Optional[str] = None


class ProviderStatusEntry(BaseModel):
    configured: bool
    initialized: bool
    usable: bool
    init_reason: Optional[str] = None


class BackendAvailability(BaseModel):
    cloud: bool
    ollama: bool


class LLMOptionsConfigAuthority(BaseModel):
    model_config = ConfigDict(extra="allow")

    llm_mode: Dict[str, Any]
    cloud_provider_chain: Dict[str, Any]
    mode_dependency: Dict[str, Any]
    effective_timeouts: Dict[str, Any]
    timeout_ownership: Dict[str, Any]
    deprecated_env_active: List[str] = []


class LLMOptionsResponseContract(BaseModel):
    llm_modes: List[LLMMode]
    cloud_providers: List[str]
    defaults: LLMOptionsDefaults
    provider_status: Dict[str, ProviderStatusEntry]
    usable_cloud_providers: List[str]
    cloud_usable: bool
    cloud_enabled_by_config: bool
    provider_switch_enabled: bool
    effective_default_provider: Optional[str] = None
    effective_mode: LLMMode
    backend_availability: BackendAvailability
    config_authority: LLMOptionsConfigAuthority


class VersionResponseContract(BaseModel):
    git_commit: str
    file_mtime: float
