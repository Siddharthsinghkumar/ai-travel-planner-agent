from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Union


LEGACY_SCENARIO_KEY = "__validation_scenario"
LEGACY_EXPECTATIONS_KEY = "__validation_expectations"
LEGACY_CASE_NAME_KEY = "__validation_case_name"


@dataclass(frozen=True)
class FrontendValidationContext:
    """Validation context passed from orchestrator to frontend validator."""

    scenario: str = ""
    expectations: Dict[str, Any] = field(default_factory=dict)
    case_name: str = ""


@dataclass(frozen=True)
class FrontendValidationRequest:
    """Explicit frontend validation request contract."""

    payload: Dict[str, Any]
    context: FrontendValidationContext = field(default_factory=FrontendValidationContext)


def _normalize_context(
    *,
    scenario: str,
    expectations: Mapping[str, Any] | None,
    case_name: str,
) -> FrontendValidationContext:
    normalized_expectations = dict(expectations) if isinstance(expectations, Mapping) else {}
    return FrontendValidationContext(
        scenario=str(scenario or "").strip().lower(),
        expectations=normalized_expectations,
        case_name=str(case_name or "").strip(),
    )


def _strip_legacy_contract_keys(payload: Mapping[str, Any]) -> Dict[str, Any]:
    clean = dict(payload)
    clean.pop(LEGACY_SCENARIO_KEY, None)
    clean.pop(LEGACY_EXPECTATIONS_KEY, None)
    clean.pop(LEGACY_CASE_NAME_KEY, None)
    return clean


def coerce_frontend_validation_request(
    request: Union[FrontendValidationRequest, Mapping[str, Any]],
) -> FrontendValidationRequest:
    """
    Normalize legacy payload calls and modern typed requests into one contract.

    Legacy compatibility:
    - payload["__validation_scenario"]
    - payload["__validation_expectations"]
    - payload["__validation_case_name"]
    """
    if isinstance(request, FrontendValidationRequest):
        payload = _strip_legacy_contract_keys(request.payload)
        context = _normalize_context(
            scenario=request.context.scenario,
            expectations=request.context.expectations,
            case_name=request.context.case_name,
        )
        return FrontendValidationRequest(payload=payload, context=context)

    payload = dict(request)
    context = _normalize_context(
        scenario=str(payload.get(LEGACY_SCENARIO_KEY) or ""),
        expectations=payload.get(LEGACY_EXPECTATIONS_KEY) if isinstance(payload.get(LEGACY_EXPECTATIONS_KEY), Mapping) else None,
        case_name=str(payload.get(LEGACY_CASE_NAME_KEY) or ""),
    )
    return FrontendValidationRequest(
        payload=_strip_legacy_contract_keys(payload),
        context=context,
    )
