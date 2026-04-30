from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse


FeatureId = str
ValidationModeId = str
SoftPassPolicyId = str

MODE_BACKEND_INTERNAL = "backend_internal"
MODE_API_CONTRACT = "api_contract"
MODE_RUNTIME_HEALTH = "runtime_health"
MODE_FRONTEND_FIXTURE_BROWSER = "frontend_fixture_browser"
MODE_FRONTEND_REAL_BACKEND_BROWSER = "frontend_real_backend_browser"
MODE_LIVE_CANARY_BROWSER = "live_canary_browser"

MODE_BUCKET_ORDER: Sequence[ValidationModeId] = (
    MODE_BACKEND_INTERNAL,
    MODE_API_CONTRACT,
    MODE_FRONTEND_FIXTURE_BROWSER,
    MODE_FRONTEND_REAL_BACKEND_BROWSER,
    MODE_LIVE_CANARY_BROWSER,
)

SOFT_PASS_HARD_FAIL_ONLY = "hard_fail_only"
SOFT_PASS_ALLOWED = "soft_pass_allowed"
SOFT_PASS_LIVE_ONLY = "live_only_soft_pass_allowed"

FRONTEND_FIXTURE_LEGACY_ALIASES: Dict[str, str] = {
    "mock_stream_success_one_way": "fixture_stream_one_way",
    "mock_stream_success_round_trip": "fixture_stream_round_trip",
    "mock_stream_success_via_stopover": "fixture_stream_via_stopover",
    "mock_stream_fallback_non_stream": "fixture_stream_fallback_non_stream",
    "mock_degraded_result": "fixture_degraded_result",
    "mock_no_flights": "fixture_no_flights",
    "mock_booking_handoff": "fixture_booking_handoff",
}


@dataclass(frozen=True)
class ValidationMeta:
    scenario: str
    layers: Sequence[str]
    validation_type: str
    features: Sequence[FeatureId] = field(default_factory=tuple)
    mode_bucket: ValidationModeId = MODE_BACKEND_INTERNAL
    soft_pass_policy: SoftPassPolicyId = SOFT_PASS_HARD_FAIL_ONLY
    criticality: str = "core"


@dataclass(frozen=True)
class FrontendRuntimeCase:
    case_name: str
    payload: Dict[str, Any]
    fixture_scenario: str
    expectations: Dict[str, Any]
    features: Sequence[FeatureId]
    mode_tags: Sequence[ValidationModeId] = field(default_factory=tuple)
    criticality: str = "core"
    soft_pass_policy: SoftPassPolicyId = SOFT_PASS_HARD_FAIL_ONLY
    dimensions: Dict[str, str] = field(default_factory=dict)
    ui_assertions: Sequence[str] = field(default_factory=tuple)
    contract_assertions: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class FrontendFixtureScenario:
    name: str
    ask_mode: str
    variant: str
    default_payload: Dict[str, Any]
    initial_bookings: Sequence[Dict[str, Any]] = field(default_factory=tuple)
    initial_alerts: Sequence[Dict[str, Any]] = field(default_factory=tuple)
    tracking_status: Dict[str, Any] = field(default_factory=dict)
    hold_response: Optional[Dict[str, Any]] = None
    track_response: Optional[Dict[str, Any]] = None
    cancel_response: Optional[Dict[str, Any]] = None


# Canonical feature list used by coverage reporting.
FEATURE_CATALOG: Sequence[FeatureId] = (
    "ask.non_stream",
    "ask.stream",
    "ask.degraded",
    "ask.no_flights",
    "trip.one_way",
    "trip.round_trip",
    "trip.via_stopover",
    "intent.cheapest",
    "intent.fastest",
    "intent.direct",
    "intent.cabin",
    "intent.vague",
    "intent.typo",
    "intent.relative_date",
    "booking.hold",
    "booking.cancel",
    "booking.list",
    "booking.handoff",
    "booking.hold_local_only",
    "booking.policy.no_google_fallback",
    "booking.navigation.link_visible",
    "booking.navigation.provider_handoff_present",
    "booking.navigation.checkout_ready",
    "booking.navigation.checkout_unavailable",
    "booking.navigation.local_hold_only",
    "booking.navigation.real_provider_browser_proof",
    "seller.ota_diversity",
    "tracking.status",
    "tracking.track_price",
    "tracking.alerts",
    "tracking.alert_ack",
    "jobs.create",
    "jobs.poll",
    "jobs.events",
    "jobs.cancel",
    "frontend.ui.behavior",
    "frontend.ui.fields.cabin",
    "frontend.ui.fields.direct",
    "frontend.ui.fields.return_date",
    "frontend.ui.fields.baggage",
    "frontend.ui.fields.async_toggle",
    "ops.health",
    "ops.version",
    "ops.llm_options",
)


def _meta(
    scenario: str,
    layers: Sequence[str],
    validation_type: str,
    features: Sequence[FeatureId] = (),
    *,
    mode_bucket: ValidationModeId,
    soft_pass_policy: SoftPassPolicyId = SOFT_PASS_HARD_FAIL_ONLY,
    criticality: str = "core",
) -> ValidationMeta:
    return ValidationMeta(
        scenario=scenario,
        layers=layers,
        validation_type=validation_type,
        features=features,
        mode_bucket=mode_bucket,
        soft_pass_policy=soft_pass_policy,
        criticality=criticality,
    )


VALIDATION_META_BY_PREFIX: Dict[str, ValidationMeta] = {
    "pytest_unit": _meta(
        "unit-and-contract-suite",
        ("backend", "api", "runtime"),
        "unit+integration",
        (),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "quick_sync_ask": _meta(
        "one-way-non-stream",
        ("backend", "api", "e2e"),
        "integration",
        ("ask.non_stream", "trip.one_way"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "missing_date_test": _meta(
        "non-stream-defaulting",
        ("backend", "api"),
        "integration",
        ("ask.non_stream", "intent.relative_date"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "nl_relative_date": _meta(
        "one-way-non-stream",
        ("backend", "api", "e2e"),
        "integration",
        ("ask.non_stream", "intent.relative_date"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "misspelled_city": _meta(
        "one-way-non-stream",
        ("backend", "api", "e2e"),
        "integration",
        ("ask.non_stream", "intent.typo"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "round_trip_duration": _meta(
        "round-trip",
        ("backend", "api", "e2e"),
        "integration",
        ("ask.non_stream", "trip.round_trip", "intent.fastest"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "time_pref_morning": _meta(
        "one-way-non-stream",
        ("backend", "api"),
        "integration",
        ("ask.non_stream", "trip.one_way"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "price_cap": _meta(
        "one-way-non-stream",
        ("backend", "api"),
        "integration",
        ("ask.non_stream", "intent.cheapest"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "direct_only": _meta(
        "one-way-non-stream",
        ("backend", "api"),
        "integration",
        ("ask.non_stream", "intent.direct"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "preferred_airline": _meta(
        "one-way-non-stream",
        ("backend", "api"),
        "integration",
        ("ask.non_stream", "trip.one_way"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "layover_limit": _meta(
        "one-way-non-stream",
        ("backend", "api"),
        "integration",
        ("ask.non_stream", "trip.one_way"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "baggage_hand": _meta(
        "one-way-non-stream",
        ("backend", "api"),
        "integration",
        ("ask.non_stream", "trip.one_way"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "stopover_via": _meta(
        "via-stopover",
        ("backend", "api", "e2e"),
        "integration",
        ("ask.non_stream", "trip.via_stopover"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "streaming_test": _meta(
        "streaming-success",
        ("backend", "api", "e2e"),
        "integration",
        ("ask.stream", "trip.one_way"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "streaming_nl_relative": _meta(
        "streaming-success",
        ("backend", "api", "e2e"),
        "integration",
        ("ask.stream", "intent.relative_date"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "health_light": _meta(
        "health-runtime-truth",
        ("api", "runtime"),
        "runtime",
        ("ops.health",),
        mode_bucket=MODE_RUNTIME_HEALTH,
    ),
    "health_deep": _meta(
        "health-runtime-truth",
        ("api", "runtime"),
        "runtime",
        ("ops.health",),
        mode_bucket=MODE_RUNTIME_HEALTH,
    ),
    "health_keys": _meta(
        "health-runtime-truth",
        ("api", "runtime"),
        "runtime",
        ("ops.health",),
        mode_bucket=MODE_RUNTIME_HEALTH,
    ),
    "health_runtime_topology": _meta(
        "health-runtime-truth",
        ("api", "runtime"),
        "runtime",
        ("ops.health",),
        mode_bucket=MODE_RUNTIME_HEALTH,
    ),
    "llm_options": _meta(
        "runtime-topology-options",
        ("api", "runtime"),
        "contract",
        ("ops.llm_options",),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "version_info": _meta(
        "runtime-version-truth",
        ("api", "runtime"),
        "contract",
        ("ops.version",),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "capability_constraints": _meta(
        "non-stream-success",
        ("backend", "api", "e2e"),
        "integration",
        ("ask.non_stream", "trip.one_way"),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "async_parallel": _meta(
        "parallel-requests",
        ("backend", "api"),
        "smoke",
        ("ask.non_stream",),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "eco_flight": _meta(
        "one-way-non-stream",
        ("backend", "api"),
        "integration",
        ("ask.non_stream",),
        mode_bucket=MODE_BACKEND_INTERNAL,
    ),
    "contract_no_flights": _meta(
        "no-flights",
        ("backend", "api"),
        "contract",
        ("ask.no_flights",),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "contract_degraded_stream": _meta(
        "degraded-result",
        ("backend", "api"),
        "contract",
        ("ask.degraded", "ask.stream"),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "contract_booking_bridge": _meta(
        "booking-handoff",
        ("backend", "api", "e2e"),
        "integration",
        (
            "booking.handoff",
            "booking.navigation.provider_handoff_present",
            "booking.navigation.link_visible",
        ),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "contract_jobs_flow": _meta(
        "jobs-flow",
        ("api", "runtime", "e2e"),
        "integration",
        ("jobs.create", "jobs.poll", "jobs.events", "jobs.cancel"),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "contract_hardening_duplicate_guard": _meta(
        "hardening-duplicate-handling",
        ("backend", "api", "runtime"),
        "hardening-contract",
        (),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "contract_hardening_backpressure": _meta(
        "hardening-backpressure",
        ("backend", "api", "runtime"),
        "hardening-contract",
        (),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "contract_hardening_consume_race": _meta(
        "hardening-consume-race",
        ("backend", "api", "runtime"),
        "hardening-contract",
        (),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "contract_hardening_retry_budget": _meta(
        "hardening-retry-budget",
        ("backend", "runtime"),
        "hardening-contract",
        (),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "contract_hardening_key_cooldown": _meta(
        "hardening-key-cooldown-recovery",
        ("backend", "runtime"),
        "hardening-contract",
        (),
        mode_bucket=MODE_API_CONTRACT,
    ),
    "result_machine_integration": _meta(
        "health-runtime-truth",
        ("api", "runtime"),
        "runtime",
        ("ops.health",),
        mode_bucket=MODE_RUNTIME_HEALTH,
    ),
    "result_machine_integration_failed": _meta(
        "health-runtime-truth",
        ("api", "runtime"),
        "runtime",
        ("ops.health",),
        mode_bucket=MODE_RUNTIME_HEALTH,
    ),
    "docker_hosted_smoke": _meta(
        "health-runtime-truth",
        ("api", "runtime"),
        "runtime",
        ("ops.health",),
        mode_bucket=MODE_RUNTIME_HEALTH,
    ),
    "docker_hosted_failed": _meta(
        "health-runtime-truth",
        ("api", "runtime"),
        "runtime",
        ("ops.health",),
        mode_bucket=MODE_RUNTIME_HEALTH,
    ),
    # Fixture-backed browser matrix.
    "frontend_fixture_": _meta(
        "frontend-fixture-matrix",
        ("frontend", "api"),
        "frontend-fixture",
        ("frontend.ui.behavior",),
        mode_bucket=MODE_FRONTEND_FIXTURE_BROWSER,
    ),
    # Real-backend browser matrix.
    "frontend_real_backend_": _meta(
        "frontend-real-backend-matrix",
        ("frontend", "api", "e2e"),
        "frontend-real-backend",
        ("frontend.ui.behavior",),
        mode_bucket=MODE_FRONTEND_REAL_BACKEND_BROWSER,
    ),
    # Live-provider browser canary.
    "frontend_live_canary_": _meta(
        "frontend-live-canary-matrix",
        ("frontend", "api", "e2e"),
        "live-canary",
        (
            "booking.navigation.real_provider_browser_proof",
            "seller.ota_diversity",
        ),
        mode_bucket=MODE_LIVE_CANARY_BROWSER,
        soft_pass_policy=SOFT_PASS_LIVE_ONLY,
        criticality="canary",
    ),
    # Backward compatibility for older frontend case prefixes.
    "frontend_runtime_": _meta(
        "frontend-fixture-matrix",
        ("frontend", "api"),
        "frontend-fixture",
        ("frontend.ui.behavior",),
        mode_bucket=MODE_FRONTEND_FIXTURE_BROWSER,
    ),
    # Non-browser real-provider checks.
    "real_simple_flight": _meta(
        "one-way-non-stream",
        ("backend", "api", "e2e"),
        "live-canary",
        ("ask.non_stream", "trip.one_way"),
        mode_bucket=MODE_LIVE_CANARY_BROWSER,
        soft_pass_policy=SOFT_PASS_LIVE_ONLY,
        criticality="canary",
    ),
    "real_weather_query": _meta(
        "one-way-non-stream",
        ("backend", "api", "e2e"),
        "live-canary",
        ("ask.non_stream",),
        mode_bucket=MODE_LIVE_CANARY_BROWSER,
        soft_pass_policy=SOFT_PASS_LIVE_ONLY,
        criticality="canary",
    ),
    "real_combined_query": _meta(
        "non-stream-success",
        ("backend", "api", "e2e"),
        "live-canary",
        ("ask.non_stream", "intent.direct", "intent.cheapest"),
        mode_bucket=MODE_LIVE_CANARY_BROWSER,
        soft_pass_policy=SOFT_PASS_LIVE_ONLY,
        criticality="canary",
    ),
}


def known_features() -> List[FeatureId]:
    return list(FEATURE_CATALOG)


def known_mode_buckets() -> List[ValidationModeId]:
    return list(MODE_BUCKET_ORDER)


def validation_meta_for_prefix(base_name: str) -> ValidationMeta:
    for prefix, meta in VALIDATION_META_BY_PREFIX.items():
        if str(base_name).startswith(prefix):
            return meta
    return ValidationMeta(
        "uncategorized",
        ("uncategorized",),
        "uncategorized",
        (),
        mode_bucket=MODE_BACKEND_INTERNAL,
        soft_pass_policy=SOFT_PASS_HARD_FAIL_ONLY,
        criticality="extended",
    )


def validation_meta_prefix_map() -> Dict[str, Dict[str, Any]]:
    return {
        prefix: {
            "scenario": meta.scenario,
            "layers": list(meta.layers),
            "validation_type": meta.validation_type,
            "features": list(meta.features),
            "mode_bucket": meta.mode_bucket,
            "soft_pass_policy": meta.soft_pass_policy,
            "criticality": meta.criticality,
        }
        for prefix, meta in VALIDATION_META_BY_PREFIX.items()
    }


def resolve_frontend_fixture_scenario_name(
    scenario: str,
    *,
    fixture_catalog: Optional[Dict[str, FrontendFixtureScenario]] = None,
) -> str:
    """Resolve scenario aliases to canonical fixture names when available."""
    name = str(scenario or "").strip().lower()
    if not name:
        return ""
    normalized = FRONTEND_FIXTURE_LEGACY_ALIASES.get(name, name)
    catalog = fixture_catalog if isinstance(fixture_catalog, dict) else frontend_fixture_scenarios()
    return normalized if normalized in catalog else ""


def classify_frontend_endpoint_request(method: str, url: str) -> str:
    """Classify API calls seen during frontend validation."""
    method_u = str(method or "").upper()
    parsed = urlparse(str(url or ""))
    path = parsed.path or ""
    query = parse_qs(parsed.query or "")

    if path.endswith("/ask"):
        if "async_job" in query and str(query.get("async_job", [""])[0]).lower() in {"true", "1", "yes"}:
            return "ask_async"
        if str(query.get("stream", [""])[0]).lower() in {"true", "1", "yes"}:
            return "ask_stream"
        return "ask_non_stream"
    if method_u == "GET" and path.endswith("/bookings"):
        return "bookings_list"
    if method_u == "POST" and path.endswith("/booking/hold"):
        return "booking_hold"
    if method_u == "POST" and path.endswith("/booking/cancel"):
        return "booking_cancel"
    if method_u == "POST" and path.endswith("/booking/track-price"):
        return "booking_track_price"
    if method_u == "GET" and path.endswith("/price-tracking/status"):
        return "price_tracking_status"
    if method_u == "GET" and path.endswith("/price-tracking/alerts"):
        return "price_tracking_alerts"
    if method_u == "POST" and re.search(r"/price-tracking/alerts/[^/]+/ack$", path):
        return "price_tracking_alert_ack"
    if method_u == "GET" and re.search(r"/jobs/[^/]+/events$", path):
        return "jobs_events"
    if method_u == "GET" and re.search(r"/jobs/[^/]+$", path):
        return "jobs_poll"
    if method_u == "POST" and re.search(r"/jobs/[^/]+/cancel$", path):
        return "jobs_cancel"
    if method_u == "GET" and path.endswith("/llm/options"):
        return "llm_options"
    if method_u == "GET" and path.endswith("/version"):
        return "version"
    if method_u == "GET" and path.endswith("/health"):
        return "health"
    return ""


def frontend_fixture_scenarios() -> Dict[str, FrontendFixtureScenario]:
    return {
        "fixture_stream_one_way": FrontendFixtureScenario(
            name="fixture_stream_one_way",
            ask_mode="stream",
            variant="one_way",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Find flights from Delhi to Mumbai on 2026-07-18",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_non_stream_one_way": FrontendFixtureScenario(
            name="fixture_non_stream_one_way",
            ask_mode="non_stream",
            variant="one_way",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Cheapest flight Delhi to Mumbai on 2026-07-18",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_stream_round_trip": FrontendFixtureScenario(
            name="fixture_stream_round_trip",
            ask_mode="stream",
            variant="round_trip",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "round-trip",
                "user_query": "Round-trip Delhi to Mumbai on 2026-07-18 returning in 3 days",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_stream_via_stopover": FrontendFixtureScenario(
            name="fixture_stream_via_stopover",
            ask_mode="stream",
            variant="via_stopover",
            default_payload={
                "origin": "DEL",
                "destination": "MAA",
                "date": "2026-07-18",
                "trip_type": "via-stopover",
                "user_query": "Delhi to Chennai via Bangalore on 2026-07-18",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_stream_fallback_non_stream": FrontendFixtureScenario(
            name="fixture_stream_fallback_non_stream",
            ask_mode="stream_fallback_non_stream",
            variant="one_way",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Fallback validation route from Delhi to Mumbai",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_degraded_result": FrontendFixtureScenario(
            name="fixture_degraded_result",
            ask_mode="stream",
            variant="degraded",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Show degraded but usable result",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_no_flights": FrontendFixtureScenario(
            name="fixture_no_flights",
            ask_mode="stream_fallback_no_flights",
            variant="no_flights",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "No flights for this route",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_booking_handoff": FrontendFixtureScenario(
            name="fixture_booking_handoff",
            ask_mode="stream",
            variant="booking_handoff",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Show booking handoff readiness",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_booking_local_only": FrontendFixtureScenario(
            name="fixture_booking_local_only",
            ask_mode="stream",
            variant="booking_local_only",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Hold flow where provider checkout is unavailable",
            },
            initial_bookings=(
                {
                    "id": 9001,
                    "status": "HELD",
                    "checkout_ready": False,
                    "checkout_status": "provider_handoff_unavailable",
                    "hold_outcome": "held_local_only",
                    "handoff_url": None,
                    "flight": {
                        "airline": "MockAir",
                        "flight_no": "MK101",
                        "origin": "DEL",
                        "destination": "BOM",
                        "departure_time": "09:20",
                        "arrival_time": "11:35",
                        "date": "2026-07-18",
                        "price_inr": 6200,
                    },
                },
            ),
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_booking_hold_cancel": FrontendFixtureScenario(
            name="fixture_booking_hold_cancel",
            ask_mode="stream",
            variant="one_way",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Book and cancel a held option",
            },
            hold_response={
                "action": "hold",
                "success": True,
                "checkout_ready": False,
                "hold_outcome": "held_local_only",
                "message": "Held locally. Provider checkout is unavailable.",
            },
            cancel_response={"action": "cancel", "success": True, "message": "Booking cancelled."},
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-09T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_tracking_alerts": FrontendFixtureScenario(
            name="fixture_tracking_alerts",
            ask_mode="stream",
            variant="one_way",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Track this route and inspect alerts",
            },
            initial_alerts=(
                {
                    "alert_id": 501,
                    "booking_id": 41,
                    "origin": "DEL",
                    "destination": "BOM",
                    "travel_date": "2026-07-18",
                    "held_price_inr": 7100,
                    "new_price_inr": 6400,
                    "drop_pct": 9.9,
                    "new_handoff_url": "/booking/handoff/post/mock-alert-501",
                    "created_at": "2026-04-10T08:30:00Z",
                },
            ),
            track_response={
                "action": "track-price",
                "success": True,
                "monitoring_active": True,
                "message": "Price tracking activated.",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-10T08:30:00Z", "last_alert_count": 1}},
        ),
        "fixture_async_jobs": FrontendFixtureScenario(
            name="fixture_async_jobs",
            ask_mode="async_job",
            variant="one_way",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Run this search as async job",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-10T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_async_jobs_cancel": FrontendFixtureScenario(
            name="fixture_async_jobs_cancel",
            ask_mode="async_job_running",
            variant="one_way",
            default_payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Run this search as async job and cancel",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-10T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_cabin_business_no_match": FrontendFixtureScenario(
            name="fixture_cabin_business_no_match",
            ask_mode="non_stream",
            variant="cabin_no_match",
            default_payload={
                "origin": "MAA",
                "destination": "DEL",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Business class flight from Chennai to Delhi on 2026-07-18",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-10T08:30:00Z", "last_alert_count": 0}},
        ),
        "fixture_direct_truthful": FrontendFixtureScenario(
            name="fixture_direct_truthful",
            ask_mode="non_stream",
            variant="direct_truthful",
            default_payload={
                "origin": "DEL",
                "destination": "GOI",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Direct flights only from Delhi to Goa on 2026-07-18",
            },
            tracking_status={"enabled": True, "status": {"last_completed_at": "2026-04-10T08:30:00Z", "last_alert_count": 0}},
        ),
    }


def _case(
    *,
    case_name: str,
    payload: Dict[str, Any],
    fixture_scenario: str,
    expectations: Dict[str, Any],
    features: Sequence[FeatureId],
    mode_tags: Sequence[ValidationModeId],
    dimensions: Dict[str, str],
    ui_assertions: Sequence[str],
    contract_assertions: Sequence[str],
    criticality: str = "core",
    soft_pass_policy: SoftPassPolicyId = SOFT_PASS_HARD_FAIL_ONLY,
) -> FrontendRuntimeCase:
    return FrontendRuntimeCase(
        case_name=case_name,
        payload=payload,
        fixture_scenario=fixture_scenario,
        expectations=expectations,
        features=features,
        mode_tags=mode_tags,
        criticality=criticality,
        soft_pass_policy=soft_pass_policy,
        dimensions=dimensions,
        ui_assertions=ui_assertions,
        contract_assertions=contract_assertions,
    )


def _frontend_runtime_case_catalog() -> List[FrontendRuntimeCase]:
    cases: List[FrontendRuntimeCase] = [
        # Fixture-backed matrix (safe default).
        _case(
            case_name="frontend_fixture_stream_one_way",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Find a one-way flight from Delhi to Mumbai on 2026-07-18",
            },
            fixture_scenario="fixture_stream_one_way",
            expectations={"expect_stream_request": True},
            features=("ask.stream", "trip.one_way", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "one_way",
                "date_basis": "explicit_date",
                "user_intent": "generic",
            },
            ui_assertions=("proof_overview", "ranked_shortlist", "trip_brief"),
            contract_assertions=("ask.stream",),
        ),
        _case(
            case_name="frontend_fixture_non_stream_one_way",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Cheapest flight from Delhi to Mumbai on 2026-07-18",
            },
            fixture_scenario="fixture_non_stream_one_way",
            expectations={"expect_stream_request": False},
            features=("ask.non_stream", "intent.cheapest", "trip.one_way", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "one_way",
                "date_basis": "explicit_date",
                "user_intent": "cheapest",
            },
            ui_assertions=("proof_overview", "ranked_shortlist", "weather_panel"),
            contract_assertions=("ask.non_stream",),
        ),
        _case(
            case_name="frontend_fixture_round_trip",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "round-trip",
                "return_date": "2026-07-21",
                "user_query": "Round-trip Delhi to Mumbai starting 2026-07-18 and returning 2026-07-21",
            },
            fixture_scenario="fixture_stream_round_trip",
            expectations={"expect_round_trip": True, "expect_stream_request": True},
            features=("ask.stream", "trip.round_trip", "frontend.ui.fields.return_date", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "round_trip",
                "date_basis": "explicit_date",
                "user_intent": "fastest",
            },
            ui_assertions=("return_leg", "proof_overview"),
            contract_assertions=("ask.stream",),
        ),
        _case(
            case_name="frontend_fixture_via_stopover",
            payload={
                "origin": "DEL",
                "destination": "MAA",
                "date": "2026-07-18",
                "trip_type": "via-stopover",
                "user_query": "Delhi to Chennai via Bangalore on 2026-07-18",
            },
            fixture_scenario="fixture_stream_via_stopover",
            expectations={"expect_via_stopover": True},
            features=("ask.stream", "trip.via_stopover", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "route_shape": "multi_leg",
                "trip_type": "via_stopover",
                "date_basis": "explicit_date",
                "user_intent": "stopover",
            },
            ui_assertions=("multicity_itinerary",),
            contract_assertions=("ask.stream",),
        ),
        _case(
            case_name="frontend_fixture_degraded",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Show degraded but usable planning output",
            },
            fixture_scenario="fixture_degraded_result",
            expectations={"expect_degraded": True, "require_notice_contains": "partial result"},
            features=("ask.degraded", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "one_way",
                "date_basis": "explicit_date",
                "user_intent": "degraded_recovery",
            },
            ui_assertions=("degraded_notice", "proof_overview"),
            contract_assertions=("result_status.degraded",),
        ),
        _case(
            case_name="frontend_fixture_no_flights",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Show no flights result path",
            },
            fixture_scenario="fixture_no_flights",
            expectations={"expect_no_flights": True},
            features=("ask.no_flights", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "one_way",
                "date_basis": "explicit_date",
                "user_intent": "no_inventory",
            },
            ui_assertions=("no_flights_notice", "proof_overview"),
            contract_assertions=("failure_reason.no_flights",),
        ),
        _case(
            case_name="frontend_fixture_booking_handoff",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Show booking handoff link behavior",
            },
            fixture_scenario="fixture_booking_handoff",
            expectations={"expect_booking_link": True},
            features=(
                "booking.handoff",
                "booking.policy.no_google_fallback",
                "booking.navigation.link_visible",
                "booking.navigation.provider_handoff_present",
                "frontend.ui.behavior",
            ),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "route_shape": "point_to_point",
                "booking_outcome": "provider_handoff_present",
                "seller_provider_breadth": "single_provider",
            },
            ui_assertions=("booking_link",),
            contract_assertions=("handoff_url",),
        ),
        _case(
            case_name="frontend_fixture_booking_local_only",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Hold locally when provider checkout is unavailable",
            },
            fixture_scenario="fixture_booking_local_only",
            expectations={"require_notice_contains": "checkout link is currently unavailable", "expect_booking_panel": True},
            features=(
                "booking.hold_local_only",
                "booking.policy.no_google_fallback",
                "booking.list",
                "booking.navigation.local_hold_only",
                "booking.navigation.checkout_unavailable",
                "frontend.ui.behavior",
            ),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "booking_outcome": "held_local_only",
                "seller_provider_breadth": "provider_unavailable",
            },
            ui_assertions=("booking_panel", "checkout_unavailable_notice"),
            contract_assertions=("checkout_status.provider_handoff_unavailable",),
        ),
        _case(
            case_name="frontend_fixture_booking_hold_cancel",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Hold the top option and cancel it",
            },
            fixture_scenario="fixture_booking_hold_cancel",
            expectations={
                "post_actions": ["hold", "cancel_latest", "refresh_bookings"],
                "required_endpoint_calls": ["booking_hold", "booking_cancel", "bookings_list"],
                "require_notice_contains": "booking cancelled",
            },
            features=("booking.hold", "booking.cancel", "booking.list", "booking.policy.no_google_fallback", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "booking_outcome": "hold_then_cancel",
                "seller_provider_breadth": "provider_unavailable",
            },
            ui_assertions=("booking_panel", "booking_action_notice"),
            contract_assertions=("booking_hold", "booking_cancel", "bookings_list"),
        ),
        _case(
            case_name="frontend_fixture_tracking_alerts",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Track this option and acknowledge an alert",
            },
            fixture_scenario="fixture_tracking_alerts",
            expectations={
                "post_actions": ["track", "refresh_alerts", "ack_first_alert"],
                "required_endpoint_calls": [
                    "booking_track_price",
                    "price_tracking_status",
                    "price_tracking_alerts",
                    "price_tracking_alert_ack",
                ],
            },
            features=("tracking.track_price", "tracking.status", "tracking.alerts", "tracking.alert_ack", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "tracking_outcome": "alert_acknowledged",
                "seller_provider_breadth": "tracked_route",
            },
            ui_assertions=("tracking_panel", "alert_ack"),
            contract_assertions=("price_tracking_status", "price_tracking_alerts", "price_tracking_alert_ack"),
        ),
        _case(
            case_name="frontend_fixture_async_jobs",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Run the route in background async mode",
            },
            fixture_scenario="fixture_async_jobs",
            expectations={
                "enable_async_mode": True,
                "required_endpoint_calls": ["ask_async", "jobs_events"],
                "optional_endpoint_calls": ["jobs_poll"],
                "expect_stream_request": False,
            },
            features=("jobs.create", "jobs.poll", "jobs.events", "frontend.ui.fields.async_toggle", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "async_behavior": "create_poll_events",
                "trip_type": "one_way",
            },
            ui_assertions=("async_toggle",),
            contract_assertions=("ask_async", "jobs_poll", "jobs_events"),
        ),
        _case(
            case_name="frontend_fixture_async_cancel",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Cancel the async job quickly",
            },
            fixture_scenario="fixture_async_jobs_cancel",
            expectations={
                "enable_async_mode": True,
                "pre_settle_actions": ["cancel_async_job"],
                "required_endpoint_calls": ["ask_async", "jobs_cancel"],
                "expect_stream_request": False,
                "allow_sparse_result": True,
            },
            features=("jobs.cancel", "frontend.ui.fields.async_toggle", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            dimensions={
                "async_behavior": "cancel",
                "trip_type": "one_way",
            },
            ui_assertions=("async_toggle",),
            contract_assertions=("ask_async", "jobs_cancel"),
        ),
        _case(
            case_name="frontend_fixture_cabin_truth",
            payload={
                "origin": "MAA",
                "destination": "DEL",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "cabin": "business",
                "user_query": "Business class flight from Chennai to Delhi",
            },
            fixture_scenario="fixture_cabin_business_no_match",
            expectations={"require_notice_contains": "business class", "expect_stream_request": False},
            features=("intent.cabin", "frontend.ui.fields.cabin", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER, MODE_FRONTEND_REAL_BACKEND_BROWSER),
            dimensions={
                "user_intent": "cabin",
                "trip_type": "one_way",
            },
            ui_assertions=("constraint_notice",),
            contract_assertions=("ask.non_stream",),
        ),
        _case(
            case_name="frontend_fixture_direct_truth",
            payload={
                "origin": "DEL",
                "destination": "GOI",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "direct_only": True,
                "user_query": "Direct flights only from Delhi to Goa",
            },
            fixture_scenario="fixture_direct_truthful",
            expectations={"require_notice_contains": "direct", "expect_stream_request": False},
            features=("intent.direct", "frontend.ui.fields.direct", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER, MODE_FRONTEND_REAL_BACKEND_BROWSER),
            dimensions={
                "user_intent": "direct",
                "trip_type": "one_way",
            },
            ui_assertions=("constraint_notice",),
            contract_assertions=("ask.non_stream",),
        ),
        _case(
            case_name="frontend_fixture_dev_mode_operator_endpoints",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-18",
                "trip_type": "one-way",
                "user_query": "Dev mode contract check with normal planning response",
            },
            fixture_scenario="fixture_non_stream_one_way",
            expectations={
                "enable_dev_mode": True,
                "expect_stream_request": True,
                "required_endpoint_calls": ["llm_options", "version", "ask_stream"],
            },
            features=("ops.llm_options", "ops.version", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_FIXTURE_BROWSER,),
            criticality="extended",
            dimensions={
                "surface": "dev_operator",
                "dev_mode": "true",
                "route_shape": "point_to_point",
            },
            ui_assertions=("proof_overview",),
            contract_assertions=("llm_options", "version", "ask.stream"),
        ),
        # Real-backend browser matrix (explicit opt-in).
        _case(
            case_name="frontend_real_backend_ask_non_stream_basic",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Cheapest one-way flight Delhi to Mumbai on 2026-07-19",
            },
            fixture_scenario="",
            expectations={"expect_stream_request": False, "allow_live_backend": True},
            features=("ask.non_stream", "trip.one_way", "intent.cheapest", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "one_way",
                "date_basis": "explicit_date",
                "user_intent": "cheapest",
            },
            ui_assertions=("proof_overview", "ranked_shortlist", "weather_panel"),
            contract_assertions=("ask.non_stream",),
        ),
        _case(
            case_name="frontend_real_backend_ask_stream_basic",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Find a flight Delhi to Mumbai on 2026-07-19",
            },
            fixture_scenario="",
            expectations={"expect_stream_request": True, "allow_live_backend": True},
            features=("ask.stream", "trip.one_way", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "one_way",
                "date_basis": "explicit_date",
                "user_intent": "generic",
            },
            ui_assertions=("proof_overview", "trip_brief"),
            contract_assertions=("ask.stream",),
        ),
        _case(
            case_name="frontend_real_backend_dev_mode_operator_endpoints",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Dev mode endpoint contract check against real backend",
            },
            fixture_scenario="",
            expectations={
                "allow_live_backend": True,
                "enable_dev_mode": True,
                "expect_stream_request": True,
                "required_endpoint_calls": ["llm_options", "version", "ask_stream"],
            },
            features=("ops.llm_options", "ops.version", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            criticality="extended",
            dimensions={
                "surface": "dev_operator",
                "dev_mode": "true",
                "route_shape": "point_to_point",
            },
            ui_assertions=("proof_overview",),
            contract_assertions=("llm_options", "version", "ask.stream"),
        ),
        _case(
            case_name="frontend_real_backend_direct_truth",
            payload={
                "origin": "DEL",
                "destination": "GOI",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "direct_only": True,
                "user_query": "Direct nonstop flights only from Delhi to Goa on 2026-07-19",
            },
            fixture_scenario="",
            expectations={"expect_stream_request": False, "allow_live_backend": True, "require_notice_contains_any": ["direct", "non-stop", "nonstop"]},
            features=("intent.direct", "frontend.ui.fields.direct", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "user_intent": "direct",
                "trip_type": "one_way",
            },
            ui_assertions=("constraint_notice",),
            contract_assertions=("ask.non_stream",),
        ),
        _case(
            case_name="frontend_real_backend_cabin_truth",
            payload={
                "origin": "MAA",
                "destination": "DEL",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "cabin": "business",
                "user_query": "Business class flight from Chennai to Delhi on 2026-07-19",
            },
            fixture_scenario="",
            expectations={"expect_stream_request": False, "allow_live_backend": True, "require_notice_contains_any": ["business", "cabin", "available"]},
            features=("intent.cabin", "frontend.ui.fields.cabin", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "user_intent": "cabin",
                "trip_type": "one_way",
            },
            ui_assertions=("constraint_notice",),
            contract_assertions=("ask.non_stream",),
        ),
        _case(
            case_name="frontend_real_backend_booking_hold",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Find flight options Delhi to Mumbai for booking hold validation",
            },
            fixture_scenario="",
            expectations={
                "expect_stream_request": True,
                "allow_live_backend": True,
                "post_actions": ["hold", "refresh_bookings"],
                "required_endpoint_calls": ["booking_hold", "bookings_list"],
            },
            features=("booking.hold", "booking.list", "booking.navigation.local_hold_only", "booking.policy.no_google_fallback", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "booking_outcome": "hold",
                "seller_provider_breadth": "local_or_provider",
            },
            ui_assertions=("booking_panel",),
            contract_assertions=("booking_hold", "bookings_list"),
        ),
        _case(
            case_name="frontend_real_backend_booking_cancel",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Find flight options Delhi to Mumbai for booking cancel validation",
            },
            fixture_scenario="",
            expectations={
                "expect_stream_request": True,
                "allow_live_backend": True,
                "post_actions": ["hold", "refresh_bookings", "cancel_latest", "refresh_bookings"],
                "required_endpoint_calls": ["booking_hold", "booking_cancel", "bookings_list"],
            },
            features=("booking.hold", "booking.cancel", "booking.list", "booking.policy.no_google_fallback", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "booking_outcome": "hold_then_cancel",
                "seller_provider_breadth": "local_or_provider",
            },
            ui_assertions=("booking_panel",),
            contract_assertions=("booking_hold", "booking_cancel", "bookings_list"),
        ),
        _case(
            case_name="frontend_real_backend_bookings_refresh",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Show current booking list and refresh it",
            },
            fixture_scenario="",
            expectations={
                "expect_stream_request": False,
                "allow_live_backend": True,
                "post_actions": ["refresh_bookings"],
                "required_endpoint_calls": ["bookings_list"],
                "allow_sparse_result": True,
            },
            features=("booking.list", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "booking_outcome": "list_refresh",
            },
            ui_assertions=("booking_panel",),
            contract_assertions=("bookings_list",),
        ),
        _case(
            case_name="frontend_real_backend_tracking_status",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Check tracking status for Delhi Mumbai options",
            },
            fixture_scenario="",
            expectations={
                "expect_stream_request": False,
                "allow_live_backend": True,
                "required_endpoint_calls": ["price_tracking_status"],
                "allow_sparse_result": True,
            },
            features=("tracking.status", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "tracking_outcome": "status",
            },
            ui_assertions=("tracking_panel",),
            contract_assertions=("price_tracking_status",),
        ),
        _case(
            case_name="frontend_real_backend_alerts_list",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Track Delhi Mumbai and inspect alert list",
            },
            fixture_scenario="",
            expectations={
                "expect_stream_request": True,
                "allow_live_backend": True,
                "post_actions": ["track", "refresh_alerts"],
                "required_endpoint_calls": ["booking_track_price", "price_tracking_status", "price_tracking_alerts"],
            },
            features=("tracking.track_price", "tracking.status", "tracking.alerts", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "tracking_outcome": "alerts_list",
            },
            ui_assertions=("tracking_panel",),
            contract_assertions=("booking_track_price", "price_tracking_alerts"),
        ),
        _case(
            case_name="frontend_real_backend_alert_ack",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Track Delhi Mumbai and acknowledge the first alert if available",
            },
            fixture_scenario="",
            expectations={
                "expect_stream_request": True,
                "allow_live_backend": True,
                "post_actions": ["track", "refresh_alerts", "ack_first_alert_if_present"],
                "required_endpoint_calls": ["booking_track_price", "price_tracking_alerts"],
                "optional_endpoint_calls": ["price_tracking_alert_ack"],
            },
            features=("tracking.track_price", "tracking.alerts", "tracking.alert_ack", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "tracking_outcome": "alert_ack_if_present",
            },
            ui_assertions=("tracking_panel",),
            contract_assertions=("booking_track_price", "price_tracking_alerts"),
        ),
        _case(
            case_name="frontend_real_backend_async_jobs",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Run this search in async mode and follow job events",
            },
            fixture_scenario="",
            expectations={
                "enable_async_mode": True,
                "allow_live_backend": True,
                "required_endpoint_calls": ["ask_async", "jobs_poll", "jobs_events"],
                "expect_stream_request": False,
            },
            features=("jobs.create", "jobs.poll", "jobs.events", "frontend.ui.fields.async_toggle", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "async_behavior": "create_poll_events",
            },
            ui_assertions=("async_toggle",),
            contract_assertions=("ask_async", "jobs_poll", "jobs_events"),
        ),
        _case(
            case_name="frontend_real_backend_seller_ota_breadth",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "user_query": "Compare seller options and booking handoff for Delhi to Mumbai",
            },
            fixture_scenario="",
            expectations={
                "expect_stream_request": False,
                "allow_live_backend": True,
                "require_seller_or_handoff_signal": True,
                "allow_sparse_result": True,
            },
            features=(
                "seller.ota_diversity",
                "booking.navigation.link_visible",
                "booking.navigation.provider_handoff_present",
                "frontend.ui.behavior",
            ),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            dimensions={
                "seller_provider_breadth": "ota_or_provider_handoff",
            },
            ui_assertions=("seller_signal_or_handoff",),
            contract_assertions=("booking_sellers_or_handoff",),
        ),
        _case(
            case_name="frontend_real_backend_no_flights_or_degraded",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-19",
                "trip_type": "one-way",
                "direct_only": True,
                "cabin": "first",
                "baggage_pref": "hand",
                "user_query": "First class direct nonstop Delhi to Mumbai with cabin baggage only under INR 1000",
            },
            fixture_scenario="",
            expectations={
                "expect_stream_request": False,
                "allow_live_backend": True,
                "allow_sparse_result": True,
                "accept_any_outcome": ["no_flights", "degraded", "success_with_constraint_notice"],
            },
            features=("ask.no_flights", "ask.degraded", "intent.direct", "intent.cabin", "frontend.ui.fields.baggage", "frontend.ui.behavior"),
            mode_tags=(MODE_FRONTEND_REAL_BACKEND_BROWSER,),
            criticality="extended",
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "one_way",
                "user_intent": "stress_constraints",
                "booking_outcome": "none_or_partial",
            },
            ui_assertions=("constraint_notice",),
            contract_assertions=("fallback_or_error_path",),
        ),
        # Live canary browser checks (small, explicit, curated).
        _case(
            case_name="frontend_live_canary_direct_one_way",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-20",
                "trip_type": "one-way",
                "direct_only": True,
                "user_query": "Cheapest direct flight Delhi to Mumbai on 2026-07-20",
            },
            fixture_scenario="",
            expectations={"allow_live_backend": True, "expect_stream_request": True},
            features=("ask.stream", "intent.direct", "intent.cheapest", "booking.navigation.real_provider_browser_proof"),
            mode_tags=(MODE_LIVE_CANARY_BROWSER,),
            criticality="canary",
            soft_pass_policy=SOFT_PASS_LIVE_ONLY,
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "one_way",
                "user_intent": "direct_cheapest",
                "seller_provider_breadth": "provider_canary",
            },
            ui_assertions=("proof_overview",),
            contract_assertions=("ask.stream",),
        ),
        _case(
            case_name="frontend_live_canary_round_trip",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-20",
                "return_date": "2026-07-23",
                "trip_type": "round-trip",
                "user_query": "Round-trip Delhi to Mumbai leaving 2026-07-20 returning 2026-07-23",
            },
            fixture_scenario="",
            expectations={"allow_live_backend": True, "expect_stream_request": True},
            features=("ask.stream", "trip.round_trip", "frontend.ui.fields.return_date", "booking.navigation.real_provider_browser_proof"),
            mode_tags=(MODE_LIVE_CANARY_BROWSER,),
            criticality="canary",
            soft_pass_policy=SOFT_PASS_LIVE_ONLY,
            dimensions={
                "route_shape": "point_to_point",
                "trip_type": "round_trip",
                "user_intent": "round_trip",
                "seller_provider_breadth": "provider_canary",
            },
            ui_assertions=("proof_overview",),
            contract_assertions=("ask.stream",),
        ),
        _case(
            case_name="frontend_live_canary_seller_diversity",
            payload={
                "origin": "DEL",
                "destination": "BOM",
                "date": "2026-07-20",
                "trip_type": "one-way",
                "user_query": "Show seller diversity and provider handoff options for Delhi to Mumbai",
            },
            fixture_scenario="",
            expectations={
                "allow_live_backend": True,
                "expect_stream_request": False,
                "require_seller_or_handoff_signal": True,
                "allow_sparse_result": True,
            },
            features=(
                "seller.ota_diversity",
                "booking.navigation.link_visible",
                "booking.navigation.provider_handoff_present",
                "booking.navigation.real_provider_browser_proof",
            ),
            mode_tags=(MODE_LIVE_CANARY_BROWSER,),
            criticality="canary",
            soft_pass_policy=SOFT_PASS_LIVE_ONLY,
            dimensions={
                "seller_provider_breadth": "ota_or_provider_handoff",
                "user_intent": "seller_diversity",
            },
            ui_assertions=("seller_signal_or_handoff",),
            contract_assertions=("booking_sellers_or_handoff",),
        ),
        _case(
            case_name="frontend_live_canary_cabin_direct_stress",
            payload={
                "origin": "DEL",
                "destination": "BLR",
                "date": "2026-07-20",
                "trip_type": "one-way",
                "cabin": "business",
                "direct_only": True,
                "user_query": "Business class direct flight Delhi to Bangalore on 2026-07-20",
            },
            fixture_scenario="",
            expectations={
                "allow_live_backend": True,
                "expect_stream_request": False,
                "require_notice_contains_any": ["business", "direct", "non-stop", "nonstop"],
                "allow_sparse_result": True,
            },
            features=("intent.cabin", "intent.direct", "frontend.ui.fields.cabin", "frontend.ui.fields.direct", "booking.navigation.real_provider_browser_proof"),
            mode_tags=(MODE_LIVE_CANARY_BROWSER,),
            criticality="canary",
            soft_pass_policy=SOFT_PASS_LIVE_ONLY,
            dimensions={
                "user_intent": "cabin_direct_stress",
                "trip_type": "one_way",
            },
            ui_assertions=("constraint_notice",),
            contract_assertions=("ask.non_stream",),
        ),
    ]
    return cases


def frontend_runtime_cases(
    *,
    mode: ValidationModeId = MODE_FRONTEND_FIXTURE_BROWSER,
    include_live_canary: bool = False,
) -> List[FrontendRuntimeCase]:
    """
    Curated-but-dimensional browser validation catalog.

    - Default mode returns fixture-backed scenarios only (safe-by-default).
    - Real-backend mode returns explicit browser journeys against local backend.
    - include_live_canary appends small live-provider canaries when in real-backend mode.
    """
    catalog = _frontend_runtime_case_catalog()
    selected: List[FrontendRuntimeCase] = []
    for case in catalog:
        tags = set(case.mode_tags)
        if mode in tags:
            selected.append(case)
        elif mode == MODE_FRONTEND_REAL_BACKEND_BROWSER and include_live_canary and MODE_LIVE_CANARY_BROWSER in tags:
            selected.append(case)

    if include_live_canary and mode == MODE_LIVE_CANARY_BROWSER:
        selected = [case for case in catalog if MODE_LIVE_CANARY_BROWSER in set(case.mode_tags)]

    return selected
