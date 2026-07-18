import contextlib
import asyncio
import logging
from unittest.mock import AsyncMock

import pytest
import httpx
from prometheus_client import generate_latest

import tools.airline_api as airline_api
from tools.airline_api import (
    AirlineAPIError,
    FLIGHT_CACHE_SCHEMA_VERSION,
    _parse_duration,
    _redact_request_params,
    _redact_sensitive_url,
    _build_flight_cache_key,
    expand_airports,
    search_flights,
)


class _DummyCircuitBreaker:
    async def call(self, fn):
        return await fn()


class _DummyResponse:
    def __init__(self, *, status_code: int, text: str = "", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data if json_data is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class _DummyClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def get(self, *_args, **_kwargs):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


class _DummyKeyManager:
    def __init__(self):
        self.reserve_calls = 0
        self.mark_calls = []
        self.record_calls = []

    @contextlib.asynccontextmanager
    async def reserve_key(self, _service):
        self.reserve_calls += 1
        if self.reserve_calls > 1:
            raise RuntimeError("No available keys for service: serpapi")
        yield 0, "serpapi-key-1"

    async def mark_exhausted(self, service, idx, *, until=None, reason=""):
        self.mark_calls.append(
            {
                "service": service,
                "idx": idx,
                "until": until,
                "reason": reason,
            }
        )

    async def record_usage(self, service, idx):
        self.record_calls.append((service, idx))


def test_parse_duration():
    assert _parse_duration("2h 15m") == 135
    assert _parse_duration("2:15") == 135
    assert _parse_duration("45m") == 45
    assert _parse_duration("3h") == 180
    assert _parse_duration(120) == 120
    assert _parse_duration("Unknown duration") is None


def test_redact_request_params_masks_api_key_and_auth_fields():
    redacted = _redact_request_params(
        {
            "api_key": "secret",
            "authorization": "Bearer secret-token",
            "x-api-key": "secret-header",
            "appid": "secret-app-id",
            "departure_id": "DEL",
        }
    )
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["authorization"] == "***REDACTED***"
    assert redacted["x-api-key"] == "***REDACTED***"
    assert redacted["appid"] == "***REDACTED***"
    assert redacted["departure_id"] == "DEL"


def test_redact_sensitive_url_masks_query_api_key():
    raw = "https://serpapi.com/account.json?api_key=topsecret&engine=google_flights"
    redacted = _redact_sensitive_url(raw)
    assert "topsecret" not in redacted
    assert "api_key=%2A%2A%2AREDACTED%2A%2A%2A" in redacted


def test_expand_airports_uses_central_resolver_for_city_name():
    assert expand_airports("Ahmedabad") == "AMD"
    assert expand_airports("Kolkata") == "CCU"


def test_expand_airports_ignores_generic_location_qualifiers():
    assert expand_airports("Ahmedabad airport") == "AMD"
    assert expand_airports("Pune city") == "PNQ"


def test_expand_airports_returns_empty_for_unresolved_multiword_phrase():
    assert expand_airports("Unknown City Name") == ""


def test_build_flight_cache_key_is_schema_versioned():
    key = _build_flight_cache_key(
        departure_ids="DEL",
        arrival_ids="BOM",
        date="2026-03-20",
        return_date=None,
        serpapi_type="2",
        eco_mode=False,
        min_layover=None,
        max_layover=None,
        deep_search=False,
    )
    assert key[0] == FLIGHT_CACHE_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_flight_merging_logic(monkeypatch):
    monkeypatch.setenv("TESTING", "1")

    flights, price_insights = await search_flights("DEL", "BOM", "2026-03-20")

    assert len(flights) == 1
    assert flights[0].airline == "TestAir"
    assert flights[0].duration_min == 120
    assert flights[0].carbon_emissions_g == 45000


@pytest.mark.asyncio
async def test_search_flights_marks_key_exhausted_on_http_403(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient([_DummyResponse(status_code=403, text="forbidden key")])

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())

    with pytest.raises(AirlineAPIError, match="All SerpAPI keys are currently exhausted"):
        await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert len(dummy_km.mark_calls) == 1
    mark = dummy_km.mark_calls[0]
    assert mark["service"] == "serpapi"
    assert mark["idx"] == 0
    assert "unauthorized_http_403" in mark["reason"]


@pytest.mark.asyncio
async def test_search_flights_honors_total_attempt_budget_under_degraded_network(monkeypatch):
    dummy_km = _DummyKeyManager()

    class _RaisingClient:
        def __init__(self):
            self.calls = 0

        async def get(self, *_args, **_kwargs):
            self.calls += 1
            raise httpx.TimeoutException("simulated timeout")

    client = _RaisingClient()

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(airline_api, "SERPAPI_MAX_RETRIES", 5)
    monkeypatch.setattr(airline_api, "SERPAPI_TOTAL_ATTEMPT_BUDGET", 2)
    monkeypatch.setattr(airline_api, "SERPAPI_RETRY_BASE_DELAY", 0.0)

    with pytest.raises(AirlineAPIError, match="retry budget exhausted"):
        await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert client.calls == 2


@pytest.mark.asyncio
async def test_search_flights_retry_budget_is_stable_across_repeated_degraded_calls(monkeypatch):
    class _AlwaysKeyManager:
        @contextlib.asynccontextmanager
        async def reserve_key(self, _service):
            yield 0, "serpapi-key-stable"

        async def mark_exhausted(self, *_args, **_kwargs):
            return None

        async def record_usage(self, *_args, **_kwargs):
            return None

    class _RaisingClient:
        def __init__(self):
            self.calls = 0

        async def get(self, *_args, **_kwargs):
            self.calls += 1
            raise httpx.TimeoutException("simulated repeated timeout")

    client = _RaisingClient()

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", _AlwaysKeyManager())
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(airline_api, "SERPAPI_MAX_RETRIES", 5)
    monkeypatch.setattr(airline_api, "SERPAPI_TOTAL_ATTEMPT_BUDGET", 2)
    monkeypatch.setattr(airline_api, "SERPAPI_RETRY_BASE_DELAY", 0.0)

    for _ in range(3):
        with pytest.raises(AirlineAPIError, match="retry budget exhausted"):
            await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert client.calls == 6

    metrics_text = generate_latest().decode()
    assert 'retry_budget_exhausted_total{component="airline_search_flights"}' in metrics_text


@pytest.mark.asyncio
async def test_search_flights_rejects_unresolved_location_before_http(monkeypatch):
    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)

    with pytest.raises(AirlineAPIError, match="Could not resolve departure location"):
        await search_flights("Unknown City Name", "DEL", "2026-03-20", use_cache=False)


@pytest.mark.asyncio
@pytest.mark.xfail(reason="pre-M1 drift: SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_ENABLED removed in f9457ea, pending Sid review", strict=False)
async def test_search_flights_logs_key_source_and_masked_fingerprint(monkeypatch, caplog):
    dummy_km = _DummyKeyManager()
    client = _DummyClient(
        [
            _DummyResponse(
                status_code=200,
                json_data={
                    "best_flights": [
                        {
                            "price": "6200",
                            "total_duration": 120,
                            "flights": [
                                {
                                    "airline": "TraceAir",
                                    "flight_number": "TR100",
                                    "departure_airport": {"time": "2026-03-20 06:00", "id": "DEL"},
                                    "arrival_airport": {"time": "2026-03-20 08:00", "id": "BOM"},
                                    "duration": 120,
                                }
                            ],
                        }
                    ],
                    "other_flights": [],
                },
            )
        ]
    )

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(airline_api, "SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_ENABLED", True)

    with caplog.at_level("DEBUG"):
        flights, _ = await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert flights
    record = next(r for r in caplog.records if "SerpAPI attempt succeeded" in r.message)
    assert record.key_source == "api_key_manager.reserve_key:serpapi"
    assert record.client_mode == "shared_get_client"
    assert isinstance(record.key_fp, str)
    assert len(record.key_fp) == 10


@pytest.mark.asyncio
@pytest.mark.xfail(reason="pre-M1 drift: SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_ENABLED removed in f9457ea, pending Sid review", strict=False)
async def test_search_flights_success_account_check_non_200_is_non_blocking_info(monkeypatch, caplog):
    dummy_km = _DummyKeyManager()
    airline_api._last_account_check.clear()
    client = _DummyClient(
        [
            _DummyResponse(
                status_code=200,
                json_data={
                    "best_flights": [
                        {
                            "price": "6200",
                            "total_duration": 120,
                            "flights": [
                                {
                                    "airline": "TraceAir",
                                    "flight_number": "TR100",
                                    "departure_airport": {"time": "2026-03-20 06:00", "id": "DEL"},
                                    "arrival_airport": {"time": "2026-03-20 08:00", "id": "BOM"},
                                    "duration": 120,
                                }
                            ],
                        }
                    ],
                    "other_flights": [],
                },
            ),
            _DummyResponse(status_code=401, text="unauthorized"),
        ]
    )

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(airline_api, "SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_ENABLED", True)

    with caplog.at_level("INFO"):
        flights, _ = await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert flights
    for _ in range(5):
        if any(r.message == "Post-success account check skipped (non-blocking)" for r in caplog.records):
            break
        await asyncio.sleep(0)
    assert all(record.message != "Account check failed" for record in caplog.records)
    assert not any(
        record.levelno >= logging.WARNING and "account check" in record.message.lower()
        for record in caplog.records
    )
    deferred_records = [
        r for r in caplog.records if r.message == "Post-success account check skipped (non-blocking)"
    ]
    if deferred_records:
        assert deferred_records[0].status == 401


@pytest.mark.asyncio
@pytest.mark.xfail(reason="pre-M1 drift: SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_ENABLED removed in f9457ea, pending Sid review", strict=False)
async def test_search_flights_success_skips_post_success_account_check_when_disabled(monkeypatch):
    dummy_km = _DummyKeyManager()
    airline_api._last_account_check.clear()
    client = _DummyClient(
        [
            _DummyResponse(
                status_code=200,
                json_data={
                    "best_flights": [
                        {
                            "price": "6200",
                            "total_duration": 120,
                            "flights": [
                                {
                                    "airline": "TraceAir",
                                    "flight_number": "TR100",
                                    "departure_airport": {"time": "2026-03-20 06:00", "id": "DEL"},
                                    "arrival_airport": {"time": "2026-03-20 08:00", "id": "BOM"},
                                    "duration": 120,
                                }
                            ],
                        }
                    ],
                    "other_flights": [],
                },
            ),
            _DummyResponse(status_code=429, text="quota should not be checked in success hot path"),
        ]
    )

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(airline_api, "SERPAPI_POST_SUCCESS_ACCOUNT_CHECK_ENABLED", False)

    flights, _ = await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert flights
    assert client.calls == 1


@pytest.mark.asyncio
async def test_health_check_quarantines_quota_exhausted_key(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient(
        [
            _DummyResponse(status_code=429, text="quota exceeded"),
            _DummyResponse(status_code=200, json_data={"plan_searches_left": 0, "plan_name": "monthly"}),
        ]
    )

    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())

    status = await airline_api.health_check()

    assert status == "degraded"
    assert len(dummy_km.mark_calls) == 1
    mark = dummy_km.mark_calls[0]
    assert mark["service"] == "serpapi"
    assert mark["idx"] == 0
    assert "health_quota_http_429" in mark["reason"]


@pytest.mark.asyncio
async def test_health_check_non_destructive_mode_skips_key_quarantine(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient(
        [
            _DummyResponse(status_code=429, text="quota exceeded"),
            _DummyResponse(status_code=200, json_data={"plan_searches_left": 0, "plan_name": "monthly"}),
        ]
    )

    monkeypatch.setenv("HEALTHCHECK_NON_DESTRUCTIVE", "1")
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())

    status = await airline_api.health_check()

    assert status == "degraded"
    assert dummy_km.mark_calls == []


@pytest.mark.asyncio
async def test_health_check_returns_degraded_on_upstream_5xx(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient([_DummyResponse(status_code=503, text="upstream unavailable")])

    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())

    status = await airline_api.health_check()

    assert status == "degraded"


@pytest.mark.asyncio
async def test_search_flights_recovers_from_transient_non_json_payload(monkeypatch):
    dummy_km = _DummyKeyManager()

    class _BadJsonResponse:
        status_code = 200
        text = "<html>temporary gateway page</html>"

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("not json")

    client = _DummyClient(
        [
            _BadJsonResponse(),
            _DummyResponse(
                status_code=200,
                json_data={
                    "best_flights": [
                        {
                            "price": "6200",
                            "total_duration": 120,
                            "flights": [
                                {
                                    "airline": "RecoverAir",
                                    "flight_number": "RC200",
                                    "departure_airport": {"time": "2026-03-20 06:00", "id": "DEL"},
                                    "arrival_airport": {"time": "2026-03-20 08:00", "id": "BOM"},
                                    "duration": 120,
                                }
                            ],
                        }
                    ],
                    "other_flights": [],
                },
            ),
        ]
    )

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())

    flights, _ = await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert flights
    assert flights[0].airline == "RecoverAir"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_search_flights_keeps_candidates_with_unusable_price(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient(
        [
            _DummyResponse(
                status_code=200,
                json_data={
                    "best_flights": [
                        {
                            "price": None,
                            "total_duration": 120,
                            "flights": [
                                {
                                    "airline": "NoPriceAir",
                                    "flight_number": "NP100",
                                    "departure_airport": {"time": "2026-03-20 06:00", "id": "DEL"},
                                    "arrival_airport": {"time": "2026-03-20 08:00", "id": "BOM"},
                                    "duration": 120,
                                }
                            ],
                        },
                        {
                            "price": "6500",
                            "total_duration": 125,
                            "flights": [
                                {
                                    "airline": "PricedAir",
                                    "flight_number": "PA200",
                                    "departure_airport": {"time": "2026-03-20 09:00", "id": "DEL"},
                                    "arrival_airport": {"time": "2026-03-20 11:05", "id": "BOM"},
                                    "duration": 125,
                                }
                            ],
                        },
                    ],
                    "other_flights": [],
                },
            )
        ]
    )

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())

    flights, meta = await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert len(flights) == 2
    assert any(getattr(f, "price_unavailable", False) for f in flights)
    assert any(getattr(f, "price_inr", None) == "Price unavailable" for f in flights)
    assert isinstance(meta, dict)
    assert "_search_meta" in meta
    assert meta["_search_meta"]["missing_price_count"] >= 1


@pytest.mark.asyncio
async def test_search_flights_round_trip_interleave_parse_window_recovers_third_candidate(monkeypatch):
    dummy_km = _DummyKeyManager()
    client = _DummyClient(
        [
            _DummyResponse(
                status_code=200,
                json_data={
                    "best_flights": [
                        {
                            "price": "6200",
                            "total_duration": 120,
                            "flights": [
                                {
                                    # Missing airline on purpose: invalid parse candidate.
                                    "flight_number": "RB100",
                                    "departure_airport": {"time": "2026-03-20 06:00", "id": "DEL"},
                                    "arrival_airport": {"time": "2026-03-20 08:00", "id": "BOM"},
                                    "duration": 120,
                                }
                            ],
                        },
                        {
                            "price": "6800",
                            "total_duration": 125,
                            "flights": [
                                {
                                    "airline": "RoundTripAir",
                                    "flight_number": "RB201",
                                    "departure_airport": {"time": "2026-03-20 09:00", "id": "DEL"},
                                    "arrival_airport": {"time": "2026-03-20 11:05", "id": "BOM"},
                                    "duration": 125,
                                }
                            ],
                        },
                    ],
                    "other_flights": [
                        {
                            "price": "6400",
                            "total_duration": 121,
                            "flights": [
                                {
                                    # Missing airline on purpose: invalid parse candidate.
                                    "flight_number": "RB150",
                                    "departure_airport": {"time": "2026-03-20 07:00", "id": "DEL"},
                                    "arrival_airport": {"time": "2026-03-20 09:01", "id": "BOM"},
                                    "duration": 121,
                                }
                            ],
                        }
                    ],
                },
            )
        ]
    )

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())

    flights, meta = await search_flights(
        "DEL",
        "BOM",
        "2026-03-20",
        return_date="2026-03-24",
        max_results=1,
        use_cache=False,
    )

    # Round-trip complex path should interleave buckets and parse one extra row.
    assert len(flights) == 1
    assert flights[0].flight_no == "RB201"
    assert meta["_search_meta"]["merge_mode"] == "interleave_best_other"
    assert meta["_search_meta"]["parse_window"] == 3


@pytest.mark.asyncio
async def test_search_flights_cache_round_trip_preserves_booking_artifacts(monkeypatch):
    dummy_km = _DummyKeyManager()
    response = _DummyResponse(
        status_code=200,
        json_data={
            "best_flights": [
                {
                    "price": "7000",
                    "total_duration": 120,
                    "booking_token": "tok_abc",
                    "shareable_link": "https://partner.example/share/abc",
                    "booking_request": {
                        "method": "GET",
                        "url": "https://partner.example/checkout/abc",
                    },
                    "booking_options": [
                        {"booking_url": "https://partner.example/checkout/opt1"}
                    ],
                    "flights": [
                        {
                            "airline": "CacheAir",
                            "flight_number": "CA101",
                            "departure_airport": {"time": "2026-03-20 08:00", "id": "DEL"},
                            "arrival_airport": {"time": "2026-03-20 10:00", "id": "BOM"},
                            "duration": 120,
                        }
                    ],
                }
            ],
            "other_flights": [],
        },
    )
    client = _DummyClient([response])

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())
    airline_api._flight_cache.clear()

    flights_1, _ = await search_flights("DEL", "BOM", "2026-03-20", use_cache=True)
    assert len(flights_1) == 1
    assert flights_1[0].shareable_link == "https://partner.example/share/abc"
    assert flights_1[0].booking_request["url"] == "https://partner.example/checkout/abc"
    assert flights_1[0].booking_options[0]["booking_url"] == "https://partner.example/checkout/opt1"
    calls_after_first = client.calls

    flights_2, _ = await search_flights("DEL", "BOM", "2026-03-20", use_cache=True)
    assert len(flights_2) == 1
    assert flights_2[0].shareable_link == "https://partner.example/share/abc"
    assert flights_2[0].booking_request["url"] == "https://partner.example/checkout/abc"
    assert client.calls == calls_after_first


@pytest.mark.asyncio
async def test_search_flights_extracts_nested_booking_artifacts_when_root_missing(monkeypatch):
    dummy_km = _DummyKeyManager()
    response = _DummyResponse(
        status_code=200,
        json_data={
            "best_flights": [
                {
                    "price": "7100",
                    "total_duration": 126,
                    "flights": [
                        {
                            "airline": "NestedAir",
                            "flight_number": "NA222",
                            "departure_airport": {"time": "2026-03-20 08:10", "id": "DEL"},
                            "arrival_airport": {"time": "2026-03-20 10:16", "id": "BOM"},
                            "duration": 126,
                            "booking_token": "tok_nested_leg",
                            "shareable_link": "https://partner.example/share/nested",
                            "booking_request": {
                                "method": "POST",
                                "url": "https://partner.example/checkout/nested",
                                "post_data": {"token": "n1"},
                            },
                        }
                    ],
                }
            ],
            "other_flights": [],
        },
    )
    client = _DummyClient([response])

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())

    flights, _ = await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert len(flights) == 1
    assert flights[0].flight_no == "NA222"
    assert flights[0].booking_token == "tok_nested_leg"
    assert flights[0].shareable_link == "https://partner.example/share/nested"
    assert flights[0].booking_request["url"] == "https://partner.example/checkout/nested"
    assert flights[0].booking_request["method"] == "POST"
    assert flights[0].booking_request["post_data"]["token"] == "n1"


@pytest.mark.asyncio
async def test_search_flights_does_not_promote_generic_nested_url_as_provider_artifact(monkeypatch):
    dummy_km = _DummyKeyManager()
    response = _DummyResponse(
        status_code=200,
        json_data={
            "best_flights": [
                {
                    "price": "7200",
                    "total_duration": 132,
                    "flights": [
                        {
                            "airline": "NoiseAir",
                            "flight_number": "NA401",
                            "departure_airport": {"time": "2026-03-20 09:00", "id": "DEL"},
                            "arrival_airport": {"time": "2026-03-20 11:12", "id": "BOM"},
                            "duration": 132,
                            "url": "https://partner.example/generic-unverified-url",
                        }
                    ],
                }
            ],
            "other_flights": [],
        },
    )
    client = _DummyClient([response])

    async def _fake_get_circuit_breaker(_name: str):
        return _DummyCircuitBreaker()

    monkeypatch.setattr(airline_api, "TESTING", False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(airline_api, "api_key_manager", dummy_km)
    monkeypatch.setattr(airline_api, "get_client", lambda: client)
    monkeypatch.setattr(airline_api, "get_circuit_breaker", _fake_get_circuit_breaker)
    monkeypatch.setattr(airline_api, "_rate_limit", AsyncMock())
    monkeypatch.setattr(airline_api.asyncio, "sleep", AsyncMock())

    flights, _ = await search_flights("DEL", "BOM", "2026-03-20", use_cache=False)

    assert len(flights) == 1
    assert flights[0].provider_link is None
    assert flights[0].booking_url is None
