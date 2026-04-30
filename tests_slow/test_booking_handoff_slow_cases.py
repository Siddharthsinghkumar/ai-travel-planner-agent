import pytest

from tests.test_booking_handoff import (
    _slow_case_hold_booking_does_not_trust_client_supplied_handoff_url_without_cache,
    _slow_case_hold_booking_persists_owner_principal,
    _slow_case_hold_booking_persists_route_primitives_for_tracker,
    _slow_case_hold_booking_resolves_when_booking_handoff_not_ready,
    _slow_case_hold_booking_reuses_booking_handoff_when_booking_ready,
)


@pytest.mark.asyncio
async def test_hold_booking_reuses_booking_handoff_when_booking_ready(monkeypatch):
    await _slow_case_hold_booking_reuses_booking_handoff_when_booking_ready(monkeypatch)


@pytest.mark.asyncio
async def test_hold_booking_does_not_trust_client_supplied_handoff_url_without_cache(monkeypatch):
    await _slow_case_hold_booking_does_not_trust_client_supplied_handoff_url_without_cache(monkeypatch)


@pytest.mark.asyncio
async def test_hold_booking_resolves_when_booking_handoff_not_ready(monkeypatch):
    await _slow_case_hold_booking_resolves_when_booking_handoff_not_ready(monkeypatch)


@pytest.mark.asyncio
async def test_hold_booking_persists_route_primitives_for_tracker(monkeypatch):
    await _slow_case_hold_booking_persists_route_primitives_for_tracker(monkeypatch)


@pytest.mark.asyncio
async def test_hold_booking_persists_owner_principal(monkeypatch):
    await _slow_case_hold_booking_persists_owner_principal(monkeypatch)
