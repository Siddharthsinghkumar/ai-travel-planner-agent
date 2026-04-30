import pytest

from tests.booking_tracking_service_authz_cases import (
    _slow_case_booking_cancel_returns_404_when_not_owner,
    _slow_case_booking_get_enforces_owner,
    _slow_case_booking_list_scopes_by_owner,
    _slow_case_price_tracking_alerts_scopes_by_owner,
    _slow_case_price_tracking_ack_returns_404_when_not_owner,
)


@pytest.mark.asyncio
async def test_booking_get_enforces_owner(monkeypatch):
    await _slow_case_booking_get_enforces_owner(monkeypatch)


@pytest.mark.asyncio
async def test_booking_list_scopes_by_owner(monkeypatch):
    await _slow_case_booking_list_scopes_by_owner(monkeypatch)


@pytest.mark.asyncio
async def test_booking_cancel_returns_404_when_not_owner(monkeypatch):
    await _slow_case_booking_cancel_returns_404_when_not_owner(monkeypatch)


@pytest.mark.asyncio
async def test_price_tracking_alerts_scopes_by_owner(monkeypatch):
    await _slow_case_price_tracking_alerts_scopes_by_owner(monkeypatch)


@pytest.mark.asyncio
async def test_price_tracking_ack_returns_404_when_not_owner(monkeypatch):
    await _slow_case_price_tracking_ack_returns_404_when_not_owner(monkeypatch)
