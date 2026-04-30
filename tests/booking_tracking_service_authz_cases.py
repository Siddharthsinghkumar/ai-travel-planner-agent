import pytest
from fastapi import HTTPException

import api.services_booking_tracking as booking_tracking_service
from core.auth import AuthenticatedPrincipal


@pytest.mark.asyncio
async def _slow_case_booking_get_enforces_owner(monkeypatch):
    seen = {}

    def fake_get_booking(booking_id, owner_principal_id=None):
        seen["booking_id"] = booking_id
        seen["owner"] = owner_principal_id
        return None

    monkeypatch.setattr("tools.booking_handoff.get_booking", fake_get_booking)
    principal = AuthenticatedPrincipal(principal_id="owner-a")

    with pytest.raises(HTTPException) as exc:
        await booking_tracking_service.booking_get(booking_id=99, principal=principal)

    assert exc.value.status_code == 404
    assert seen["booking_id"] == 99
    assert seen["owner"] == "owner-a"


@pytest.mark.asyncio
async def _slow_case_booking_list_scopes_by_owner(monkeypatch):
    seen = {}

    def fake_list_bookings(status=None, limit=100, owner_principal_id=None):
        seen["status"] = status
        seen["limit"] = limit
        seen["owner"] = owner_principal_id
        return [{"id": 1, "status": "HELD"}]

    monkeypatch.setattr("tools.booking_handoff.list_bookings", fake_list_bookings)
    principal = AuthenticatedPrincipal(principal_id="owner-a")

    payload = await booking_tracking_service.booking_list(
        status="HELD",
        limit=5,
        principal=principal,
    )

    assert payload["count"] == 1
    assert seen == {"status": "HELD", "limit": 5, "owner": "owner-a"}


@pytest.mark.asyncio
async def _slow_case_booking_cancel_returns_404_when_not_owner(monkeypatch):
    seen = {}

    def fake_cancel_booking(booking_id, owner_principal_id=None):
        seen["booking_id"] = booking_id
        seen["owner"] = owner_principal_id
        return False

    monkeypatch.setattr("tools.booking_handoff.cancel_booking", fake_cancel_booking)
    principal = AuthenticatedPrincipal(principal_id="owner-b")

    with pytest.raises(HTTPException) as exc:
        await booking_tracking_service.booking_cancel(booking_id=7, principal=principal)

    assert exc.value.status_code == 404
    assert seen == {"booking_id": 7, "owner": "owner-b"}


@pytest.mark.asyncio
async def _slow_case_price_tracking_alerts_scopes_by_owner(monkeypatch):
    seen = {}

    def fake_alerts(booking_id=None, owner_principal_id=None):
        seen["booking_id"] = booking_id
        seen["owner"] = owner_principal_id
        return [{"alert_id": 1}]

    monkeypatch.setattr("tools.price_tracker.get_unacknowledged_alerts", fake_alerts)
    principal = AuthenticatedPrincipal(principal_id="owner-c")

    payload = await booking_tracking_service.price_tracking_alerts(
        principal=principal,
        booking_id=42,
    )

    assert payload["count"] == 1
    assert seen == {"booking_id": 42, "owner": "owner-c"}


@pytest.mark.asyncio
async def _slow_case_price_tracking_ack_returns_404_when_not_owner(monkeypatch):
    seen = {}

    def fake_ack(alert_id, owner_principal_id=None):
        seen["alert_id"] = alert_id
        seen["owner"] = owner_principal_id
        return False

    monkeypatch.setattr("tools.price_tracker.acknowledge_alert", fake_ack)
    principal = AuthenticatedPrincipal(principal_id="owner-d")

    with pytest.raises(HTTPException) as exc:
        await booking_tracking_service.price_tracking_ack(9, principal=principal)

    assert exc.value.status_code == 404
    assert seen == {"alert_id": 9, "owner": "owner-d"}
