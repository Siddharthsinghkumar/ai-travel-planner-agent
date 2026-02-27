import pytest
from core.http_client import get_client, close_client

@pytest.mark.asyncio
async def test_http_client_shutdown():

    client = get_client()
    assert client is not None

    await close_client()

    assert client.is_closed
