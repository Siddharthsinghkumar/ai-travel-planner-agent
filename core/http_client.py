# core/http_client.py
import asyncio
import httpx
from weakref import WeakKeyDictionary

# Check if HTTP/2 is supported (requires 'h2' package)
try:
    import h2  # noqa: F401
    HTTP2_ENABLED = True
except ImportError:
    HTTP2_ENABLED = False

# One client per event loop
_clients: WeakKeyDictionary = WeakKeyDictionary()


def get_client() -> httpx.AsyncClient:
    """
    Returns a shared AsyncClient bound to the current running event loop.
    Safe for FastAPI and async usage.
    """
    loop = asyncio.get_running_loop()

    client = _clients.get(loop)
    if client is not None:
        return client

    # Create client lazily
    timeout = httpx.Timeout(
        connect=5.0,
        read=15.0,
        write=5.0,
        pool=5.0
    )

    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20
    )

    client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
        http2=HTTP2_ENABLED,  # Auto‑enable if h2 is installed
    )

    _clients[loop] = client
    return client


async def close_client():
    """
    Gracefully close all AsyncClient instances.
    Call this during application shutdown.
    """
    for client in list(_clients.values()):
        try:
            await client.aclose()
        except Exception:
            pass

    _clients.clear()