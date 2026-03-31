import pytest
import asyncio
from core.circuit_breaker import AsyncCircuitBreaker

@pytest.mark.asyncio
async def test_circuit_breaker_opens():
    breaker = AsyncCircuitBreaker(failure_threshold=2, recovery_timeout=10)

    async def fail():
        raise Exception("fail")

    # Two failures
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(fail)

    # Now it should be open
    with pytest.raises(Exception):
        await breaker.call(fail)

    assert breaker._state == "open"


@pytest.mark.asyncio
async def test_circuit_breaker_cancellation_is_neutral_by_default():
    breaker = AsyncCircuitBreaker(failure_threshold=1, recovery_timeout=10)

    async def cancelled():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await breaker.call(cancelled)

    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_circuit_breaker_can_treat_cancellation_as_failure():
    breaker = AsyncCircuitBreaker(failure_threshold=1, recovery_timeout=10)

    async def cancelled():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await breaker.call(cancelled, treat_cancelled_as_failure=True)

    assert breaker.state == "open"
