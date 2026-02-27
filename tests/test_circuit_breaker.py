import pytest
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
