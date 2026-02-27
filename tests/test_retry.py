import pytest
from core.retry import retry_async, RetryConfig

@pytest.mark.asyncio
async def test_retry_retries_and_succeeds():
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception("fail")
        return "ok"

    result = await retry_async(
        flaky,
        config=RetryConfig(retries=3, base_delay=0),
        retry_exceptions=(Exception,)
    )

    assert result == "ok"
    assert attempts == 3
