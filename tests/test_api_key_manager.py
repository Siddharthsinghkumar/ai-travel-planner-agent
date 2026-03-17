#tests/test_api_key_manager.py
import pytest
import time
from core.api_key_manager import APIKeyManager, KeyEntry

@pytest.mark.asyncio
async def test_key_rotation_and_exhaustion():
    # 1. Create a fresh key manager and load fake keys using KeyEntry dataclass
    km = APIKeyManager()
    km._keys = {
        "fake_api": [
            KeyEntry(value="key1", fingerprint="hash1", exhausted_until=None),
            KeyEntry(value="key2", fingerprint="hash2", exhausted_until=None)
        ]
    }
    km._rr_index = {"fake_api": 0}

    # 2. Test Reservation (Should get key1)
    async with km.reserve_key("fake_api") as (idx1, key_val1):
        assert key_val1 == "key1"
        assert idx1 == 0

    # 3. Test Rotation (Should get key2 next)
    async with km.reserve_key("fake_api") as (idx2, key_val2):
        assert key_val2 == "key2"
        assert idx2 == 1

    # 4. Test Exhaustion 
    # Mark key2 as exhausted for 10 seconds
    future_time = time.time() + 10
    await km.mark_exhausted("fake_api", 1, until=future_time, reason="test_quota")
    
    # Verify key2 is skipped and it wraps around to key1
    async with km.reserve_key("fake_api") as (idx3, key_val3):
        assert key_val3 == "key1"
        assert idx3 == 0

    # 5. Test All Keys Exhausted
    await km.mark_exhausted("fake_api", 0, until=future_time, reason="test_quota")
    
    # Should raise RuntimeError because no keys are available
    with pytest.raises(RuntimeError, match="No available keys"):
        async with km.reserve_key("fake_api") as (idx4, key_val4):
            pass