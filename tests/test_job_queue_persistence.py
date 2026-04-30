import asyncio
import os
import pytest
import json
import uuid
from core import job_queue

@pytest.mark.asyncio
async def test_job_persistence():
    # Setup: ensure we use a temporary DB
    test_db = f"test_jobs_{uuid.uuid4().hex}.db"
    os.environ["DATABASE_URL"] = f"sqlite:///./{test_db}"
    
    # Reset job_queue module state
    job_queue._jobs.clear()
    job_queue._job_event_queues.clear()
    job_queue.DB_PATH = test_db
    
    try:
        # 1. Initialize
        await job_queue.initialize_job_queue()
        
        # 2. Enqueue a job
        payload = {"test": "data"}
        job_id = await job_queue.enqueue_job(payload, owner_principal_id="test_user")
        
        # Verify it's in memory
        job = await job_queue.get_job(job_id)
        assert job is not None
        assert job["status"] == "queued"
        
        # 3. Simulate "restart" by clearing memory and re-initializing
        job_queue._jobs.clear()
        job_queue._job_event_queues.clear()
        # Empty the queue
        while not job_queue._queue.empty():
            job_queue._queue.get_nowait()
            
        await job_queue.initialize_job_queue()
        
        # 4. Verify job is restored
        restored_job = await job_queue.get_job(job_id)
        assert restored_job is not None
        assert restored_job["job_id"] == job_id
        assert restored_job["owner_principal_id"] == "test_user"
        assert restored_job["status"] == "queued"
        
        # 5. Verify it was re-enqueued
        assert not job_queue._queue.empty()
        qid, qpayload = job_queue._queue.get_nowait()
        assert qid == job_id
        assert qpayload == payload
        
        # 6. Update status and verify persistence
        await job_queue._set_job_status(job_id, "done", result={"success": True})
        
        # Clear memory again
        job_queue._jobs.clear()
        await job_queue.initialize_job_queue()
        
        final_job = await job_queue.get_job(job_id)
        assert final_job["status"] == "done"
        assert final_job["result"] == {"success": True}

    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.remove(test_db)
        # Restore environment or just let it be since it's a test process
