import json
import time
from pathlib import Path
from threading import Lock

_LOG_PATH = Path("eval_results/kpi_events.jsonl")
_LOCK = Lock()


def log_event(event_type: str, plan_id: str, **fields):
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), "event": event_type, "plan_id": plan_id, **fields}
    with _LOCK:
        with _LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
