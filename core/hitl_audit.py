"""
HITL Audit Logger — structured JSONL audit trail for human-in-the-loop approval decisions.

Each day gets its own JSONL file under logs/hitl_audit/.
Entries contain timestamps, plan IDs, user IDs, decisions, and latency metrics.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class HITLAuditLogger:
    """Thread-safe JSONL audit logger for HITL approval gates."""

    def __init__(self, audit_dir: str = "logs/hitl_audit") -> None:
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _today_file(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.audit_dir / f"audit_{today}.jsonl"

    def log_request(
        self,
        plan_id: str,
        user_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "request",
            "plan_id": plan_id,
            "user_id": user_id,
            "action": action,
            "details": details or {},
        }
        self._append(entry)

    def log_decision(
        self,
        plan_id: str,
        user_id: str,
        approved: bool,
        latency_ms: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "decision",
            "plan_id": plan_id,
            "user_id": user_id,
            "approved": approved,
            "latency_ms": round(latency_ms, 2),
            "details": details or {},
        }
        self._append(entry)

    def _append(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            path = self._today_file()
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_audit_trail(
        self,
        plan_id: Optional[str] = None,
        user_id: Optional[str] = None,
        date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if date:
            files = [self.audit_dir / f"audit_{date}.jsonl"]
        else:
            files = sorted(self.audit_dir.glob("audit_*.jsonl"))

        results: List[Dict[str, Any]] = []
        for fpath in files:
            if not fpath.exists():
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if plan_id and entry.get("plan_id") != plan_id:
                        continue
                    if user_id and entry.get("user_id") != user_id:
                        continue
                    results.append(entry)
        return results

    def get_metrics(self, date: Optional[str] = None) -> Dict[str, Any]:
        trail = self.get_audit_trail(date=date)
        decisions = [e for e in trail if e.get("event") == "decision"]
        requests = [e for e in trail if e.get("event") == "request"]

        total_requests = len(requests)
        total_decisions = len(decisions)
        approved = sum(1 for d in decisions if d.get("approved"))
        rejected = total_decisions - approved

        latencies = [d["latency_ms"] for d in decisions if "latency_ms" in d]
        latencies.sort()

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p50_latency = latencies[len(latencies) // 2] if latencies else 0.0
        p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

        approval_rate = approved / total_decisions if total_decisions > 0 else 0.0
        rejection_rate = rejected / total_decisions if total_decisions > 0 else 0.0

        return {
            "total_requests": total_requests,
            "total_decisions": total_decisions,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": round(approval_rate, 4),
            "rejection_rate": round(rejection_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "p50_latency_ms": round(p50_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
        }
