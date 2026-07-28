#!/usr/bin/env python3
"""
HITL Audit Summary — CLI tool to display audit metrics from JSONL log files.

Usage:
    python scripts/hitl_audit_summary.py
    python scripts/hitl_audit_summary.py --date 2026-04-29
    python scripts/hitl_audit_summary.py --json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.hitl_audit import HITLAuditLogger  # noqa: E402 — script-level sys.path.insert before import


def main() -> None:
    parser = argparse.ArgumentParser(description="HITL Audit Summary")
    parser.add_argument("--date", help="Filter by date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--audit-dir", default="logs/hitl_audit", help="Audit log directory")
    args = parser.parse_args()

    logger = HITLAuditLogger(audit_dir=args.audit_dir)
    metrics = logger.get_metrics(date=args.date)

    if getattr(args, "json"):
        print(json.dumps(metrics, indent=2))
        return

    print("=" * 60)
    print("HITL Audit Summary" + (f" — {args.date}" if args.date else ""))
    print("=" * 60)
    print(f"  Total requests:    {metrics['total_requests']}")
    print(f"  Total decisions:   {metrics['total_decisions']}")
    print(f"  Approved:          {metrics['approved']}")
    print(f"  Rejected:          {metrics['rejected']}")
    print(f"  Approval rate:     {metrics['approval_rate']:.1%}")
    print(f"  Rejection rate:    {metrics['rejection_rate']:.1%}")
    print(f"  Avg latency:       {metrics['avg_latency_ms']:.1f} ms")
    print(f"  P50 latency:       {metrics['p50_latency_ms']:.1f} ms")
    print(f"  P95 latency:       {metrics['p95_latency_ms']:.1f} ms")
    print("=" * 60)

    trail = logger.get_audit_trail(date=args.date)
    if trail:
        print("\nRecent entries (last 10):")
        print(f"  {'Timestamp':<28} {'Event':<10} {'Plan ID':<38} {'User':<12} {'Result'}")
        print("  " + "-" * 110)
        for entry in trail[-10:]:
            ts = entry.get("timestamp", "")[:26]
            event = entry.get("event", "")
            plan_id = entry.get("plan_id", "")[:36]
            user = entry.get("user_id", "")[:10]
            if event == "decision":
                result = "APPROVED" if entry.get("approved") else "REJECTED"
            else:
                result = entry.get("action", "")
            print(f"  {ts:<28} {event:<10} {plan_id:<38} {user:<12} {result}")


if __name__ == "__main__":
    main()
