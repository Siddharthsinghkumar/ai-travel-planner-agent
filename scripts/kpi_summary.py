#!/usr/bin/env python3
"""KPI summary script — reads eval_results/kpi_events.jsonl and prints a markdown table."""

import json
import sys
from pathlib import Path
from collections import defaultdict

LOG_PATH = Path("eval_results/kpi_events.jsonl")


def main():
    if not LOG_PATH.exists():
        print(f"No KPI events found at {LOG_PATH}")
        print("Run a booking flow first to generate events.")
        sys.exit(1)

    events = []
    for line in LOG_PATH.read_text().strip().split("\n"):
        if line.strip():
            events.append(json.loads(line))

    if not events:
        print("No events in log file.")
        sys.exit(1)

    # Group by plan_id
    plans = defaultdict(dict)
    for ev in events:
        pid = ev.get("plan_id", "unknown")
        evt = ev.get("event", "")
        if evt == "plan_start":
            plans[pid]["start_ts"] = ev.get("ts")
            plans[pid]["intent_type"] = ev.get("intent_type", "unknown")
        elif evt == "approval_requested":
            plans[pid]["approval_requested"] = True
            plans[pid]["approval_action"] = ev.get("action_type", "")
        elif evt == "approval_decision":
            plans[pid]["approved"] = ev.get("approved", False)
            plans[pid]["approval_latency_ms"] = ev.get("latency_ms", 0)
        elif evt == "plan_complete":
            plans[pid]["complete"] = True
            plans[pid]["duration_ms"] = ev.get("total_duration_ms", 0)
            plans[pid]["action_count"] = ev.get("action_count", 0)
        elif evt == "plan_error":
            plans[pid]["error"] = True
            plans[pid]["error_type"] = ev.get("error_type", "unknown")

    total_plans = len(plans)
    completed = sum(1 for p in plans.values() if p.get("complete"))
    errors = sum(1 for p in plans.values() if p.get("error"))
    approval_requested = sum(1 for p in plans.values() if p.get("approval_requested"))
    approval_granted = sum(1 for p in plans.values() if p.get("approved"))

    durations = [p["duration_ms"] for p in plans.values() if "duration_ms" in p]
    approval_latencies = [p["approval_latency_ms"] for p in plans.values() if "approval_latency_ms" in p]

    median_duration = sorted(durations)[len(durations) // 2] if durations else 0
    median_approval_latency = sorted(approval_latencies)[len(approval_latencies) // 2] if approval_latencies else 0
    approval_grant_rate = approval_granted / approval_requested if approval_requested else 0.0
    errors_per_100 = (errors / total_plans * 100) if total_plans else 0.0

    print("## KPI Summary")
    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Total plans | {total_plans} |")
    print(f"| Completed | {completed} |")
    print(f"| Errors | {errors} |")
    print(f"| Median plan duration | {median_duration} ms |")
    print(f"| Approval grant rate | {approval_grant_rate:.1%} |")
    print(f"| Median approval latency | {median_approval_latency} ms |")
    print(f"| Errors per 100 plans | {errors_per_100:.1f} |")
    print()

    if plans:
        print("### Per-Plan Details")
        print()
        print("| Plan ID | Intent | Duration (ms) | Approved | Error |")
        print("|---------|--------|---------------|----------|-------|")
        for pid, p in sorted(plans.items()):
            intent = p.get("intent_type", "n/a")
            duration = p.get("duration_ms", "n/a")
            approved = "Yes" if p.get("approved") else ("No" if p.get("approval_requested") else "n/a")
            error = p.get("error_type", "") if p.get("error") else ""
            print(f"| {pid[:20]}... | {intent} | {duration} | {approved} | {error} |")


if __name__ == "__main__":
    main()
