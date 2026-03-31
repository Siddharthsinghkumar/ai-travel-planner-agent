#!/usr/bin/env bash
set -euo pipefail

# Quick command pack for showcase-safe backend invocation.
# Usage:
#   bash docs/showcase-commands.sh
# Then copy/run the printed commands or uncomment function calls below.

BASE="${BASE:-http://127.0.0.1:8000}"
DATE_ONE_WAY="${DATE_ONE_WAY:-2030-01-15}"
DATE_STREAM="${DATE_STREAM:-2030-01-16}"
DATE_RT_OUT="${DATE_RT_OUT:-2030-01-20}"
DATE_RT_BACK="${DATE_RT_BACK:-2030-01-24}"

must_1_health() {
  curl -sS "$BASE/health" | jq
}

must_2_llm_options() {
  curl -sS "$BASE/llm/options" | jq
}

must_3_ask_nonstream() {
  curl -sS -X POST "$BASE/ask" \
    -H "Content-Type: application/json" \
    -d "{\"origin\":\"DEL\",\"destination\":\"BOM\",\"date\":\"$DATE_ONE_WAY\",\"trip_type\":\"one-way\",\"user_query\":\"Find the best value option and explain why.\"}" | jq
}

must_4_ask_stream() {
  curl -N -sS -X POST "$BASE/ask?stream=true" \
    -H "Content-Type: application/json" \
    -d "{\"origin\":\"DEL\",\"destination\":\"BLR\",\"date\":\"$DATE_STREAM\",\"trip_type\":\"one-way\",\"user_query\":\"Give a concise recommendation with tradeoffs.\"}"
}

must_5_round_trip() {
  curl -sS -X POST "$BASE/ask" \
    -H "Content-Type: application/json" \
    -d "{\"user_query\":\"Round-trip Delhi to Mumbai, leave $DATE_RT_OUT and return $DATE_RT_BACK, prioritize cheapest acceptable option.\"}" | jq
}

diag_1_health_deep() {
  curl -sS "$BASE/health/deep" | jq
}

diag_2_health_keys() {
  curl -sS "$BASE/health/keys" | jq
}

diag_3_health_ready() {
  curl -sS "$BASE/health/ready" | jq
}

diag_4_version() {
  curl -sS "$BASE/version" | jq
}

diag_5_metrics_head() {
  curl -sS "$BASE/metrics" | head -n 40
}

cat <<EOF
BASE=$BASE

Must-have demo commands:
  must_1_health
  must_2_llm_options
  must_3_ask_nonstream
  must_4_ask_stream
  must_5_round_trip

Fallback diagnostic commands:
  diag_1_health_deep
  diag_2_health_keys
  diag_3_health_ready
  diag_4_version
  diag_5_metrics_head

Notes:
  - Admin/debug endpoints are intentionally excluded.
  - Unsafe async override flows are intentionally excluded.
  - Async jobs can be topology-guarded in multi-worker mode.
EOF

# Optional quick run (uncomment as needed):
# must_1_health
# must_2_llm_options
# must_3_ask_nonstream
