#!/usr/bin/env bash
set -euo pipefail

# Canonical deployment smoke test for this repo (Phase 7D.3).
# Verifies readiness, lightweight/deep health, and one /ask contract response.

usage() {
  echo "Usage: BASE_URL=https://travel.example.com $0"
  echo "   or: $0 https://travel.example.com"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

BASE_URL="${BASE_URL:-${1:-}}"
if [[ -z "${BASE_URL}" ]]; then
  usage
  exit 2
fi

INSECURE_TLS="${INSECURE_TLS:-0}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-120}"
READY_POLL_SEC="${READY_POLL_SEC:-3}"
SMOKE_ORIGIN="${SMOKE_ORIGIN:-DEL}"
SMOKE_DESTINATION="${SMOKE_DESTINATION:-BOM}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERROR: python3 or python is required for deploy_smoke date/json parsing."
    exit 2
  fi
fi

SMOKE_DATE="${SMOKE_DATE:-$(${PYTHON_BIN} - <<'PY'
from datetime import datetime, timezone, timedelta
print((datetime.now(timezone.utc) + timedelta(days=21)).strftime("%Y-%m-%d"))
PY
)}"

CURL_COMMON=(-sS --connect-timeout 5 --max-time 30)
if [[ "${INSECURE_TLS}" == "1" ]]; then
  CURL_COMMON+=(-k)
fi

tmpdir="$(mktemp -d /tmp/deploy_smoke.XXXXXX)"
trap 'rm -rf "$tmpdir"' EXIT

request_json() {
  local method="$1"
  local path="$2"
  local body_file="${3:-}"
  local out_file="$4"
  local status

  if [[ -n "${body_file}" ]]; then
    status="$(curl "${CURL_COMMON[@]}" -X "${method}" \
      -H "Content-Type: application/json" \
      --data @"${body_file}" \
      -o "${out_file}" -w "%{http_code}" \
      "${BASE_URL}${path}")"
  else
    status="$(curl "${CURL_COMMON[@]}" -X "${method}" \
      -o "${out_file}" -w "%{http_code}" \
      "${BASE_URL}${path}")"
  fi
  echo "${status}"
}

echo "==> Smoke target: ${BASE_URL}"
echo "==> Waiting for readiness (/health/ready) ..."

ready_status_file="${tmpdir}/ready.json"
deadline=$((SECONDS + READY_TIMEOUT_SEC))
ready_code=""

while (( SECONDS < deadline )); do
  ready_code="$(request_json GET /health/ready "" "${ready_status_file}" || true)"
  if [[ "${ready_code}" == "200" ]]; then
    break
  fi
  sleep "${READY_POLL_SEC}"
done

if [[ "${ready_code}" != "200" ]]; then
  echo "ERROR: /health/ready did not become 200 within ${READY_TIMEOUT_SEC}s (last code=${ready_code:-none})."
  if [[ -s "${ready_status_file}" ]]; then
    cat "${ready_status_file}"
  fi
  exit 1
fi

${PYTHON_BIN} - <<'PY' "${ready_status_file}"
import json,sys
p=sys.argv[1]
d=json.load(open(p,"r",encoding="utf-8"))
print("readiness_status:", d.get("status"))
print("readiness_prewarm:", (d.get("llm_prewarm") or {}).get("status"))
PY

echo "==> Checking /health ..."
health_file="${tmpdir}/health.json"
health_code="$(request_json GET /health "" "${health_file}")"
if [[ "${health_code}" != "200" ]]; then
  echo "ERROR: /health returned HTTP ${health_code}"
  cat "${health_file}" || true
  exit 1
fi

${PYTHON_BIN} - <<'PY' "${health_file}"
import json,sys
d=json.load(open(sys.argv[1],"r",encoding="utf-8"))
print("health_status:", d.get("status"))
deps=d.get("dependencies") or {}
print("health_dependencies:", {
    "app": deps.get("app"),
    "key_manager": deps.get("key_manager"),
    "database": deps.get("database"),
    "ollama": deps.get("ollama"),
    "cloud": deps.get("cloud"),
})
rt=(d.get("runtime_topology") or {})
print("runtime_topology:", {
    "worker_role": rt.get("worker_role"),
    "async_jobs_enabled": rt.get("async_jobs_enabled"),
})
PY

echo "==> Checking /health/deep (diagnostic) ..."
deep_file="${tmpdir}/health_deep.json"
deep_code="$(request_json GET /health/deep "" "${deep_file}")"
if [[ "${deep_code}" != "200" ]]; then
  echo "ERROR: /health/deep returned HTTP ${deep_code}"
  cat "${deep_file}" || true
  exit 1
fi

${PYTHON_BIN} - <<'PY' "${deep_file}"
import json,sys
d=json.load(open(sys.argv[1],"r",encoding="utf-8"))
print("deep_status:", d.get("status"))
summary=d.get("dependency_summary") or {}
print("deep_dependency_summary:", {
    "failed": summary.get("failed"),
    "degraded": summary.get("degraded"),
    "unavailable": summary.get("unavailable"),
})
PY

echo "==> Running /ask smoke case (${SMOKE_ORIGIN} -> ${SMOKE_DESTINATION} on ${SMOKE_DATE}) ..."
ask_req="${tmpdir}/ask.json"
cat > "${ask_req}" <<JSON
{
  "origin": "${SMOKE_ORIGIN}",
  "destination": "${SMOKE_DESTINATION}",
  "date": "${SMOKE_DATE}",
  "trip_type": "one-way",
  "user_query": ""
}
JSON

ask_file="${tmpdir}/ask_resp.json"
ask_code="$(request_json POST /ask "${ask_req}" "${ask_file}")"
if [[ "${ask_code}" != "200" ]]; then
  echo "ERROR: /ask returned HTTP ${ask_code}"
  cat "${ask_file}" || true
  exit 1
fi

${PYTHON_BIN} - <<'PY' "${ask_file}"
import json,sys
d=json.load(open(sys.argv[1],"r",encoding="utf-8"))
best=d.get("best_flight") or {}
handoff=(best.get("booking_handoff") or {})
print("ask_result_status:", d.get("result_status"))
print("ask_failure_reason:", d.get("failure_reason"))
print("ask_primary_handoff_url:", best.get("handoff_url"))
print("ask_booking_exit_quality:", handoff.get("booking_exit_quality"))
print("ask_booking_availability:", handoff.get("booking_availability"))
PY

echo "==> Smoke PASS (service reachable, readiness ok, health/deep responded, /ask contract returned JSON)."
