#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 https://<public-domain>/health/live"
  echo "   or: TARGET_URL=https://<public-domain>/health/live $0"
  echo "Optional: INSECURE_TLS=1 to skip certificate verification for temporary diagnostics"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

TARGET_URL="${1:-${TARGET_URL:-}}"
if [[ -z "${TARGET_URL}" ]]; then
  usage
  exit 2
fi

if [[ "${TARGET_URL}" != https://* ]]; then
  echo "ERROR: target must be HTTPS for production header validation: ${TARGET_URL}" >&2
  exit 2
fi

CURL_TLS_FLAGS=()
if [[ "${INSECURE_TLS:-0}" == "1" ]]; then
  CURL_TLS_FLAGS=(-k)
fi

RAW_HEADERS="$(curl -sS "${CURL_TLS_FLAGS[@]}" -D - -o /dev/null "${TARGET_URL}")"
STATUS_CODE="$(printf '%s\n' "${RAW_HEADERS}" | awk '/^HTTP\// {code=$2} END {print code}')"
if [[ -z "${STATUS_CODE}" ]]; then
  echo "ERROR: could not read HTTP status from response." >&2
  exit 1
fi

get_header() {
  local name="$1"
  printf '%s\n' "${RAW_HEADERS}" \
    | grep -im1 "^${name}:" \
    | sed -E 's/^[^:]+:[[:space:]]*//'
}

check_contains() {
  local header_name="$1"
  local must_contain="$2"
  local value
  value="$(get_header "${header_name}")"
  if [[ -z "${value}" ]]; then
    echo "FAIL missing header: ${header_name}"
    return 1
  fi
  if [[ "${value,,}" != *"${must_contain,,}"* ]]; then
    echo "FAIL header ${header_name} value mismatch: ${value}"
    return 1
  fi
  echo "PASS ${header_name}: ${value}"
  return 0
}

echo "Target: ${TARGET_URL}"
echo "Status: ${STATUS_CODE}"
echo "Checking security headers..."

failures=0
check_contains "Strict-Transport-Security" "max-age=" || failures=$((failures + 1))
check_contains "X-Content-Type-Options" "nosniff" || failures=$((failures + 1))
check_contains "X-Frame-Options" "deny" || failures=$((failures + 1))
check_contains "Referrer-Policy" "strict-origin-when-cross-origin" || failures=$((failures + 1))
check_contains "Content-Security-Policy" "frame-ancestors" || failures=$((failures + 1))

if [[ "${failures}" -gt 0 ]]; then
  echo "Header verification FAILED (${failures} checks failed)." >&2
  exit 1
fi

echo "Header verification PASS."
