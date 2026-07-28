#!/usr/bin/env bash
# deploy/security/zap-full.sh — OWASP ZAP full scan wrapper
# Usage: ZAP_FULL_TARGET=https://<domain> ./deploy/security/zap-full.sh
# Reports land in plans/qa/ (untracked).

set -euo pipefail

TARGET_URL="${ZAP_FULL_TARGET:-${TARGET_URL:-}}"
if [ -z "${TARGET_URL:-}" ]; then
  echo "ERROR: Set ZAP_FULL_TARGET (or TARGET_URL)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORT_DIR="$REPO_ROOT/plans/qa"
mkdir -p "$REPORT_DIR"

NOW="$(date -u +%Y%m%d-%H%M%S)"
REPORT_NAME="zap-full-${NOW}"

TUNING_FILE="$SCRIPT_DIR/zap-full.tsv"
RULES_ARG=()
if [ -f "$TUNING_FILE" ]; then
  RULES_ARG=(-c "$TUNING_FILE")
fi

echo "=== ZAP full scan ==="
echo "Target:  $TARGET_URL"
echo "Report:  $REPORT_DIR/${REPORT_NAME}.md"
echo ""

docker run --rm \
  -v "$REPORT_DIR":/zap/wrk \
  ghcr.io/zaproxy/zaproxy:stable \
  zap-full-scan.py \
  -t "$TARGET_URL" \
  "${RULES_ARG[@]}" \
  -r "${REPORT_NAME}.md" \
  -w "${REPORT_NAME}.md" \
  -I

echo ""
echo "=== Done ==="
echo "Report: $REPORT_DIR/${REPORT_NAME}.md"
echo "NOTE: §6 M2-T12 rule: new HIGH findings → STOP."
