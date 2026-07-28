#!/usr/bin/env bash
# deploy/security/nuclei-scan.sh — nuclei scan wrapper
# Usage: NUCLEI_TARGET=https://<domain> ./deploy/security/nuclei-scan.sh
# Reports land in plans/qa/ (untracked).

set -euo pipefail

TARGET_URL="${NUCLEI_TARGET:-${TARGET_URL:-}}"
if [ -z "${TARGET_URL:-}" ]; then
  echo "ERROR: Set NUCLEI_TARGET (or TARGET_URL)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORT_DIR="$REPO_ROOT/plans/qa"
mkdir -p "$REPORT_DIR"

NOW="$(date -u +%Y%m%d-%H%M%S)"
REPORT_FILE="$REPORT_DIR/nuclei-${NOW}.txt"

# Template set: http/exposures, misconfiguration, ssl, plus http/technologies
# These directories are shipped with nuclei. If nuclei is not installed, the
# live run uses the official Docker image: projectdiscovery/nuclei.
TEMPLATES=(
  "http/exposures"
  "http/misconfiguration"
  "ssl"
)

TEMPLATE_ARGS=()
for t in "${TEMPLATES[@]}"; do
  TEMPLATE_ARGS+=(-t "$t")
done

echo "=== nuclei scan ==="
echo "Target:  $TARGET_URL"
echo "Report:  $REPORT_FILE"
echo ""

if command -v nuclei &>/dev/null; then
  nuclei -u "$TARGET_URL" "${TEMPLATE_ARGS[@]}" -o "$REPORT_FILE" -silent
  EXIT_CODE=$?
else
  echo "nuclei not installed — running via Docker"
  # nuclei default template dir inside the image: /root/nuclei-templates
  DOCKER_TEMPLATES=()
  for t in "${TEMPLATES[@]}"; do
    DOCKER_TEMPLATES+=(-t "/root/nuclei-templates/$t")
  done
  docker run --rm \
    -v "$REPORT_DIR":/output \
    projectdiscovery/nuclei:latest \
    -u "$TARGET_URL" \
    "${DOCKER_TEMPLATES[@]}" \
    -o "/output/nuclei-${NOW}.txt" \
    -silent
  EXIT_CODE=$?
fi

echo ""
echo "=== Done (exit $EXIT_CODE) ==="
echo "Report: $REPORT_FILE"
echo "NOTE: §6 M2-T12 rule: new HIGH finding → STOP."
