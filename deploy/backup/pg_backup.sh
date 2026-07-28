#!/usr/bin/env bash
set -euo pipefail

# PostgreSQL backup script for single-node production.
# See: docs/persistence-backups.md
#
# Usage: deploy/backup/pg_backup.sh
# Env:
#   PGHOST              (default: localhost)
#   PGPORT              (default: 5432)
#   PGUSER              (placeholder: POSTGRES_USER)
#   PGPASSWORD          (placeholder: POSTGRES_PASSWORD)
#   PGDATABASE          (placeholder: POSTGRES_DB)
#   BACKUP_DIR          (default: /var/backups/llm-travel-agent)
#   RETAIN_COUNT         (default: 14)

BACKUP_DIR="${BACKUP_DIR:-/var/backups/llm-travel-agent}"
RETAIN_COUNT="${RETAIN_COUNT:-14}"
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-PLACEHOLDER_PGUSER}"
PGPASSWORD="${PGPASSWORD:-PLACEHOLDER_PGPASSWORD}"
PGDATABASE="${PGDATABASE:-PLACEHOLDER_PGDATABASE}"

export PGHOST PGPORT PGUSER PGPASSWORD PGDATABASE

if [[ "$PGUSER" == "PLACEHOLDER_PGUSER" ]]; then
  echo "ERROR: PGUSER is a placeholder — set via environment." >&2
  exit 2
fi

mkdir -p "$BACKUP_DIR"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_file="$BACKUP_DIR/pg_backup_${stamp}.dump"

pg_dump -Fc --no-owner --no-acl -f "$out_file"
sha256sum "$out_file" > "$out_file.sha256"

mapfile -t backups < <(ls -1t "$BACKUP_DIR"/pg_backup_*.dump 2>/dev/null || true)
if (( ${#backups[@]} > RETAIN_COUNT )); then
  for old in "${backups[@]:RETAIN_COUNT}"; do
    rm -f "$old" "$old.sha256"
  done
fi

echo "backup_path=$out_file"
echo "checksum_path=$out_file.sha256"
echo "retain_count=$RETAIN_COUNT"
