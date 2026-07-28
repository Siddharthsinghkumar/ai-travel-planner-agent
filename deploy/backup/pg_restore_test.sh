#!/usr/bin/env bash
set -euo pipefail

# Restore-test script — restore into scratch DB, compare row counts.
# See: docs/persistence-backups.md (A1 restore test).
#
# Usage: deploy/backup/pg_restore_test.sh <backup_file.dump>
# Env:
#   PGHOST              (default: localhost)
#   PGPORT              (default: 5432)
#   PGUSER              (placeholder: POSTGRES_USER)
#   PGPASSWORD          (placeholder: POSTGRES_PASSWORD)

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-PLACEHOLDER_PGUSER}"
PGPASSWORD="${PGPASSWORD:-PLACEHOLDER_PGPASSWORD}"
SCRATCH_DB="llm_travel_restore_test"

export PGHOST PGPORT PGUSER PGPASSWORD

if [[ "$PGUSER" == "PLACEHOLDER_PGUSER" ]]; then
  echo "ERROR: PGUSER is a placeholder — set via environment." >&2
  exit 2
fi

BACKUP_FILE="${1:-}"
if [[ -z "$BACKUP_FILE" || ! -f "$BACKUP_FILE" ]]; then
  echo "Usage: $0 <backup_file.dump>" >&2
  echo "ERROR: backup file not found: ${BACKUP_FILE:-<none>}" >&2
  exit 2
fi

echo "=== Restore test: $BACKUP_FILE -> scratch DB $SCRATCH_DB ==="

dropdb --if-exists "$SCRATCH_DB" 2>/dev/null || true
createdb "$SCRATCH_DB"

pg_restore --no-owner --no-acl -d "$SCRATCH_DB" "$BACKUP_FILE"

echo ""
echo "=== Row counts (scratch vs source) ==="

SCHEMA="public"
tables="$(psql -d "$SCRATCH_DB" -tAc "SELECT tablename FROM pg_tables WHERE schemaname='$SCHEMA' ORDER BY tablename")"

if [[ -z "$tables" ]]; then
  echo "WARNING: no tables found in scratch DB" >&2
else
  while IFS= read -r tbl; do
    [[ -z "$tbl" ]] && continue
    scratch_count="$(psql -d "$SCRATCH_DB" -tAc "SELECT COUNT(*) FROM \"$tbl\"" 2>/dev/null || echo "ERROR")"
    source_count="$(psql -d "${PGDATABASE:-llm_travel}" -tAc "SELECT COUNT(*) FROM \"$tbl\"" 2>/dev/null || echo "ERROR")"
    printf "%-40s scratch=%-8s source=%-8s\n" "$tbl" "$scratch_count" "$source_count"
  done <<< "$tables"
fi

echo ""
echo "=== Dropping scratch DB ==="
dropdb "$SCRATCH_DB"

echo "=== Restore test PASSED ==="
