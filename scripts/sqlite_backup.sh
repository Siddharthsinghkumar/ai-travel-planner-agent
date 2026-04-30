#!/usr/bin/env bash
set -euo pipefail

# Canonical SQLite backup helper for Phase 7D.5.
# Uses SQLite-supported VACUUM INTO to create a consistent snapshot.

usage() {
  cat <<'USAGE'
Usage:
  scripts/sqlite_backup.sh [DB_PATH]

Inputs:
  DB_PATH       Optional explicit SQLite DB file path.

Environment:
  DATABASE_URL  Optional. Used when DB_PATH is omitted.
                Expected sqlite URL, e.g. sqlite:////var/lib/llm-travel-agent/local.db
  BACKUP_DIR    Optional. Default: /var/backups/llm-travel-agent
  RETAIN_COUNT  Optional. Default: 14

Examples:
  DATABASE_URL='sqlite:////var/lib/llm-travel-agent/local.db' scripts/sqlite_backup.sh
  BACKUP_DIR='/var/backups/llm-travel-agent' RETAIN_COUNT=30 scripts/sqlite_backup.sh /var/lib/llm-travel-agent/local.db
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "ERROR: sqlite3 is required for VACUUM INTO backups." >&2
  exit 2
fi

resolve_db_path() {
  local arg_path="${1:-}"
  if [[ -n "$arg_path" ]]; then
    printf '%s\n' "$arg_path"
    return 0
  fi

  local db_url="${DATABASE_URL:-}"
  if [[ -z "$db_url" ]]; then
    printf '%s\n' ""
    return 0
  fi

  if [[ "$db_url" != sqlite:///* ]]; then
    echo "ERROR: DATABASE_URL must be a sqlite file URL (got: $db_url)" >&2
    exit 2
  fi

  local path_part="${db_url#sqlite:///}"
  path_part="${path_part%%\?*}"
  printf '%s\n' "$path_part"
}

DB_PATH="$(resolve_db_path "${1:-}")"
if [[ -z "$DB_PATH" ]]; then
  echo "ERROR: missing DB path. Provide DB_PATH arg or DATABASE_URL." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$DB_PATH" ]]; then
  echo "ERROR: DB file not found: $DB_PATH" >&2
  exit 2
fi

BACKUP_DIR="${BACKUP_DIR:-/var/backups/llm-travel-agent}"
RETAIN_COUNT="${RETAIN_COUNT:-14}"
if ! [[ "$RETAIN_COUNT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: RETAIN_COUNT must be an integer (got: $RETAIN_COUNT)" >&2
  exit 2
fi

mkdir -p "$BACKUP_DIR"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_file="$BACKUP_DIR/backup_${stamp}.db"
tmp_file="$out_file.tmp"

# Escape single quotes for SQLite SQL string literal.
out_sql="${tmp_file//\'/\'\'}"

sqlite3 "$DB_PATH" ".timeout 5000" "VACUUM INTO '$out_sql';"
mv "$tmp_file" "$out_file"
sha256sum "$out_file" > "$out_file.sha256"

# Prune old backups, keep newest RETAIN_COUNT backups.
mapfile -t backups < <(ls -1t "$BACKUP_DIR"/backup_*.db 2>/dev/null || true)
if (( ${#backups[@]} > RETAIN_COUNT )); then
  for old in "${backups[@]:RETAIN_COUNT}"; do
    rm -f "$old" "$old.sha256"
  done
fi

echo "backup_path=$out_file"
echo "checksum_path=$out_file.sha256"
echo "retain_count=$RETAIN_COUNT"
