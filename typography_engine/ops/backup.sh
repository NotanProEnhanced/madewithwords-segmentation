#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Typortrait / Loved in Words — encrypted, incremental, OFF-SITE backup.
#
# Uses restic: deduplicated + encrypted + retention in one tool. Safe to run
# hourly (only changes are stored), which gives a ~1-hour recovery point.
#
# Backs up: consistent SQLite snapshots, the private/ data (recipes + consent),
# the .env secrets, and the server config (nginx + TLS certs).
#
# No secrets live in this file — they're read from a root-only creds file
# (see ops/BACKUP-SETUP.md). This script is safe to commit to the repo.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- CONFIG (adjust APP_DIR if your checkout path differs) -----------------
APP_DIR="/root/typortrait-staging/typography_engine"
STAGING="/root/.typortrait-backup-staging"     # transient consistent DB snapshots
ENVFILE="/root/.typortrait-backup.env"         # restic + provider creds (chmod 600)
LOG="/var/log/typortrait-backup.log"
HEALTHCHECK_URL=""                             # optional healthchecks.io ping URL
# ---------------------------------------------------------------------------

exec >>"$LOG" 2>&1
echo "===== $(date -u +%FT%TZ) backup start ====="

fail(){
  echo "BACKUP FAILED: $*"
  [ -n "$HEALTHCHECK_URL" ] && curl -fsS -m 10 --retry 3 "${HEALTHCHECK_URL}/fail" >/dev/null 2>&1 || true
  exit 1
}
trap 'fail "error near line $LINENO"' ERR

[ -f "$ENVFILE" ] || fail "missing creds file $ENVFILE (see ops/BACKUP-SETUP.md)"
set -a; . "$ENVFILE"; set +a
command -v restic  >/dev/null || fail "restic not installed (apt-get install restic)"
command -v sqlite3 >/dev/null || fail "sqlite3 not installed (apt-get install sqlite3)"

# 1) Consistent SQLite snapshots — never back up a live DB file directly.
rm -rf "$STAGING"; mkdir -p "$STAGING"; chmod 700 "$STAGING"
for db in "$APP_DIR/data/orders.db" "$APP_DIR/data/gather/gather.db"; do
  [ -f "$db" ] && sqlite3 "$db" ".backup '$STAGING/$(basename "$db")'"
done

# 2) Initialise the repo on first run (no-op once it exists).
restic snapshots >/dev/null 2>&1 || restic init

# 3) Back up DB snapshots + private data + secrets + server config.
restic backup \
  "$STAGING" \
  "$APP_DIR/.env" \
  "$APP_DIR/private" \
  /etc/nginx/sites-available \
  /etc/letsencrypt \
  --tag typortrait --exclude-caches

# 4) Retention: ~1h RPO recent, thinning to long-term (yearly x7 covers the
#    ~7-year consent-record retention).
restic forget --prune \
  --keep-hourly 24 --keep-daily 30 --keep-monthly 12 --keep-yearly 7

# 5) Fast structural integrity check (non-fatal warning if it flags anything).
restic check || echo "WARN: restic check reported an issue — investigate."

rm -rf "$STAGING"
echo "===== backup ok ====="
[ -n "$HEALTHCHECK_URL" ] && curl -fsS -m 10 --retry 3 "$HEALTHCHECK_URL" >/dev/null 2>&1 || true
