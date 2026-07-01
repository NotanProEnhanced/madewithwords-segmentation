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

# 3) DR set — everything, tagged 'data'. SHORT retention (below) so source
#    photos roll off in ~35 days, matching our ~30-day deletion promise to users.
#    Include the live marketing web roots: some of their assets aren't tracked in
#    git, so the backup is their authoritative recovery source (present-dirs only).
WWW=()
for d in /var/www/typortrait.com /var/www/lovedinwords.com; do [ -d "$d" ] && WWW+=("$d"); done
restic backup \
  "$STAGING" \
  "$APP_DIR/.env" \
  "$APP_DIR/data/private" \
  "${WWW[@]}" \
  /etc/nginx/sites-available \
  /etc/letsencrypt \
  --tag typortrait --tag data --exclude-caches

# 3b) Consent records ONLY, tagged 'consent'. LONG retention (legal evidence,
#     ~7 years). They also live in the 'data' snapshot; this is what keeps them
#     after the photos have rolled off, WITHOUT retaining photos for years.
find "$APP_DIR/data/private" \( -name '*.consent.json' -o -name '*.biometric_consent.json' \) -print \
  > "$STAGING/consent-list.txt" || true
if [ -s "$STAGING/consent-list.txt" ]; then
  restic backup --files-from "$STAGING/consent-list.txt" --tag typortrait --tag consent
fi

# 4) Two-tier retention, then a single prune.
#    data    -> photos/DBs/config roll off ~35 days (matches the deletion promise)
#    consent -> kept long for the legal retention window
restic forget --tag data    --keep-hourly 24 --keep-daily 35
restic forget --tag consent --keep-daily 30 --keep-monthly 12 --keep-yearly 7
restic prune

# 5) Fast structural integrity check (non-fatal warning if it flags anything).
restic check || echo "WARN: restic check reported an issue — investigate."

rm -rf "$STAGING"
echo "===== backup ok ====="
[ -n "$HEALTHCHECK_URL" ] && curl -fsS -m 10 --retry 3 "$HEALTHCHECK_URL" >/dev/null 2>&1 || true
