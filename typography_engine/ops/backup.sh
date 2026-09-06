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

# SCOPE: the trees listed in APP_DIRS below. The remaining live trees have their
# config, orders and consent records covered nightly by backup-config.sh, but
# NOT their source photos or gather.db. See "The gap between them" in
# ops/README.md before assuming the whole fleet is in here.
#
# ---- CONFIG ---------------------------------------------------------------
# One line per tree. Each contributes its .env, data/private and a consistent
# snapshot of its databases. Snapshots are staged per tree, so two trees with an
# orders.db do not overwrite each other -- they did not, before, because there
# was only ever one tree.
APP_DIRS=(
  "/root/typortrait-prod/typography_engine"   # production
  "/root/typortrait/typography_engine"        # hand-run workspace: marketing and reel renders
)
STAGING="/root/.typortrait-backup-staging"     # transient consistent DB snapshots
ENVFILE="/root/.typortrait-backup.env"         # restic + provider creds (chmod 600)
LOG="/var/log/typortrait-backup.log"
HEALTHCHECK_URL="https://hc-ping.com/692a664c-1ee1-45bd-8144-1a7524aba3d6"
# ---------------------------------------------------------------------------

exec >>"$LOG" 2>&1
echo "===== $(date -u +%FT%TZ) backup start ====="

fail(){
  echo "BACKUP FAILED: $*"
  [ -n "$HEALTHCHECK_URL" ] && curl -fsS -m 10 --retry 3 "${HEALTHCHECK_URL}/fail" >/dev/null 2>&1 || true
  exit 1
}
trap 'fail "error near line $LINENO"' ERR

# These paths are hardcoded above and have been wrong before: the production
# tree was renamed from typortrait-staging to typortrait-prod in August 2026,
# and every path in this file pointed at the old name until it was caught. Check
# them explicitly, so a wrong path says so instead of failing three commands
# later with a line number.
for _d in "${APP_DIRS[@]}"; do
  [ -d "$_d" ] || fail "tree does not exist: $_d -- has it been renamed or removed? (see ops/README.md)"
done
[ -f "$ENVFILE" ] || fail "missing creds file $ENVFILE (see ops/BACKUP-SETUP.md)"
set -a; . "$ENVFILE"; set +a
command -v restic  >/dev/null || fail "restic not installed (apt-get install restic)"
command -v sqlite3 >/dev/null || fail "sqlite3 not installed (apt-get install sqlite3)"

# 1) Consistent SQLite snapshots — never back up a live DB file directly.
#    Staged under STAGING/<tree>/ so each tree's databases stay distinguishable
#    in the snapshot and in a restore.
rm -rf "$STAGING"; mkdir -p "$STAGING"; chmod 700 "$STAGING"
for _d in "${APP_DIRS[@]}"; do
  _tree=$(basename "$(dirname "$_d")")
  mkdir -p "$STAGING/$_tree"
  for db in "$_d/data/orders.db" "$_d/data/gather/gather.db"; do
    [ -f "$db" ] && sqlite3 "$db" ".backup '$STAGING/$_tree/$(basename "$db")'"
  done
done

# 2) Initialise the repo on first run (no-op once it exists).
#
# This used to be `restic snapshots >/dev/null 2>&1 || restic init`, which
# answers "I cannot READ the repository" and "there is no repository" with the
# same action. On 2 Sep 2026 B2 returned 403 on every read -- the daily Class B
# transaction cap -- and the job's response was to try to initialise ON TOP of a
# live repository holding every backup we have. It failed only because init also
# needs a read. That must never be left to luck: a transient read error is not
# evidence that the repository is gone.
if _cfg_err="$(restic cat config 2>&1 >/dev/null)"; then
  :                                   # repository is there and readable
else
  case "$_cfg_err" in
    *403*|*"Access Denied"*|*"denied"*)
      fail "cannot READ the repository: $_cfg_err
  A 403 from B2 usually means the daily Class B transaction cap is reached.
  restic must read the config and indexes before it can write, so BACKUPS ARE
  STOPPED until the cap resets at 00:00 UTC or is raised on the Caps & Alerts
  page. Deliberately NOT initialising -- the repository is almost certainly
  intact and unreadable, not missing." ;;
  esac
  echo "-- no repository at this location; initialising a new one"
  restic init
fi

# 3) DR set — everything, tagged 'data'. SHORT retention (below) so source
#    photos roll off in ~35 days, matching our ~30-day deletion promise to users.
#    Include the live marketing web roots: some of their assets aren't tracked in
#    git, so the backup is their authoritative recovery source (present-dirs only).
WWW=()
for d in /var/www/typortrait.com /var/www/lovedinwords.com; do [ -d "$d" ] && WWW+=("$d"); done
PATHS=("$STAGING")
for _d in "${APP_DIRS[@]}"; do
  [ -f "$_d/.env" ]         && PATHS+=("$_d/.env")
  [ -d "$_d/data/private" ] && PATHS+=("$_d/data/private")
done
restic backup \
  "${PATHS[@]}" \
  "${WWW[@]}" \
  /etc/nginx/sites-available \
  /etc/letsencrypt \
  --tag typortrait --tag data --exclude-caches

# 3b) Consent records ONLY, tagged 'consent'. LONG retention (legal evidence,
#     ~7 years). They also live in the 'data' snapshot; this is what keeps them
#     after the photos have rolled off, WITHOUT retaining photos for years.
: > "$STAGING/consent-list.txt"
for _d in "${APP_DIRS[@]}"; do
  [ -d "$_d/data/private" ] || continue
  find "$_d/data/private" \( -name '*.consent.json' -o -name '*.biometric_consent.json' \) -print \
    >> "$STAGING/consent-list.txt" || true
done
if [ -s "$STAGING/consent-list.txt" ]; then
  restic backup --files-from "$STAGING/consent-list.txt" --tag typortrait --tag consent
fi

# 4) Two-tier retention, then a single prune.
#    data    -> photos/DBs/config roll off at 35 days (matches the deletion promise)
#    consent -> kept long for the legal retention window
#
# The data line uses --keep-within-*, which expires by ELAPSED TIME. It used to
# say --keep-daily 35, and that is a different thing: it keeps the 35 most
# recent days that HAVE a snapshot, not the last 35 days. This script was never
# actually scheduled, so it ran three times in two months -- and under
# --keep-daily 35 those three days were all "within the last 35 daily
# snapshots" and were kept indefinitely. Customer source photos from 1 July
# were still in the repository on 29 August, 59 days later, against a promise
# to delete at ~30. Count-based retention silently stops honouring a time-based
# promise the moment the backup cadence slips, and fails in the direction that
# keeps personal data longer.
restic forget --tag data    --keep-within-hourly 24h --keep-within-daily 35d
#
# The consent line stays COUNT-based on purpose. These are legal records and no
# photos, so the safe direction is to over-retain: when the cadence slips,
# count-based keeps more than intended, while time-based would delete records
# that no newer backup has replaced. Tiny text files, so over-retention costs
# nothing and under-retention is unrecoverable.
restic forget --tag consent --keep-daily 30 --keep-monthly 12 --keep-yearly 7

# 5) MAINTENANCE -- only with --maintain, which cron runs ONCE A DAY.
#
# `prune` and `check` both read the whole repository's metadata: check downloads
# every index and snapshot file, prune reads the indexes and rewrites packs. On
# B2 each of those file reads is a billable Class B transaction, and this script
# runs hourly -- so 24 full metadata reads a day exhausted the free Class B
# allowance and Backblaze capped the account on 2 Sep 2026.
#
# Nothing above this line needs them. `forget` still runs hourly, so retention is
# decided on time; prune is what actually reclaims the bytes, and daily is well
# inside the ~30-day deletion promise (data is kept 35d).
#
# The cost of skipping a day is bytes left in the repo a few hours longer. The
# cost of running them hourly is that reads get denied at the cap -- including a
# restore, exactly when it is needed.
if [ "${1:-}" = "--maintain" ]; then
  echo "-- maintenance: prune + check"
  restic prune
  restic check || echo "WARN: restic check reported an issue — investigate."
else
  echo "-- maintenance skipped (run with --maintain; cron does this daily)"
fi

rm -rf "$STAGING"
echo "===== backup ok ====="
[ -n "$HEALTHCHECK_URL" ] && curl -fsS -m 10 --retry 3 "$HEALTHCHECK_URL" >/dev/null 2>&1 || true
