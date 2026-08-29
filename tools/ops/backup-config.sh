#!/bin/bash
# Encrypted backup of everything that is NOT in git and cannot be rebuilt.
#
# WHAT AND WHY
#   .env + docker-compose.yml per tree   the entire configuration of five
#                                        deployments, deliberately untracked
#                                        because they hold credentials
#   data/orders.db per tree              customer orders. The existing nightly
#                                        cron backs up ~/typortrait/data/orders.db,
#                                        which is NOT any of the live trees --
#                                        so these may have had no backup at all
#   data/private/*.json per tree         order metadata and consent records
#
#   Customer source photos (data/private/*.src) are deliberately EXCLUDED: they
#   are large, they are personal data the product promises to delete after ~30
#   days, and copying them into a long-lived archive would quietly break that.
#
# ENCRYPTION
#   gpg symmetric, passphrase read from /root/.backup-pass (chmod 600).
#   Create it once:
#       head -c 32 /dev/urandom | base64 > /root/.backup-pass
#       chmod 600 /root/.backup-pass
#   Then store that passphrase in your password manager. An encrypted archive you
#   cannot decrypt is not a backup, and the passphrase must not live only on the
#   machine the backup protects.
#
# OFF-BOX
#   A backup on the same disk as the original protects against fat fingers and
#   nothing else. Set one of these and the archive is pushed after it is written:
#       BACKUP_REMOTE=user@host:/backups/typortrait   ./backup-config.sh   # scp
#       BACKUP_RCLONE=b2:my-bucket/typortrait          ./backup-config.sh   # rclone
#   Both may be set, for two independent copies.
#
#   Verify a restore afterwards with restore-verify.sh. An untested backup is a
#   guess, and the failure mode is silent until the day it matters.
#
# Usage:  ./backup-config.sh
set -euo pipefail

TREES="typortrait-stg typortrait-prod typortrait-faithinwords typortrait-lovedinwords typortrait-pawsinwords"
OUT="${OUT:-/root/backups}"
PASS="${PASS:-/root/.backup-pass}"
KEEP_DAYS="${KEEP_DAYS:-30}"
STAMP=$(date +%Y%m%d-%H%M)

[ -f "$PASS" ] || { echo "no passphrase file at $PASS -- see the header of this script"; exit 1; }
[ "$(stat -c '%a' "$PASS")" = "600" ] || { echo "$PASS must be chmod 600"; exit 1; }
command -v gpg >/dev/null || { echo "gpg not installed"; exit 1; }

mkdir -p "$OUT"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

missing=0
for T in $TREES; do
  E="/root/$T/typography_engine"
  [ -d "$E" ] || { echo "WARNING: no such tree: $E"; missing=1; continue; }
  D="$STAGE/$T"
  mkdir -p "$D/data/private"
  cp "$E/.env" "$D/.env" 2>/dev/null || { echo "WARNING: no .env in $T"; missing=1; }
  cp "$E/docker-compose.yml" "$D/docker-compose.yml" 2>/dev/null || { echo "WARNING: no compose in $T"; missing=1; }
  if [ -f "$E/data/orders.db" ]; then
    # sqlite3 .backup takes a consistent copy of a live database; a plain cp of a
    # WAL-mode file mid-write can be torn.
    if command -v sqlite3 >/dev/null; then
      sqlite3 "$E/data/orders.db" ".backup '$D/data/orders.db'"
    else
      cp "$E/data/orders.db" "$D/data/orders.db"
      echo "NOTE: sqlite3 not installed -- orders.db copied without a consistent snapshot"
    fi
  else
    echo "NOTE: no orders.db in $T"
  fi
  # order metadata and consent records; NOT the customer photos
  find "$E/data/private" -maxdepth 1 -name '*.json' -exec cp {} "$D/data/private/" \; 2>/dev/null || true
done

ARCHIVE="$OUT/typortrait-config-$STAMP.tar.gz.gpg"
tar -czf - -C "$STAGE" . \
  | gpg --batch --yes --symmetric --cipher-algo AES256 \
        --passphrase-file "$PASS" -o "$ARCHIVE"
chmod 600 "$ARCHIVE"

echo "wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
echo "contents:"
gpg --batch --quiet --decrypt --passphrase-file "$PASS" "$ARCHIVE" 2>/dev/null \
  | tar -tzf - | head -30

pushed=0
if [ -n "${BACKUP_REMOTE:-}" ]; then
  echo "pushing to $BACKUP_REMOTE (scp)"
  if scp -q "$ARCHIVE" "$BACKUP_REMOTE/"; then echo "pushed"; pushed=1
  else echo "PUSH FAILED -- archive is still local only"; fi
fi
if [ -n "${BACKUP_RCLONE:-}" ]; then
  if command -v rclone >/dev/null; then
    echo "pushing to $BACKUP_RCLONE (rclone)"
    if rclone copy "$ARCHIVE" "$BACKUP_RCLONE"; then echo "pushed"; pushed=1
    else echo "PUSH FAILED -- archive is still local only"; fi
  else
    echo "BACKUP_RCLONE is set but rclone is not installed"
  fi
fi
if [ "$pushed" = "0" ]; then
  echo
  echo "BACKUP_REMOTE is not set, so this archive is on the same disk as the"
  echo "originals. Set it to an off-box destination or this protects very little."
fi

find "$OUT" -name 'typortrait-config-*.tar.gz.gpg' -mtime +"$KEEP_DAYS" -delete
[ "$missing" = "0" ] || echo "COMPLETED WITH WARNINGS -- see above"
