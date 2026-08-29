#!/bin/bash
# Verify that a backup archive can actually be restored.
#
# An untested backup is a guess. This decrypts the most recent archive into a
# temporary directory, checks that every tree is present with its config, and
# runs SQLite's own integrity check on each orders.db. It writes nothing to any
# live tree and removes the temporary copy afterwards.
#
#   ./restore-verify.sh                 verify the newest archive
#   ./restore-verify.sh <archive.gpg>   verify a specific one
#
# Run it after the first backup, and again whenever the backup changes shape.
set -uo pipefail

OUT="${OUT:-/root/backups}"
PASS="${PASS:-/root/.backup-pass}"
TREES="typortrait-stg typortrait-prod typortrait-faithinwords typortrait-lovedinwords typortrait-pawsinwords"

ARCHIVE="${1:-}"
if [ -z "$ARCHIVE" ]; then
  ARCHIVE=$(ls -t "$OUT"/typortrait-config-*.tar.gz.gpg 2>/dev/null | head -1)
fi
[ -n "$ARCHIVE" ] && [ -f "$ARCHIVE" ] || { echo "no archive found in $OUT"; exit 1; }
[ -f "$PASS" ] || { echo "no passphrase file at $PASS"; exit 1; }

echo "verifying $ARCHIVE"
echo "size: $(du -h "$ARCHIVE" | cut -f1)   written: $(stat -c '%y' "$ARCHIVE" | cut -d. -f1)"
echo

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if ! gpg --batch --quiet --decrypt --passphrase-file "$PASS" "$ARCHIVE" 2>/dev/null \
     | tar -xzf - -C "$TMP"; then
  echo "FAILED: could not decrypt or extract. The archive or the passphrase is wrong."
  exit 1
fi
echo "decrypt + extract: OK"
echo

fail=0
for T in $TREES; do
  printf '%-26s ' "$T"
  [ -d "$TMP/$T" ] || { echo "MISSING FROM ARCHIVE"; fail=1; continue; }
  parts=""
  [ -s "$TMP/$T/.env" ] && parts="$parts .env" || { parts="$parts NO-ENV"; fail=1; }
  [ -s "$TMP/$T/docker-compose.yml" ] && parts="$parts compose" || { parts="$parts NO-COMPOSE"; fail=1; }
  if [ -f "$TMP/$T/data/orders.db" ]; then
    if command -v sqlite3 >/dev/null; then
      chk=$(sqlite3 "$TMP/$T/data/orders.db" 'PRAGMA integrity_check;' 2>&1 | head -1)
      n=$(sqlite3 "$TMP/$T/data/orders.db" 'SELECT COUNT(*) FROM orders;' 2>/dev/null || echo '?')
      if [ "$chk" = "ok" ]; then
        parts="$parts orders.db(ok,$n rows)"
      else
        parts="$parts orders.db(CORRUPT: $chk)"; fail=1
      fi
    else
      parts="$parts orders.db(present, sqlite3 not installed to check)"
    fi
  else
    parts="$parts no-orders.db"
  fi
  j=$(ls "$TMP/$T/data/private"/*.json 2>/dev/null | wc -l)
  echo "$parts  private-json=$j"
done

printf '%-26s ' "system"
sys=""
[ -s "$TMP/system/crontab.txt" ] && sys="$sys crontab($(wc -l < "$TMP/system/crontab.txt") lines)" || { sys="$sys NO-CRONTAB"; fail=1; }
[ -s "$TMP/system/etc-nginx.tar.gz" ] && sys="$sys nginx($(du -h "$TMP/system/etc-nginx.tar.gz" | cut -f1))" || { sys="$sys NO-NGINX"; fail=1; }
[ -s "$TMP/system/letsencrypt-renewal.tar.gz" ] && sys="$sys letsencrypt-renewal" || sys="$sys no-letsencrypt"
echo "$sys"

echo
if [ "$fail" = "0" ]; then
  echo "RESTORE VERIFIED -- this archive contains a usable copy of every tree."
else
  echo "PROBLEMS FOUND -- see above. Do not rely on this archive."
  exit 1
fi
