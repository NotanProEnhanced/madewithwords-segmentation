#!/bin/bash
# Monthly proof that the backup still works, from Backblaze rather than from the
# local copy.
#
# A nightly backup that silently stopped working looks identical to one that is
# fine -- until the day you need it. This pulls the newest archive back down,
# decrypts it, and checks every tree plus the system config. It is deliberately
# quiet on success and loud on failure.
#
#   ./verify-monthly.sh            verify; email only if something is wrong
#   ./verify-monthly.sh --always   email either way (useful for the first run)
#
# Email uses the SMTP settings already in the production .env. If they are not
# there, the result still goes to the log and the exit code is still non-zero on
# failure, so cron's own mail (if configured) picks it up.
set -uo pipefail

REMOTE="${BACKUP_RCLONE:-}"
PASS="${PASS:-/root/.backup-pass}"
ENVFILE="${ENVFILE:-/root/typortrait-prod/typography_engine/.env}"
LOG="${LOG:-/var/log/typortrait-backup-verify.log}"
ALWAYS=0
[ "${1:-}" = "--always" ] && ALWAYS=1

[ -n "$REMOTE" ] || { echo "BACKUP_RCLONE is not set -- source /root/.backup-env first"; exit 2; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

LATEST=$(rclone lsf "$REMOTE" 2>/dev/null | sort | tail -1)
if [ -z "$LATEST" ]; then
  OUT="FAILED: no archives found at $REMOTE"
  rc=1
else
  if rclone copy "$REMOTE/$LATEST" "$TMP/" 2>/dev/null; then
    OUT=$(/root/restore-verify.sh "$TMP/$LATEST" 2>&1)
    rc=$?
  else
    OUT="FAILED: could not download $LATEST from $REMOTE"
    rc=1
  fi
fi

# Age check: a backup that stopped running still verifies fine, so check the
# newest archive is recent as well as valid.
AGE_DAYS=99
if [ -n "$LATEST" ]; then
  STAMP=$(echo "$LATEST" | sed -n 's/.*-\([0-9]\{8\}\)-[0-9]\{4\}\..*/\1/p')
  if [ -n "$STAMP" ]; then
    AGE_DAYS=$(( ( $(date +%s) - $(date -d "$STAMP" +%s) ) / 86400 ))
    if [ "$AGE_DAYS" -gt 3 ]; then
      OUT="$OUT

WARNING: newest archive is $AGE_DAYS days old -- the nightly job may have stopped."
      rc=1
    fi
  fi
fi

STATUS=$([ "$rc" = "0" ] && echo "OK" || echo "PROBLEM")
{
  echo "=== $(date '+%Y-%m-%d %H:%M')  $STATUS  archive=$LATEST age=${AGE_DAYS}d"
  echo "$OUT"
  echo
} >> "$LOG"

if [ "$rc" != "0" ] || [ "$ALWAYS" = "1" ]; then
  echo "$OUT"
  # best-effort email using the app's own SMTP settings
  if [ -f "$ENVFILE" ]; then
    python3 - "$STATUS" "$LATEST" <<'PY' 2>/dev/null || true
import os, re, smtplib, sys
from email.message import EmailMessage
env = {}
for line in open(os.environ.get("ENVFILE", "/root/typortrait-prod/typography_engine/.env"),
                 encoding="utf-8", errors="ignore"):
    m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
    if m:
        env[m.group(1)] = m.group(2)
host = env.get("TYPO_SMTP_HOST"); user = env.get("TYPO_SMTP_USER")
pw = env.get("TYPO_SMTP_PASS") or env.get("TYPO_SMTP_PASSWORD")
to = env.get("TYPO_ADMIN_EMAIL")
if not (host and user and pw and to):
    sys.exit(0)
msg = EmailMessage()
msg["Subject"] = "Typortrait backup verification: %s (%s)" % (sys.argv[1], sys.argv[2])
msg["From"] = user
msg["To"] = to
msg.set_content(open(os.environ.get("LOG", "/var/log/typortrait-backup-verify.log")).read()[-4000:])
s = smtplib.SMTP(host, int(env.get("TYPO_SMTP_PORT", "587")), timeout=20)
s.starttls(); s.login(user, pw); s.send_message(msg); s.quit()
PY
  fi
fi
exit "$rc"
