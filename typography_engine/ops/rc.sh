# ops/rc.sh — load the restic backup environment into your CURRENT shell.
#
#   Usage:  source /root/typortrait-prod/typography_engine/ops/rc.sh
#           (or, from the repo dir:  . ops/rc.sh)
#
# Then run restic directly:  restic snapshots | restic stats | restic check
# The env only lasts for this terminal session — source again in a new shell.
#
# This file is HAND-EDITED whenever the B2 keys are rotated, so it validates
# before and after sourcing. It used to print "restic env loaded" regardless:
# with an unterminated quote in the creds file, bash reported EOF, sourcing
# stopped part-way, and this script still declared success — so the real fault
# only surfaced later as restic complaining about an empty key. A loader that
# says "loaded" when it did not is worse than no message at all.
_e="/root/.typortrait-backup.env"
if [ ! -f "$_e" ]; then
  echo "creds file not found: $_e   (see ops/BACKUP-SETUP.md)"
elif ! bash -n "$_e" 2>/dev/null; then
  echo "PROBLEM: $_e has a syntax error and was NOT loaded."
  bash -n "$_e"
  echo
  echo "Usually an unterminated quote after a key rotation. Each line should be:"
  echo '    export B2_ACCOUNT_KEY="...."      quote at BOTH ends, no trailing text'
  echo "Fix it with: nano $_e"
else
  set -a; . "$_e"; set +a
  _missing=""
  for _v in RESTIC_REPOSITORY RESTIC_PASSWORD_FILE B2_ACCOUNT_ID B2_ACCOUNT_KEY; do
    [ -n "${!_v:-}" ] || _missing="$_missing $_v"
  done
  if [ -n "$_missing" ]; then
    echo "PROBLEM: loaded $_e but these are empty:$_missing"
    echo "restic will fail. Check those lines in $_e"
  elif [ ! -s "${RESTIC_PASSWORD_FILE:-/nonexistent}" ]; then
    echo "PROBLEM: RESTIC_PASSWORD_FILE is empty or missing:"
    echo "    ${RESTIC_PASSWORD_FILE}"
    echo "Without it the repository cannot be decrypted at all."
  else
    echo "restic env loaded  ->  repo: $RESTIC_REPOSITORY"
    echo "try: restic snapshots | restic stats | restic check | restic forget <id> --prune"
  fi
  unset _missing _v
fi
unset _e
