# ops/rc.sh — load the restic backup environment into your CURRENT shell.
#
#   Usage:  source /root/typortrait-prod/typography_engine/ops/rc.sh
#           (or, from the repo dir:  . ops/rc.sh)
#
# Then run restic directly:  restic snapshots | restic stats | restic check
# The env only lasts for this terminal session — source again in a new shell.
_e="/root/.typortrait-backup.env"
if [ -f "$_e" ]; then
  set -a; . "$_e"; set +a
  echo "restic env loaded  ->  repo: ${RESTIC_REPOSITORY:-?}"
  echo "try: restic snapshots | restic stats | restic check | restic forget <id> --prune"
else
  echo "creds file not found: $_e   (see ops/BACKUP-SETUP.md)"
fi
unset _e
