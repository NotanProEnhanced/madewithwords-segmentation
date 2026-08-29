#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Typortrait / Loved in Words — restore from an off-site restic backup.
#
# SAFE BY DEFAULT: restores into a staging directory for review; it does NOT
# overwrite live data. You copy the reviewed files into place (guidance printed
# at the end). See ops/BACKUP-SETUP.md for the full recovery runbook.
#
# Usage:
#   ./restore.sh                      # restore 'latest' to /root/typortrait-restore
#   ./restore.sh /root/somewhere      # restore 'latest' to a chosen dir
#   ./restore.sh /root/somewhere <id> # restore a specific snapshot id
# ---------------------------------------------------------------------------
set -euo pipefail

APP_DIR="/root/typortrait-prod/typography_engine"
ENVFILE="/root/.typortrait-backup.env"
RESTORE_TO="${1:-/root/typortrait-restore}"
SNAPSHOT="${2:-latest}"

[ -f "$ENVFILE" ] || { echo "missing creds file $ENVFILE (see ops/BACKUP-SETUP.md)"; exit 1; }
set -a; . "$ENVFILE"; set +a
command -v restic >/dev/null || { echo "restic not installed"; exit 1; }

echo "Available snapshots:"
restic snapshots --tag typortrait
echo
echo "Restoring snapshot '$SNAPSHOT' into $RESTORE_TO ..."
mkdir -p "$RESTORE_TO"
restic restore "$SNAPSHOT" --target "$RESTORE_TO"

cat <<EOF

Restore complete. Files are under: $RESTORE_TO  (original absolute paths preserved)

Review, then place each piece:
  • Databases   $RESTORE_TO/root/.typortrait-backup-staging/orders.db
                $RESTORE_TO/root/.typortrait-backup-staging/gather.db
                ->  $APP_DIR/data/orders.db   and   $APP_DIR/data/gather/gather.db
  • Private     $RESTORE_TO$APP_DIR/data/private/   ->  $APP_DIR/data/private/
  • Secrets     $RESTORE_TO$APP_DIR/.env       ->  $APP_DIR/.env      (chmod 600)
  • nginx       $RESTORE_TO/etc/nginx/sites-available/  ->  /etc/nginx/sites-available/
  • TLS certs   $RESTORE_TO/etc/letsencrypt/            ->  /etc/letsencrypt/   (or re-issue via certbot)

Then: rebuild the app (docker compose up -d --build), reload nginx, repoint DNS,
and reconcile any in-flight orders against Stripe and Printful.
EOF
