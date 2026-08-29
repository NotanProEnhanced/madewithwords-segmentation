#!/bin/bash
# Deploy the typortrait.com marketing site from git to the web root.
#
# The site is now edited in the repo at sites/typortrait.com and deployed from
# there. Before this, /var/www was the only copy and every edit was unversioned.
#
#   ./deploy-site.sh            show what WOULD change (default; nothing written)
#   ./deploy-site.sh --apply    actually deploy
#
# Deliberately protective:
#   - dry-run is the default, so an accidental run cannot destroy the live site
#   - --delete removes files the repo does not have, but the excludes below keep
#     backups, retired files and the review directory, which live only on the
#     server by design
#   - refuses to run if the source looks empty or index.html is missing
set -euo pipefail

SRC="${SRC:-/root/typortrait-stg/sites/typortrait.com}"
DST="${DST:-/var/www/typortrait.com}"

EXCLUDES=(
  --exclude 'review/'
  --exclude '*.bak*'
  --exclude '*.retired'
  --exclude '*-new*.jpg'
)

[ -f "$SRC/index.html" ] || { echo "refusing: no index.html in $SRC"; exit 1; }
[ -d "$DST" ] || { echo "refusing: destination $DST does not exist"; exit 1; }

if [ "${1:-}" = "--apply" ]; then
  echo "deploying $SRC -> $DST"
  rsync -a --delete "${EXCLUDES[@]}" "$SRC/" "$DST/"
  echo "done. Live files:"
  ls -la "$DST" | head -15
  echo
  echo "Reminder: bump the ?v= query on any changed asset, or browsers keep the old one."
else
  echo "DRY RUN -- nothing written. Re-run with --apply to deploy."
  echo "$SRC -> $DST"
  echo
  rsync -a --delete --itemize-changes --dry-run "${EXCLUDES[@]}" "$SRC/" "$DST/"
  echo
  echo "Lines above starting > are files that would be written;"
  echo "lines starting *deleting are files that would be REMOVED from the live site."
fi
