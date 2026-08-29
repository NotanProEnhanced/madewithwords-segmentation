#!/bin/bash
# Detect the live marketing site drifting from the repo.
#
# The site is now edited in git and published with deploy-site.sh, but nothing
# stops someone editing /var/www directly -- which is how it ended up
# unversioned in the first place. This reports any difference so drift is caught
# in a day rather than discovered months later.
#
#   ./site-drift.sh          report differences (exit 1 if any)
#   ./site-drift.sh --quiet  print only when drift is found (for cron)
#
# Reports both directions:
#   only in /var/www  someone edited the live site, or a file was never committed
#   differs           the same file has different contents in each place
#   only in repo      committed but never deployed -- run deploy-site.sh
set -uo pipefail

SRC="${SRC:-/root/typortrait-stg/sites/typortrait.com}"
DST="${DST:-/var/www/typortrait.com}"
QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

EXCLUDES=(
  --exclude 'review/'
  --exclude '*.bak*'
  --exclude '*.retired'
  --exclude '*-new*.jpg'
)

[ -d "$SRC" ] || { echo "site-drift: no repo copy at $SRC"; exit 2; }
[ -d "$DST" ] || { echo "site-drift: no live site at $DST"; exit 2; }

OUT=$(rsync -rn --itemize-changes --delete "${EXCLUDES[@]}" "$SRC/" "$DST/" 2>/dev/null \
      | grep -vE '^\.d\.\.t' || true)

if [ -z "$OUT" ]; then
  [ "$QUIET" = "1" ] || echo "site-drift: live site matches the repo"
  exit 0
fi

echo "site-drift: $DST differs from $SRC"
echo
echo "$OUT" | while read -r line; do
  case "$line" in
    \*deleting*) echo "  only in /var/www (not in repo): ${line#\*deleting }" ;;
    \>f+++++++*) echo "  only in repo (not deployed):    ${line##* }" ;;
    \>f*)        echo "  differs:                        ${line##* }" ;;
    *)           echo "  $line" ;;
  esac
done
echo
echo "If /var/www is right: copy those files into $SRC and commit."
echo "If the repo is right: run deploy-site.sh --apply."
exit 1
