#!/usr/bin/env bash
# stage-lovedinwords.sh — publish the Loved in Words MARKETING pages to the
# staging mirror at https://staging.typortrait.com/lw/ (behind Basic Auth).
#
# Single source of truth: the pages live once in the repo pointing at PROD.
# This script generates a *staging* copy on the fly — rewriting prod URLs to
# staging — so there are no parallel staging HTML files to drift. The CTAs end
# up pointing at the staging studio, so the full journey (page -> create ->
# Stripe TEST checkout) is testable on staging.
#
# Run on the VPS from anywhere inside the repo (e.g. the staging worktree):
#   bash deploy/stage-lovedinwords.sh
#
# Overridable:
#   BRANCH=feat/displacement-style   the source branch (origin/<BRANCH>)
#   DOCROOT=/var/www/staging-lovedinwords
set -euo pipefail

BRANCH="${BRANCH:-feat/displacement-style}"
DOCROOT="${DOCROOT:-/var/www/staging-lovedinwords}"
SUBDIR="typography_engine/marketing/lovedinwords"

# Work from inside the git repo (this file lives at typography_engine/deploy/).
cd "$(dirname "$0")/.."

echo "→ Fetching origin/$BRANCH …"
git fetch origin --quiet

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git archive "origin/$BRANCH" "$SUBDIR" | tar -x -C "$TMP"

# PROD -> STAGING rewrites for the staging copy only:
#   - app.typortrait.com   -> staging.typortrait.com  (studio CTAs + policy links)
#   - lovedinwords.com/     -> staging.typortrait.com/lw/  (the gallery "Home" link)
sed -i \
  -e 's#https://app\.typortrait\.com#https://staging.typortrait.com#g' \
  -e 's#https://lovedinwords\.com/#https://staging.typortrait.com/lw/#g' \
  "$TMP/$SUBDIR/index.html" "$TMP/$SUBDIR/gallery.html"

sudo mkdir -p "$DOCROOT"
# --exclude '_*' drops the working/preview files; no --delete (leave unknowns).
sudo rsync -av --exclude '_*' "$TMP/$SUBDIR/" "$DOCROOT/"
sudo chown -R www-data:www-data "$DOCROOT"

echo "✓ Published to $DOCROOT"
echo "  → https://staging.typortrait.com/lw/index.html  (Basic Auth)"
