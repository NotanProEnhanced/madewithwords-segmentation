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

DOCROOT="${DOCROOT:-/var/www/staging-lovedinwords}"

# Work from typography_engine (this file lives at typography_engine/deploy/).
# Copy from the WORKING TREE rather than `git archive` so it's robust to how the
# repo is rooted on the box; make sure the tree is current first.
cd "$(dirname "$0")/.."

SRC="marketing/lovedinwords"
if [ ! -d "$SRC" ]; then
  echo "✗ '$SRC' not found. Run this from the staging worktree (cd ~/typortrait-staging/typography_engine)."
  exit 1
fi
echo "→ Updating the worktree …"
git pull --ff-only --quiet || echo "  (skipping pull — not fast-forwardable; publishing the current tree)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
rsync -a --exclude '_*' "$SRC/" "$TMP/"

# PROD -> STAGING rewrites for the staging copy only:
#   - app.typortrait.com   -> staging.typortrait.com  (studio CTAs + policy links)
#   - lovedinwords.com/     -> staging.typortrait.com/lw/  (the gallery "Home" link)
sed -i \
  -e 's#https://app\.typortrait\.com#https://staging.typortrait.com#g' \
  -e 's#https://lovedinwords\.com/#https://staging.typortrait.com/lw/#g' \
  "$TMP/index.html" "$TMP/gallery.html"

sudo mkdir -p "$DOCROOT"
# --exclude '_*' drops the working/preview files; no --delete (leave unknowns).
sudo rsync -av --exclude '_*' "$TMP/" "$DOCROOT/"
sudo chown -R www-data:www-data "$DOCROOT"

echo "✓ Published to $DOCROOT"
echo "  → https://staging.typortrait.com/lw/index.html  (Basic Auth)"
