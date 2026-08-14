#!/usr/bin/env bash
# promote.sh — promote tested code from the `staging` branch to PRODUCTION.
#
# Run from the PRODUCTION working tree (e.g. ~/typortrait/typography_engine).
# It verifies you're on the prod branch with a clean tree, shows exactly which
# commits + files will go live, asks for confirmation, then merges staging,
# pushes, and rebuilds the prod container.
#
# Safe by design: secrets and data never move. Each environment keeps its own
# git-ignored .env and data/ dir, so a merge can never push staging's TEST Stripe
# keys (or PRINTFUL_CONFIRM=false) into production.
set -euo pipefail

PROD_BRANCH="${PROD_BRANCH:-claude/printful-integration}"
STAGING_BRANCH="${STAGING_BRANCH:-staging}"

# 1. Must be on the prod branch.
cur="$(git rev-parse --abbrev-ref HEAD)"
if [ "$cur" != "$PROD_BRANCH" ]; then
  echo "✗ On '$cur', not the prod branch '$PROD_BRANCH'. Run this from the PRODUCTION tree."
  exit 1
fi

# 2. Clean working tree (don't promote on top of uncommitted edits).
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "✗ Uncommitted changes in the prod tree — commit or stash first:"
  git status --short
  exit 1
fi

# 3. Fetch and use the REMOTE staging branch as the source of truth. We compare
# against origin/<branch>, NEVER a local copy: a stale local branch in this tree
# (git fetch updates origin/* refs, not local branches) would make us think there
# is nothing to promote and silently skip the rebuild.
echo "→ Fetching latest…"
git fetch origin --quiet || true
SRC="origin/$STAGING_BRANCH"
if ! git rev-parse --verify --quiet "$SRC" >/dev/null; then
  echo "✗ '$SRC' not found. Did you push the staging branch to origin?"
  exit 1
fi

PENDING="$(git log --oneline "$PROD_BRANCH..$SRC" || true)"
if [ -z "$PENDING" ]; then
  echo "✓ Nothing to promote — production already has everything in $SRC."
  exit 0
fi

echo
echo "Commits to PROMOTE ($SRC → $PROD_BRANCH):"
echo "------------------------------------------------------------"
echo "$PENDING"
echo "------------------------------------------------------------"
echo "Files changed:"
git diff --stat "$PROD_BRANCH..$SRC"
echo

# 4. Confirm.
read -r -p "Promote to production and rebuild now? [y/N] " ans
case "$ans" in
  y|Y|yes|YES) ;;
  *) echo "Aborted. Nothing changed."; exit 0 ;;
esac

# 5. Merge.
if ! git merge --no-edit "$SRC"; then
  echo "✗ Merge conflict. Resolve it, 'git commit', then run: docker compose up -d --build"
  exit 1
fi

# 6. Push (non-fatal: if it fails, the deploy still proceeds).
git push || echo "⚠ push failed — merge is local; fix auth and 'git push' later."

# 7. Rebuild + restart production.
echo "→ Rebuilding production container…"
docker compose up -d --build

# Reclaim space from the now-dangling OLD image layers the rebuild superseded, so
# repeated promotes don't fill the disk. Dangling (untagged) images only — never a
# tagged image in use. Best-effort; never fails the promote.
docker image prune -f >/dev/null 2>&1 || true

echo
echo "✓ Promoted and live. Recent prod logs:"
docker compose logs --tail=15
