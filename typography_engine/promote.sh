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

# 3. Fetch, pick the staging source (local branch is authoritative; worktrees share it).
echo "→ Fetching latest…"
git fetch origin --quiet || true
if git show-ref --verify --quiet "refs/heads/$STAGING_BRANCH"; then
  SRC="$STAGING_BRANCH"
else
  SRC="origin/$STAGING_BRANCH"
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

echo
echo "✓ Promoted and live. Recent prod logs:"
docker compose logs --tail=15
