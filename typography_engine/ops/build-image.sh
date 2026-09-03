#!/bin/bash
# Build ONE image, from a pinned commit, for every tree to share.
#
# WHY
#   Today each of the five trees runs its own `docker compose up -d --build` --
#   five independent builds of what should be identical source, each keyed to
#   whatever happens to be checked out in that tree's working directory at that
#   moment. That gap is what let four live brands sit nine commits behind while
#   their `tt list` commit column looked current (2026-09-03): git HEAD said one
#   thing, the served image said another, and nothing forced them to agree.
#
#   This builds once, from `git archive` at an exact commit -- never from an
#   ambient working tree, which might carry an uncommitted debug flag the way
#   staging's .env once carried TYPO_MASK_DEBUG. The result is tagged by the
#   commit it came from, so "which commit is this container running" becomes a
#   literal string comparison instead of a live HTTP probe.
#
# USE
#   ./build-image.sh              build HEAD of the source tree (below)
#   ./build-image.sh <sha>        build a specific commit
#   SRC=/root/typortrait-prod ./build-image.sh
#
#   Prints the tag it built, on its own final line, so a caller can capture it:
#     TAG=$(./build-image.sh)
#
# WHAT IT DOES NOT DO
#   Does not touch any tree's .env or running container. Building is separate
#   from promoting a tree to the new image -- see promote.sh. A build that goes
#   wrong costs a few minutes and an unused image tag; nothing live changes
#   until a tree is deliberately promoted.
set -euo pipefail

# Defaults to the tree THIS COPY OF THE SCRIPT lives in, not a hardcoded tree -- a real
# run tonight, invoked from staging with no SRC=, silently built from prod's source
# instead (the previous default) because prod happened to still exist at some other
# commit. The image looked fine (built, tagged, no error) and was built from the WRONG,
# STALE commit, missing everything staging had just pulled. Whichever tree's ./ops/
# you actually run this from is now what it builds, matching how `tt` already resolves
# its own path -- no tree name to remember or get wrong.
_arg="${SRC:-$(cd "$(dirname "$0")/.." && pwd)}"
# `git -C <anywhere inside a repo> ...` succeeds -- git walks UPWARD looking for .git, so
# it can't be used to tell "this is the repo root" from "this is some subdirectory of it".
# The first version of this script tried `[ -d "$TREE/.git" ]` plus a `dirname` fallback,
# which broke two different ways: a worktree's .git is a FILE, not a directory, so that
# check fails on a perfectly good repo; and passing SRC=.../typography_engine (a valid
# subdirectory) satisfied the check WITHOUT correcting to the root, so the later `git
# archive ... -- typography_engine` pathspec looked for a typography_engine/ folder
# inside typography_engine/ and found nothing. `--show-toplevel` asks git directly for
# the actual root, correct for a plain clone, a worktree, or a subdirectory of either.
TREE="$(git -C "$_arg" rev-parse --show-toplevel 2>/dev/null)" || true
# `|| true` above matters under `set -e`: without it, a FAILING command inside a bare
# assignment kills the script right there -- silently, with git's own exit code, before
# the friendly message below ever runs. A real thing that happened testing this: SRC
# pointing at a path with no git repo exited 128 with no output at all.
[ -n "$TREE" ] || { echo "no git repo at or above $_arg -- pass SRC=<tree with a working checkout>, e.g. SRC=/root/typortrait-stg"; exit 1; }

SHA="${1:-$(git -C "$TREE" rev-parse --short HEAD)}"
git -C "$TREE" cat-file -e "$SHA" 2>/dev/null || { echo "no commit $SHA in $TREE"; exit 1; }

TAG="typortrait:$SHA"
echo "building $TAG from the COMMITTED state at $SHA in $TREE" >&2
echo "(uncommitted changes in that working tree, if any, are ignored -- git archive exports only what is committed)" >&2

# .git lives at the REPO ROOT ($TREE), one level above typography_engine/ -- the Dockerfile
# and everything docker build needs are in that subdirectory, not the repo root, which also
# holds sites/ and other unrelated content. `git archive <sha> -- typography_engine` scopes
# the export to that subtree, but its paths still read "typography_engine/Dockerfile" etc;
# extracted with --strip-components=1 that leading directory is removed so Dockerfile lands
# at the context root, which is where `docker build` requires it. Piping the archive straight
# into `docker build -` (the first version of this script) would have built successfully
# -looking images from the WRONG context -- caught only by extracting a real archive to a
# directory and checking for Dockerfile before this ever ran against a live tree.
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
git -C "$TREE" archive "$SHA" -- typography_engine | tar -x --strip-components=1 -C "$CTX"
[ -f "$CTX/Dockerfile" ] || { echo "no Dockerfile at the extracted context root -- something is wrong, refusing to build"; exit 1; }
docker build -t "$TAG" -t typortrait:latest "$CTX"

echo "built $TAG" >&2
echo "$TAG"
