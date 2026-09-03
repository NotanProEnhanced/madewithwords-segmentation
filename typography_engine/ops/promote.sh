#!/bin/bash
# Point ONE tree at an already-built image and recreate its container. Never builds.
#
# WHY
#   Separates "a new image exists" from "this tree is running it" into two explicit
#   steps, so a rollout can be staggered on purpose -- staging first, watched, THEN
#   the live brands one at a time -- instead of five trees racing five independent
#   builds of the same commit and hoping they agree.
#
# USE
#   ./promote.sh typortrait-lovedinwords typortrait:af209dd
#   ./promote.sh typortrait-lovedinwords              # re-promote to whatever IMAGE_TAG
#                                                      # already says (e.g. after --build
#                                                      # was mistakenly used and drifted it)
#
# SAFETY
#   Refuses if the image is not already built locally -- run build-image.sh first. Backs
#   up .env before touching it, same convention as rotate-printful-secret.sh.
set -euo pipefail

TREE="${1:?usage: promote.sh <tree-dir-name> [image-tag]}"
ENG="/root/$TREE/typography_engine"
[ -d "$ENG" ] || { echo "no tree at $ENG"; exit 1; }
ENVF="$ENG/.env"
[ -f "$ENVF" ] || { echo "no .env at $ENVF"; exit 1; }

TAG="${2:-$(grep -E '^IMAGE_TAG=' "$ENVF" 2>/dev/null | head -1 | cut -d= -f2-)}"
[ -n "$TAG" ] || { echo "no image tag given and none set in $ENVF"; exit 1; }

docker image inspect "$TAG" >/dev/null 2>&1 || {
    echo "no local image '$TAG' -- build it first:"
    echo "    ./build-image.sh"
    exit 1
}

cp -a "$ENVF" "$ENVF.bak-$(date +%Y%m%d-%H%M%S)"
if grep -q '^IMAGE_TAG=' "$ENVF"; then
    sed -i "s|^IMAGE_TAG=.*|IMAGE_TAG=$TAG|" "$ENVF"
else
    printf 'IMAGE_TAG=%s\n' "$TAG" >> "$ENVF"
fi

echo "-- $TREE -> $TAG"
( cd "$ENG" && docker compose up -d )

# Prove the promotion took, not just that the command exited 0 -- see tonight's
# `tt build` incident, where a command exiting 0 having done nothing was mistaken
# for success. Ask `docker compose ps`, run from the SAME directory with the SAME
# bare invocation `tt` already uses -- every tree is a full checkout and so holds
# all five docker-compose*.yml files, and only that tree's own .env (COMPOSE_FILE
# or equivalent, not tracked in git) says which one is actually in play. Grepping
# a filename here would guess; asking compose from this directory does not.
CN="$(cd "$ENG" && docker compose ps --format '{{.Name}}' 2>/dev/null | head -1)" || true
# `|| true` matters under set -euo pipefail: without it, `docker compose ps` failing
# (empty project, container not yet registered) kills the whole SCRIPT right here --
# silently, no message, right after "docker compose up -d" may have just succeeded.
# Proven, not assumed: the equivalent bug in build-image.sh did exactly this on the
# first real run tonight (exit 128, nothing printed) before this same fix was applied
# there; the identical shape here was found by testing for it, not by inspection.
[ -n "$CN" ] || { echo "   could not identify the running container -- check by hand: (cd $ENG && docker compose ps)"; exit 1; }
RUNNING="$(docker inspect "$CN" --format '{{.Config.Image}}' 2>/dev/null || echo '?')"
if [ "$RUNNING" = "$TAG" ]; then
    echo "   confirmed: container is running $TAG"
else
    echo "   WARNING: container reports image '$RUNNING', expected '$TAG' -- check by hand"
fi
