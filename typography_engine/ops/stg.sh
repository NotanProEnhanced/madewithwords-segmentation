#!/bin/bash
# Save and restore a COMPLETE working state of a tree: code, settings and image.
#
# WHY
#   Staging is where new code is tried, so the thing that matters most is getting back to
#   something that worked. Git alone does not do it: `app/` is baked into the image, so a
#   revert still costs a rebuild, and the settings that change the product live in .env,
#   which git never sees. Reverting the code and leaving PET_SUBJECT_BASE at an
#   experimental value restores nothing.
#
#   So a state here is all three:
#     the commit           what the code was
#     a copy of .env       what the settings were
#     a tagged Docker image  what was actually running
#
#   Restoring re-tags the saved image rather than rebuilding, so it takes seconds. The
#   rebuild is the slow part and the saved image makes it unnecessary.
#
# USE
#   ./stg.sh save good-hat-render      BEFORE experimenting, and after anything you like
#   ./stg.sh list                      what is saved
#   ./stg.sh restore good-hat-render   back to it, no rebuild
#   ./stg.sh drop old-name             remove a state and its image
#
# HABIT THAT MATTERS
#   Save at the START of a session, not only when something turns out well. You only know
#   a render is good after you have tested it, and by then you may have already replaced
#   the state you wanted to keep.
#
# Defaults to the staging tree; TREE=/root/typortrait-prod ./stg.sh ... for another.
set -uo pipefail

TREE="${TREE:-/root/typortrait-stg}"
ENG="$TREE/typography_engine"
STORE="${STORE:-/root/.tree-states}"

[ -d "$ENG" ] || { echo "no tree at $ENG"; exit 1; }

_img() {   # the image this tree runs, from its own .env
    local t
    t=$(grep -E '^IMAGE_TAG=' "$ENG/.env" 2>/dev/null | head -1 | cut -d= -f2-)
    echo "${t:-typortrait-staging:latest}"
}
_base() { _img | cut -d: -f1; }

# Read one field from a manifest. Deliberately NOT `source`: a value with a space in it --
# saved=2026-09-01 03:16 -- makes the shell try to run "03:16", which silently broke both
# restore and list. Parsing also means a state directory can never execute anything.
_m() { grep -E "^$1=" "$2/manifest" 2>/dev/null | head -1 | cut -d= -f2-; }

cmd="${1:-}"
name="${2:-}"

case "$cmd" in
save)
    [ -n "$name" ] || { echo "usage: $0 save <name>"; exit 1; }
    d="$STORE/$name"
    mkdir -p "$d" || exit 1
    img="$(_img)"; base="$(_base)"
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "no image $img -- start the container once before saving"; exit 1
    fi
    docker image tag "$img" "$base:state-$name" || exit 1
    cp "$ENG/.env" "$d/env" 2>/dev/null || { echo "could not copy .env"; exit 1; }
    chmod 600 "$d/env"
    {
        echo "tree=$TREE"
        echo "commit=$(git -C "$TREE" rev-parse HEAD 2>/dev/null)"
        echo "branch=$(git -C "$TREE" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        echo "image=$base:state-$name"
        echo "saved=$(date '+%Y-%m-%d %H:%M')"
        echo "dirty=$(git -C "$TREE" status --porcelain | wc -l)"
    } > "$d/manifest"
    echo "saved '$name'"
    sed 's/^/  /' "$d/manifest"
    ;;

list)
    [ -d "$STORE" ] || { echo "no states saved yet"; exit 0; }
    printf '%-24s %-12s %-18s %s\n' NAME COMMIT SAVED DIRTY
    for d in "$STORE"/*/; do
        [ -f "$d/manifest" ] || continue
        c=$(_m commit "$d"); sv=$(_m saved "$d"); dy=$(_m dirty "$d")
        printf '%-24s %-12s %-18s %s\n' "$(basename "$d")" "${c:0:10}" "$sv" \
               "$([ "${dy:-0}" = "0" ] && echo "-" || echo "$dy uncommitted")"
    done
    ;;

restore)
    [ -n "$name" ] || { echo "usage: $0 restore <name>"; exit 1; }
    d="$STORE/$name"
    [ -f "$d/manifest" ] || { echo "no state '$name' -- try: $0 list"; exit 1; }
    commit=$(_m commit "$d"); image=$(_m image "$d"); saved=$(_m saved "$d")
    echo "restoring '$name'  (commit ${commit:0:10}, saved $saved)"

    # 1) code. Checked out into the working tree, not a detached HEAD, so work can
    #    continue from here without a surprise later.
    if [ -n "${commit:-}" ]; then
        git -C "$TREE" checkout -q "$commit" -- typography_engine/ 2>/dev/null \
            && echo "  code restored" || echo "  WARNING: could not restore code at $commit"
        # ...but NOT ops/. These scripts are the instruments -- the harness, the state
        # store, the comparison. Rolling them back to whatever they were when the state was
        # saved removes the tools you are in the middle of using, and does it silently. A
        # state is the PRODUCT (app/, .env, image); the instruments stay at HEAD.
        if git -C "$TREE" rev-parse --verify -q HEAD >/dev/null; then
            git -C "$TREE" checkout -q HEAD -- typography_engine/ops/ 2>/dev/null \
                && echo "  ops/ left at HEAD (tools are not part of a state)"
        fi
    fi

    # 2) settings. The half that git cannot see, and the half that silently keeps an
    #    experiment alive after the code is back.
    cp "$d/env" "$ENG/.env" && chmod 600 "$ENG/.env" && echo "  .env restored"

    # 3) the image. Re-tagging is what makes this fast; without it every restore is a
    #    rebuild, and a slow rollback is one you avoid using when you most need it.
    if docker image inspect "$image" >/dev/null 2>&1; then
        docker image tag "$image" "$(_img)" && echo "  image re-tagged (no rebuild)"
        (cd "$ENG" && docker compose up -d 2>&1 | tail -2)
    else
        # The saved image is gone (pruned, or a state from before 2026-09-03). Compose
        # files no longer carry `build: .` -- see ops/build-image.sh -- so a blind
        # `docker compose up -d --build` here would now fail instead of quietly working.
        # Rebuild the SAME commit this state recorded, then land it exactly where the
        # fast path above would have: retagged as this tree's IMAGE_TAG.
        echo "  saved image is gone -- rebuilding commit ${commit:0:10} instead (slower)"
        if [ -n "${commit:-}" ] && NEWTAG="$(SRC="$ENG" "$(dirname "$0")/build-image.sh" "$commit")"; then
            docker image tag "$NEWTAG" "$(_img)" && echo "  built and tagged as $(_img)"
            (cd "$ENG" && docker compose up -d 2>&1 | tail -2)
        else
            echo "  could not rebuild (no commit recorded, or the build failed) -- restore incomplete"
        fi
    fi
    echo "restored. Hard-reload the browser before judging a render."
    ;;

drop)
    [ -n "$name" ] || { echo "usage: $0 drop <name>"; exit 1; }
    d="$STORE/$name"
    [ -f "$d/manifest" ] || { echo "no state '$name'"; exit 1; }
    image=$(_m image "$d")
    docker image rm "$image" >/dev/null 2>&1 && echo "removed image $image"
    rm -rf "$d" && echo "dropped '$name'"
    ;;

*)
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
