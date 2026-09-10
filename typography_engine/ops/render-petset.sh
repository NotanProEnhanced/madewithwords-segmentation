#!/bin/bash
# Render the fixed PET test set through the running container and keep the results, filed
# by commit, with the engine's own claim metrics beside each image.
#
# WHY
#   render-testset.sh does this for the human engine. The pet engine (app/pet_v2) is judged
#   on four numbers it prints per render -- type-only likeness, letter-body collisions,
#   exposed space, footprint -- and on the image. Both are deterministic (rng is seeded), so
#   the same photo, words and settings give the same bytes and the same numbers. A change
#   that moves either is visible by comparing two runs, not by remembering.
#
#   The numbers come from the container's log (PET_V2_VERBOSE=1 in the tree's .env), so this
#   script reads the log after each render and stores what the engine said as <image>.metrics.
#
# USE
#   ./render-petset.sh                  render every pet photo at the current commit
#   ./render-petset.sh 02 04            just those (by leading number of the filename)
#   ./pet-gate.sh <A> <B>               compare two runs: which images moved, and how the
#                                       numbers moved; non-zero exit if any got worse
#
# WHERE (outside every tree, like the human set, so a revert cannot take it)
#   sources   /root/typortrait-testset/pets/src/NN-name.jpg
#   words     /root/typortrait-testset/pets/words.txt          default for every photo
#             /root/typortrait-testset/pets/src/NN-name.words  per-photo override
#   results   /root/typortrait-testset/pets/out/<commit>/
#
# The site's own pet preview parameters: pawsinwords brand, Gallery Dark, 4:5, slider at
# Small (0.30), preview width 1400. Change these only deliberately; they make runs comparable.
set -uo pipefail
SET="${SET:-/root/typortrait-testset/pets}"
TREE="${TREE:-/root/typortrait-stg}"
PORT="${PORT:-8078}"
BASE="http://127.0.0.1:$PORT"
CONTAINER="${CONTAINER:-typortrait-staging}"
DEF_PNG_W=1400; DEF_GROUND=dark; DEF_ASPECT=0.8; DEF_TYPE=0.30; DEF_BRAND=pawsinwords
PNG_W="${PNG_W:-$DEF_PNG_W}"
GROUND="${GROUND:-$DEF_GROUND}"
ASPECT="${ASPECT:-$DEF_ASPECT}"
PET_TYPE="${PET_TYPE:-$DEF_TYPE}"
BRAND="${BRAND:-$DEF_BRAND}"
# How to read the engine's log lines. Default: the container's log since the render began.
# Override for a non-docker service: LOG_CMD='tail -n 400 /path/to/uvicorn.log'.
LOG_CMD="${LOG_CMD:-}"

[ -d "$SET/src" ] || { echo "no sources at $SET/src"; exit 1; }
[ -s "$SET/words.txt" ] || { echo "no default words at $SET/words.txt"; exit 1; }
DEF_WORDS="$(tr -d '\r\n' < "$SET/words.txt")"
COMMIT="$(git -C "$TREE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY="$(git -C "$TREE" status --porcelain --untracked-files=no 2>/dev/null | wc -l)"   # tracked edits only: a tree's .env and gallery images are not code

SIG=""
[ "$PNG_W"    = "$DEF_PNG_W" ]  || SIG="$SIG-png$PNG_W"
[ "$GROUND"   = "$DEF_GROUND" ] || SIG="$SIG-$GROUND"
[ "$ASPECT"   = "$DEF_ASPECT" ] || SIG="$SIG-ar$ASPECT"
[ "$PET_TYPE" = "$DEF_TYPE" ]   || SIG="$SIG-ts$PET_TYPE"
if [ -n "${NAME:-}" ]; then
    OUT="$SET/out/$NAME"
else
    OUT="$SET/out/$COMMIT$SIG"
    [ "$DIRTY" = "0" ] || OUT="$OUT-dirty"
fi
mkdir -p "$OUT"

WAIT="${WAIT:-90}"
_up() {
    curl -sf --max-time 5 "$BASE/healthz" >/dev/null 2>&1 && return 0
    curl -sf --max-time 5 -o /dev/null "$BASE/static/index.html" 2>/dev/null
}
if ! _up; then
    printf 'waiting for %s ' "$BASE"
    _t0=$(date +%s)
    until _up; do
        if [ $(( $(date +%s) - _t0 )) -ge "$WAIT" ]; then
            echo; echo "nothing answering on $BASE after ${WAIT}s."; exit 1
        fi
        printf '.'; sleep 3
    done
    echo " up"
fi

# The engine's report card for the render that just finished. Two lines are read:
#   CLAIM METRICS: typography footprint covers 85.6% of the animal; exposed space (>4px ...) 0.9%;
#       ... 10618 words; collisions: letter BODIES overlapping 1.08%, any antialiased touch 1.97%
#   type-only likeness score (face-weighted SSIM, blur sigma=28.1): 0.5721
# and written as key=value so pet-gate.sh can diff them without parsing prose.
_metrics() {   # $1 = ISO time the render started, $2 = output file
    local since="$1" outf="$2" log
    if [ -n "$LOG_CMD" ]; then
        log=$(eval "$LOG_CMD" 2>&1)
    else
        log=$(docker logs --since "$since" "$CONTAINER" 2>&1)
    fi
    local claim like
    claim=$(printf '%s\n' "$log" | grep 'CLAIM METRICS' | tail -1)
    like=$(printf '%s\n' "$log" | grep 'type-only likeness score' | tail -1)
    if [ -z "$claim" ] && [ -z "$like" ]; then
        echo "metrics=missing   (is PET_V2_VERBOSE=1 set in the tree's .env?)" > "$outf"
        return 1
    fi
    {
        printf 'footprint=%s\n'  "$(printf '%s' "$claim" | sed -n 's/.*footprint covers \([0-9.]*\)%.*/\1/p')"
        printf 'exposed=%s\n'    "$(printf '%s' "$claim" | sed -n 's/.*exposed space ([^)]*) \([0-9.]*\)%.*/\1/p')"
        printf 'words=%s\n'      "$(printf '%s' "$claim" | sed -n 's/.*; \([0-9]*\) words;.*/\1/p')"
        printf 'collisions=%s\n' "$(printf '%s' "$claim" | sed -n 's/.*BODIES overlapping \([0-9.]*\)%.*/\1/p')"
        printf 'likeness=%s\n'   "$(printf '%s' "$like"  | sed -n 's/.*: \([0-9.]*\)$/\1/p')"
    } > "$outf"
}

note=""; [ "$DIRTY" = "0" ] || note=" ($DIRTY uncommitted)"
echo "commit $COMMIT$note   brand $BRAND   slider $PET_TYPE   -> $OUT"
echo

want=("$@")
ok=0; bad=0
for f in "$SET"/src/*.jpg "$SET"/src/*.jpeg "$SET"/src/*.png; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"; n="${b%%-*}"; stem="${b%.*}"
    if [ ${#want[@]} -gt 0 ]; then
        hit=0; for w in "${want[@]}"; do [ "$w" = "$n" ] && hit=1; done
        [ "$hit" = "1" ] || continue
    fi
    words="$DEF_WORDS"
    [ -s "$SET/src/$stem.words" ] && words="$(tr -d '\r\n' < "$SET/src/$stem.words")"
    printf '  %-22s ' "$b"
    since=$(date -u +%Y-%m-%dT%H:%M:%S)
    t0=$(date +%s)
    resp=$(curl -s --max-time 600 -X POST "$BASE/render" \
        -F "image=@$f" \
        -F "words=$words" \
        -F "pet=1" -F "pet_type=$PET_TYPE" -F "ground=$GROUND" \
        -F "png_width=$PNG_W" -F "aspect=$ASPECT" -F "remove_bg=true" -F "uppercase=true" \
        -F "brand=$BRAND" -F "ref=$BRAND" -F "biometric_consent=on" 2>&1)
    dt=$(( $(date +%s) - t0 ))
    prev=$(printf '%s' "$resp" | sed -n 's/.*"preview"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
    if [ -z "$prev" ]; then
        err=$(printf '%s' "$resp" | sed -n 's/.*"error"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
        echo "FAILED  ${err:-no preview in response}"
        bad=$((bad + 1))
        continue
    fi
    if curl -s --max-time 120 -o "$OUT/$stem.png" "$BASE$prev"; then
        _metrics "$since" "$OUT/$stem.metrics" || true
        printf 'ok  %3ds  %s  %s\n' "$dt" "$(du -h "$OUT/$stem.png" | cut -f1)" \
            "$(tr '\n' ' ' < "$OUT/$stem.metrics")"
        ok=$((ok + 1))
    else
        echo "FAILED to fetch $prev"; bad=$((bad + 1))
    fi
done

echo
echo "$ok rendered, $bad failed   ->  $OUT"
[ "$DIRTY" = "0" ] || echo "NOTE: the tree has uncommitted changes, so this is not a reproducible point."
