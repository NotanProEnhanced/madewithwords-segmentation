#!/bin/bash
# Render the fixed test set and keep the results, so "is this better?" becomes a comparison.
#
# WHY
#   Every render judged during a day of engine work was a fresh photograph with different
#   words, sometimes captured mid-load. Under those conditions a change that made things
#   worse survived several rounds before anyone could prove it, and three separate
#   regressions shipped that way.
#
#   The engine is DETERMINISTIC -- rng is seeded with a fixed value -- so the same photo
#   and the same words give a byte-identical render. That makes a real before/after
#   possible, but only if the photo and the words never change. Hence a fixed set and a
#   fixed block of text, both stored outside any tree so a revert cannot take them.
#
#   Outputs are filed under the commit they were produced at, so a change that breaks the
#   couple or the side-lit face is visible by opening two files rather than by remembering.
#
# USE
#   ./render-testset.sh                 render every image at the current commit
#   ./render-testset.sh 05 06           render only those
#   BRAND=faithinwords ./render-testset.sh
#
#   Talks to the container directly on its host port, so no basic-auth credentials are
#   needed and nothing depends on nginx.
#
# WHERE
#   sources   /root/typortrait-testset/src/*.jpg
#   words     /root/typortrait-testset/words.txt
#   results   /root/typortrait-testset/out/<commit>/
set -uo pipefail

SET="${SET:-/root/typortrait-testset}"
TREE="${TREE:-/root/typortrait-stg}"
PORT="${PORT:-8078}"
BASE="http://127.0.0.1:$PORT"
BRAND="${BRAND:-lovedinwords}"

# The studio's own memorial-preview parameters, so a baseline reflects what a visitor
# actually sees rather than some other configuration. Change these only deliberately: they
# are part of what makes two runs comparable.
DEF_PNG_W=900; DEF_RENDER_W=1500; DEF_STYLE=displacement; DEF_MIN_FONT=57
DEF_BRAND=lovedinwords
PNG_W="${PNG_W:-$DEF_PNG_W}"
RENDER_W="${RENDER_W:-$DEF_RENDER_W}"
STYLE="${STYLE:-$DEF_STYLE}"
INK="${INK:-photo}"
GROUND="${GROUND:-navy}"
BACKDROP="${BACKDROP:-studio}"
ASPECT="${ASPECT:-0.8}"
MIN_FONT="${MIN_FONT:-$DEF_MIN_FONT}"
PET="${PET:-}"                 # PET=1 to exercise the landmark-free engine instead

[ -d "$SET/src" ] || { echo "no sources at $SET/src"; exit 1; }
[ -s "$SET/words.txt" ] || { echo "no words at $SET/words.txt"; exit 1; }
WORDS="$(tr -d '\r\n' < "$SET/words.txt")"

COMMIT="$(git -C "$TREE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY="$(git -C "$TREE" status --porcelain 2>/dev/null | wc -l)"
# A run at non-default parameters is NOT comparable with the baseline, so it must not be
# filed on top of it. Rendering the download size into out/<commit>/ would overwrite the
# preview-size files the baseline is made of, and nothing would say it had happened.
SIG=""
[ "$PNG_W"    = "$DEF_PNG_W" ]    || SIG="$SIG-png$PNG_W"
[ "$RENDER_W" = "$DEF_RENDER_W" ] || SIG="$SIG-rw$RENDER_W"
[ "$MIN_FONT" = "$DEF_MIN_FONT" ] || SIG="$SIG-mf$MIN_FONT"
[ "$STYLE"    = "$DEF_STYLE" ]    || SIG="$SIG-$STYLE"
[ "$BRAND"    = "$DEF_BRAND" ]    || SIG="$SIG-$BRAND"
[ -z "$PET" ]                     || SIG="$SIG-pet"

OUT="$SET/out/$COMMIT$SIG"
[ "$DIRTY" = "0" ] || OUT="$OUT-dirty"
mkdir -p "$OUT"

# WAIT for the container, do not probe once. `docker compose up -d` returns when the
# container has STARTED; uvicorn then spends fifteen to twenty seconds loading MediaPipe
# before it answers anything. A single probe loses that race every time the two commands
# are run as a block, and reports it as "is the container up?" -- which it is.
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
            echo
            echo "nothing answering on $BASE after ${WAIT}s."
            echo "The container may have failed to start. Look at why:"
            echo "    docker compose logs --tail 40"
            exit 1
        fi
        printf '.'
        sleep 3
    done
    echo " up"
fi

note=""; [ "$DIRTY" = "0" ] || note=" ($DIRTY uncommitted)"
echo "commit $COMMIT$note   brand $BRAND   -> $OUT"
echo

want=("$@")
ok=0; bad=0
for f in "$SET"/src/*.jpg "$SET"/src/*.png; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"; n="${b%%-*}"
    if [ ${#want[@]} -gt 0 ]; then
        hit=0; for w in "${want[@]}"; do [ "$w" = "$n" ] && hit=1; done
        [ "$hit" = "1" ] || continue
    fi
    printf '  %-22s ' "$b"
    t0=$(date +%s)
    resp=$(curl -s --max-time 300 -X POST "$BASE/render" \
        -F "image=@$f" \
        -F "words=$WORDS" \
        -F "message=$WORDS" \
        -F "style=$STYLE" -F "ink=$INK" -F "ground=$GROUND" -F "backdrop=$BACKDROP" \
        -F "png_width=$PNG_W" -F "render_w=$RENDER_W" -F "aspect=$ASPECT" \
        -F "min_font_px=$MIN_FONT" -F "remove_bg=true" -F "uppercase=true" \
        -F "brand=$BRAND" -F "ref=$BRAND" -F "biometric_consent=on" \
        ${PET:+-F "pet=1" -F "pet_type=small"} 2>&1)
    dt=$(( $(date +%s) - t0 ))

    # `preview` is a URL path on the same host; pull it down and keep it under the commit.
    prev=$(printf '%s' "$resp" | sed -n 's/.*"preview"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
    if [ -z "$prev" ]; then
        err=$(printf '%s' "$resp" | sed -n 's/.*"error"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
        det=$(printf '%s' "$resp" | sed -n 's/.*"detail"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
        echo "FAILED  ${err:-no preview in response} ${det}"
        bad=$((bad + 1))
        continue
    fi
    if curl -s --max-time 120 -o "$OUT/${b%.*}.png" "$BASE$prev"; then
        printf 'ok  %3ds  %s\n' "$dt" "$(du -h "$OUT/${b%.*}.png" | cut -f1)"
        ok=$((ok + 1))
    else
        echo "FAILED to fetch $prev"; bad=$((bad + 1))
    fi
done

echo
echo "$ok rendered, $bad failed   ->  $OUT"
[ "$DIRTY" = "0" ] || echo "NOTE: the tree has uncommitted changes, so this is not a reproducible point."
echo
echo "Compare against another run:"
ls -1dt "$SET"/out/*/ 2>/dev/null | head -3 | sed 's/^/  /'
exit $([ "$bad" = "0" ] && echo 0 || echo 1)
