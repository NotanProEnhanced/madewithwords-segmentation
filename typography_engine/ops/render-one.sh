#!/bin/bash
# Render ONE arbitrary photograph with arbitrary words, against any tree.
#
# WHY
#   render-testset.sh answers "did this change break the fixed set?". It cannot answer
#   "why does THIS customer's render look wrong?", because it only ever renders its own ten
#   images with its own fixed block of text.
#
#   When a live render looks wrong, the question is always which input is responsible: the
#   photograph, the words, the crop, the word-size floor, or the code. This changes one at a
#   time against the same engine, so the answer is a comparison rather than a theory.
#
# USE
#   ./render-one.sh <image> "<words>" [output.png]
#   ./render-one.sh photo.jpg "$(cat /root/typortrait-testset/words.txt)" long.png
#   MIN_FONT=120 ./render-one.sh photo.jpg "short phrase" giant.png
#   PORT=8080 BRAND=lovedinwords ./render-one.sh photo.jpg "..." live.png
#
# Same knobs as render-testset.sh, same defaults, so the two are directly comparable:
#   PORT PNG_W RENDER_W STYLE INK GROUND BACKDROP ASPECT MIN_FONT BRAND PET
#
# PRIVACY
#   A customer's source photograph is personal data under a ~30-day deletion promise.
#   Render it where you must, look, and delete both the copy and the output. Do not leave
#   either in data/outputs, which is inside the backup set.
set -uo pipefail

PORT="${PORT:-8078}"
BASE="http://127.0.0.1:$PORT"
BRAND="${BRAND:-lovedinwords}"
PNG_W="${PNG_W:-900}"
RENDER_W="${RENDER_W:-1500}"
STYLE="${STYLE:-displacement}"
INK="${INK:-photo}"
GROUND="${GROUND:-navy}"
BACKDROP="${BACKDROP:-studio}"
ASPECT="${ASPECT:-0.8}"
MIN_FONT="${MIN_FONT:-57}"
PET="${PET:-}"

IMG="${1:-}"
WORDS="${2:-}"
OUT="${3:-render-one.png}"

[ -f "$IMG" ] || { echo "usage: $0 <image> \"<words>\" [output.png]"; exit 1; }
[ -n "$WORDS" ] || { echo "no words given -- pass them as the second argument"; exit 1; }

echo "$BASE   brand=$BRAND  min_font=$MIN_FONT  png=$PNG_W  render_w=$RENDER_W  aspect=$ASPECT"
echo "image:  $IMG"
echo "words:  $(printf '%s' "$WORDS" | wc -w) words, $(printf '%s' "$WORDS" | wc -c) characters"
echo

resp=$(curl -s --max-time 300 -X POST "$BASE/render" \
    -F "image=@$IMG" \
    -F "words=$WORDS" -F "message=$WORDS" \
    -F "style=$STYLE" -F "ink=$INK" -F "ground=$GROUND" -F "backdrop=$BACKDROP" \
    -F "png_width=$PNG_W" -F "render_w=$RENDER_W" -F "aspect=$ASPECT" \
    -F "min_font_px=$MIN_FONT" -F "remove_bg=true" -F "uppercase=true" \
    -F "brand=$BRAND" -F "ref=$BRAND" -F "biometric_consent=on" \
    ${PET:+-F "pet=1" -F "pet_type=small"} 2>&1)

prev=$(printf '%s' "$resp" | sed -n 's/.*"preview"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
if [ -z "$prev" ]; then
    echo "FAILED"
    printf '%s\n' "$resp" | head -c 600
    echo
    exit 1
fi
curl -s --max-time 120 -o "$OUT" "$BASE$prev" && echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
