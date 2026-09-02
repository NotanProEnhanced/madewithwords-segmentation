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
#   FULL=1 renders at DELIVERED size (3600px) instead of preview size. Needs
#   TYPO_OPS_PREVIEW_PX set on the tree; it says so if the cap clamped it anyway.
#     FULL=1 ./render-one.sh photo.jpg "..." delivered.png
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

# FULL=1 -- render at the size the customer receives, not the size the browser shows.
#
# /render clamps output to PREVIEW_PNG_WIDTH, so by default nothing reachable from here can
# show you the delivered file. Measured on 01-hat: a typical glyph is 4px on screen against
# ~15px in the download. Judging type, tone or eyes on the preview is judging a quarter of
# the pixels, and the two smallest tiers are texture there and letterforms in the file.
#
# The tree must have TYPO_OPS_PREVIEW_PX set (staging only) or the cap still applies -- so
# this checks the result rather than trusting the request, and says so if it was clamped.
if [ "${FULL:-}" = "1" ]; then
    PNG_W="${PNG_W_OVERRIDE:-3600}"
    RENDER_W="${RENDER_W_OVERRIDE:-2600}"
fi

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

# Report the size actually returned. A clamped render looks exactly like a full one on
# disk, and a judgement made on it would be wrong in the direction that matters.
_w=$(python3 -c "import cv2,sys;i=cv2.imread(sys.argv[1]);print(i.shape[1] if i is not None else 0)" "$OUT" 2>/dev/null || echo 0)
[ "$_w" != "0" ] && echo "        $_w px wide"
if [ "${FULL:-}" = "1" ] && [ "$_w" != "0" ] && [ "$_w" -lt "$PNG_W" ]; then
    cat <<EOF

  NOT the delivered size: asked for ${PNG_W}px, got ${_w}px.
  /render caps at PREVIEW_PNG_WIDTH unless the tree sets TYPO_OPS_PREVIEW_PX.
  On staging, add to .env and recreate:   TYPO_OPS_PREVIEW_PX=3600
  Leave it unset on the live trees -- a 3600px preview is ~4x the pixels on every swatch.
EOF
fi
