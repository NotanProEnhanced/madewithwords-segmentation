#!/bin/bash
# Does the on-screen PREVIEW show the same artwork as the file the customer buys?
#
# WHY
#   The preview renders at supersample 1 and ~900-1400px; the paid file at supersample 2
#   and 3600px (see disp_ss in main.py, DOWNLOAD_PNG_WIDTH in config.py). Every aesthetic
#   judgement anyone makes -- "the type is too small", "27 is unreadable", "the eyes look
#   wrong" -- is made through the preview. If the preview composes DIFFERENTLY, then all of
#   those judgements are about the preview rather than about the product.
#
#   displacement.py claims it does not: `_ssn = SS / 2.0` scales the four type tiers by the
#   same factor as the canvas, so a glyph should occupy the same FRACTION of the frame at
#   either setting. That is a claim in a comment. This measures it.
#
# WHAT IT ANSWERS
#   1. Do glyphs occupy the same fraction of the frame in both?  <- the composition question
#      Measured as glyph height in % of image height, so the two sizes compare directly.
#   2. Downscaled to the same size, how different are the two images?
#
#   The RATIO in (1) is the load-bearing number. The image difference in (2) is context,
#   not a verdict: two renders of identical composition still differ by 20+ levels once
#   one has been resampled from 4x the pixels -- antialiasing alone accounts for it, as a
#   synthetic pair of known-identical layouts confirms. Read it for WHERE they disagree
#   (see diff.png) rather than for a pass/fail number.
#
# USE
#   ./preview-fidelity.sh <image> "<words>" [outdir]
#   PORT=8080 BRAND=lovedinwords ./preview-fidelity.sh photo.jpg "GRACE, KIND, MOM"
#
# PRIVACY
#   Same rule as render-one.sh: a customer photograph is personal data under a ~30-day
#   deletion promise. Render it, look, then delete the copy AND the outputs. Do not leave
#   either in data/outputs, which is inside the backup set.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IMG="${1:-}"
WORDS="${2:-}"
OUTDIR="${3:-./preview-fidelity}"

[ -f "$IMG" ] || { echo "usage: $0 <image> \"<words>\" [outdir]"; exit 1; }
[ -n "$WORDS" ] || { echo "no words given -- pass them as the second argument"; exit 1; }
mkdir -p "$OUTDIR"

# The two paths, differing ONLY in the size knobs. png_width < 1200 is what selects
# supersample 1 in main.py, so 900 is the real preview path and 3600 the real paid one.
echo "== preview path  (png 900, supersample 1)"
PNG_W=900  RENDER_W=1500 "$HERE/render-one.sh" "$IMG" "$WORDS" "$OUTDIR/preview.png" || exit 1
echo
echo "== paid path     (png 3600, supersample 2)"
PNG_W=3600 RENDER_W=2600 "$HERE/render-one.sh" "$IMG" "$WORDS" "$OUTDIR/paid.png"    || exit 1
echo

python3 - "$OUTDIR/preview.png" "$OUTDIR/paid.png" "$OUTDIR/diff.png" <<'PY'
import sys, os, cv2, numpy as np

prev_p, paid_p, diff_p = sys.argv[1:4]
prev = cv2.imread(prev_p); paid = cv2.imread(paid_p)
if prev is None or paid is None:
    sys.exit("could not read one of the renders -- did both paths succeed?")

def glyph_heights(bgr):
    """Height of each ink blob, as a percentage of image height.

    Glyphs are the dark marks on the ground (or the light marks on a dark ground), so
    threshold against the frame's own median rather than a fixed value -- the two grounds
    and every ink choice would otherwise need their own constant. Blobs under 2px or over
    a tenth of the frame are dropped: those are speckle and background regions, not type."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    med = float(np.median(g))
    ink = (g < med - 18) if med > 127 else (g > med + 18)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    H = bgr.shape[0]
    h = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    h = h[(h >= 2) & (h <= H * 0.10)]
    return (h / H) * 100.0, H

# MEASURE BOTH AT THE SAME PIXEL SIZE.
#
# Connected components are not scale-invariant: at 900px adjacent letters merge into one
# blob, at 3600px they separate. The same artwork gave 1843 blobs at 900px and 5460 at
# 3600px, and the median blob height fell 31% purely from that -- which this script first
# reported as "COMPOSITION DIFFERS". It was the instrument, not the engine. The tell was
# p75 being IDENTICAL in both: a real composition change scales every percentile together.
#
# So the ratio is computed on the paid render downscaled to the preview's size, where both
# sides merge alike. Native-resolution numbers are still reported below, for legibility --
# a different question, and the one that is genuinely about delivered pixels.
paid_small = cv2.resize(paid, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
ph, pH = glyph_heights(prev)
dh, dH = glyph_heights(paid_small)
native_dh, _ = glyph_heights(paid)       # native resolution: the legibility question
print(f"preview  {prev.shape[1]}x{prev.shape[0]}   {len(ph):6d} ink blobs")
print(f"paid     {paid.shape[1]}x{paid.shape[0]}, measured at {prev.shape[1]}x{prev.shape[0]}"
      f"   {len(dh):6d} ink blobs")
print()
print("glyph height as % of frame height -- these should MATCH if composition is faithful")
print(f"{'':10s} {'p25':>7s} {'median':>7s} {'p75':>7s} {'p95':>7s}")
for name, a in (("preview", ph), ("paid", dh)):
    if len(a) == 0: print(f"{name:10s}   (no blobs found)"); continue
    q = np.percentile(a, [25, 50, 75, 95])
    print(f"{name:10s} " + " ".join(f"{v:7.3f}" for v in q))
if len(ph) and len(dh):
    r = np.median(dh) / np.median(ph)
    # The band comes from the instrument's resolution, not from taste.
    #
    # A median glyph in the 900px preview is FOUR PIXELS. Blob heights are integers, so the
    # ratio can only land on 4/4, 4/5, 5/4 -- steps of roughly 25%. Synthetic controls at 4x
    # resolution confirm it: an identical layout reads 1.11, 1.2x larger reads 1.33, 1.4x
    # reads 1.56, 2x reads 2.11. Real differences separate cleanly; anything under ~25% does
    # not, and no band makes that untrue.
    #
    # That limit is itself worth knowing: the preview is too coarse to MEASURE its own type,
    # which is the same reason it is too coarse to judge it by eye. The fault this is hunting
    # -- _ssn failing to normalize supersample -- would read 0.5 or 2.0, far outside the noise.
    _ok = 0.75 <= r <= 1.30
    print(f"\nratio paid/preview at the median: {r:.3f}   "
          f"({'faithful -- within what a 4px preview glyph can resolve' if _ok else 'COMPOSITION DIFFERS -- look at _ssn'})")

# Same picture, same size: how far apart are they once resolution is taken out?
small = cv2.resize(paid, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
d = np.abs(small.astype(np.int16) - prev.astype(np.int16))
print(f"\npaid downscaled to preview size: mean abs diff {d.mean():6.2f} / 255"
      f"   (p95 {np.percentile(d, 95):.0f})")
cv2.imwrite(diff_p, (255 - np.clip(d.max(axis=2) * 3, 0, 255)).astype(np.uint8))
print(f"wrote {diff_p} -- dark areas are where the two disagree most")

# /render clamps to PREVIEW_PNG_WIDTH (main.py: preview_w = min(png_width, _preview_cap)),
# so asking it for 3600 does NOT produce the paid file -- it produces a capped one at
# supersample 2. The supersample comparison above is still valid; the RESOLUTION comparison
# is not, and saying so is the difference between a measurement and a misleading number.
DOWNLOAD_W = int(os.environ.get("TYPO_DOWNLOAD_PX", "3600") or 3600)
if paid.shape[1] < DOWNLOAD_W:
    print(f"\nNOTE: the 'paid' render came back {paid.shape[1]}px wide, not {DOWNLOAD_W}px --")
    print(f"  /render caps output at PREVIEW_PNG_WIDTH. Supersample 2 still applied, so the")
    print(f"  ratio above is a real SS=1 vs SS=2 comparison. Delivered sizes below are")
    print(f"  computed from the measured percentage against a {DOWNLOAD_W}px-wide file.")

if len(ph) and len(dh):
    dev_H = DOWNLOAD_W / 0.8                       # 4:5, the studio's print aspect
    print("\ntypical glyph, in the pixels each viewer actually gets:")
    print(f"  on screen (preview {pH}px tall) {np.median(ph)/100*pH:6.1f}px")
    print(f"  in the download ({int(dev_H)}px tall) {np.median(native_dh)/100*dev_H:6.1f}px")
    print("  Under ~10px a mark reads as texture, not as a word. If the first number is")
    print("  below that and the second is not, every judgement about small type has been")
    print("  made on something the customer never sees.")
PY
