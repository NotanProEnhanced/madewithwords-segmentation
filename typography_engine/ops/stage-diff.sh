#!/bin/bash
# Find which numbered stage introduced a dark region, from a TYPO_DUMP_STAGES dump.
#
# WHY
#   The staged dump writes 16 full-frame PNGs, one per rendering pass. For a defect
#   confined to a small region (an eye, say), scanning 16 full images by eye to spot which
#   one changed is exactly the kind of thing this project keeps building instruments to
#   avoid. The dump's own comment says as much: "Reasoning from the code about which one
#   did it failed four times on a single defect -- each wrong guess costing a rebuild and
#   an upload." This automates the pairwise comparison it already recommends.
#
# USE
#   TYPO_DUMP_STAGES=/app/outputs/stagedump ./ops/render-one.sh photo.jpg "words" out.png
#   ./ops/stage-diff.sh data/outputs/stagedump
#
#   (TYPO_DUMP_STAGES is a container-side path; if it's under /app/outputs, the host sees
#   it at data/outputs/<name> because that directory is bind-mounted.)
set -uo pipefail

DIR="${1:-}"
[ -n "$DIR" ] && [ -d "$DIR" ] || { echo "usage: $0 <dump directory>"; exit 1; }

python3 - "$DIR" <<'PY'
import sys, glob, os
import numpy as np
try:
    import cv2
except ImportError:
    sys.exit("needs opencv-python (cv2) -- run inside the container, or pip install opencv-python-headless")

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "a-*.png")))
if len(files) < 2:
    sys.exit(f"found {len(files)} files in {d} -- expected ~16 (a-01-*.png .. a-16-*.png)")

def load(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        sys.exit(f"could not read {p}")
    return im.astype(np.float32) / 255.0

prev_name, prev = os.path.basename(files[0]), load(files[0])
print(f"{len(files)} stages, {prev.shape[1]}x{prev.shape[0]}")
print(f"{'stage':30s} {'new-dark px':>12s} {'darkest region (x,y,w,h)':>28s} {'region mean before->after':>28s}")

for f in files[1:]:
    name, cur = os.path.basename(f), load(f)
    # "Newly dark" = pixels that dropped below 0.05 (near-black) in THIS pass and were not
    # already there -- isolates what THIS stage darkened, not what earlier stages already had.
    went_dark = (cur < 0.05) & (prev >= 0.05)
    n = int(went_dark.sum())
    region = ""
    means = ""
    if n > 0:
        ys, xs = np.where(went_dark)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        region = f"({x0},{y0},{x1-x0+1},{y1-y0+1})"
        means = f"{prev[y0:y1+1, x0:x1+1].mean():.3f} -> {cur[y0:y1+1, x0:x1+1].mean():.3f}"
    print(f"{prev_name+' -> '+name:30.30s} {n:12d} {region:>28s} {means:>28s}")
    prev_name, prev = name, cur

print("\nRead this as: the pass with the largest 'new-dark px' in a small, eye-sized region")
print("(not spread across the whole frame) is the one that painted the disc. Open that")
print("stage's PNG and the one just before it to confirm by eye.")
PY
