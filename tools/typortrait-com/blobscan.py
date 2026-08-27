#!/usr/bin/env python3
"""Measure the white blobs in a rendered preview instead of guessing at them.

Four variables have now been proposed and eliminated by eye. This reads the
actual failing image off disk and reports what the blobs ARE: how many, where,
how big, whether they are clipped to pure white or merely pale, and how round or
ragged their outlines are.

That distinguishes the candidate causes, which predict different signatures:

  saturated ink  -- clipped at 255, ragged outline following glyph rows,
                    scattered through bright regions, area varies with the words
  a filled shape -- uniform interior, smooth convex outline, one per face,
                    anchored to a facial feature
  matte hole     -- background colour rather than white, at the subject edge
  highlight wash -- soft-edged, not clipped, confined to the lit side

Usage:
    python3 blobscan.py                       # newest *_preview.png in outputs
    python3 blobscan.py <path-to-image>
    python3 blobscan.py <path> 245            # custom brightness threshold

Writes <name>-blobs.png alongside the input: the detected regions filled in
magenta over a dimmed copy, so what the numbers describe can be seen.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

OUTPUTS = "/root/typortrait-stg/typography_engine/data/outputs"
MIN_AREA = 400          # ignore glyph strokes and watermark text


def newest_preview():
    hits = sorted(glob.glob(os.path.join(OUTPUTS, "*_preview.png")),
                  key=os.path.getmtime)
    if not hits:
        raise SystemExit("no *_preview.png under %s" % OUTPUTS)
    return hits[-1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest_preview()
    thr = int(sys.argv[2]) if len(sys.argv) > 2 else 245
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    H, W = a.shape[:2]
    print("image      %s  %dx%d" % (path, W, H))
    print("threshold  all channels >= %d" % thr)

    bright = (a[..., 0] >= thr) & (a[..., 1] >= thr) & (a[..., 2] >= thr)
    print("bright px  %d (%.2f%% of frame)" % (bright.sum(), 100.0 * bright.mean()))
    clipped = (a[..., 0] >= 254) & (a[..., 1] >= 254) & (a[..., 2] >= 254)
    print("clipped px %d (%.2f%%)  <- pure white means the composite saturated"
          % (clipped.sum(), 100.0 * clipped.mean()))

    try:
        import cv2
    except Exception:                                      # noqa: BLE001
        raise SystemExit("cv2 unavailable -- run this inside the container")

    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        bright.astype(np.uint8), 8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        comp = (lab == i)
        # solidity: area / convex hull area. A filled shape is near 1.0; text
        # rows merging into a patch are much lower.
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        sol = 0.0
        if cnts:
            hull = cv2.convexHull(cnts[0])
            ha = float(cv2.contourArea(hull))
            if ha > 0:
                sol = float(cv2.contourArea(cnts[0])) / ha
        inner = a[comp]
        blobs.append((area, x, y, w, h, sol, float(inner.mean()),
                      float(inner.std())))

    blobs.sort(reverse=True)
    print("\nregions >= %d px: %d" % (MIN_AREA, len(blobs)))
    print("%8s %6s %6s %6s %6s %7s %8s %7s"
          % ("area", "x", "y", "w", "h", "solid", "mean", "sd"))
    for b in blobs[:15]:
        print("%8d %6d %6d %6d %6d %7.2f %8.1f %7.2f"
              % (b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]))

    if blobs:
        vis = (np.asarray(im).astype(np.float32) * 0.45).astype(np.uint8)
        vis[bright] = (255, 0, 255)
        out = os.path.splitext(path)[0] + "-blobs.png"
        Image.fromarray(vis).save(out)
        print("\nwrote %s  (detected regions in magenta)" % out)
    else:
        print("\nnothing above threshold -- lower it, e.g. blobscan.py <path> 235")


if __name__ == "__main__":
    main()
