#!/usr/bin/env python3
"""Find smooth patches inside the subject -- places where the typography is missing.

WHY NOT BRIGHTNESS
  The first version of this looked for near-white pixels. That was wrong. On a
  measured failing render: nothing clipped at 255, nothing above 235 formed a
  region larger than 400px, and the only large bright region was the studio
  backdrop itself (mean 230.6, sd 1.7, full height, touching the left border).
  The blobs sit ON the face at roughly the same level as the background, so no
  brightness threshold can separate them.

WHAT ACTUALLY DISTINGUISHES THEM
  Everywhere else on the subject carries glyph texture. A blob is smooth. So:

    1. local standard deviation over a small window -> texture
    2. smooth = texture below TEX, i.e. no type
    3. drop every component touching the image border -> that is the background,
       which is also smooth
    4. what remains is a smooth patch surrounded by type: a blob

  Reports position, size, mean level, and solidity (area / convex hull area) so
  a filled shape can be told from type that merely thinned out.

Usage:
    python3 blobscan.py <image> [tex] [minarea]
        tex      local sd below this counts as smooth (default 6.0)
        minarea  ignore components smaller than this (default 900)

Writes <name>-blobs.png: detected patches in magenta over a dimmed copy.
"""
import os
import sys

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:                                          # noqa: BLE001
    raise SystemExit("cv2 unavailable -- run this inside the container")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: blobscan.py <image> [tex] [minarea]")
    path = sys.argv[1]
    tex_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    min_area = int(sys.argv[3]) if len(sys.argv) > 3 else 900

    im = Image.open(path).convert("RGB")
    rgb = np.asarray(im).astype(np.float32)
    g = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = g.shape
    print("image     %s  %dx%d" % (path, W, H))

    # local sd over a 15px window -- large enough to span a glyph, small enough
    # to stay inside a blob
    k = 15
    mean = cv2.boxFilter(g, -1, (k, k), normalize=True)
    sq = cv2.boxFilter(g * g, -1, (k, k), normalize=True)
    sd = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    print("texture   sd: p5=%.1f p50=%.1f p95=%.1f   smooth if < %.1f"
          % (np.percentile(sd, 5), np.percentile(sd, 50),
             np.percentile(sd, 95), tex_thr))

    smooth = (sd < tex_thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(smooth, 8)

    # the background is smooth too, but it touches the frame edge
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    print("components %d, of which %d touch the border (background)"
          % (n - 1, len([i for i in border if i])))

    rows = []
    for i in range(1, n):
        if i in border:
            continue
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comp = (lab == i)
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        sol = 0.0
        if cnts:
            ha = float(cv2.contourArea(cv2.convexHull(cnts[0])))
            if ha > 0:
                sol = float(cv2.contourArea(cnts[0])) / ha
        vals = g[comp]
        rows.append((int(area), int(x), int(y), int(w), int(h), sol,
                     float(vals.mean()), float(np.percentile(vals, 95))))

    rows.sort(reverse=True)
    print("\ninterior smooth patches >= %d px: %d" % (min_area, len(rows)))
    print("%9s %6s %6s %6s %6s %7s %8s %8s"
          % ("area", "x", "y", "w", "h", "solid", "mean", "p95"))
    for r in rows[:20]:
        print("%9d %6d %6d %6d %6d %7.2f %8.1f %8.1f" % r)

    vis = (rgb * 0.40).astype(np.uint8)
    for i in range(1, n):
        if i in border or stats[i][4] < min_area:
            continue
        vis[lab == i] = (255, 0, 255)
    out = os.path.splitext(path)[0] + "-blobs.png"
    Image.fromarray(vis).save(out)
    print("\nwrote %s" % out)
    if not rows:
        print("nothing found -- raise tex (e.g. 9) or lower minarea")


if __name__ == "__main__":
    main()
