#!/usr/bin/env python3
"""Compare color saturation across regions of a source photograph.
WHY
  A render defect can look like "this region went dark/gray" without the ink-density
  field (TYPO_DUMP_STAGES) showing anything unusual -- if the ink formula preserves HUE
  but a source region is already close to neutral (low saturation), the formula produces
  something close to flat gray by construction, not by any code fault. This checks that
  directly against the real photograph, rather than arguing it from the formula alone.

  Written chasing the closed-eye disc defect (2026-09-03): every explicit
  disc-drawing/darkening mechanism in the render code was ruled out, and the density
  field showed nothing concentrated in that region either. If eye-region saturation is
  measurably lower than nearby lit skin here, that is direct, checkable support for "the
  hue-preserving ink formula collapses toward gray in a naturally low-saturation region,
  exposed only because nothing else (photo-eye overlay, synthetic iris) draws over it for
  a closed eye" -- rather than one more guess added to a list of ruled-out ones.

USE
  python3 region-saturation.py <photo.jpg>

  Prints mean saturation and value (HSV, 0-255 each) for a handful of approximate face
  regions -- eyes, cheeks, forehead. The comparison matters, not the exact coordinates:
  the built-in guesses assume a head-and-shoulders 4:5-ish portrait with eyes roughly
  35-42% down the frame. If they land visibly off-target for a given photo (crop it and
  re-run, or edit the region() calls below with the real fractional coordinates), the
  absolute numbers are still meaningful, just check they landed where intended.

READING IT
  Eye-region saturation clearly lower than cheek/forehead -> supports the hypothesis: the
  eye socket is a naturally low-saturation region, and the photo-hue-preserving ink
  formula renders that as closer to gray than the surrounding warmer, more saturated skin.
  Eye-region saturation similar to or higher than cheek/forehead -> does NOT support it;
  the disc's cause is still open, and the color-compositing code needs a closer read.
"""
import sys, cv2, numpy as np

im = cv2.imread(sys.argv[1])
if im is None:
    sys.exit(f"could not read {sys.argv[1]}")
H, W = im.shape[:2]
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
sat, val = hsv[..., 1], hsv[..., 2]


def region(cx_frac, cy_frac, r_frac, label):
    cx, cy, r = int(cx_frac * W), int(cy_frac * H), int(r_frac * W)
    y0, y1 = max(0, cy - r), min(H, cy + r)
    x0, x1 = max(0, cx - r), min(W, cx + r)
    s = sat[y0:y1, x0:x1]
    v = val[y0:y1, x0:x1]
    print(f"{label:14s} sat mean={s.mean():6.1f}  val mean={v.mean():6.1f}  (0-255 each)")


region(0.50, 0.38, 0.05, "between-eyes")
region(0.38, 0.38, 0.04, "left-eye")
region(0.62, 0.38, 0.04, "right-eye")
region(0.35, 0.55, 0.05, "cheek(L)")
region(0.65, 0.55, 0.05, "cheek(R)")
region(0.50, 0.30, 0.05, "forehead")
