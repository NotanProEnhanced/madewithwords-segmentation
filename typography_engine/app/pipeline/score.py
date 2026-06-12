"""Likeness scoring: how well a finished render preserves the subject's face.

Used to recommend the best STYLE for a given photo ("measure, don't guess"):
the studio probes each style with a coarse render, the server scores each one
here, and the style whose score is highest gets a quiet "best for this photo"
recommendation. The score is the zero-normalized cross-correlation between the
render's face region and the source photo's face region (both lightly blurred,
so we compare light/shadow STRUCTURE, not text grain) -- in [-1, 1], higher is
better. Best-effort: returns None whenever anything can't be resolved.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# The renderers compose onto a 4:5 print canvas (tonal._fit_print_canvas): the
# art keeps the working-image aspect, centred, ground-padded. The canvas width
# always equals the pre-pad render width, which lets us reconstruct the art
# rectangle here without threading geometry through every renderer.
_PRINT_ASPECT = 0.8


def likeness_score(an, png_bytes: bytes) -> Optional[float]:
    """Score a rendered PNG against the analysis' source photo. None on failure."""
    try:
        fbb = an.face_bbox
        if not fbb or not png_bytes:
            return None
        arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return None
        ch_, cw_ = arr.shape[:2]
        w0, h0 = float(an.img.w), float(an.img.h)
        if w0 < 8 or h0 < 8 or cw_ < 8 or ch_ < 8:
            return None
        # Locate the art rectangle inside the (possibly padded) canvas.
        if abs(cw_ / ch_ - _PRINT_ASPECT) < 0.01:
            hr = cw_ * h0 / w0                      # pre-pad render height
            if hr > ch_ + 1:                        # was downscaled, padded in x
                aw, ah = int(round(cw_ * ch_ / hr)), ch_
                ox, oy = (cw_ - aw) // 2, 0
            else:                                   # padded in y
                aw, ah = cw_, int(round(hr))
                ox, oy = 0, (ch_ - ah) // 2
        else:                                       # unpadded/legacy render
            aw, ah, ox, oy = cw_, ch_, 0, 0
        s = aw / w0
        x, y, bw, bh = (float(v) for v in fbb)
        rx0, ry0 = int(ox + x * s), int(oy + y * s)
        rx1, ry1 = int(ox + (x + bw) * s), int(oy + (y + bh) * s)
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(cw_, rx1), min(ch_, ry1)
        crop_r = arr[ry0:ry1, rx0:rx1]
        crop_p = an.img.gray[int(max(0, y)):int(y + bh), int(max(0, x)):int(x + bw)]
        if min(crop_r.shape[:2]) < 8 or min(crop_p.shape[:2]) < 8:
            return None
        a = cv2.resize(crop_r, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(crop_p, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Blur away glyph texture so the comparison sees tonal structure.
        a = cv2.GaussianBlur(a, (0, 0), 2.5)
        b = cv2.GaussianBlur(b, (0, 0), 2.5)
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float(np.clip((a * b).mean(), -1.0, 1.0))
    except Exception:  # noqa: BLE001
        return None
