"""In-house source-photo enhancement.

Faithful, identity-safe preprocessing that hands the typographic renderer a
cleaner tonal map from imperfect photos (old snapshots, grainy phone pictures,
small scans, faded prints). Deliberately LIGHT: it upsizes and denoises but never
reconstructs or invents facial features. No AI face restoration runs here, so a
memorial likeness can never be silently changed — the family's face stays exactly
their face. Heavier, opt-in AI restoration (CodeFormer/GFPGAN) is a separate,
user-approved path, not this stage.

Both steps are gated and best-effort: a clean, well-sized photo passes through
untouched, and any failure returns the original image.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .warnings import WarningCollector

# Photos whose longest side is under this get upscaled toward it, so the renderer
# has enough pixels to resolve the silhouette and features cleanly.
_MIN_LONG = 1100
_MAX_UPSCALE = 3.0
# Estimated Gaussian-noise sigma (0..255) above which we denoise. Gentle, clean
# photos fall below this and are left untouched (no over-smoothing).
_NOISE_SIGMA = 3.2


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise estimate: high-pass the image with a Laplacian-like
    kernel and scale the mean absolute response. Returns sigma in [0, 255]."""
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)
    return float(np.abs(conv).sum() * math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2)))


def enhance_source(bgr: np.ndarray, warns: WarningCollector) -> np.ndarray:
    """Upscale a small photo and denoise a grainy one before analysis/rendering.
    Identity-preserving (no feature reconstruction). Best-effort: returns the
    input unchanged on any error."""
    try:
        h, w = bgr.shape[:2]

        # 1. Denoise grainy photos FIRST, on native pixels where the grain is
        #    measurable and non-local-means works on the real noise (phone/film
        #    grain, scan noise). Gated on an estimate so clean photos are
        #    untouched; strength tracks the measured noise. Colour-aware so skin
        #    tones don't shift.
        sigma = _estimate_noise_sigma(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if sigma > _NOISE_SIGMA:
            strength = float(np.clip(sigma * 1.1, 3.0, 14.0))
            bgr = cv2.fastNlMeansDenoisingColored(
                bgr, None, h=strength, hColor=strength,
                templateWindowSize=7, searchWindowSize=21)
            warns.warn("enhance", "denoised",
                       "Grainy photo cleaned up for a sharper portrait.")

        # 2. Upscale small photos (old snapshots, low-res scans, phone crops) AFTER
        #    denoising, so the silhouette and features carry enough clean detail
        #    for the word-mosaic without amplifying grain.
        longest = max(h, w)
        if longest < _MIN_LONG:
            scale = min(_MAX_UPSCALE, _MIN_LONG / float(longest))
            if scale > 1.02:
                bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_CUBIC)
                warns.warn("enhance", "upscaled",
                           "Low-resolution photo upscaled for a cleaner portrait.")
        return bgr
    except Exception:  # noqa: BLE001
        return bgr
