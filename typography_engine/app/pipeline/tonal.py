"""Tonal word-fill portrait.

Reproduce the photo's light and shadow as a monospace grid of the approved
words. The subject's tone is sharpened (CLAHE + unsharp) and contrast-stretched,
then each grid cell takes the area-averaged darkness beneath it so the field is
smooth. The whole masked subject is filled with words -- even bright skin and
white hair render as faint light-gray words -- while only the background stays
blank. Every inked glyph is shaded by the
exact tone it lands on (light gray on skin midtones, near-black on hair, brows,
eyes and lips), so the assembled grid carries smooth gradients and reads as the
person's face. When facial landmarks are available the eyes, brows, lips and
nostrils are deepened so the recognition features anchor the likeness. Masked to
the silhouette so the background stays clean.

Each contiguous dark run is packed with whole words from the approved list
(cycled in order, one blank cell between words); a word is placed only when it
fits the run entirely, and runs too short to hold any word stay blank. No word
is ever cut and no stranded single letters appear.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import RenderConfig
from .svgbuild import SvgDoc, esc
from .textlayout import TextRun, normalize_words
from .warnings import WarningCollector

_MONO_FAMILY = "'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace"
_MONO_ADVANCE = 0.6  # glyph advance as a fraction of em for monospace fonts

# Legibility/texture controls. Per-word jitter (fraction of cell/row) scatters
# whole words to break the grid without wobbling letters within a word; word gap
# is the blank cells between words (clearer separation reads better).
# X jitter breaks vertical rivers without hurting reading; Y is kept low so rows
# stay on clean baselines (most legible). Word gap separates words for clarity.
# Banding is otherwise held off by the random word order.
_JITTER_X = 0.16
_JITTER_Y = 0.10
_WORD_GAP = 2

# Per-glyph gray ramp (0-255): lightest inked cells near this gray, darkest
# features near-black, so tone gradients carry the likeness. Kept just below
# white so the brightest skin/hair still render as very faint words (not blank)
# while highlights read light enough to give the portrait real contrast.
_SHADE_LIGHT = 214
_SHADE_DARK = 0

# Named ink treatments. Each duotone is (light_end, dark_end): the colour at the
# brightest tone and at the darkest. Light end is near the background so
# highlights melt into it; dark end carries the features. "photo" samples the
# source image's own colour. Background pairs with the chosen ink.
_PALETTES = {
    "mono":     ("#bebebe", "#000000", "#ffffff"),   # reference grayscale
    "navy":     ("#dbe4f1", "#08111f", "#ffffff"),
    "sepia":    ("#ead9b6", "#2a1808", "#fbf7ee"),
    "burgundy": ("#ecd2d3", "#42101a", "#ffffff"),
    "forest":   ("#d4e3d8", "#0c2618", "#ffffff"),
    "gold_noir": ("#15171c", "#e8c66a", "#101216"),  # bright ink on dark ground
}


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# Positional-gradient inks: hue runs top->bottom independent of the photo; tone
# still drives density. Stops are (vertical_fraction, colour).
_GRADIENTS = {
    "spectrum": [(0.00, "#f2b705"), (0.18, "#f25c05"), (0.38, "#e6002e"),
                 (0.58, "#b5179e"), (0.78, "#6a1fb5"), (1.00, "#1f3fb5")],
    "aurora":   [(0.00, "#10c8b8"), (0.30, "#19a6e8"), (0.55, "#3a6df0"),
                 (0.80, "#7b3ff0"), (1.00, "#a62ee0")],
}


def _grad_rgb(stops, v: float) -> Tuple[int, int, int]:
    v = 0.0 if v < 0 else (1.0 if v > 1 else v)
    for i in range(len(stops) - 1):
        v0, c0 = stops[i]
        v1, c1 = stops[i + 1]
        if v <= v1:
            t = 0.0 if v1 == v0 else (v - v0) / (v1 - v0)
            a, b = _hex_to_rgb(c0), _hex_to_rgb(c1)
            return tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))
    return _hex_to_rgb(stops[-1][1])

# Calligram looks: each entry is (ink_hex, bg_hex). After the 2026-05-30
# pivot we ship dark-ground palettes only -- the modulation pass (with
# per-glyph photo tonal gradation + feature emphasis + painted pupil)
# produces gallery-grade depth on dark backgrounds; the light-ground path
# could not match the same density without highlights melting into white.
_CALLIGRAM = {
    "gold_noir":  ("#e8c66a", "#101216"),  # warm gold on near-black
    "navy_marigold": ("#f3c34a", "#0f1a35"),  # warm yellow on deep navy
    "forest_bone":   ("#f1e8d4", "#143226"),  # cream on hunter green
    "burgundy_champagne": ("#e9d39a", "#3a0f17"),  # pale gold on oxblood
}

# MediaPipe 478-point mesh index groups for the recognition features we deepen.
_EYE_L = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_EYE_R = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
_BROW_L = (70, 63, 105, 66, 107, 46, 53, 52, 65, 55)
_BROW_R = (336, 296, 334, 293, 300, 276, 283, 282, 295, 285)
_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
_NOSE = (1, 2, 98, 327, 97, 326, 5, 4, 275, 440, 220, 45)
_FEATURE_GROUPS = (_EYE_L, _EYE_R, _BROW_L, _BROW_R, _LIPS, _NOSE)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Local-contrast (CLAHE) + unsharp mask so features keep their edges.

    A fairly strong CLAHE deepens the subtle shadows on soft, evenly-lit faces
    (eyes, nose, smile lines) so flat subjects don't render washed-out."""
    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(7, 7)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 2.4)
    return cv2.addWeighted(clahe, 1.8, blur, -0.8, 0)


def _tone_field(gray: np.ndarray, mask: np.ndarray, gamma: float, floor: float) -> np.ndarray:
    """Return per-pixel darkness in [0,1] (1 = ink), contrast-stretched within
    the subject so the full tonal range is used and bright skin drops toward 0."""
    m = mask > 127
    vals = gray[m] if int(m.sum()) > 50 else gray.reshape(-1)
    lo, hi = np.percentile(vals, [1.5, 98.5])
    if hi - lo < 1.0:
        lo, hi = float(vals.min()), float(max(vals.min() + 1.0, vals.max()))
    g = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    dark = 1.0 - g
    dark = np.clip((dark - floor) / max(1e-3, 1.0 - floor), 0.0, 1.0)
    if gamma != 1.0:
        dark = dark ** gamma
    dark[~m] = 0.0
    return dark


def _auto_tone(dark: np.ndarray, mset: np.ndarray, target: float, max_shift: float) -> np.ndarray:
    """Even out overall brightness across images by shifting the in-subject mean
    darkness toward `target` (clamped). Gentle and contrast-preserving: it tames
    unusually dark or light subjects without flattening local detail (feature
    contrast is handled by the CLAHE/unsharp step instead)."""
    vals = dark[mset]
    if vals.size < 50:
        return dark
    shift = float(np.clip(target - float(vals.mean()), -max_shift, max_shift))
    if abs(shift) < 1e-3:
        return dark
    d = np.clip(dark + shift, 0.0, 1.0)
    d[~mset] = 0.0
    return d


def _faces_of(an):
    faces = getattr(an, "faces", None)
    if faces:
        return faces
    lm = getattr(an, "landmarks", None)
    return [lm] if lm is not None else []


def _balance_faces(dark: np.ndarray, an, scale: float, mset: np.ndarray) -> np.ndarray:
    """Per-face local-contrast normalization so a pale, low-contrast face renders
    with the same ink depth as a high-contrast one.

    The global tone stretch in `_tone_field` spans the whole subject, so on a
    light face beside dark clothing the face collapses into a narrow bright band
    and washes out. Here each detected face's own darkness range is stretched and
    lifted toward a target mean, blended through a feathered oval so there is no
    seam at the jaw/neck and the rest of the portrait is untouched."""
    faces = _faces_of(an)
    if not faces:
        return dark
    H, W = dark.shape[:2]
    out = dark.copy()
    for face in faces:
        pts = face.points * scale
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        fw = float(pts[:, 0].max() - pts[:, 0].min())
        fh = float(pts[:, 1].max() - pts[:, 1].min())
        if fw < 4 or fh < 4:
            continue
        # Feathered oval over the face (a bit larger than the landmark hull).
        wm = np.zeros((H, W), np.float32)
        cv2.ellipse(wm, (int(cx), int(cy)), (int(fw * 0.72), int(fh * 0.85)),
                    0, 0, 360, 1.0, -1)
        wm = cv2.GaussianBlur(wm, (0, 0), max(2.0, fw * 0.18)) * mset
        core = wm > 0.6
        if int(core.sum()) < 50:
            continue
        vals = dark[core]
        lo, hi = np.percentile(vals, [10, 90])
        if hi - lo < 0.04:
            hi = lo + 0.04
        # Linear remap into a wide band (near full range) so each face uses most
        # of the ink scale -- light skin reads light, shadows read dark -- then
        # blend with the original to keep the smooth gradients that read as
        # detail. (A hard 0->1 stretch plus a gamma lift instead compounds with
        # the sharpen + shade S-curve and turns subtle modeling into blotches.)
        t_lo, t_hi = 0.05, 0.92
        remap = np.clip((dark - lo) / (hi - lo), 0.0, 1.0) * (t_hi - t_lo) + t_lo
        alpha = 0.65
        balanced = dark * (1.0 - alpha) + remap * alpha
        out = out * (1.0 - wm) + balanced * wm
    return np.clip(out, 0.0, 1.0)


def _eye_ellipses(an, scale: float) -> List[Tuple[float, float, float, float]]:
    """Per-eye (cx, cy, rx, ry) ellipses in render coords -- the regions the main
    grid skips and the finer eye pass fills, so eyes resolve their structure."""
    out: List[Tuple[float, float, float, float]] = []
    for face in _faces_of(an):
        pts = face.points * scale
        for grp in (_EYE_L, _EYE_R):
            ep = pts[list(grp)]
            cx, cy = float(ep[:, 0].mean()), float(ep[:, 1].mean())
            rx = (float(ep[:, 0].max() - ep[:, 0].min()) / 2.0) * 1.30
            ry = (float(ep[:, 1].max() - ep[:, 1].min()) / 2.0) * 1.55
            if rx >= 2.0 and ry >= 2.0:
                out.append((cx, cy, rx, ry))
    return out


def _emphasize_features(dark: np.ndarray, an, scale: float, mset: np.ndarray) -> np.ndarray:
    """Deepen the brows, lips and nostrils of every face so the likeness anchors
    there. Eyes are handled separately (_sharpen_eyes) -- they need local
    contrast, not the uniform darkening that flattens iris/sclera/catchlight."""
    faces = _faces_of(an)
    if not faces:
        return dark
    H, W = dark.shape[:2]
    fm = np.zeros((H, W), np.uint8)
    for face in faces:
        pts = face.points * scale
        for grp in (_BROW_L, _BROW_R, _LIPS, _NOSE):
            hull = cv2.convexHull(np.array([pts[i] for i in grp], np.int32))
            cv2.fillConvexPoly(fm, hull, 255)
    fm = cv2.dilate(fm, np.ones((5, 5), np.uint8), 1)
    w = (cv2.GaussianBlur(fm, (0, 0), 3.0).astype(np.float32) / 255.0) * mset
    return dark * (1.0 - w) + np.clip(dark ** 0.55, 0.0, 1.0) * w


def _sharpen_eyes(dark: np.ndarray, an, scale: float, mset: np.ndarray) -> np.ndarray:
    """Make each eye read as a *live* eye: strong local contrast so iris/lash go
    dark and sclera goes light, crisp lid edges (unsharp), and a preserved
    catchlight -- the small bright glint that makes a portrait look back at you.
    Strong contrast is desirable here (unlike the gentle whole-face balance)."""
    faces = _faces_of(an)
    if not faces:
        return dark
    H, W = dark.shape[:2]
    out = dark.copy()
    for face in faces:
        pts = face.points * scale
        for grp in (_EYE_L, _EYE_R):
            ep = pts[list(grp)]
            x0, y0 = float(ep[:, 0].min()), float(ep[:, 1].min())
            x1, y1 = float(ep[:, 0].max()), float(ep[:, 1].max())
            ew, eh = x1 - x0, y1 - y0
            if ew < 6 or eh < 4:
                continue
            bx0 = int(max(0, x0 - ew * 0.30)); bx1 = int(min(W, x1 + ew * 0.30))
            by0 = int(max(0, y0 - eh * 0.80)); by1 = int(min(H, y1 + eh * 0.80))
            if bx1 - bx0 < 6 or by1 - by0 < 6:
                continue
            patch = out[by0:by1, bx0:bx1]
            lo, hi = np.percentile(patch, [4, 96])
            if hi - lo < 0.05:
                continue
            st = np.clip((patch - lo) / (hi - lo), 0.0, 1.0)
            blur = cv2.GaussianBlur(st, (0, 0), max(1.0, ew * 0.04))
            sharp = np.clip(st + (st - blur) * 1.1, 0.0, 1.0)   # crisper iris/lid
            ph, pw = patch.shape[:2]
            fm = np.zeros((ph, pw), np.float32)
            cv2.ellipse(fm, (pw // 2, ph // 2), (pw // 2, ph // 2), 0, 0, 360, 1.0, -1)
            fm = cv2.GaussianBlur(fm, (0, 0), max(1.0, ew * 0.12)) * mset[by0:by1, bx0:bx1]
            out[by0:by1, bx0:bx1] = patch * (1.0 - fm) + sharp * fm

            # Catchlight: keep the brightest spot inside the eye a crisp light
            # glint (only if a real highlight exists), so eyes don't read dead.
            ix0, iy0 = int(max(0, x0)), int(max(0, y0))
            ix1, iy1 = int(min(W, x1)), int(min(H, y1))
            eye_in = out[iy0:iy1, ix0:ix1]
            if eye_in.size and float(eye_in.min()) < 0.30:
                cyl, cxl = np.unravel_index(int(np.argmin(eye_in)), eye_in.shape)
                r = max(3, int(round(eh * 0.09)))
                cv2.circle(out, (ix0 + cxl, iy0 + cyl), r, 0.0, -1)
    return np.clip(out, 0.0, 1.0)


def build_calligram(
    an,
    text: str,
    cfg: RenderConfig,
    warns: WarningCollector,
    render_w: int = 2600,
    font_px: float = 22.0,
    contrast: float = 3.5,
    pivot: float = 0.45,
    power: float = 0.65,
    ink_hex: str = "#15202b",
    bg_hex: str = "#ffffff",
    subject_only: bool = False,
) -> Tuple[str, List[TextRun]]:
    """Story calligram: lay the user's passage as continuous prose, multi-size,
    so the face emerges from text density. Small font in detail areas (eyes,
    brows, lip line), medium on cheek/forehead, large on hair / body / edges.
    Brightness modulated by photo tone -- bright letters on lit skin, dim
    letters in shadow / outside the subject -- yields the high-contrast Margot
    look on dark-ground inks (gold_noir, etc).

    When `subject_only=True`, words are placed only on cells whose centre falls
    inside the silhouette -- the background tiers stay blank so the subject
    reads as a portrait on a clean ground, not a full-canvas type field."""
    words = [w for w in str(text).split() if w]
    if not words:
        warns.error("text", "no_words", "No passage supplied for the calligram.")
        return "", []

    gray = an.img.gray
    mask = an.silhouette.mask
    h0, w0 = gray.shape[:2]
    if mask.shape[:2] != (h0, w0):
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    if w0 < render_w:
        scale = render_w / float(w0)
        W, H = int(round(w0 * scale)), int(round(h0 * scale))
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        scale, W, H = 1.0, w0, h0

    mset = mask > 127
    dark = _tone_field(_sharpen(gray), mask, gamma=1.0, floor=0.0)
    dark = _auto_tone(dark, mset, 0.50, max_shift=0.18)
    dark = _balance_faces(dark, an, scale, mset)
    dark = _emphasize_features(dark, an, scale, mset)
    dark = _sharpen_eyes(dark, an, scale, mset)
    # Cast back to float32 -- the helpers above can promote to float64 via
    # numpy ops, and cv2.GaussianBlur rejects float64 inputs.
    dark = dark.astype(np.float32)

    # Local tonal-detail signal -- secondary input to font sizing.
    gx = cv2.Sobel(dark, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dark, cv2.CV_32F, 0, 1, ksize=5)
    detail = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    detail = cv2.GaussianBlur(detail, (0, 0), max(3.0, W * 0.008))
    if mset.sum() > 50:
        lo, hi = np.percentile(detail[mset], [10, 95])
    else:
        lo, hi = float(detail.min()), float(max(detail.min() + 1e-3, detail.max()))
    detail = np.clip((detail - lo) / max(1e-3, hi - lo), 0.0, 1.0)
    detail[~mset] = 0.0

    # Letter size map: four structural zones, smoothly feathered so size
    # transitions are gradual not stepped.
    #   FACE FEATURES (eyes/brows/lips/nose) -> 0.00 ... 0.16  (tiniest)
    #   FACE HULL (rest)                     -> 0.16 ... 0.36  (small)
    #   BODY (inside silhouette, off face)   -> 0.36 ... 0.66  (medium)
    #   BACKGROUND                            -> 0.66 ... 1.00  (largest)
    size_signal = np.zeros((H, W), np.float32)
    face_hull_mask = np.zeros((H, W), np.uint8)
    feature_mask = np.zeros((H, W), np.uint8)
    eye_mask = np.zeros((H, W), np.uint8)   # eyes only, for sharper contrast
    eye_centers = []                         # list of (cx, cy, rx, ry) per eye
    have_face = False
    for face in _faces_of(an):
        have_face = True
        pts = (face.points * scale).astype(np.int32)
        # Face hull (encompasses the whole face area).
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(face_hull_mask, hull, 255)
        # Per-feature convex hulls (eyes / brows / lips / nose) for the very
        # finest tier so eye sclera / iris / lashes / brow line / nostrils /
        # lip corners get the smallest possible letters.
        for grp in _FEATURE_GROUPS:
            try:
                fhull = cv2.convexHull(np.array([pts[i] for i in grp], np.int32))
                cv2.fillConvexPoly(feature_mask, fhull, 255)
            except Exception:
                pass
        # Eye-only mask + per-eye centre/extent, used for pupil/catchlight pass.
        for grp in (_EYE_L, _EYE_R):
            try:
                ep = pts[list(grp)]
                ehull = cv2.convexHull(np.array([pts[i] for i in grp], np.int32))
                cv2.fillConvexPoly(eye_mask, ehull, 255)
                cx, cy = float(ep[:, 0].mean()), float(ep[:, 1].mean())
                rx = (float(ep[:, 0].max() - ep[:, 0].min()) / 2.0)
                ry = (float(ep[:, 1].max() - ep[:, 1].min()) / 2.0)
                eye_centers.append((cx, cy, rx, ry))
            except Exception:
                pass
        feature_mask = cv2.dilate(feature_mask, np.ones((3, 3), np.uint8))
    if have_face:
        # Use feathered distance transforms so each zone fades smoothly into
        # the next -- no harsh tier boundaries.
        # Distance into feature mask (0 inside, growing outward).
        feat_d = cv2.distanceTransform(255 - feature_mask, cv2.DIST_L2, 5)
        feat_d = np.clip(feat_d / max(8.0, W * 0.005), 0.0, 1.0)
        # Distance into face hull (0 inside face hull, 1 far away). WIDER
        # normalization so the transition from face-hull-size to body-size
        # spans a generous band rather than stepping right at the hairline.
        hull_d = cv2.distanceTransform(255 - face_hull_mask, cv2.DIST_L2, 5)
        hull_d = np.clip(hull_d / max(40.0, W * 0.045), 0.0, 1.0)
        # Distance into silhouette (0 inside silhouette, 1 far in bg).
        sil_d = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        sil_d = np.clip(sil_d / max(40.0, W * 0.04), 0.0, 1.0)
        # Compose: each zone contributes a range of the signal; smooth
        # interpolation between them via the three distance fields.
        # In feature: signal = 0.00 - 0.16 (slight variation by position).
        # In hull but not feature: signal = 0.16 + 0.20 * feat_d.
        # In silhouette but not hull: signal = 0.36 + 0.64 * hull_d. Body /
        #   hair / clothing now spans md(18) -> xl(36) so hair gets visibly
        #   larger letters as it moves away from the face, per 2026-05-30
        #   feedback that body tiers were too uniform with only md/md+.
        # In background: signal stays high but subject_only stops placement.
        size_signal = np.where(
            feature_mask > 0,
            0.16 * (1.0 - feat_d),  # 0 at feature centre, 0.16 at feature edge
            np.where(
                face_hull_mask > 0,
                0.16 + 0.20 * feat_d,
                np.where(
                    mset,
                    0.36 + 0.64 * hull_d,
                    0.66 + 0.34 * sil_d,
                ),
            ),
        ).astype(np.float32)
    else:
        size_signal = np.full((H, W), 0.45, dtype=np.float32)
    size_signal = np.clip(size_signal, 0.0, 1.0)

    ir, ig, ib = _hex_to_rgb(ink_hex)
    br, bgc, bb = _hex_to_rgb(bg_hex)
    family = "'Courier New', 'DejaVu Sans Mono', 'Liberation Mono', monospace"
    bg_luma = 0.299 * br + 0.587 * bgc + 0.114 * bb
    dark_bg = bg_luma < 100
    dim_floor = 0.10
    # Per-cell contrast curve overrides for light_bg. The default 3.5/0.45/0.65
    # were tuned for the dark_bg modulation path and clip too aggressively for
    # direct per-cell tspan rendering (mid-tones cluster near full ink,
    # highlights clip to bg). The historical 71d531b curve (2.3/0.5/1.0)
    # gives a broader value spread; an ink floor below ensures highlights
    # stay visibly inked rather than melting into the white ground.
    if not dark_bg:
        contrast, pivot, power = 2.3, 0.5, 1.0
    light_bg_ink_floor = 0.12   # min visible ink (12%) on light_bg highlights

    doc = SvgDoc(width=W, height=H, background=bg_hex)
    runs: List[TextRun] = []

    # FIVE tiers, indexed by `size_signal`. Each zone (face/body/bg) spans
    # ~1/3 of the signal so all tiers are reachable.
    # Tier (label, font_px, size_low, size_high)
    # EIGHT tiers for very smooth size gradation. Smallest (6px) for the
    # innermost feature centres, largest (36px) for far background.
    tiers = [
        ("xl",   36.0, 0.86, 1.01),   # deep background
        ("lg",   28.0, 0.72, 0.86),   # silhouette edge / background near subject
        ("md+",  22.0, 0.58, 0.72),   # body / hair body
        ("md",   18.0, 0.46, 0.58),   # shoulders / hair near face
        ("sm",   14.0, 0.34, 0.46),   # silhouette-near-face / outer hair
        ("xs+",  11.0, 0.22, 0.34),   # face hull / cheek / forehead
        ("xs",    9.0, 0.10, 0.22),   # feature edges (lid line, lip line)
        ("xxs",   6.0, 0.00, 0.10),   # eye sclera/iris, lash line, lip corners
    ]
    # claim grid at the finest tier's resolution; once claimed by a finer tier,
    # coarser tiers skip those cells.
    fine_cw = tiers[-1][1] * _MONO_ADVANCE
    fine_rh = tiers[-1][1] * 1.05
    claim_cols = max(1, int(W / fine_cw))
    claim_rows = max(1, int(H / fine_rh))
    claimed = np.zeros((claim_rows, claim_cols), dtype=bool)

    wi = 0  # advances through the words list across all tiers

    def _color_for(yi: int, xi: int, p: float) -> str:
        """Pick the ink colour for a letter at (yi, xi). For dark-ground inks
        we render all letters PURE WHITE in the SVG and add the photo's tonal
        gradient via the per-pixel modulation pass below -- that gives the
        Margot-style internal-glyph tonal variation. Light-ground inks keep
        the per-cell colour modulation."""
        if dark_bg:
            return "#ffffff"
        tone = float(tone_s[yi, xi])
        norm = (tone - pivot) * contrast + pivot
        norm = 1.0 if norm > 1.0 else (0.0 if norm < 0.0 else norm)
        f = 1.0 - norm ** power
        # Ink floor: cap `f` so the brightest in-silhouette letters still
        # carry visible ink rather than vanishing into the white ground.
        # (Outside the silhouette, subject_only stops placement entirely.)
        f_max = 1.0 - light_bg_ink_floor
        if f > f_max:
            f = f_max
        cr = int(round(ir + (br - ir) * f))
        cg = int(round(ig + (bgc - ig) * f))
        cb = int(round(ib + (bb - ib) * f))
        return f"#{cr:02x}{cg:02x}{cb:02x}"

    # Render tiers COARSE -> MEDIUM -> FINE. Each tier claims the pixel
    # footprint of each LETTER it draws (not whole-word bounds), so finer
    # tiers fill the SPACES between larger letters / words without overlapping
    # the large letters themselves. This gives Margot-style clean tier
    # transitions: smaller text fits BETWEEN bigger text with no visual smear.
    claimed_px = np.zeros((H, W), dtype=bool)
    for label, fp, d_lo, d_hi in tiers:
        cw = fp * _MONO_ADVANCE
        # Smoother row-spacing gradient so face-to-hair transition isn't a
        # density step change. Smallest tiers get 0.90x, middle tiers 0.95x,
        # largest tiers 1.0x -- a gentle slope instead of the prior 0.85
        # vs 1.05 jump that made forehead-vs-hair look like two different
        # textures collide.
        if fp <= 11.0:
            row_ratio = 0.90
        elif fp <= 18.0:
            row_ratio = 0.95
        else:
            row_ratio = 1.00
        rh = fp * row_ratio
        cols = max(1, int(W / cw))
        rows = max(1, int(H / rh))
        tone_s = cv2.GaussianBlur(dark, (0, 0), max(1.0, fp * 0.5))
        for r in range(rows):
            baseline = (r + 1) * rh
            yi = min(H - 1, max(0, int(baseline - 0.32 * fp)))
            y_lo = max(0, int(baseline - rh))
            y_hi = min(H, int(baseline) + 1)
            spans = []
            line_chars = []
            c = 0
            while c < cols:
                word = words[wi % len(words)]
                wl = len(word)
                if wl > cols:
                    word = word[:cols]; wl = cols
                if c + wl > cols:
                    break
                # Try to place the word starting at column c. Check the start
                # cell's detail bucket; if it doesn't match the tier, skip a
                # cell and try again.
                xi_start = min(W - 1, max(0, int(c * cw + cw * 0.5)))
                # Subject-only mode: skip cells whose centre falls outside the
                # silhouette so the background tiers never paint.
                if subject_only and not mset[yi, xi_start]:
                    c += 1
                    continue
                s = float(size_signal[yi, xi_start])
                if not (d_lo <= s < d_hi):
                    c += 1
                    continue
                # Check that no letter of this word would land on a pixel
                # already claimed by an earlier (larger) tier. If any would,
                # advance one cell and retry; this lets the word slide right
                # until it finds a clear stretch.
                wx_lo = max(0, int(c * cw))
                wx_hi = min(W, int((c + wl) * cw) + 1)
                if claimed_px[y_lo:y_hi, wx_lo:wx_hi].any():
                    c += 1
                    continue
                # Place the word. Per-glyph fill: photo tone runs through each
                # letter via the modulation pass.
                for k, ch in enumerate(word):
                    col = c + k
                    xi = min(W - 1, max(0, int(col * cw + cw * 0.5)))
                    fill = _color_for(yi, xi, 0.55)
                    spans.append(
                        f'<tspan x="{col * cw:.1f}" fill="{fill}">{esc(ch)}</tspan>'
                    )
                    line_chars.append(ch)
                # Claim this word's pixel footprint so finer tiers don't paint
                # over its letters (they're free to fill the gaps after it).
                claimed_px[y_lo:y_hi, wx_lo:wx_hi] = True
                # Tight packing: NO inter-word space on the finest tiers (so
                # face features get maximum letter density), 1-cell gap on
                # larger tiers (keeps words readable in the body / bg).
                gap = 0 if fp <= 14.0 else 1
                c += wl + gap
                wi += 1
            if not spans:
                continue
            doc.add(
                f'<text y="{baseline:.1f}" xml:space="preserve" font-family="{esc(family)}" '
                f'font-size="{fp:.1f}">' + "".join(spans) + "</text>"
            )
            runs.append(TextRun(region=f"calligram_{label}", path_id=f"{label}_r{r}",
                                path_d="", text="".join(line_chars),
                                font_size=round(fp, 1), kind="primary"))

    # Per-glyph tonal modulation: render the SVG normally with WHITE text on
    # solid bg, then use numpy to mask the photo through the rendered letter
    # shapes. Each letter ends up filled with the actual photo tones beneath
    # its glyph -- the "lip-contour-inside-each-letter" effect from the
    # Margot reference. We pass the modulation image back to the caller; if
    # present it overrides the standard cairosvg-only path.
    modulation_png = None  # type: Optional[bytes]
    # Modulation pass runs ONLY on dark_bg. On light_bg the per-cell tspan
    # tonal colours already encode the photo tone per letter (the 71d531b
    # approach Jeff approved), and the modern 8-tier hierarchy + feature
    # emphasis in the tone field give graduated typography with proper
    # value contrast. Running modulation on top of that washes the per-cell
    # colours into a flat field; skipping it keeps the dimensional look.
    if dark_bg:
        # Run modulation for dark grounds. The per-glyph photo tonal
        # gradation is what gives the typography dimension on gold_noir.
        from .raster import svg_to_png_bytes
        import io as _io2
        import re as _re2
        from PIL import Image as _PILImage2
        # For alpha extraction we need uniform letter contrast against the
        # ground. On dark_bg the SVG already has white letters on a dark
        # ground (good alpha as-is). On light_bg the letters are per-cell
        # coloured -- swap every tspan fill to black for the alpha render so
        # the alpha mask reflects letter SHAPE only, not pre-baked cell tone
        # (otherwise we'd double-modulate when photo_fill is applied below).
        svg_text = doc.to_svg()
        if dark_bg:
            alpha_svg = svg_text
        else:
            alpha_svg = _re2.sub(
                r'(<tspan\b[^>]*?)fill="#[0-9a-fA-F]{6}"',
                r'\1fill="#000000"',
                svg_text,
            )
        try:
            png_bytes_raw = svg_to_png_bytes(alpha_svg, output_width=W)
        except Exception:
            png_bytes_raw = None
        if png_bytes_raw:
            text_img = _PILImage2.open(_io2.BytesIO(png_bytes_raw)).convert("RGB")
            text_arr = np.asarray(text_img).astype(np.float32) / 255.0  # H,W,3
            # Recompute the high-contrast tone-mapped photo at render resolution.
            t = dark.copy()
            if mset.sum() > 50:
                plo, phi = np.percentile(t[mset], [3, 97])
            else:
                plo, phi = float(t.min()), float(max(t.min() + 1e-3, t.max()))
            t = np.clip((t - plo) / max(1e-3, phi - plo), 0.0, 1.0)
            brightness = (1.0 - t).astype(np.float32)
            # FEATURE CONTRAST BOOST: inside eye/brow/lip/nose hulls, push
            # darks darker and brights brighter -- WITH a 0.10 floor so the
            # darkest feature pixels (iris, pupil, eyebrow) stay visible as
            # dim letters rather than collapsing into pure black.
            if have_face:
                fm = (feature_mask > 0).astype(np.float32)
                fm = cv2.GaussianBlur(fm, (0, 0), max(2.0, W * 0.003))
                # Feature S-curve: on dark_bg keep a 0.10 brightness floor
                # so dim features stay visible against the dark ground; on
                # light_bg drop the floor so brows / iris / lips can sink to
                # full ink (brightness 0 -> ink_amount near 1.0).
                if dark_bg:
                    boosted = np.clip(((brightness - 0.5) * 1.6) + 0.5, 0.10, 1.0)
                else:
                    boosted = np.clip(((brightness - 0.5) * 2.0) + 0.5, 0.0, 1.0)
                brightness = brightness * (1.0 - fm) + boosted * fm

                # EYES PASS: sharper contrast specifically in eye hulls --
                # pupil drops near-black, sclera goes near-white.
                em = (eye_mask > 0).astype(np.float32)
                em = cv2.GaussianBlur(em, (0, 0), max(1.5, W * 0.0025))
                if dark_bg:
                    eye_boost = np.clip(((brightness - 0.5) * 2.6) + 0.5, 0.04, 1.0)
                else:
                    eye_boost = np.clip(((brightness - 0.5) * 3.0) + 0.5, 0.0, 1.0)
                brightness = brightness * (1.0 - em) + eye_boost * em
                # On dark_bg only: force darkest spot per eye into the
                # brightness map (so letters at the pupil render as bg-black
                # and a catchlight spot reads as full bright ink). This trick
                # only works for the dark_bg direction where brightness=0
                # maps to bg and brightness=1 maps to full ink. On light_bg
                # the same forcing inverts the meaning. Per 2026-05-30
                # feedback: pupil/catchlight should emerge from typography
                # carrying the natural photo tonal values, not from forced
                # uniform extremes. _sharpen_eyes (in the `dark` field)
                # provides local contrast boost; we let that flow through
                # the modulation pass naturally.

            # Smooth interior-silhouette transitions so the letter colour
            # gradient flows without a stark edge between bright skin and
            # dark hair. Small sigma so feature detail is kept.
            brightness = cv2.GaussianBlur(brightness, (0, 0), max(1.5, W * 0.0025))
            # Map brightness -> ink_amount in [0,1]: how much ink shows at
            # each pixel (0 = full bg, 1 = full ink). Direction-specific
            # curves keep mid-tones clearly visible regardless of which way
            # the ink runs:
            #   dark_bg  : BRIGHT photo -> high ink_amount (visible gold).
            #              Floor 0.22 so even shadows have a hint of ink.
            #   light_bg : DARK photo  -> high ink_amount (visible navy).
            #              Floor 0.30 so mid-skin still reads as solidly
            #              inked rather than melting wispy into the white
            #              ground; ceiling 0.95 so the darkest shadow isn't
            #              a solid ink blob.
            if dark_bg:
                ink_amount_face = 0.22 + 0.78 * (brightness ** 0.55)
                ink_amount_outside = 0.12   # quiet bg letters, face dominates
            else:
                # Light_bg needs more density than dark_bg -- on white,
                # mid-skin and hair must read as solidly inked navy, not
                # wispy gray, for the face to have any value contrast.
                # Floor 0.40 so even highlights are clearly inked; curve
                # exponent 0.55 (vs the 0.70 of the earlier attempt) skews
                # the bulk of the tone range toward more ink.
                ink_amount_face = 0.40 + 0.60 * ((1.0 - brightness) ** 0.55)
                ink_amount_outside = 0.0    # bg letters fully melt into white
            outside = np.full_like(brightness, ink_amount_outside)
            # Wide silhouette feather so face-to-bg transitions over ~60 px.
            mset_f = mset.astype(np.float32)
            mset_f = cv2.GaussianBlur(mset_f, (0, 0), max(14.0, W * 0.028))
            mset_f = np.clip(mset_f, 0.0, 1.0)
            ink_amount = ink_amount_face * mset_f + outside * (1.0 - mset_f)
            # Unified photo_fill: bg + ink_amount * (ink - bg). Same formula
            # in both directions -- the per-direction work is encoded in how
            # ink_amount maps from brightness above.
            ink_rgb_arr = np.array([ir, ig, ib], dtype=np.float32)
            bg_rgb_arr = np.array([br, bgc, bb], dtype=np.float32)
            photo_fill = (
                bg_rgb_arr * (1.0 - ink_amount[..., None])
                + ink_rgb_arr * ink_amount[..., None]
            ).astype(np.float32) / 255.0
            # Resize photo_fill to match the rendered text image.
            if photo_fill.shape[:2] != text_arr.shape[:2]:
                photo_fill_img = _PILImage2.fromarray((photo_fill * 255).astype(np.uint8))
                photo_fill_img = photo_fill_img.resize(
                    (text_arr.shape[1], text_arr.shape[0]), _PILImage2.LANCZOS
                )
                photo_fill = np.asarray(photo_fill_img).astype(np.float32) / 255.0
            # Alpha mask from the rendered text image:
            #   dark_bg:  white-on-dark -> alpha = max channel.
            #   light_bg: black-on-white (after fill swap) -> alpha = 1 - min.
            if dark_bg:
                alpha = text_arr.max(axis=2, keepdims=True)
            else:
                alpha = np.clip(1.0 - text_arr.min(axis=2, keepdims=True), 0.0, 1.0)
            bg_rgb = np.array([br, bgc, bb], dtype=np.float32) / 255.0
            modulated = photo_fill * alpha + bg_rgb * (1.0 - alpha)
            modulated_u8 = (modulated * 255).clip(0, 255).astype(np.uint8)
            # Eyes are NOT painted as solid shapes. Pupils and catchlights
            # emerge from the typography itself -- the _sharpen_eyes pass
            # increases local contrast in the eye region so iris/pupil land
            # at dense ink and sclera/catchlight land at near-bg; the
            # modulation pass then renders those values through whatever
            # glyphs occupy the eye area. This keeps the look "made of
            # words" instead of stamped circles on top of words.
            out_img = _PILImage2.fromarray(modulated_u8)
            buf2 = _io2.BytesIO()
            out_img.save(buf2, format="PNG", optimize=True)
            modulation_png = buf2.getvalue()

    if not runs:
        warns.error("text", "no_runs", "Calligram produced no text (mask empty?).")
    return doc.to_svg(), runs, modulation_png


def build_tonal_portrait(
    an,
    words: Sequence[str],
    cfg: RenderConfig,
    warns: WarningCollector,
    uppercase: bool = True,
    render_w: int = 2600,
    gamma: float = 1.0,
    floor: float = 0.0,
    level: float = 0.015,
    power: float = 1.0,
    auto_tone: bool = True,
    target_tone: float = 0.50,
    jitter: float = 0.7,
    seed: int = 1234,
    contrast: float = 2.4,
    pivot: float = 0.46,
    ink: str = "mono",
) -> Tuple[str, List[TextRun]]:
    approved = normalize_words(words, uppercase)
    if not approved:
        warns.error("text", "no_words", "No approved words supplied; cannot place typography.")
        return "", []

    tokens = [t for t in (re.sub(r"\s+", "", w) for w in approved) if t]
    if not tokens:
        warns.error("text", "no_words", "Approved words contained no letters.")
        return "", []
    shortest = min(len(t) for t in tokens)
    ntok = len(tokens)

    gray = an.img.gray
    mask = an.silhouette.mask
    h0, w0 = gray.shape[:2]
    if mask.shape[:2] != (h0, w0):
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

    # Upsample so the grid is fine (more cells) while glyphs stay >= min_font.
    if w0 < render_w:
        scale = render_w / float(w0)
        W = int(round(w0 * scale))
        H = int(round(h0 * scale))
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        scale = 1.0
        W, H = w0, h0

    mset = mask > 127
    dark = _tone_field(_sharpen(gray), mask, gamma=gamma, floor=floor)
    if auto_tone:
        dark = _auto_tone(dark, mset, target_tone, max_shift=0.18)
    dark = _balance_faces(dark, an, scale, mset)
    dark = _emphasize_features(dark, an, scale, mset)
    dark = _sharpen_eyes(dark, an, scale, mset)

    # Photo-underlay mode uses a larger font so individual words are clearly
    # readable on top of the photograph (otherwise the grid stipples into
    # texture). Standard rendering keeps the fine-grained tonal grid.
    photo_ink = ink == "photo"
    if photo_ink:
        font = max(cfg.min_font_px * 1.7, 34.0)
    else:
        font = min(cfg.max_font_px, max(cfg.min_font_px, cfg.min_font_px))
    cell_w = font * _MONO_ADVANCE
    row_h = font * 0.80
    cols = max(1, int(W / cell_w))
    rows = max(1, int(H / row_h))

    # Area-average the tone into the glyph grid so each letter reflects the mean
    # darkness of its whole cell -> smooth, faithful gradients (not point noise).
    grid = cv2.resize(dark, (cols, rows), interpolation=cv2.INTER_AREA)
    # Coarse silhouette mask at the glyph grid resolution. In photo-underlay
    # mode we fill every silhouette cell with text (the photo carries the
    # tonal signal), so hair / bright skin still receive words -- not just the
    # dark zones the global `level` threshold lets through.
    sil_grid = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA) > 96

    # Ink treatment: grayscale (mono), a named duotone, or colour sampled from
    # the source photo. Mono keeps the existing gray ramp untouched.
    # (photo_ink set above so we could size the grid for photo mode.)
    grad = _GRADIENTS.get(ink)
    duo = _PALETTES[ink][:2] if (ink in _PALETTES and ink != "mono") else None
    bg = _PALETTES[ink][2] if (ink in _PALETTES and ink != "mono") else cfg.background_hex
    color_grid = (
        cv2.resize(an.img.bgr, (cols, rows), interpolation=cv2.INTER_AREA)
        if photo_ink else None
    )
    lo_rgb, hi_rgb = (_hex_to_rgb(duo[0]), _hex_to_rgb(duo[1])) if duo is not None else (None, None)

    span = max(1, _SHADE_LIGHT - _SHADE_DARK)
    inv_level = max(1e-3, 1.0 - level)

    def tdark_of(tone: float) -> float:
        n = (tone - level) / inv_level
        n = (n - pivot) * contrast + pivot
        n = 1.0 if n > 1.0 else (0.0 if n < 0.0 else n)
        return n ** power

    def fill_for(t_dark: float, src=None, vfrac: float = 0.0) -> str:
        g = _SHADE_LIGHT - int(round(span * t_dark))
        g = 0 if g < 0 else (255 if g > 255 else g)
        if grad is not None:
            # Hue from vertical position; blend from white (faint highlight) to
            # the full hue (saturated shadow) by tone, so features read crisp.
            hr, hg, hb = _grad_rgb(grad, vfrac)
            cr = int(round(255 + (hr - 255) * t_dark))
            cg = int(round(255 + (hg - 255) * t_dark))
            cb = int(round(255 + (hb - 255) * t_dark))
            return f"#{cr:02x}{cg:02x}{cb:02x}"
        if photo_ink:
            # Strong dark ink over the photo so the typography is unambiguously
            # legible against skin / hair highlights. The tonal modulation goes
            # mid-grey -> near-black so features still anchor the gradient,
            # but even the lightest cell stays dark enough to read.
            lo_p, hi_p = (80, 84, 96), (0, 0, 0)
            cr = int(round(lo_p[0] + (hi_p[0] - lo_p[0]) * t_dark))
            cg = int(round(lo_p[1] + (hi_p[1] - lo_p[1]) * t_dark))
            cb = int(round(lo_p[2] + (hi_p[2] - lo_p[2]) * t_dark))
            return f"#{cr:02x}{cg:02x}{cb:02x}"
        if duo is not None:
            cr = int(round(lo_rgb[0] + (hi_rgb[0] - lo_rgb[0]) * t_dark))
            cg = int(round(lo_rgb[1] + (hi_rgb[1] - lo_rgb[1]) * t_dark))
            cb = int(round(lo_rgb[2] + (hi_rgb[2] - lo_rgb[2]) * t_dark))
            return f"#{cr:02x}{cg:02x}{cb:02x}"
        return f"#{g:02x}{g:02x}{g:02x}"

    # Eyes get a finer pass (below) for crisp iris/lid/catchlight; the main grid
    # skips these regions so the two don't overprint.
    eyes = _eye_ellipses(an, scale)

    def in_eyes(px: float, py: float) -> bool:
        for ex, ey, rx, ry in eyes:
            if ((px - ex) / rx) ** 2 + ((py - ey) / ry) ** 2 <= 1.0:
                return True
        return False

    doc = SvgDoc(width=W, height=H, background=bg)
    # Photo-underlay mode: when ink="photo", embed the source image as the
    # canvas background -- silhouette-masked so the background drops to white
    # and only the subject shows under the typography. This produces the
    # "exceptional likeness" look: the photo carries the recognition, the
    # words carry the personalization, and the silhouette stays clean.
    if photo_ink:
        import base64
        import io as _io
        from PIL import Image as _PILImage
        # Composite source-photo over white at native resolution using the
        # ORIGINAL silhouette mask, then resize. Doing the masking at native
        # size keeps the soft mask feather sharp and avoids upscaling artefacts.
        src_bgr = an.img.bgr
        orig_mask = an.silhouette.mask
        if orig_mask.shape[:2] != src_bgr.shape[:2]:
            orig_mask = cv2.resize(orig_mask, (src_bgr.shape[1], src_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        m = orig_mask.astype(np.float32) / 255.0
        m = cv2.GaussianBlur(m, (0, 0), max(1.5, src_bgr.shape[1] * 0.004))
        m3 = np.dstack([m, m, m])
        white = np.full_like(rgb, 255, dtype=np.uint8)
        composed = (rgb.astype(np.float32) * m3 + white.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
        pil = _PILImage.fromarray(composed).resize((W, H), _PILImage.LANCZOS)
        buf = _io.BytesIO()
        pil.save(buf, format="JPEG", quality=82, optimize=True)
        doc.bg_image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        doc.bg_image_mime = "image/jpeg"
    runs: List[TextRun] = []
    # Seeded (reproducible) per-row jitter offsets break up the rigid column grid
    # so words don't form vertical "rivers" or horizontal banding.
    rng = np.random.default_rng(seed)

    for r in range(rows):
        ox = (rng.random() - 0.5) * cell_w * jitter
        oy = (rng.random() - 0.5) * row_h * jitter * 0.5
        baseline = (r + 0.5) * row_h + font * 0.34 + oy
        row = grid[r]
        # Photo-underlay: fill every silhouette cell. Otherwise: only cells
        # darker than `level` -- standard tonal-density behaviour.
        ink = sil_grid[r] if photo_ink else (row > level)
        if eyes:
            cy_nom = (r + 0.5) * row_h
            for c in range(cols):
                if ink[c] and in_eyes((c + 0.5) * cell_w, cy_nom):
                    ink[c] = False
        c = 0
        while c < cols:
            if not ink[c]:
                c += 1
                continue
            start = c
            while c < cols and ink[c]:
                c += 1
            end = c
            pos = start
            glyphs: List[str] = []
            first = True
            while True:
                need = 0 if first else _WORD_GAP  # blank cells between words
                avail = end - pos - need
                if avail < shortest:
                    break
                # Random word order (seeded) so word-boundary spaces fall at
                # different x on each row instead of stacking into vertical
                # "rivers"; pick the first that fits the remaining run.
                chosen = None
                for j in rng.permutation(ntok):
                    if len(tokens[j]) <= avail:
                        chosen = tokens[j]
                        break
                if chosen is None:
                    break
                if not first:
                    glyphs.extend([" "] * _WORD_GAP)
                    pos += _WORD_GAP
                glyphs.extend(chosen)
                pos += len(chosen)
                first = False
            if not glyphs:
                continue
            spans = []
            # Per-WORD jitter (not per-glyph): every letter in a word shares one
            # offset, so within a word letters stay aligned and evenly spaced
            # (legible), while whole words scatter enough to break the rigid grid
            # / banding. Each new word (after a space) gets a fresh offset.
            wx = (rng.random() - 0.5) * cell_w * _JITTER_X
            wy = (rng.random() - 0.5) * row_h * _JITTER_Y
            prev_space = False
            for k, ch in enumerate(glyphs):
                if ch == " ":
                    if not prev_space:
                        wx = (rng.random() - 0.5) * cell_w * _JITTER_X
                        wy = (rng.random() - 0.5) * row_h * _JITTER_Y
                    prev_space = True
                    continue
                prev_space = False
                cell = start + k
                t_dark = tdark_of(row[cell])
                gx = cell * cell_w + ox + wx
                gy = baseline + wy
                fill = fill_for(t_dark, color_grid[r, cell] if photo_ink else None, gy / H)
                spans.append(
                    f'<tspan x="{gx:.1f}" y="{gy:.1f}" fill="{fill}">'
                    f"{esc(ch)}</tspan>"
                )
            if not spans:
                continue
            doc.add(
                f'<text xml:space="preserve" '
                f'font-family="{esc(_MONO_FAMILY)}" font-size="{font:.2f}" '
                f'font-weight="{esc(cfg.font_weight)}">' + "".join(spans) + "</text>"
            )
            runs.append(
                TextRun(
                    region="tonal",
                    path_id=f"row{r}_{start}",
                    path_d="",
                    text="".join(glyphs),
                    font_size=round(font, 2),
                    kind="primary",
                )
            )

    # ---- Finer eye pass: resolve iris / lid / catchlight inside the eye
    # ellipses (which the main grid skipped). Half-size glyphs, marked "detail"
    # so they're exempt from the readable min-font floor that governs the body.
    if eyes:
        fe = max(6.0, font * 0.5)
        ecw, erh = fe * _MONO_ADVANCE, fe * 0.80
        ex0 = max(0, int(min(e[0] - e[2] for e in eyes)))
        ey0 = max(0, int(min(e[1] - e[3] for e in eyes)))
        ex1 = min(W, int(max(e[0] + e[2] for e in eyes)) + 1)
        ey1 = min(H, int(max(e[1] + e[3] for e in eyes)) + 1)
        if ex1 > ex0 + 2 and ey1 > ey0 + 2:
            cols_f = max(1, int((ex1 - ex0) / ecw))
            rows_f = max(1, int((ey1 - ey0) / erh))
            sub = cv2.resize(dark[ey0:ey1, ex0:ex1], (cols_f, rows_f), interpolation=cv2.INTER_AREA)
            msub = cv2.resize(mset.astype(np.uint8)[ey0:ey1, ex0:ex1], (cols_f, rows_f),
                              interpolation=cv2.INTER_NEAREST) > 0
            csub = (cv2.resize(an.img.bgr[ey0:ey1, ex0:ex1], (cols_f, rows_f),
                               interpolation=cv2.INTER_AREA) if photo_ink else None)
            for rf in range(rows_f):
                cyf = ey0 + (rf + 0.5) * erh
                baseline = cyf + fe * 0.34
                rowf = sub[rf]
                inkf = (rowf > level) & msub[rf]
                for cf in range(cols_f):
                    if inkf[cf] and not in_eyes(ex0 + (cf + 0.5) * ecw, cyf):
                        inkf[cf] = False
                c = 0
                while c < cols_f:
                    if not inkf[c]:
                        c += 1
                        continue
                    start = c
                    while c < cols_f and inkf[c]:
                        c += 1
                    end = c
                    pos, glyphs, first = start, [], True
                    while True:
                        avail = end - pos - (0 if first else _WORD_GAP)
                        if avail < shortest:
                            break
                        chosen = None
                        for j in rng.permutation(ntok):
                            if len(tokens[j]) <= avail:
                                chosen = tokens[j]
                                break
                        if chosen is None:
                            break
                        if not first:
                            glyphs.extend([" "] * _WORD_GAP)
                            pos += _WORD_GAP
                        glyphs.extend(chosen)
                        pos += len(chosen)
                        first = False
                    if not glyphs:
                        continue
                    spans = []
                    wx = (rng.random() - 0.5) * ecw * _JITTER_X
                    wy = (rng.random() - 0.5) * erh * _JITTER_Y
                    prev_space = False
                    for k, ch in enumerate(glyphs):
                        if ch == " ":
                            if not prev_space:
                                wx = (rng.random() - 0.5) * ecw * _JITTER_X
                                wy = (rng.random() - 0.5) * erh * _JITTER_Y
                            prev_space = True
                            continue
                        prev_space = False
                        cellf = start + k
                        gx = ex0 + cellf * ecw + wx
                        gy = baseline + wy
                        fill = fill_for(tdark_of(rowf[cellf]), csub[rf, cellf] if photo_ink else None, gy / H)
                        spans.append(f'<tspan x="{gx:.1f}" y="{gy:.1f}" fill="{fill}">{esc(ch)}</tspan>')
                    if not spans:
                        continue
                    doc.add(
                        f'<text xml:space="preserve" font-family="{esc(_MONO_FAMILY)}" '
                        f'font-size="{fe:.2f}" font-weight="{esc(cfg.font_weight)}">'
                        + "".join(spans) + "</text>"
                    )
                    runs.append(TextRun(region="eye", path_id=f"eye{rf}_{start}",
                                        path_d="", text="".join(glyphs),
                                        font_size=round(fe, 2), kind="detail"))

    if not runs:
        warns.error("text", "no_runs", "Tonal fill produced no text (subject too bright or mask empty).")
    return doc.to_svg(), runs
