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

import base64
import io
import os
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
# Row pitch as a fraction of the font size. Lower packs rows tighter (denser, more
# vertical overlap); higher gives cleaner separation but opens black bands when the
# pitch exceeds the glyph height. 0.88 ≈ the glyph height: rows clear the overlap of
# the original 0.80 pack while still fully covering (no gaps). Words/Passage only.
_ROW_HEIGHT_FRAC = 0.88

# Supersampling: rasterize the text mask + composite at N× the output width, then
# Lanczos-downsample to the final size. This is higher-quality anti-aliasing for
# small/fine type than rasterizing straight to the output resolution. Set to 1 to
# disable (exact prior behavior). Gated by _SS_MAX_RENDER_W so already-large
# download renders don't multiply memory — only the smaller preview is supersampled.
_SUPERSAMPLE = 2
_SS_MAX_RENDER_W = 3200

# Selective colour: in the COLOUR inks (and Custom), the eyes + teeth keep their
# true photo colours (the same look "Original" gives) so they pop against the
# tinted face. 0 = off (fully monochrome ink), 1 = full natural in those regions;
# values between dial the strength. Edges are always feathered.
_SELCOLOR = 1.0


def _eff_supersample(out_width: int) -> int:
    """Effective supersample factor for this output width (1 = off). Drops to 1
    when N× would exceed the render-width cap, so big downloads stay cheap."""
    ss = max(1, int(_SUPERSAMPLE))
    while ss > 1 and out_width * ss > _SS_MAX_RENDER_W:
        ss -= 1
    return ss


def _lanczos_down(img, out_width: int):
    """Downsample a PIL image to out_width (preserving aspect) with Lanczos."""
    from PIL import Image
    W, H = img.size
    if W <= out_width:
        return img
    target_h = max(1, int(round(H * (out_width / float(W)))))
    return img.resize((out_width, target_h), Image.LANCZOS)

# Legibility/texture controls. Per-word jitter (fraction of cell/row) scatters
# whole words to break the grid without wobbling letters within a word; word gap
# is the blank cells between words (clearer separation reads better).
# X jitter breaks vertical rivers without hurting reading; Y is kept low so rows
# stay on clean baselines (most legible). Word gap separates words for clarity.
# Banding is otherwise held off by the random word order.
_JITTER_X = 0.16
_JITTER_Y = 0.10
_WORD_GAP = 2
# River control. A SUB-CELL jitter is not enough to break vertical "rivers" when
# the input is a single word (or several equal-length words): every row repeats
# the same word-length+gap period, so the blank gaps between words stack into
# vertical channels. We desync the period two ways that DON'T move a glyph off
# the tonal cell it samples (unlike the poster's large uniform x-shift): (1) a
# random whole-cell leading phase per row, and (2) a variable blank gap between
# words. Together the word boundaries fall at different x on every row, so even
# one repeated word renders river-free while each letter still reflects its own
# cell's tone.
_WORD_GAP_MIN = 1
_WORD_GAP_MAX = 3
_RIVER_PHASE_CELLS = 5  # max random leading blank cells at the start of a row run
# Silhouette edge feather. Cells within this many body-cell-widths of the mask
# edge are thinned on a dither screen (densest drop right at the edge), so hard
# boundaries -- especially wispy hair atop the head -- dissolve into stippled
# letters that fade to the ground rather than rendering as a solid block.
_EDGE_FEATHER_CELLS = 2.2

# Ordered-dither (Bayer 8x8) thresholds in [0,1). Used by the optional
# tone-density pass to thin the DEEPEST shadows into a regular halftone so the
# dark ground shows through and the face lifts off it. Ordered (a fixed
# repeating screen), never stochastic -- random skipping punches a Swiss-cheese
# face; a Bayer screen reads as an even, intentional texture.
_BAYER8 = (np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) + 0.5) / 64.0
# Tone above which the density pass starts thinning (1 = darkest). Only the
# deepest shadows screen out; midtones/highlights stay fully dense so the lit
# face keeps its continuous gold and never goes holey.
_DENSITY_KNEE = 0.60

# Per-glyph gray ramp (0-255): lightest inked cells near this gray, darkest
# features near-black, so tone gradients carry the likeness. Kept just below
# white so the brightest skin/hair still render as very faint words (not blank)
# while highlights read light enough to give the portrait real contrast.
_SHADE_LIGHT = 172
_SHADE_DARK = 0

# Message/prose renders read darker than Words: uniform rows of text over the
# photo leave large shadow-toned bands that crush to near-black. This gamma lift
# (applied to the finished Message composite only, before vibrance) opens the
# shadows/midtones toward life while leaving highlights near-white. 0 disables.
_MSG_BOOST = 0.5

# Every render is composed onto a STANDARD PRINT canvas: 4:5 (16"x20", also
# 8"x10") so the digital file, the Printful print, and the studio's framed
# presentation all share one true aspect -- no letterbox gaps in the mat and no
# crop at the printer. The art is fitted centred and the canvas filled with the
# ground colour, which is seamless because the composite ground is solid.
_PRINT_ASPECT = 0.8          # width / height = 4:5


def _fit_print_canvas(arr: np.ndarray, ground, aspect: float = _PRINT_ASPECT) -> np.ndarray:
    """Fit an HxWx3 image onto a `aspect` (width/height) canvas filled with
    `ground` (RGB/BGR triple matching the array's channel order). Centred;
    downscales only when the image is taller than the canvas allows. Returns the
    array unchanged when it already matches the target aspect.

    `aspect` defaults to 4:5 (the digital download / on-screen proof); the print
    path passes each product's true aspect (e.g. 0.75 for an 18x24 poster) so the
    fulfilment file matches the physical size with the face centred and ground-
    padded -- never cropped or stretched to fit."""
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return arr
    cw, ch = w, int(round(w / aspect))
    if abs(w / h - aspect) < 0.005:
        return arr
    if h > ch:                       # too tall for this width -> scale down to fit
        f = ch / float(h)
        arr = cv2.resize(arr, (max(1, int(round(w * f))), ch), interpolation=cv2.INTER_AREA)
        h, w = arr.shape[:2]
    canvas = np.empty((ch, cw, 3), dtype=arr.dtype)
    canvas[:] = np.asarray(ground, dtype=arr.dtype)
    ox, oy = (cw - w) // 2, (ch - h) // 2
    canvas[oy:oy + h, ox:ox + w] = arr
    return canvas


def _auto_message_font(text: str) -> float:
    """Pick the Message/prose font size from message length. build_poster repeats
    the message to fill the silhouette, so size is purely a legibility<->density
    trade: a short note renders bold and readable (the phrase reads as a unit); a
    long letter goes finer so more of it shows before the phrase cycles. Smooth
    log falloff, clamped to a readable band [13, 25]px. No manual knob."""
    import math
    n = len(str(text).strip())
    if n <= 1:
        return 24.0
    f = 24.0 - 3.5 * math.log2(max(n, 24) / 24.0)
    return float(max(13.0, min(25.0, f)))

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

# Calligram looks, keyed by the same swatch names as the mosaic inks:
# (ink/full-darkness colour, background). gold_noir is light ink on a dark page.
_CALLIGRAM = {
    "navy":      ("#0d1b3a", "#ffffff"),
    "sepia":     ("#2a1808", "#fbf6ea"),
    "burgundy":  ("#3f0d16", "#ffffff"),
    "forest":    ("#0d2418", "#ffffff"),
    "gold_noir": ("#e8c66a", "#101216"),
    "mono":      ("#141414", "#ffffff"),
    "photo":     ("#15202b", "#ffffff"),
}

# Poster looks: a near-black ground and a bright ink, keyed by swatch. The
# poster renders brightness-positive (lit face = brightest text), so the photo
# is faithfully evident -- the gallery/tribute "type-poster" style.
_POSTER = {
    "navy":      ("#090d18", "#dbe4f1"),
    "sepia":     ("#0c0a06", "#ecdab4"),
    "burgundy":  ("#120709", "#edc6ca"),
    "forest":    ("#06110b", "#cfe6d6"),
    "gold_noir": ("#0b0a06", "#e8c66a"),
    "mono":      ("#0a0a0a", "#f2ece0"),
    "photo":     ("#0a0a0c", None),       # tint from the source colour
    # Original on PAPER: the source colour rendered as a coloured engraving on a
    # warm white. Darkness drives the ink (shadows/features bold) with a floor so
    # light skin + grey hair still tint instead of vanishing into the paper.
    "photo_paper": ("#f6f1e8", None),
}


def custom_poster(hex_in: str):
    """Build a (ground, ink) poster pair from a user-picked colour for the 'custom'
    ink. The near-black ground is fixed; the chosen hue is lifted toward white if
    it's too dark to read on that ground (hue preserved, just brightened). Returns
    None on a malformed hex so the caller can fall back."""
    try:
        r, g, b = _hex_to_rgb(hex_in)
    except Exception:  # noqa: BLE001
        return None
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b        # 0..255
    if lum < 150:                                      # too dark for the dark ground -> lift
        t = min(0.78, (150.0 - lum) / 200.0)
        r = int(round(r + (255 - r) * t))
        g = int(round(g + (255 - g) * t))
        b = int(round(b + (255 - b) * t))
    return ("#0a0a0c", "#%02x%02x%02x" % (r, g, b))

# MediaPipe 478-point mesh index groups for the recognition features we deepen.
_EYE_L = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_EYE_R = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
_BROW_L = (70, 63, 105, 66, 107, 46, 53, 52, 65, 55)
_BROW_R = (336, 296, 334, 293, 300, 276, 283, 282, 295, 285)
_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
_NOSE = (1, 2, 98, 327, 97, 326, 5, 4, 275, 440, 220, 45)
_FEATURE_GROUPS = (_EYE_L, _EYE_R, _BROW_L, _BROW_R, _LIPS, _NOSE)
# Luminance ceiling for the composited real-eye overlay on dark grounds. The photo's
# own eye can be blown out (255 catchlight, ~250 sclera); on the dark word-face that
# reads as a glowing orb. Matching the Lifelike path (catchlight 238 / sclera ~200),
# we soft-pull the eye below blow-out so it stays faithful but cannot bloom.
_EYE_LUMA_CAP = 198.0

# Eye treatment is chosen by INK, not by style: the realistic photographic eye only
# matches a full-colour face (the Photo / "Original" ink), so it is used there; the
# tinted/monochrome inks (Noir/Sepia/Navy/Sage) render the stylized TYPOGRAPHIC eye
# instead, in the ink's palette, so a full-colour eye never clashes with a tinted
# face. (A brief detour rendered a typographic eye for ALL Mosaic/Passage inks; the
# user rejected it for the Photo ink -- realism wins there -- so the split is by ink.)


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
        # No detected face. The human app (Typortrait / Loved in Words) leaves
        # this untouched -- return `dark` exactly as before, so faceless human
        # renders are byte-identical to prod.
        #
        # Only an EXPLICIT pet render opts in via `an.pet_subject = True`. There,
        # without the per-face rescue, a pale, low-contrast coat -- a golden, a
        # white or spotted dog -- collapses into a narrow bright band under the
        # global `_tone_field` stretch and washes out to near-blank. Balance the
        # whole silhouette as one region so it still reaches full ink depth,
        # using the same wide-band remap as the per-face path.
        if not getattr(an, "pet_subject", False):
            return dark
        core = mset > 0.5  # robust to a bool or a float subject mask
        if int(core.sum()) < 50:
            return dark
        vals = dark[core]
        lo, hi = np.percentile(vals, [10, 90])
        if hi - lo < 0.04:
            hi = lo + 0.04
        t_lo, t_hi = 0.12, 0.96
        remap = np.clip((dark - lo) / (hi - lo), 0.0, 1.0) * (t_hi - t_lo) + t_lo
        alpha = 0.72
        out = dark.copy()
        out[core] = (dark * (1.0 - alpha) + remap * alpha)[core]
        return np.clip(out, 0.0, 1.0)
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
        # Lift the highlight floor (t_lo) so light skin carries visible ink and
        # the face reads solid instead of washing to white, and extend the dark
        # end so shadows go deep -- more presence and pop, kept linear so no S-
        # curve brightens the highlights back out or blotches the modeling.
        t_lo, t_hi = 0.12, 0.96
        remap = np.clip((dark - lo) / (hi - lo), 0.0, 1.0) * (t_hi - t_lo) + t_lo
        alpha = 0.72
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
            # Tight to the EYEBALL (the lid aperture): lids/brows take normal
            # face-tier words; only the eyeball itself is reserved for the iris
            # pass, so type appears in the iris and nowhere else inside the eye.
            rx = (float(ep[:, 0].max() - ep[:, 0].min()) / 2.0) * 1.05
            ry = (float(ep[:, 1].max() - ep[:, 1].min()) / 2.0) * 1.10
            if rx >= 2.0 and ry >= 2.0:
                out.append((cx, cy, rx, ry))
    return out


# --- Living eyes: the person's true iris colour carried by the glyphs ---------
# MediaPipe's 478-point mesh includes the irises: centre + 4-point ring per eye.
_IRIS_L = (468, (469, 470, 471, 472))
_IRIS_R = (473, (474, 475, 476, 477))
# Faithfulness gate: tint ONLY when the photo itself clearly carries the eye
# colour. Both irises must be reasonably saturated and hue-consistent with each
# other; faded/B&W/dim photos fail the gate and render exactly as before. We
# sample colour, never invent it (same doctrine as the enhancement stage).
_IRIS_MIN_SAT = 50.0     # OpenCV HSV S (0-255); pale blue irises sit ~60-80
_IRIS_HUE_TOL = 22.0     # max circular hue difference between the two eyes


def _iris_circles(an, scale: float) -> List[Tuple[float, float, float]]:
    """Per-iris (cx, cy, r) circles in render coords from the 478-mesh iris
    landmarks (geometry only, no colour gate); [] when unavailable/too small."""
    out: List[Tuple[float, float, float]] = []
    for face in _faces_of(an):
        pts = face.points
        if pts.shape[0] < 478:
            continue
        for c, ring in (_IRIS_L, _IRIS_R):
            cx, cy = float(pts[c][0]), float(pts[c][1])
            r = float(np.mean([np.hypot(pts[i][0] - cx, pts[i][1] - cy) for i in ring]))
            if r * scale >= 2.5:
                out.append((cx * scale, cy * scale, r * scale))
    return out


# Inner-lip ring (MediaPipe 478-mesh): bounds the mouth OPENING, i.e. the teeth.
_INNER_LIP = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
              308, 324, 318, 402, 317, 14, 87, 178, 88, 95)


def _teeth_mask(pts, h: int, w: int):
    """Soft mask of the inner mouth (where teeth show) from already-scaled
    landmark points. None when the mouth is ~closed (no teeth to clear) or the
    mesh is unavailable -- so a closed-mouth portrait is left untouched."""
    if pts is None or len(pts) <= max(_INNER_LIP):
        return None
    p = np.array([pts[i] for i in _INNER_LIP], np.float32)
    pw = float(p[:, 0].max() - p[:, 0].min())
    ph = float(p[:, 1].max() - p[:, 1].min())
    if pw < 3.0 or ph < 2.0 or ph / pw < 0.12:    # lips together -> no teeth
        return None
    mm = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(mm, cv2.convexHull(p.astype(np.int32)), 1.0)
    # Tight feather: enough to anti-alias the boundary, not so much that the teeth
    # blur into the lips (the fill itself is unsharp-masked to read crisp).
    return cv2.GaussianBlur(mm, (0, 0), max(1.0, pw * 0.03))


def _catchlight_points(an) -> List[Tuple[float, float, float]]:
    """Catchlight positions, one per iris, in WORKING coords: (gx, gy, glint_r).
    Deterministic and consistent between the two eyes -- the classic upper
    diagonal at ~0.34 r from the iris centre, on the side the face is lit from.
    (The photo's own brightest-pixel is unreliable: it often sits on the iris
    circle's rim, landing the glint off the iris and differently per eye.)"""
    circles = _iris_circles(an, 1.0)
    if not circles:
        return []
    side = -1.0                                   # default: light from viewer-left
    try:
        fbb = an.face_bbox
        if fbb:
            x, y, bw, bh = (int(v) for v in fbb)
            g = an.img.gray[max(0, y):y + bh, max(0, x):x + bw]
            if g.size:
                half = g.shape[1] // 2
                side = -1.0 if float(g[:, :half].mean()) >= float(g[:, half:].mean()) else 1.0
    except Exception:  # noqa: BLE001
        pass
    return [(cx + side * 0.24 * r, cy - 0.24 * r, 0.13 * r) for cx, cy, r in circles]


def _photo_eye_overlay(bgr_hw, pts, eye_groups, H: int, W: int):
    """REAL-eye overlay -- the single biggest realism lever. Returns
    (eye_bgr (H,W,3) float32, alpha (H,W) float32): the photo's OWN eye openings,
    per-eye tone-normalised so they read on any ground while KEEPING every bit of the
    real modelling -- the spherical-curvature falloff, the upper-lid cast shadow, the
    lashes, the iris colour + texture, the true catchlight. A flat synthetic sclera
    disc (one uniform grey, no gradient) was the failure this replaces.

    Per eye, each eye is processed in its OWN bounding box at a consistent internal
    resolution (small/low-res eyes -- e.g. a phone screenshot, an old scan -- are
    upscaled; large ones stay native), so the tone + sharpen are stable regardless of
    canvas size and the on-screen PREVIEW eye is as crisp as the paid file. An
    edge-preserving (bilateral) clean removes low-res blockiness while keeping the
    iris rim, lid and lash edges; the tone is FAITHFUL -- the photo's own brightness
    and colour are kept, only the deepest socket shadows are gently lifted, so the
    sclera is never brighter than the source and the lids/lashes keep their real
    shading. The composited region is the whole ORBITAL area (lids + lashes + socket),
    feathered into the surrounding words, so the eye has a natural transition rather
    than a bare eyeball in a dark hole. `pts` are in (H,W) coords; `eye_groups` are the
    two eye-contour index lists. Memorial families upload what they have, so this must
    hold up on a low-resolution source, not just a studio photo."""
    eye_bgr = bgr_hw.astype(np.float32).copy()
    alpha = np.zeros((H, W), np.float32)
    for grp in eye_groups:
        p = np.array([pts[i] for i in grp if i < len(pts)], np.int32)
        if len(p) < 4:
            continue
        hull = cv2.convexHull(p)
        bx, by, bw, bh = cv2.boundingRect(hull)
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        # ORBITAL region, not just the eye opening: cover the eyeball AND the lids,
        # lashes and socket so the eye has a real lid + lash + transition into the
        # skin, instead of a bare eyeball in a dark hole. Eyes are wider than tall, so
        # size the vertical half-axis off the width too, and bias the centre up toward
        # the upper lid.
        # Halo geometry: covers the eyeball + lids + a soft socket transition. Kept
        # modest, and the radial alpha below only PARTIALLY composites the orbital, so
        # the photo's socket shadow never paints a dark "raccoon" ring on a bright face.
        sx = max(bw * 0.70, 4.0)
        sy = max(bh * 0.95, bw * 0.34, 4.0)
        ecy = cy - 0.10 * sy
        pad = int(round(max(sx, sy) * 2.2))
        X0, Y0 = max(0, int(cx - pad)), max(0, int(ecy - pad))
        X1, Y1 = min(W, int(cx + pad)), min(H, int(ecy + pad))
        sw, sh = X1 - X0, Y1 - Y0
        if sw < 6 or sh < 6:
            continue
        sub = eye_bgr[Y0:Y1, X0:X1]
        sc = max(1.0, 280.0 / float(max(sw, sh)))            # upscale small/low-res eyes; large stay native
        bw2, bh2 = int(round(sw * sc)), int(round(sh * sc))
        big = cv2.resize(sub, (bw2, bh2), interpolation=cv2.INTER_CUBIC)
        # Edge-preserving clean: low-res sources are blocky after the upscale; bilateral
        # smooths the blocks while keeping the iris/lid/lash edges.
        big = cv2.bilateralFilter(np.clip(big, 0, 255).astype(np.uint8), 7, 30, 7).astype(np.float32)
        # FAITHFUL tone: keep the photo's OWN brightness + colour. Lift ONLY the deep
        # shadows (a recessed socket) so it isn't pure black; the sclera and midtones
        # are left exactly as shot -> the eye-white is never brighter than the source,
        # and the lids/lashes keep their natural shading.
        lab = cv2.cvtColor(np.clip(big, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        n = lab[..., 0].astype(np.float32) / 255.0
        bump = 0.10 * np.clip((0.30 - n) / 0.30, 0.0, 1.0)   # only n<0.30 lifted; highlights untouched
        lab[..., 0] = np.clip((n + bump) * 255.0, 0, 255).astype(np.uint8)
        toned = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
        sft = cv2.GaussianBlur(toned, (0, 0), sigmaX=max(1.0, bw2 * 0.010))
        toned = np.clip(toned * 1.28 - sft * 0.28, 0, 255)   # mild definition only, no harsh contrast
        eye_bgr[Y0:Y1, X0:X1] = cv2.resize(toned, (sw, sh), interpolation=cv2.INTER_AREA)
        # Alpha = a FAITHFUL core over the eye opening (so the eyeball reads true) PLUS
        # a gentle radial halo over the orbital that only partially composites -- the
        # lid/socket BLENDS with the surrounding face. On a bright face the socket
        # shadow no longer paints a dark ring; on a dark face the lid still picks up the
        # photo's structure. The halo peak is well below 1 so it can never dominate.
        core = np.zeros((H, W), np.float32)
        cv2.fillConvexPoly(core, hull, 1.0)
        core = cv2.GaussianBlur(core, (0, 0), sigmaX=max(1.5, bh * 0.22))
        yy = np.arange(H, dtype=np.float32)[:, None]
        xx = np.arange(W, dtype=np.float32)[None, :]
        halo = 0.45 * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - ecy) / sy) ** 2)).astype(np.float32)
        alpha = np.maximum(alpha, np.clip(np.maximum(core, halo), 0.0, 1.0))
    return eye_bgr, np.clip(alpha, 0.0, 1.0)


def _sclera_shade(gray, an, scale: float, sm, floor: float = 0.58):
    """Per-eye floored luminance stretch + synthesised upper-lid shadow, returning a
    0..1 value map over the sclera mask `sm`. Mirrors the Sculpt engine's
    `_sclera_value`: the sclera must NOT be painted from the photo's own eye pixels
    (on a deep-set/shadowed eye those are near-black, so the white collapses to the
    dark ground -- the eye reads as a black socket with only a glint). Stretching
    each eye's OWN luminance restores the natural gradient; normalising PER EYE with
    a floor keeps even a shaded eye bright. `scale` maps analysis coords -> (H,W)."""
    H, W = gray.shape
    val = np.full((H, W), floor, np.float32)
    faces = _faces_of(an)
    if not faces:
        return val
    gb = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    yy = np.arange(H, dtype=np.float32)[:, None]
    pts = faces[0].points * scale
    for grp in (_EYE_L, _EYE_R):
        p = np.array([pts[i] for i in grp if i < len(pts)], np.int32)
        if len(p) < 3:
            continue
        em = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(em, cv2.convexHull(p), 1)
        m = (em > 0) & (sm > 0.05)
        if int(m.sum()) < 12:
            continue
        gp = gb[m]
        lo, hi = np.percentile(gp, [12, 94])
        if hi - lo < 6.0:
            hi = lo + 6.0
        v = np.clip((gb - lo) / (hi - lo), 0.0, 1.0)
        base = floor + (1.0 - floor) * v
        y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
        vy = np.clip((yy - y0) / max(8.0, (y1 - y0)), 0.0, 1.0)         # 0 top -> 1 bottom
        lid = 0.52 + 0.48 * np.clip((vy - 0.08) / 0.55, 0.0, 1.0)       # darker top, full lower
        val[m] = np.clip(base * np.broadcast_to(lid, (H, W)), 0.0, 1.0)[m]
    return val


def _iris_tint(an):
    """Sample the primary face's iris colour. Returns ([(cx, cy, r)] in working
    coords, lifted RGB tip colour) when both irises pass the colour gate, else
    None (renders fall back to the plain ink, byte-identical)."""
    faces = _faces_of(an)
    if not faces:
        return None
    pts = faces[0].points                      # primary face (renderer features it)
    if pts.shape[0] < 478:
        return None                            # no iris landmarks (fallback mesh)
    bgr = an.img.bgr
    H0, W0 = bgr.shape[:2]
    circles: List[Tuple[float, float, float]] = []
    hsvs: List[np.ndarray] = []
    for c, ring in (_IRIS_L, _IRIS_R):
        cx, cy = float(pts[c][0]), float(pts[c][1])
        r = float(np.mean([np.hypot(pts[i][0] - cx, pts[i][1] - cy) for i in ring]))
        if r < 3.0:
            return None
        # Sample the inner iris only (0.68 r skips lid/lash overlap), dropping
        # the darkest 30% (pupil) and brightest 20% (catchlight) of pixels.
        x0, x1 = int(max(0, cx - r)), int(min(W0, cx + r + 1))
        y0, y1 = int(max(0, cy - r)), int(min(H0, cy + r + 1))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        yy, xx = np.ogrid[y0:y1, x0:x1]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.68 * r) ** 2
        px = bgr[y0:y1, x0:x1][m].astype(np.float32)
        if px.shape[0] < 12:
            return None
        lum = px.mean(1)
        lo, hi = np.percentile(lum, [30.0, 80.0])
        sel = px[(lum >= lo) & (lum <= hi)]
        if sel.shape[0] < 6:
            return None
        med = np.clip(np.median(sel, 0), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(med.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
        circles.append((cx, cy, r))
        hsvs.append(hsv)
    if len(circles) < 2:
        return None
    if any(h[1] < _IRIS_MIN_SAT for h in hsvs):
        return None                            # photo doesn't carry the colour
    dh = abs(float(hsvs[0][0]) - float(hsvs[1][0]))
    if min(dh, 180.0 - dh) > _IRIS_HUE_TOL:
        return None                            # eyes disagree -> unreliable sample
    # Shared colour with a gentle saturation nudge. The ink VALUE tracks the true
    # eye lightness: pale blue / light-brown irises render lighter than deep
    # brown, so the rendered eye colour is honest to the person, not uniform.
    hsv = np.mean(hsvs, 0)
    hsv[1] = min(255.0, hsv[1] * 1.25)
    hsv[2] = float(np.interp(hsv[2], [30.0, 160.0], [140.0, 235.0]))
    rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8).reshape(1, 1, 3),
                       cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)
    return circles, rgb


def _face_ovals(an, scale: float) -> List[Tuple[float, float, float, float]]:
    """Per-face (cx, cy, rx, ry) ellipses (render coords) marking the area that
    gets the finer 'face' word tier, so the likeness keeps its detail while the
    body is rendered with larger, more legible words."""
    out: List[Tuple[float, float, float, float]] = []
    for face in _faces_of(an):
        pts = face.points * scale
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        fw = float(pts[:, 0].max() - pts[:, 0].min())
        fh = float(pts[:, 1].max() - pts[:, 1].min())
        if fw < 4 or fh < 4:
            continue
        out.append((cx, cy, fw * 0.60, fh * 0.72))
    return out


def _emphasize_features(dark: np.ndarray, an, scale: float, mset: np.ndarray,
                        gamma: float = 0.48) -> np.ndarray:
    """Deepen the brows and lips of every face so the likeness anchors there.
    The nose is intentionally NOT uniformly darkened -- filling its whole hull
    turns it into a dark blob ("muddled nose"); its shape reads from natural
    shading plus the edge-separation pass (nostril/side shadows). Eyes are
    handled separately (_sharpen_eyes)."""
    faces = _faces_of(an)
    if not faces:
        return dark
    H, W = dark.shape[:2]
    fm = np.zeros((H, W), np.uint8)
    for face in faces:
        pts = face.points * scale
        for grp in (_BROW_L, _BROW_R, _LIPS):
            hull = cv2.convexHull(np.array([pts[i] for i in grp], np.int32))
            cv2.fillConvexPoly(fm, hull, 255)
    fm = cv2.dilate(fm, np.ones((3, 3), np.uint8), 1)
    w = (cv2.GaussianBlur(fm, (0, 0), 2.2).astype(np.float32) / 255.0) * mset
    return dark * (1.0 - w) + np.clip(dark ** gamma, 0.0, 1.0) * w


def _sharpen_eyes(dark: np.ndarray, an, scale: float, mset: np.ndarray,
                  contrast: float = 1.32, catchlight: str = "hard") -> np.ndarray:
    """Make each eye read as a *live* eye: strong local contrast so iris/lash go
    dark and sclera goes light, crisp lid edges (unsharp), and a preserved
    catchlight -- the small bright glint that makes a portrait look back at you.
    Strong contrast is desirable here (unlike the gentle whole-face balance).

    catchlight: "hard" = flat painted disk (legacy), "soft" = Gaussian glint at
    the eye's real brightest pixel (reads as light, not a sticker), "off"."""
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
            # Push iris/lash darker and sclera lighter so the eye reads alive.
            st = np.clip(0.5 + (st - 0.5) * contrast, 0.0, 1.0)
            blur = cv2.GaussianBlur(st, (0, 0), max(1.0, ew * 0.04))
            sharp = np.clip(st + (st - blur) * 1.45, 0.0, 1.0)   # crisper iris/lid
            ph, pw = patch.shape[:2]
            fm = np.zeros((ph, pw), np.float32)
            cv2.ellipse(fm, (pw // 2, ph // 2), (pw // 2, ph // 2), 0, 0, 360, 1.0, -1)
            fm = cv2.GaussianBlur(fm, (0, 0), max(1.0, ew * 0.12)) * mset[by0:by1, bx0:bx1]
            out[by0:by1, bx0:bx1] = patch * (1.0 - fm) + sharp * fm

            # Catchlight: lift the brightest spot inside the eye to a light
            # glint (only if a real highlight exists), so eyes don't read dead.
            if catchlight != "off":
                ix0, iy0 = int(max(0, x0)), int(max(0, y0))
                ix1, iy1 = int(min(W, x1)), int(min(H, y1))
                eye_in = out[iy0:iy1, ix0:ix1]
                if eye_in.size and float(eye_in.min()) < 0.30:
                    cyl, cxl = np.unravel_index(int(np.argmin(eye_in)), eye_in.shape)
                    if catchlight == "soft":
                        # Gaussian glint at the photo's real brightest eye pixel:
                        # soft falloff reads as light, not a stamped disk.
                        r = max(2, int(round(eh * 0.09)))
                        gm = np.zeros(eye_in.shape, np.float32)
                        cv2.circle(gm, (cxl, cyl), r, 1.0, -1)
                        gm = cv2.GaussianBlur(gm, (0, 0), max(1.0, r * 0.7))
                        gm *= mset[iy0:iy1, ix0:ix1]
                        out[iy0:iy1, ix0:ix1] = eye_in * (1.0 - gm)
                    else:
                        r = max(3, int(round(eh * 0.11)))
                        cv2.circle(out, (ix0 + cxl, iy0 + cyl), r, 0.0, -1)
    return np.clip(out, 0.0, 1.0)


def build_calligram(
    an,
    text: str,
    cfg: RenderConfig,
    warns: WarningCollector,
    render_w: int = 2600,
    font_px: float = 22.0,
    contrast: float = 2.8,
    pivot: float = 0.34,
    power: float = 1.0,
    ink_hex: str = "#15202b",
    bg_hex: str = "#ffffff",
) -> Tuple[str, List[TextRun]]:
    """Story calligram: lay the user's own passage as continuous, ordered,
    readable prose in lines across the subject, shading each glyph by the photo's
    tone so the face emerges from the text's density. Unlike the word-mosaic this
    keeps the words in order and unbroken (you can read it), fills the whole
    masked region (faint in highlights, dark in shadows), and runs in straight
    lines like a page. Ink colour is derived from `ink_hex` (the dark end)."""
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
    sharp = _sharpen(gray)
    dark = _tone_field(sharp, mask, gamma=1.0, floor=0.0)
    dark = _auto_tone(dark, mset, 0.55, max_shift=0.18)
    dark = _balance_faces(dark, an, scale, mset)
    dark = _emphasize_features(dark, an, scale, mset)
    dark = _sharpen_eyes(dark, an, scale, mset)
    dark = _edge_separate(dark, sharp, mset, amount=0.40)
    tone_s = cv2.GaussianBlur(dark, (0, 0), max(1.0, font_px * 0.32))

    cell_w = font_px * _MONO_ADVANCE
    row_h = font_px * 1.05
    cols = max(1, int(W / cell_w))
    rows = max(1, int(H / row_h))
    ir, ig, ib = _hex_to_rgb(ink_hex)
    br, bgc, bb = _hex_to_rgb(bg_hex)
    family = "'Courier New', 'DejaVu Sans Mono', 'Liberation Mono', monospace"

    doc = SvgDoc(width=W, height=H, background=bg_hex)
    runs: List[TextRun] = []
    wi = 0
    for r in range(rows):
        baseline = (r + 1) * row_h
        yi = min(H - 1, max(0, int(baseline - 0.32 * font_px)))
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
            drew = False
            for k, ch in enumerate(word):
                col = c + k
                xi = min(W - 1, max(0, int(col * cell_w + cell_w * 0.5)))
                if not mset[yi, xi]:
                    continue
                tone = float(tone_s[yi, xi])
                norm = (tone - pivot) * contrast + pivot
                norm = 1.0 if norm > 1.0 else (0.0 if norm < 0.0 else norm)
                f = 1.0 - norm ** power           # 1 = melt into background, 0 = full ink
                if f > 0.9:                        # keep faint ink in highlights so the
                    f = 0.9                        # form stays solid (matches the mosaic)
                cr = int(round(ir + (br - ir) * f))
                cg = int(round(ig + (bgc - ig) * f))
                cb = int(round(ib + (bb - ib) * f))
                spans.append(
                    f'<tspan x="{col * cell_w:.1f}" fill="#{cr:02x}{cg:02x}{cb:02x}">{esc(ch)}</tspan>'
                )
                line_chars.append(ch); drew = True
            c += wl + 1
            if drew:                              # only consume a word when it's actually shown
                wi += 1
        if not spans:
            continue
        doc.add(
            f'<text y="{baseline:.1f}" xml:space="preserve" font-family="{esc(family)}" '
            f'font-size="{font_px:.1f}">' + "".join(spans) + "</text>"
        )
        runs.append(TextRun(region="calligram", path_id=f"row{r}", path_d="",
                            text="".join(line_chars), font_size=round(font_px, 1), kind="primary"))

    if not runs:
        warns.error("text", "no_runs", "Calligram produced no text (mask empty?).")
    return doc.to_svg(), runs


def _edge_separate(dark: np.ndarray, gray: np.ndarray, mset: np.ndarray, amount: float) -> np.ndarray:
    """Darken local edges/texture so regions of similar brightness but different
    texture separate -- fine gray hair vs smooth light skin is the classic case
    (a pure tone map renders them identically because their luminance matches).
    Gray hair carries dense fine edges, smooth skin almost none, so this pulls
    them apart and sharpens feature boundaries (hairline, nose, jaw, lips)
    without needing to segment hair."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    vals = mag[mset] if int(mset.sum()) > 50 else mag.reshape(-1)
    hi = float(np.percentile(vals, 95))
    mag = np.clip(mag / max(hi, 1e-3), 0.0, 1.0)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.2)
    # Weight by how light the area is (1 - dark): edges in light/mid regions
    # (gray hair, hairline against skin) darken for separation, while already-
    # dark hair is barely touched -- otherwise dense dark hair saturates into a
    # solid "helmet" with no internal variation.
    out = dark + amount * mag * (1.0 - dark)
    out[~mset] = dark[~mset]
    return np.clip(out, 0.0, 1.0)


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
    target_tone: float = 0.55,
    jitter: float = 0.7,
    seed: int = 1234,
    contrast: float = 2.8,
    pivot: float = 0.34,
    ink: str = "mono",
    tone_density: float = 0.0,
    gap_fill: bool = True,
    gap_fill_passes: int = 12,
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
    # Distance (px) from the silhouette edge, inward. Used to FEATHER the outer
    # boundary: cells near the edge are dropped on a dither screen so a hard mask
    # edge (e.g. wispy hair atop the head) dissolves into scattered letters that
    # fade to the ground, instead of tiling into a solid block.
    medge = cv2.distanceTransform(mset.astype(np.uint8), cv2.DIST_L2, 5)
    sharp = _sharpen(gray)
    dark = _tone_field(sharp, mask, gamma=gamma, floor=floor)
    if auto_tone:
        dark = _auto_tone(dark, mset, target_tone, max_shift=0.18)
    dark = _balance_faces(dark, an, scale, mset)
    dark = _emphasize_features(dark, an, scale, mset)
    dark = _sharpen_eyes(dark, an, scale, mset)
    # Separate similar-brightness regions (gray hair vs light skin) and crisp up
    # feature boundaries via local edge/texture darkening.
    dark = _edge_separate(dark, sharp, mset, amount=0.40)

    # ---- Size tiers ---------------------------------------------------------
    # Words step DOWN in size toward the face: BIG across the outer body, MID in
    # a ring around the head/shoulders, SMALL on the face, and the FINEST glyphs
    # in the eyes. The gentle steps read as a smooth large->mid->small gradient
    # (not a hard two-size seam), so the typography is legible AND the likeness
    # holds where detail matters.
    # Tier ratios kept close together so words step down GENTLY toward the face
    # (body -> mid -> face), avoiding a harsh size discontinuity where the small
    # face tier meets the larger hair/headwear tiers.
    # Subject-relative type scale: size the words to the face's SHARE of the
    # frame, so a close-up and a far/loosely-cropped shot of the same person
    # render with consistent word density and recognisability. The size control
    # (min_font_px) is the MULTIPLIER on this subject-normalised base, not an
    # absolute pixel size -- so "Giant" is reliably giant relative to the person
    # on every photo. face_frac is the face width as a fraction of the source;
    # ref_frac is a typical head-and-shoulders framing (norm = 1.0 there).
    ref_frac = 0.42
    face_frac = (float(an.face_bbox[2]) / float(w0)
                 if (getattr(an, "face_bbox", None) and w0 > 0) else ref_frac)
    norm = float(np.clip(face_frac / ref_frac, 0.45, 2.2))
    base = cfg.min_font_px * norm
    body_font = float(min(cfg.max_font_px, max(12.0, base * 2.2)))
    mid_font = float(min(cfg.max_font_px, max(10.0, base * 1.45)))
    face_font = float(max(8.0, base * 1.0))
    eye_font = float(max(5.0, face_font * 0.45))   # finest: words sized to eye anatomy

    # Ink treatment: grayscale (mono), a named duotone, or colour sampled from
    # the source photo. Mono keeps the existing gray ramp untouched.
    photo_ink = ink == "photo"
    grad = _GRADIENTS.get(ink)
    duo = _PALETTES[ink][:2] if (ink in _PALETTES and ink != "mono") else None
    bg = _PALETTES[ink][2] if (ink in _PALETTES and ink != "mono") else cfg.background_hex
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
        if photo_ink and src is not None:
            b0, g0, r0 = int(src[0]), int(src[1]), int(src[2])  # BGR
            luma = 0.299 * r0 + 0.587 * g0 + 0.114 * b0
            s = g / max(luma, 1.0)  # tint source colour to our tonal lightness
            return f"#{min(255,int(r0*s)):02x}{min(255,int(g0*s)):02x}{min(255,int(b0*s)):02x}"
        if duo is not None:
            cr = int(round(lo_rgb[0] + (hi_rgb[0] - lo_rgb[0]) * t_dark))
            cg = int(round(lo_rgb[1] + (hi_rgb[1] - lo_rgb[1]) * t_dark))
            cb = int(round(lo_rgb[2] + (hi_rgb[2] - lo_rgb[2]) * t_dark))
            return f"#{cr:02x}{cg:02x}{cb:02x}"
        return f"#{g:02x}{g:02x}{g:02x}"

    # The face gets a smaller-word tier and the eyes the finest one; the larger
    # tiers skip those regions so the passes don't overprint.
    eyes = _eye_ellipses(an, scale)

    def in_eyes(px: float, py: float) -> bool:
        for ex, ey, rx, ry in eyes:
            if ((px - ex) / rx) ** 2 + ((py - ey) / ry) ** 2 <= 1.0:
                return True
        return False

    # Inside the eyeball, type lives ONLY in the iris annulus: outside the round
    # pupil (0.40 r), inside the iris circle. Sclera and pupil carry no glyphs at
    # all, so the eye reads as anatomy -- round pupil, typed iris -- not as text.
    iris_cl = _iris_circles(an, scale)

    def in_iris_annulus(px: float, py: float) -> bool:
        for icx, icy, irr in iris_cl:
            d2 = (px - icx) ** 2 + (py - icy) ** 2
            if (0.40 * irr) ** 2 <= d2 <= irr * irr:
                return True
        return False

    face_ov = _face_ovals(an, scale)

    def in_face(px: float, py: float) -> bool:
        for cx, cy, rx, ry in face_ov:
            if ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 <= 1.0:
                return True
        return False

    # A larger ring around the face takes the mid size, so words step down
    # large -> mid -> small toward the face instead of jumping at one hard edge.
    mid_ov = [(cx, cy, rx * 1.95, ry * 1.85) for cx, cy, rx, ry in face_ov]

    def in_mid(px: float, py: float) -> bool:
        for cx, cy, rx, ry in mid_ov:
            if ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 <= 1.0:
                return True
        return False

    doc = SvgDoc(width=W, height=H, background=bg)
    runs: List[TextRun] = []
    # Seeded (reproducible) per-row jitter offsets break up the rigid column grid
    # so words don't form vertical "rivers" or horizontal banding.
    rng = np.random.default_rng(seed)
    # Occupancy grid: pixels already covered by a placed glyph. The multi-scale
    # gap-fill passes consult this so they pack smaller words ONLY into regions
    # the larger tiers left blank -- filling the silhouette across the full size
    # range with no empty black holes.
    occ = np.zeros((H, W), dtype=bool)

    def emit(font: float, bx0: int, by0: int, bx1: int, by1: int, region: str, kind: str) -> None:
        """Lay one size tier of words over [bx0:bx1, by0:by1]. Every cell inside
        the silhouette is inked (highlights -> faint glyphs, so the form stays
        continuous); the body tier skips the face, the face tier skips the eyes."""
        cw = font * _MONO_ADVANCE
        rh = font * _ROW_HEIGHT_FRAC
        bx0 = int(max(0, bx0)); by0 = int(max(0, by0))
        bx1 = int(min(W, bx1)); by1 = int(min(H, by1))
        cols = int((bx1 - bx0) / cw)
        rows = int((by1 - by0) / rh)
        if cols < 1 or rows < 1:
            return
        # Area-average the tone into this tier's grid so each letter reflects the
        # mean darkness of its whole cell (smooth gradients, not point noise).
        sub = cv2.resize(dark[by0:by1, bx0:bx1], (cols, rows), interpolation=cv2.INTER_AREA)
        mg = cv2.resize(mask[by0:by1, bx0:bx1], (cols, rows), interpolation=cv2.INTER_AREA)
        # Min-pool the edge distance into the grid (INTER_AREA would blur the
        # edge inward); each cell takes the SMALLEST distance it covers so a cell
        # straddling the boundary is treated as an edge cell.
        eg = cv2.resize(medge[by0:by1, bx0:bx1], (cols, rows), interpolation=cv2.INTER_NEAREST)
        feather_px = font * _MONO_ADVANCE * _EDGE_FEATHER_CELLS
        csub = (cv2.resize(an.img.bgr[by0:by1, bx0:bx1], (cols, rows),
                           interpolation=cv2.INTER_AREA) if photo_ink else None)
        for r in range(rows):
            ox = (rng.random() - 0.5) * cw * jitter
            oy = (rng.random() - 0.5) * rh * jitter * 0.5
            baseline = by0 + (r + 0.5) * rh + font * 0.34 + oy
            cy_nom = by0 + (r + 0.5) * rh
            row = sub[r]
            ink = mg[r] > 110
            fill_mode = (region == "fill")
            for c in range(cols):
                if not ink[c]:
                    continue
                px = bx0 + (c + 0.5) * cw
                if fill_mode:
                    # Gap-fill: ink only where no larger pass already drew; leave
                    # the dedicated eye pass region to that finer pass.
                    yy = min(H - 1, max(0, int(cy_nom))); xx = min(W - 1, max(0, int(px)))
                    if in_eyes(px, cy_nom) or occ[yy, xx]:
                        ink[c] = False
                    continue
                if in_eyes(px, cy_nom):
                    ink[c] = False
                elif region == "body" and in_mid(px, cy_nom):
                    ink[c] = False
                elif region == "mid" and (in_face(px, cy_nom) or not in_mid(px, cy_nom)):
                    ink[c] = False
                elif region == "face" and not in_face(px, cy_nom):
                    ink[c] = False
                elif tone_density > 0.0 and region in ("body", "mid") and row[c] > _DENSITY_KNEE:
                    # Structured shadow thinning: drop cells on a Bayer screen
                    # so the deepest shadows open to the dark ground and the
                    # face emerges. Face/eye tiers are exempt (keep full detail).
                    keep = 1.0 - tone_density * (row[c] - _DENSITY_KNEE) / (1.0 - _DENSITY_KNEE)
                    if keep <= _BAYER8[r & 7, c & 7]:
                        ink[c] = False
                if ink[c] and region == "body" and feather_px > 0.0 and eg[r, c] < feather_px:
                    # Silhouette-edge feather: keep-probability ramps 0 (at the
                    # very edge) -> 1 (at the band's inner limit), dithered on the
                    # Bayer screen, so hard mask edges (wispy hair) stipple out
                    # into the ground instead of forming a blocky slab.
                    if (eg[r, c] / feather_px) <= _BAYER8[r & 7, c & 7]:
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
                # Random whole-cell leading phase so the first word starts at a
                # different x on each row -> word boundaries (and their gaps)
                # don't stack into vertical rivers. Capped so a run always keeps
                # room for at least the shortest word.
                pad = int(rng.integers(0, _RIVER_PHASE_CELLS + 1))
                pad = min(pad, max(0, (end - start) - shortest))
                glyphs: List[str] = [" "] * pad
                pos = start + pad
                first = True
                while True:
                    # Variable blank gap between words (not a fixed period) so the
                    # inter-word channels desync row-to-row.
                    gap = 0 if first else int(rng.integers(_WORD_GAP_MIN, _WORD_GAP_MAX + 1))
                    avail = end - pos - gap
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
                        glyphs.extend([" "] * gap)
                        pos += gap
                    glyphs.extend(chosen)
                    pos += len(chosen)
                    first = False
                if not glyphs or all(g == " " for g in glyphs):
                    continue
                spans = []
                # Per-WORD jitter (not per-glyph): letters in a word share one
                # offset (stay aligned/legible) while whole words scatter to break
                # the rigid grid. Each new word (after a space) gets a fresh one.
                wx = (rng.random() - 0.5) * cw * _JITTER_X
                wy = (rng.random() - 0.5) * rh * _JITTER_Y
                prev_space = False
                for k, ch in enumerate(glyphs):
                    if ch == " ":
                        if not prev_space:
                            wx = (rng.random() - 0.5) * cw * _JITTER_X
                            wy = (rng.random() - 0.5) * rh * _JITTER_Y
                        prev_space = True
                        continue
                    prev_space = False
                    cell = start + k
                    gx = bx0 + cell * cw + ox + wx
                    gy = baseline + wy
                    fill = fill_for(tdark_of(row[cell]), csub[r, cell] if photo_ink else None, gy / H)
                    spans.append(f'<tspan x="{gx:.1f}" y="{gy:.1f}" fill="{fill}">{esc(ch)}</tspan>')
                    # Record this glyph's cell so later fill passes don't overprint it.
                    oy0 = max(0, int(cy_nom - rh * 0.5)); oy1 = min(H, int(cy_nom + rh * 0.5))
                    oxa = max(0, int(bx0 + cell * cw)); oxb = min(W, int(bx0 + (cell + 1) * cw))
                    if oxb > oxa and oy1 > oy0:
                        occ[oy0:oy1, oxa:oxb] = True
                if not spans:
                    continue
                doc.add(
                    f'<text xml:space="preserve" font-family="{esc(_MONO_FAMILY)}" '
                    f'font-size="{font:.2f}" font-weight="{esc(cfg.font_weight)}">'
                    + "".join(spans) + "</text>"
                )
                runs.append(
                    TextRun(region=region, path_id=f"{region}{r}_{start}", path_d="",
                            text="".join(glyphs), font_size=round(font, 2), kind=kind)
                )

    # Body: big, legible words across the silhouette, outside the head region.
    emit(body_font, 0, 0, W, H, "body", "primary")
    # Mid: medium words in the ring around the face (the large->small step-down).
    if mid_ov:
        mx0 = min(cx - rx for cx, cy, rx, ry in mid_ov)
        my0 = min(cy - ry for cx, cy, rx, ry in mid_ov)
        mx1 = max(cx + rx for cx, cy, rx, ry in mid_ov)
        my1 = max(cy + ry for cx, cy, rx, ry in mid_ov)
        emit(mid_font, mx0, my0, mx1, my1, "mid", "primary")
    # Face: finer words to resolve nose/lips/features (exempt from the readable
    # min-font floor that governs the body, like the eye pass).
    if face_ov:
        fx0 = min(cx - rx for cx, cy, rx, ry in face_ov)
        fy0 = min(cy - ry for cx, cy, rx, ry in face_ov)
        fx1 = max(cx + rx for cx, cy, rx, ry in face_ov)
        fy1 = max(cy + ry for cx, cy, rx, ry in face_ov)
        emit(face_font, fx0, fy0, fx1, fy1, "face", "detail")

    # ---- Finer eye pass: resolve iris / lid / catchlight inside the eye
    # ellipses (which the main grid skipped). Half-size glyphs, marked "detail"
    # so they're exempt from the readable min-font floor that governs the body.
    if eyes:
        fe = eye_font
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
                    if inkf[cf]:
                        pxf = ex0 + (cf + 0.5) * ecw
                        # Iris annulus only (when the iris resolves); otherwise the
                        # legacy whole-eye fill so far/small faces still get eyes.
                        ok = in_eyes(pxf, cyf) and (not iris_cl or in_iris_annulus(pxf, cyf))
                        if not ok:
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
                    pad = int(rng.integers(0, _RIVER_PHASE_CELLS + 1))
                    pad = min(pad, max(0, (end - start) - shortest))
                    glyphs, pos, first = [" "] * pad, start + pad, True
                    while True:
                        gap = 0 if first else int(rng.integers(_WORD_GAP_MIN, _WORD_GAP_MAX + 1))
                        avail = end - pos - gap
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
                            glyphs.extend([" "] * gap)
                            pos += gap
                        glyphs.extend(chosen)
                        pos += len(chosen)
                        first = False
                    if not glyphs or all(g == " " for g in glyphs):
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

    # ---- Multi-scale gap fill -----------------------------------------------
    # The single-size tiers leave blank holes wherever a word couldn't fit the
    # available run (most visible at large sizes, where a big word needs many
    # cells). Pack progressively smaller words into those gaps -- skipping any
    # pixel a larger pass already inked -- so the portrait covers the full size
    # range with no empty black regions, finer where big words never fit.
    # Start the cascade just below the body size and step DOWN gently (each pass
    # ~78% of the previous), so the sizes form a smooth gradient instead of a few
    # stark jumps -- the intermediate sizes capture the detail that big->small
    # steps were skipping. More, smaller steps = smoother transition + more detail.
    fill_font = body_font * 0.78
    _fills = 0
    while gap_fill and fill_font >= 8.0 and _fills < gap_fill_passes:
        emit(fill_font, 0, 0, W, H, "fill", "fill")
        fill_font *= 0.78
        _fills += 1

    if not runs:
        warns.error("text", "no_runs", "Tonal fill produced no text (subject too bright or mask empty).")
    return doc.to_svg(), runs


def build_poster(
    an,
    text: str,
    cfg: RenderConfig,
    warns: WarningCollector,
    render_w: int = 2200,
    font_px: float = 20.0,
    ink: str = "gold_noir",
    remove_bg: bool = True,
    contrast: float = 2.8,
    pivot: float = 0.34,
    power: float = 1.0,
    level: float = 0.015,
) -> Tuple[str, List[TextRun]]:
    """Type-poster: the message in clean, fully-readable straight rows, with the
    portrait formed by per-letter brightness (tone), like a printed letter that
    *is* the face. Reuses the same tone pipeline and ink palettes as the mosaic.
    remove_bg keeps text only on the subject (clean ground); otherwise the whole
    frame is type and the surroundings fade into the ground."""
    words = [w for w in str(text).split() if w]
    if not words:
        warns.error("text", "no_words", "No message supplied for the poster.")
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
    sharp = _sharpen(gray)
    dark = _tone_field(sharp, mask, gamma=1.0, floor=0.0)
    dark = _auto_tone(dark, mset, 0.55, max_shift=0.18)
    dark = _balance_faces(dark, an, scale, mset)
    dark = _emphasize_features(dark, an, scale, mset)
    dark = _sharpen_eyes(dark, an, scale, mset)
    dark = _edge_separate(dark, sharp, mset, amount=0.40)
    # Full-frame background darkness (raw luminance) for the keep-background mode;
    # the enhanced subject tone overrides it inside the silhouette.
    lf = sharp.astype(np.float32) / 255.0
    lo, hi = np.percentile(lf, [2, 98])
    lf = np.clip((lf - lo) / max(1e-3, hi - lo), 0.0, 1.0)
    combined = (1.0 - lf) * 0.85  # background a touch lighter-toned so it recedes
    combined[mset] = dark[mset]
    if remove_bg:
        combined[~mset] = 0.0

    ground_hex, ink_hex = _POSTER.get(ink, ("#0a0a0c", "#f2ece0"))
    ground = _hex_to_rgb(ground_hex)
    photo_ink = ink == "photo"
    grad = _GRADIENTS.get(ink)
    ink_rgb = _hex_to_rgb(ink_hex) if (ink_hex and grad is None and not photo_ink) else None
    color_grid = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA) if photo_ink else None
    inv_level = max(1e-3, 1.0 - level)

    def tdark_of(tone: float) -> float:
        n = (tone - level) / inv_level
        n = (n - pivot) * contrast + pivot
        n = 1.0 if n > 1.0 else (0.0 if n < 0.0 else n)
        return n ** power

    def fill_for(t_dark: float, src, vfrac: float) -> str:
        # Brightness-positive on a dark ground: lit (t_dark->0) = bright ink,
        # shadow (t_dark->1) fades to the ground, so the lit face is the
        # brightest type and the photo reads faithfully.
        v = 1.0 - t_dark
        if grad is not None:
            tip = _grad_rgb(grad, vfrac)
        elif photo_ink and src is not None:
            tip = (int(src[2]), int(src[1]), int(src[0]))   # BGR -> RGB
        else:
            tip = ink_rgb
        cr = int(round(ground[0] + (tip[0] - ground[0]) * v))
        cg = int(round(ground[1] + (tip[1] - ground[1]) * v))
        cb = int(round(ground[2] + (tip[2] - ground[2]) * v))
        return f"#{cr:02x}{cg:02x}{cb:02x}"

    doc = SvgDoc(width=W, height=H, background=ground_hex)
    runs: List[TextRun] = []
    cw = font_px * _MONO_ADVANCE
    rh = font_px * 1.16
    cols = max(1, int(W / cw))
    rows = max(1, int(H / rh))
    rng = np.random.default_rng(7)
    nwords = max(1, len(words))
    for r in range(rows):
        # Start each row at a different point in the message AND a different
        # horizontal offset, so the repeating phrase doesn't stack into vertical
        # "rivers." (A sub-cell shift isn't enough -- the word boundaries realign.)
        wi = int(rng.integers(0, nwords))
        ox = float(rng.random()) * cw * 3.0
        y = (r + 0.9) * rh
        yi = min(H - 1, max(0, int(y - rh * 0.34)))
        spans: List[str] = []
        line: List[str] = []
        c = 0
        while c < cols:
            word = words[wi % len(words)]
            wl = len(word)
            if wl > cols:
                word, wl = word[:cols], cols
            if c + wl > cols:
                break
            drew = False
            for k, ch in enumerate(word):
                col = c + k
                gx = col * cw + ox
                xi = min(W - 1, max(0, int(gx + cw * 0.5)))
                if remove_bg and not mset[yi, xi]:
                    continue
                t_dark = tdark_of(float(combined[yi, xi]))
                fill = fill_for(t_dark, color_grid[yi, xi] if photo_ink else None, y / float(H))
                spans.append(f'<tspan x="{gx:.1f}" fill="{fill}">{esc(ch)}</tspan>')
                line.append(ch)
                drew = True
            c += wl + 1
            if drew:
                line.append(" ")
                wi += 1
        if spans:
            doc.add(
                f'<text xml:space="preserve" font-family="{esc(_MONO_FAMILY)}" '
                f'font-size="{font_px:.1f}" font-weight="{esc(cfg.font_weight)}" y="{y:.1f}">'
                + "".join(spans) + "</text>"
            )
            runs.append(TextRun(region="poster", path_id=f"poster{r}", path_d="",
                                text="".join(line).strip(), font_size=round(font_px, 2), kind="primary"))
    if not runs:
        warns.error("text", "no_runs", "Poster produced no text (mask empty or message too short).")
    return doc.to_svg(), runs


# ---------------------------------------------------------------------------
# Layered renderer: the photo shows THROUGH the typography (text used as a
# mask), giving real photographic richness inside each letter. Shared by both
# user-facing styles -- "Words" (the mosaic layout) and "Message" (poster rows).
# Output is a raster (the richness is photographic), composited in code because
# CairoSVG does not honour SVG masks.
# ---------------------------------------------------------------------------

def _tint_photo(an, W: int, H: int, ink: str, remove_bg: bool, light: bool = False,
                custom=None) -> np.ndarray:
    """Processed, ink-tinted photo that shows through the text mask.

    Dark ground (default): brightness-positive -- lit areas are the bright ink,
    shadows fall to the dark ground. Light ground: dark ink on a light paper --
    shadows/features are the dark ink, highlights melt into the paper (an
    engraving look). Colours are more muted/inky on light paper than on dark."""
    gray = _sharpen(cv2.resize(an.img.gray, (W, H), interpolation=cv2.INTER_CUBIC)).astype(np.float32) / 255.0
    lo, hi = np.percentile(gray, [2, 99])
    lum = np.clip((gray - lo) / max(1e-3, hi - lo), 0.0, 1.0)
    # Midtone-punch S-curve. A steep linear contrast stretch clips features to
    # one flat ink; a smoothstep deepens shadows and lifts highlights while
    # asymptoting smoothly at 0/1, so eye sockets sink and lit planes pop
    # without losing tonal separation between adjacent features.
    lum = lum * lum * (3.0 - 2.0 * lum)
    lum = np.clip((lum - 0.5) * 1.25 + 0.5, 0.0, 1.0)
    # Anchor the likeness in the recognition features. The layered renderer
    # builds tone purely from photo luminance, so without this the eyes/brows/
    # lips read no stronger than skin. Apply the tonal-path emphasis on a
    # darkness field (1 = dark feature) so eye sockets/iris deepen and lit
    # sclera pops, then convert back to the brightness `lum`.
    smask = an.silhouette.mask
    if smask.shape[:2] != (H, W):
        smask = cv2.resize(smask, (W, H), interpolation=cv2.INTER_NEAREST)
    mset = (smask > 127).astype(np.float32)
    fscale = W / float(an.img.gray.shape[1])
    darkf = 1.0 - lum
    darkf = _emphasize_features(darkf, an, fscale, mset, gamma=0.38)
    darkf = _sharpen_eyes(darkf, an, fscale, mset, contrast=1.6, catchlight="soft")
    lum = np.clip(1.0 - darkf, 0.0, 1.0)
    if light:
        ck = _CALLIGRAM.get(ink, ("#15202b", "#ffffff"))   # (dark ink, light paper)
        ground_hex, ink_hex = ck[1], ck[0]
    elif ink == "custom" and custom:
        ground_hex, ink_hex = custom                       # (ground, ink) from a user-picked colour
    else:
        ground_hex, ink_hex = _POSTER.get(ink, ("#0a0a0c", "#f2ece0"))
    ground = np.array(_hex_to_rgb(ground_hex), dtype=np.float32)
    grad = _GRADIENTS.get(ink)
    if grad is not None:
        col = np.array([_grad_rgb(grad, float(y)) for y in np.linspace(0, 1, H)], dtype=np.float32)
        if not light:
            col = col + (255.0 - col) * 0.22    # lift toward white so hues stay luminous on the dark ground
        tip = col[:, None, :]
    elif ink in ("photo", "photo_paper"):
        bgr = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        rgb = bgr[..., ::-1]
        g = rgb.mean(axis=2, keepdims=True)
        tip = np.clip(g + (rgb - g) * 1.12, 0.0, 255.0)     # keep close to the photo's own saturation (natural skin, not cartoonish)
        if ink == "photo_paper":
            tip = tip * 0.60        # darken the hue so it reads as ink on white paper
    else:
        tip = np.array(_hex_to_rgb(ink_hex), dtype=np.float32)
    # Dark ground: brightness drives ink. Light paper: darkness drives ink
    # (engraving). In light mode push highlights hard toward white (gamma<1) so
    # the lit face stays white paper even under dense gap-fill -- only shadows and
    # features pick up ink, so the portrait reads instead of becoming a gray mass.
    if ink == "photo_paper":
        # Photo on PAPER: darkness drives the ink (features/shadows bold) with a
        # 0.30 floor so the brightest skin + grey hair still carry a soft tint
        # rather than vanishing into the white -- a coloured engraving that reads
        # whole. (Distinct from the mono light-engraving formula below.)
        v = np.clip(0.30 + 0.80 * (1.0 - lum), 0.0, 1.0)
        # Eye detail is injected into the ink array just below (after `out`), so it
        # rides the SAME word mask in compose -> a photo-true eye made of type, never
        # a smooth patch. Teeth stay the natural engraving (lips dark, teeth light).
    elif not light:
        v = lum
    else:
        # Engraving: keep the lightest areas as faint near-white (small 0.08
        # floor, so the brightest skin isn't pure blank paper) but push mids and
        # shadows DARKER (1.25 gain) so the visible words are bold, not washed
        # out. With the sparser light-mode fill this reads as a crisp engraving
        # with white space, not a faint gray jumble.
        v = np.clip(0.08 + 1.25 * (1.0 - np.clip(lum ** 0.8, 0.0, 1.0)), 0.0, 1.0)
    out = ground + (tip - ground) * v[..., None]
    if ink == "photo_paper":
        # Eye STRUCTURE forced into the ink array (pre-mask), so it renders THROUGH
        # the words (typographic, never a smooth patch). The photo's own eye is too
        # light to read on paper for light-eyed/grey subjects, so we lay an explicit
        # light sclera, a dark iris in the person's HUE, a near-black pupil, and dark
        # upper+lower lids -- the gradation an eye needs. compose() then textures it
        # with the type. Coarse at large word sizes (eye = a few words), finer as the
        # word size drops.
        try:
            _ee = _eye_ellipses(an, fscale); _ic = _iris_circles(an, fscale)
            if _ee:
                _be = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)[..., ::-1]
                eye_r = float(np.mean([e[2] for e in _ee]))
                def _stamp(mask, colour, strength):
                    nonlocal out
                    m = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.4, eye_r * 0.06))[..., None] * strength
                    out = out * (1.0 - m) + np.asarray(colour, np.float32) * m
                # 1) Sclera: a light base across the whole eye (the eye-white).
                scl = np.zeros((H, W), np.float32)
                for ex, ey, rx, ry in _ee:
                    cv2.ellipse(scl, (int(round(ex)), int(round(ey))), (int(round(rx)), int(round(ry * 0.95))), 0, 0, 360, 1.0, -1)
                _stamp(scl, (236.0, 233.0, 228.0), 0.92)
                # 2) Iris (the person's hue, forced dark) + 3) near-black pupil.
                for cx, cy, r in _ic:
                    irm = np.zeros((H, W), np.float32)
                    cv2.circle(irm, (int(round(cx)), int(round(cy))), max(2, int(round(r * 0.92))), 1.0, -1, cv2.LINE_AA)
                    hue = _be[max(0, int(cy)) % H, max(0, int(cx)) % W]
                    _stamp(irm, np.clip(hue * 0.34, 18.0, 120.0), 0.95)
                    pum = np.zeros((H, W), np.float32)
                    cv2.circle(pum, (int(round(cx)), int(round(cy))), max(1, int(round(r * 0.44))), 1.0, -1, cv2.LINE_AA)
                    _stamp(pum, (20.0, 19.0, 18.0), 0.95)
                # 4) Lids: dark upper + lower arcs frame the eye.
                lid = np.zeros((H, W), np.float32)
                for ex, ey, rx, ry in _ee:
                    cv2.ellipse(lid, (int(round(ex)), int(round(ey))), (int(round(rx)), int(round(ry))),
                                0, 180, 360, 1.0, max(2, int(round(ry * 0.26))), cv2.LINE_AA)
                    cv2.ellipse(lid, (int(round(ex)), int(round(ey))), (int(round(rx * 0.94)), int(round(ry * 0.92))),
                                0, 6, 174, 1.0, max(1, int(round(ry * 0.16))), cv2.LINE_AA)
                _stamp(lid, (30.0, 28.0, 26.0), 0.88)
        except Exception:
            pass
    # Living eyes: inside each iris, carry the person's TRUE eye colour in the
    # glyphs (sampled, never invented -- _iris_tint gates on the photo actually
    # holding the colour; gated-off photos render byte-identical). The tonal
    # field `v` is kept, so iris structure and the catchlight stay; only the
    # ink hue changes, feathered at the iris edge. Dark ground only: the dormant
    # light/engraving path uses dark ink on paper, where a brightness-lifted
    # tint would read wrong.
    # Selective colour: bring the EYES (whole eyeball) and TEETH back to their TRUE
    # photo colours -- the exact natural rendering "Original" (photo ink) produces,
    # confined to those regions and feathered -- so they pop against the tinted
    # face (living eyes, a natural smile). Skipped for the photo ink (already fully
    # natural), in light/engraving mode, and when _SELCOLOR is 0.
    if not light and ink not in ("photo", "photo_paper") and _SELCOLOR > 0:   # all colour inks incl. Custom; not Original
        sel = np.zeros((H, W), np.float32)
        ircs = _iris_circles(an, fscale)
        if ircs:
            # Colour the iris only, SMALLER than the iris radius so it can never spill
            # past the eyeball onto the lower lid / under-eye skin (the pink patches).
            for (cx, cy, r) in ircs:
                cv2.circle(sel, (int(round(cx)), int(round(cy))), max(2, int(round(r * 0.85))), 1.0, -1)
        else:
            # No iris landmarks: fall back to a TINY central-eyeball ellipse.
            for (cx, cy, rx, ry) in _eye_ellipses(an, fscale):
                cv2.ellipse(sel, (int(round(cx)), int(round(cy))),
                            (max(2, int(round(rx * 0.45))), max(2, int(round(ry * 0.40)))), 0, 0, 360, 1.0, -1)
        if an.landmarks is not None:
            tmask = _teeth_mask(an.landmarks.points * fscale, H, W)   # inner mouth (None if closed)
            if tmask is not None:
                # Teeth are bright AND neutral; the lip is bright but SATURATED (pink).
                # Gate on both so only teeth take colour, never the lip.
                _b2 = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
                _sat = (_b2.max(2) - _b2.min(2)) / (_b2.max(2) + 1e-3)
                bright = np.clip((v - 0.55) / 0.30, 0.0, 1.0)
                lowsat = np.clip((0.30 - _sat) / 0.18, 0.0, 1.0)
                sel = np.maximum(sel, tmask * bright * lowsat)
        if float(sel.max()) > 0.0:
            # Tight feather so the colour stays on the eyeball/teeth, not bleeding
            # onto surrounding skin.
            sel = np.clip(cv2.GaussianBlur(sel, (0, 0), max(1.0, W * 0.0016)), 0.0, 1.0) * float(_SELCOLOR)
            # Render these pixels as the photo ink would, but on a NEUTRAL dark
            # ground (not the tinted ink ground) so the hue matches the source
            # instead of picking up the ink colour in the shadows. Mild saturation.
            ngd = np.array(_hex_to_rgb("#0a0a0c"), dtype=np.float32)
            bgr2 = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            rgb2 = bgr2[..., ::-1]
            gg = rgb2.mean(axis=2, keepdims=True)
            nat_tip = np.clip(gg + (rgb2 - gg) * 1.2, 0.0, 255.0)
            nat = ngd + (nat_tip - ngd) * v[..., None]
            out = out * (1.0 - sel[..., None]) + nat * sel[..., None]
    if remove_bg:
        m = an.silhouette.mask
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out[m <= 127] = ground
    return out.clip(0, 255).astype(np.uint8)


def _mask_svg(colored_svg: str) -> str:
    """Coloured layout SVG -> white-on-black text mask for clean CairoSVG text."""
    s = re.sub(r'fill="#[0-9a-fA-F]{6}"', 'fill="#ffffff"', colored_svg)
    s = re.sub(r'(<rect x="0" y="0"[^>]*?)fill="#ffffff"', r'\1fill="#000000"', s, count=1)
    return s


def _ground_hex(ink: str, light: bool, custom=None) -> str:
    if light:
        return _CALLIGRAM.get(ink, ("#15202b", "#ffffff"))[1]
    if ink == "custom" and custom:
        return custom[0]
    return _POSTER.get(ink, ("#0a0a0c", None))[0]


def render_layered_png(an, text: str, style: str, cfg: RenderConfig, warns: WarningCollector,
                       ink: str = "mono", remove_bg: bool = True, light: bool = False,
                       out_width: int = 1400, render_w: int = 2200, tone_density: float = 0.6,
                       uppercase: bool = True, print_aspect: float = _PRINT_ASPECT, custom=None):
    """Layered portrait -> PNG bytes. style='message' = poster rows; else Words
    (mosaic) layout. Returns (png_bytes, runs, ground_hex, mask_svg)."""
    # --- Dedicated light/engraving renderer (Words style) --------------------
    # On white paper, compositing the photo through a text mask washes out or
    # jumbles. Instead, draw the tonally-shaded words DIRECTLY on the paper: each
    # glyph is coloured by the tone it sits on (faint near-white in highlights,
    # bold/dark in shadows), which reads as a clean engraving. This is a separate
    # path from the dark/photo-composite renderer. No mask is returned, so the
    # paid high-res download re-renders through here.
    if light and style != "message":
        from .portrait import build_portrait
        from .raster import svg_to_png_bytes
        ewords = [w for w in re.split(r"[\s,]+", text) if w]
        eng_ink = ink if ink in ("navy", "sepia", "burgundy", "forest", "mono") else "navy"
        eres = build_portrait(an, ewords, cfg, warns, uppercase=uppercase, ink=eng_ink,
                              render_w=render_w, gap_fill=True, gap_fill_passes=12)
        eground = _PALETTES.get(eng_ink, _PALETTES["mono"])[2]
        if not eres.svg:
            return b"", eres.runs, eground, ""
        ss = _eff_supersample(out_width)
        epng = svg_to_png_bytes(eres.svg, output_width=out_width * ss)
        from PIL import Image
        eimg = Image.open(io.BytesIO(epng)).convert("RGB")
        if ss > 1:                   # supersampled -> Lanczos down to final size
            eimg = _lanczos_down(eimg, out_width)
        # Same print canvas as the composite path, padded with paper.
        eimg = Image.fromarray(_fit_print_canvas(np.asarray(eimg), _hex_to_rgb(eground), print_aspect))
        ebuf = io.BytesIO(); eimg.save(ebuf, format="PNG"); epng = ebuf.getvalue()
        return epng, eres.runs, eground, ""

    # The layout only needs glyph POSITIONS (we whiten them into a mask); the
    # colour/ink is applied separately by _tint_photo. Build the layout with a
    # neutral ink so the ink choice never touches the layout (and we avoid the
    # mosaic's photo-ink colour path, which isn't needed here).
    if style == "message":
        # Message/prose has no manual size knob: build_poster repeats the message
        # to fill the face, so size never affects fit — only legibility vs density.
        # Auto-size from message length: a short note reads bold and clear; a long
        # letter goes finer so more of it is visible before the phrase cycles.
        msg_font = _auto_message_font(text)
        colored, runs = build_poster(an, text, cfg, warns, render_w=render_w,
                                     font_px=msg_font, ink="mono", remove_bg=remove_bg)
    else:
        from .portrait import build_portrait
        words = [w for w in re.split(r"[\s,]+", text) if w]
        # Density thinning opens deep shadows to the GROUND so the face emerges.
        # That only reads on a dark ground; on light paper a thinned shadow cell
        # becomes white (sparse, invisible non-face areas), so disable it in light
        # mode and keep full dark-ink-on-paper shadows (the engraving look).
        eff_density = 0.0 if light else tone_density
        # Paper engraving needs FINER words than the dark hero: features like eyes
        # span only a word or two at the larger sizes, too coarse to resolve an
        # iris/pupil/lids. Pin photo_paper to a small min-font (more, smaller words
        # = the resolution the eyes need). Save/restore so the shared cfg + recipe
        # are untouched; the download re-renders through here and pins identically.
        _orig_mf = cfg.min_font_px
        if ink == "photo_paper":
            cfg.min_font_px = min(float(cfg.min_font_px), 28.0)
        # Multi-scale gap-fill packs the full size range into the silhouette in
        # both modes. In light/engraving mode the lit face is pushed to white
        # (see _tint_photo), so the dense fill there renders as white paper and
        # only shadows/features take ink -- full range without a gray jumble.
        # Dark ground wants the full dense fill (the hero look). Light/engraving
        # wants white space, so use far fewer fill passes -- sparser + cleaner.
        res = build_portrait(an, words, cfg, warns, uppercase=uppercase, ink="mono",
                             render_w=render_w, tone_density=eff_density,
                             gap_fill=True, gap_fill_passes=12)
        cfg.min_font_px = _orig_mf
        colored, runs = res.svg, res.runs
    ground_hex = _ground_hex(ink, light, custom)
    if not colored:
        return b"", runs, ground_hex, ""
    mask_svg = _mask_svg(colored)
    boost = _MSG_BOOST if style == "message" else 0.0
    png = compose_layered(mask_svg, an, ink, remove_bg, out_width, light=light, boost=boost,
                          print_aspect=print_aspect, custom=custom)
    return png, runs, ground_hex, mask_svg


def compose_layered(mask_svg: str, an, ink: str, remove_bg: bool, out_width: int,
                    light: bool = False, boost: float = 0.0,
                    print_aspect: float = _PRINT_ASPECT, custom=None) -> bytes:
    """Composite the tinted photo through a prebuilt white-text mask SVG. Reused
    at download from the stored mask, so the costly layout build runs only once
    (at render), not again per sale. `boost` (>0) gamma-lifts the finished
    composite's shadows/midtones — used to brighten Message/prose renders."""
    from PIL import Image
    from .raster import svg_to_png_bytes
    if not mask_svg:
        return b""
    ss = _eff_supersample(out_width)
    render_w = out_width * ss
    mpng = svg_to_png_bytes(mask_svg, output_width=render_w)
    mask = np.asarray(Image.open(io.BytesIO(mpng)).convert("L"))
    H, W = mask.shape[:2]
    photo = _tint_photo(an, W, H, ink, remove_bg, light=light, custom=custom).astype(np.float32)
    ground = np.array(_hex_to_rgb(_ground_hex(ink, light, custom)), dtype=np.float32)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    out = (ground + (photo - ground) * m).clip(0, 255).astype(np.uint8)
    # Catchlight: a SPECULAR white glint at the eye's real brightest pixel inside
    # each iris, painted over the finished composite (above the text mask) so the
    # eye looks back. Always white -- the lightest thing on the face -- never ink-
    # or iris-coloured. Dark grounds only; before the pad so coords hold.
    # photo_paper takes its OWN white-ground eye treatment below -- the dark-ground
    # catchlight/limbal here darkens toward the ground, which inverts on white.
    if not light and ink != "photo_paper":
        # Sclera wash: the whites of the eyes read LIGHT (carrying no typography),
        # painted as a soft warm-white modulated by the photo's own shading so the
        # eye keeps its natural gradient -- not a flat disc, dimmer than glyphs.
        fsc0 = W / float(an.img.gray.shape[1])
        eyes_e = _eye_ellipses(an, fsc0)
        iris_c = _iris_circles(an, fsc0)
        # Dark-lens (sunglasses) guard -- mirrors the Lifelike path: a tinted lens has no
        # bright sclera (p90 < 95 vs 134-193 for real eyes), so suppress the fabricated /
        # see-through eye and paint an opaque lens below. Gated by env TYPO_DARKLENS (ON).
        _dl_on = os.environ.get("TYPO_DARKLENS", "1").strip().lower() not in ("0", "false", "off", "no", "")
        _lens_eyes = []
        # Dark-lens (sunglasses) guard, PER FACE: a face reads as tinted lenses only when
        # BOTH its eyes are dark (max sclera p90 < 95; real sunglasses measure 39-73). So a
        # single shadowed/side-lit real eye must NOT suppress the face, and one person's
        # shades must NOT suppress everyone else's eyes in a group. iris_c/eyes_e are
        # 2-per-face in the same order; only shaded faces are suppressed + lens-filled.
        if _dl_on and iris_c and len(iris_c) == len(eyes_e):
            _glum = (photo[..., 0] * 0.299 + photo[..., 1] * 0.587 + photo[..., 2] * 0.114)
            def _sp90(_c):
                _icx, _icy, _irr = _c
                _y0, _y1 = max(0, int(_icy - _irr * 2.4)), int(_icy + _irr * 2.4)
                _x0, _x1 = max(0, int(_icx - _irr * 2.4)), int(_icx + _irr * 2.4)
                _reg = _glum[_y0:_y1, _x0:_x1]
                return float(np.percentile(_reg, 90)) if _reg.size >= 16 else None
            _keep_e, _keep_i = [], []
            for _k in range(0, len(iris_c), 2):
                _pair_i, _pair_e = iris_c[_k:_k + 2], eyes_e[_k:_k + 2]
                _ps = [p for p in (_sp90(_c) for _c in _pair_i) if p is not None]
                if len(_ps) >= 2 and max(_ps) < 95.0:
                    _lens_eyes.extend(_pair_e)          # shaded face -> opaque lens fill
                else:
                    _keep_i.extend(_pair_i)
                    _keep_e.extend(_pair_e)
            eyes_e, iris_c = _keep_e, _keep_i
        _dark_lens = bool(_lens_eyes)
        # Eye-white + teeth fill. In Photo ink the photo's OWN pixels can carry a
        # warm cast (warm light + warm ink), so take mostly LUMINANCE + a trace of
        # colour -> natural neutral whites. Other inks keep the tinted photo (it
        # already matches the duotone face), so they're left as-is.
        if ink == "photo":
            gp = (photo[..., 0] * 0.299 + photo[..., 1] * 0.587 + photo[..., 2] * 0.114)[..., None]
            eye_fill = photo * 0.15 + gp * 0.85
        else:
            eye_fill = photo
        if eyes_e:
            ir_mean = float(np.mean([r for _, _, r in iris_c])) if iris_c else 1.0
            # Limbal ring: dark rim at the iris edge -- the cue that reads as a
            # real iris, not a flat disc. Darken the thin annulus toward the ground.
            if iris_c:
                lm = np.zeros((H, W), np.float32)
                for icx, icy, irr in iris_c:
                    cv2.circle(lm, (int(round(icx)), int(round(icy))), int(round(irr)), 1.0, -1, cv2.LINE_AA)
                    cv2.circle(lm, (int(round(icx)), int(round(icy))), int(round(irr * 0.80)), 0.0, -1, cv2.LINE_AA)
                lm = cv2.GaussianBlur(lm, (0, 0), sigmaX=max(1.0, ir_mean * 0.05))[..., None]
                out = (out.astype(np.float32) * (1.0 - 0.60 * lm)
                       + ground * (0.60 * lm)).clip(0, 255).astype(np.uint8)
            # NO synthetic sclera is painted. A bright floored white disc (~200) drawn
            # from _eye_ellipses geometry pokes out BEYOND the hull-based real-eye overlay
            # below, and that exposed rim BLOOMS under the vibrance pass -- it is the
            # "glow" on Mosaic/Passage. (Displacement doesn't glow because it uses the
            # same geometry for both.) The real-eye overlay supplies the sclera; any
            # uncovered eye-corner pixels stay as the word-tone, which never glows.
        # Teeth carry NO typography. Where the mouth is open, clear the glyphs from
        # the inner mouth and let the photo's OWN pixels show through -- the same
        # tinted source the rest of the portrait is built from, so the teeth keep
        # their real ivory and shading and match the face, never an invented white.
        # A closed mouth yields no mask, so it is left exactly as composed.
        tm = _teeth_mask(_faces_of(an)[0].points * fsc0 if _faces_of(an) else None, H, W)
        if tm is not None:
            # The mouth is a small source region scaled up, so the bare photo reads
            # soft against the crisp type. Unsharp-mask the fill so the tooth edges
            # and gum line are defined -- crisp teeth, still the photo's own tone.
            ph_sharp = cv2.addWeighted(eye_fill, 1.7, cv2.GaussianBlur(eye_fill, (0, 0), 1.4), -0.7, 0.0)
            tw = (tm * 0.92)[..., None]
            out = (out.astype(np.float32) * (1.0 - tw) + ph_sharp * tw).clip(0, 255).astype(np.uint8)
        # NO synthetic catchlight either -- a 250-white glint outside the overlay blooms
        # under vibrance. The real-eye overlay below carries the photo's own catchlight.
        # Real-eye overlay for EVERY ink -- the photo's own eye, which never glows (it
        # overrides the bright synthetic sclera/catchlight above). For the Photo ink it
        # stays full colour; for the tinted inks (Noir/Sepia/Navy/Sage) it is then
        # DESATURATED into the ink's palette so a full-colour eye doesn't clash with the
        # tinted face -- realistic structure, no glow, no colour clash.
        if eyes_e:
            bgr_eye = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            # Photographic eye overlay for EVERY subject (primary + all secondary), so a
            # group portrait renders identical real eyes on each person -- not just the
            # largest face. Each face's eyes are disjoint; keep the strongest alpha per px.
            eye_bgr = bgr_eye.copy()
            eye_a = np.zeros((H, W), np.float32)
            for _f in _faces_of(an):
                _ebgr, _ea = _photo_eye_overlay(bgr_eye, _f.points * fsc0, (_EYE_L, _EYE_R), H, W)
                _take = _ea > eye_a
                eye_a = np.where(_take, _ea, eye_a)
                eye_bgr[_take] = _ebgr[_take]
            # THE Mosaic/Passage "blue glowing eyes" -- the actual root cause (reproduced &
            # measured on the user's photo): a CHANNEL SWAP. `_photo_eye_overlay` returns the
            # eye in its input order (BGR, from an.img.bgr), but in this compose path `out`/
            # `photo`/`ground` are RGB (ground = _hex_to_rgb; final Image.fromarray = RGB).
            # Compositing the BGR eye into the RGB canvas flips R<->B, so a BROWN iris
            # (R71/B37) rendered BLUE (R38/B50). Displacement never showed it because it works
            # in BGR throughout. Fix = convert the overlay to RGB before any of it touches
            # `out`. (Verified: Jesus iris R38/B50 -> R49/B38; hero-before unchanged.)
            eye_bgr = eye_bgr[..., ::-1].copy()              # BGR -> RGB to match `out`
            _elum = (eye_bgr[..., 0] * 0.299 + eye_bgr[..., 1] * 0.587 + eye_bgr[..., 2] * 0.114)
            _esc = np.minimum(1.0, _EYE_LUMA_CAP / np.maximum(_elum, 1e-3))[..., None]
            eye_bgr = eye_bgr * _esc                          # trim only blow-out (cap matches Lifelike's ~238)
            _ir = float(np.mean([r for _, _, r in iris_c])) if iris_c else max(2.0, eyes_e[0][2] * 0.5)
            _kr = max(1, int(round(_ir * 0.85)))
            _base = cv2.dilate(eye_a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_kr * 2 + 1, _kr * 2 + 1)))
            _base = cv2.GaussianBlur(_base, (0, 0), sigmaX=max(1.0, _ir * 0.55))[..., None]
            # Socket = the photo's own orbital skin (mid-toned), not the dark ground, so the
            # bright eye transitions into a face -> no high-contrast halo.
            out = (out.astype(np.float32) * (1.0 - 0.70 * _base) + photo.astype(np.float32) * (0.70 * _base)).clip(0, 255).astype(np.uint8)
            a3 = eye_a[..., None]   # FULL alpha -- no bright-word bleed through the eyeball
            out = (out.astype(np.float32) * (1.0 - a3) + eye_bgr * a3).clip(0, 255).astype(np.uint8)
            # Light desaturation so the eye sits in the ink's palette: tinted inks pull most
            # of the way to mono (a full-colour eye would clash with the tinted face), while
            # Photo/Original keeps the iris its true (now correctly-coloured) hue with only a
            # whisper of taming. (The old heavy pull was masking the BGR/RGB swap above; with
            # the swap fixed the eye is the right colour, so it no longer needs hiding.)
            _desat = 0.12 if ink == "photo" else 0.78
            of = out.astype(np.float32)
            lum = (of[..., 0] * 0.299 + of[..., 1] * 0.587 + of[..., 2] * 0.114)[..., None]   # RGB luma
            grayed = of * (1.0 - _desat) + lum * _desat
            em3 = eye_a[..., None]
            out = (of * (1.0 - em3) + grayed * em3).clip(0, 255).astype(np.uint8)
        # Opaque dark lens: with the eye suppressed above, paint the lens region toward the
        # ground so it reads as a solid dark lens instead of the word-face showing through.
        if _dark_lens and _lens_eyes:
            _lm = np.zeros((H, W), np.float32)
            for _ex, _ey, _erx, _ery in _lens_eyes:
                cv2.circle(_lm, (int(round(_ex)), int(round(_ey))),
                           max(2, int(round(max(_erx, _ery) * 1.9))), 1.0, -1, cv2.LINE_AA)
            _mr = float(np.mean([max(rx, ry) for _, _, rx, ry in _lens_eyes]))
            _lm = np.clip(cv2.GaussianBlur(_lm, (0, 0), sigmaX=max(1.0, _mr * 0.30)), 0, 1)[..., None]
            out = (out.astype(np.float32) * (1.0 - 0.90 * _lm) + ground * (0.90 * _lm)).clip(0, 255).astype(np.uint8)
    # (photo_paper paints NO compose-level eye/teeth patch -- that read as a sticker
    # glued on the typography. Its eyes + smile are formed by the WORDS themselves,
    # deepened in the tonal field in _tint_photo so the features emerge from type.)
    # Standard print canvas (4:5 = 16x20): pad with the ground BEFORE the boost
    # and vibrance passes so the band is processed identically to the interior
    # ground (no visible seam).
    # Shield the photographic eye from the vibrance pass below. Pad the eye-overlay
    # alpha onto the print canvas with the SAME call as `out`, so the mask stays
    # pixel-aligned through any centre/downscale/pad.
    _eye_guard = None
    if eyes_e:
        # DILATE the eye alpha before using it as the vibrance guard. Vibrance's clarity
        # term paints a bright HALO just OUTSIDE the bright eye (along the eye<->word edge);
        # the bare overlay alpha feathers to 0 exactly there, so the halo leaked through and
        # WAS the residual bloom. A solid guard ring over the eye edge stops clarity from
        # forming a halo at all.
        _gr = int(round(max(2.0, (np.mean([r for _, _, r in iris_c]) if iris_c else eyes_e[0][2]) * 0.9)))
        _eg = cv2.dilate((eye_a > 0.05).astype(np.float32),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_gr * 2 + 1, _gr * 2 + 1)))
        _eg = np.maximum(eye_a, cv2.GaussianBlur(_eg, (0, 0), sigmaX=max(1.0, _gr * 0.5)))
        _gm = _fit_print_canvas(np.repeat((_eg[..., None] * 255.0).clip(0, 255).astype(np.uint8), 3, axis=2),
                                (0, 0, 0), print_aspect)
        _eye_guard = _gm[..., 0].astype(np.float32) / 255.0
    # Background lightening: render the image background LIGHTER than the subject's
    # ground, so the worded subject (on its dark navy/black ground) sits against a
    # lighter backdrop. The subject's OWN ground is untouched -- only pixels outside
    # the silhouette are lifted. Gated by env TYPO_BG_LIGHTEN (0..1; 0 = off, current
    # behaviour: background == ground). The lift is a fraction of the way from the
    # ground toward white, so it tracks whatever ink/ground is in use.
    try:
        _bg_lift = float(os.environ.get("TYPO_BG_LIGHTEN", "0"))
    except ValueError:
        _bg_lift = 0.0
    _bg_lift = min(max(_bg_lift, 0.0), 1.0)
    _pad_col = ground
    if remove_bg and not light and _bg_lift > 0.0:
        _silm = an.silhouette.mask
        if _silm.shape[:2] != (H, W):
            _silm = cv2.resize(_silm, (W, H), interpolation=cv2.INTER_NEAREST)
        _bg_light = ground + (255.0 - ground) * _bg_lift
        # Soft boundary so the subject-ground -> lighter-background transition is a
        # gentle halo, not a jagged hard cut. Feather stays outside the subject.
        _outside = cv2.GaussianBlur((_silm <= 127).astype(np.float32), (0, 0),
                                    sigmaX=max(1.0, W * 0.002))[..., None]
        out = (out.astype(np.float32) * (1.0 - _outside)
               + _bg_light * _outside).clip(0, 255).astype(np.uint8)
        _pad_col = _bg_light                       # print band matches the lighter background
    out = _fit_print_canvas(out, _pad_col, print_aspect)
    if boost and boost > 0.0:        # lift dark Message renders (shadows/midtones)
        f = (out.astype(np.float32) / 255.0) ** (1.0 / (1.0 + float(boost)))
        out = (f * 255.0).clip(0, 255).astype(np.uint8)
    if not light:                    # gentle life (clarity); restrained so colour stays natural, not cartoonish
        from .preprocess import apply_vibrance
        _pre = out
        out = apply_vibrance(out, strength=0.34, bgr=False)
        if _eye_guard is not None:
            # Keep the eye's PRE-vibrance values. Vibrance's clarity term amplifies the
            # bright sclera/catchlight against the dark word-face, and its saturation lift
            # over-vivifies the iris -- together that is the Mosaic/Passage eye "glow" that
            # only shows on the full-fidelity (cairosvg) render. The real-eye overlay is
            # already faithful and sharp; excluding it from vibrance gives Lifelike clarity
            # with no glow. (Displacement avoids it instead by keeping the eye below blow-out.)
            # The guard is already dilated to cover the eye edge + halo zone, so it needs
            # no extra boost (that would over-exclude the face from the clarity pass).
            _m = np.clip(_eye_guard, 0.0, 1.0)[..., None]
            out = (out.astype(np.float32) * (1.0 - _m) + _pre.astype(np.float32) * _m).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    if ss > 1:                       # supersampled -> Lanczos down to final size
        img = _lanczos_down(img, out_width)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
