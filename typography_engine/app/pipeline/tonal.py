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
            rx = (float(ep[:, 0].max() - ep[:, 0].min()) / 2.0) * 1.30
            ry = (float(ep[:, 1].max() - ep[:, 1].min()) / 2.0) * 1.55
            if rx >= 2.0 and ry >= 2.0:
                out.append((cx, cy, rx, ry))
    return out


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
    body_font = float(min(cfg.max_font_px, max(cfg.min_font_px, cfg.min_font_px * 2.2)))
    mid_font = float(min(cfg.max_font_px, max(cfg.min_font_px, cfg.min_font_px * 1.45)))
    face_font = float(max(8.0, cfg.min_font_px * 1.0))
    eye_font = float(max(6.0, face_font * 0.62))

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
        rh = font * 0.80
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
    while gap_fill and fill_font >= 8.0 and _fills < 12:
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

def _tint_photo(an, W: int, H: int, ink: str, remove_bg: bool, light: bool = False) -> np.ndarray:
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
    else:
        ground_hex, ink_hex = _POSTER.get(ink, ("#0a0a0c", "#f2ece0"))
    ground = np.array(_hex_to_rgb(ground_hex), dtype=np.float32)
    grad = _GRADIENTS.get(ink)
    if grad is not None:
        col = np.array([_grad_rgb(grad, float(y)) for y in np.linspace(0, 1, H)], dtype=np.float32)
        if not light:
            col = col + (255.0 - col) * 0.22    # lift toward white so hues stay luminous on the dark ground
        tip = col[:, None, :]
    elif ink == "photo":
        bgr = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        rgb = bgr[..., ::-1]
        g = rgb.mean(axis=2, keepdims=True)
        tip = np.clip(g + (rgb - g) * 1.5, 0.0, 255.0)      # +50% saturation
    else:
        tip = np.array(_hex_to_rgb(ink_hex), dtype=np.float32)
    # Dark ground: brightness drives ink. Light paper: darkness drives ink
    # (engraving). In light mode push highlights hard toward white (gamma<1) so
    # the lit face stays white paper even under dense gap-fill -- only shadows and
    # features pick up ink, so the portrait reads instead of becoming a gray mass.
    if not light:
        v = lum
    else:
        # gamma<1 keeps the lit face light; the small 0.12 floor stops the
        # BRIGHTEST areas from going fully white (= blank paper). The lightest
        # skin keeps a faint gray so the form/edges read and the transitional
        # fill words stay visible there, while darks still reach full ink.
        v = 0.12 + 0.88 * (1.0 - np.clip(lum ** 0.55, 0.0, 1.0))
    out = ground + (tip - ground) * v[..., None]
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


def _ground_hex(ink: str, light: bool) -> str:
    if light:
        return _CALLIGRAM.get(ink, ("#15202b", "#ffffff"))[1]
    return _POSTER.get(ink, ("#0a0a0c", None))[0]


def render_layered_png(an, text: str, style: str, cfg: RenderConfig, warns: WarningCollector,
                       ink: str = "mono", remove_bg: bool = True, light: bool = False,
                       out_width: int = 1400, render_w: int = 2200, tone_density: float = 0.6):
    """Layered portrait -> PNG bytes. style='message' = poster rows; else Words
    (mosaic) layout. Returns (png_bytes, runs, ground_hex, mask_svg)."""
    # The layout only needs glyph POSITIONS (we whiten them into a mask); the
    # colour/ink is applied separately by _tint_photo. Build the layout with a
    # neutral ink so the ink choice never touches the layout (and we avoid the
    # mosaic's photo-ink colour path, which isn't needed here).
    if style == "message":
        # Message/prose is a single uniform text size (no face tiers), so the
        # word-size control maps straight to the poster font size. Clamp to a
        # sane readable band; default min_font_px=20 reproduces the prior look.
        msg_font = float(min(cfg.max_font_px, max(12.0, cfg.min_font_px)))
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
        # Multi-scale gap-fill packs the full size range into the silhouette in
        # both modes. In light/engraving mode the lit face is pushed to white
        # (see _tint_photo), so the dense fill there renders as white paper and
        # only shadows/features take ink -- full range without a gray jumble.
        res = build_portrait(an, words, cfg, warns, uppercase=True, ink="mono",
                             render_w=render_w, tone_density=eff_density,
                             gap_fill=True)
        colored, runs = res.svg, res.runs
    ground_hex = _ground_hex(ink, light)
    if not colored:
        return b"", runs, ground_hex, ""
    mask_svg = _mask_svg(colored)
    png = compose_layered(mask_svg, an, ink, remove_bg, out_width, light=light)
    return png, runs, ground_hex, mask_svg


def compose_layered(mask_svg: str, an, ink: str, remove_bg: bool, out_width: int, light: bool = False) -> bytes:
    """Composite the tinted photo through a prebuilt white-text mask SVG. Reused
    at download from the stored mask, so the costly layout build runs only once
    (at render), not again per sale."""
    from PIL import Image
    from .raster import svg_to_png_bytes
    if not mask_svg:
        return b""
    mpng = svg_to_png_bytes(mask_svg, output_width=out_width)
    mask = np.asarray(Image.open(io.BytesIO(mpng)).convert("L"))
    H, W = mask.shape[:2]
    photo = _tint_photo(an, W, H, ink, remove_bg, light=light).astype(np.float32)
    ground = np.array(_hex_to_rgb(_ground_hex(ink, light)), dtype=np.float32)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    out = (ground + (photo - ground) * m).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue()
