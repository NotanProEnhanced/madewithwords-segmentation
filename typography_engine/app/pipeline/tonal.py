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
    contrast: float = 2.3,
    pivot: float = 0.5,
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
    dark = _tone_field(_sharpen(gray), mask, gamma=1.0, floor=0.0)
    dark = _auto_tone(dark, mset, 0.50, max_shift=0.18)
    dark = _balance_faces(dark, an, scale, mset)
    dark = _emphasize_features(dark, an, scale, mset)
    dark = _sharpen_eyes(dark, an, scale, mset)
    tone_s = cv2.GaussianBlur(dark, (0, 0), max(1.0, font_px * 0.5))

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

    font = min(cfg.max_font_px, max(cfg.min_font_px, cfg.min_font_px))
    cell_w = font * _MONO_ADVANCE
    row_h = font * 0.80
    cols = max(1, int(W / cell_w))
    rows = max(1, int(H / row_h))

    # Area-average the tone into the glyph grid so each letter reflects the mean
    # darkness of its whole cell -> smooth, faithful gradients (not point noise).
    grid = cv2.resize(dark, (cols, rows), interpolation=cv2.INTER_AREA)

    # Ink treatment: grayscale (mono), a named duotone, or colour sampled from
    # the source photo. Mono keeps the existing gray ramp untouched.
    photo_ink = ink == "photo"
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
    # canvas background so the actual portrait shows through and the typography
    # overlays on top. This is what produces the "exceptional likeness" look --
    # the photo carries the recognition, the words carry the personalization.
    if photo_ink:
        import base64
        import io as _io
        from PIL import Image as _PILImage
        # Resize the source to match the SVG canvas so the embedded payload
        # isn't gratuitously large. JPEG is far smaller than PNG for portraits.
        rgb = cv2.cvtColor(an.img.bgr, cv2.COLOR_BGR2RGB)
        pil = _PILImage.fromarray(rgb).resize((W, H), _PILImage.LANCZOS)
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
        ink = row > level
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
