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

# Per-glyph gray ramp (0-255): the lightest inked cells render near this light
# gray, the darkest features near-black, so tone gradients carry the likeness.
# The light end is clamped well below white so bright skin and white hair still
# render as faint words instead of dropping out to blank.
_SHADE_LIGHT = 194
_SHADE_DARK = 0

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
        # Gentle linear remap into a moderate band (not full black->white) so a
        # washed-out pale face gains presence and contrast without over-
        # amplifying its faint cell-level variation. Mapping to [0.10,0.84]
        # rather than [0,1] and blending with the original keeps the smooth
        # gradients that read as detail -- a hard full-range stretch (or a gamma
        # lift) compounds with the sharpen + shade S-curve and turns subtle
        # modeling into blotches.
        t_lo, t_hi = 0.10, 0.84
        remap = np.clip((dark - lo) / (hi - lo), 0.0, 1.0) * (t_hi - t_lo) + t_lo
        alpha = 0.6
        balanced = dark * (1.0 - alpha) + remap * alpha
        out = out * (1.0 - wm) + balanced * wm
    return np.clip(out, 0.0, 1.0)


def _emphasize_features(dark: np.ndarray, an, scale: float, mset: np.ndarray) -> np.ndarray:
    """Deepen the eyes, brows, lips and nostrils of every face so each person's
    likeness anchors there (supports group portraits, not just one subject)."""
    faces = _faces_of(an)
    if not faces:
        return dark
    H, W = dark.shape[:2]
    fm = np.zeros((H, W), np.uint8)
    for face in faces:
        pts = face.points * scale
        for grp in _FEATURE_GROUPS:
            hull = cv2.convexHull(np.array([pts[i] for i in grp], np.int32))
            cv2.fillConvexPoly(fm, hull, 255)
    fm = cv2.dilate(fm, np.ones((5, 5), np.uint8), 1)
    w = (cv2.GaussianBlur(fm, (0, 0), 3.0).astype(np.float32) / 255.0) * mset
    return dark * (1.0 - w) + np.clip(dark ** 0.55, 0.0, 1.0) * w


def build_tonal_portrait(
    an,
    words: Sequence[str],
    cfg: RenderConfig,
    warns: WarningCollector,
    uppercase: bool = True,
    render_w: int = 2600,
    gamma: float = 1.0,
    floor: float = 0.0,
    level: float = 0.02,
    power: float = 1.0,
    auto_tone: bool = True,
    target_tone: float = 0.50,
    jitter: float = 0.7,
    seed: int = 1234,
    contrast: float = 2.2,
    pivot: float = 0.42,
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

    font = min(cfg.max_font_px, max(cfg.min_font_px, cfg.min_font_px))
    cell_w = font * _MONO_ADVANCE
    row_h = font * 0.80
    cols = max(1, int(W / cell_w))
    rows = max(1, int(H / row_h))

    # Area-average the tone into the glyph grid so each letter reflects the mean
    # darkness of its whole cell -> smooth, faithful gradients (not point noise).
    grid = cv2.resize(dark, (cols, rows), interpolation=cv2.INTER_AREA)

    doc = SvgDoc(width=W, height=H, background=cfg.background_hex)
    runs: List[TextRun] = []
    cursor = 0
    span = max(1, _SHADE_LIGHT - _SHADE_DARK)
    inv_level = max(1e-3, 1.0 - level)
    # Seeded (reproducible) per-row jitter offsets break up the rigid column grid
    # so words don't form vertical "rivers" or horizontal banding.
    rng = np.random.default_rng(seed)

    for r in range(rows):
        ox = (rng.random() - 0.5) * cell_w * jitter
        oy = (rng.random() - 0.5) * row_h * jitter * 0.5
        baseline = (r + 0.5) * row_h + font * 0.34 + oy
        row = grid[r]
        ink = row > level
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
                need = 0 if first else 1  # one blank cell between words
                avail = end - pos - need
                if avail < shortest:
                    break
                chosen = None
                for i in range(ntok):
                    t = tokens[(cursor + i) % ntok]
                    if len(t) <= avail:
                        chosen = t
                        cursor = (cursor + i + 1) % ntok
                        break
                if chosen is None:
                    break
                if not first:
                    glyphs.append(" ")
                    pos += 1
                glyphs.extend(chosen)
                pos += len(chosen)
                first = False
            if not glyphs:
                continue
            spans = []
            for k, ch in enumerate(glyphs):
                if ch == " ":
                    continue
                cell = start + k
                norm = (row[cell] - level) / inv_level
                # Contrast S-curve: push darks toward black, lights toward light.
                norm = (norm - pivot) * contrast + pivot
                norm = 1.0 if norm > 1.0 else (0.0 if norm < 0.0 else norm)
                g = _SHADE_LIGHT - int(round(span * (norm ** power)))
                # Per-glyph jitter so baselines and columns aren't dead straight;
                # breaks the rigid grid into organic texture (kills the residual
                # horizontal-row / vertical-column banding) while each letter
                # stays within its own tonal cell, so the likeness is preserved.
                gx = cell * cell_w + ox + (rng.random() - 0.5) * cell_w * 0.22
                gy = baseline + (rng.random() - 0.5) * row_h * 0.24
                spans.append(
                    f'<tspan x="{gx:.1f}" y="{gy:.1f}" fill="#{g:02x}{g:02x}{g:02x}">'
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

    if not runs:
        warns.error("text", "no_runs", "Tonal fill produced no text (subject too bright or mask empty).")
    return doc.to_svg(), runs
