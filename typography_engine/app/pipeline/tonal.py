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


def _emphasize_features(dark: np.ndarray, an, scale: float, mset: np.ndarray) -> np.ndarray:
    """Deepen the eyes, brows, lips and nostrils so the likeness anchors there."""
    lm = getattr(an, "landmarks", None)
    if lm is None:
        return dark
    H, W = dark.shape[:2]
    fm = np.zeros((H, W), np.uint8)
    pts = lm.points * scale
    for grp in _FEATURE_GROUPS:
        hull = cv2.convexHull(np.array([pts[i] for i in grp], np.int32))
        cv2.fillConvexPoly(fm, hull, 255)
    fm = cv2.dilate(fm, np.ones((5, 5), np.uint8), 1)
    w = (cv2.GaussianBlur(fm, (0, 0), 3.0).astype(np.float32) / 255.0) * mset
    return dark * (1.0 - w) + np.clip(dark ** 0.55, 0.0, 1.0) * w


# Feature placement priority: the first approved word anchors the most
# important feature (eyes/brow), then jaw, then the silhouette crown frames it.
_FEATURE_PRIORITY = {"brow_line": 0, "jaw_line": 1, "lip_line": 1, "silhouette": 2}


def _add_feature_words(
    doc: SvgDoc,
    an,
    scale: float,
    approved: Sequence[str],
    cfg: RenderConfig,
    H: int,
    warns: WarningCollector,
    uppercase: bool,
) -> List[TextRun]:
    """Tier 1: large readable words flowed along the key facial features.

    The tonal grid (tier 2) carries shading/likeness; this layer places the
    user's words at a legible size on the eyes/brow, jaw and silhouette so a
    viewer reads them at arm's length. Words are haloed to stay legible over the
    texture. Failures here are non-fatal -- the texture portrait stands alone."""
    from .textlayout import RegionPath, layout_text_runs

    regions = getattr(getattr(an, "regions", None), "paths", None)
    if not regions:
        return []

    scaled = [
        RegionPath(rp.name, rp.points * scale, rp.closed, rp.kind) for rp in regions
    ]
    scaled.sort(key=lambda rp: _FEATURE_PRIORITY.get(rp.name, 9))

    local = WarningCollector()  # don't let supplementary-layer warnings fail the render
    runs = layout_text_runs(scaled, approved, cfg, image_h=H, warns=local, uppercase=uppercase)
    for r in runs:
        doc.add_haloed_text_on_path(
            path_id=f"feat_{r.path_id}",
            d=r.path_d,
            text=r.text,
            font_size=r.font_size,
            fill=cfg.foreground_hex,
            halo=cfg.background_hex,
            font_family=cfg.primary_font_family,
            font_weight=cfg.font_weight,
            start_offset=r.start_offset,
        )
    return runs


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
    feature_words: bool = True,
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
    dark = _emphasize_features(dark, an, scale, mset)

    font = min(cfg.max_font_px, max(cfg.min_font_px, cfg.min_font_px))
    cell_w = font * _MONO_ADVANCE
    row_h = font * 0.92
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
                spans.append(
                    f'<tspan x="{cell * cell_w + ox:.1f}" fill="#{g:02x}{g:02x}{g:02x}">'
                    f"{esc(ch)}</tspan>"
                )
            if not spans:
                continue
            doc.add(
                f'<text y="{baseline:.1f}" xml:space="preserve" '
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

    if feature_words:
        runs.extend(_add_feature_words(doc, an, scale, approved, cfg, H, warns, uppercase))

    return doc.to_svg(), runs
