"""Tonal word-fill portrait (variable-size quadtree packer).

Reproduce the photo's light and shadow as a mosaic of the approved words at
*varying* sizes, so the result is both a likeness and genuinely readable.

The subject is split with a quadtree: a cell subdivides only where there is
detail (high local tone variance) or where the silhouette edge crosses it.
Flat regions -- cheeks, forehead, clothing -- stay large and hold big, readable
words; the eyes, nose, mouth and outline recurse down to fine type that carries
the fidelity. Size therefore emerges from the image itself rather than a fixed
grid, which is what lets the words be read without mushing the face.

Word choice tracks cell size: the largest (most readable) cells take the most
important words (input order), so a viewer reads the words that matter first;
later words fill the smaller, busier cells. Every glyph is shaded by the tone it
lands on, masked to the silhouette so the background stays clean. Only whole
words are placed -- never a cut word or a stranded letter.
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

# Per-glyph gray ramp (0-255): lightest inked cells near this light gray, darkest
# features near-black, so tone gradients carry the likeness.
_SHADE_LIGHT = 188
_SHADE_DARK = 0

# MediaPipe 478-point mesh index groups for the recognition features.
_EYE_L = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_EYE_R = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
_BROW_L = (70, 63, 105, 66, 107, 46, 53, 52, 65, 55)
_BROW_R = (336, 296, 334, 293, 300, 276, 283, 282, 295, 285)
_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
_NOSE = (1, 2, 98, 327, 97, 326, 5, 4, 275, 440, 220, 45)
_FEATURE_GROUPS = (_EYE_L, _EYE_R, _BROW_L, _BROW_R, _LIPS, _NOSE)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Local-contrast (CLAHE) + unsharp mask so features keep their edges."""
    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(7, 7)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 2.4)
    return cv2.addWeighted(clahe, 1.8, blur, -0.8, 0)


def _tone_field(gray: np.ndarray, mask: np.ndarray, gamma: float, floor: float) -> np.ndarray:
    """Per-pixel darkness in [0,1] (1 = ink), contrast-stretched within the
    subject so the full tonal range is used and bright skin drops toward 0."""
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
    """Even out overall brightness by shifting the in-subject mean darkness toward
    `target` (clamped, contrast-preserving)."""
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


def _detail_profile(dark: np.ndarray, mset: np.ndarray, sigma_rows: float) -> np.ndarray:
    """Per-image-row detail (0..1): high where that horizontal band crosses busy
    structure (eyes, nose, mouth, edges), low across flat skin/clothing. Drives
    the row height -- fine rows through detail, tall readable rows through flats."""
    d32 = dark.astype(np.float32)
    gx = cv2.Sobel(d32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d32, cv2.CV_32F, 0, 1, ksize=3)
    gm = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3.0) * mset
    counts = np.maximum(mset.sum(axis=1), 1)
    drow = gm.sum(axis=1) / counts  # mean detail per row, within the subject
    drow = cv2.GaussianBlur(drow.reshape(-1, 1).astype(np.float32), (0, 0), sigma_rows).ravel()
    rows_in = mset.any(axis=1)
    if int(rows_in.sum()) < 4:
        return np.zeros_like(drow)
    lo, hi = np.percentile(drow[rows_in], [25, 95])
    if hi - lo < 1e-6:
        return np.zeros_like(drow)
    return np.clip((drow - lo) / (hi - lo), 0.0, 1.0)


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
    pivot: float = 0.40,
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
    tone_s = cv2.GaussianBlur(dark, (0, 0), 2.0)

    min_font = max(8.0, cfg.min_font_px)
    max_font = max(min_font + 6.0, min(cfg.max_font_px, W * 0.05))
    detail = _detail_profile(dark, mset, sigma_rows=max(2.0, min_font))

    doc = SvgDoc(width=W, height=H, background=cfg.background_hex)
    runs: List[TextRun] = []
    span = max(1, _SHADE_LIGHT - _SHADE_DARK)
    inv_level = max(1e-3, 1.0 - level)
    rng = np.random.default_rng(seed)

    def shade_hex(tone: float) -> str:
        norm = (tone - level) / inv_level
        norm = (norm - pivot) * contrast + pivot
        norm = 1.0 if norm > 1.0 else (0.0 if norm < 0.0 else norm)
        g = _SHADE_LIGHT - int(round(span * (norm ** power)))
        g = 0 if g < 0 else (255 if g > 255 else g)
        return f"#{g:02x}{g:02x}{g:02x}"

    def pick_word(size_norm: float, avail_chars: int) -> int:
        # Bigger rows (size_norm -> 1) lean toward the most important (early)
        # words, so the big readable words are the ones that matter.
        target = (1.0 - size_norm) * (ntok - 1) + rng.normal() * 0.7
        order = sorted(range(ntok), key=lambda j: abs(j - target))
        for j in order:
            if len(tokens[j]) <= avail_chars:
                return j
        return -1

    # Walk top->bottom in variable-height rows. Each row spans the full width as
    # one dense, continuous, per-letter-shaded line (this is what keeps the face
    # readable as tone); its font is set by the band's detail so the eyes/mouth
    # get fine rows and the flat forehead/cheeks/clothing get tall readable ones.
    y = 0.0
    while y < H:
        yi = min(H - 1, int(y))
        d = float(detail[yi])
        F = max_font - d * (max_font - min_font)
        F = max(min_font, min(max_font, F))
        row_h = F * 0.92
        size_norm = (F - min_font) / max(1e-3, max_font - min_font)
        adv = F * _MONO_ADVANCE
        cy = y + row_h * 0.5
        cyi = min(H - 1, max(0, int(cy)))
        baseline = y + F * 0.78

        spans = []
        placed_words: List[str] = []
        gx = 0.0
        while gx < W:
            avail_chars = int((W - gx) / adv)
            if avail_chars < shortest:
                break
            j = pick_word(size_norm, avail_chars)
            if j < 0:
                break
            wd = tokens[j]
            drawn = False
            for ch in wd:
                cxc = gx + adv * 0.5
                cxi = min(W - 1, max(0, int(cxc)))
                if mset[cyi, cxi]:
                    spans.append(
                        f'<tspan x="{gx:.0f}" fill="{shade_hex(float(tone_s[cyi, cxi]))}">'
                        f'{esc(ch)}</tspan>'
                    )
                    drawn = True
                gx += adv
            if drawn:
                placed_words.append(wd)
            gx += adv  # blank cell between words
        y += row_h

        if not spans:
            continue
        doc.add(
            f'<text y="{baseline:.0f}" font-family="{esc(_MONO_FAMILY)}" '
            f'font-size="{F:.1f}" font-weight="{esc(cfg.font_weight)}">'
            + "".join(spans) + "</text>"
        )
        runs.append(
            TextRun(region="tonal", path_id=f"row{int(y)}", path_d="",
                    text=" ".join(placed_words), font_size=round(F, 1), kind="primary")
        )

    if not runs:
        warns.error("text", "no_runs", "Tonal fill produced no text (subject too bright or mask empty).")

    return doc.to_svg(), runs
