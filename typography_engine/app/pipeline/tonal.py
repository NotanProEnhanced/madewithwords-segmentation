"""Tonal word-fill portrait.

Reproduce the photo's light and shadow as a single mosaic of the approved words.
The subject's tone is sharpened (CLAHE + unsharp) and contrast-stretched, then
every glyph is shaded by the tone it lands on (light gray on skin midtones,
near-black on hair, brows, eyes and lips), so the assembled field carries smooth
gradients and reads as the person's face. Masked to the silhouette so the
background stays clean.

Two things make the text read as part of the face rather than stamped on it:

  * Rows are traced as streamlines through a flow field derived from the face's
    own form, so lines of text bend around the eyes, nose, cheeks and jaw, and
    each glyph is rotated to the local contour tangent.
  * Word choice is importance-weighted: the most important words (input order)
    are steered onto the recognition features (eyes, brows, lips, nose) while
    later words fill the flatter areas. There is no separate size tier -- every
    glyph is the same size; only placement and shade vary.

Each contiguous run is packed with whole words from the approved list; a word is
placed only when it fits the run entirely. No word is ever cut and no stranded
single letters appear.
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
_SHADE_LIGHT = 194
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
    darkness toward `target` (clamped, contrast-preserving)."""
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


def _flow_field(gray: np.ndarray, mask: np.ndarray, sigma: float, max_tilt_deg: float = 18.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Gentle unit vector field the text rows bend along.

    Built from the iso-brightness tangent (perpendicular to the smoothed image
    gradient) so flow leans *along* the face's shading transitions -- around the
    eye sockets, down the nose, along the cheek and jaw. Only the strongest edges
    bend it; flat areas stay horizontal. Biased rightward and tilt-limited so the
    mosaic stays orderly."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    # Tangent to iso-contours, oriented to point rightward.
    tx, ty = -gy, gx
    flip = tx < 0
    tx = np.where(flip, -tx, tx)
    ty = np.where(flip, -ty, ty)
    n = np.maximum(mag, 1e-6)
    tx, ty = tx / n, ty / n

    inside = mask > 127
    mref = float(np.percentile(mag[inside], 88)) if int(inside.sum()) > 50 else float(mag.max() or 1.0)
    w = np.clip(mag / max(mref, 1e-6), 0.0, 1.0)  # only strong edges bend the flow
    vx = (1.0 - w) * 1.0 + w * tx
    vy = w * ty

    vx = cv2.GaussianBlur(vx, (0, 0), sigma * 1.5)
    vy = cv2.GaussianBlur(vy, (0, 0), sigma * 1.5)
    vx = np.maximum(vx, 0.05)  # keep sweeping rightward
    maxt = float(np.tan(np.radians(max_tilt_deg)))
    slope = np.clip(vy / np.maximum(vx, 1e-6), -maxt, maxt)
    vy = slope * vx
    norm = np.maximum(np.sqrt(vx * vx + vy * vy), 1e-6)
    return vx / norm, vy / norm


def _importance_field(an, scale: float, W: int, H: int, row_h: float) -> Optional[np.ndarray]:
    """[0,1] field, high on the recognition features, used to steer the most
    important words onto the eyes/brows/lips/nose. None when no landmarks."""
    lm = getattr(an, "landmarks", None)
    if lm is None:
        return None
    fm = np.zeros((H, W), np.float32)
    pts = lm.points * scale
    for grp in _FEATURE_GROUPS:
        hull = cv2.convexHull(np.array([pts[i] for i in grp], np.int32))
        cv2.fillConvexPoly(fm, hull, 1.0)
    fm = cv2.GaussianBlur(fm, (0, 0), max(1.0, row_h * 2.0))
    peak = float(fm.max())
    return fm / peak if peak > 0 else None


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

    # Upsample so the mosaic is fine while glyphs stay >= min_font.
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
    rows = max(1, int(H / row_h))
    cols = int(W / cell_w) + 2

    # Tone smoothed at cell scale (so each glyph reflects its neighbourhood) and
    # the flow field whose vertical lean bends the rows and rotates the glyphs.
    tone_s = cv2.GaussianBlur(dark, (0, 0), max(1.0, cell_w * 0.8))
    vx_f, vy_f = _flow_field(gray, mask, sigma=max(2.0, row_h * 1.2))
    deg_f = np.degrees(np.arctan2(vy_f, vx_f)).astype(np.float32)
    imp = _importance_field(an, scale, W, H, row_h)

    doc = SvgDoc(width=W, height=H, background=cfg.background_hex)
    runs: List[TextRun] = []
    cursor = 0
    span = max(1, _SHADE_LIGHT - _SHADE_DARK)
    inv_level = max(1e-3, 1.0 - level)
    rng = np.random.default_rng(seed)

    # Rows stay in vertical lanes (fixed horizontal advance) but drift with the
    # flow's vertical lean, hard-clamped so a row bows around features without
    # ever crossing its neighbours; a mild spring relaxes it back in flat areas.
    flow_gain = 1.2
    spring = 0.12
    max_drift = 0.45 * row_h

    def pick_word(avail: int, ix: int, iy: int) -> int:
        """Index of a word that fits `avail` cells, biased by local importance:
        high importance -> top-priority (early) words."""
        if imp is not None:
            target = (1.0 - float(imp[iy, ix])) * (ntok - 1) + rng.normal() * 0.5
            order = sorted(range(ntok), key=lambda j: abs(j - target))
        else:
            order = [(cursor + j) % ntok for j in range(ntok)]
        for j in order:
            if len(tokens[j]) <= avail:
                return j
        return -1

    for r in range(rows):
        y_nom = (r + 0.5) * row_h + (rng.random() - 0.5) * row_h * jitter * 0.4
        y = y_nom
        pts: List[Tuple[float, float, float, int, int, bool, float]] = []  # x,y,deg,ix,iy,inside,tone
        for c in range(cols):
            x = c * cell_w
            xi = 0 if x < 0 else (W - 1 if x >= W else int(x))
            yi = 0 if y < 0 else (H - 1 if y >= H else int(y))
            inside = bool(mset[yi, xi])
            pts.append((x, y, float(deg_f[yi, xi]), xi, yi, inside, float(tone_s[yi, xi])))
            y += float(vy_f[yi, xi]) * cell_w * flow_gain
            y -= spring * (y - y_nom)
            d = y - y_nom
            if d > max_drift:
                y = y_nom + max_drift
            elif d < -max_drift:
                y = y_nom - max_drift

        npts = len(pts)
        c = 0
        while c < npts:
            if not (pts[c][5] and pts[c][6] > level):
                c += 1
                continue
            start = c
            while c < npts and pts[c][5] and pts[c][6] > level:
                c += 1
            end = c
            pos = start
            placed: List[Tuple[str, float, float, float, float]] = []  # ch,x,y,deg,tone
            first = True
            while True:
                need = 0 if first else 1  # one blank cell between words
                avail = end - pos - need
                if avail < shortest:
                    break
                j = pick_word(avail, pts[pos][3], pts[pos][4])
                if j < 0:
                    break
                cursor = (j + 1) % ntok
                if not first:
                    px, py, pd = pts[pos][0], pts[pos][1], pts[pos][2]
                    placed.append((" ", px, py, pd, 0.0))
                    pos += 1
                for ch in tokens[j]:
                    px, py, pd, pt = pts[pos][0], pts[pos][1], pts[pos][2], pts[pos][6]
                    placed.append((ch, px, py, pd, pt))
                    pos += 1
                first = False
            if not any(ch != " " for ch, *_ in placed):
                continue

            spans = []
            for ch, px, py, pd, pt in placed:
                if ch == " ":
                    continue
                norm = (pt - level) / inv_level
                norm = (norm - pivot) * contrast + pivot
                norm = 1.0 if norm > 1.0 else (0.0 if norm < 0.0 else norm)
                g = _SHADE_LIGHT - int(round(span * (norm ** power)))
                rot = f' rotate="{pd:.0f}"' if abs(pd) > 0.5 else ""
                spans.append(
                    f'<tspan x="{px:.0f}" y="{py:.0f}"{rot} '
                    f'fill="#{g:02x}{g:02x}{g:02x}">{esc(ch)}</tspan>'
                )
            if not spans:
                continue
            doc.add(
                f'<text font-family="{esc(_MONO_FAMILY)}" font-size="{font:.2f}" '
                f'font-weight="{esc(cfg.font_weight)}">' + "".join(spans) + "</text>"
            )
            runs.append(
                TextRun(
                    region="tonal",
                    path_id=f"row{r}_{start}",
                    path_d="",
                    text="".join(ch for ch, *_ in placed),
                    font_size=round(font, 2),
                    kind="primary",
                )
            )

    if not runs:
        warns.error("text", "no_runs", "Tonal fill produced no text (subject too bright or mask empty).")

    return doc.to_svg(), runs
