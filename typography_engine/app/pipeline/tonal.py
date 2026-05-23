"""Tonal word-fill portrait.

Reproduce the photo's light and shadow as a monospace grid of the approved
words: cells dark enough to clear a low tone threshold are inked, while
near-white highlights and the background stay blank. Each inked glyph is shaded
by the exact tone it lands on (light gray on skin midtones, near-black on hair,
brows, eyes and lips), so the assembled grid carries smooth gradients and reads
as the person's face. Masked to the silhouette so the background stays clean.

Each contiguous dark run is packed with whole words from the approved list
(cycled in order); a word is placed only when it fits the run entirely, and
runs too short to hold any word stay blank. No word is ever cut and no stranded
single letters appear.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..config import RenderConfig
from .svgbuild import SvgDoc, esc
from .textlayout import TextRun, normalize_words
from .warnings import WarningCollector

_MONO_FAMILY = "'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace"
_MONO_ADVANCE = 0.6  # glyph advance as a fraction of em for monospace fonts

# Per-glyph gray ramp (0-255): midtone cells render light, the darkest features
# render near-black, so tone gradients carry the likeness.
_SHADE_LIGHT = 205
_SHADE_DARK = 18


def _tone_field(gray: np.ndarray, mask: np.ndarray, gamma: float, floor: float) -> np.ndarray:
    """Return per-pixel darkness in [0,1] (1 = ink), contrast-stretched within
    the subject and gamma-shaped so bright skin drops out and features stay."""
    m = mask > 127
    vals = gray[m] if int(m.sum()) > 50 else gray.reshape(-1)
    lo, hi = np.percentile(vals, [4.0, 96.0])
    if hi - lo < 1.0:
        lo, hi = float(vals.min()), float(max(vals.min() + 1.0, vals.max()))
    g = (gray.astype(np.float32) - lo) / (hi - lo)
    g = np.clip(g, 0.0, 1.0)
    dark = 1.0 - g
    # Drop everything below `floor`, then renormalize and gamma-shape so skin
    # thins out while hair/eyes/lips stay solid.
    dark = np.clip((dark - floor) / max(1e-3, 1.0 - floor), 0.0, 1.0)
    dark = dark ** gamma
    dark[~m] = 0.0
    return dark


def build_tonal_portrait(
    an,
    words: Sequence[str],
    cfg: RenderConfig,
    warns: WarningCollector,
    uppercase: bool = True,
    render_w: int = 1700,
    gamma: float = 1.7,
    floor: float = 0.32,
    level: float = 0.16,
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
        W, H = w0, h0

    dark = _tone_field(gray, mask, gamma=gamma, floor=floor)
    mset = mask > 127

    font = max(cfg.min_font_px, cfg.min_font_px)
    font = min(cfg.max_font_px, font)
    cell_w = font * _MONO_ADVANCE
    row_h = font * 0.92
    cols = max(1, int(W / cell_w))
    rows = max(1, int(H / row_h))

    doc = SvgDoc(width=W, height=H, background=cfg.background_hex)
    runs: List[TextRun] = []
    cursor = 0

    span = max(1, _SHADE_LIGHT - _SHADE_DARK)
    inv_level = max(1e-3, 1.0 - level)
    ntok = len(tokens)
    for r in range(rows):
        cy = (r + 0.5) * row_h
        yi = int(min(H - 1, max(0, round(cy))))
        baseline = cy + font * 0.34
        # Sample the tone under each cell; ink cells brighter than `level`
        # stay blank so near-white highlights and the background read clean.
        tone = [0.0] * cols
        ink = [False] * cols
        for c in range(cols):
            cx = (c + 0.5) * cell_w
            xi = int(min(W - 1, max(0, round(cx))))
            d = dark[yi, xi] if mset[yi, xi] else 0.0
            tone[c] = d
            ink[c] = d > level
        # Fill each contiguous inked run with whole words only (a word is placed
        # only when it fits the run entirely), then shade every glyph by the
        # tone of the cell it lands on so the face carries smooth gradients.
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
            while end - pos >= shortest:
                remaining = end - pos
                chosen = None
                for i in range(ntok):
                    t = tokens[(cursor + i) % ntok]
                    if len(t) <= remaining:
                        chosen = t
                        cursor = (cursor + i + 1) % ntok
                        break
                if chosen is None:
                    break
                glyphs.extend(chosen)
                pos += len(chosen)
            if not glyphs:
                continue
            spans = []
            for k, ch in enumerate(glyphs):
                cell = start + k
                norm = (tone[cell] - level) / inv_level
                norm = 1.0 if norm > 1.0 else (0.0 if norm < 0.0 else norm)
                g = _SHADE_LIGHT - int(round(span * norm))
                spans.append(
                    f'<tspan x="{cell * cell_w:.1f}" fill="#{g:02x}{g:02x}{g:02x}">'
                    f"{esc(ch)}</tspan>"
                )
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
    return doc.to_svg(), runs
