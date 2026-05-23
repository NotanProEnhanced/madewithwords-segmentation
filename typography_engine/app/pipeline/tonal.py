"""Tonal word-fill portrait.

Reproduce the photo's light and shadow as a monospace grid of the approved
words: cells where the image is dark (hair, brows, eyes, lips, shadows) are
inked and cells that are bright (skin highlights, background) stay blank. A
tone threshold turns the continuous image into solid dark masses, so the
assembled grid reads as the person's face. Masked to the silhouette so the
background stays clean.

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
    render_w: int = 1500,
    gamma: float = 1.7,
    floor: float = 0.32,
    level: float = 0.32,
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

    ntok = len(tokens)
    for r in range(rows):
        cy = (r + 0.5) * row_h
        yi = int(min(H - 1, max(0, round(cy))))
        # Mark which cells fall on dark (inked) tone via the ordered dither.
        ink = [False] * cols
        for c in range(cols):
            cx = (c + 0.5) * cell_w
            xi = int(min(W - 1, max(0, round(cx))))
            d = dark[yi, xi] if mset[yi, xi] else 0.0
            ink[c] = d > level
        # Fill each contiguous inked run with whole words only. Runs shorter
        # than the shortest word stay blank, so no word is ever cut and no
        # stranded single letters appear.
        chars = [" "] * cols
        placed = 0
        c = 0
        while c < cols:
            if not ink[c]:
                c += 1
                continue
            start = c
            while c < cols and ink[c]:
                c += 1
            run = c - start  # length of this contiguous dark run, in cells
            pos = start
            end = start + run
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
                for ch in chosen:
                    chars[pos] = ch
                    pos += 1
                placed += len(chosen)
        if placed == 0:
            continue
        row_str = "".join(chars).rstrip()
        if not row_str:
            continue
        baseline = cy + font * 0.34
        doc.add(
            f'<text x="0" y="{baseline:.1f}" xml:space="preserve" '
            f'font-family="{esc(_MONO_FAMILY)}" font-size="{font:.2f}" '
            f'font-weight="{esc(cfg.font_weight)}" fill="{cfg.foreground_hex}">'
            f"{esc(row_str)}</text>"
        )
        runs.append(
            TextRun(
                region="tonal",
                path_id=f"row_{r}",
                path_d="",
                text=row_str,
                font_size=round(font, 2),
                kind="primary",
            )
        )

    if not runs:
        warns.error("text", "no_runs", "Tonal fill produced no text (subject too bright or mask empty).")
    return doc.to_svg(), runs
