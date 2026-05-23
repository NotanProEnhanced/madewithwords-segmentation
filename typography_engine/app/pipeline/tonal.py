"""Tonal word portrait.

Lay the approved words out as readable, whole words in their submitted order,
and drive each word's FONT SIZE from the local luminance: dark regions (hair,
brows, eyes, lips, shadows) get large, heavy words; bright skin gets small
words; highlights stay blank. The varying ink density reads as the subject's
face. Masked to the silhouette so the background stays clean.

No random letters: words are emitted intact, cycling through the approved list
in order.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..config import RenderConfig
from .svgbuild import SvgDoc, esc
from .textlayout import TextRun, normalize_words
from .textmeasure import text_width
from .warnings import WarningCollector


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
    target_rows: int = 30,
    gamma: float = 1.5,
    floor: float = 0.32,
) -> Tuple[str, List[TextRun]]:
    approved = normalize_words(words, uppercase)
    if not approved:
        warns.error("text", "no_words", "No approved words supplied; cannot place typography.")
        return "", []

    gray = an.img.gray
    mask = an.silhouette.mask
    h0, w0 = gray.shape[:2]
    if mask.shape[:2] != (h0, w0):
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

    # Upsample so words can be small where bright while staying >= min_font.
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

    row_h = H / float(target_rows)
    min_size = float(cfg.min_font_px)
    max_size = min(cfg.max_font_px, max(min_size + 2.0, row_h * 0.98))

    doc = SvgDoc(width=W, height=H, background=cfg.background_hex)
    runs: List[TextRun] = []
    cursor = 0

    for r in range(target_rows):
        y_top = r * row_h
        baseline = y_top + row_h * 0.82
        yc = int(min(H - 1, max(0, round(y_top + row_h * 0.5))))
        cols = np.where(mset[yc])[0]
        if cols.size == 0:
            continue
        x_left, x_right = float(cols.min()), float(cols.max())
        wy0 = int(max(0, round(y_top + row_h * 0.5 - row_h * 0.3)))
        wy1 = int(min(H, round(y_top + row_h * 0.5 + row_h * 0.3)))
        x = x_left
        row_words: List[str] = []
        row_min = max_size
        while x < x_right:
            xi = int(min(W - 1, max(0, round(x))))
            if not mset[yc, xi]:
                x += min_size * 0.6  # blank highlight/background: keep word order
                continue
            wx0 = int(max(0, round(x)))
            wx1 = int(min(W, round(x + max_size * 0.7)))
            region = dark[wy0:wy1, wx0:wx1]
            d = float(region.mean()) if region.size else 0.0
            if d <= 0.04:
                x += min_size * 0.6  # lit highlight: leave white, keep word order
                continue
            size = min_size + min(1.0, d) * (max_size - min_size)
            word = approved[cursor % len(approved)]
            ww = text_width(word, size)
            if x + ww > x_right + max_size:
                break
            doc.add(
                f'<text x="{x:.1f}" y="{baseline:.1f}" '
                f'font-family="{esc(cfg.primary_font_family)}" font-size="{size:.2f}" '
                f'font-weight="{esc(cfg.font_weight)}" fill="{cfg.foreground_hex}">'
                f"{esc(word)}</text>"
            )
            cursor += 1
            row_words.append(word)
            row_min = min(row_min, size)
            x += ww + size * 0.3
        if row_words:
            runs.append(
                TextRun(
                    region="tonal",
                    path_id=f"row_{r}",
                    path_d="",
                    text=" ".join(row_words),
                    font_size=round(row_min, 2),
                    kind="primary",
                )
            )

    if not runs:
        warns.error("text", "no_runs", "Tonal fill produced no text (subject too bright or mask empty).")
    return doc.to_svg(), runs
