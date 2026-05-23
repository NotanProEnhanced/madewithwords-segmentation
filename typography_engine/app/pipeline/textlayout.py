"""Place readable words from the approved list along region paths via textPath.

Guarantees:
  * Only words from the user-approved list are ever emitted (cycled/repeated).
  * No text is rendered below cfg.min_font_px (skipped + warned instead).
  * Bold black glyphs flowing along each contour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..config import RenderConfig
from .regions import RegionPath
from .textmeasure import text_width
from .warnings import WarningCollector

# Fraction of a path's length to fill with text. Closed loops leave a gap so the
# end does not collide with the start; open paths can fill almost fully.
_FILL_RATIO_CLOSED = 0.92
_FILL_RATIO_OPEN = 0.98

# Per-region font size as a fraction of image height, before clamping.
_REGION_FONT_FRACTION = {
    "silhouette": 0.055,
    "jaw_line": 0.030,
    "brow_line": 0.026,
    "lip_line": 0.026,
}
_DEFAULT_FRACTION = 0.030


@dataclass
class TextRun:
    region: str
    path_id: str
    path_d: str
    text: str
    font_size: float
    kind: str
    start_offset: str = "0%"


# Per-region (start, end) fractions of path length to occupy with text. The
# silhouette starts slightly in and stops before its ends so words don't curl
# into the sharp shoulder/frame corners where the path turns vertical.
_REGION_SPAN = {
    "silhouette": (0.065, 0.91),
}
_DEFAULT_SPAN_OPEN = (0.0, _FILL_RATIO_OPEN)
_DEFAULT_SPAN_CLOSED = (0.0, _FILL_RATIO_CLOSED)


def polyline_length(points: np.ndarray, closed: bool) -> float:
    if len(points) < 2:
        return 0.0
    pts = points
    if closed:
        pts = np.vstack([points, points[:1]])
    d = np.diff(pts, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def normalize_words(words: Sequence[str], uppercase: bool) -> List[str]:
    out = []
    for w in words:
        w = (w or "").strip()
        if not w:
            continue
        out.append(w.upper() if uppercase else w)
    return out


def _fill_string_measured(
    words: List[str], target_px: float, font_size: float, start_word: int
) -> str:
    """Append cycled words while the measured width stays under target_px.

    Stops *before* exceeding the budget so text never overflows / wraps past the
    path start. Always returns at least one word if one fits.
    """
    if not words:
        return ""
    parts: List[str] = []
    i = start_word
    guard = 0
    max_iters = 4000
    while guard < max_iters:
        w = words[i % len(words)]
        candidate = " ".join(parts + [w])
        if text_width(candidate, font_size) > target_px and parts:
            break
        parts.append(w)
        i += 1
        guard += 1
        if not parts:  # safety
            break
    return " ".join(parts)


def _orient_left_to_right(rp: RegionPath) -> np.ndarray:
    """Orient a path so flowed text reads upright.

    Open paths: ensure points run left->right.
    Closed loops: ensure the path direction at the topmost point points right,
    so text along the top edge is upright (not mirrored) and starts there.
    """
    pts = rp.points
    if len(pts) < 3:
        return pts

    if not rp.closed:
        if pts[-1, 0] < pts[0, 0]:
            return pts[::-1].copy()
        return pts

    n = len(pts)
    top = int(np.argmin(pts[:, 1]))
    nxt = pts[(top + 1) % n]
    prv = pts[(top - 1) % n]
    tangent_x = nxt[0] - prv[0]
    rotated = np.roll(pts, -top, axis=0)  # start the loop at the topmost point
    if tangent_x < 0:
        # Reverse winding but keep the topmost point first.
        rotated = rotated[::-1].copy()
        rotated = np.roll(rotated, 1, axis=0)
    return rotated


def font_size_for(region: str, image_h: int, cfg: RenderConfig) -> float:
    frac = _REGION_FONT_FRACTION.get(region, _DEFAULT_FRACTION)
    size = frac * image_h
    return float(max(cfg.min_font_px, min(cfg.max_font_px, size)))


def layout_text_runs(
    regions: Sequence[RegionPath],
    words: Sequence[str],
    cfg: RenderConfig,
    image_h: int,
    warns: WarningCollector,
    uppercase: bool = True,
) -> List[TextRun]:
    approved = normalize_words(words, uppercase)
    if not approved:
        warns.error("text", "no_words", "No approved words supplied; cannot place typography.")
        return []

    # Shortest approved word governs whether a path can host any readable word.
    shortest_word = min(approved, key=len)

    from .pathgen import catmull_rom_to_bezier_d

    runs: List[TextRun] = []
    word_cursor = 0
    for i, rp in enumerate(regions):
        pts = _orient_left_to_right(rp)
        oriented = RegionPath(rp.name, pts, rp.closed, rp.kind)
        length = polyline_length(oriented.points, oriented.closed)
        default_span = _DEFAULT_SPAN_CLOSED if oriented.closed else _DEFAULT_SPAN_OPEN
        start_frac, end_frac = _REGION_SPAN.get(oriented.name, default_span)
        budget = length * (end_frac - start_frac)

        # Shrink font from the region's base size toward the minimum until the
        # shortest approved word fits the path. Never go below cfg.min_font_px.
        base = font_size_for(oriented.name, image_h, cfg)
        size = base
        while size > cfg.min_font_px and text_width(shortest_word, size) > budget:
            size = max(cfg.min_font_px, size * 0.9)

        if text_width(shortest_word, size) > budget:
            warns.warn(
                "text",
                "path_too_short",
                f"Region '{oriented.name}' path ({length:.0f}px) cannot fit even "
                f"'{shortest_word}' at the minimum {cfg.min_font_px:.0f}px; skipped "
                f"to preserve readability.",
            )
            continue

        text = _fill_string_measured(approved, budget, size, word_cursor)
        if not text:
            continue
        word_cursor += 1  # rotate starting word so adjacent regions differ

        path_d = catmull_rom_to_bezier_d(oriented.points, oriented.closed)
        runs.append(
            TextRun(
                region=oriented.name,
                path_id=f"tp_{oriented.name}_{i}",
                path_d=path_d,
                text=text,
                font_size=round(size, 2),
                kind=oriented.kind,
                start_offset=f"{start_frac * 100:.1f}%",
            )
        )

    if not runs:
        warns.error("text", "no_runs", "No region path could host readable text.")
    return runs
