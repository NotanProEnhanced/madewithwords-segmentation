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
from .warnings import WarningCollector

# Approximate average glyph advance as a fraction of font size for bold sans
# (includes spaces). Used only to estimate how many characters fill a path.
_GLYPH_ADVANCE = 0.62

# Per-region font size as a fraction of image height, before clamping.
_REGION_FONT_FRACTION = {
    "face_oval": 0.052,
    "hair": 0.050,
    "neck": 0.048,
    "lips": 0.030,
    "left_eye": 0.022,
    "right_eye": 0.022,
    "left_brow": 0.022,
    "right_brow": 0.022,
    "nose": 0.024,
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


def _fill_string(words: List[str], target_chars: int, start_word: int) -> str:
    """Build a space-joined string from cycled words up to ~target_chars."""
    if not words:
        return ""
    parts: List[str] = []
    length = 0
    i = start_word
    guard = 0
    max_iters = target_chars * 2 + 50
    while length < target_chars and guard < max_iters:
        w = words[i % len(words)]
        parts.append(w)
        length += len(w) + 1
        i += 1
        guard += 1
    return " ".join(parts) if parts else words[0]


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

    runs: List[TextRun] = []
    word_cursor = 0
    for i, rp in enumerate(regions):
        size = font_size_for(rp.name, image_h, cfg)
        length = polyline_length(rp.points, rp.closed)

        # Can at least one approved word fit at the (already min-clamped) size?
        shortest = min(len(w) for w in approved)
        min_word_px = shortest * size * _GLYPH_ADVANCE
        if length < min_word_px:
            warns.warn(
                "text",
                "path_too_short",
                f"Region '{rp.name}' path ({length:.0f}px) too short for a readable word "
                f"at {size:.0f}px; skipped to preserve readability.",
            )
            continue

        target_chars = max(1, int(length / (size * _GLYPH_ADVANCE)))
        text = _fill_string(approved, target_chars, word_cursor)
        word_cursor += 1  # rotate starting word so regions vary

        from .pathgen import catmull_rom_to_bezier_d

        path_d = catmull_rom_to_bezier_d(rp.points, rp.closed)
        runs.append(
            TextRun(
                region=rp.name,
                path_id=f"tp_{rp.name}_{i}",
                path_d=path_d,
                text=text,
                font_size=round(size, 2),
                kind=rp.kind,
            )
        )

    if not runs:
        warns.error("text", "no_runs", "No region path could host readable text.")
    return runs
