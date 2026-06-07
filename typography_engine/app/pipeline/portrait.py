"""Assemble a full typographic portrait SVG from analysis + approved words.

The portrait is rendered as a tonal word-fill: the photo's light and shadow are
reproduced as a monospace grid of the approved words (dark areas dense with
letters, bright skin blank), which reads as the person's face.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..config import RenderConfig
from .analyze import Analysis
from .svgbuild import validate_svg
from .textlayout import TextRun
from .tonal import build_tonal_portrait
from .warnings import WarningCollector


@dataclass
class PortraitResult:
    svg: str
    runs: List[TextRun]


def build_portrait(
    an: Analysis,
    words: Sequence[str],
    cfg: RenderConfig,
    warns: WarningCollector,
    uppercase: bool = True,
    ink: str = "navy",
    render_w: int = 2600,
    tone_density: float = 0.0,
    gap_fill: bool = True,
    gap_fill_passes: int = 12,
) -> PortraitResult:
    cfg.validate()
    svg, runs = build_tonal_portrait(an, words, cfg, warns, uppercase=uppercase, ink=ink,
                                     render_w=render_w, tone_density=tone_density,
                                     gap_fill=gap_fill, gap_fill_passes=gap_fill_passes)
    if svg:
        validate_svg(svg)
    return PortraitResult(svg=svg, runs=runs)
