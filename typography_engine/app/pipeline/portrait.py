"""Assemble a full typographic portrait SVG from analysis + approved words."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..config import RenderConfig
from .analyze import Analysis
from .svgbuild import SvgDoc, validate_svg
from .textlayout import TextRun, layout_text_runs
from .warnings import WarningCollector


@dataclass
class PortraitResult:
    svg: str
    runs: List[TextRun]


# Draw order: large primary contours first, fine detail last (sits on top).
_ORDER = ["hair", "face_oval", "neck", "left_brow", "right_brow",
          "left_eye", "right_eye", "nose", "lips"]


def _ordered_regions(an: Analysis):
    rank = {n: i for i, n in enumerate(_ORDER)}
    return sorted(an.regions.paths, key=lambda rp: rank.get(rp.name, len(_ORDER)))


def build_portrait(
    an: Analysis,
    words: Sequence[str],
    cfg: RenderConfig,
    warns: WarningCollector,
    uppercase: bool = True,
) -> PortraitResult:
    cfg.validate()
    regions = _ordered_regions(an)
    runs = layout_text_runs(regions, words, cfg, an.img.h, warns, uppercase=uppercase)

    doc = SvgDoc(width=an.img.w, height=an.img.h, background=cfg.background_hex)
    for r in runs:
        doc.add_text_on_path(
            path_id=r.path_id,
            d=r.path_d,
            text=r.text,
            font_size=r.font_size,
            fill=cfg.foreground_hex,
            font_family=cfg.primary_font_family,
            font_weight=cfg.font_weight,
            letter_spacing=cfg.letter_spacing_px,
        )

    svg = doc.to_svg()
    validate_svg(svg)
    return PortraitResult(svg=svg, runs=runs)
