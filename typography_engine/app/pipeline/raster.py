"""SVG -> PNG rasterization via CairoSVG."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def svg_to_png_bytes(svg_text: str, output_width: Optional[int] = None) -> bytes:
    import cairosvg

    kwargs = {}
    if output_width:
        kwargs["output_width"] = int(output_width)
    return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), **kwargs)


def write_png(svg_text: str, out_path: Path, output_width: Optional[int] = None) -> Path:
    data = svg_to_png_bytes(svg_text, output_width=output_width)
    out_path.write_bytes(data)
    return out_path
