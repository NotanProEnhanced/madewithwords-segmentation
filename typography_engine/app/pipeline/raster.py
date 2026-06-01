"""SVG -> PNG rasterization. Prefers CairoSVG (production); falls back to
resvg-py when Cairo's native library is unavailable (e.g. local Windows)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def svg_to_png_bytes(svg_text: str, output_width: Optional[int] = None) -> bytes:
    try:
        import cairosvg
    except (ImportError, OSError):
        return _resvg_png_bytes(svg_text, output_width)
    kwargs = {}
    if output_width:
        kwargs["output_width"] = int(output_width)
    try:
        return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), **kwargs)
    except OSError:
        # cairosvg imported but its native cairo library is missing at call time.
        return _resvg_png_bytes(svg_text, output_width)


def _resvg_png_bytes(svg_text: str, output_width: Optional[int] = None) -> bytes:
    import resvg_py

    kwargs = {}
    if output_width:
        kwargs["width"] = int(output_width)
    return bytes(resvg_py.svg_to_bytes(svg_string=svg_text, **kwargs))


def write_png(svg_text: str, out_path: Path, output_width: Optional[int] = None) -> Path:
    data = svg_to_png_bytes(svg_text, output_width=output_width)
    out_path.write_bytes(data)
    return out_path
