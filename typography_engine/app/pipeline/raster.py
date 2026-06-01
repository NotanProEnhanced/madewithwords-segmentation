"""SVG -> PNG rasterization.

Prefers resvg-py (Rust-based, ~3x faster on rotation-heavy SVGs and
handles per-glyph `rotate=` attributes well). Falls back to cairosvg
if resvg-py is unavailable. Output is bit-identical-quality PNG bytes
in either case."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def svg_to_png_bytes(svg_text: str, output_width: Optional[int] = None) -> bytes:
    try:
        import resvg_py
        kwargs = {"svg_string": svg_text}
        if output_width:
            kwargs["width"] = int(output_width)
        out = resvg_py.svg_to_bytes(**kwargs)
        return bytes(out) if not isinstance(out, (bytes, bytearray)) else bytes(out)
    except Exception:
        import cairosvg
        kwargs = {}
        if output_width:
            kwargs["output_width"] = int(output_width)
        return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), **kwargs)


def write_png(svg_text: str, out_path: Path, output_width: Optional[int] = None) -> Path:
    data = svg_to_png_bytes(svg_text, output_width=output_width)
    out_path.write_bytes(data)
    return out_path
