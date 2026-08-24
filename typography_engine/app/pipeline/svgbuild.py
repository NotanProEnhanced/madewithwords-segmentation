"""SVG document assembly.

Strict rules (from spec):
  * Hex colors only (no rgba()/hsl()).
  * Output must be well-formed XML that validates.
  * Text content is XML-escaped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional
from xml.sax.saxutils import escape as _xml_escape

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def require_hex(color: str) -> str:
    if not _HEX_RE.match(color or ""):
        raise ValueError(f"Color must be hex (e.g. #000000); got {color!r}")
    return color


def esc(text: str) -> str:
    return _xml_escape(text, {'"': "&quot;"})


@dataclass
class SvgDoc:
    width: float
    height: float
    background: str = "#ffffff"
    # When set, an embedded base64 PNG/JPEG used as the canvas instead of (or
    # behind) the solid background. This is what enables photo-underlay mode:
    # the original portrait shows through and the typography overlays on top.
    bg_image_b64: Optional[str] = None
    bg_image_mime: str = "image/png"
    # Uniform inset (fraction of each side) so contours touching the image edge
    # are scaled inward, leaving whitespace and preventing glyph clipping.
    content_margin: float = 0.0
    defs: List[str] = field(default_factory=list)
    body: List[str] = field(default_factory=list)

    def add_def(self, fragment: str) -> None:
        self.defs.append(fragment)

    def add(self, fragment: str) -> None:
        self.body.append(fragment)

    def add_path(
        self,
        path_id: str,
        d: str,
        stroke: Optional[str] = None,
        fill: str = "none",
        stroke_width: float = 1.0,
    ) -> None:
        attrs = [f'id="{esc(path_id)}"', f'd="{esc(d)}"', f'fill="{fill if fill == "none" else require_hex(fill)}"']
        if stroke is not None:
            attrs.append(f'stroke="{require_hex(stroke)}"')
            attrs.append(f'stroke-width="{stroke_width}"')
            attrs.append('stroke-linecap="round"')
            attrs.append('stroke-linejoin="round"')
        self.body.append(f"<path {' '.join(attrs)} />")

    def add_def_path(self, path_id: str, d: str) -> None:
        """Path placed in <defs> purely as a textPath target (not drawn)."""
        self.defs.append(f'<path id="{esc(path_id)}" d="{esc(d)}" />')

    def add_text_on_path(
        self,
        path_id: str,
        d: str,
        text: str,
        font_size: float,
        fill: str = "#000000",
        font_family: str = "Arial, Helvetica, sans-serif",
        font_weight: str = "bold",
        letter_spacing: float = 0.0,
        start_offset: str = "0%",
    ) -> None:
        """Register the target path in defs and flow text along it."""
        self.add_def_path(path_id, d)
        ls = f' letter-spacing="{letter_spacing}"' if letter_spacing else ""
        self.body.append(
            f'<text font-family="{esc(font_family)}" font-size="{font_size}" '
            f'font-weight="{esc(font_weight)}" fill="{require_hex(fill)}"{ls}>'
            f'<textPath xlink:href="#{esc(path_id)}" startOffset="{esc(start_offset)}">'
            f"{esc(text)}</textPath></text>"
        )

    def to_svg(self) -> str:
        require_hex(self.background)
        defs = "\n".join(self.defs)
        body = "\n".join(self.body)
        m = max(0.0, min(0.3, self.content_margin))
        if m > 0:
            s = 1.0 - 2.0 * m
            tx = m * self.width
            ty = m * self.height
            body = f'<g transform="translate({tx},{ty}) scale({s})">\n{body}\n</g>'
        # Background: solid rect always (so the canvas has a colour fallback),
        # then -- when supplied -- the source photo on top, scaled to fill.
        bg_layer = (
            f'  <rect x="0" y="0" width="{self.width}" height="{self.height}" '
            f'fill="{self.background}" />\n'
        )
        if self.bg_image_b64:
            bg_layer += (
                f'  <image x="0" y="0" width="{self.width}" height="{self.height}" '
                f'preserveAspectRatio="xMidYMid slice" '
                f'xlink:href="data:{self.bg_image_mime};base64,{self.bg_image_b64}" />\n'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'{bg_layer}'
            f"  <defs>\n{defs}\n  </defs>\n"
            f"{body}\n"
            "</svg>\n"
        )


def validate_svg(svg_text: str) -> None:
    """Parse the SVG with a strict XML parser; raise if malformed.

    Also rejects rgba()/hsl() color usage which several SVG consumers reject.
    """
    import xml.etree.ElementTree as ET

    ET.fromstring(svg_text)  # raises ParseError if not well-formed
    if "rgba(" in svg_text or "hsl(" in svg_text or "rgb(" in svg_text:
        raise ValueError("SVG contains unsupported color function (rgb/rgba/hsl); use hex only.")
