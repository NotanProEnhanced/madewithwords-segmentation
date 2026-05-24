"""Designed-composition layer: wrap a bare word-portrait into a titled keepsake.

Opt-in. Embeds the portrait SVG (which uses no <defs>/xlink, so it nests
cleanly) into a larger poster canvas with generous margins, a thin rule, the
subject's name in an elegant serif, and an optional custom caption/date. Colours
are derived from the chosen ink so the framing matches the portrait (navy ink ->
navy on white, gold_noir -> gold on charcoal).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from .svgbuild import esc, require_hex

_SERIF = "Georgia, 'Times New Roman', serif"

# Per-ink poster colours: (background, title/ink, muted caption+rule).
_POSTER_COLORS = {
    "navy":      ("#ffffff", "#0d1b3a", "#6b7790"),
    "sepia":     ("#fbf7ee", "#2e1c0a", "#8a795f"),
    "burgundy":  ("#ffffff", "#4a0d18", "#9a6b72"),
    "forest":    ("#ffffff", "#0f2e1e", "#5f7d6b"),
    "gold_noir": ("#101216", "#e8c66a", "#8a7a4e"),
    "mono":      ("#ffffff", "#141414", "#777777"),
    "photo":     ("#ffffff", "#1a1a1a", "#777777"),
}


def _portrait_inner(portrait_svg: str) -> Tuple[str, float, float]:
    """Strip the XML decl and outer <svg> wrapper, returning inner content plus
    the portrait's intrinsic width/height (from its viewBox)."""
    m = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', portrait_svg)
    w, h = (float(m.group(1)), float(m.group(2))) if m else (1000.0, 1250.0)
    body = re.sub(r"^<\?xml[^>]*\?>\s*", "", portrait_svg.strip())
    body = re.sub(r"^<svg\b[^>]*>", "", body, count=1)
    body = re.sub(r"</svg>\s*$", "", body.strip())
    return body, w, h


def compose_poster(
    portrait_svg: str,
    ink: str,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    canvas_w: int = 1500,
) -> str:
    bg, fg, muted = _POSTER_COLORS.get(ink, _POSTER_COLORS["mono"])
    for c in (bg, fg, muted):
        require_hex(c)

    inner, pw0, ph0 = _portrait_inner(portrait_svg)
    CW = float(canvas_w)
    side = CW * 0.085
    box_w = CW - 2 * side
    box_h = box_w * (ph0 / pw0)
    top = CW * 0.075

    has_title = bool(title and title.strip())
    has_cap = bool(caption and caption.strip())

    title_size = CW * 0.060
    cap_size = CW * 0.0185
    gap = CW * 0.055          # portrait -> rule
    rule_y = top + box_h + gap
    title_base = rule_y + title_size * 0.95 if has_title else rule_y
    cap_base = (title_base + cap_size * 1.9) if has_cap else title_base
    last = cap_base if has_cap else (title_base if has_title else rule_y)
    CH = last + CW * 0.06

    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{CW:.0f}" height="{CH:.0f}" viewBox="0 0 {CW:.0f} {CH:.0f}">',
        f'<rect x="0" y="0" width="{CW:.0f}" height="{CH:.0f}" fill="{bg}" />',
        f'<svg x="{side:.1f}" y="{top:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" '
        f'viewBox="0 0 {pw0:.0f} {ph0:.0f}" preserveAspectRatio="xMidYMid meet">{inner}</svg>',
    ]
    if has_title or has_cap:
        rx0, rx1 = CW * 0.40, CW * 0.60
        parts.append(
            f'<line x1="{rx0:.1f}" y1="{rule_y:.1f}" x2="{rx1:.1f}" y2="{rule_y:.1f}" '
            f'stroke="{muted}" stroke-width="2" />'
        )
    if has_title:
        parts.append(
            f'<text x="{CW/2:.1f}" y="{title_base:.1f}" text-anchor="middle" '
            f'font-family="{esc(_SERIF)}" font-size="{title_size:.1f}" font-weight="bold" '
            f'letter-spacing="{CW*0.004:.1f}" fill="{fg}">{esc(title.strip().upper())}</text>'
        )
    if has_cap:
        parts.append(
            f'<text x="{CW/2:.1f}" y="{cap_base:.1f}" text-anchor="middle" '
            f'font-family="{esc(_SERIF)}" font-size="{cap_size:.1f}" '
            f'letter-spacing="{CW*0.010:.1f}" fill="{muted}">{esc(caption.strip().upper())}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
