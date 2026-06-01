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
# Updated 2026-06-01 for the dark-ground palette set we ship in production.
_POSTER_COLORS = {
    "gold_noir":          ("#0b0c10", "#e8c66a", "#7a6a44"),
    "navy_marigold":      ("#0a1126", "#f3c34a", "#6f5e30"),
    "forest_bone":        ("#0b2618", "#f1e8d4", "#7a8c7a"),
    "burgundy_champagne": ("#2a0a11", "#e9d39a", "#7a6b53"),
    # Light-bg legacy keys still keep working in case anyone refers to them.
    "mono":               ("#ffffff", "#141414", "#777777"),
    "photo":              ("#ffffff", "#1a1a1a", "#777777"),
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


def compose_poster_png(
    portrait_png_bytes: bytes,
    ink: str,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    canvas_w: int = 2200,
) -> bytes:
    """Wrap the photo-modulated PNG portrait into a keepsake design with
    title/caption. Used when modulation_png_bytes is the source of the
    rendered output (story-style calligram with per-glyph photo tonal
    gradation); embeds the PNG verbatim rather than re-rasterising the
    SVG (which would lose the modulation pass). Returns PNG bytes."""
    import io as _io
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:  # noqa: BLE001
        # Without Pillow we can't compose; let the caller fall back.
        raise RuntimeError(f"compose_poster_png needs Pillow: {e}")
    bg_hex, fg_hex, muted_hex = _POSTER_COLORS.get(ink, _POSTER_COLORS["gold_noir"])
    def _hex(h: str):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    bg, fg, muted = _hex(bg_hex), _hex(fg_hex), _hex(muted_hex)

    portrait = Image.open(_io.BytesIO(portrait_png_bytes)).convert("RGB")
    pw, ph = portrait.size
    CW = int(canvas_w)
    side = int(round(CW * 0.085))
    box_w = CW - 2 * side
    box_h = int(round(box_w * (ph / pw)))
    top = int(round(CW * 0.075))
    title_size = int(round(CW * 0.060))
    cap_size = int(round(CW * 0.0185))
    gap = int(round(CW * 0.055))
    rule_y = top + box_h + gap

    has_title = bool(title and title.strip())
    has_cap = bool(caption and caption.strip())
    title_base = rule_y + int(title_size * 0.95) if has_title else rule_y
    cap_base = (title_base + int(cap_size * 1.9)) if has_cap else title_base
    last = cap_base if has_cap else (title_base if has_title else rule_y)
    CH = last + int(CW * 0.06)

    canvas = Image.new("RGB", (CW, CH), bg)
    portrait_resized = portrait.resize((box_w, box_h), Image.LANCZOS)
    canvas.paste(portrait_resized, (side, top))

    draw = ImageDraw.Draw(canvas)
    # Use a system-available serif. We don't ship a TTF here; PIL's default
    # works in a pinch but is sans. Try the DejaVu Serif that the deploy
    # Dockerfile installs (fonts-dejavu); fall back gracefully.
    serif_paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    )
    def _serif(size: int):
        for p in serif_paths:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
        return ImageFont.load_default()
    serif_bold = _serif(title_size) if has_title else None
    serif_cap = _serif(cap_size) if has_cap else None

    # Thin rule between portrait and title
    if has_title or has_cap:
        rx0 = int(CW * 0.40)
        rx1 = int(CW * 0.60)
        draw.line([(rx0, rule_y), (rx1, rule_y)], fill=muted, width=2)
    if has_title:
        txt = title.strip().upper()
        tw = draw.textlength(txt, font=serif_bold)
        draw.text(((CW - tw) / 2, title_base - title_size), txt,
                  font=serif_bold, fill=fg)
    if has_cap:
        txt = caption.strip().upper()
        tw = draw.textlength(txt, font=serif_cap)
        draw.text(((CW - tw) / 2, cap_base - cap_size), txt,
                  font=serif_cap, fill=muted)

    out = _io.BytesIO()
    canvas.save(out, format="PNG", optimize=True)
    return out.getvalue()
