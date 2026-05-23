"""Accurate text width measurement using the same font CairoSVG will render.

CairoSVG resolves `Arial, Helvetica, sans-serif` + bold via fontconfig, which on
this platform maps to Liberation Sans Bold (a metric-compatible Arial clone). We
measure with that exact TTF so path-fitting matches the rasterized output.
"""
from __future__ import annotations

import glob
from functools import lru_cache
from typing import Optional

from PIL import ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/**/LiberationSans-Bold.ttf",
    "/usr/share/fonts/**/Arial*Bold*.ttf",
    "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
]
_REF_SIZE = 100  # measure once at this size, scale linearly


@lru_cache(maxsize=1)
def _font() -> Optional[ImageFont.FreeTypeFont]:
    for pat in _FONT_CANDIDATES:
        matches = sorted(glob.glob(pat, recursive=True))
        if matches:
            try:
                return ImageFont.truetype(matches[0], _REF_SIZE)
            except Exception:
                continue
    return None


@lru_cache(maxsize=4096)
def _ref_width(text: str) -> float:
    f = _font()
    if f is None:
        # Fallback: rough average advance for bold sans.
        return len(text) * _REF_SIZE * 0.62
    return float(f.getlength(text))


def text_width(text: str, font_size: float) -> float:
    """Width in px of `text` rendered at `font_size`."""
    if not text:
        return 0.0
    return _ref_width(text) * (font_size / _REF_SIZE)


def have_metric_font() -> bool:
    return _font() is not None
