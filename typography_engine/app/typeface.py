"""The one place a typeface is opened.

Both engines drew in DejaVu Sans Bold because it was on the box. TYPO_FONT names a font
file (a path, or a file name under typography_engine/fonts/, which ships open-licence
faces from Google Fonts); unset keeps the box font, so every existing render is
unchanged. TYPO_FONT_WEIGHT sets the weight axis of a variable font (a static face
ignores it). Sizes may be fractional: the sharp print paths draw at exactly the working
size times the print scale.

Fail-safe: a face that cannot be opened falls back to the box font, then to Pillow's
default, and a warning is printed once.
"""
from __future__ import annotations

import os
from typing import Optional

from PIL import ImageFont

from . import settings as _settings

_FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
_warned = set()


def font_file(fallback: Optional[str] = None) -> Optional[str]:
    """The face in force: TYPO_FONT resolved to a file, else `fallback` (the box font)."""
    name = (_settings.raw("TYPO_FONT") or "").strip()
    if name:
        for cand in (name, os.path.join(_FONTS_DIR, name)):
            if os.path.isfile(cand):
                return cand
        if name not in _warned:
            _warned.add(name)
            print(f"[typeface] TYPO_FONT={name!r} not found; using the box font")
    return fallback


def font_id() -> str:
    """A short identity for cache keys: changes whenever the face or weight would."""
    return f"{_settings.raw('TYPO_FONT') or ''}|{_settings.raw('TYPO_FONT_WEIGHT') or ''}"


def load(size: float, fallback: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """Open the face in force at `size` (fractional allowed), weight axis applied."""
    path = font_file(fallback)
    if not path:
        return ImageFont.load_default()
    try:
        f = ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001
        try:
            f = ImageFont.truetype(path, max(1, int(size)))
        except Exception:  # noqa: BLE001
            if fallback and fallback != path:
                return ImageFont.truetype(fallback, max(1, int(size)))
            return ImageFont.load_default()
    w = (_settings.raw("TYPO_FONT_WEIGHT") or "").strip()
    if w:
        try:
            axes = f.get_variation_axes()
            vals = []
            for a in axes:
                nm = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
                vals.append(float(w) if nm.lower().startswith("weight") else float(a["default"]))
            f.set_variation_by_axes(vals)
        except Exception:  # noqa: BLE001 -- a static face has no axes
            pass
    return f
