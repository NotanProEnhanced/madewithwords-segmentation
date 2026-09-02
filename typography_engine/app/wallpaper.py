"""Wallpaper set — turn a finished 4:5 print master into device-shaped keepsakes.

The digital download that ships with every LovedInWords / EverLoved keepsake is not
just the print file: it's the same portrait re-matted for a phone lock screen, a
desktop background, and a square for sharing — so the loved one is *with you
everywhere*, not only on a wall. That "digital everywhere" bundle is a zero-COGS
value-add and a genuine differentiator versus flowers or a plain print.

Approach: never crop the face. Each target is a solid canvas in the portrait's own
ground color (sampled from the master's border, so the seam is invisible) with the
whole portrait scaled to *contain* inside it, centered, plus a whisper of vignette for
depth. Pure Pillow + NumPy (both already dependencies); no new packages.
"""
from __future__ import annotations

import io
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def _ground_color(im: Image.Image) -> Tuple[int, int, int]:
    """Median color of the master's outer border — the flat ground the portrait
    sits on. Robust to the odd stray bright letter that reaches the edge."""
    a = np.asarray(im.convert("RGB"))
    h, w = a.shape[:2]
    b = max(2, int(min(h, w) * 0.02))
    border = np.concatenate([
        a[:b, :, :].reshape(-1, 3), a[-b:, :, :].reshape(-1, 3),
        a[:, :b, :].reshape(-1, 3), a[:, -b:, :].reshape(-1, 3),
    ], axis=0)
    return tuple(int(v) for v in np.median(border, axis=0))


def _vignette(size: Tuple[int, int], strength: float = 0.16) -> Image.Image:
    """A soft radial darkening mask (L) — subtle richness at the edges."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    d = np.clip(d / np.sqrt(2), 0, 1)
    darken = (d ** 2.2) * strength
    return Image.fromarray((darken * 255).astype("uint8"), "L")


def _mat(master: Image.Image, tw: int, th: int, ground: Tuple[int, int, int],
         fit_w: float, fit_h: float, center_y: float = 0.5,
         vignette: float = 0.16) -> Image.Image:
    """Place the whole portrait on a `tw x th` ground canvas, scaled to fit within
    `fit_w`/`fit_h` of the canvas (contain — never cropped), centered at `center_y`."""
    canvas = Image.new("RGB", (tw, th), ground)
    if vignette > 0:
        dark = Image.new("RGB", (tw, th), (0, 0, 0))
        canvas = Image.composite(dark, canvas, _vignette((tw, th), vignette))
    mw, mh = master.size
    scale = min((tw * fit_w) / mw, (th * fit_h) / mh)
    nw, nh = max(1, int(mw * scale)), max(1, int(mh * scale))
    art = master.resize((nw, nh), Image.LANCZOS)
    x = (tw - nw) // 2
    y = int(th * center_y - nh / 2)
    canvas.paste(art, (x, max(0, min(th - nh, y))))
    return canvas


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_wallpaper_set(master_png: bytes) -> Dict[str, bytes]:
    """From the 4:5 print master, return {'phone','desktop','square'} PNG bytes.

    - phone   1290x2796 (9:19.5) — lock screen; face sits above where you grip it.
    - desktop 2560x1440 (16:9)   — background; portrait centered, matted sides.
    - square  2048x2048 (1:1)    — social / profile sharing.
    """
    master = Image.open(io.BytesIO(master_png)).convert("RGB")
    g = _ground_color(master)
    return {
        "phone": _png(_mat(master, 1290, 2796, g, fit_w=0.90, fit_h=0.60, center_y=0.50)),
        "desktop": _png(_mat(master, 2560, 1440, g, fit_w=0.42, fit_h=0.86, center_y=0.50)),
        "square": _png(_mat(master, 2048, 2048, g, fit_w=0.90, fit_h=0.90, center_y=0.50)),
    }
