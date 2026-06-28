"""Burn a free-preview watermark into a rendered PNG.

Adds two removable marks (removed only in the paid, clean download):
  * the text "Typortrait.com"
  * a QR code linking back to the site

Both sit on small translucent-white plates so they read on any background. The
watermark is rasterised into the pixels (not an overlay element), so it cannot
be stripped from the public preview.
"""
from __future__ import annotations

import glob
import io
import math
from functools import lru_cache
from typing import Optional

import qrcode
from PIL import Image, ImageDraw, ImageFont

_NAVY = (13, 27, 58)


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for pat in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/**/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",          # local dev (Windows) fallbacks
        "C:/Windows/Fonts/seguisb.ttf",
    ):
        m = sorted(glob.glob(pat, recursive=True))
        if m:
            return ImageFont.truetype(m[0], size)
    return ImageFont.load_default()


def _tile_watermark(im: Image.Image, mark: str = "typortrait.com", alpha: int = 66) -> Image.Image:
    """Faint repeating diagonal mark across the WHOLE image so it can't be cropped
    off (the corner QR/label can). White text with a dark outline, both at low
    alpha, so it stays legible on dark grounds AND light/paper grounds without
    obscuring the art. Burned into the pixels."""
    W, H = im.size
    wf = _font(max(14, int(W * 0.024)))
    diag = int(math.hypot(W, H)) + 4               # big enough that the rotation still covers every corner
    layer = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    mw = ld.textlength(mark, font=wf)
    step_x = int(mw + max(48, W * 0.12))
    step_y = max(30, int(H * 0.105))
    row = 0
    for yy in range(0, diag, step_y):
        off = (step_x // 2) if (row % 2) else 0    # brick offset so columns don't line up
        for xx in range(-step_x, diag, step_x):
            ld.text((xx + off, yy), mark, font=wf, fill=(255, 255, 255, alpha),
                    stroke_width=1, stroke_fill=(0, 0, 0, alpha))
        row += 1
    layer = layer.rotate(30, resample=Image.BICUBIC, expand=False)
    cx, cy = (diag - W) // 2, (diag - H) // 2
    layer = layer.crop((cx, cy, cx + W, cy + H))
    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


@lru_cache(maxsize=4)
def _qr(url: str) -> Image.Image:
    qr = qrcode.QRCode(border=2, box_size=10, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="#0d1b3a", back_color="white").convert("RGB")


def add_watermark(png_bytes: bytes, url: str = "https://typortrait.com",
                  label: str = "Typortrait.com") -> bytes:
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    W, H = im.size
    # Unccroppable tiled mark first (background), then the corner QR + label on top.
    im = _tile_watermark(im)
    d = ImageDraw.Draw(im, "RGBA")
    m = max(10, int(W * 0.028))
    plate = (255, 255, 255, 232)

    # QR (bottom-right)
    q = max(60, int(W * 0.13))
    qr = _qr(url).resize((q, q), Image.NEAREST)
    pad = max(4, int(q * 0.06))
    d.rounded_rectangle([W - q - m - pad, H - q - m - pad, W - m + pad, H - m + pad],
                        radius=int(q * 0.08), fill=plate)
    im.paste(qr, (W - q - m, H - q - m))

    # Label (bottom-left)
    fs = max(16, int(W * 0.034))
    font = _font(fs)
    tw = d.textlength(label, font=font)
    tp = int(fs * 0.45)
    d.rounded_rectangle([m, H - m - fs - 2 * tp, m + tw + 2 * tp, H - m],
                        radius=int(fs * 0.4), fill=plate)
    d.text((m + tp, H - m - fs - tp), label, font=font, fill=_NAVY)

    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()
