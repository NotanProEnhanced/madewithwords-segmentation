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

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

_NAVY = (13, 27, 58)

# The preview wordmark is brand-aware: on a PARTNER skin we must NOT stamp
# "typortrait.com" -- it would advertise the direct site to a customer who came
# through the partner (diverting the sale + their commission) and break the
# white-label. First-party brands use their own domain; partner skins use the
# partner's name; anything else falls back to typortrait.com.
_BRAND_MARK = {
    "lovedinwords": "lovedinwords.com",
    "everloved": "Ever Loved",
}


def _brand_mark(brand: str) -> str:
    return _BRAND_MARK.get((brand or "").strip().lower(), "typortrait.com")


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


def _tile_watermark(im: Image.Image, mark: str = "typortrait.com", alpha: int = 120) -> Image.Image:
    """A sparse repeating diagonal mark confined to the SUBJECT (the figure), not the
    background -- fewer marks, a clean background, and unccroppable (you can't crop out
    the face). White text with a dark outline so it reads on dark and light grounds."""
    W, H = im.size
    wf = _font(max(14, int(W * 0.023)))
    diag = int(math.hypot(W, H)) + 4               # big enough that the rotation still covers every corner
    layer = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    mw = ld.textlength(mark, font=wf)
    step_x = int(mw + max(70, W * 0.22))           # sparse -> few marks
    step_y = max(46, int(H * 0.17))
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
    # Confine the mark to the figure: multiply its alpha by the subject mask.
    layer.putalpha(ImageChops.multiply(layer.split()[3], _subject_mask(im)))
    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


def _subject_mask(im: Image.Image) -> Image.Image:
    """Rough 'figure vs flat ground' mask. The preview's background is a flat ground
    colour (renders run bg-removed), so pixels far from the sampled corner colour are
    the subject. Returns an L image (255 = subject) used to confine the watermark to
    the figure so the background stays clean."""
    arr = np.asarray(im.convert("RGB")).astype(np.int16)
    H, W = arr.shape[:2]
    cs = max(6, int(min(W, H) * 0.02))
    corners = np.concatenate([
        arr[:cs, :cs].reshape(-1, 3), arr[:cs, -cs:].reshape(-1, 3),
        arr[-cs:, :cs].reshape(-1, 3), arr[-cs:, -cs:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    msk = Image.fromarray(((dist > 42) * 255).astype(np.uint8), "L")
    # close pinholes, then soften the edge so marks don't cling to the silhouette rim
    return msk.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))


def add_watermark(png_bytes: bytes, brand: str = "", mark: str = "") -> bytes:
    """Burn the free-preview watermark into a render: a single sparse diagonal
    wordmark over the figure only (no QR, no corner plate). The wordmark is
    BRAND-AWARE -- partner skins never show "typortrait.com" (see _BRAND_MARK).
    Preview-only -- the paid download recomposes the clean PNG without this."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    im = _tile_watermark(im, mark=(mark or _brand_mark(brand)))
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()
