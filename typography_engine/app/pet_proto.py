"""Landmark-free typographic-portrait prototype (pets / any subject), as an importable
render used by the flag-gated /pet-test page.

The production engine (displacement.py) needs a human MediaPipe FACE MESH + the selfie
person-segmenter, so it raises `displacement_needs_face` on a pet. This proves the core
idea works WITHOUT any landmarks:

  1) GrabCut foreground (no model download),
  2) Laplacian DETAIL map -> where the eyes/nose/fur are,
  3) two size tiers driven by that detail (fine on features, coarse on body),
  4) every glyph coloured by the photo, so the subject emerges FROM the words.

Species-agnostic: it never looks for a face, only for photographic detail.
This is a PROTOTYPE (crude GrabCut matte, no warp/edge-ink) -- a quality gate, not the
finished look.
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

_FONT = next((p for p in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
) if os.path.exists(p)), None)

GROUNDS = {                     # BGR
    # TRUE-TONE grounds: glyphs keep the photo's real luminance/colour (nothing faded), so a
    # black-AND-white subject keeps BOTH -- white fur reads as bright text, black fur as dark.
    "mid":      (128, 128, 128),  # neutral mid-grey  -> both extremes contrast (BEST for B&W pets)
    "dark":     (40, 26, 20),     # deep navy         -> light fur pops; dark fur can go muddy
    "charcoal": (60, 56, 52),     # warm charcoal
    # INK-DENSITY grounds (stylised): light tones fade INTO the ground, so white fur disappears.
    # Only suitable for a subject DARKER than the ground (e.g. an all-black or brown pet).
    "paper":    (232, 240, 244),  # warm ivory (ink-on-paper look; dark-furred pets only)
    "slate":    (216, 221, 226),  # cool gallery grey (dark-furred pets only)
}
_FADE_GROUNDS = ("paper", "slate")   # ink-density styling -> fades light tones (loses white fur)


def _foreground_mask(bgr):
    """GrabCut with a border-init rectangle -- no model download. Serviceable for a
    centred subject on a distinct background (the common pet-photo case)."""
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.float32)
        if float(m.mean()) < 0.03:                       # grabcut collapsed -> use the rect
            raise cv2.error("empty")
    except cv2.error:
        m = np.zeros((h, w), np.float32)
        m[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1.0
    m = cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(1.0, w * 0.004)) / 255.0
    return np.clip(m, 0, 1)


def _detail_map(gray):
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    lap = cv2.GaussianBlur(lap, (0, 0), sigmaX=max(1.0, gray.shape[1] * 0.012))
    lap /= (np.percentile(lap, 99) + 1e-6)
    return np.clip(lap, 0, 1)


def _char_stream(words):
    toks = [t for t in "".join(c if (c.isalnum() or c in " ,") else " " for c in words.upper()).split() if t]
    if not toks:
        toks = ["LOVE"]
    i = 0
    while True:
        for ch in toks[i % len(toks)]:
            yield ch
        yield " "
        i += 1


def _render_tier(bgr, mask, size, words, fade):
    h, w = bgr.shape[:2]
    font = ImageFont.truetype(_FONT, size) if _FONT else ImageFont.load_default()
    layer = Image.new("RGB", (w, h), (0, 0, 0))
    alpha = Image.new("L", (w, h), 0)
    dl, da = ImageDraw.Draw(layer), ImageDraw.Draw(alpha)
    cw = max(3, int(font.getlength("M")))
    lh = int(size * 1.02)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    stream = _char_stream(words)
    y = 0
    while y < h:
        x = 0
        while x < w:
            ch = next(stream)
            cx, cy = min(w - 1, x + cw // 2), min(h - 1, y + size // 2)
            if ch != " " and mask[cy, cx] > 0.5:
                b, g, r = bgr[cy, cx]
                col = (int(r), int(g), int(b))
                if fade:                                  # ink-density styling (paper/slate)
                    a = int(255 * (0.35 + 0.65 * (1.0 - gray[cy, cx])))
                    col = tuple(int(c * 0.72) for c in col)
                else:                                     # true-tone: keep the photo's real luminance
                    a = 255
                dl.text((x, y), ch, font=font, fill=col)
                da.text((x, y), ch, font=font, fill=a)
            x += cw
        y += lh
    return np.asarray(layer, np.float32), np.asarray(alpha, np.float32) / 255.0


def _render_word_portrait(bgr, mask, words, ground="mid"):
    h, w = bgr.shape[:2]
    gbgr = GROUNDS.get(ground, GROUNDS["mid"])
    fade = ground in _FADE_GROUNDS
    det = _detail_map(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    base = max(9, int(round(w / 46)))
    fine = max(6, int(round(base * 0.55)))
    c_rgb, c_a = _render_tier(bgr, mask, base, words, fade)
    f_rgb, f_a = _render_tier(bgr, mask, fine, words, fade)
    sel = np.clip((det - 0.32) / 0.33, 0, 1)[..., None]
    rgb = c_rgb * (1 - sel) + f_rgb * sel
    a = (c_a[..., None] * (1 - sel) + f_a[..., None] * sel) * mask[..., None]
    ground_rgb = np.full((h, w, 3), gbgr[::-1], np.float32)
    out = np.clip(ground_rgb * (1 - a) + rgb * a, 0, 255).astype(np.uint8)   # RGB
    return out


def render_pet_portrait(image_bytes: bytes, words: str, ground: str = "mid", height: int = 900) -> bytes:
    """Decode a photo, render a landmark-free word-portrait, return PNG bytes."""
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    if bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (max(1, int(bgr.shape[1] * height / bgr.shape[0])), height),
                         interpolation=cv2.INTER_AREA)
    mask = _foreground_mask(bgr)
    out_rgb = _render_word_portrait(bgr, mask, words, ground=ground)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("encode failed")
    return buf.tobytes()
