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
import tempfile
import urllib.request
from threading import Lock

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


def _grabcut_mask(bgr):
    """GrabCut border-init FALLBACK (no model). Serviceable for a centred subject on a
    distinct background; struggles on white fur against a white background."""
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


# ---- U2-Net foreground matte (general objects: pets, any subject). Run directly through
# onnxruntime (already a dependency) -- no rembg, so no extra deps to conflict with. The
# ~176MB model is fetched once to a cache dir on first use (needs runtime internet), then
# reused. Any failure -> None, and the caller falls back to GrabCut. --------------------
_U2NET_URL = os.environ.get(
    "PET_MATTE_URL",
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx")
_U2NET_PATH = os.path.join(os.environ.get("PET_MATTE_DIR", tempfile.gettempdir()), "u2net.onnx")
_U2_SESSION = None
_U2_LOCK = Lock()
_U2_FAILED = False


def _ensure_u2net():
    if os.path.exists(_U2NET_PATH) and os.path.getsize(_U2NET_PATH) > 1_000_000:
        return _U2NET_PATH
    os.makedirs(os.path.dirname(_U2NET_PATH) or ".", exist_ok=True)
    urllib.request.urlretrieve(_U2NET_URL, _U2NET_PATH)
    return _U2NET_PATH if (os.path.exists(_U2NET_PATH) and os.path.getsize(_U2NET_PATH) > 1_000_000) else None


def _u2net_session():
    global _U2_SESSION, _U2_FAILED
    with _U2_LOCK:
        if _U2_SESSION is not None or _U2_FAILED:
            return _U2_SESSION
        try:
            import onnxruntime as ort  # already a project dependency
            path = _ensure_u2net()
            if not path:
                _U2_FAILED = True
                return None
            _U2_SESSION = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        except Exception:  # noqa: BLE001  -- any failure -> fall back to GrabCut
            _U2_FAILED = True
            _U2_SESSION = None
    return _U2_SESSION


def _u2net_mask(bgr):
    """Per-pixel foreground matte from U2-Net, or None on any failure. Standard U2-Net I/O:
    RGB resized to 320x320, ImageNet-normalised; output saliency min-max normalised."""
    sess = _u2net_session()
    if sess is None:
        return None
    try:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA)
        im = (im - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        inp = np.transpose(im, (2, 0, 1))[None].astype(np.float32)
        pred = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0, 0]
        mn, mx = float(pred.min()), float(pred.max())
        pred = (pred - mn) / (mx - mn + 1e-8)
        m = cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(1.0, w * 0.0015))
        if float((m > 0.5).mean()) < 0.004:      # matte collapsed -> not usable
            return None
        return np.clip(m, 0, 1)
    except Exception:  # noqa: BLE001
        return None


def _foreground_mask(bgr):
    """Real matte first (U2-Net, clean fur-vs-background even white-on-white); GrabCut only
    as a fallback when the model is unavailable."""
    m = _u2net_mask(bgr)
    return m if m is not None else _grabcut_mask(bgr)


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


def _enhance_contrast(bgr, mask):
    """Stretch the subject's tones to the full range so black fur reads black and white fur
    white. The flat grey wash came from compressed midtones -- this is the punch that makes a
    black-and-white pet read on any ground."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inside = gray[mask > 0.5]
    if inside.size < 50:
        return bgr
    lo, hi = float(np.percentile(inside, 2.0)), float(np.percentile(inside, 98.0))
    if hi - lo < 24.0:
        hi = lo + 24.0
    out = (bgr.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def _edge_ink(gray):
    """Strong internal edges (eyes, nose, muzzle line, fur boundaries) as a 0..1 field."""
    e = cv2.Canny(gray, 45, 130).astype(np.float32) / 255.0
    return np.clip(cv2.GaussianBlur(e, (0, 0), sigmaX=1.1), 0, 1)


def _render_word_portrait(bgr, mask, words, ground="mid"):
    h, w = bgr.shape[:2]
    gbgr = GROUNDS.get(ground, GROUNDS["mid"])
    fade = ground in _FADE_GROUNDS
    bgr = _enhance_contrast(bgr, mask)                  # punch up the tonal range first
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    det = _detail_map(gray)
    base = max(9, int(round(w / 46)))
    fine = max(6, int(round(base * 0.55)))
    c_rgb, c_a = _render_tier(bgr, mask, base, words, fade)
    f_rgb, f_a = _render_tier(bgr, mask, fine, words, fade)
    sel = np.clip((det - 0.26) / 0.34, 0, 1)[..., None]     # a touch more fine text on features
    rgb = c_rgb * (1 - sel) + f_rgb * sel
    a = (c_a[..., None] * (1 - sel) + f_a[..., None] * sel) * mask[..., None]
    ground_rgb = np.full((h, w, 3), gbgr[::-1], np.float32)
    out = ground_rgb * (1 - a) + rgb * a
    # Feature definition: darken along real internal edges so the FACE reads (eyes, nose,
    # muzzle, fur boundaries) instead of a flat text field. Subtle; PET_EDGE_INK tunes it.
    edge = (_edge_ink(gray) * mask)[..., None]
    ink = np.array([28.0, 24.0, 20.0], np.float32)      # near-black warm ink (RGB)
    k = float(os.environ.get("PET_EDGE_INK", "0.5") or 0.5)
    out = out * (1 - k * edge) + ink * (k * edge)
    return np.clip(out, 0, 255).astype(np.uint8)


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
