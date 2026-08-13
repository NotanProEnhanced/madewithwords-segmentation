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
import random
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


def _tokens(words):
    toks = [t for t in "".join(c if (c.isalnum() or c in " ,") else " " for c in words.upper()).split() if t]
    return toks or ["LOVE"]


def _rows(tokens, W, H, fs, rng):
    """A full-canvas grayscale INK-COVERAGE map (1 = ink, 0 = ground) of horizontal word
    rows at font size `fs`. Ported from the human engine's row generator (landmark-free);
    each row's horizontal start is jittered so a short list doesn't tile into wallpaper."""
    fs = max(6, int(round(fs)))
    font = ImageFont.truetype(_FONT, fs) if _FONT else ImageFont.load_default()
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    base = " ".join(tokens) + "  "
    bw = max(1.0, float(d.textlength(base, font=font)))
    line = base * max(2, int((W + fs * 7) / bw) + 2)
    y = 0
    while y < H + fs:
        d.text((-rng.randint(0, int(fs * 6)), y), line, font=font, fill=0)
        y += max(6, int(fs))
    return 1.0 - (np.asarray(im).astype(np.float32) / 255.0)


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
    """Sculpted landmark-free word-portrait: word rows are WARPED by the photo's luminance so
    the type drapes over the subject's form, blended across detail tiers (fine on features,
    coarse on the body), then coloured by the photo. Ported from the human displacement engine
    but driven by the U2-Net silhouette + saliency -- this module never touches that engine."""
    H, W = bgr.shape[:2]
    gbgr = GROUNDS.get(ground, GROUNDS["mid"])
    fade = ground in _FADE_GROUNDS
    bgr = _enhance_contrast(bgr, mask)                       # punch up the tonal range first
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rng = random.Random(20260813)                           # fixed seed -> deterministic
    toks = _tokens(words)
    det = _detail_map(gray.astype(np.uint8))                # 0..1 saliency: high on features/edges
    sc = W / 900.0                                          # size reference

    # 1) Four ink-coverage tiers, coarse -> micro. PET_TYPE_SCALE (<1 = finer type).
    _tsc = float(os.environ.get("PET_TYPE_SCALE", "0.6") or 0.6)
    tL = _rows(toks, W, H, 64 * sc * _tsc, rng)
    tM = _rows(toks, W, H, 40 * sc * _tsc, rng)
    tF = _rows(toks, W, H, 26 * sc * _tsc, rng)
    tMi = _rows(toks, W, H, 16 * sc * _tsc, rng)

    # 2) Drape: warp the rows VERTICALLY by smoothed luminance so they ride the form. Damp the
    #    warp on high-detail features (eyes/nose) so they stay crisp -- the saliency stand-in
    #    for the human engine's feature band. PET_DRAPE tunes the wrap depth.
    D = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, W * 0.02))
    dn = (D / 255.0 - 0.5) * 2.0
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    featdamp = np.clip(det * 1.3, 0, 1)
    amp = float(os.environ.get("PET_DRAPE", "82") or 82.0) * sc * (1.0 - 0.85 * featdamp)
    my = (yy + amp * dn).astype(np.float32)
    mx = xx

    def R(t):
        return cv2.remap(t, mx, my, cv2.INTER_LINEAR, borderValue=0.0)

    wL, wM, wF, wMi = R(tL), R(tM), R(tF), R(tMi)

    # 3) Blend tiers by the detail field: coarse text on flat body, fine on features.
    df = np.clip(det, 0, 1)
    warped = wL.copy()
    for a0, b0, ia, ib in ((0.0, 0.45, wL, wM), (0.45, 0.75, wM, wF), (0.75, 1.0001, wF, wMi)):
        bt = np.clip((df - a0) / (b0 - a0), 0, 1)
        warped = np.where((df >= a0) & (df < b0), ia * (1 - bt) + ib * bt, warped)
    warped = np.where(df >= 1.0, wMi, warped)               # ink density 0..1

    # 4) Colourise: the ink carries the photo's own colour, composited on the ground.
    col = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
    dens = np.ones_like(gray)
    if fade:                                                # ink-density styling (dark-furred pets)
        dens = 0.35 + 0.65 * (1.0 - gray / 255.0)
        col = col * 0.72
    a = (warped * dens * mask)[..., None]
    ground_rgb = np.full((H, W, 3), gbgr[::-1], np.float32)
    out = ground_rgb * (1.0 - a) + col * a

    # 5) Feature edge-ink: darken along real internal edges so the face reads.
    edge = (_edge_ink(gray.astype(np.uint8)) * mask)[..., None]
    ink = np.array([28.0, 24.0, 20.0], np.float32)
    k = float(os.environ.get("PET_EDGE_INK", "0.62") or 0.62)
    out = out * (1.0 - k * edge) + ink * (k * edge)

    # 6) Tonal depth: deepen shadows + lift highlights within the subject so black fur reads
    #    deep (not flat grey) and lit areas glow -- the pet analogue of the engine's 'breathe'.
    #    PET_TONAL scales it (0 = off).
    _tone = float(os.environ.get("PET_TONAL", "1.0") or 1.0)
    if _tone > 0.0:
        m3 = mask[..., None]
        o = np.clip(out / 255.0, 0, 1)
        o2 = np.power(o, 1.0 + 0.18 * _tone)                        # deepen darks
        o2 = np.clip((o2 - 0.5) * (1.0 + 0.14 * _tone) + 0.5, 0, 1)  # gentle overall contrast
        out = (o * (1.0 - m3) + o2 * m3) * 255.0
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
