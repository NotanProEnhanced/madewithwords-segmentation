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
import re
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


# ---- Foreground matte via onnxruntime (already a dependency -- no rembg). Default model is
# isnet-general-use: markedly better than u2net on fur edges and white-fur-on-white background.
# u2net is a lighter, revertible fallback (PET_MATTE_MODEL=u2net). ~170MB, fetched once to the
# cache dir (needs runtime internet), then reused. Any failure -> None -> GrabCut fallback. -----
_MATTE_MODELS = {
    "isnet": ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx", 1024),
    "u2net": ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx", 320),
}
_MATTE_NAME = (os.environ.get("PET_MATTE_MODEL", "isnet").strip().lower() or "isnet")
if _MATTE_NAME not in _MATTE_MODELS:
    _MATTE_NAME = "isnet"
_MATTE_URL = (os.environ.get("PET_MATTE_URL", "").strip() or _MATTE_MODELS[_MATTE_NAME][0])
_MATTE_SIZE = _MATTE_MODELS[_MATTE_NAME][1]
_MATTE_PATH = os.path.join(os.environ.get("PET_MATTE_DIR", tempfile.gettempdir()), _MATTE_NAME + ".onnx")
_U2_SESSION = None
_U2_LOCK = Lock()
_U2_FAILED = False


def _ensure_u2net():
    if os.path.exists(_MATTE_PATH) and os.path.getsize(_MATTE_PATH) > 1_000_000:
        return _MATTE_PATH
    os.makedirs(os.path.dirname(_MATTE_PATH) or ".", exist_ok=True)
    urllib.request.urlretrieve(_MATTE_URL, _MATTE_PATH)
    return _MATTE_PATH if (os.path.exists(_MATTE_PATH) and os.path.getsize(_MATTE_PATH) > 1_000_000) else None


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
    """Per-pixel foreground matte from the selected model (isnet/u2net), or None on failure.
    isnet: 1024x1024, /max, mean 0.5 std 1. u2net: 320x320, ImageNet-normalised. Output saliency
    min-max normalised either way."""
    sess = _u2net_session()
    if sess is None:
        return None
    try:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        im = cv2.resize(rgb, (_MATTE_SIZE, _MATTE_SIZE), interpolation=cv2.INTER_AREA)
        if _MATTE_NAME == "isnet":
            im = im / max(float(im.max()), 1.0) - 0.5
        else:
            im = (im / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
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


def _solidify_matte(m, w):
    """Keep the WHOLE subject. U2-Net gives a LIGHT-fur-on-WHITE neck/chest low confidence, so it
    drops out -> a 'floating head'. Threshold to a solid silhouette (largest component + filled
    interior holes), feather it, then UNION with the confident soft matte so wispy fur edges
    survive. PET_MATTE_FILL is the keep threshold (0 disables)."""
    thr = float(os.environ.get("PET_MATTE_FILL", "0.35") or 0.35)
    if thr <= 0:
        return m
    b = (m > thr).astype(np.uint8)
    if int(b.sum()) < 20:
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    if n > 1:
        b = (lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(np.uint8)
    ff = b.copy()                                             # fill interior holes (flood from a corner)
    cv2.floodFill(ff, np.zeros((b.shape[0] + 2, b.shape[1] + 2), np.uint8), (0, 0), 1)
    b = np.clip(b + (ff == 0).astype(np.uint8), 0, 1)
    solid = cv2.GaussianBlur(b.astype(np.float32), (0, 0), sigmaX=max(1.0, w * 0.006))
    return np.clip(np.maximum(m, solid), 0, 1)


def _foreground_mask(bgr):
    """Real matte first (U2-Net); GrabCut fallback when the model is unavailable. Solidified so a
    light-fur-on-white body is kept (no 'floating head')."""
    m = _u2net_mask(bgr)
    if m is None:
        m = _grabcut_mask(bgr)
    return _solidify_matte(m, bgr.shape[1])


def _detail_map(gray):
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    lap = cv2.GaussianBlur(lap, (0, 0), sigmaX=max(1.0, gray.shape[1] * 0.012))
    lap /= (np.percentile(lap, 99) + 1e-6)
    return np.clip(lap, 0, 1)


def _phrases(words):
    ph = [re.sub(r"[^A-Z0-9' ]+", "", p).strip() for p in words.upper().replace("\n", ",").split(",")]
    return [p for p in ph if p] or ["LOVE"]


def _weighted_stream(words):
    """Importance-weighted phrase stream: the NAME + lead words repeat most (up to ~3x) with
    their copies spread evenly, so the pet's name is findable across the portrait -- the
    'oh, it says his name' moment that drives the sale. Ported from the human engine's weighting."""
    ph = _phrases(words)
    n = len(ph)
    if n <= 1:
        return ph
    top = 3.2
    items = []
    for i, p in enumerate(ph):
        w = max(1, int(round(1.0 + (top - 1.0) * (1.0 - i / (n - 1)) ** 1.25)))
        for c in range(w):
            items.append(((c + 0.5) / w, i, p))
    items.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in items]


def _rows(stream, W, H, fs, rng):
    """A full-canvas grayscale INK-COVERAGE map (1 = ink, 0 = ground) of horizontal rows of the
    (importance-weighted) phrase stream at font size `fs`. Each row's horizontal start is
    jittered so a short list doesn't tile into wallpaper."""
    fs = max(6, int(round(fs)))
    font = ImageFont.truetype(_FONT, fs) if _FONT else ImageFont.load_default()
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    base = ", ".join(stream) + ",  "
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


def _render_word_portrait(bgr, mask, words, ground="dark"):
    """Sculpted landmark-free word-portrait: word rows are WARPED by the photo's luminance so
    the type drapes over the subject's form, blended across detail tiers (fine on features,
    coarse on the body), then coloured by the photo. Ported from the human displacement engine
    but driven by the U2-Net silhouette + saliency -- this module never touches that engine."""
    H, W = bgr.shape[:2]
    gbgr = GROUNDS.get(ground, GROUNDS["dark"])
    fade = ground in _FADE_GROUNDS
    bgr = _enhance_contrast(bgr, mask)                       # punch up the tonal range first
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rng = random.Random(20260813)                           # fixed seed -> deterministic
    stream = _weighted_stream(words)                        # name + lead words weighted -> findable
    det = _detail_map(gray.astype(np.uint8))                # 0..1 saliency: high on features/edges
    sc = W / 900.0                                          # size reference

    # 1) Four ink-coverage tiers, coarse -> micro. PET_TYPE_SCALE (<1 = finer type).
    _tsc = float(os.environ.get("PET_TYPE_SCALE", "0.42") or 0.42)
    tL = _rows(stream, W, H, 64 * sc * _tsc, rng)
    tM = _rows(stream, W, H, 40 * sc * _tsc, rng)
    tF = _rows(stream, W, H, 26 * sc * _tsc, rng)
    tMi = _rows(stream, W, H, 16 * sc * _tsc, rng)

    # 2) Drape: warp the rows VERTICALLY by smoothed luminance so they ride the form. Damp the
    #    warp on high-detail features (eyes/nose) so they stay crisp -- the saliency stand-in
    #    for the human engine's feature band. PET_DRAPE tunes the wrap depth.
    # Smooth the drape FIELD generously so the type rides the broad form, not sharp local
    # features -- a hard bright-fur -> dark-nose edge otherwise SHEARS the warp into a glitchy
    # blob (worst on light-furred pets with a dark nose). PET_DRAPE_SMOOTH tunes it.
    _dsm = float(os.environ.get("PET_DRAPE_SMOOTH", "0.045") or 0.045)
    D = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, W * _dsm))
    dn = np.tanh((D / 255.0 - 0.5) * 2.4) * 0.85            # soft-limit: no over-stretch on the darkest/brightest fur (kills the 'melt')
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # Feature protection: SPREAD the detail edges inward (blur) so the eye/nose INTERIORS -- a
    # flat dark pupil has ~0 local detail but sits inside a high-detail rim -- inherit their
    # rim's protection and don't get warped into a corrupted blob. Then damp the drape there.
    featdamp = np.clip(cv2.GaussianBlur(det, (0, 0), sigmaX=max(1.0, W * 0.010)) * 1.9, 0, 1)
    amp = float(os.environ.get("PET_DRAPE", "68") or 68.0) * sc * (1.0 - 0.92 * featdamp)
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
    else:
        # SHADOW LIFT: a dark ground swallows dark-fur words -- so a shadowed neck/chest reads as
        # an empty void (a "floating head"). Floor the word colour to a dim WARM ink so shadowed
        # fur still shows readable type against the dark ground. PET_SHADOW_LIFT scales it (0 off).
        _slift = float(os.environ.get("PET_SHADOW_LIFT", "1.0") or 1.0)
        if _slift > 0.0:
            col = np.maximum(col, np.array([50.0, 42.0, 34.0], np.float32) * _slift)
    a = (warped * dens * mask)[..., None]
    ground_rgb = np.full((H, W, 3), gbgr[::-1], np.float32)
    out = ground_rgb * (1.0 - a) + col * a

    # 5) Feature edge-ink: darken along real internal edges so the face reads.
    edge = (_edge_ink(gray.astype(np.uint8)) * mask)[..., None]
    ink = np.array([28.0, 24.0, 20.0], np.float32)
    k = float(os.environ.get("PET_EDGE_INK", "0.62") or 0.62)
    out = out * (1.0 - k * edge) + ink * (k * edge)

    # 6) Photographic FEATURE realism: blend the real (contrast-enhanced) photo into the render
    #    weighted by DETAIL, so the eyes / nose / whiskers gain photographic definition while the
    #    flat body stays words -- the landmark-free analogue of the human engine's real-photo eyes.
    #    PET_PHOTO scales it (0 = pure typography; higher = more photographic).
    _pf = float(os.environ.get("PET_PHOTO", "0.45") or 0.45)
    if _pf > 0.0:
        photo_rgb = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
        # Spread the weight into the feature interiors (same reason as featdamp) so the eyes/nose
        # read as the REAL photo -- covering any residual warp -- while the body stays words.
        wgt = cv2.GaussianBlur(np.clip(det * 1.6, 0, 1), (0, 0), sigmaX=max(1.0, W * 0.008)) * mask
        wgt = (wgt * _pf)[..., None]
        out = out * (1.0 - wgt) + photo_rgb * wgt

    # 7) Tonal depth: deepen shadows + lift highlights within the subject so black fur reads
    #    deep (not flat grey) and lit areas glow -- the pet analogue of the engine's 'breathe'.
    #    PET_TONAL scales it (0 = off).
    _tone = float(os.environ.get("PET_TONAL", "1.0") or 1.0)
    if _tone > 0.0:
        m3 = mask[..., None]
        o = np.clip(out / 255.0, 0, 1)
        o2 = np.power(o, 1.0 + 0.18 * _tone)                        # deepen darks
        o2 = np.clip((o2 - 0.5) * (1.0 + 0.14 * _tone) + 0.5, 0, 1)  # gentle overall contrast
        out = (o * (1.0 - m3) + o2 * m3) * 255.0

    # 8) Eye/feature crispness: a gentle unsharp within the subject so the eyes and nose snap.
    _shp = float(os.environ.get("PET_SHARPEN", "0.5") or 0.5)
    if _shp > 0.0:
        m3 = mask[..., None]
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
        out = out * (1.0 - m3) + np.clip(out + _shp * (out - blur), 0, 255) * m3

    # 9) Vibrance: boost less-saturated colours more so warm fur glows without going garish.
    _vib = float(os.environ.get("PET_VIBRANCE", "0.35") or 0.35)
    if _vib > 0.0 and not fade:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        s = hsv[..., 1] / 255.0
        hsv[..., 1] = np.clip((s + _vib * s * (1.0 - s)) * 255.0, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    # 10) Studio spotlight vignette: darken toward the canvas corners so the subject is lit like
    #     a gallery portrait. Applied to the whole canvas (ground included). PET_VIGNETTE tunes it.
    _vg = float(os.environ.get("PET_VIGNETTE", "0.32") or 0.32)
    if _vg > 0.0:
        cy0, cx0 = H * 0.46, W * 0.5
        r = np.sqrt(((xx - cx0) / (0.72 * W)) ** 2 + ((yy - cy0) / (0.72 * H)) ** 2)
        vig = np.clip(1.0 - _vg * np.clip(r - 0.45, 0, 1) ** 1.4, 0.45, 1.0)
        out = out * vig[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


def _fit_print_aspect(bgr, mask, aspect):
    """Pad the (matted) subject to a target print aspect (width/height) so the render composes
    on a proper canvas -- e.g. 0.8 = 4:5. Margins get mask=0 so the render fills them with the
    ground, and the vignette then lights the whole print. Subject is centred (a touch high)."""
    H, W = bgr.shape[:2]
    cur = W / max(1, H)
    if abs(cur - aspect) < 0.005:
        return bgr, mask
    if cur > aspect:                                   # too wide -> pad top/bottom
        newH = int(round(W / aspect)); pad = newH - H
        top = int(pad * 0.42); bot = pad - top         # subject sits slightly high (portrait framing)
        l = r = 0
    else:                                              # too tall -> pad left/right
        newW = int(round(H * aspect)); pad = newW - W
        l = pad // 2; r = pad - l; top = bot = 0
    bgr2 = cv2.copyMakeBorder(bgr, top, bot, l, r, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    mask2 = cv2.copyMakeBorder(mask, top, bot, l, r, cv2.BORDER_CONSTANT, value=0)
    return bgr2, mask2


def render_pet_portrait(image_bytes: bytes, words: str, ground: str = "dark", height: int = 900,
                        print_aspect: float | None = None) -> bytes:
    """Decode a photo, render a landmark-free word-portrait, return PNG bytes. `height` sets the
    render resolution (preview ~900; print ~4500). `print_aspect` (e.g. 0.8 for 4:5) pads the
    subject onto a gallery canvas for print -- omit for the raw preview crop."""
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    if bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (max(1, int(bgr.shape[1] * height / bgr.shape[0])), height),
                         interpolation=cv2.INTER_AREA)
    mask = _foreground_mask(bgr)
    if print_aspect:
        bgr, mask = _fit_print_aspect(bgr, mask, float(print_aspect))
    out_rgb = _render_word_portrait(bgr, mask, words, ground=ground)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("encode failed")
    return buf.tobytes()
