"""Landmark-free typographic-portrait prototype (pets / any subject), as an importable
render used by the flag-gated /pet-test page.

The production engine (displacement.py) needs a human MediaPipe FACE MESH + the selfie
person-segmenter, so it raises `displacement_needs_face` on a pet. This proves the core
idea works WITHOUT any landmarks:

  1) GrabCut foreground (no model download),
  2) Laplacian DETAIL map -> where the eyes/nose/fur are,
  3) two size tiers driven by that detail (fine on features, coarse on body),
  4) every glyph colored by the photo, so the subject emerges FROM the words.

Species-agnostic: it never looks for a face, only for photographic detail.
This is a PROTOTYPE (crude GrabCut matte, no warp/edge-ink) -- a quality gate, not the
finished look.
"""
from __future__ import annotations

import os
import random
import re
import tempfile
import time
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
    # TRUE-TONE grounds: glyphs keep the photo's real luminance/color (nothing faded), so a
    # black-AND-white subject keeps BOTH -- white fur reads as bright text, black fur as dark.
    "mid":      (128, 128, 128),  # neutral mid-gray  -> both extremes contrast (BEST for B&W pets)
    "dark":     (40, 26, 20),     # deep navy         -> light fur pops; dark fur can go muddy
    "charcoal": (60, 56, 52),     # warm charcoal
    # INK-DENSITY grounds (stylised): light tones fade INTO the ground, so white fur disappears.
    # Only suitable for a subject DARKER than the ground (e.g. an all-black or brown pet).
    "paper":    (232, 240, 244),  # warm ivory (ink-on-paper look; dark-furred pets only)
    "slate":    (216, 221, 226),  # cool gallery gray (dark-furred pets only)
}
_FADE_GROUNDS = ("paper", "slate")   # ink-density styling -> fades light tones (loses white fur)


def _grabcut_mask(bgr):
    """GrabCut border-init FALLBACK (no model). Serviceable for a centered subject on a
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
# Both models are ~170MB. Used to reject a TRUNCATED file instead of handing it to
# onnxruntime; the old check was 1MB, which a 170MB download passes almost instantly.
_MATTE_MIN_BYTES = 100_000_000
_U2_SESSION = None
_U2_LOCK = Lock()
_U2_FAILED_AT = 0.0        # monotonic time of the last failed attempt; 0 = none
_U2_RETRY_AFTER = 300.0    # seconds to wait before trying again


def _ensure_u2net():
    """Return the model path, or None.

    Downloads to a TEMPORARY file in the same directory and renames it into place, so
    a partly-written model is never visible at the final path.

    This mattered. urlretrieve() wrote straight to _MATTE_PATH, and a 170MB download
    passes a 1MB size check within a fraction of a second -- so a render arriving
    during the download saw a file that looked complete, loaded a truncated ONNX,
    and raised. The failure then latched for the life of the process and every later
    render fell back to GrabCut's rectangle mask. That is what produced the near-black
    portraits on Loved in Words the first time the pet engine ran there, two minutes
    after the model started downloading.
    """
    if os.path.exists(_MATTE_PATH):
        if os.path.getsize(_MATTE_PATH) >= _MATTE_MIN_BYTES:
            return _MATTE_PATH
        try:
            os.remove(_MATTE_PATH)     # truncated by an interrupted earlier attempt
        except OSError:
            return None
    d = os.path.dirname(_MATTE_PATH) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".part")   # same filesystem -> atomic rename
    os.close(fd)
    try:
        urllib.request.urlretrieve(_MATTE_URL, tmp)
        if os.path.getsize(tmp) < _MATTE_MIN_BYTES:
            return None
        os.replace(tmp, _MATTE_PATH)
        return _MATTE_PATH
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _u2net_session():
    """The ONNX session, or None to fall back to GrabCut.

    A failure used to latch permanently: one transient error -- a slow download, a
    truncated file, a momentary memory shortage -- degraded EVERY later render in the
    process to the GrabCut rectangle, silently, until someone restarted the container.
    For a portrait people pay for, that is the worst possible failure shape: it looks
    like the product working, and it stays broken. Back off and retry instead, so a
    transient fault costs one render rather than all of them.
    """
    global _U2_SESSION, _U2_FAILED_AT
    with _U2_LOCK:
        if _U2_SESSION is not None:
            return _U2_SESSION
        if _U2_FAILED_AT and (time.monotonic() - _U2_FAILED_AT) < _U2_RETRY_AFTER:
            return None
        try:
            import onnxruntime as ort  # already a project dependency
            path = _ensure_u2net()
            if not path:
                _U2_FAILED_AT = time.monotonic()
                return None
            _U2_SESSION = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            _U2_FAILED_AT = 0.0
        except Exception:  # noqa: BLE001  -- any failure -> fall back to GrabCut
            _U2_FAILED_AT = time.monotonic()
            _U2_SESSION = None
    return _U2_SESSION


def _u2net_mask(bgr):
    """Per-pixel foreground matte from the selected model (isnet/u2net), or None on failure.
    isnet: 1024x1024, /max, mean 0.5 std 1. u2net: 320x320, ImageNet-normalized. Output saliency
    min-max normalized either way."""
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
    # Fill ONLY true interior holes. PAD a background frame first so background pockets walled
    # off by the subject touching the image edge (ears/shoulders in a tight crop) stay connected
    # to the border and are NOT mistaken for holes -- otherwise they fill and leak type into the
    # background at the sides.
    padded = cv2.copyMakeBorder(b, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 1)
    holes = (padded[1:-1, 1:-1] == 0).astype(np.uint8)       # 0s unreachable from the border = real holes
    # ...except when TWO subjects stand together. The background between them is walled off
    # by the pair and is, to a flood fill from the border, indistinguishable from a hole
    # inside one body. Measured on a two-person portrait it filled as subject and rendered
    # with typography and skin tone in the gap between them.
    #
    # Size separates the cases. Measured across the ten-image test set, every genuine dropout
    # on a single subject was under 0.6% of the silhouette, while the pocket between two
    # people was 2.8% -- a clean gap with nothing in it:
    #
    #     06-sidelight 0.059%   10-smile 0.242%   07-dark-on-dark 0.533%   |   05-couple 2.781%
    #
    # PET_HOLE_MAX is the largest hole, as a fraction of the subject's own area, that will
    # still be filled. 0.012 sits in that gap with about a factor of two of margin on each
    # side. PET_HOLE_MAX=0 restores the old behavior of filling everything.
    #
    # Calibrated on ONE two-subject photograph and on no animals at all. Three subjects, or a
    # dog with a real gap between its legs, are not represented; treat it as well-founded for
    # couples and provisional elsewhere until the test set covers them.
    _hmax = float(os.environ.get("PET_HOLE_MAX", "0.012") or 0.012)
    _subj = float(b.sum()) or 1.0
    if os.environ.get("PET_HOLE_DEBUG", "").strip() or _hmax > 0.0:
        _nh, _lh, _sh, _ = cv2.connectedComponentsWithStats(holes, 8)
        _keep = np.zeros_like(holes)
        _rep = []
        for _i in range(1, _nh):
            _a = float(_sh[_i, cv2.CC_STAT_AREA])
            _frac = _a / _subj
            _fill = (_hmax <= 0.0) or (_frac <= _hmax)
            if _fill:
                _keep[_lh == _i] = 1
            if _frac >= 0.0005:
                _rep.append((_frac, _fill))
        if os.environ.get("PET_HOLE_DEBUG", "").strip():
            print("[holes] %d found, %d over 0.05%% of subject: %s  (PET_HOLE_MAX=%s)"
                  % (_nh - 1, len(_rep),
                     ", ".join("%.3f%%%s" % (100 * f, "" if k else " REJECTED")
                               for f, k in sorted(_rep, reverse=True)[:8]) or "none",
                     _hmax or "off"))
        holes = _keep
    b = np.clip(b + holes, 0, 1)
    # Torso continuation. The hole fill above can only recover regions the border cannot
    # reach; a neck and chest run OFF the bottom edge, so they are correctly judged "not a
    # hole" -- and then lost anyway, because the model's confidence fades downward. Measured
    # on a head-and-shoulders portrait the mask ran 0.93 coverage at the face and 0.00 in the
    # bottom band, which is the 'floating head' this function exists to prevent.
    #
    # So: for any column whose silhouette already extends into the lower part of the frame,
    # carry it down to the bottom edge. A torso that reaches that far does not stop in mid-air.
    # Columns that end higher up are untouched, so a subject sitting fully inside the frame
    # (a pet with floor visible below) does not smear downward.
    #
    # PET_TORSO_FILL is the fraction of the height below which a column counts as "reaching":
    # 0 disables. Off by default -- turn it on per tree and look before adopting it.
    _tf = float(os.environ.get("PET_TORSO_FILL", "0") or 0.0)
    if _tf > 0.0 and b.any():
        H = b.shape[0]
        cut = int(H * min(0.95, max(0.05, _tf)))
        reaches = b[cut:].any(axis=0)                       # columns present below the cut
        if reaches.any():
            has = b.any(axis=0)
            last = H - 1 - np.argmax(b[::-1], axis=0)       # lowest subject row per column
            rows = np.arange(H)[:, None]
            b = np.maximum(b, ((rows >= last[None, :]) & has[None, :]
                               & reaches[None, :]).astype(np.uint8))
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


def _rows(stream, W, H, fs, rng, pad=0):
    """A grayscale INK-COVERAGE map (1 = ink, 0 = ground) of horizontal rows of the
    (importance-weighted) phrase stream at font size `fs`. Each row's horizontal start is
    jittered so a short list doesn't tile into wallpaper.

    `pad` extends the canvas BELOW the frame. The drape samples downward on bright regions,
    so without it the bottom edge has nothing real to read and the type stops short."""
    fs = max(6, int(round(fs)))
    pad = max(0, int(pad))
    font = ImageFont.truetype(_FONT, fs) if _FONT else ImageFont.load_default()
    im = Image.new("L", (W, H + pad), 255)
    d = ImageDraw.Draw(im)
    # Subtle letter tracking so glyphs in a row don't crowd -- a hair of space between words
    # opens the type up (reads less like a solid mass). PET_TRACK adds inter-phrase spacing.
    _trk = float(os.environ.get("PET_TRACK", "1.0") or 1.0)
    sep = "," + (" " * max(1, int(round(2 * _trk))))
    base = sep.join(stream) + "," + (" " * max(2, int(round(2 * _trk))))
    bw = max(1.0, float(d.textlength(base, font=font)))
    line = base * max(2, int((W + fs * 7) / bw) + 2)
    # Row gap: leave a sliver of ground between rows so lines breathe instead of colliding
    # into a busy wall of text. PET_ROW_GAP multiplies the line step (1.0 = touching).
    _gap = float(os.environ.get("PET_ROW_GAP", "1.12") or 1.12)
    step = max(6, int(round(fs * _gap)))
    y = 0
    while y < H + pad + fs:
        d.text((-rng.randint(0, int(fs * 6)), y), line, font=font, fill=0)
        y += step
    return 1.0 - (np.asarray(im).astype(np.float32) / 255.0)


def _enhance_contrast(bgr, mask):
    """Stretch the subject's tones to the full range so black fur reads black and white fur
    white. The flat gray wash came from compressed midtones -- this is the punch that makes a
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


def _render_word_portrait(bgr, mask, words, ground="dark", type_scale=None):
    """Sculpted landmark-free word-portrait: word rows are WARPED by the photo's luminance so
    the type drapes over the subject's form, blended across detail tiers (fine on features,
    coarse on the body), then colored by the photo. Ported from the human displacement engine
    but driven by the U2-Net silhouette + saliency -- this module never touches that engine."""
    H, W = bgr.shape[:2]
    gbgr = GROUNDS.get(ground, GROUNDS["dark"])
    fade = ground in _FADE_GROUNDS
    # --- Preprocess the PHOTO first, so the portrait is built from WORDS, not a photo with type
    #     laid over it: the coat should read as type; only the eyes/nose stay photographic. ---
    bgr = _enhance_contrast(bgr, mask)                       # punch up the tonal range first
    rng = random.Random(20260813)                           # fixed seed -> deterministic
    stream = _weighted_stream(words)                        # name + lead words weighted -> findable
    sc = W / 900.0                                          # size reference

    # (The silhouette "halo/glow" -- a ring of the original background the matte let bleed in --
    #  is removed at the very END by fading the outer band into the ground; see EDGE FADE below.)

    gray0 = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    # FEATURE FIELD (eyes/nose), color-agnostic: markedly DARKER (pupil, wet nose, dark-eyed dog)
    # OR BRIGHTER (light iris, catchlight) than the broad neighborhood. Computed UP FRONT so it can
    # PROTECT the eyes from the de-whisker and CONFINE the photographic blend to the features. A
    # uniform coat sits ~= its neighborhood and scores ~0, so it is never over-processed.
    _fp = float(os.environ.get("PET_FEATURE_PROTECT", "0.7") or 0.7)
    feat = np.zeros_like(gray0)                                            # 0..1 feature field (eyes/nose)
    broad = cv2.GaussianBlur(gray0, (0, 0), sigmaX=max(1.0, W * float(os.environ.get("PET_FEATURE_SCOPE","0.06") or 0.06)))  # neighborhood luminance
    if _fp > 0.0:
        localdark = np.clip((broad - gray0) / 55.0, 0, 1) * mask
        locallight = np.clip((gray0 - broad) / 70.0, 0, 1) * mask          # bright side less sensitive (÷70) -> fur/stripes don't register
        raw = np.maximum(localdark, locallight)
        # A feature is a COMPACT region (eye, nose); a whisker or ear-line is a THIN stroke that
        # also stands out from its neighborhood. Morphologically OPEN with a kernel wider than a
        # whisker: compact features survive, thin strokes are erased. This keeps whiskers OUT of the
        # feature field, so they are neither photo-painted nor shielded from the de-whisker below.
        ok = int(max(3, round(W * 0.011))) | 1
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)))
        feat = np.clip(cv2.GaussianBlur(raw, (0, 0), sigmaX=max(1.0, W * 0.012)) * 1.6, 0, 1)
        # The neighborhood comparison only fires at a feature's RIM: in the middle of a
        # large dark region (a nose in a tight crop) the neighborhood is equally dark, so
        # feat reads 0 and the interior falls to the COARSE tier -- big words on the nose.
        # Seal the rim with a small close, then flood-fill from the border: anything the
        # fill cannot reach is interior. Same technique as _solidify_matte. 0 = off.
        _ff = float(os.environ.get("PET_FEATURE_FILL", "0") or 0.0)
        if _ff > 0.0:
            _k = int(max(3, round(W * _ff))) | 1
            _b = (feat > 0.30).astype(np.uint8)
            _b = cv2.morphologyEx(_b, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_k, _k)))
            _p = cv2.copyMakeBorder(_b, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            cv2.floodFill(_p, np.zeros((_p.shape[0] + 2, _p.shape[1] + 2), np.uint8), (0, 0), 1)
            _b = np.clip(_b + (_p[1:-1, 1:-1] == 0).astype(np.uint8), 0, 1)
            feat = np.maximum(feat, cv2.GaussianBlur(_b.astype(np.float32), (0, 0),
                                                     sigmaX=max(1.0, W * 0.006)))

    # DE-WHISKER: thin bright strokes (whiskers, the diagonal lines inside ears) read as real fur,
    # not type, and overpower the words. A white top-hat isolates bright structures THINNER than the
    # kernel; replace those with a median (line-free) version so they dissolve into the coat. The
    # feature field is subtracted so eye catchlights (also small + bright) survive. PET_DEWHISKER scales it.
    _dw = float(os.environ.get("PET_DEWHISKER", "0.85") or 0.0)
    if _dw > 0.0:
        kk = int(max(3, round(W * 0.006))) | 1
        opened = cv2.morphologyEx(gray0.astype(np.uint8), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        tophat = np.clip((gray0 - opened.astype(np.float32)) / 32.0, 0, 1)  # thin bright structures
        thin = np.clip(tophat * mask * (1.0 - feat) * _dw, 0, 1)[..., None]
        med = cv2.medianBlur(np.clip(bgr, 0, 255).astype(np.uint8), kk).astype(np.float32)
        bgr = bgr * (1.0 - thin) + med * thin

    # LOCAL CONTRAST: lift local separation (CLAHE on L) so the darker muzzle / lower face doesn't go
    # muddy -- letters keep structure against the ground instead of smearing. PET_LOCAL_CONTRAST blends.
    _lc = float(os.environ.get("PET_LOCAL_CONTRAST", "0.4") or 0.0)
    if _lc > 0.0:
        lab = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        Lc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        lab[..., 0] = np.clip(lab[..., 0].astype(np.float32) * (1.0 - _lc) + Lc.astype(np.float32) * _lc, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    # Luminance + detail from the PROCESSED photo (de-whiskered, contrast-lifted).
    gray = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    det = _detail_map(gray.astype(np.uint8))                # 0..1 saliency: high on features/edges

    # 1) Four ink-coverage tiers, coarse -> micro. type_scale (<1 = finer type) sets the
    #    typography size the buyer picked (Small/Medium/Large -> ~0.30/0.42/0.56); when the
    #    caller doesn't pass one, fall back to the PET_TYPE_SCALE env default.
    _tsc = float(type_scale) if type_scale is not None else float(os.environ.get("PET_TYPE_SCALE", "0.42") or 0.42)
    # Coarse tier size. 64 is the historic value; lowering it shrinks only the LARGEST
    # words (the ones flat areas get) without touching the fine end, so gradation is kept.
    _tc = float(os.environ.get("PET_TIER_COARSE", "64") or 64.0)
    # PET_WORD_TIERS: importance becomes SIZE, not just repetition. _weighted_stream already
    # repeats the lead phrases more often, but every tier drew from the same stream -- so which
    # words came out large was an accident of where a row happened to land. Feeding the coarse
    # tier only the leading phrases (the name, the quirk, the bond -- whatever the form put
    # first) makes the largest words in the portrait always the meaningful ones, and leaves the
    # adjectives as texture. Off (default) = every tier draws the full stream, as before.
    _hero, _mid = stream, stream
    if os.environ.get("PET_WORD_TIERS", "").strip().lower() in ("1", "true", "on", "yes"):
        _ph = _phrases(words)
        _n = len(_ph)
        if _n >= 3:
            # Hero = the LEADING phrases that are also SHORT. Importance alone is not enough:
            # a nine-word memory at coarse size becomes a banner across the subject, while the
            # same words read beautifully one tier down. Names and quirks are naturally brief,
            # so this keeps them largest without asking the customer to write telegraphically.
            _lead = _ph[:max(2, int(round(_n * 0.55)))]
            _brief = [q for q in _lead if len(q.split()) <= 3]
            _hero = _brief[:max(1, int(round(_n * 0.30)))] or _lead[:1]
            _mid = _lead
    # The text canvas is drawn TALLER than the frame, because the drape below samples
    # DOWNWARD on bright regions -- my = yy + amp*dn -- and at the bottom edge that reaches
    # past the canvas. A zero border returned no glyphs there; replicating returned row H-1
    # smeared down, which is a gap if that row happens to fall between two lines. Measured on
    # a pale chest, glyph coverage held at 0.25 to 94% of the frame and then collapsed to
    # 0.098 -- a stripe that no row-gap or border setting could close, because there was
    # nothing real to sample.
    #
    # Padding by the drape's maximum reach gives it real, varied rows to find.
    _drape_max = float(os.environ.get("PET_DRAPE", "68") or 68.0) * sc
    _pad = int(round(_drape_max)) + int(round(_tc * sc * _tsc)) + 8
    tL = _rows(_hero, W, H, _tc * sc * _tsc, rng, pad=_pad)
    tM = _rows(_mid, W, H, 40 * sc * _tsc, rng, pad=_pad)
    tF = _rows(stream, W, H, 26 * sc * _tsc, rng, pad=_pad)
    tMi = _rows(stream, W, H, 16 * sc * _tsc, rng, pad=_pad)

    # 2) Drape: warp the rows VERTICALLY by smoothed luminance so they ride the form. Damp the
    #    warp on high-detail features (eyes/nose) so they stay crisp. PET_DRAPE tunes wrap depth;
    #    PET_DRAPE_SMOOTH smooths the field so type rides the broad form, not sharp local edges.
    _dsm = float(os.environ.get("PET_DRAPE_SMOOTH", "0.045") or 0.045)
    D = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, W * _dsm))
    dn = np.tanh((D / 255.0 - 0.5) * 2.4) * 0.85            # soft-limit: no over-stretch on the darkest/brightest fur (kills the 'melt')
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # Spread the detail edges inward, then UNION the feature field, so eye/nose interiors inherit
    # their rim's drape-protection and stay crisp instead of warping into a corrupted blob.
    featdamp = np.clip(cv2.GaussianBlur(det, (0, 0), sigmaX=max(1.0, W * 0.010)) * float(os.environ.get("PET_DRAPE_DETAIL_DAMP","1.9") or 1.9), 0, 1)
    featdamp = np.maximum(featdamp, feat * _fp)
    amp = float(os.environ.get("PET_DRAPE", "68") or 68.0) * sc * (1.0 - float(os.environ.get("PET_DRAPE_DAMP","0.92") or 0.92) * featdamp)
    my = (yy + amp * dn).astype(np.float32)
    mx = xx

    def R(t):
        # REPLICATE, not a zero border. my = yy + amp*dn, so a BRIGHT region drapes the
        # sample point downward -- at PET_DRAPE=110 that is ~110px. Near the bottom edge it
        # lands outside the canvas, and a zero border returns no glyphs: a blank band exactly
        # where the fur is brightest and the frame ends. Measured on a French bulldog, the
        # bottom tenth carried 70% of the glyph coverage the rest of the frame had, and the
        # ratio held whatever the row gap was set to.
        #
        # The text field is a tiling of identical rows, so continuing it past the edge is what
        # the drape was always reaching for.
        return cv2.remap(t, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    wL, wM, wF, wMi = R(tL), R(tM), R(tF), R(tMi)

    # 3) Blend tiers by the detail field: coarse text on flat body, fine on features. The feature
    #    field is unioned in so an eye/nose INTERIOR (smooth -> low raw detail -> would otherwise
    #    get the COARSE tier and show big words) is pushed to the FINE tier like its detailed rim.
    # PET_TIER_GAMMA: `det` is normalized by its own 99th percentile. Measured on a real coat it
    # spans ~0.25-0.90 (p5-p95), which already crosses all three tier bands -- so the
    # default 1.0 is usually correct. Gamma < 1 compresses toward the FINE tiers and
    # REDUCES size range; > 1 shifts area toward COARSE. 1.0 = previous behavior.
    _tg = float(os.environ.get("PET_TIER_GAMMA", "1.0") or 1.0)
    df = np.clip(np.maximum(det, feat), 0, 1)
    if _tg > 0.0 and _tg != 1.0:
        df = np.power(df, _tg)
    # PET_HERO_CENTRE: tier choice is driven by image detail, so the COARSE tier lands on
    # whatever is flattest -- often a smooth patch of ruff at the frame edge rather than the
    # face. Raise df toward the subject's periphery so the outer region takes finer tiers and
    # the largest words stay on the head, where a hero word reads as an anchor rather than a
    # stray banner. Radius is normalized to the mask's own bounding box, so it follows the
    # animal rather than the canvas. 0 (default) = unchanged.
    _hc = float(os.environ.get("PET_HERO_CENTRE", "0") or 0.0)
    if _hc > 0.0:
        _ys, _xs = np.nonzero(mask > 0.5)
        if _ys.size > 32:
            _cy, _cx = float(_ys.mean()), float(_xs.mean())
            _ry = max(1.0, (float(_ys.max()) - float(_ys.min())) * 0.5)
            _rx = max(1.0, (float(_xs.max()) - float(_xs.min())) * 0.5)
            _rr = np.sqrt(((xx - _cx) / _rx) ** 2 + ((yy - _cy) / _ry) ** 2)
            df = np.clip(df + _hc * np.clip(_rr - 0.45, 0.0, 1.0), 0.0, 1.0)
    warped = wL.copy()
    for a0, b0, ia, ib in ((0.0, 0.45, wL, wM), (0.45, 0.75, wM, wF), (0.75, 1.0001, wF, wMi)):
        bt = np.clip((df - a0) / (b0 - a0), 0, 1)
        warped = np.where((df >= a0) & (df < b0), ia * (1 - bt) + ib * bt, warped)
    warped = np.where(df >= 1.0, wMi, warped)               # ink density 0..1

    # 4) Colorise: the ink carries the photo's own color, composited on the ground.
    col = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
    dens = np.ones_like(gray)
    if fade:                                                # ink-density styling (dark-furred pets)
        dens = 0.35 + 0.65 * (1.0 - gray / 255.0)
        col = col * 0.72
    else:
        # SHADOW LIFT: a dark ground swallows dark-fur words -- so a shadowed neck/chest reads as
        # an empty void (a "floating head"). Floor the word color to a dim WARM ink so shadowed
        # fur still shows readable type against the dark ground. PET_SHADOW_LIFT scales it (0 off).
        _slift = float(os.environ.get("PET_SHADOW_LIFT", "1.0") or 1.0)
        if _slift > 0.0:
            col = np.maximum(col, np.array([50.0, 42.0, 34.0], np.float32) * _slift)
    # PET_NEGATIVE_SPACE: coverage is otherwise uniform across the whole subject -- every
    # region equally busy, so the eye has nowhere to rest and meaningful words become wallpaper.
    # Thin the type as the photo goes dark, so the deepest shadows dissolve toward the ground
    # and read as depth rather than as texture. Highlights keep full density, so the lit
    # structure stays built from words. 0 (default) = uniform, exactly as before.
    _ns = float(os.environ.get("PET_NEGATIVE_SPACE", "0") or 0.0)
    if _ns > 0.0:
        _lq = np.clip(gray / 255.0, 0.0, 1.0)
        _quiet = np.clip((0.45 - _lq) / 0.45, 0.0, 1.0)      # 1 at black -> 0 at mid-tone
        _quiet = cv2.GaussianBlur(_quiet, (0, 0), sigmaX=max(1.0, W * 0.012))
        dens = dens * (1.0 - min(max(_ns, 0.0), 0.95) * _quiet)
    a = (warped * dens * mask)[..., None]
    ground_rgb = np.full((H, W, 3), gbgr[::-1], np.float32)
    # SUBJECT BASE: the flat ground is painted across the WHOLE canvas, so INSIDE the
    # silhouette it shows through every gap between glyphs and every row gap -- the coat
    # reads as ground color instead of the animal's own. PET_SUBJECT_BASE swaps the base
    # inside the mask for the SOURCE PHOTO (dimmed by PET_SUBJECT_DIM so the words still
    # read on top of it), leaving the flat ground only BEHIND the subject. The glyphs
    # themselves are unchanged -- this only alters what sits behind them within the mask.
    # PET_SUBJECT_BASE=0 (default) is byte-identical to the original behavior.
    _sb = float(os.environ.get("PET_SUBJECT_BASE", "0") or 0.0)
    if _sb > 0.0:
        _dim = float(os.environ.get("PET_SUBJECT_DIM", "0.45") or 0.0)
        _pbase = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8),
                              cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 - _dim)
        _m3 = (mask * min(max(_sb, 0.0), 1.0))[..., None]
        ground_rgb = ground_rgb * (1.0 - _m3) + _pbase * _m3
    out = ground_rgb * (1.0 - a) + col * a

    # PET_DUMP_FIELDS=<dir>. The human engine grew this hook today and it ended several
    # rounds of guessing; the pet engine had none, so "the chest has no typography" could
    # only be argued about. Reports, band by band down the frame and INSIDE the mask only:
    #
    #   glyph   the text field (warped)  -- is any type laid here at all?
    #   dens    the density multiplier   -- is it being thinned away?
    #   alpha   glyph * dens * mask      -- what actually reaches the composite
    #   ink     the word COLOR's luma   -- near-white words on pale fur are invisible,
    #   base    what sits behind them       not absent, and only these two together say which
    _pd = os.environ.get("PET_DUMP_FIELDS", "").strip()
    if _pd:
        try:
            os.makedirs(_pd, exist_ok=True)
            _lum = lambda _x: (_x[..., 0] * 0.299 + _x[..., 1] * 0.587 + _x[..., 2] * 0.114)
            _al = np.asarray(a, np.float32)[..., 0]
            _ink_l, _base_l = _lum(np.asarray(col, np.float32)), _lum(np.asarray(ground_rgb, np.float32))
            for _nm, _arr in (("mask", mask), ("glyph", warped), ("dens", dens), ("alpha", _al)):
                _q = np.asarray(_arr, np.float32)
                cv2.imwrite(os.path.join(_pd, "pet-%s.png" % _nm),
                            np.clip(_q * (255.0 if float(_q.max()) <= 1.001 else 1.0),
                                    0, 255).astype(np.uint8))
            cv2.imwrite(os.path.join(_pd, "pet-out.png"),
                        cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            print("[pet] band  %8s %8s %8s %8s %8s" % ("glyph", "dens", "alpha", "ink", "base"))
            _mk = np.asarray(mask, np.float32) > 0.5
            for _i in range(10):
                _y0, _y1 = int(H * _i / 10), int(H * (_i + 1) / 10)
                _b = _mk[_y0:_y1]
                if not _b.any():
                    print("[pet] %3d%%  (no subject)" % (10 * _i + 5)); continue
                print("[pet] %3d%%  %8.3f %8.3f %8.3f %8.1f %8.1f"
                      % (10 * _i + 5,
                         float(np.asarray(warped, np.float32)[_y0:_y1][_b].mean()),
                         float(np.asarray(dens, np.float32)[_y0:_y1][_b].mean()),
                         float(_al[_y0:_y1][_b].mean()),
                         float(_ink_l[_y0:_y1][_b].mean()),
                         float(_base_l[_y0:_y1][_b].mean())))
        except Exception as _e:  # noqa: BLE001
            print("[pet] dump failed: %s" % _e)

    # 5) Feature edge-ink: darken along real internal edges so the face reads.
    edge = (_edge_ink(gray.astype(np.uint8)) * mask)[..., None]
    ink = np.array([28.0, 24.0, 20.0], np.float32)
    k = float(os.environ.get("PET_EDGE_INK", "0.62") or 0.62)
    out = out * (1.0 - k * edge) + ink * (k * edge)

    # 6) Photographic realism -- CONFINED to the features. The coat must read as WORDS, not a photo
    #    with type on top: general fur detail gets only a WHISPER of real photo (PET_PHOTO_FUR), while
    #    the eyes/nose keep the strong photographic blend that anchors the piece (feat field, scaled
    #    by PET_PHOTO). A uniform coat scores feat~0, so it is never over-photographed.
    _pf = float(os.environ.get("PET_PHOTO", "0.45") or 0.0)          # eyes/nose photographic anchor
    _pfur = float(os.environ.get("PET_PHOTO_FUR", "0.10") or 0.0)    # residual coat photo -- keep low
    if _pf > 0.0 or _pfur > 0.0:
        photo_rgb = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
        wgt = cv2.GaussianBlur(np.clip(det * 1.6, 0, 1), (0, 0), sigmaX=max(1.0, W * 0.008)) * mask * _pfur
        wgt = np.clip(np.maximum(wgt, feat * _pf * 1.4), 0, 1)
        wgt = wgt[..., None]
        out = out * (1.0 - wgt) + photo_rgb * wgt

    # 7) Tonal depth: deepen shadows + lift highlights within the subject so black fur reads
    #    deep (not flat gray) and lit areas glow -- the pet analogue of the engine's 'breathe'.
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

    # 8b) Eye/feature POP: make the eyes read alive and the nose glisten. Within the feature field
    #     (eyes of any iris color, wet nose), add crisp local contrast AND amplify the existing
    #     catchlight/shine (the bright specular point already in the photo). Landmark-free: it
    #     rides the same feature field, so a uniform coat (field ~0) is untouched.
    #     PET_EYE_POP scales it (0 disables).
    _pop = float(os.environ.get("PET_EYE_POP", "0.6") or 0.6)
    if _pop > 0.0 and _fp > 0.0 and float(feat.max()) > 0.05:
        g2 = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        hp = g2 - cv2.GaussianBlur(g2, (0, 0), sigmaX=max(1.0, W * 0.006))   # high-freq detail
        fpk = (feat * _pop)[..., None]
        out = np.clip(out + hp[..., None] * fpk * 1.3, 0, 255)               # snap the feature detail
        spec = np.clip((g2 - 175.0) / 60.0, 0, 1)[..., None]                 # existing catchlight / nose shine
        out = np.clip(out + spec * fpk * 95.0, 0, 255)                       # lift it -> a living glint

    # 9) Vibrance: boost less-saturated colors more so warm fur glows without going garish.
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

    # 11) EDGE FADE: dissolve the silhouette's OUTER band into the ground. The matte's background
    #     bleed (the pale "halo/glow") sits exactly at the mask perimeter, so fading from ground at
    #     the very edge up to the full render a short way in erases it -- and reads as a tasteful
    #     soft gallery edge, not a hard cutout. PET_EDGE_TIGHTEN sets the band width (0 disables).
    _et = float(os.environ.get("PET_EDGE_TIGHTEN", "0.18") or 0.0)
    if _et > 0.0:
        dist = cv2.distanceTransform((mask > 0.35).astype(np.uint8), cv2.DIST_L2, 3)  # px inward from the edge
        bandw = max(2.0, W * 0.11 * _et)                                              # band width scales with the knob
        ef = np.clip(dist / bandw, 0.0, 1.0)[..., None]
        ground_rgb = np.full((H, W, 3), gbgr[::-1], np.float32)
        out = ground_rgb * (1.0 - ef) + out * ef

    return np.clip(out, 0, 255).astype(np.uint8)


def _fit_print_aspect(bgr, mask, aspect):
    """Pad the (matted) subject to a target print aspect (width/height) so the render composes
    on a proper canvas -- e.g. 0.8 = 4:5. Margins get mask=0 so the render fills them with the
    ground, and the vignette then lights the whole print. Subject is centered (a touch high)."""
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
                        print_aspect: float | None = None, type_scale: float | None = None) -> bytes:
    """Decode a photo, render a landmark-free word-portrait, return PNG bytes. `height` sets the
    render resolution (preview ~900; print ~4500). `print_aspect` (e.g. 0.8 for 4:5) pads the
    subject onto a gallery canvas for print -- omit for the raw preview crop. `type_scale` sets
    the typography size (Small/Medium/Large); None uses the PET_TYPE_SCALE env default."""
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    # Cap the WORKING render resolution. The sculpt cost scales with pixel count: a full 4500px
    # print render takes ~70s -- past the request/proxy timeout, so /download hangs and the buyer
    # sees "preparing files" forever. Render at a capped height, then upscale the finished portrait
    # to the requested print height (the human engine uses the same render-then-upscale pattern).
    # PET_MAX_RENDER_PX tunes the cap; the preview (<=1600) is already below it and unaffected.
    cap = int(os.environ.get("PET_MAX_RENDER_PX", "2400") or 2400)
    work_h = min(height, cap) if height and height > 0 else height
    if bgr.shape[0] != work_h:
        bgr = cv2.resize(bgr, (max(1, int(bgr.shape[1] * work_h / bgr.shape[0])), work_h),
                         interpolation=cv2.INTER_AREA)
    mask = _foreground_mask(bgr)
    if print_aspect:
        bgr, mask = _fit_print_aspect(bgr, mask, float(print_aspect))
    out_rgb = _render_word_portrait(bgr, mask, words, ground=ground, type_scale=type_scale)
    if work_h < height:                                   # upscale the finished portrait to print size
        out_w = int(round(out_rgb.shape[1] * height / out_rgb.shape[0]))
        out_rgb = cv2.resize(out_rgb, (max(1, out_w), height), interpolation=cv2.INTER_LANCZOS4)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("encode failed")
    return buf.tobytes()
