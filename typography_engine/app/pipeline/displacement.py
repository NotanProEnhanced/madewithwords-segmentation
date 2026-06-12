"""Displacement typographic portrait renderer.

A premium human-portrait style: horizontal rows of the approved words are warped
by the photo's luminance so the text drapes over the facial form (the "type
follows the form" look), with a multi-tier feature-detail system (coarse rows on
the broad form, finer text on the features, finest in the eye rings) and explicit
eye/lip anchoring so features read regardless of lighting.

This renderer is raster-based (PIL text + OpenCV remap) and returns PNG bytes
directly, unlike the SVG-based tonal renderer. It reuses the shared analysis
(MediaPipe 478-point face mesh + silhouette) produced by ``analyze_image``.

Validated across diverse faces; see docs/displacement-style-findings.md.
"""
from __future__ import annotations

import glob
import random
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analyze import Analysis

# Ground (background + ink) options. BGR colours. ``tone`` selects whether the
# ink follows the photo's highlights ("light" -> light ink on a dark ground) or
# its shadows ("dark" -> dark ink on a light ground).
GROUNDS = {
    "paper": {"bg": (240, 247, 250), "ink": (58, 33, 20), "tone": "dark"},   # near-black on cream
    "navy":  {"bg": (58, 27, 13),    "ink": (248, 248, 248), "tone": "light"},  # white on navy (hero)
    "black": {"bg": (14, 14, 14),    "ink": (248, 248, 248), "tone": "light"},  # white on black
}

# Sculpted ink colours: the WORD colour (BGR) draped on the dark ground. These are
# light/bright tints (mirroring the studio's ink swatches) so they read on navy.
# "photo" is handled separately (per-pixel from the source). Keeps Sculpted's
# light-on-dark aesthetic while giving it the same palette as Words/Passage.
_SCULPT_INK = {
    "mono":      (248, 248, 248),   # Noir   — near-white
    "navy":      (241, 228, 219),   # Navy   — pale blue
    "sepia":     (182, 217, 234),   # Sepia  — warm cream
    "burgundy":  (211, 210, 236),   # Rose   — soft rose
    "forest":    (216, 227, 212),   # Sage   — pale green
    "gold_noir": (106, 198, 232),   # Ember  — gold
}

# MediaPipe FaceMesh landmark groups (subset rings) used for feature detail +
# anchoring. Indices <= 467 are stable across the 468/478-point variants.
_GROUPS = {
    "Leye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
    "Reye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 49, 279, 220, 440],
    "Lbrow": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
    "Rbrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
}


@lru_cache(maxsize=1)
def _font_path() -> Optional[str]:
    pats = [
        "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/**/*Bold*.ttf",
        "/usr/share/fonts/**/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in pats:
        m = sorted(glob.glob(p, recursive=True))
        if m:
            return m[0]
    return None


def _font(sz: int) -> ImageFont.FreeTypeFont:
    sz = max(6, int(sz))
    fp = _font_path()
    if fp:
        try:
            return ImageFont.truetype(fp, sz)
        except Exception:  # noqa: BLE001
            pass
    return ImageFont.load_default()


def _normalize_words(words: Sequence[str], uppercase: bool = True) -> List[str]:
    out: List[str] = []
    for w in words:
        s = str(w).upper() if uppercase else str(w)
        t = "".join(ch for ch in s if ch.isalnum() or ch in "-'")
        if t:
            out.append(t)
    return out or (["LOVE"] if uppercase else ["love"])


def render_displacement_portrait(
    an: Analysis,
    words: Sequence[str],
    ground: str = "navy",
    out_width: int = 1400,
    supersample: int = 2,
    seed: int = 7,
    uppercase: bool = True,
    ink: Optional[str] = None,
) -> bytes:
    """Render a displacement typographic portrait to PNG bytes.

    Raises ValueError("displacement_needs_face") if no face mesh is available
    (this style is driven by the 478-point landmarks).
    """
    pts0 = an.landmarks.points if an.landmarks is not None else None
    if pts0 is None:
        raise ValueError("displacement_needs_face")

    g = GROUNDS.get(ground, GROUNDS["navy"])
    rng = random.Random(seed)
    vocab = _normalize_words(words, uppercase)

    g0 = an.img.gray.astype(np.float32)
    m0 = (an.silhouette.mask > 127).astype(np.float32)
    h0, w0 = g0.shape
    SS = max(1, int(supersample))
    W, H = w0 * SS, h0 * SS
    gray = cv2.resize(g0, (W, H), interpolation=cv2.INTER_CUBIC)
    mask01 = cv2.resize(m0, (W, H), interpolation=cv2.INTER_LINEAR)
    pts = pts0 * SS
    fbb = an.face_bbox
    fw = (fbb[2] * SS) if fbb else W * 0.55
    face_frac = (fbb[2] / w0) if fbb else 0.55
    s = float(np.clip(face_frac / 0.47, 0.5, 1.3))   # subject-relative scale (hero anchor = 0.47)

    # --- Living eyes: true iris geometry from MediaPipe's iris landmarks ------
    # (centre + 4-point ring per eye, 478-point mesh only). Drives a round pupil,
    # an iris-scaled text tier, a real catchlight, and -- separately gated -- the
    # person's true eye colour. Both irises must resolve large enough to carry
    # structure; otherwise every step below falls back to the legacy behavior.
    irises: List[Tuple[float, float, float]] = []
    if len(pts) >= 478:
        for ic, ring in ((468, (469, 470, 471, 472)), (473, (474, 475, 476, 477))):
            icx, icy = float(pts[ic][0]), float(pts[ic][1])
            ir = float(np.mean([np.hypot(pts[i][0] - icx, pts[i][1] - icy) for i in ring]))
            if ir >= 8.0:
                irises.append((icx, icy, ir))
    if len(irises) < 2:
        irises = []

    def rows(fs: float) -> np.ndarray:
        f = _font(fs)
        im = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(im)
        y = 0
        while y < H + fs:
            # Keep the words in the order they were entered (a sentence stays a
            # sentence); only the row's horizontal start is jittered for variety.
            line = (" ".join(vocab) + " ") * (W // max(1, int(fs * 3)) + 18)
            d.text((-rng.randint(0, int(fs * 6)), y), line, font=f, fill=0)
            y += max(6, int(fs))
        return 1.0 - (np.asarray(im).astype(np.float32) / 255.0)

    # Four size tiers blended *continuously* (below) so the type eases from large
    # to small instead of snapping between discrete sizes.
    t_large, t_mid, t_fine, t_micro = rows(64 * s), rows(40 * s), rows(26 * s), rows(16 * s)
    # Fifth tier, scaled to the EYE rather than the face: even "micro" type spans
    # a whole iris on a close-up, so the iris gets rows proportional to its own
    # radius -- typography that fits inside the eye.
    t_iris = rows(max(7.0, float(np.mean([r for _, _, r in irises])) * 0.45)) if irises else None

    def mask_of(keys, dil, sig) -> np.ndarray:
        mm = np.zeros((H, W), np.uint8)
        for k in keys:
            p = np.array([pts[i] for i in _GROUPS[k] if i < len(pts)], np.int32)
            if len(p) >= 3:
                cv2.fillConvexPoly(mm, cv2.convexHull(p), 1)
        if dil > 0:
            mm = cv2.dilate(mm, np.ones((dil | 1, dil | 1), np.uint8), 1)
        return np.clip(cv2.GaussianBlur(mm.astype(np.float32), (0, 0), sigmaX=max(1.0, sig)), 0, 1)

    feat_damp = mask_of(_GROUPS.keys(), int(fw * 0.06), fw * 0.045)

    fmh = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(fmh, cv2.convexHull(pts.astype(np.int32)), 1)
    face_w = cv2.GaussianBlur(fmh.astype(np.float32), (0, 0), sigmaX=W * 0.02)

    # Smooth "detail field" df in [0,1] that drives a CONTINUOUS size gradient:
    # ~0 on the body (large text) -> ~0.45 on the broad face (mid) -> ~1 at the
    # features (small). Heavily feathered so the size transition is gradual.
    face_norm = np.clip(face_w / (face_w.max() + 1e-6), 0, 1)
    feat_union = mask_of(_GROUPS.keys(), int(fw * 0.04), fw * 0.10)
    feat_norm = np.clip(feat_union / (feat_union.max() + 1e-6), 0, 1)
    df = np.clip(0.45 * face_norm + 0.70 * feat_norm, 0, 1)

    # Clean vertical drape, dampened in the feature band (keeps features crisp).
    D = cv2.GaussianBlur(gray, (0, 0), sigmaX=W * 0.020)
    dn = (D / 255.0 - 0.5) * 2.0
    xx, yy = np.meshgrid(np.arange(W).astype(np.float32), np.arange(H).astype(np.float32))
    amp = 64.0 * s * (1.0 - 0.85 * feat_damp)
    my = (yy + amp * dn).astype(np.float32)
    mx = xx.astype(np.float32)

    def R(t):
        return cv2.remap(t, mx, my, cv2.INTER_LINEAR, borderValue=0.0)

    # Continuous bracket-blend across the 4 tiers -> smooth large -> mid -> small.
    wL, wM, wF, wMi = R(t_large), R(t_mid), R(t_fine), R(t_micro)
    warped = wL.copy()
    for a, b, ia, ib in ((0.0, 0.45, wL, wM), (0.45, 0.75, wM, wF), (0.75, 1.0001, wF, wMi)):
        bt = np.clip((df - a) / (b - a), 0, 1)
        warped = np.where((df >= a) & (df < b), ia * (1 - bt) + ib * bt, warped)
    warped = np.where(df >= 1.0, wMi, warped)

    # Iris circles take the eye-scaled tier (feathered edge); iris_m is reused
    # below for the colour blend.
    iris_m = None
    if irises and t_iris is not None:
        iris_m = np.zeros((H, W), np.float32)
        ir_mean = float(np.mean([r for _, _, r in irises]))
        for icx, icy, ir in irises:
            cv2.circle(iris_m, (int(round(icx)), int(round(icy))), int(round(ir)), 1.0, -1, cv2.LINE_AA)
        iris_m = np.clip(cv2.GaussianBlur(iris_m, (0, 0), sigmaX=max(1.0, ir_mean * 0.18)), 0, 1)
        warped = warped * (1.0 - iris_m) + R(t_iris) * iris_m

    # Tonal field: percentile-stretch within the subject.
    vals = gray[mask01 > 0]
    if vals.size == 0:
        vals = gray.reshape(-1)
    lo, hi = np.percentile(vals, [4, 96])
    lum = np.clip((gray - lo) / (hi - lo + 1e-6), 0, 1)
    ink_field = lum if g["tone"] == "light" else (1.0 - lum)

    # Local-contrast boost so flat-lit features separate.
    hp = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=fw * 0.06)
    hp /= (np.std(hp[mask01 > 0]) + 1e-6)
    sign = 1.0 if g["tone"] == "light" else -1.0
    ink_field = np.clip(ink_field + 0.40 * sign * np.clip(hp, -2, 2) * face_w, 0, 1)

    # Progressive density: thicken text where ink is strongest.
    b1 = cv2.dilate(warped, np.ones((2, 2), np.uint8), 1)
    b2 = cv2.dilate(warped, np.ones((3, 3), np.uint8), 1)
    gd1 = np.clip((ink_field - 0.40) / 0.60, 0, 1)
    gd2 = np.clip((ink_field - 0.70) / 0.30, 0, 1)
    w2 = np.clip(warped + (b1 - warped) * gd1 + (b2 - b1) * gd2, 0, 1)

    a = np.clip(w2 * (0.04 + 0.96 * np.power(ink_field, 0.62)), 0, 1)
    a = a * np.clip(cv2.GaussianBlur(mask01, (0, 0), sigmaX=W * 0.007), 0, 1)   # feathered edge

    # Feature anchoring: eye rings + lip seam + pupils + nostrils.
    anchor = np.zeros((H, W), np.float32)
    th = max(1, int(fw * 0.006))
    for k in ["Leye", "Reye", "lips"]:
        p = np.array([pts[i] for i in _GROUPS[k] if i < len(pts)], np.int32)
        if len(p) >= 3:
            cv2.polylines(anchor, [cv2.convexHull(p)], True, 1.0, th, cv2.LINE_AA)
    if not irises:
        # Legacy eye presence: an ink blob at the lid centroid. Only used when the
        # iris landmarks can't resolve -- with real irises the round pupil and
        # catchlight below model the eye properly instead.
        for k in ["Leye", "Reye"]:
            c = np.mean([pts[i] for i in _GROUPS[k]], 0).astype(int)
            cv2.circle(anchor, tuple(c), max(2, int(fw * 0.020)), 1.0, -1, cv2.LINE_AA)
    for i in (98, 327, 2):
        if i < len(pts):
            cv2.circle(anchor, (int(pts[i][0]), int(pts[i][1])), max(1, int(fw * 0.012)), 1.0, -1, cv2.LINE_AA)
    anchor = cv2.GaussianBlur(anchor, (0, 0), sigmaX=max(1.0, fw * 0.004))
    anchor = np.clip(anchor, 0, 1)
    if g["tone"] == "light":
        a = a * (1.0 - 0.65 * anchor)          # dark feature lines = less light ink (ground shows)
    else:
        a = np.clip(a + 0.70 * anchor, 0, 1)    # dark feature lines = more dark ink on paper

    # Round pupil + catchlight from the true iris geometry. The pupil is a
    # feathered DISC at the iris centre (not the blocky gap the text rows happen
    # to leave), and the catchlight sits at the eye's real brightest pixel inside
    # the iris -- the glint that makes the portrait look back at you.
    if irises:
        pup = np.zeros((H, W), np.float32)
        glint = np.zeros((H, W), np.float32)
        for icx, icy, ir in irises:
            cv2.circle(pup, (int(round(icx)), int(round(icy))),
                       max(2, int(round(ir * 0.42))), 1.0, -1, cv2.LINE_AA)
            gx0, gx1 = int(max(0, icx - ir)), int(min(W, icx + ir + 1))
            gy0, gy1 = int(max(0, icy - ir)), int(min(H, icy + ir + 1))
            win = gray[gy0:gy1, gx0:gx1]
            if win.size:
                wyy, wxx = np.ogrid[gy0:gy1, gx0:gx1]
                inside = ((wxx - icx) ** 2 + (wyy - icy) ** 2) <= ir * ir
                bright = np.where(inside, win, -1.0)
                by, bx = np.unravel_index(int(np.argmax(bright)), bright.shape)
                cv2.circle(glint, (gx0 + bx, gy0 + by),
                           max(1, int(round(ir * 0.18))), 1.0, -1, cv2.LINE_AA)
        ir_mean = float(np.mean([r for _, _, r in irises]))
        pup = np.clip(cv2.GaussianBlur(pup, (0, 0), sigmaX=max(1.0, ir_mean * 0.10)), 0, 1)
        glint = np.clip(cv2.GaussianBlur(glint, (0, 0), sigmaX=max(1.0, ir_mean * 0.10)), 0, 1)
        if g["tone"] == "light":                  # light ink on a dark ground
            a = a * (1.0 - 0.88 * pup)            # pupil: round, dark (ground shows)
            a = np.clip(a + 0.85 * glint, 0, 1)   # catchlight: bright glint
        else:                                     # dark ink on light paper
            a = np.clip(a + 0.80 * pup, 0, 1)     # pupil: round dark ink
            a = a * (1.0 - 0.85 * glint)          # catchlight: paper shows

    al = a[..., None]
    if ink == "photo":
        # Words take the photo's OWN colours, draped over the form, on the ground.
        bgr_full = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        hsv = cv2.cvtColor(np.clip(bgr_full, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)          # lift saturation
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.18 + 18, 0, 255)      # lift value vs dark ground
        ink_col = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        out = np.array(g["bg"], np.float32) * (1 - al) + ink_col * al
    elif ink in _SCULPT_INK:
        word = np.array(_SCULPT_INK[ink], np.float32)
        out = np.array(g["bg"], np.float32) * (1 - al) + word * al
    else:
        out = np.array(g["bg"], np.float32) * (1 - al) + np.array(g["ink"], np.float32) * al

    # Living eyes, colour: glyphs inside the iris carry the person's TRUE eye
    # colour -- sampled by the shared gated helper (both irises saturated and
    # hue-consistent, else no tint; sampled, never invented). Dark grounds only:
    # the lifted tint is designed for light-ink-on-dark.
    if irises and iris_m is not None and g["tone"] == "light":
        from .tonal import _iris_tint
        tint = _iris_tint(an)
        if tint is not None:
            tip = np.array(tint[1][::-1], np.float32)        # lifted RGB -> BGR
            iout = np.array(g["bg"], np.float32) * (1 - al) + tip * al
            im3 = iris_m[..., None]
            out = out * (1.0 - im3) + iout * im3
    oh = max(1, int(out_width * h0 / w0))
    out = cv2.resize(out, (int(out_width), oh), interpolation=cv2.INTER_AREA)
    from .preprocess import apply_vibrance
    out = apply_vibrance(out, bgr=True)       # give the render life (clarity/glow/saturation)
    ok, buf = cv2.imencode(".png", np.clip(out, 0, 255).astype(np.uint8))
    if not ok:
        raise ValueError("encode_failed")
    return buf.tobytes()
