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
    "paper": {"bg": (232, 240, 244), "ink": (58, 33, 20), "tone": "dark"},   # ink-drawing on warm ivory (BGR)
    "navy":  {"bg": (58, 27, 13),    "ink": (248, 248, 248), "tone": "light"},  # white on navy (hero)
    "black": {"bg": (14, 14, 14),    "ink": (248, 248, 248), "tone": "light"},  # white on black
}

# Paper = an INK-DRAWING on warm ivory. Colouring words by the photo's brightness
# fails on a light ground (light hair/skin are highlights -> they vanish), so here
# tone comes from ink DENSITY instead: dark photo areas get heavy dark ink; light
# areas fade to paper; and an EDGE pass adds ink along contours (hair strands,
# silhouette, features) so light hair is DRAWN by its structure, not erased. The
# ink itself is always dark (a warm hue) so wherever it lands it reads on the ivory.
_PAPER_INK_VALUE = 102      # HSV V cap of the ink -- dark enough to read on ivory
_PAPER_INK_SAT = 1.95       # COLOUR lives in the GLYPHS: push hue so skin/lips/eyes read
_PAPER_DARK_GAMMA = 0.80    # <1 lifts faint mid-darks so the face isn't too empty
_PAPER_DARK_GAIN = 1.12     # overall ink weight from darkness
_PAPER_EDGE_GAIN = 0.85     # extra ink along contours (this is what draws light hair)
_PAPER_INK_FLOOR = 0.30     # minimum ink on the subject so the face stays densely typographic
_PAPER_IRIS_SAT = 1.36

# Eye-aspect-ratio (lid aperture / eye width) below which an eye is treated as
# CLOSED. MediaPipe still places an iris on a shut eye, so without this gate the
# "living eyes" treatment fabricates open eyes on a closed-eye photo. Open eyes
# run ~0.25-0.35; a relaxed/closed eye ~<0.12. 0.15 leaves headroom for squints.
_EYE_OPEN_EAR = 0.15
# Appearance backstop for what geometry/blendshapes MISS (e.g. closed eyes behind a
# reflective lens, where MediaPipe still reports "open"). A real open eye is a DARK
# iris/pupil in a brighter sclera; ratio = iris-centre luminance / eye-region bright
# pixels. Real eyes measured 0.08-0.30; a closed/glare eye (no dark pupil) ~0.45+.
# Above this there's no real eye to model -> suppress instead of fabricating one.
_EYE_OPEN_IRIS_MAX = 0.40

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


def _sclera_value(gray, pts, scl, floor=0.60):
    """Per-eye contrast-stretched sclera shading. Real sclera is not a flat disc:
    the upper lid shadows its top, it falls off toward the inner/outer corners, and
    it curves away at the edges. Stretching each eye's OWN luminance restores that
    natural gradient, while normalising PER EYE keeps even a shaded eye bright (so
    the dark-merge fix holds without the artificial, uniform-value look)."""
    H, W = gray.shape
    val = np.full((H, W), floor, np.float32)
    gb = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    yy = np.arange(H, dtype=np.float32)[:, None]
    for k in ("Leye", "Reye"):
        p = np.array([pts[i] for i in _GROUPS[k] if i < len(pts)], np.int32)
        if len(p) < 3:
            continue
        em = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(em, cv2.convexHull(p), 1)
        m = (em > 0) & (scl > 0.05)
        if int(m.sum()) < 12:
            continue
        # 1) Per-eye luminance stretch -> the photo's own sclera gradient, bright.
        gp = gb[m]
        lo, hi = np.percentile(gp, [12, 94])
        if hi - lo < 6.0:
            hi = lo + 6.0
        v = np.clip((gb - lo) / (hi - lo), 0.0, 1.0)
        base = floor + (1.0 - floor) * v
        # 2) Upper-lid shadow: the dominant natural cue -- the top of the sclera (just
        # under the lid/lashes) is markedly darker, easing to full toward the exposed
        # lower sclera. Synthesised from the eye's own vertical extent so it's reliable
        # even when the photo's tiny sclera is too flat to carry it.
        y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
        vy = np.clip((yy - y0) / max(8.0, (y1 - y0)), 0.0, 1.0)        # 0 top -> 1 bottom
        lid = 0.52 + 0.48 * np.clip((vy - 0.08) / 0.55, 0.0, 1.0)      # darker top, full lower
        val[m] = np.clip(base * np.broadcast_to(lid, (H, W)), 0.0, 1.0)[m]
    return val


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
    print_aspect: float = 0.8,    # width/height of the print canvas (4:5 default)
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
    # A few absolute pixel sizes below (text tiers, drape amplitude) were tuned at
    # the default SS=2. Everything else (face width, blurs, warp coords) scales
    # with the canvas, so at a LOWER SS those absolutes come out coarse relative to
    # the face. Normalize them by SS so a light preview (SS=1) keeps the SAME
    # typography + drape as the full SS=2 paid file -- just at lower resolution.
    _ssn = SS / 2.0
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
    eye_centers = list(irises)   # remember detected eyes; reused for glare clean-up if gated off
    # Eye-OPENNESS gate. MediaPipe places an iris even on a CLOSED eye (glasses make
    # it worse), so the living-eye treatment would paint synthetic OPEN eyes onto a
    # peaceful, closed-eye photo -- unacceptable, especially for a memorial. Only
    # keep the synthetic eyes when BOTH eyes read as open via the eye-aspect-ratio
    # (lid aperture / eye width) from the eyelid landmarks; otherwise the eye region
    # renders as plain words (a naturally closed eye).
    if irises and len(pts) >= 478:
        def _ear(p1, p2, p3, p4, p5, p6):
            horiz = float(np.hypot(pts[p1][0] - pts[p4][0], pts[p1][1] - pts[p4][1]))
            if horiz < 1e-3:
                return 0.0
            v = (float(np.hypot(pts[p2][0] - pts[p6][0], pts[p2][1] - pts[p6][1]))
                 + float(np.hypot(pts[p3][0] - pts[p5][0], pts[p3][1] - pts[p5][1])))
            return v / (2.0 * horiz)
        ear_r = _ear(33, 160, 158, 133, 153, 144)     # subject's right eye
        ear_l = _ear(362, 385, 387, 263, 373, 380)    # subject's left eye
        if min(ear_r, ear_l) < _EYE_OPEN_EAR:
            irises = []                                # closed -> no fabricated eyes
    # Appearance backstop: a real open eye is a DARK iris/pupil in a brighter sclera.
    # If the iris-centre isn't clearly darker than the eye's bright pixels -- because
    # it's eyelid skin (closed) or a glasses GLARE filling the lens -- there's no eye
    # to model, so suppress rather than fabricate one. Catches the cases MediaPipe's
    # geometry/blendshapes get wrong (closed eyes it still reports as open).
    if irises:
        ratios = []
        for icx, icy, ir in irises:
            y0, y1 = max(0, int(icy - ir * 2.4)), int(icy + ir * 2.4)
            x0, x1 = max(0, int(icx - ir * 2.4)), int(icx + ir * 2.4)
            reg = gray[y0:y1, x0:x1]
            if reg.size < 16:
                continue
            inner = np.zeros((H, W), np.uint8)
            cv2.circle(inner, (int(round(icx)), int(round(icy))), max(1, int(ir * 0.5)), 1, -1)
            sclera = max(float(np.percentile(reg, 90)), 1.0)
            ratios.append(float(gray[inner > 0].mean()) / sclera)
        if len(ratios) >= 2 and min(ratios) > _EYE_OPEN_IRIS_MAX:
            irises = []                                # no dark iris -> not a real eye

    # Glare clean-up: when the eyes are SUPPRESSED (closed / glare) AND the photo has a
    # blown-out specular reflection over the eye (e.g. glasses glare), tone it down
    # toward the surrounding skin so it doesn't render as a bright, eye-like blob. Only
    # runs when there is no real eye to model, so real open eyes are never touched.
    _eye_deglare = None
    if not irises and eye_centers:
        emask = np.zeros((H, W), np.uint8)
        for icx, icy, ir in eye_centers:
            cv2.circle(emask, (int(round(icx)), int(round(icy))), int(ir * 4.0), 1, -1)
        # The reflection is bright AND/OR strongly COLOURED (a blue lens glare spans a
        # wide luminance, so brightness alone misses most of it). Catch both, then tone
        # toward neutral skin so the lens reads as a muted area, not a coloured,
        # eye-like blob. Only runs when eyes are suppressed -> real eyes untouched.
        _sat = cv2.cvtColor(cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
        spec = (((gray > 175.0) | (_sat > 55.0)) & (emask > 0)).astype(np.uint8)
        if int(spec.sum()) > 0:
            spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            skin_px = gray[(emask > 0) & (spec == 0)]
            skin = float(np.median(skin_px)) * 0.9 if skin_px.size else 120.0
            gm = cv2.GaussianBlur(spec.astype(np.float32), (0, 0), max(2.0, _ssn * 4.0))
            gray = gray * (1.0 - gm) + skin * gm        # density/displacement sees no glare
            _eye_deglare = (gm, float(skin))            # reused on the colour source below

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
    t_large, t_mid, t_fine, t_micro = (rows(64 * s * _ssn), rows(40 * s * _ssn),
                                       rows(26 * s * _ssn), rows(16 * s * _ssn))
    # Fifth tier, scaled to the EYE rather than the face: even "micro" type spans
    # a whole iris on a close-up, so the iris gets rows proportional to its own
    # radius -- typography that fits inside the eye.
    t_iris = rows(max(6.0, float(np.mean([r for _, _, r in irises])) * 0.30)) if irises else None

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
    # FACE-relative feathering (not image-relative): on a tight crop the face fills
    # the frame, so an image-relative blur transitions over too thin a band and the
    # type SNAPS large->small. Scaling the blur to the face width keeps the size
    # gradient gradual across the forehead/cheeks at any crop tightness.
    face_w = cv2.GaussianBlur(fmh.astype(np.float32), (0, 0), sigmaX=max(W * 0.012, fw * 0.22))

    # Smooth "detail field" df in [0,1] that drives a CONTINUOUS size gradient:
    # ~0 on the body (large text) -> ~0.45 on the broad face (mid) -> ~1 at the
    # features (small). Heavily feathered so the size transition is gradual.
    face_norm = np.clip(face_w / (face_w.max() + 1e-6), 0, 1)
    # Wide feature feathering -> feat_norm DECAYS smoothly outward from the eyes/
    # nose/mouth, so the type grows continuously (small at features -> mid -> large)
    # instead of snapping at a hard feature boundary.
    feat_union = mask_of(_GROUPS.keys(), int(fw * 0.04), fw * 0.24)
    feat_norm = np.clip(feat_union / (feat_union.max() + 1e-6), 0, 1)
    df = np.clip(0.45 * face_norm + 0.70 * feat_norm, 0, 1)
    df = cv2.GaussianBlur(df, (0, 0), sigmaX=max(2.0, fw * 0.06))   # ease the size steps further

    # Clean vertical drape, dampened in the feature band (keeps features crisp).
    D = cv2.GaussianBlur(gray, (0, 0), sigmaX=W * 0.020)
    dn = (D / 255.0 - 0.5) * 2.0
    xx, yy = np.meshgrid(np.arange(W).astype(np.float32), np.arange(H).astype(np.float32))
    amp = 64.0 * s * _ssn * (1.0 - 0.85 * feat_damp)
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
        # Catchlight: deterministic, consistent between both eyes (the classic
        # upper diagonal on the lit side) -- shared helper, working coords -> xSS.
        from .tonal import _catchlight_points
        for gx, gy, gr in _catchlight_points(an):
            cv2.circle(glint, (int(round(gx * SS)), int(round(gy * SS))),
                       max(1, int(round(gr * SS))), 1.0, -1, cv2.LINE_AA)
        ir_mean = float(np.mean([r for _, _, r in irises]))
        pup = np.clip(cv2.GaussianBlur(pup, (0, 0), sigmaX=max(1.0, ir_mean * 0.10)), 0, 1)
        glint = np.clip(cv2.GaussianBlur(glint, (0, 0), sigmaX=max(1.0, ir_mean * 0.10)), 0, 1)
        # No typography in the sclera: inside the eyelid hull but outside the
        # iris, ink is suppressed entirely -- the eye reads as anatomy (clean
        # sclera, typed iris, round pupil, glint), not as text.
        scl = np.zeros((H, W), np.float32)
        for k in ("Leye", "Reye"):
            p = np.array([pts[i] for i in _GROUPS[k] if i < len(pts)], np.int32)
            if len(p) >= 3:
                cv2.fillConvexPoly(scl, cv2.convexHull(p), 1.0)
        iris_full = np.zeros((H, W), np.float32)
        limbal = np.zeros((H, W), np.float32)      # dark rim at the iris edge
        for icx, icy, ir in irises:
            ci = (int(round(icx)), int(round(icy)))
            cv2.circle(iris_full, ci, int(round(ir)), 1.0, -1, cv2.LINE_AA)
            cv2.circle(limbal, ci, int(round(ir)), 1.0, -1, cv2.LINE_AA)
            cv2.circle(limbal, ci, int(round(ir * 0.80)), 0.0, -1, cv2.LINE_AA)
        limbal = cv2.GaussianBlur(limbal, (0, 0), sigmaX=max(1.0, ir_mean * 0.05))
        scl = np.clip(scl - iris_full, 0, 1)
        scl = cv2.GaussianBlur(scl, (0, 0), sigmaX=max(1.0, fw * 0.004))
        a = a * (1.0 - 0.92 * scl)
        if g["tone"] == "light":                  # light ink on a dark ground
            a = a * (1.0 - 0.88 * pup)            # pupil: round, dark (ground shows)
            a = np.clip(a + 0.85 * glint, 0, 1)   # catchlight: bright glint
        else:                                     # dark ink on light paper
            a = np.clip(a + 0.80 * pup, 0, 1)     # pupil: round dark ink
            a = a * (1.0 - 0.85 * glint)          # catchlight: paper shows

    # Teeth carry NO typography. Where the mouth is open, suppress ink across the
    # inner mouth (both tones); on a dark ground the cleared teeth get a soft
    # light wash below, on light paper the paper already reads as teeth. A closed
    # mouth yields no mask and is left untouched.
    from .tonal import _teeth_mask
    teeth = _teeth_mask(pts, H, W)
    if teeth is not None:
        # Appearance gate (parallels the eyes): the mesh mis-reads a resting/closed
        # mouth on a tilted or lying-down photo as "open". A REAL open mouth has a
        # dark cavity OR bright teeth; a falsely-detected one is uniform lip tone.
        # Keep teeth only with a dark cavity (p10 low) or genuinely bright teeth.
        tpx = gray[teeth > 0.5]
        if tpx.size > 10:
            p10 = float(np.percentile(tpx, 10))
            p90 = float(np.percentile(tpx, 90))
            if p10 > 60.0 and p90 < 205.0:
                teeth = None                           # no cavity, no teeth -> closed mouth
    if teeth is not None:
        a = a * (1.0 - 0.92 * teeth)
    if ground == "paper":
        # INK-DRAWING density: tone is how much ink lands, not its colour. Heavy ink
        # where the photo is dark; fade to paper where it's light; an edge boost draws
        # contours/hair strands so light hair isn't erased on the ivory.
        valn = gray / 255.0
        dark = np.clip(1.0 - valn, 0.0, 1.0) ** _PAPER_DARK_GAMMA
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.hypot(gx, gy)
        edge = np.clip(edge / (float(np.percentile(edge, 99.0)) + 1e-6), 0.0, 1.0)
        edge = cv2.GaussianBlur(edge, (0, 0), max(0.6, 0.8 * _ssn))
        ink_amt = np.clip(dark * _PAPER_DARK_GAIN + edge * _PAPER_EDGE_GAIN + _PAPER_INK_FLOOR, 0.0, 1.0)
        a = a * ink_amt
    al = a[..., None]
    if ink == "photo":
        # Words take the photo's OWN colours, draped over the form, on the ground.
        bgr_full = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        if _eye_deglare is not None:        # tone the suppressed-eye glare out of the colour too
            gm, skin = _eye_deglare
            bgr_full = bgr_full * (1.0 - gm[..., None]) + np.float32(skin) * gm[..., None]
        hsv = cv2.cvtColor(np.clip(bgr_full, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        if ground == "paper":
            # Ink-drawing with COLOURED glyphs: the words keep the photo's hue at high
            # saturation but a capped dark value, so each word reads as coloured TYPE on
            # ivory (skin/lips/eyes show) -- colour from the glyphs, not a photo overlay.
            # Tone is the ink DENSITY applied above; minimum() keeps deep shadows deep.
            hsv[..., 1] = np.clip(hsv[..., 1] * _PAPER_INK_SAT, 0, 255)
            hsv[..., 2] = np.minimum(hsv[..., 2], np.float32(_PAPER_INK_VALUE))
        else:
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
    if irises and iris_m is not None and g["tone"] == "light" and ink == "photo":
        from .tonal import _iris_tint
        tint = _iris_tint(an)
        if tint is not None:
            tip = np.array(tint[1][::-1], np.float32)        # lifted RGB -> BGR
            iout = np.array(g["bg"], np.float32) * (1 - al) + tip * al
            im3 = iris_m[..., None]
            out = out * (1.0 - im3) + iout * im3
    elif irises and iris_m is not None and ground == "paper" and ink == "photo":
        # Paper: keep the iris its TRUE source colour. The Keep-Paper-Light lift
        # mutes everything toward the paper, which would wash the eye colour out --
        # so re-lay the source's own iris pixels here, saturated but NOT lifted, so
        # the real hue (her green/hazel/blue) pops against the airy face.
        bf = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        hv = cv2.cvtColor(np.clip(bf, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hv[..., 1] = np.clip(hv[..., 1] * _PAPER_IRIS_SAT, 0, 255)   # true hue, balanced with the face
        iris_col = cv2.cvtColor(hv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        iout = np.array(g["bg"], np.float32) * (1 - al) + iris_col * al
        im3 = iris_m[..., None]
        out = out * (1.0 - im3) + iout * im3
    # Limbal ring: a dark rim at the iris edge -- the single strongest cue that
    # reads as a real iris rather than a flat tinted disc. Darken toward the
    # ground in that thin annulus.
    if irises and g["tone"] == "light":
        lim = limbal[..., None]
        out = out * (1.0 - 0.60 * lim) + np.array(g["bg"], np.float32) * (0.60 * lim)
    # Eye-white + teeth (dark ground): neither carries typography. A neutral, dim
    # light tone shaded by the photo's OWN luminance (so it keeps the real bright/
    # shadow gradient, not a flat disc) -- the SAME treatment in every ink. Using
    # the photo's actual pixels in Photo mode read too bright/warm (the whites
    # glowed and picked up the render's cast); the neutral tone never glows and
    # never takes the ink's colour. The iris still carries the subject's real eye
    # colour in Photo mode (handled separately above).
    if (g["tone"] == "light" or ground == "paper") and (irises or teeth is not None):
        gshade = np.clip((gray / 255.0 - 0.20) / 0.55, 0.0, 1.0)
        # On the mid greige paper ground the sclera/teeth must be painted brighter
        # and stronger than on a dark ground, or they read as dirty greige instead
        # of white -- this is what makes the eyes/smile come alive on paper.
        paper_feat = ground == "paper"
        s_str, t_str = (0.90, 0.92) if paper_feat else (0.70, 0.66)
        s_col = (236, 238, 240) if paper_feat else (198, 200, 202)
        t_col = (238, 240, 242) if paper_feat else (200, 202, 204)
        if irises:
            # Natural sclera shading from each eye's OWN luminance, stretched PER EYE:
            # the real upper-lid shadow and corner falloff come through as a GRADIENT
            # instead of a flat grey disc, while per-eye normalisation keeps even a
            # shaded eye bright (preserving the dark-merge fix without the uniform,
            # artificial look). A faint warm-neutral tint reads more like sclera than
            # a cool grey.
            scl_val = _sclera_value(gray, pts, scl, floor=(0.70 if paper_feat else 0.58))
            sw = (scl * scl_val * s_str)[..., None]
            out = out * (1.0 - sw) + np.array(s_col, np.float32) * sw
        if teeth is not None:
            tw = (teeth * gshade * t_str)[..., None]
            out = out * (1.0 - tw) + np.array(t_col, np.float32) * tw
    # Catchlight is a SPECULAR highlight: always white (the lightest thing on the
    # face), never ink- or iris-coloured -- painted over the colour composite.
    if irises and (g["tone"] == "light" or ground == "paper"):
        gl3 = glint[..., None]
        out = out * (1.0 - gl3) + np.float32(250.0) * gl3
    oh = max(1, int(out_width * h0 / w0))
    out = cv2.resize(out, (int(out_width), oh), interpolation=cv2.INTER_AREA)
    # Standard print canvas (4:5 = 16x20), padded with the ground BEFORE vibrance
    # so the band is processed identically to the interior ground (no seam).
    from .tonal import _fit_print_canvas
    out = _fit_print_canvas(out, g["bg"], print_aspect)
    from .preprocess import apply_vibrance
    out = apply_vibrance(out, bgr=True)       # give the render life (clarity/glow/saturation)
    ok, buf = cv2.imencode(".png", np.clip(out, 0, 255).astype(np.uint8))
    if not ok:
        raise ValueError("encode_failed")
    return buf.tobytes()
