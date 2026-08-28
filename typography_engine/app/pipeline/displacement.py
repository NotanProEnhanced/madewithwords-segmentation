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
import os
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

# Grounds that share paper's ink-drawing treatment (density polarity, edge-drawn hair,
# words-form-the-eye). Currently just "paper"; kept as a set so the treatment can be
# extended without touching every gate.
PAPER_FAMILY = frozenset({"paper"})

# "Match your space" backdrop swatches. Unlike a GROUND (which re-renders the whole
# subject on that colour), a backdrop recolours ONLY the segmented background --
# the region OUTSIDE the subject silhouette. The subject is rendered exactly as its
# ground dictates (e.g. the navy Lifelike sculpt), untouched. Values are BGR wall
# colours a buyer can match to a room. `None`/unknown => the legacy TYPO_BG_LIGHTEN
# behaviour (navy sculpt on a lifted-grey backdrop).
BACKDROPS = {
    "studio": (230, 230, 230),  # the DEFAULT "Studio" wall -- a light neutral grey (#e6e6e6).
                                # Explicit (not the legacy TYPO_BG_LIGHTEN lift) so the studio swatch
                                # tile matches the render exactly; near-neutral => vibrance/sat leave it put.
    "gray":  (236, 236, 236),   # soft neutral gallery grey
    "ivory": (232, 240, 244),   # warm off-white
    "sand":  (208, 228, 238),   # warm oat / beige
    "slate": (226, 221, 216),   # cool light slate
    "sage":  (208, 222, 214),   # muted green-grey
    "blush": (222, 222, 236),   # soft warm rose
}

# Floral background FRAMES (memorial). Unlike a BACKDROP (a solid wall colour), a floral
# fills the region OUTSIDE the subject silhouette with a curated watercolour frame on a
# cream ground -- so the portrait reads as ink-on-paper inside a floral mat. A floral pairs
# ONLY with the ink-on-ivory Paper sculpt (forced below), and is composited AFTER the print
# canvas pad so the blooms (corners / borders / side columns) land on the true canvas edges.
# Art lives in static/florals/<key>.png (bind-mounted -> swap the art without a rebuild) at
# 4:5, ideally 4800x6000. A missing/broken file falls back to a flat cream mat (never crashes).
_FLORAL_KEYS = ("wildflowers", "roses", "eucalyptus", "line")
_FLORAL_CREAM = (232.0, 240.0, 244.0)   # BGR, matches the Paper ground so the pad is seamless
_FLORAL_DIR = (os.environ.get("TYPO_FLORAL_DIR", "").strip()
               or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__)))), "static", "florals"))
_floral_cache: Dict[str, Optional[np.ndarray]] = {}


def _pad_floral_4x5(img: np.ndarray) -> np.ndarray:
    """Pad a floral frame to EXACTLY 4:5 (w:h) with the cream ground. The compositor resizes the
    frame to the 4:5 print canvas, so art generated at a slightly different aspect (AI output is
    rarely exact) would otherwise be STRETCHED. The frame's ground is cream, so padding is
    seamless -- edge blooms just sit a hair off the very border instead of being distorted."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    tgt = 4.0 / 5.0                                   # target width/height
    r = w / float(h)
    if abs(r - tgt) < 0.004:                          # already 4:5 -> untouched
        return img
    cream = np.array(_FLORAL_CREAM, np.float32)
    if r < tgt:                                       # too tall -> pad left/right (widen)
        nw = int(round(h * tgt)); l = max(0, (nw - w) // 2)
        out = np.full((h, nw, 3), cream, np.float32); out[:, l:l + w] = img
    else:                                             # too wide -> pad top/bottom (heighten)
        nh = int(round(w / tgt)); t = max(0, (nh - h) // 2)
        out = np.full((nh, w, 3), cream, np.float32); out[t:t + h] = img
    return out


def _load_floral(key: str) -> Optional[np.ndarray]:
    """Return the floral frame as a float32 BGR image (H, W, 3), padded to exact 4:5, or None if
    unavailable (missing file / unreadable) so the caller can fall back to a plain cream mat."""
    key = (key or "").strip().lower()
    if key not in _FLORAL_KEYS:
        return None
    if key not in _floral_cache:
        path = os.path.join(_FLORAL_DIR, key + ".png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)   # BGR; alpha (if any) dropped onto the frame's own ground
        _floral_cache[key] = None if img is None else _pad_floral_4x5(img.astype(np.float32))
    return _floral_cache[key]

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
# reflective lens, where MediaPipe still reports "open"). A real open eye has a DARK
# PUPIL; ratio = pupil darkness (p10 of the central disc) / eye-region bright pixels.
# Real open eyes (dark OR light irises) measured 0.02-0.09; a closed/glare eye (no
# dark pupil) sits far higher. Above this there's no real eye -> suppress instead of
# fabricating one. Threshold keeps a wide margin so light irises are never dropped.
_EYE_OPEN_IRIS_MAX = 0.40
# Dark-lens (sunglasses) guard. A real open eye has a BRIGHT sclera (p90 measured 134-193
# across many faces); a tinted lens darkens the whole eye region (sunglasses measured
# 39-73). Below this there's no real sclera to model -> suppress the fabricated eye and
# render the lens as a solid dark lens, instead of a glowing catchlight/iris or the real
# eye bleeding through the shades. Gated by env TYPO_DARKLENS (default ON; =0 reverts).
_EYE_SCLERA_MIN = 95.0
# Reflective / mirrored sunglasses aren't DARK -- a broad lens reflection reads bright,
# fooling the dark-sclera test. The reliable tell is that a reflective lens makes BOTH
# eyes uniformly bright, whereas a real face is asymmetrically lit -- one eye catches the
# light (its p75 can run high, ~160) while the other stays in shadow (~100). So key off
# the DIMMER eye: only when BOTH eye windows are broadly bright (min p75 over the two eyes
# exceeds this) is the region too uniformly bright to be a real pair of eyes. Keying off
# the max instead false-fired on a single bright real eye (and any auto-levels/brighter
# crop tipped it over), painting dark discs on an open-eyed face.
_EYE_REFLECTIVE_BOTH = 135.0

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


def _sclera_value(gray, all_face_pts, scl, floor=0.60):
    """Per-eye contrast-stretched sclera shading for EVERY subject's eyes. Real sclera
    is not a flat disc: the upper lid shadows its top, it falls off toward the inner/
    outer corners, and it curves away at the edges. Stretching each eye's OWN luminance
    restores that natural gradient, while normalising PER EYE keeps even a shaded eye
    bright (so the dark-merge fix holds without the artificial, uniform-value look).
    `all_face_pts` is a list of per-face landmark arrays; a one-face list reproduces the
    single-subject result exactly (each eye's pixels are set directly, never clamped)."""
    H, W = gray.shape
    val = np.full((H, W), floor, np.float32)
    gb = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    yy = np.arange(H, dtype=np.float32)[:, None]
    for pts in all_face_pts:
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


# Characters kept inside a token. Word-cloud modes keep only hyphen + apostrophe (commas
# etc. are separators there, and "KIND," would read as clutter). A flowing Passage/Letter
# also keeps sentence punctuation so a verse or note reads naturally.
_WORD_PUNCT = "-'"
_FLOW_PUNCT = _WORD_PUNCT + ".,;:!?()&" + '"' + "‘’“”–—…"


def _normalize_words(words: Sequence[str], uppercase: bool = True, keep_punct: bool = False) -> List[str]:
    out: List[str] = []
    allow = _FLOW_PUNCT if keep_punct else _WORD_PUNCT
    for w in words:
        s = str(w).upper() if uppercase else str(w)
        t = "".join(ch for ch in s if ch.isalnum() or ch in allow)
        if t:
            out.append(t)
    return out or (["LOVE"] if uppercase else ["love"])


def _add_discovery(out, pts, fw, H, W, marks):
    """Opt-in 'discovery layer': a small Latin cross plus a few tiny text marks
    (e.g. ["IHS", "JN 8:12"]) placed off the landmarks at low contrast, so they are
    legible only on close inspection -- a detail collectors find over time. ``marks``
    is a sequence of short strings; an empty/None sequence draws only the cross. Drawn
    from geometry/ASCII so it never depends on a glyph the font lacks."""
    marks = [str(m).strip() for m in (marks or []) if str(m).strip()]
    from PIL import Image as _I, ImageDraw as _D
    ov = _I.new("L", (W, H), 0)
    dr = _D.Draw(ov)
    cx = float(np.mean(pts[:, 0]))
    chin = float(pts[:, 1].max())
    f = _font(max(8, int(fw * 0.05)))

    def place(txt, x, y):
        w = dr.textlength(txt, font=f)
        dr.text((x - w / 2.0, y), txt, font=f, fill=255)

    # Latin cross on the chest, below the chin (drawn from lines -> glyph-independent).
    ccx, ccy = cx, min(H - 1.0, chin + fw * 0.85)
    ch = fw * 0.15
    cw = ch * 0.60
    t = max(1, int(fw * 0.012))
    dr.line([(ccx, ccy - ch * 0.5), (ccx, ccy + ch * 0.5)], fill=255, width=t)
    dr.line([(ccx - cw * 0.5, ccy - ch * 0.16), (ccx + cw * 0.5, ccy - ch * 0.16)], fill=255, width=t)
    # Marks: the first sits under the cross; extras spread to a few set spots.
    spots = [(ccx, ccy + ch * 0.55),
             (cx - fw * 0.40, chin - fw * 0.12),
             (cx + fw * 0.34, chin - fw * 0.02),
             (ccx, min(H - 1.0, ccy + ch * 1.15))]
    for i, m in enumerate(marks):
        sx, sy = spots[i % len(spots)]
        place(m, sx, sy)

    ovf = cv2.GaussianBlur(np.asarray(ov).astype(np.float32) / 255.0, (0, 0), 0.6)
    col = np.array((150.0, 158.0, 170.0), np.float32)   # muted warm light (BGR)
    al = (ovf * 0.50)[..., None]
    return out * (1.0 - al) + col * al


def render_displacement_portrait(
    an: Analysis,
    words: Sequence[str],
    ground: str = "navy",
    out_width: int = 1400,
    supersample: int = 2,
    seed: int = 7,
    uppercase: bool = True,
    ink: Optional[str] = None,
    ink_hex: Optional[str] = None,  # only when ink=="custom": the user-picked colour.
                                  # Draped as a single light tint (lifted to read on the
                                  # dark ground) so a Custom pick SCULPTS in its own hue
                                  # instead of falling back to a flat/photo render.
    print_aspect: float = 0.8,    # width/height of the print canvas (4:5 default)
    flow: bool = False,           # True => the text is a MESSAGE: keep it in written
                                  # order and stream it continuously (a sculpted letter),
                                  # instead of importance-weighting a word list.
    variety: float = 0.0,         # 0 = current importance skew (leading words repeat up to
                                  # ~3x, so the portrait is built mostly OF them); ->1 flattens
                                  # the skew so a varied word list reads as varied, not name-dominated.
    breathe: bool = False,        # opt-in Phase-1: tonal 'breathing' -- deep-shadow negative
                                  # space + crisp specular highlight relief, compressing the type
                                  # into the midtones for more sculptural depth. Default OFF.
    discovery: Optional[Sequence[str]] = None,  # opt-in Phase-1: hide a small cross + these tiny
                                  # marks (e.g. ["IHS", "JN 8:12"]) for close-inspection 'discovery'.
                                  # None => nothing drawn (default).
    graduate: bool = True,        # graduate the type OUTSIDE the face -- body (below chin) and hair
                                  # (above face) step down from the largest tier -- plus a hair
                                  # local-contrast 'sculpt'. Default ON; env TYPO_GRADUATE_BODY=0
                                  # reverts with no code change.
    backdrop: Optional[str] = None,  # "match your space" background colour. Recolours ONLY the
                                  # region outside the subject silhouette (a named key in BACKDROPS);
                                  # the subject render is untouched. None => legacy TYPO_BG_LIGHTEN.
    sunglasses: bool = False,     # MANUAL sunglasses control. Pixels cannot reliably tell a
                                  # tinted/reflective lens from a real eye (a dark sclera and a
                                  # mirror lens look identical), so opacity is caller-driven:
                                  # False (default) => NO face is ever dark-lensed, real eyes are
                                  # never blacked out; True => every detected face's eye region is
                                  # rendered as an opaque lens.
    sunglass_faces: Optional[Sequence[int]] = None,  # PER-SUBJECT sunglasses: left-to-right face
                                  # indices the user explicitly marked. When given (not None) this
                                  # is AUTHORITATIVE -- lens EXACTLY those faces, unconditionally,
                                  # and never touch anyone else. No brightness guessing. Overrides
                                  # the global `sunglasses` flag. [] => nobody wears sunglasses.
    _diag: Optional[dict] = None,  # test hook: if a dict is passed, the per-face eye
                                  # classification counts are recorded into it (no effect on
                                  # output). Lets a regression test assert e.g. that a bright
                                  # single face is NOT flagged as a sunglasses/dark lens.
) -> bytes:
    """Render a displacement typographic portrait to PNG bytes.

    Raises ValueError("displacement_needs_face") if no face mesh is available
    (this style is driven by the 478-point landmarks).
    """
    pts0 = an.landmarks.points if an.landmarks is not None else None
    if pts0 is None:
        raise ValueError("displacement_needs_face")

    # A floral frame is chosen via the `backdrop` slot. It must frame the SAME rich sculpt as
    # the default Original look -- the dense photographic word-portrait on the hero dark ground
    # -- NOT the muted ink-on-ivory Paper treatment (which caps brightness/boosts ink and reads
    # flat and washed). So force the hero DARK ground here: the subject renders exactly like a
    # normal Original portrait, and the floral frame (composited below) fills everything OUTSIDE
    # the silhouette with the cream watercolour mat, so the visible background is cream, not navy.
    _floral_key = (backdrop or "").strip().lower()
    _floral_key = _floral_key if _floral_key in _FLORAL_KEYS else None
    if _floral_key:
        ground = "navy"
    g = GROUNDS.get(ground, GROUNDS["navy"])
    rng = random.Random(seed)
    vocab = _normalize_words(words, uppercase, keep_punct=flow)   # keep sentence punctuation only for a flowing Passage/Letter
    # Importance-weighted frequency: words arrive most-important-first (the AI returns
    # them that way; people type the name first), so repeat the leading words more --
    # the portrait is built mostly OF them. Each word's copies are spread evenly across
    # the stream so none clump, and the skew is gentle (top ~3x the tail) to avoid a
    # monotonous look. Tiny lists (<3 words) stay flat.
    if flow or len(vocab) < 3:
        # flow: a message must read in its written order (importance-weighting would
        # multiply and scatter the opening words, shredding the sentence). Tiny lists
        # also stay flat. Everything else gets the importance skew below.
        _vocab_stream = list(vocab)
    else:
        _n = len(vocab)
        # variety dial: 0 keeps the default 3x lead-word skew; 1 flattens it to 1x (every
        # word equally frequent) so a varied list shows its variety instead of the name.
        _top = 3.0 - max(0.0, min(1.0, variety)) * 2.0
        _wts = [max(1, int(round(1.0 + (_top - 1.0) * (1.0 - i / (_n - 1)) ** 1.3))) for i in range(_n)]
        _items = []
        for _i, _w in enumerate(vocab):
            for _c in range(_wts[_i]):
                _items.append(((_c + 0.5) / _wts[_i], _i, _w))   # spread each word's copies evenly
        _items.sort(key=lambda t: (t[0], t[1]))
        _vocab_stream = [t[2] for t in _items]

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
    # Soft alpha matte (hair-preserving feathered edge). Used for the SUBJECT edge and all
    # subject/background compositing so the silhouette doesn't read as a hard "cardboard"
    # cut. Falls back to a gently-blurred binary edge when matting is off/unavailable, so
    # behaviour is unchanged then. mask01 stays BINARY for density/geometry/guards.
    _soft = getattr(an.silhouette, "soft", None)
    if _soft is not None:
        soft01 = np.clip(cv2.resize(_soft.astype(np.float32) / 255.0, (W, H),
                                    interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
        # Clean the faint transition band so background GAPS between hair strands don't
        # carry stray words: push the low end toward 0 while keeping the wisps. A soft
        # knee (below TYPO_MATTE_FLOOR -> 0) plus a gentle gamma. Only when a real matte
        # is present (the coarse fallback edge is already soft, no band to clean).
        _mf = float(os.environ.get("TYPO_MATTE_FLOOR", "0.12") or 0.12)
        _mgam = float(os.environ.get("TYPO_MATTE_GAMMA", "1.5") or 1.5)
        if _mf > 0.0:
            soft01 = np.clip((soft01 - _mf) / max(1e-3, 1.0 - _mf), 0.0, 1.0)
        if _mgam != 1.0:
            soft01 = np.power(soft01, _mgam)
    else:
        soft01 = np.clip(cv2.GaussianBlur(mask01, (0, 0), sigmaX=W * 0.007), 0.0, 1.0)
    pts = pts0 * SS
    # Every detected face's landmarks (primary first), so eyes + facial-feature
    # typography are rendered identically for EVERY subject, not just the largest.
    all_pts = [np.asarray(f.points) * SS for f in (an.faces or [an.landmarks])]
    fbb = an.face_bbox
    fw = (fbb[2] * SS) if fbb else W * 0.55
    face_frac = (fbb[2] / w0) if fbb else 0.55
    s = float(np.clip(face_frac / 0.47, 0.5, 1.3))   # subject-relative scale (hero anchor = 0.47)

    # --- Living eyes: true iris geometry from MediaPipe's iris landmarks ------
    # (centre + 4-point ring per eye, 478-point mesh only). Drives a round pupil,
    # an iris-scaled text tier, a real catchlight, and -- separately gated -- the
    # person's true eye colour. Both irises must resolve large enough to carry
    # structure; otherwise every step below falls back to the legacy behavior.
    # Collect open, unshaded eyes across EVERY face (primary + secondary) so a group
    # portrait renders identical eyes on all subjects. Each face runs the SAME gates:
    # openness (eye-aspect-ratio), dark-pupil backstop, and dark-lens (sunglasses). A
    # face that fails a gate renders as plain words there; the other faces are
    # unaffected. Single-subject renders reproduce the previous behaviour exactly.
    # Opacity is MANUAL now (the `sunglasses` flag), not auto-detected: brightness/pupil
    # heuristics cannot separate a tinted lens from a real eye and kept mis-firing (black
    # holes on real eyes, or fabricated eyes on real sunglasses). TYPO_DARKLENS stays only
    # as a hard kill-switch that can force it off even if a caller passes True.
    _darklens_ok = os.environ.get("TYPO_DARKLENS", "1").strip().lower() not in ("0", "false", "off", "no", "")
    _dl_on = bool(sunglasses) and _darklens_ok
    # PER-SUBJECT sunglasses: the user tapped WHICH faces wear them (left-to-right indices). When
    # provided this is authoritative -- lens exactly those, never guess. Map each face to its
    # left-to-right rank (rank 0 = leftmost) so the client's "Left / Right / Nth-from-left"
    # labels line up with the faces the renderer sees, regardless of MediaPipe's primary-first order.
    _sel_faces = (set(int(i) for i in sunglass_faces) if (sunglass_faces is not None and _darklens_ok) else None)
    _face_cx = [float(np.mean(np.asarray(_p)[:, 0])) for _p in all_pts]
    _lr_order = sorted(range(len(all_pts)), key=lambda i: _face_cx[i])
    _rank_of = {i: r for r, i in enumerate(_lr_order)}     # all_pts index -> left-to-right rank
    irises: List[Tuple[float, float, float]] = []
    _iris_face_idx: List[int] = []     # parallel to `irises`: which face each circle came from
    eye_centers: List[Tuple[float, float, float]] = []
    _eye_face_pts = []            # faces whose eyes ARE rendered (drive the sclera/anchor hulls)
    _dark_lens_active = False
    _dark_lens_eyes = []
    _dark_lens_face_pts = []      # shaded faces (drive the opaque-lens fill)
    _misfit_face_pts = []         # meshes fit so badly the eyes can't be trusted (skip entirely)
    for _pi, _fp in enumerate(all_pts):
        if len(_fp) < 478:
            continue
        # Anatomical sanity: a real iris sits BELOW its own eyebrow. On a busy group a
        # face can get a badly-fit mesh whose iris landmarks land on the FOREHEAD, well
        # above the brows -- and the bright forehead then false-triggers the reflective-
        # lens gate, painting two dark dots up there (and eye rings on the brows). When
        # BOTH irises read clearly above their brow, distrust the whole eye region for
        # this face: no fabricated eyes, no dark-lens fill, no eye anchor -- just words.
        _fh = float(_fp[:, 1].max() - _fp[:, 1].min())
        _gaps = []
        for _ic, _bk in ((468, "Rbrow"), (473, "Lbrow")):
            _brow = np.array([_fp[j] for j in _GROUPS[_bk] if j < len(_fp)], np.float32)
            if _brow.size:
                _gaps.append(float(_fp[_ic][1]) - float(_brow[:, 1].max()))
        # A real iris ALWAYS sits BELOW its own brow (measured gaps +22..+24px, ~ +7% of
        # face height). A misfit/occluded mesh -- e.g. a sunglasses face whose landmarks
        # slide up onto the forehead -- puts the iris ABOVE its brow (measured -9.5..-14.9px,
        # ~ -4% of face height), which fabricates eyes on the lens and a tinted iris on the
        # forehead. Brightness tests are useless there (the mesh is displaced, so they sample
        # the wrong regions); the reliable tell is the iris-above-brow geometry. Suppress the
        # whole face if EITHER iris sits above its brow by more than TYPO_MISFIT_GAP of face
        # height (default 0.02 -- real faces are at +0.07, misfits at -0.04, so this separates
        # them with a wide margin and never touches a real eye).
        _mgap = float(os.environ.get("TYPO_MISFIT_GAP", "0.02") or 0.02)
        if len(_gaps) == 2 and any(g < -_mgap * _fh for g in _gaps):
            _misfit_face_pts.append(_fp)
            continue
        _fi = []
        for ic, ring in ((468, (469, 470, 471, 472)), (473, (474, 475, 476, 477))):
            icx, icy = float(_fp[ic][0]), float(_fp[ic][1])
            ir = float(np.mean([np.hypot(_fp[i][0] - icx, _fp[i][1] - icy) for i in ring]))
            # 8px min iris was tuned at SS=2; scale it by _ssn so the SS=1 PREVIEW
            # resolves the SAME irises the SS=2 paid file does. Without this, a light
            # preview finds <2 irises per face, skips the whole living-eyes + dark-lens
            # detection, and falls back to legacy eye-blobs/rings on EVERY face -- so
            # sunglasses wearers render see-through in the preview though the paid file
            # is correct. At SS=2 the threshold is unchanged (byte-identical).
            # Minimum iris radius for a face to get real eyes. Below this the
            # face is skipped entirely and the legacy blob fallback takes over,
            # which paints a dark disc. Small faces in a group photo sit under
            # the original fixed 8.0. TYPO_IRIS_MIN_PX exposes it.
            if ir >= float(os.environ.get("TYPO_IRIS_MIN_PX", "8.0") or 8.0) * _ssn:
                _fi.append((icx, icy, ir))
        if len(_fi) < 2:
            continue
        eye_centers.extend(_fi)
        # Openness gate (both eyes must read open) -- else plain words for this face.
        def _ear(p1, p2, p3, p4, p5, p6, _p=_fp):
            horiz = float(np.hypot(_p[p1][0] - _p[p4][0], _p[p1][1] - _p[p4][1]))
            if horiz < 1e-3:
                return 0.0
            v = (float(np.hypot(_p[p2][0] - _p[p6][0], _p[p2][1] - _p[p6][1]))
                 + float(np.hypot(_p[p3][0] - _p[p5][0], _p[p3][1] - _p[p5][1])))
            return v / (2.0 * horiz)
        if min(_ear(33, 160, 158, 133, 153, 144), _ear(362, 385, 387, 263, 373, 380)) < _EYE_OPEN_EAR:
            continue
        # PER-SUBJECT sunglasses (authoritative): the user tapped exactly which faces wear them.
        # Lens the selected faces unconditionally; leave everyone else with their real eyes. No
        # brightness heuristic runs at all -- this is what ends the "dark circles on a bare face"
        # class of bug for good.
        if _sel_faces is not None:
            if _rank_of.get(_pi) in _sel_faces:
                _dark_lens_active = True
                _dark_lens_eyes.extend(_fi)
                _dark_lens_face_pts.append(_fp)
                continue
            # not selected -> a bare-eyed subject: skip the lens entirely, render real eyes below.
        # MANUAL sunglasses (legacy global flag): the caller flagged the whole render as having
        # tinted lenses. Pixels can't reliably tell a lens from a real eye, so this heuristic path
        # is only used when no per-face selection was given (old clients / stored records).
        elif _dl_on:
            # The flag is GLOBAL (the whole render), but it must pick WHICH faces actually wear
            # a tinted lens -- in a mixed group one subject may be bare-eyed, and blacking those
            # eyes out paints "dark circles" over a real face. Sample each eye region and decide.
            _lens_max = float(os.environ.get("TYPO_LENS_DARK_MAX", "115") or 115)
            _lens_med_max = float(os.environ.get("TYPO_LENS_DARK_MED", "105") or 105)
            _sc = []      # per-eye p90     -> sclera / lens-glare brightness
            _md = []      # per-eye median  -> overall darkness of the eye region
            _pup = []     # per-eye pupil darkness   (p10 of the central disc)
            _scl = []     # per-eye sclera brightness (p75 of the RING just outside the iris)
            for _icx, _icy, _ir in _fi:
                _r = gray[max(0, int(_icy - _ir * 2.0)):int(_icy + _ir * 2.0),
                          max(0, int(_icx - _ir * 2.0)):int(_icx + _ir * 2.0)]
                if _r.size >= 16:
                    _sc.append(max(float(np.percentile(_r, 90)), 1.0))
                    _md.append(float(np.median(_r)))
                _cx, _cy = int(round(_icx)), int(round(_icy))
                _disc = np.zeros((H, W), np.uint8)                          # pupil: central disc
                cv2.circle(_disc, (_cx, _cy), max(1, int(_ir * 0.45)), 1, -1)
                _ring = np.zeros((H, W), np.uint8)                          # sclera: annulus AROUND the iris
                cv2.circle(_ring, (_cx, _cy), max(2, int(_ir * 1.35)), 1, -1)
                cv2.circle(_ring, (_cx, _cy), max(1, int(_ir * 0.80)), 0, -1)
                if np.any(_disc > 0) and np.any(_ring > 0):
                    _pup.append(float(np.percentile(gray[_disc > 0], 10)))
                    _scl.append(float(np.percentile(gray[_ring > 0], 75)))
            # Real-eye VETO (SPATIAL): a real open eye has a bright sclera in the RING right around
            # the iris and a much darker pupil at the centre. A tinted lens is dark right up to the
            # pupil -- no bright ring. Measuring the ring (not the whole box, whose edges catch
            # skin) is what reliably keeps a BARE-EYED subject from being blacked out ("dark
            # circles") when the toggle is on for someone else. Both eyes must show it.
            # TYPO_LENS_REALEYE = how many times brighter the sclera ring must be than the pupil.
            _realeye = float(os.environ.get("TYPO_LENS_REALEYE", "1.6") or 1.6)
            _real_eye = (len(_scl) >= 2 and min(_scl) >= 80.0
                         and all(_scl[i] >= _realeye * max(_pup[i], 1.0) for i in range(len(_scl))))
            # Apply the opaque lens to a single confirmed subject unconditionally (no group
            # ambiguity), or in a MIXED group only to genuinely dark faces (p90 or median) -- but
            # in BOTH cases never over a real open eye (the veto). A glossy lens whose glare spikes
            # p90 is still caught by the median test.
            _lens_dark = (not _sc) or (max(_sc) < _lens_max) or (bool(_md) and max(_md) < _lens_med_max)
            if not _real_eye and (len(all_pts) <= 1 or _lens_dark):
                _dark_lens_active = True
                _dark_lens_eyes.extend(_fi)
                _dark_lens_face_pts.append(_fp)
                continue
            # else: a real open eye (veto) or bright eyes -> render normally
        # Dark-pupil backstop: a real open eye has a dark pupil in a lighter iris. If the
        # centre isn't darker than the surround the mesh isn't sitting on an eye -- render
        # plain words there rather than fabricate an eyeball.
        ratios = []
        scleras = []
        eye_meds = []
        cheek_meds = []
        for icx, icy, ir in _fi:
            y0, y1 = max(0, int(icy - ir * 2.4)), int(icy + ir * 2.4)
            x0, x1 = max(0, int(icx - ir * 2.4)), int(icx + ir * 2.4)
            reg = gray[y0:y1, x0:x1]
            if reg.size < 16:
                continue
            inner = np.zeros((H, W), np.uint8)
            cv2.circle(inner, (int(round(icx)), int(round(icy))), max(1, int(ir * 0.45)), 1, -1)
            sclera = max(float(np.percentile(reg, 90)), 1.0)
            scleras.append(sclera)
            ratios.append(float(np.percentile(gray[inner > 0], 10)) / sclera)
            eye_meds.append(float(np.median(reg)))
            # Cheek reference: a skin patch BELOW the eye (never under a lens frame).
            cyc, cxc = int(icy + ir * 3.2), int(icx)
            cpatch = gray[max(0, cyc - int(ir)):min(H, cyc + int(ir)),
                          max(0, cxc - int(ir)):min(W, cxc + int(ir))]
            if cpatch.size >= 16:
                cheek_meds.append(float(np.median(cpatch)))
        # Sunglasses backstop (dark OR translucent-tinted lenses). Two independent tells,
        # both dark-only so a real BRIGHT eye can never be suppressed:
        #   * absolute: even the brighter eye region is below _EYE_SCLERA_MIN (opaque lens).
        #   * relative: the eye region reads far DARKER than the cheek skin just below it --
        #     a real eye (bright lids + sclera) never does; a tinted lens does. This catches
        #     translucent lenses the absolute test misses (the eye shows through, but dark).
        # A detected lens is routed to the SAME opaque-lens path as the manual flag, so it
        # also skips the eye anchor/rings (a plain skip left stray rings on the lens).
        # TYPO_DARKSCLERA gates it (default on); TYPO_LENS_DARKRATIO tunes the relative test.
        _cheek = float(np.median(cheek_meds)) if cheek_meds else 0.0
        _lens_ratio = float(os.environ.get("TYPO_LENS_DARKRATIO", "0.62") or 0.62)
        _abs_dark = len(scleras) >= 2 and max(scleras) < _EYE_SCLERA_MIN
        _rel_dark = (len(eye_meds) >= 2 and _cheek > 20.0 and max(eye_meds) < _lens_ratio * _cheek)
        if os.environ.get("TYPO_EYE_DEBUG", "").strip():
            import sys as _sys
            print(f"[eye] fh={_fh:.0f} scleras={[round(s) for s in scleras]} "
                  f"eye_meds={[round(m) for m in eye_meds]} cheek={_cheek:.0f} "
                  f"abs_dark={_abs_dark} rel_dark={_rel_dark} "
                  f"gaps={[round(g, 1) for g in _gaps]}", file=_sys.stderr, flush=True)
        # NOTE: _abs_dark / _rel_dark are computed for the DEBUG LOG ONLY and are NOT used
        # to suppress. Brightness-based lens detection was painting opaque black lenses over
        # deep-set / shadowed REAL eyes (they read just as dark as a tinted lens). Sunglasses
        # are handled by the GEOMETRY guard above (iris-above-brow misfit), which cannot
        # false-fire on a real face. Keeping the brightness signals wired (env + log) so a
        # safer detector can be built from data later, but not acting on them.
        if len(ratios) >= 2 and min(ratios) > _EYE_OPEN_IRIS_MAX:
            continue                                   # no dark pupil -> not a real eye
        irises.extend(_fi)
        _iris_face_idx.extend([_pi] * len(_fi))
        _eye_face_pts.append(_fp)

    if _diag is not None:
        _diag.update(total_faces=len(all_pts), dark_lens_faces=len(_dark_lens_face_pts),
                     eye_faces=len(_eye_face_pts), misfit_faces=len(_misfit_face_pts))

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

    # Dark-lens gray fill: darken the lens region so the density/displacement field stays
    # low there (helps the lens read as an opaque dark surface). Alpha is cleared below too.
    # Anchor it to the EYELID hull (Leye/Reye) -- the reliable lens position -- NOT the iris
    # centres: behind sunglasses MediaPipe frequently guesses the iris landmarks and can drop
    # them onto the brow, so an iris-centred fill paints stray dark circles on the forehead
    # above the lenses. The eyelid hull is where the opaque alpha fill lands too, so the two
    # stay registered on the actual lens.
    if _dark_lens_eyes:
        _dlr = float(np.mean([r for *_c, r in _dark_lens_eyes]))
        _dlm = np.zeros((H, W), np.float32)
        for _fp in _dark_lens_face_pts:
            for k in ("Leye", "Reye"):
                p = np.array([_fp[i] for i in _GROUPS[k] if i < len(_fp)], np.int32)
                if len(p) >= 3:
                    cv2.fillConvexPoly(_dlm, cv2.convexHull(p), 1.0)
        _dk = max(3, int(round(_dlr * 2.0))) | 1        # grow past the lid opening onto the lens
        _dlm = cv2.dilate(_dlm, np.ones((_dk, _dk), np.uint8), 1)
        _dlm = np.clip(cv2.GaussianBlur(_dlm, (0, 0), sigmaX=max(1.0, _dlr * 0.30)), 0, 1)
        gray = gray * (1.0 - 0.90 * _dlm)

    def rows(fs: float) -> np.ndarray:
        f = _font(fs)
        im = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(im)
        y = 0
        if flow:
            # MESSAGE mode: stream the words continuously DOWN the rows, wrapping word
            # by word and looping seamlessly, so the sentence reads in order and then
            # repeats like a refrain across the face. A SHORT looping phrase would tile into
            # a visible "wallpaper" lattice, so offset each row's horizontal start (rows are
            # built 6*fs wider than the canvas, so an offset never leaves a gap). Env-tunable
            # in units of the row font size -- TYPO_FLOW_JITTER=0 restores the old gentle,
            # near-aligned indent; higher scatters more. Seeded rng => preview == paid file.
            _fjit = float(os.environ.get("TYPO_FLOW_JITTER", "3.0") or 3.0)
            adv = {w: float(d.textlength(w + " ", font=f)) for w in set(_vocab_stream)}
            space = max(1.0, float(d.textlength(" ", font=f)))
            n = max(1, len(_vocab_stream))
            wi = 0
            target = float(W) + fs * 6.0
            ry = 0
            while y < H + fs:
                parts, row_w = [], 0.0
                while row_w < target and len(parts) < 20000:   # cap: never hang a row
                    tok = _vocab_stream[wi % n]; wi += 1
                    parts.append(tok); row_w += adv.get(tok, space)
                _ox = -(rng.randint(0, int(fs * _fjit)) if _fjit > 0 else int((ry % 5) * fs * 0.5))
                d.text((_ox, y), " ".join(parts), font=f, fill=0)
                y += max(6, int(fs)); ry += 1
        else:
            # Keep the words in the order they were entered (a sentence stays a
            # sentence); only the row's horizontal start is jittered for variety.
            base = " ".join(_vocab_stream) + " "
            # Repeat only enough to span the canvas + jitter margin. The legacy fixed
            # multiplier (W//(3*fs)+18) assumed a SHORT word list; fed a long list or a
            # pasted tribute it made every row a giant, mostly-off-canvas string -> 60s+
            # renders (the 120s timeout). Sizing by measured width is visually identical
            # (the surplus copies fell off the right edge) but bounds the work. Built
            # once -- it never varied per row.
            bw = max(1.0, float(d.textlength(base, font=f)))
            line = base * max(2, int((W + fs * 7) / bw) + 2)
            while y < H + fs:
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
        for _fp in all_pts:
            for k in keys:
                p = np.array([_fp[i] for i in _GROUPS[k] if i < len(_fp)], np.int32)
                if len(p) >= 3:
                    cv2.fillConvexPoly(mm, cv2.convexHull(p), 1)
        if dil > 0:
            mm = cv2.dilate(mm, np.ones((dil | 1, dil | 1), np.uint8), 1)
        return np.clip(cv2.GaussianBlur(mm.astype(np.float32), (0, 0), sigmaX=max(1.0, sig)), 0, 1)

    feat_damp = mask_of(_GROUPS.keys(), int(fw * 0.06), fw * 0.045)

    fmh = np.zeros((H, W), np.uint8)
    for _fp in all_pts:
        cv2.fillConvexPoly(fmh, cv2.convexHull(_fp.astype(np.int32)), 1)
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
    df = np.clip(0.52 * face_norm + 0.70 * feat_norm, 0, 1)
    # Face-detail floor: the chin, jaw, cheekbones and ears sit INSIDE the face but FAR from
    # the eyes/nose/mouth, so feat_norm ~ 0 there and they render at the same mid size as the
    # neck -- no distinction, and the chin/jaw read too large. Lift df across the whole face
    # (scaled by face_norm) so every facial region reads as FINE detail, clearly finer than
    # the neck below it. Falls off with face_norm, so the neck/body are barely touched.
    # TYPO_FACE_DETAIL is the TARGET fineness FLOOR for the face+ears (0..1): a floor, not an
    # add, so the ear (which starts at df~0 = the LARGEST tier) is forced straight to fine
    # rather than merely nudged. Default 0.8; 0 reverts.
    _fdt = float(os.environ.get("TYPO_FACE_DETAIL", "0.8") or 0.8)
    if _fdt > 0.0:
        # Detail mask = the TIGHT face interior (not the wide face_norm feather, so the chin/
        # jaw/cheekbones get the FULL lift right to the jawline) PLUS an estimated EAR region on
        # each side. MediaPipe has no ear landmarks, so anchor small ellipses just outside the
        # face's lateral extremes at eye level, kept on the subject via mask01. Lifting df here
        # makes chin/jaw/cheeks/ears read as FINE type, clearly finer than the neck below.
        # Tight feather so the floor stays SOLID right to the jawline/chin (a wider feather
        # tapered the floor at the chin edge, so the chin read larger than the mid-face).
        _detail = np.clip(cv2.GaussianBlur(fmh.astype(np.float32), (0, 0), sigmaX=max(1.0, fw * 0.025)), 0, 1)
        _ears = np.zeros((H, W), np.float32)
        for _fp in all_pts:
            _x0, _x1 = float(_fp[:, 0].min()), float(_fp[:, 0].max())
            _y0, _y1 = float(_fp[:, 1].min()), float(_fp[:, 1].max())
            _fwi, _fhi = (_x1 - _x0), (_y1 - _y0)
            _ey = _y0 + _fhi * 0.42                                   # ~eye level
            for _ex in (_x0 - _fwi * 0.03, _x1 + _fwi * 0.03):
                cv2.ellipse(_ears, (int(round(_ex)), int(round(_ey))),
                            (int(round(_fwi * 0.14)), int(round(_fhi * 0.24))), 0, 0, 360, 1.0, -1)
        _ears = np.clip(cv2.GaussianBlur(_ears, (0, 0), sigmaX=max(1.0, fw * 0.04)), 0, 1) * mask01
        _detail = np.clip(np.maximum(_detail, _ears), 0, 1)
        # FLOOR: raise df to at least _fdt across the face+ears (feathered by _detail), so the
        # ear/chin/cheeks are forced to fine no matter their starting size, while the features
        # (already ~1) and the neck (_detail~0) are untouched.
        df = np.maximum(df, _fdt * _detail)
    # === Graduate body + hair type (default ON; TYPO_GRADUATE_BODY=0 reverts) =========
    # Beyond the face the detail field falls to ~0, so the neck/clothing AND the crown of
    # the hair render at the LARGEST tier (giant words). Ramp df UP with distance away from
    # the face -- below the chin and above the face-top -- so the body and hair step DOWN
    # continuously toward the hem/crown. Gated to OUTSIDE the face (the face is unchanged).
    _grad_on = graduate and os.environ.get("TYPO_GRADUATE_BODY", "1").strip().lower() not in ("0", "false", "off", "no", "")
    if _grad_on:
        _gyv = np.arange(H, dtype=np.float32)[:, None]
        _rows_on = np.where(mask01.max(axis=1) > 0)[0]
        _notface = 1.0 - face_norm
        _bottom_y = float(_rows_on.max()) if _rows_on.size else float(H)
        _top_y = float(_rows_on.min()) if _rows_on.size else 0.0
        if len(all_pts) <= 1:
            # Single subject (the memorial portrait). Hold the neck + body + hair AT LEAST as
            # fine as the FACE'S OWN detail floor (_fdt), so they read proportional to the jaw
            # -- uniform, never coarser than the face, no bulge. (The prior fixed target 0.65
            # sat BELOW _fdt=0.8, so the neck rendered COARSER than the jaw -> "still too
            # large".) TYPO_NECK_FINE scales that floor: 1.0 = match the face; >1 = finer than
            # the face; 0 = legacy large body.
            # Use a TIGHT ramp off the chin/face-top (reaches full strength within ~0.15 face-
            # heights) -- NOT _notface, whose big fw*0.22 blur under-boosts the upper neck and
            # leaves large type there. maximum() only raises df, so overlapping the face is safe
            # (the face floor already holds it). TYPO_NECK_FINE scales the floor: 1.0 = match the
            # face; >1 = finer; 0 = legacy large body.
            # FULL-strength boost across the ENTIRE region below the chin-bottom / above the
            # face-top (hard horizontal cut, seamless because the neck target == the face floor
            # so both read the same size); _notface fills the sides. Earlier masks (_notface's
            # wide blur, and a ramp that started at 0 AT the chin) both starved the UPPER neck
            # of the boost -> large type there. This applies it evenly from just under the jaw.
            _neck_scale = float(os.environ.get("TYPO_NECK_FINE", "1.0") or 1.0)
            _neck_target = float(np.clip(_fdt * _neck_scale, 0.0, 1.0))
            # SHARP face-hull complement (tight fw*0.03 feather) so the boost reaches full
            # strength right under the JAWLINE. _notface's wide fw*0.22 blur and the chin-TIP
            # cut both left the under-jaw neck un-boosted -> large type there. Seamless because
            # the neck target == the face floor; df is smoothed downstream.
            _offmask = 1.0 - np.clip(cv2.GaussianBlur(fmh.astype(np.float32), (0, 0),
                                                      sigmaX=max(1.0, fw * 0.03)), 0, 1)
            df = np.clip(np.maximum(df, _neck_target * _offmask), 0, 1)
        else:
            # Group: each subject's neck/chest (and hair/crown) must graduate from THEIR OWN
            # chin, not one group-wide line -- otherwise a taller person's chest sizes
            # differently from a shorter one's and the type jumps between people. Build a
            # smooth per-COLUMN chin / face-top / face-height by weighting every face's value
            # by horizontal proximity (Gaussian on |x - face_centre|, sigma ~ face width), so
            # the reference blends seamlessly across neighbours with no seam. Then, instead of
            # a hard step at the chin (which snaps small->large), hold df up right under the
            # chin and DECAY it over ~0.9 face-heights: the neck/upper-chest eases through a
            # medium band before the long ramp grows the type toward the hem -- small (face) ->
            # medium (neck) -> large (chest), identically for every subject.
            _cx = np.array([0.5 * (float(_p[:, 0].min()) + float(_p[:, 0].max())) for _p in all_pts], np.float32)
            _chin_a = np.array([float(_p[:, 1].max()) for _p in all_pts], np.float32)
            _topa = np.array([float(_p[:, 1].min()) for _p in all_pts], np.float32)
            _fwa = np.array([max(1.0, float(_p[:, 0].max() - _p[:, 0].min())) for _p in all_pts], np.float32)
            _fha = np.array([max(1.0, float(_p[:, 1].max() - _p[:, 1].min())) for _p in all_pts], np.float32)
            _xs = np.arange(W, dtype=np.float32)
            _wt = np.exp(-0.5 * ((_xs[None, :] - _cx[:, None]) / (_fwa[:, None] * 1.1)) ** 2)
            _wt /= (_wt.sum(0, keepdims=True) + 1e-6)
            _chin_col = (_wt * _chin_a[:, None]).sum(0)[None, :]
            _top_col = (_wt * _topa[:, None]).sum(0)[None, :]
            _fh_col = (_wt * _fha[:, None]).sum(0)[None, :]
            _belowm = (_gyv > _chin_col).astype(np.float32) * _notface
            _below = np.clip((_gyv - _chin_col) / np.maximum(1.0, (_bottom_y - _chin_col)), 0.0, 1.0)
            _neck = np.clip(1.0 - (_gyv - _chin_col) / np.maximum(1.0, 0.9 * _fh_col), 0.0, 1.0) * _belowm
            df = np.clip(df + 0.40 * _neck + 0.55 * _below * _notface, 0, 1)
            _abovem = (_gyv < _top_col).astype(np.float32) * _notface
            _above = np.clip((_top_col - _gyv) / np.maximum(1.0, (_top_col - _top_y)), 0.0, 1.0)
            _crown = np.clip(1.0 - (_top_col - _gyv) / np.maximum(1.0, 0.9 * _fh_col), 0.0, 1.0) * _abovem
            df = np.clip(df + 0.40 * _crown + 0.55 * _above * _notface, 0, 1)
    # =================================================================================
    # #1 Forehead is a big smooth plane that large letters dominate -> push the type
    # finer above the brow line so it stops shouting.
    brows = [_p[i] for _p in all_pts for grp in ("Lbrow", "Rbrow") for i in _GROUPS[grp] if i < len(_p)]
    if brows:
        brow_y = float(np.mean([b[1] for b in brows]))
        _yy = np.arange(H, dtype=np.float32)[:, None]
        fh = ((fmh > 0) & (_yy < brow_y)).astype(np.float32)
        df = np.clip(df + 0.26 * cv2.GaussianBlur(fh, (0, 0), sigmaX=max(2.0, fw * 0.05)), 0, 1)
    df = cv2.GaussianBlur(df, (0, 0), sigmaX=max(2.0, fw * 0.06))   # ease the size steps further
    # Re-assert the face-detail FLOOR after the smoothing: the blur above pulls the chin's
    # LOWER edge down toward the larger neck (so the chin bottom read bigger than the forehead).
    # Locking the floor here keeps the whole face -- chin bottom + ears included -- at the fine
    # tier, with the size step to the neck happening right at the jaw, not inside the chin.
    if _fdt > 0.0:
        df = np.maximum(df, _fdt * _detail)
    # #4 Highlights breathe: the brightest skin otherwise keeps full-size, dense type and reads
    # as a flat wash. Push df UP in the brightest ~30% of the face so the type there goes FINER
    # -- smaller, airier words let the highlight breathe instead of caking. Face only; strength
    # TYPO_HILIGHT_FINE (default 0.30; 0 disables).
    _hf = float(os.environ.get("TYPO_HILIGHT_FINE", "0.30") or 0.30)
    if _hf > 0.0:
        _hib = np.clip((gray / 255.0 - 0.72) / 0.28, 0.0, 1.0) * face_norm
        _hib = cv2.GaussianBlur(_hib, (0, 0), sigmaX=max(1.0, fw * 0.03))
        df = np.clip(df + _hf * _hib, 0, 1)

    # Clean vertical drape, dampened in the feature band (keeps features crisp).
    D = cv2.GaussianBlur(gray, (0, 0), sigmaX=W * 0.020)
    dn = (D / 255.0 - 0.5) * 2.0
    xx, yy = np.meshgrid(np.arange(W).astype(np.float32), np.arange(H).astype(np.float32))
    # Drape amplitude: how far the rows ride the facial form (vertical remap by luminance).
    # Higher = more sculptural wrap around brow/nose/cheeks; too high distorts. Env-tunable
    # so it can be dialled on staging without a rebuild (default 64 = unchanged).
    _drape = float(os.environ.get("TYPO_DRAPE", "64") or 64.0)
    amp = _drape * s * _ssn * (1.0 - 0.85 * feat_damp)
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
    # #2/#3 Finer-scale contrast AT the features so the nose (bridge/tip/sides), the
    # smile lines and cheek transitions MODEL instead of reading flat.
    hp2 = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, fw * 0.022))
    hp2 /= (np.std(hp2[mask01 > 0]) + 1e-6)
    ink_field = np.clip(ink_field + 0.32 * sign * np.clip(hp2, -2.0, 2.0) * feat_norm, 0, 1)
    # #7 Micro-expression: the fine-scale contrast above is gated to the eyes/nose/mouth, so
    # the CREASES that make a face recognisable -- nasolabial folds, crow's feet, forehead and
    # smile lines out on the skin -- smooth away. Apply an even finer high-pass across the whole
    # face (face_norm) so those creases render as delicate darker type instead of flat skin.
    # Subject/face only; strength TYPO_CREASE (default 0.22; 0 disables).
    _cr = float(os.environ.get("TYPO_CREASE", "0.22") or 0.22)
    if _cr > 0.0:
        _hpc = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, fw * 0.012))
        _hpc /= (np.std(_hpc[mask01 > 0]) + 1e-6)
        ink_field = np.clip(ink_field + _cr * sign * np.clip(_hpc, -2.0, 2.0) * face_norm, 0, 1)
    # Sculpt the hair (same TYPO_GRADUATE_BODY gate): boost mid-scale local contrast in the
    # hair region (subject, above the chin, outside the face) so strand clumps, volume and
    # highlights MODEL instead of reading as a flat text field.
    if _grad_on:
        _hph = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, fw * 0.030))
        _hph /= (np.std(_hph[mask01 > 0]) + 1e-6)
        _hair_reg = ((mask01 > 0) & (yy < max(float(_p[:, 1].max()) for _p in all_pts))).astype(np.float32) * (1.0 - face_norm)
        _hair_reg = cv2.GaussianBlur(_hair_reg, (0, 0), sigmaX=max(2.0, fw * 0.03))
        ink_field = np.clip(ink_field + 0.60 * sign * np.clip(_hph, -2.0, 2.0) * _hair_reg, 0, 1)
    # #4 Quiet the clothing: below the chin, compress contrast toward the local mean
    # so patterned clothes stop competing with the face (hair untouched).
    chin_y = max(float(_p[:, 1].max()) for _p in all_pts)
    cloth = ((mask01 > 0) & (yy > chin_y)).astype(np.float32)
    cloth = cv2.GaussianBlur(cloth, (0, 0), sigmaX=max(2.0, fw * 0.05))
    # When graduating (same gate), quiet the clothing a bit LESS so some drape shows,
    # then add a gentle fold-scale sculpt -> plain garments gain fold depth while
    # patterned clothes are only mildly livelier (they stay quieter than the face).
    _cq = 0.40 if _grad_on else 0.55
    cm = ink_field[cloth > 0.5]
    if cm.size > 50:
        ink_field = ink_field * (1.0 - _cq * cloth) + float(cm.mean()) * (_cq * cloth)
    if _grad_on:
        _hpc = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, fw * 0.045))
        _hpc /= (np.std(_hpc[mask01 > 0]) + 1e-6)
        ink_field = np.clip(ink_field + 0.30 * sign * np.clip(_hpc, -2.0, 2.0) * cloth, 0, 1)

    # Progressive density: thicken text where ink is strongest.
    b1 = cv2.dilate(warped, np.ones((2, 2), np.uint8), 1)
    b2 = cv2.dilate(warped, np.ones((3, 3), np.uint8), 1)
    gd1 = np.clip((ink_field - 0.40) / 0.60, 0, 1)
    gd2 = np.clip((ink_field - 0.70) / 0.30, 0, 1)
    w2 = np.clip(warped + (b1 - warped) * gd1 + (b2 - b1) * gd2, 0, 1)

    a = np.clip(w2 * (0.04 + 0.96 * np.power(ink_field, 0.62)), 0, 1)
    # Light-aware far-edge softening (prototype): when the subject is DIRECTIONALLY lit, let
    # the SHADOW-side silhouette edge fall off into the ground instead of a crisp cut -- the
    # natural way a portrait's dark side melts into space. Self-gating: light direction is the
    # face's own left/right + top/bottom brightness asymmetry, and confidence is that asymmetry's
    # magnitude, so FLAT lighting -> ~0 confidence -> no effect. Only the shadow-side edge band
    # fades; the lit edge stays crisp. TYPO_EDGE_FALLOFF (0 disables).
    _ef = float(os.environ.get("TYPO_EDGE_FALLOFF", "0.45") or 0.45)
    if _ef > 0.0 and int(np.count_nonzero(mask01 > 0.5)) > 200:
        _fm = mask01 > 0.5
        _xr = np.arange(W, dtype=np.float32)[None, :]
        _yr = np.arange(H, dtype=np.float32)[:, None]
        _n = float(_fm.sum())
        _cx, _cy = float((_xr * _fm).sum() / _n), float((_yr * _fm).sum() / _n)
        _gf = gray.astype(np.float32)
        _R, _L = _fm & (_xr > _cx), _fm & (_xr <= _cx)
        _B, _T = _fm & (_yr > _cy), _fm & (_yr <= _cy)
        _dx = (float(_gf[_R].mean()) - float(_gf[_L].mean())) if _R.any() and _L.any() else 0.0
        _dy = (float(_gf[_B].mean()) - float(_gf[_T].mean())) if _B.any() and _T.any() else 0.0
        _mag = (_dx * _dx + _dy * _dy) ** 0.5
        _conf = min(1.0, _mag / 35.0)                    # 35 grey-levels of asymmetry -> full effect
        if _conf > 0.05 and _mag > 1e-3:
            _sx, _sy = -_dx / _mag, -_dy / _mag          # shadow direction (away from the light)
            _rad = 0.5 * float(fw) + 1e-3
            _proj = np.clip(((_xr - _cx) * _sx + (_yr - _cy) * _sy) / _rad, 0.0, 1.0)   # 1 on the shadow side
            _dist = cv2.distanceTransform(_fm.astype(np.uint8), cv2.DIST_L2, 3)
            _band = np.clip(1.0 - _dist / max(1.0, fw * 0.16), 0.0, 1.0)                # 1 at edge -> 0 inward
            soft01 = soft01 * (1.0 - np.clip(_ef * _conf * _band * _proj, 0.0, 1.0))
    a = a * soft01   # hair-preserving soft matte edge (+ optional light-aware shadow-edge falloff)
    # Highlight wash (light-ground only): a BRIGHT subject region -- silver/white hair, pale
    # skin, specular highlights -- otherwise reads DARK because the navy ground shows through
    # the gaps BETWEEN glyphs. Lift a gentle light floor under the type in the brightest areas
    # so the region reads light (a real highlight) while the glyphs still texture it -- silver
    # hair stays silver instead of collapsing to the ground. Only the brightest ~40% lifts, it
    # stays inside the subject mask, and it fills only the gaps (a += w*(1-a)) so glyph edges
    # keep their crispness. Default 0.0 -> byte-identical; TYPO_HILIGHT_WASH tunes the strength.
    if g["tone"] == "light":
        _hw = float(os.environ.get("TYPO_HILIGHT_WASH", "0.5") or 0.5)   # default ON; 0 disables
        if _hw > 0.0:
            _hi = np.clip((ink_field - 0.60) / 0.40, 0.0, 1.0)
            _mk = np.clip(cv2.GaussianBlur(mask01, (0, 0), sigmaX=W * 0.007), 0, 1)
            a = np.clip(a + _hw * _hi * _mk * (1.0 - a), 0, 1)
    # Shadow lift (light-ground only): the counterpart to the highlight wash. Deep shadow on
    # the subject otherwise crushes to the near-black ground, so the shaded side of a face
    # becomes a detail-less void and the portrait reads harsh. Lift a gentle floor in the
    # shadow-to-low-midtone band, gated to the FACE (face_norm) so the body + background keep
    # their drama, filling only the gaps (a += w*(1-a)) so the glyph texture and the deepest
    # pockets still read dark -- the shaded cheek/jaw keep MODELLED form instead of collapsing.
    # Default 0.0 -> byte-identical; TYPO_SHADOW_LIFT tunes the strength.
    if g["tone"] == "light":
        _sl = float(os.environ.get("TYPO_SHADOW_LIFT", "0.18") or 0.18)   # default ON; 0 disables
        if _sl > 0.0:
            _lo = np.clip((0.50 - ink_field) / 0.50, 0.0, 1.0)   # 1 at black -> 0 at mid(0.5)
            a = np.clip(a + _sl * _lo * np.clip(face_norm, 0, 1) * (1.0 - a), 0, 1)

    # Feature anchoring: eye rings + lip seam + pupils + nostrils.
    anchor = np.zeros((H, W), np.float32)
    th = max(1, int(fw * 0.006))
    # Lips seam is drawn for every face; the eye rings + legacy eye blob are skipped on
    # SHADED (sunglasses) faces only. Drawing an eye ring/blob over an opaque lens stamps
    # a dark outline onto it ("black dots"), so a shaded face gets a lips anchor but no eye
    # anchor. Every other face (real eyes, closed eyes, low-res faces whose irises never
    # resolved) keeps its eye rings exactly as before.
    _no_eye_ids = {id(_fp) for _fp in _dark_lens_face_pts} | {id(_fp) for _fp in _misfit_face_pts}
    _eye_anchor_pts = [_fp for _fp in all_pts if id(_fp) not in _no_eye_ids]
    for _fp in all_pts:
        p = np.array([_fp[i] for i in _GROUPS["lips"] if i < len(_fp)], np.int32)
        if len(p) >= 3:
            cv2.polylines(anchor, [cv2.convexHull(p)], True, 1.0, th, cv2.LINE_AA)
    for _fp in _eye_anchor_pts:
        for k in ["Leye", "Reye"]:
            p = np.array([_fp[i] for i in _GROUPS[k] if i < len(_fp)], np.int32)
            if len(p) >= 3:
                cv2.polylines(anchor, [cv2.convexHull(p)], True, 1.0, th, cv2.LINE_AA)
    # Legacy blob fallback. When no iris resolved, this draws an ink disc at each
    # lid centroid. On small faces that is a dark circle over a real eye -- worse
    # than drawing nothing, since the rest of the face is already typography.
    # TYPO_EYE_BLOB=0 renders those eyes as words instead. Default 1 = unchanged.
    _eye_blob = os.environ.get("TYPO_EYE_BLOB", "1").strip().lower() \
        not in ("0", "false", "off", "no")
    if not irises and _eye_blob:
        # Legacy eye presence: an ink blob at the lid centroid. Only used when the
        # iris landmarks can't resolve -- with real irises the round pupil and
        # catchlight below model the eye properly instead. Skipped on shaded faces so a
        # lens never gets a dark blob.
        for _fp in _eye_anchor_pts:
            for k in ["Leye", "Reye"]:
                c = np.mean([_fp[i] for i in _GROUPS[k]], 0).astype(int)
                cv2.circle(anchor, tuple(c), max(2, int(fw * 0.020)), 1.0, -1, cv2.LINE_AA)
    for _fp in all_pts:
        for i in (98, 327, 2):
            if i < len(_fp):
                cv2.circle(anchor, (int(_fp[i][0]), int(_fp[i][1])), max(1, int(fw * 0.012)), 1.0, -1, cv2.LINE_AA)
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
        # _catchlight_points covers EVERY detected face independently, but the eyes of some
        # faces are suppressed here (sunglasses, no dark pupil, closed). Only paint a glint
        # on a face whose eyes are ACTUALLY rendered (_eye_face_pts); otherwise a stray white
        # dot lands on an opaque lens / occluded eye.
        _eye_boxes = [(_p[:, 0].min(), _p[:, 1].min(), _p[:, 0].max(), _p[:, 1].max())
                      for _p in _eye_face_pts]
        for gx, gy, gr in _catchlight_points(an):
            _gx, _gy = gx * SS, gy * SS
            if not any(_a <= _gx <= _c and _b <= _gy <= _d for (_a, _b, _c, _d) in _eye_boxes):
                continue
            cv2.circle(glint, (int(round(_gx)), int(round(_gy))),
                       max(1, int(round(gr * SS))), 1.0, -1, cv2.LINE_AA)
        ir_mean = float(np.mean([r for _, _, r in irises]))
        pup = np.clip(cv2.GaussianBlur(pup, (0, 0), sigmaX=max(1.0, ir_mean * 0.10)), 0, 1)
        glint = np.clip(cv2.GaussianBlur(glint, (0, 0), sigmaX=max(1.0, ir_mean * 0.10)), 0, 1)
        # No typography in the sclera: inside the eyelid hull but outside the
        # iris, ink is suppressed entirely -- the eye reads as anatomy (clean
        # sclera, typed iris, round pupil, glint), not as text.
        scl = np.zeros((H, W), np.float32)
        for _fp in _eye_face_pts:
            for k in ("Leye", "Reye"):
                p = np.array([_fp[i] for i in _GROUPS[k] if i < len(_fp)], np.int32)
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
            a = np.clip(a + 0.55 * glint, 0, 1)   # catchlight: a tight glint, not a bloom
        else:                                     # dark ink on light paper
            a = np.clip(a + 0.80 * pup, 0, 1)     # pupil: round dark ink
            a = a * (1.0 - 0.85 * glint)          # catchlight: paper shows

    # Opaque dark lens: clear the typography inside a detected tinted lens (eyelid hull,
    # dilated to the lens where reflections sit) so it reads as a solid dark lens instead
    # of a see-through eye. The fabricated-eye pass (iris/catchlight/sclera) is already
    # suppressed above; this removes the words too.
    if _dark_lens_active:
        # Shape the opaque lens like real EYEWEAR, not a circular dilation of the narrow eye
        # slit. The old fill (eyelid hull + a uniform circular dilation) made a round disc that
        # happens to match ROUND frames but reads as "dark circles" on cat-eye / rectangular
        # frames. Instead draw a filled ellipse per eye, sized from the eyelid landmarks --
        # wider than tall (a lens, not a slit) and lifted a touch toward the brow (where glasses
        # actually sit). Two ellipses + the natural bridge gap read as a pair of lenses on any
        # frame style. TYPO_LENS_SIZE scales the whole lens (default 1.0) with no rebuild.
        _lsz = float(os.environ.get("TYPO_LENS_SIZE", "1.0") or 1.0)
        _lens = np.zeros((H, W), np.float32)
        for _fp in _dark_lens_face_pts:
            for k in ("Leye", "Reye"):
                _idx = [i for i in _GROUPS[k] if i < len(_fp)]
                if len(_idx) < 3:
                    continue
                _p = np.array([_fp[i] for i in _idx], np.float32)
                _cx = float(_p[:, 0].mean())
                _cy = float(_p[:, 1].mean())
                _ew = float(_p[:, 0].max() - _p[:, 0].min())     # eye-slit width
                _eh = float(_p[:, 1].max() - _p[:, 1].min())     # eye-slit height (small)
                _ax = max(4, int(round(_ew * 0.80 * _lsz)))                       # half-width  (~1.6x the slit)
                _ay = max(4, int(round(max(_eh * 1.15, _ew * 0.52) * _lsz)))      # half-height (a real lens, not a slit)
                _ecy = int(round(_cy - _eh * 0.25))                              # sit a touch high (toward the brow)
                cv2.ellipse(_lens, (int(round(_cx)), _ecy), (_ax, _ay), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        _lensf = np.clip(cv2.GaussianBlur(_lens, (0, 0), sigmaX=max(1.0, fw * 0.012)), 0, 1)
        a = a * (1.0 - 0.97 * _lensf)

    # Teeth carry NO typography. Where the mouth is open, suppress ink across the
    # inner mouth (both tones); on a dark ground the cleared teeth get a soft
    # light wash below, on light paper the paper already reads as teeth. A closed
    # mouth yields no mask and is left untouched.
    from .tonal import _teeth_mask
    # Per-face gate. Merging first and testing the union let one subject's dark
    # lip crease validate "open mouth" for every face in the photo -- two closed
    # mouths rendered as two pale blobs. Each face is now judged on its own
    # pixels and dropped before it can contribute to the union.
    _tdark = float(os.environ.get("TYPO_TEETH_DARK", "60.0") or 60.0)
    _tbright = float(os.environ.get("TYPO_TEETH_BRIGHT", "205.0") or 205.0)
    _tdbg = os.environ.get("TYPO_TEETH_DEBUG", "").strip().lower() in ("1", "true", "on", "yes")
    teeth = None                       # union of the mouths that really are open
    for _fi, _fp in enumerate(all_pts):
        _tm = _teeth_mask(_fp, H, W)
        if _tm is None:
            continue
        # A REAL open mouth has a dark cavity OR genuinely bright teeth; a
        # falsely-detected one is uniform lip tone.
        _tpx = gray[_tm > 0.5]
        if _tpx.size > 10:
            _p10 = float(np.percentile(_tpx, 10))
            _p90 = float(np.percentile(_tpx, 90))
            _closed = (_p10 > _tdark and _p90 < _tbright)
            if _tdbg:
                try:
                    print("[teeth] face=%d p10=%.1f p90=%.1f dark<=%.1f bright>=%.1f -> %s"
                          % (_fi, _p10, _p90, _tdark, _tbright,
                             "closed" if _closed else "KEPT"))
                except Exception:
                    pass
            if _closed:
                continue
        teeth = _tm if teeth is None else np.maximum(teeth, _tm)
    if teeth is not None:
        a = a * (1.0 - 0.92 * teeth)
    if ground in PAPER_FAMILY:
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
    # === Phase-1 (opt-in): tonal breathing =================================
    # Compress the typography into the MIDTONES so the tonal extremes rest, the way a
    # charcoal portrait reads as 3-D: let the ground show through the deepest facial
    # shadows (negative space), and give the brightest specular skin clean relief.
    # Restricted to the face plane (hair/robe untouched); eyes+lips excluded so the
    # catchlight/anchors are preserved. Default OFF -> when ``breathe`` is False this
    # block is skipped and the output is byte-identical to before.
    _hl = None
    if breathe:
        _face_reg = np.clip(face_norm, 0.0, 1.0)
        _eyeblock = np.zeros((H, W), np.float32)
        for _k in ("Leye", "Reye", "lips"):
            _p = np.array([pts[i] for i in _GROUPS[_k] if i < len(pts)], np.int32)
            if len(_p) >= 3:
                cv2.fillConvexPoly(_eyeblock, cv2.convexHull(_p), 1.0)
        _eyeblock = cv2.dilate(_eyeblock, np.ones((9, 9), np.uint8), 1)
        _eyeblock = np.clip(cv2.GaussianBlur(_eyeblock, (0, 0), sigmaX=max(1.0, fw * 0.02)), 0, 1)
        _skin = _face_reg * (1.0 - _eyeblock)
        _shadow = np.clip((0.33 - lum) / 0.33, 0.0, 1.0) * _skin      # deepest facial shadow
        a = a * (1.0 - 0.66 * _shadow)
        _hl = np.clip((lum - 0.88) / 0.12, 0.0, 1.0) * _skin          # brightest specular skin
        _hl = np.power(_hl, 1.4)                                       # roll off -> only true peaks
        _hl = cv2.GaussianBlur(_hl, (0, 0), sigmaX=max(1.0, fw * 0.009))
        a = a * (1.0 - 0.32 * _hl)
    # =======================================================================

    # Paper/ink ground: on light skin the whole face is highlight, so the density falls away
    # and the features wash out (worst on fair, older subjects). Keep a gentle ink floor inside
    # the face so the likeness reads while the surrounding paper still breathes. TYPO_PAPER_FACE
    # (default 0.30; 0 disables). Paper ground only -- dark grounds are unaffected.
    if ground in PAPER_FAMILY:
        _pf = float(os.environ.get("TYPO_PAPER_FACE", "0.30") or 0.30)
        if _pf > 0.0:
            a = np.maximum(a, w2 * _pf * np.clip(face_norm, 0, 1))
        # Light-hair floor: silver/blonde hair on the ivory washes out (light on light), so a
        # silver-haired subject reads as a floating face. Give the HAIR region an ink-density
        # floor so light hair renders as delicate grey words instead of vanishing. Dark hair
        # already has density (max() leaves it untouched). TYPO_PAPER_HAIR (default 0.35; 0 off).
        _ph = float(os.environ.get("TYPO_PAPER_HAIR", "0.35") or 0.35)
        if _ph > 0.0:
            _yy2 = np.arange(H, dtype=np.float32)[:, None]
            _chin2 = max(float(_p[:, 1].max()) for _p in all_pts)
            _hairm = ((mask01 > 0) & (_yy2 < _chin2)).astype(np.float32) * (1.0 - np.clip(face_norm, 0, 1))
            _hairm = np.clip(cv2.GaussianBlur(_hairm, (0, 0), sigmaX=max(2.0, fw * 0.03)), 0, 1)
            a = np.maximum(a, w2 * _ph * _hairm)

    al = a[..., None]
    if ink == "photo" or ink == "mono":
        # Photo Lifelike composite. Noir (mono) shares this exact path -- full tonal range,
        # polarity shadows, living eyes -- and is desaturated to black & white at the end,
        # so it's a B&W Lifelike, not the old flat single-ink sculpt.
        # Words take the photo's OWN colours, draped over the form, on the ground.
        bgr_full = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        if _eye_deglare is not None:        # tone the suppressed-eye glare out of the colour too
            gm, skin = _eye_deglare
            bgr_full = bgr_full * (1.0 - gm[..., None]) + np.float32(skin) * gm[..., None]
        hsv = cv2.cvtColor(np.clip(bgr_full, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        if ground in PAPER_FAMILY:
            # Ink-drawing with COLOURED glyphs: the words keep the photo's hue at high
            # saturation but a capped dark value, so each word reads as coloured TYPE on
            # ivory (skin/lips/eyes show) -- colour from the glyphs, not a photo overlay.
            # Tone is the ink DENSITY applied above; minimum() keeps deep shadows deep.
            hsv[..., 1] = np.clip(hsv[..., 1] * _PAPER_INK_SAT, 0, 255)
            hsv[..., 2] = np.minimum(hsv[..., 2], np.float32(_PAPER_INK_VALUE))
        else:
            hsv[..., 1] = np.clip(hsv[..., 1] * float(os.environ.get("TYPO_INK_SAT", "1.02") or 1.02), 0, 255)  # step-3 colour-fidelity knob (was fixed 1.02)
            # On a dark ground the gaps between glyphs show GROUND, so the render reads
            # darker than the source photograph. This lifts the ink value to compensate.
            # Multiplier and offset were hardcoded at 1.14 / 14 -- both now tunable.
            _ilm = float(os.environ.get("TYPO_INK_LIFT", "1.14") or 1.14)
            _ila = float(os.environ.get("TYPO_INK_LIFT_ADD", "14") or 14.0)
            hsv[..., 2] = np.clip(hsv[..., 2] * _ilm + _ila, 0, 255)     # lift value vs dark ground
        ink_col = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        # SUBJECT BASE: the ground is painted across the whole canvas, so INSIDE the
        # silhouette it shows through every gap between glyphs -- the face reads as ground
        # colour rather than skin, and the portrait comes out far darker than the source.
        # TYPO_SUBJECT_BASE swaps the base inside the mask for the SOURCE PHOTO, dimmed by
        # TYPO_SUBJECT_DIM so the words still read on top of it. The flat ground stays
        # BEHIND the subject. 0 (default) is byte-identical to the previous behaviour.
        _sb = float(os.environ.get("TYPO_SUBJECT_BASE", "0") or 0.0)
        _base = np.zeros((H, W, 3), np.float32) + np.array(g["bg"], np.float32)
        if _sb > 0.0:
            _dim = float(os.environ.get("TYPO_SUBJECT_DIM", "0.45") or 0.0)
            _m3 = (np.clip(mask01, 0, 1) * min(max(_sb, 0.0), 1.0))[..., None]
            _base = _base * (1.0 - _m3) + (bgr_full * (1.0 - _dim)) * _m3
        # Field dump (TYPO_DUMP_FIELDS=<dir>). The composite below is fully
        # determined by _base, al and ink_col, so when a render is wrong the
        # answer is in one of them -- no inference required.
        _dd = os.environ.get("TYPO_DUMP_FIELDS", "").strip()
        if _dd:
            try:
                os.makedirs(_dd, exist_ok=True)

                def _dump(_nm, _arr):
                    _a = np.asarray(_arr, np.float32)
                    if _a.ndim == 3 and _a.shape[2] == 1:
                        _a = _a[..., 0]
                    if float(_a.max()) <= 1.001:
                        _a = _a * 255.0
                    cv2.imwrite(os.path.join(_dd, _nm + ".png"),
                                np.clip(_a, 0, 255).astype(np.uint8))

                _dump("mask01", mask01)
                _dump("soft01", soft01)
                _dump("alpha", al)
                _dump("base", _base)
                _a1 = np.asarray(al, np.float32)
                if _a1.ndim == 3:
                    _a1 = _a1[..., 0]
                print("[dump] %s  alpha mean=%.3f p95=%.3f   soft01 mean=%.3f   "
                      "base mean=%.1f" % (_dd, float(_a1.mean()),
                                          float(np.percentile(_a1, 95)),
                                          float(np.asarray(soft01, np.float32).mean()),
                                          float(np.asarray(_base, np.float32).mean())))
            except Exception as _e:  # noqa: BLE001
                print("[dump] failed: %s" % _e)
        out = _base * (1 - al) + ink_col * al
        if os.environ.get("TYPO_POLARITY", "0").strip().lower() in ("1", "true", "on", "yes"):
            # Polarity model (the paper-grade shadow behaviour, brought to the dark-ground
            # Lifelike look). Instead of "light ink whose COVERAGE follows brightness"
            # (shadow -> no ink -> ground shows -> absence), make the type present at HIGH
            # coverage everywhere and carry the tone in the LETTER COLOUR across the full
            # range: near-black letters in deep shadow (a heavy dark mass to lean into),
            # light letters in highlight. Two tuning knobs (env; iterate without a rebuild):
            #   TYPO_POLARITY_GAMMA (>1 drives deep shadow harder to black; default 1.35)
            #   TYPO_POLARITY_FLOOR (how black the shadow GAPS get; 0 = true black; default 0.18)
            _pol_g = float(os.environ.get("TYPO_POLARITY_GAMMA", "1.35") or 1.35)
            _pol_f = float(os.environ.get("TYPO_POLARITY_FLOOR", "0.18") or 0.18)
            _mkf = np.clip(cv2.GaussianBlur(mask01, (0, 0), sigmaX=W * 0.007), 0, 1)
            _tone = np.clip(lum, 0.0, 1.0)[..., None] ** _pol_g  # gamma>1 -> deep shadow drives to near-black
            _pc = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            _pl = (_pc[..., 0] * 0.114 + _pc[..., 1] * 0.587 + _pc[..., 2] * 0.299)[..., None] + 1e-3
            _ink = _pc / _pl * (3.0 + 250.0 * _tone)             # keep the photo HUE, re-map brightness full-range
            _ink = np.minimum(_ink, np.float32([255, 255, 255])) # (hue*value can exceed 255 on saturated pixels)
            # Coverage rises INTO the shadows (heavier, denser type there) and eases in highlights, and
            # the glyph field w2 keeps the letterforms visible. Deep shadow = a dense near-black letter
            # mass ("black to lean into"); highlight = lighter, airier letters. Ground shows in the gaps
            # so it still reads as words, not a photo.
            _dark = 1.0 - _tone[..., 0]
            _cov = np.clip((0.28 + 0.22 * _dark) + 0.55 * w2, 0.0, 1.0) * _mkf
            # Darken the LOCAL ground (the gaps) toward black in deep shadow too, so the shadow
            # reads as true black -- not the mid navy -- giving the piece a black to lean into.
            _bg_local = np.array(g["bg"], np.float32) * np.clip(_pol_f + (1.0 - _pol_f) * _tone, 0.0, 1.0)
            out = _bg_local * (1 - _cov[..., None]) + np.clip(_ink, 0, 255) * _cov[..., None]
    elif ink in _SCULPT_INK:
        word = np.array(_SCULPT_INK[ink], np.float32)
        out = np.array(g["bg"], np.float32) * (1 - al) + word * al
    elif ink == "custom" and ink_hex:
        # A user-picked colour, sculpted as a single light tint. Reuse the poster
        # helper's dark-colour lift (hue preserved, brightened if it's too dark to
        # read on the dark ground), then drape it like any other sculpt ink.
        from .tonal import custom_poster
        _cp = custom_poster(ink_hex)
        if _cp:
            _h = _cp[1].lstrip("#")
            word = np.array((int(_h[4:6], 16), int(_h[2:4], 16), int(_h[0:2], 16)), np.float32)  # RGB hex -> BGR
        else:
            word = np.array(g["ink"], np.float32)
        out = np.array(g["bg"], np.float32) * (1 - al) + word * al
    else:
        out = np.array(g["bg"], np.float32) * (1 - al) + np.array(g["ink"], np.float32) * al

    # Living eyes, colour: glyphs inside the iris carry the person's TRUE eye
    # colour -- sampled by the shared gated helper (both irises saturated and
    # hue-consistent, else no tint; sampled, never invented). Dark grounds only:
    # the lifted tint is designed for light-ink-on-dark.
    # TYPO_EYE_PLAIN: render the eye as TYPE and nothing else. The iris tint, limbal
    # ring, sclera paint and photographic paste are all gated on `irises`, so emptying
    # it here disables the entire synthesis in one place. Nothing is drawn from landmark
    # geometry, so a badly fitted mesh cannot place a disc where no eye is. Teeth are
    # unaffected (their gate is `irises or teeth`).
    if os.environ.get("TYPO_EYE_PLAIN", "").strip().lower() in ("1", "true", "on", "yes"):
        irises = []
        iris_m = None
    if irises and iris_m is not None and g["tone"] == "light" and ink in ("photo", "mono"):
        from .tonal import _iris_tint, _iris_tint_face

        def _iris_layer(_tint):
            """The colour laid inside an iris: the sampled tint when the gate passed, else
            the source's own iris pixels lifted so a dark brown reads on the dark ground."""
            if _tint is not None:
                _tp = np.array(_tint[1][::-1], np.float32)   # lifted RGB -> BGR
                _iaa = float(os.environ.get("TYPO_IRIS_ALPHA", "0") or 0.0)
                _iall = np.maximum(al, _iaa) if _iaa > 0.0 else al
                return np.array(g["bg"], np.float32) * (1 - _iall) + _tp * _iall
            _ill = float(os.environ.get("TYPO_IRIS_LIFT", "1.35") or 1.35)
            _bff = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            _hvv = cv2.cvtColor(np.clip(_bff, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            _hvv[..., 1] = np.clip(_hvv[..., 1] * 1.3, 0, 255)
            _hvv[..., 2] = np.clip(_hvv[..., 2] * _ill + 40, 0, 255)
            _icol = cv2.cvtColor(_hvv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
            return np.array(g["bg"], np.float32) * (1 - al) + _icol * al

        if (os.environ.get("TYPO_IRIS_PER_FACE", "").strip().lower() in ("1", "true", "on", "yes")
                and _iris_face_idx):
            # Each face gets ITS OWN sampled colour on ITS OWN irises. Previously one tint
            # from faces[0] was painted onto every iris in the image, so a mixed-eye-colour
            # group inherited the primary face's eyes. A face whose gate rejects now falls
            # back alone rather than forcing the fallback on everyone.
            for _fx in sorted(set(_iris_face_idx)):
                _sel = [_c for _c, _fi2 in zip(irises, _iris_face_idx) if _fi2 == _fx]
                if not _sel:
                    continue
                _fm = np.zeros((H, W), np.float32)
                for (_ccx, _ccy, _rr) in _sel:
                    cv2.circle(_fm, (int(round(_ccx)), int(round(_ccy))), int(round(_rr)),
                               1.0, -1, cv2.LINE_AA)
                _rmean = float(np.mean([_c[2] for _c in _sel]))
                _fm = np.clip(cv2.GaussianBlur(_fm, (0, 0), sigmaX=max(1.0, _rmean * 0.18)), 0, 1)
                _fm3 = _fm[..., None]
                out = out * (1.0 - _fm3) + _iris_layer(_iris_tint_face(an, _fx)) * _fm3
            tint = None
            im3 = None
        else:
            tint = _iris_tint(an)
            im3 = iris_m[..., None]
        if im3 is not None and tint is not None:
            tip = np.array(tint[1][::-1], np.float32)        # lifted RGB -> BGR
            # The iris is composited over the GROUND, so wherever ink alpha is low the navy
            # ground (13,27,58 RGB -- a saturated dark blue) shows through and a correctly
            # sampled BROWN iris renders BLUE. TYPO_IRIS_ALPHA floors the alpha inside the
            # iris so the sampled colour wins. 0 (default) keeps the previous behaviour.
            _ia = float(os.environ.get("TYPO_IRIS_ALPHA", "0") or 0.0)
            _ial = np.maximum(al, _ia) if _ia > 0.0 else al
            iout = np.array(g["bg"], np.float32) * (1 - _ial) + tip * _ial
            out = out * (1.0 - im3) + iout * im3
        elif im3 is not None:
            # The tint gate rejected the sample -- most often a low-saturation BROWN iris in
            # shadow. Without word-eyes the photo paste covered this; with word-eyes the iris
            # would otherwise fall through to the dark navy ground and read BLUE. Re-lay the
            # source's own iris pixels (saturation + a value lift so a dark brown reads on the
            # dark ground), so a brown eye renders brown -- not blue. TYPO_IRIS_LIFT tunes it.
            _il = float(os.environ.get("TYPO_IRIS_LIFT", "1.35") or 1.35)
            bf = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            hv = cv2.cvtColor(np.clip(bf, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hv[..., 1] = np.clip(hv[..., 1] * 1.3, 0, 255)
            hv[..., 2] = np.clip(hv[..., 2] * _il + 40, 0, 255)
            iris_col = cv2.cvtColor(hv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
            iout = np.array(g["bg"], np.float32) * (1 - al) + iris_col * al
            out = out * (1.0 - im3) + iout * im3
    elif irises and iris_m is not None and ground in PAPER_FAMILY and ink == "photo":
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
    if (g["tone"] == "light" or ground in PAPER_FAMILY) and (irises or teeth is not None):
        gshade = np.clip((gray / 255.0 - 0.20) / 0.55, 0.0, 1.0)
        # On the mid greige paper ground the sclera/teeth must be painted brighter
        # and stronger than on a dark ground, or they read as dirty greige instead
        # of white -- this is what makes the eyes/smile come alive on paper.
        paper_feat = ground in PAPER_FAMILY
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
            scl_val = _sclera_value(gray, _eye_face_pts, scl, floor=(0.70 if paper_feat else 0.58))
            sw = (scl * scl_val * s_str)[..., None]
            out = out * (1.0 - sw) + np.array(s_col, np.float32) * sw
        if teeth is not None:
            tw = (teeth * gshade * t_str)[..., None]
            out = out * (1.0 - tw) + np.array(t_col, np.float32) * tw
    # Catchlight is a SPECULAR highlight: always white (the lightest thing on the
    # face), never ink- or iris-coloured -- painted over the colour composite.
    if irises and (g["tone"] == "light" or ground in PAPER_FAMILY):
        gl3 = glint[..., None]
        out = out * (1.0 - gl3) + np.float32(238.0) * gl3   # bright glint, below blow-out so vibrance doesn't bloom it
    # Realistic eyes: composite the photo's OWN eye openings, tone-normalised, OVER the
    # synthetic fill -- the real eye never glows (the synthetic bright sclera/catchlight
    # does). Applied for EVERY ink on a dark ground; the Photo ink keeps it full colour,
    # the tinted/monochrome inks (Noir/Sepia/Navy/Sage) then DESATURATE it into the ink's
    # palette so a full-colour eye doesn't clash with the tinted face. Gated by the
    # openness check (closed eyes skipped); paper keeps its words-form-the-eye treatment.
    # Word-formed eyes (paper's treatment, brought to the dark ground): when TYPO_WORD_EYES
    # is on, the Photo Lifelike look SKIPS this photographic paste and lets the eye be built
    # from the synthetic sclera + tinted-iris words + limbal ring + catchlight already laid
    # above -- so the eye reads as part of the typography, not a photo patch. Tinted inks
    # (Noir/Sepia/...) still get the photographic eye (they have no colour clash to word-form).
    _word_eyes = os.environ.get("TYPO_WORD_EYES", "0").strip().lower() in ("1", "true", "on", "yes")
    if irises and g["tone"] == "light" and not (_word_eyes and ink in ("photo", "mono")):
        from .tonal import _photo_eye_overlay
        bgr_eye = cv2.resize(an.img.bgr, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # Composite the REAL photo eye for EVERY subject's eyes (this is what makes an
        # eye read as real vs. a synthetic dark disc) -- not just the primary face.
        eye_bgr = bgr_eye.copy()
        eye_a = np.zeros((H, W), np.float32)
        for _fp in _eye_face_pts:
            _eb, _ea = _photo_eye_overlay(bgr_eye, _fp, (_GROUPS["Leye"], _GROUPS["Reye"]), H, W)
            _tk = _ea > eye_a
            eye_a = np.where(_tk, _ea, eye_a)
            eye_bgr[_tk] = _eb[_tk]
        a3 = (eye_a * float(os.environ.get("TYPO_EYE_PHOTO", "0.5") or 0.5))[..., None]   # 1=opaque photo eye; lower (default 0.5) blends the typography through so the eye reads as part of the words
        out = out * (1.0 - a3) + eye_bgr * a3
        # A NON-closeup source has small, soft eyes, so the pasted eye reads flat/muddy.
        # Sharpen + lift local contrast INSIDE the eye opening so the iris/pupil/catchlight
        # read crisp; scale the amount UP for smaller (more distant) eyes. TYPO_EYE_SHARPEN
        # (0 disables) sets the base strength.
        _esh = float(os.environ.get("TYPO_EYE_SHARPEN", "0.6") or 0.6)
        if _esh > 0.0 and float(eye_a.max()) > 0.0:
            _em = np.clip(eye_a, 0.0, 1.0)[..., None]
            _amt = _esh * float(np.clip((0.16 * W) / max(1.0, fw), 0.6, 2.4))   # smaller face -> more
            _blur = cv2.GaussianBlur(out, (0, 0), sigmaX=max(1.0, fw * 0.010))
            _enh = np.clip((out - 128.0) * 1.12 + 128.0 + _amt * (out - _blur), 0.0, 255.0)
            out = out * (1.0 - _em) + _enh * _em
        # Eye "pop": re-assert a crisp specular CATCHLIGHT + a dark LIMBAL rim ON TOP of the
        # photo eye. "Flat and muddy" = a soft source lost its key-light glint and iris rim;
        # painting them back (like real portrait retouching) makes even a low-res eye read
        # alive and defined. Scaled UP for smaller eyes; TYPO_EYE_POP (0 disables). A faint
        # pupil-core darken adds depth. Runs before the mono desaturate so Noir gets it too.
        _pop = float(os.environ.get("TYPO_EYE_POP", "0.8") or 0.8)
        if _pop > 0.0 and float(glint.max()) > 0.0:
            _psc = float(np.clip((0.16 * W) / max(1.0, fw), 0.7, 2.2))   # smaller face -> more pop
            _l3 = np.clip(limbal * (0.6 * _pop), 0.0, 1.0)[..., None]
            out = out * (1.0 - _l3) + (out * 0.32) * _l3                 # dark iris rim (definition)
            _g3 = np.clip(glint * (_pop * _psc), 0.0, 1.0)[..., None]
            out = out * (1.0 - _g3) + np.float32(246.0) * _g3           # bright specular catchlight
        if ink not in ("photo", "mono"):             # mono desaturates the WHOLE subject below
            lum = (out[..., 0] * 0.114 + out[..., 1] * 0.587 + out[..., 2] * 0.299)[..., None]
            grayed = out * 0.22 + lum * 0.78         # pull the eye toward the ink's monochrome
            em3 = eye_a[..., None]
            out = out * (1.0 - em3) + grayed * em3
    # === Phase-1 (opt-in): highlight glaze + discovery layer ===============
    if breathe and _hl is not None:
        _glz = (0.34 * _hl)[..., None]                          # crisp specular relief, not fog
        _bright = np.array((232.0, 236.0, 240.0), np.float32)   # warm near-white (BGR)
        out = out * (1.0 - _glz) + _bright * _glz
    if discovery:
        out = _add_discovery(out, pts, fw, H, W, discovery)
    # =======================================================================

    # Noir = the finished Lifelike render in black & white. Desaturate to luminance with a
    # gentle contrast lift so the grayscale is punchy, not muddy -- keeping the polarity
    # shadows, catchlight and living eyes intact. TYPO_NOIR_CONTRAST tunes the punch.
    if ink == "mono":
        _nc = float(os.environ.get("TYPO_NOIR_CONTRAST", "1.08") or 1.08)
        _lo = out[..., 0] * 0.114 + out[..., 1] * 0.587 + out[..., 2] * 0.299
        _lo = np.clip((_lo - 128.0) * _nc + 128.0, 0, 255)
        out = np.stack([_lo, _lo, _lo], axis=-1)

    # De-posterize: the tonal floors (highlight wash / shadow lift) and the discrete text-
    # density steps flatten the face into bands. Add the photo's OWN low-frequency light->dark
    # falloff back as a gentle multiply -- brighter where the photo is bright, darker where it
    # is dark -- so the flat regions regain smooth photographic gradation WITHOUT blurring the
    # glyph edges (only the LOW frequencies move, the type stays crisp). Subject only, light
    # ground only. Default 0 -> byte-identical; TYPO_DEPOSTERIZE tunes the strength.
    _dp = float(os.environ.get("TYPO_DEPOSTERIZE", "0.6") or 0.6)   # default ON; 0 disables
    if _dp > 0.0 and g["tone"] == "light":
        _plo = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (0, 0), sigmaX=max(2.0, fw * 0.11))
        _m = np.clip(cv2.GaussianBlur(mask01, (0, 0), sigmaX=W * 0.01), 0, 1)
        _mid = float(np.mean(_plo[mask01 > 0])) if np.any(mask01 > 0) else 0.5
        _mod = 1.0 + _dp * np.clip(_plo - _mid, -0.5, 0.5) * _m
        out = np.clip(out * _mod[..., None], 0, 255)

    oh = max(1, int(out_width * h0 / w0))
    out = cv2.resize(out, (int(out_width), oh), interpolation=cv2.INTER_AREA)
    # Background fill: recolour the region OUTSIDE the subject silhouette. The subject
    # (on its own ground -- e.g. the navy Lifelike sculpt) is NEVER touched; only pixels
    # outside the silhouette move. Two sources, in priority order:
    #   1. An explicit `backdrop` swatch (the "match your space" wall colour) -- fills
    #      with that colour regardless of ground. This is the user-facing feature.
    #   2. Else the legacy env TYPO_BG_LIGHTEN lift (navy sculpt on a lighter grey),
    #      dark grounds only (the light "paper" ground is already bright -> skipped).
    _transparent = (backdrop or "").strip().lower() == "transparent"
    _bd = None if _transparent else (BACKDROPS.get((backdrop or "").strip().lower()) if backdrop else None)
    try:
        _bg_lift = float(os.environ.get("TYPO_BG_LIGHTEN", "0"))
    except ValueError:
        _bg_lift = 0.0
    _bg_lift = min(max(_bg_lift, 0.0), 1.0)
    _pad_bg = g["bg"]
    _fill = None
    _alpha = None
    _floral_inside = None
    if _floral_key and getattr(an, "silhouette", None) is not None:
        # Floral frame: capture the subject's soft alpha now; the frame itself is composited
        # over the padded canvas below (so the blooms sit on the true edges). Pad with the
        # art's cream so any pad the frame doesn't cover stays seamless.
        _ow = int(out_width)
        _floral_inside = np.clip(cv2.resize(soft01, (_ow, oh), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
        _pad_bg = _FLORAL_CREAM
    elif _transparent and getattr(an, "silhouette", None) is not None:
        # Transparent cutout (digital PNG only): alpha = the soft matte (hair-preserving),
        # so wispy hair feathers into transparency instead of a hard cut. Everything outside
        # -- and the print-canvas padding -- becomes fully transparent, portrait floats free.
        _ow = int(out_width)
        _alpha = np.clip(cv2.resize(soft01, (_ow, oh), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
        _pad_bg = (0.0, 0.0, 0.0)
    elif _bd is not None:
        _fill = np.array(_bd, np.float32)
    elif _bg_lift > 0.0 and ground not in PAPER_FAMILY:
        _bgc = np.array(g["bg"], np.float32)
        _fill = _bgc + (255.0 - _bgc) * _bg_lift
    if _fill is not None and getattr(an, "silhouette", None) is not None:
        _ow = int(out_width)
        # Soft matte edge so the backdrop blends into hair instead of a hard cardboard cut.
        _inside = np.clip(cv2.resize(soft01, (_ow, oh), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
        _outside = (1.0 - _inside)[..., None]
        out = out.astype(np.float32) * (1.0 - _outside) + _fill * _outside
        _pad_bg = tuple(float(c) for c in _fill)
    # Standard print canvas (4:5 = 16x20), padded with the ground BEFORE vibrance
    # so the band is processed identically to the interior ground (no seam).
    from .tonal import _fit_print_canvas
    out = _fit_print_canvas(out, _pad_bg, print_aspect)
    from .preprocess import apply_vibrance
    _vib = float(os.environ.get("TYPO_VIBRANCE", "0.22") or 0.22)   # step-3 colour-fidelity knob (was fixed 0.34)
    out = apply_vibrance(out, strength=_vib, bgr=True)   # gentle life (clarity); restrained so colour stays natural and the sclera isn't glow-brightened
    # Colour fidelity: soft-cap HSV saturation so the oversaturated extremes -- magenta lips,
    # orange-boosted skin highlights -- compress toward a natural ceiling while ordinary skin
    # keeps its colour. Only saturation ABOVE the cap is compressed (35% slope), so nothing
    # below it is touched. TYPO_SAT_CAP=0 disables. Default 170 (gentle).
    _scap = float(os.environ.get("TYPO_SAT_CAP", "150") or 150)
    if _scap > 0:
        _hh = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        _s = _hh[..., 1]
        _hh[..., 1] = np.where(_s > _scap, _scap + (_s - _scap) * 0.35, _s)
        out = cv2.cvtColor(_hh.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if _floral_key and _floral_inside is not None:
        # Floral frame: composite the watercolour frame everywhere OUTSIDE the subject, on the
        # padded canvas (blooms land on the true edges). Pad the subject alpha to the SAME canvas
        # (0 outside) so hair feathers into the frame; a missing art file -> a plain cream mat.
        _fl = _load_floral(_floral_key)
        _hc, _wc = out.shape[:2]
        _fa = _fit_print_canvas(np.repeat(_floral_inside[..., None], 3, axis=2).astype(np.float32),
                                (0.0, 0.0, 0.0), print_aspect)
        _fai = np.clip(_fa[..., 0:1], 0.0, 1.0)
        if _fl is None:
            _fl = np.full((_hc, _wc, 3), _FLORAL_CREAM, np.float32)
        else:
            _fl = cv2.resize(_fl, (_wc, _hc), interpolation=cv2.INTER_AREA).astype(np.float32)
        out = np.clip(out, 0, 255).astype(np.float32) * _fai + _fl * (1.0 - _fai)
    if _transparent and _alpha is not None:
        # Pad the alpha to the SAME canvas as `out` (transparent border), then emit BGRA.
        _a3 = _fit_print_canvas(np.repeat(_alpha[..., None], 3, axis=2).astype(np.float32),
                                (0.0, 0.0, 0.0), print_aspect)
        _alpha_c = np.clip(_a3[..., 0], 0.0, 1.0)
        bgra = np.dstack([np.clip(out, 0, 255).astype(np.uint8),
                          (_alpha_c * 255.0).astype(np.uint8)])
        ok, buf = cv2.imencode(".png", bgra)
    else:
        ok, buf = cv2.imencode(".png", np.clip(out, 0, 255).astype(np.uint8))
    if not ok:
        raise ValueError("encode_failed")
    return buf.tobytes()
