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
    # "match your space" palette: the SAME ink-drawing renderer as paper (correct
    # polarity, typographic eyes/teeth), only the ground colour changes so buyers can
    # match a room's wall. All are LIGHT grounds (tone "dark" = dark ink) -- a dark
    # ground would flip the compositor back to photographic eyes and lose the look.
    "sand":  {"bg": (208, 228, 238), "ink": (58, 33, 20), "tone": "dark"},   # warm sand / oat
    "slate": {"bg": (226, 221, 216), "ink": (44, 38, 34),  "tone": "dark"},   # soft cool grey
    "navy":  {"bg": (58, 27, 13),    "ink": (248, 248, 248), "tone": "light"},  # white on navy (hero)
    "black": {"bg": (14, 14, 14),    "ink": (248, 248, 248), "tone": "light"},  # white on black
}

# Grounds that share paper's ink-drawing treatment (density polarity, edge-drawn hair,
# words-form-the-eye). They differ ONLY in ground colour -- the "match your space" swatches.
PAPER_FAMILY = frozenset({"paper", "sand", "slate"})

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


def _normalize_words(words: Sequence[str], uppercase: bool = True) -> List[str]:
    out: List[str] = []
    for w in words:
        s = str(w).upper() if uppercase else str(w)
        t = "".join(ch for ch in s if ch.isalnum() or ch in "-'")
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
    sunglasses: bool = False,     # MANUAL sunglasses control. Pixels cannot reliably tell a
                                  # tinted/reflective lens from a real eye (a dark sclera and a
                                  # mirror lens look identical), so opacity is caller-driven:
                                  # False (default) => NO face is ever dark-lensed, real eyes are
                                  # never blacked out; True => every detected face's eye region is
                                  # rendered as an opaque lens.
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

    g = GROUNDS.get(ground, GROUNDS["navy"])
    rng = random.Random(seed)
    vocab = _normalize_words(words, uppercase)
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
    _dl_on = bool(sunglasses) and os.environ.get("TYPO_DARKLENS", "1").strip().lower() not in ("0", "false", "off", "no", "")
    irises: List[Tuple[float, float, float]] = []
    eye_centers: List[Tuple[float, float, float]] = []
    _eye_face_pts = []            # faces whose eyes ARE rendered (drive the sclera/anchor hulls)
    _dark_lens_active = False
    _dark_lens_eyes = []
    _dark_lens_face_pts = []      # shaded faces (drive the opaque-lens fill)
    _misfit_face_pts = []         # meshes fit so badly the eyes can't be trusted (skip entirely)
    for _fp in all_pts:
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
        if len(_gaps) == 2 and all(g < -0.06 * _fh for g in _gaps):
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
            if ir >= 8.0 * _ssn:
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
        # MANUAL sunglasses: the caller flagged this render as having tinted lenses. Pixels
        # can't reliably tell a lens from a real eye, so honour the flag -- render this face's
        # eye region as an opaque lens. Off (default) => this never runs, so a real eye can
        # never be blacked out by a mis-detection.
        if _dl_on:
            _dark_lens_active = True
            _dark_lens_eyes.extend(_fi)
            _dark_lens_face_pts.append(_fp)
            continue
        # Dark-pupil backstop: a real open eye has a dark pupil in a lighter iris. If the
        # centre isn't darker than the surround the mesh isn't sitting on an eye -- render
        # plain words there rather than fabricate an eyeball.
        ratios = []
        for icx, icy, ir in _fi:
            y0, y1 = max(0, int(icy - ir * 2.4)), int(icy + ir * 2.4)
            x0, x1 = max(0, int(icx - ir * 2.4)), int(icx + ir * 2.4)
            reg = gray[y0:y1, x0:x1]
            if reg.size < 16:
                continue
            inner = np.zeros((H, W), np.uint8)
            cv2.circle(inner, (int(round(icx)), int(round(icy))), max(1, int(ir * 0.45)), 1, -1)
            sclera = max(float(np.percentile(reg, 90)), 1.0)
            ratios.append(float(np.percentile(gray[inner > 0], 10)) / sclera)
        if len(ratios) >= 2 and min(ratios) > _EYE_OPEN_IRIS_MAX:
            continue                                   # no dark pupil -> not a real eye
        irises.extend(_fi)
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
            # repeats like a refrain across the face. A gentle, deterministic per-row
            # indent (no random jitter) breaks vertical seams while staying legible.
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
                d.text((-(ry % 5) * fs * 0.5, y), " ".join(parts), font=f, fill=0)
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
            # Single subject: unchanged (protects the memorial single-portrait product).
            _chin_y = max(float(_p[:, 1].max()) for _p in all_pts)
            _below = np.clip((_gyv - _chin_y) / max(1.0, (_bottom_y - _chin_y)), 0.0, 1.0)
            df = np.clip(df + 0.30 * _notface * (_gyv > _chin_y).astype(np.float32) + 0.55 * _below * _notface, 0, 1)
            _face_top_y = min(float(_p[:, 1].min()) for _p in all_pts)
            _above = np.clip((_face_top_y - _gyv) / max(1.0, (_face_top_y - _top_y)), 0.0, 1.0)
            df = np.clip(df + 0.30 * _notface * (_gyv < _face_top_y).astype(np.float32) + 0.55 * _above * _notface, 0, 1)
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
    a = a * np.clip(cv2.GaussianBlur(mask01, (0, 0), sigmaX=W * 0.007), 0, 1)   # feathered edge
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
    if not irises:
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
        _lens = np.zeros((H, W), np.uint8)
        for _fp in _dark_lens_face_pts:
            for k in ("Leye", "Reye"):
                p = np.array([_fp[i] for i in _GROUPS[k] if i < len(_fp)], np.int32)
                if len(p) >= 3:
                    cv2.fillConvexPoly(_lens, cv2.convexHull(p), 1)
        _dk = max(3, int(fw * 0.075)) | 1
        _lensf = cv2.dilate(_lens, np.ones((_dk, _dk), np.uint8), 1).astype(np.float32)
        _lensf = np.clip(cv2.GaussianBlur(_lensf, (0, 0), sigmaX=max(1.0, fw * 0.010)), 0, 1)
        a = a * (1.0 - 0.97 * _lensf)

    # Teeth carry NO typography. Where the mouth is open, suppress ink across the
    # inner mouth (both tones); on a dark ground the cleared teeth get a soft
    # light wash below, on light paper the paper already reads as teeth. A closed
    # mouth yields no mask and is left untouched.
    from .tonal import _teeth_mask
    teeth = None                       # union every subject's open-mouth teeth region
    for _fp in all_pts:
        _tm = _teeth_mask(_fp, H, W)
        if _tm is not None:
            teeth = _tm if teeth is None else np.maximum(teeth, _tm)
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
    if ink == "photo":
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
            hsv[..., 2] = np.clip(hsv[..., 2] * 1.14 + 14, 0, 255)      # lift value vs dark ground
        ink_col = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        out = np.array(g["bg"], np.float32) * (1 - al) + ink_col * al
        if os.environ.get("TYPO_POLARITY", "0").strip().lower() in ("1", "true", "on", "yes"):
            # PROTOTYPE -- polarity model. Instead of "light ink whose COVERAGE follows brightness"
            # (shadow -> no ink -> ground shows -> absence), make the type present at HIGH coverage
            # everywhere and carry the tone in the LETTER COLOUR across the full range: near-black
            # letters in deep shadow (a heavy dark mass to lean into), light letters in highlight.
            _mkf = np.clip(cv2.GaussianBlur(mask01, (0, 0), sigmaX=W * 0.007), 0, 1)
            _tone = np.clip(lum, 0.0, 1.0)[..., None] ** 1.35     # gamma>1 -> deep shadow drives to near-black
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
            _bg_local = np.array(g["bg"], np.float32) * np.clip(0.18 + 0.86 * _tone, 0.0, 1.0)
            out = _bg_local * (1 - _cov[..., None]) + np.clip(_ink, 0, 255) * _cov[..., None]
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
    if irises and g["tone"] == "light":
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
        if ink != "photo":
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
    # Background lightening (Sculpt): lift the region OUTSIDE the silhouette toward
    # white so the sculpted subject (on its dark ground) sits against a lighter
    # backdrop. The subject's own ground is untouched -- only pixels outside the
    # silhouette move. Env TYPO_BG_LIGHTEN (0..1; 0 = off). Dark grounds only; the
    # light "paper" ground is skipped (it's already bright).
    try:
        _bg_lift = float(os.environ.get("TYPO_BG_LIGHTEN", "0"))
    except ValueError:
        _bg_lift = 0.0
    _bg_lift = min(max(_bg_lift, 0.0), 1.0)
    _pad_bg = g["bg"]
    if _bg_lift > 0.0 and ground not in PAPER_FAMILY:
        _ow = int(out_width)
        _sil = cv2.resize((an.silhouette.mask > 127).astype(np.uint8) * 255, (_ow, oh),
                          interpolation=cv2.INTER_NEAREST)
        _bgc = np.array(g["bg"], np.float32)
        _bg_light = _bgc + (255.0 - _bgc) * _bg_lift
        _outside = cv2.GaussianBlur((_sil <= 127).astype(np.float32), (0, 0),
                                    sigmaX=max(1.0, _ow * 0.002))[..., None]
        out = out.astype(np.float32) * (1.0 - _outside) + _bg_light * _outside
        _pad_bg = tuple(float(c) for c in _bg_light)
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
    ok, buf = cv2.imencode(".png", np.clip(out, 0, 255).astype(np.uint8))
    if not ok:
        raise ValueError("encode_failed")
    return buf.tobytes()
