"""Flow placement for the Words (Lifelike) style: TYPO_FLOW_PLACE=1.

The rows renderer draws horizontal rows of words and warps them vertically by the
photograph's luminance, so the rows ride the form. Every word stays horizontal; the form
shows as a displacement of the row. This module sets the words ALONG the form instead:
each word is an upright bitmap rotated to the local direction of a flow field and placed
along traced streamlines, the way the pet engine lays type along fur. Words stay crisp
(they are never warped), and the direction itself carries the form: the hair's own grain,
the contours of the face, orbits around the eyes, the tangent of the silhouette.

The flow field, per pixel, as a direction mod pi and a confidence:
  form      the tangent of the iso-luminance lines of the smoothed photograph (the contour
            lines of the face), confidence by the gradient's strength;
  hair      the multi-scale structure tensor the pet engine uses for fur, trusted on the
            subject above the chin and outside the face;
  reading   a constant pull toward horizontal, so a flat patch reads as rows;
  eyes      the pet engine's attractor swirl around each iris (TYPO_FLOW_EYE, 0 = none).
Combined as double-angle vectors (a line has no direction, only an orientation), which is
also how each source's confidence becomes its weight.

Placement: the pet engine's evenly spaced streamlines (Jobard-Lefer) over a separation
field equal to the local type size times TYPO_FLOW_LEADING, so lines sit one row apart
just as the rows did. The type size is the continuous size the rows' four tiers blend
between, read from the same detail field (large on the body, mid on the broad face, fine at
the features), so the size hierarchy the customer sees is unchanged. Every line is read
left to right; the word stream runs on from line to line, so a Passage keeps its order.
A word whose letters would land on more than a tenth of already placed ink is skipped
(a gap beats a pile-up). Each placement is logged (word, size, angle, centre), and the
sharp print draws the log again at print scale with the font at size x k, exactly as the
pet engine's print path does.

TYPO_FLOW_ZONE=hair (the default) is the hybrid: the rows stay on the face and body and
the flow is placed only in the hair, blended through a feathered hair weight (hair_zone).
Judged on staging 2026-09-20: whole-subject flow lost to the rows on a soft frontal face
(busier, darker, ripples round the highlights) and won only in the hair. "all" is the
whole-subject mode.

Off (the default) nothing here runs and the rows renderer is byte-identical.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import settings as _settings
from .. import typeface as _typeface

# (word, font px, angle deg, centre x, centre y) at the working scale.
Glyph = Tuple[str, int, int, float, float]


def on() -> bool:
    return (_settings.raw("TYPO_FLOW_PLACE") or "").strip().lower() in ("1", "true", "on", "yes")


# The rows renderer blends four tiers by the detail field df: large below 0.45, mid to
# 0.75, fine to 1.0, micro at 1.0. The flow renderer reads one continuous size off the same
# knots, so a region that was a 60/40 blend of two tiers is now one size between them.
_DF_KNOTS = (0.0, 0.45, 0.75, 1.0)
_TIER_PX = (64.0, 40.0, 26.0, 16.0)


def size_field(df: np.ndarray, s: float, ssn: float, wsc: float) -> np.ndarray:
    sizes = np.array(_TIER_PX, np.float32) * float(s * ssn * wsc)
    return np.interp(np.asarray(df, np.float32), np.array(_DF_KNOTS, np.float32), sizes).astype(np.float32)


def hair_zone(mask01: np.ndarray, fmh: np.ndarray, chin_y: float, fw: float, W: int, H: int,
              gray: np.ndarray, bgr: Optional[np.ndarray] = None) -> np.ndarray:
    """Where the flow places words in the hybrid (TYPO_FLOW_ZONE=hair): the subject outside
    the face hull above the chin, and below it only long hair: grain-confident, clear of
    the neck column, joined to the hair above, and of the hair's own colour (a neck crease
    and a woven garment are confident and joined too; the first trials set flow words on
    a neck and a shirt). Feathered by a few pixels so the seam at the hairline is a short
    cross-fade: a wide one doubled the words in the band."""
    from ..pet_v2.engine import multi_scale_orientation
    d = int(fw * 0.03) | 1
    hull8 = cv2.dilate(np.asarray(fmh, np.uint8), np.ones((d, d), np.uint8), 1)
    # The mesh stops short of the hairline; the forehead above it is skin and stays in
    # rows. Grow the hull upward by a fifth of the face height (anchor at the kernel's
    # foot, so only upward).
    ys_h = np.nonzero(hull8.any(axis=1))[0]
    fh = float(ys_h.max() - ys_h.min()) if len(ys_h) else fw
    up = max(1, int(fh * 0.20))
    hull_face = hull8.copy()
    hull8 = cv2.dilate(hull8, np.ones((up, 1), np.uint8), anchor=(0, up - 1), iterations=1)
    hull = hull8.astype(np.float32)
    hull = np.clip(cv2.GaussianBlur(hull, (0, 0), sigmaX=max(1.0, fw * 0.012)), 0, 1)
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    inside = mask01 > 0.5
    above = inside & (yy < chin_y) & (hull < 0.5)
    ys, xs = np.nonzero(np.asarray(fmh) > 0)
    cx = float(xs.mean()) if len(xs) else W / 2.0
    _th, coh = multi_scale_orientation(np.asarray(gray, np.float32), W)
    coh_s = cv2.GaussianBlur(coh, (0, 0), sigmaX=max(2.0, fw * 0.04))
    # Confidence alone cannot tell hair from skin: a forehead's shading is one smooth
    # gradient and measures 0.96, the hair 0.98. Texture can (local contrast: hair 16.6,
    # forehead 7.7, cheek 9.0 on the smile portrait), normalised to this subject's own
    # hair so a soft or a contrasty photograph reads the same. Grain = confidence x texture.
    g32 = np.asarray(gray, np.float32)
    tex = np.sqrt(np.maximum(cv2.GaussianBlur((g32 - cv2.GaussianBlur(g32, (0, 0), sigmaX=max(1.0, fw * 0.01))) ** 2,
                                              (0, 0), sigmaX=max(2.0, fw * 0.03)), 0.0))
    ref_t = float(np.median(tex[above])) if above.any() else float(np.median(tex[inside])) if inside.any() else 1.0
    grain = coh_s * np.clip((tex / max(ref_t, 1e-3) - 0.5) / 0.4, 0, 1)
    # Below the chin the weight is soft (a hard confidence cut drew a horizontal seam
    # through long hair at chin height): the grain's confidence, times the colour test.
    below_w = np.clip((grain - 0.25) / 0.25, 0, 1) * (inside & (yy >= chin_y) & (np.abs(xx - cx) > 0.55 * fw))
    lab = None
    if bgr is not None and above.any():
        # The hair's colour, from the hair above the chin. Chroma (a, b) separates hair
        # from a garment where the full Lab distance did not (dark brown hair and a navy
        # shirt are 37 apart in Lab, 35 of it chroma); lightness gets a wide band so the
        # strand highlights pass.
        lab = cv2.cvtColor(cv2.GaussianBlur(np.asarray(bgr, np.uint8), (0, 0), sigmaX=max(1.0, fw * 0.02)),
                           cv2.COLOR_BGR2LAB).astype(np.float32)
        ref = np.median(lab[above], axis=0)
        chroma = np.sqrt(((lab[..., 1:] - ref[1:]) ** 2).sum(axis=2))
        colour_ok = (chroma < 16.0) & (np.abs(lab[..., 0] - ref[0]) < 50.0)
        below_w = below_w * colour_ok
    # A fringe falling onto the forehead sits inside the grown hull and was set in rows,
    # with rows and flow meeting on the strands. Inside the growth band (and the top of
    # the face hull itself) confident, hair-coloured strands near the hair's own
    # lightness count as hair again. Skin shares brown hair's chroma, so the lightness
    # band and the grain carry the test there.
    fringe_w = None
    if lab is not None:
        top_y = float(ys_h.min()) if len(ys_h) else 0.0
        band = (hull8 > 0) & ((hull_face == 0) | (yy < top_y + 0.15 * fh)) & inside
        fringe_ok = colour_ok & (np.abs(lab[..., 0] - ref[0]) < 25.0) & (chroma < 12.0)
        fringe_w = np.clip((grain - 0.30) / 0.25, 0, 1) * (band & fringe_ok)
        fringe_w = np.clip(cv2.GaussianBlur(fringe_w.astype(np.float32), (0, 0), sigmaX=max(1.0, fw * 0.01)), 0, 1)
    # The mesh sits low on a high forehead, so the band above the grown hull can be
    # skin. Above the hull, smooth pixels of the face's own colour are skin, not hair:
    # measured on the smile portrait, a band of forehead a fifth of the face high took
    # curved words. Hair of a skin-like colour keeps its grain, which is the tie-break.
    above_w = above.astype(np.float32)
    if lab is not None and (hull_face > 0).any():
        skin_ref = np.median(lab[hull_face > 0], axis=0)
        skin_chroma = np.sqrt(((lab[..., 1:] - skin_ref[1:]) ** 2).sum(axis=2))
        skin_like = (skin_chroma < 10.0) & (np.abs(lab[..., 0] - skin_ref[0]) < 22.0)
        smooth = np.clip((0.40 - grain) / 0.15, 0, 1)
        above_w = above_w * (1.0 - smooth * skin_like)
        above_w = np.clip(cv2.GaussianBlur(above_w, (0, 0), sigmaX=max(1.0, fw * 0.01)), 0, 1) * above
    cand = (above_w > 0.3) | (below_w > 0.05) | ((fringe_w > 0.3) if fringe_w is not None else False)
    n, lab_cc = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab_cc[above_w > 0.3])] = True
    keep[0] = False
    zone = np.where(above, above_w, below_w).astype(np.float32) * keep[lab_cc] * (1.0 - hull)
    if fringe_w is not None:
        zone = np.maximum(zone, fringe_w * keep[lab_cc])
    return np.clip(cv2.GaussianBlur(zone, (0, 0), sigmaX=max(1.0, fw * 0.008)), 0, 1).astype(np.float32)


def flow_field(gray: np.ndarray, mask01: np.ndarray, face_norm: np.ndarray, chin_y: float,
               irises: Sequence[Tuple[float, float, float]], fw: float, W: int, H: int,
               amp: np.ndarray, base_px: float, hair: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """The direction (mod pi) and confidence (0..1) the streamlines follow.

    On the face the direction is the draped row's own: a row y = c becomes the curve
    y + F(x, y) = c under the drape (F = amplitude x normalised luminance), whose tangent
    is atan2(-dF/dx, 1 + dF/dy). It is smooth, it never crosses itself, and it carries the
    relief the rows have always had; the words now turn with it instead of shearing. Two
    trials with the photograph's own iso-luminance contours came out as spaghetti: on a
    side-lit face the terminator runs vertically and every highlight is a small island.
    In the hair the multi-scale structure tensor (the pet engine's fur grain) takes over
    where it is confident; at the silhouette the lines turn along the edge; around each
    iris the pet engine's orbit is blended in. `hair` is the hair weight (0..1); None
    derives one from the face feather (the whole-subject mode)."""
    from ..pet_v2.engine import multi_scale_orientation, blend_attractor_field
    g = np.asarray(gray, np.float32)
    D = cv2.GaussianBlur(g, (0, 0), sigmaX=W * 0.020)         # the drape's own smoothing
    F = np.asarray(amp, np.float32) * ((D / 255.0 - 0.5) * 2.0)
    Fy, Fx = np.gradient(F)
    # The fold: where 1 + dF/dy nears zero the drape compresses a row to nothing and the
    # tangent swings vertical; past zero it loops, and a traced line circled every cheek
    # highlight (the ripples seen on staging). A row is bounded by the amplitude; the
    # tangent is floored so a line cannot loop.
    th_drape = np.arctan2(-Fx, np.maximum(0.5, 1.0 + Fy)).astype(np.float32)
    # Hair grain, trusted on the subject above the chin and off the face.
    th_hair, coh_hair = multi_scale_orientation(g, W)
    if hair is None:
        yy = np.arange(H, dtype=np.float32)[:, None]
        hair = ((mask01 > 0.5) & (yy < chin_y)).astype(np.float32) * (1.0 - np.clip(face_norm, 0, 1))
        hair = np.clip(cv2.GaussianBlur(hair, (0, 0), sigmaX=max(2.0, fw * 0.03)), 0, 1)
    hair_k = float(_settings.raw("TYPO_FLOW_HAIR") or 1.0)
    w_hair = np.clip(np.clip(coh_hair * 1.5, 0, 1) ** 2 * hair * hair_k, 0.0, 0.92)
    c2 = np.cos(2 * th_drape) * (1.0 - w_hair) + np.cos(2 * th_hair) * w_hair
    s2 = np.sin(2 * th_drape) * (1.0 - w_hair) + np.sin(2 * th_hair) * w_hair
    # Silhouette: within a row or two of the edge the lines run along it (the level sets
    # of the distance transform are parallel to the edge; their tangent is its tangent).
    edge_rows = float(_settings.raw("TYPO_FLOW_EDGE") or 1.5)
    if edge_rows > 0.0:
        m8 = (mask01 > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
        dist_b = cv2.GaussianBlur(dist, (0, 0), sigmaX=max(1.5, W * 0.006))
        dgy, dgx = np.gradient(dist_b)
        th_edge = np.arctan2(dgy, dgx) + math.pi / 2.0
        w_edge = np.clip(1.0 - dist / max(1.0, base_px * edge_rows), 0, 1) ** 1.5 * (m8 > 0)
        c2 = c2 * (1.0 - w_edge) + np.cos(2 * th_edge) * w_edge
        s2 = s2 * (1.0 - w_edge) + np.sin(2 * th_edge) * w_edge
    # Neighbouring lines must run parallel to read as one sweep: smooth the direction
    # vectors over about a row of type.
    _sm = max(1.0, fw * float(_settings.raw("TYPO_FLOW_SMOOTH") or 0.02))
    c2 = cv2.GaussianBlur(c2.astype(np.float32), (0, 0), sigmaX=_sm)
    s2 = cv2.GaussianBlur(s2.astype(np.float32), (0, 0), sigmaX=_sm)
    theta = (0.5 * np.arctan2(s2, c2)).astype(np.float32)
    coh = np.clip(np.hypot(c2, s2), 0.0, 1.0).astype(np.float32)
    # Eyes: the words orbit each iris, the pet engine's swirl.
    eye_k = float(_settings.raw("TYPO_FLOW_EYE") or 0.6)
    if irises and eye_k > 0.0:
        pts = [(float(cx), float(cy)) for cx, cy, _r in irises]
        radius = float(np.mean([r for _, _, r in irises])) * 2.2
        theta, coh = blend_attractor_field(theta, coh, pts, radius, strength=min(1.0, eye_k))
    return theta.astype(np.float32), coh.astype(np.float32)


class _Bitmaps:
    """Upright word bitmaps ("L", ink = 255) and their rotations, per (word, px, deg)."""

    def __init__(self, font_path: Optional[str], k: float = 1.0):
        self.font_path, self.k = font_path, float(k)
        self.fonts: Dict[int, object] = {}
        self.text: Dict[Tuple[str, int], Image.Image] = {}
        self.rot: Dict[Tuple[str, int, int], np.ndarray] = {}
        self.space: Dict[int, float] = {}
        self.adv: Dict[Tuple[str, int], float] = {}

    def font(self, px: int):
        f = self.fonts.get(px)
        if f is None:
            f = _typeface.load(px * self.k if self.k != 1.0 else px, self.font_path)
            self.fonts[px] = f
        return f

    def space_px(self, px: int) -> float:
        v = self.space.get(px)
        if v is None:
            v = max(1.0, float(self.font(px).getlength(" ")))
            self.space[px] = v
        return v

    def advance(self, word: str, px: int) -> float:
        """The word's own advance, as a row would set it; the bitmap carries 2 px of
        padding each side that must not widen the spacing."""
        key = (word, px)
        v = self.adv.get(key)
        if v is None:
            v = max(1.0, float(self.font(px).getlength(word)))
            self.adv[key] = v
        return v

    def upright(self, word: str, px: int) -> Image.Image:
        key = (word, px)
        im = self.text.get(key)
        if im is None:
            f = self.font(px)
            tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
            bb = tmp.textbbox((0, 0), word, font=f)
            w, h = bb[2] - bb[0] + 4, bb[3] - bb[1] + 4
            im = Image.new("L", (max(1, w), max(1, h)), 0)
            ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), word, font=f, fill=255)
            if len(self.text) > 6000:
                self.text.clear()
            self.text[key] = im
        return im

    def rotated(self, word: str, px: int, deg: int) -> np.ndarray:
        key = (word, px, deg)
        a = self.rot.get(key)
        if a is None:
            im = self.upright(word, px)
            if deg:
                im = im.rotate(-deg, expand=True, resample=Image.BICUBIC)
            a = np.asarray(im, np.float32) / 255.0
            if len(self.rot) > 20000:
                self.rot.clear()
            self.rot[key] = a
        return a


def _paste_max(field: np.ndarray, bmp: np.ndarray, cx: float, cy: float, occupancy_test: bool,
               max_overlap: float) -> bool:
    """Union the bitmap into the field centred at (cx, cy). With occupancy_test, refuse (and
    leave the field alone) when more than max_overlap of its letters land on placed ink."""
    H, W = field.shape
    h, w = bmp.shape
    px, py = int(round(cx - w / 2.0)), int(round(cy - h / 2.0))
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(W, px + w), min(H, py + h)
    if x1 <= x0 or y1 <= y0:
        return False
    sub = bmp[y0 - py:y1 - py, x0 - px:x1 - px]
    roi = field[y0:y1, x0:x1]
    if occupancy_test:
        letters = sub > 0.16
        n = int(letters.sum())
        if n == 0:
            return False
        if int((roi[letters] > 0.16).sum()) > max_overlap * n:
            return False
    np.maximum(roi, sub, out=roi)
    return True


def _upright_deg(angle: float) -> int:
    a = math.atan2(math.sin(angle), math.cos(angle))
    if a > math.pi / 2:
        a -= math.pi
    elif a < -math.pi / 2:
        a += math.pi
    return int(round(math.degrees(a)))


def place(theta: np.ndarray, coh: np.ndarray, size_px: np.ndarray, mask01: np.ndarray,
          stream: Sequence[str], font_path: Optional[str], W: int, H: int,
          zone: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Glyph]]:
    """Trace the streamlines and set the words along them. Returns the ink field (H x W,
    0..1) and the placement log the sharp print redraws."""
    from ..pet_v2.engine import evenly_spaced_streamlines
    leading = float(_settings.raw("TYPO_FLOW_LEADING") or 0.92)
    sep = np.maximum(4.0, size_px * leading).astype(np.float32)
    # Lines run past the silhouette by a couple of rows so the matte, not a missing line,
    # cuts the edge; the rows renderer covered the whole frame for the same reason.
    reach = int(round(float(size_px.max()) * 2.0)) | 1
    where = (mask01 > 0.5) if zone is None else (zone > 0.15)
    region = cv2.dilate(where.astype(np.uint8), np.ones((reach, reach), np.uint8), 1).astype(np.float32)
    if not region.any():
        return np.zeros((H, W), np.float32), []
    # A line never dies for want of confidence: where the field is uncertain it is already
    # near-horizontal, and the rows renderer drew there too. The floor keeps the tracer's
    # own test off; the seed is still the most confident point.
    coh_t = np.maximum(coh, 0.3).astype(np.float32)
    step = max(2.0, 0.22 * float(size_px.min()))
    # The tracer's usual closeness test (a line dies within 0.62 of a row of another) keeps
    # lines long; with a stricter 0.85 the lines broke into fragments a word long (median
    # 136 px) and every junction dropped a word. Where two lines pinch, the placement below
    # shrinks the word a step or two before giving up.
    _tf = float(_settings.raw("TYPO_FLOW_TEST") or 0.62)
    lines = evenly_spaced_streamlines(theta, coh_t, region, sep, step=step, max_steps=20000,
                                      min_coherence=0.0, max_turn=0.10, test_frac=_tf,
                                      max_lines=60000, reseed=True)
    # Gap fill. Where the field diverges (the drape spreads rows on a vertical gradient)
    # two lines drift up to two rows apart before a seed a full row from either fits, and
    # the tracer's own reseed looks only three rows out. The lanes the lines claimed are
    # painted, and a second wave grows in what is left, confined to it (a line dies at the
    # lane's edge); its words still pass the collision test below.
    for _pass in range(int(_settings.raw("TYPO_FLOW_FILL") or 1)):
        lane = np.zeros((H, W), np.uint8)
        for line in lines:
            if len(line) < 2:
                continue
            P = np.asarray([(x, y) for x, y, _a in line], np.int32)
            xi, yi = min(W - 1, max(0, int(P[len(P) // 2, 0]))), min(H - 1, max(0, int(P[len(P) // 2, 1])))
            cv2.polylines(lane, [P.reshape(-1, 1, 2)], False, 1, max(1, int(round(1.7 * float(sep[yi, xi])))))
        gap = region * (lane == 0)
        if float(gap.sum()) < 4.0 * float(sep.mean()) ** 2:
            break
        more = evenly_spaced_streamlines(theta, coh_t, gap, sep, step=step, max_steps=20000,
                                         min_coherence=0.0, max_turn=0.10, test_frac=_tf,
                                         max_lines=60000, reseed=True)
        if not more:
            break
        lines = lines + more
    field = np.zeros((H, W), np.float32)
    bm = _Bitmaps(font_path)
    log: List[Glyph] = []
    n = max(1, len(stream))
    wi = 0
    max_overlap = float(_settings.raw("TYPO_FLOW_OVERLAP") or 0.10)
    shrink = (1.0, 0.85, 0.72)
    n_lines = 0

    def _at(P, seg, cum, d):
        i = min(max(int(np.searchsorted(cum, d, side="right")), 1), len(P) - 1)
        t = 0.0 if seg[i - 1] < 1e-6 else min(1.0, max(0.0, (d - cum[i - 1]) / seg[i - 1]))
        return (float(P[i - 1, 0] + (P[i, 0] - P[i - 1, 0]) * t),
                float(P[i - 1, 1] + (P[i, 1] - P[i - 1, 1]) * t),
                math.atan2(P[i, 1] - P[i - 1, 1], P[i, 0] - P[i - 1, 0]))

    for line in lines:
        if len(line) < 2:
            continue
        P = np.asarray([(x, y) for x, y, _a in line], np.float32)
        if P[-1, 0] < P[0, 0]:          # read left to right
            P = P[::-1]
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 1.0:
            continue
        n_lines += 1
        d = 0.0
        while d < total:
            x, y, _ang = _at(P, seg, cum, d)
            xi, yi = min(W - 1, max(0, int(round(x)))), min(H - 1, max(0, int(round(y))))
            px0 = float(size_px[yi, xi])
            word = stream[wi % n]
            placed = False
            for sh in shrink:
                px = max(6, int(round(px0 * sh)))
                wide = bm.advance(word, px)
                # The word's centre sits half its width along the line from where it
                # starts, and its angle is the line's there. A word that would overhang
                # the line's end by more than a third of itself is left for the next line.
                if d + wide * 0.67 > total:
                    break
                cx, cy, ang = _at(P, seg, cum, d + wide / 2.0)
                deg = _upright_deg(ang)
                if _paste_max(field, bm.rotated(word, px, deg), cx, cy, True, max_overlap):
                    log.append((word, px, deg, cx, cy))
                    placed = True
                    break
            if d + bm.advance(word, max(6, int(round(px0 * shrink[-1])))) * 0.67 > total:
                break                   # the end of the line
            wi += 1
            if not placed:
                d += bm.space_px(max(6, int(round(px0)))) * 2.0   # a pinch: step on and try the next word
                continue
            d += wide + bm.space_px(px)
    if (_settings.raw("TYPO_FLOW_DEBUG") or "").strip():
        inside = (mask01 > 0.5) if zone is None else (zone > 0.5)
        _ln = [float(np.hypot(np.diff([p[0] for p in l]), np.diff([p[1] for p in l])).sum()) for l in lines if len(l) > 1]
        print("[flow] lines=%d (median %.0f px) words=%d coverage(subject)=%.3f step=%.1f sep=%.0f-%.0f"
              % (n_lines, float(np.median(_ln)) if _ln else 0.0, len(log),
                 float(field[inside].mean()) if inside.any() else 0.0,
                 step, float(sep.min()), float(sep.max())), flush=True)
    return field, log


def typed_iris_on() -> bool:
    return (_settings.raw("TYPO_TYPED_IRIS") or "").strip().lower() in ("1", "true", "on", "yes")


def _set_along(field, log, bm, P, words, px, max_overlap, fit_len=None):
    """Set words along the polyline P (N x 2) at size px, left to right along the path.
    With fit_len the FIRST word of the stream whose advance fits that length is set once
    (a spoke holds one word); otherwise the words run on along the path."""
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1.0:
        return 0

    def _at(d):
        i = min(max(int(np.searchsorted(cum, d, side="right")), 1), len(P) - 1)
        t = 0.0 if seg[i - 1] < 1e-6 else min(1.0, max(0.0, (d - cum[i - 1]) / seg[i - 1]))
        return (float(P[i - 1, 0] + (P[i, 0] - P[i - 1, 0]) * t),
                float(P[i - 1, 1] + (P[i, 1] - P[i - 1, 1]) * t),
                math.atan2(P[i, 1] - P[i - 1, 1], P[i, 0] - P[i - 1, 0]))

    n_set = 0
    if fit_len is not None:
        for _k in range(len(words)):
            word = words[(words.i + _k) % len(words)]
            wide = bm.advance(word, px)
            if wide <= fit_len:
                cx, cy, ang = _at(total / 2.0)
                deg = _upright_deg(ang)
                if _paste_max(field, bm.rotated(word, px, deg), cx, cy, True, max_overlap):
                    log.append((word, px, deg, cx, cy))
                    n_set += 1
                words.i += _k + 1
                return n_set
        # No word fits (a small iris): the next word's initial, still typography.
        word = words[words.i % len(words)][:1] or "I"
        words.i += 1
        cx, cy, ang = _at(total / 2.0)
        deg = _upright_deg(ang)
        if _paste_max(field, bm.rotated(word, px, deg), cx, cy, True, max_overlap):
            log.append((word, px, deg, cx, cy))
            n_set += 1
        return n_set
    d = 0.0
    while d < total:
        word = words[words.i % len(words)]
        words.i += 1
        wide = bm.advance(word, px)
        if d + wide * 0.6 > total:
            break
        cx, cy, ang = _at(d + wide / 2.0)
        deg = _upright_deg(ang)
        if _paste_max(field, bm.rotated(word, px, deg), cx, cy, True, max_overlap):
            log.append((word, px, deg, cx, cy))
            n_set += 1
        d += wide + bm.space_px(px)
    return n_set


class _Stream(list):
    """The word stream with a cursor, so the rings and spokes read on in order."""
    i = 0


def typed_iris(irises: Sequence[Tuple[float, float, float]], stream: Sequence[str],
               font_path: Optional[str], W: int, H: int) -> Tuple[np.ndarray, List[Glyph]]:
    """The iris built from type (TYPO_TYPED_IRIS=1): at each iris a ring of words running
    round the limbus (the rim), and short words set radially from the pupil's edge to the
    rim like the fibres of a real iris; the pupil is left clear for the dark disc the
    feature pass paints, the glint sits on top. Returns the glyph field and the log the
    sharp print redraws at print scale. Sizes follow the iris radius; the smallest word
    is 6 px, so on a small iris the fibres are single short words and the rim one ring."""
    field = np.zeros((H, W), np.float32)
    log: List[Glyph] = []
    bm = _Bitmaps(font_path)
    words = _Stream(list(stream) or ["LOVE"])
    ring_k = float(_settings.raw("TYPO_IRIS_RING") or 0.20)
    spoke_k = float(_settings.raw("TYPO_IRIS_SPOKE") or 0.18)
    for cx, cy, r in irises:
        r = float(r)
        # The rim: one ring of words at 0.87 r, read clockwise from the top left.
        px = max(6, int(round(r * ring_k)))
        rr = r * 0.87
        ang = np.linspace(-math.pi * 0.75, math.pi * 1.25, 180)
        P = np.stack([cx + rr * np.cos(ang), cy + rr * np.sin(ang)], axis=1).astype(np.float32)
        _set_along(field, log, bm, P, words, px, 0.10)
        # A large iris (the ring's words above 9 px) takes a second ring inside the first.
        px_s = max(6, int(round(r * spoke_k)))
        if px >= 9:
            rr2 = r * 0.66
            P = np.stack([cx + rr2 * np.cos(ang), cy + rr2 * np.sin(ang)], axis=1).astype(np.float32)
            _set_along(field, log, bm, P, words, px_s, 0.10)
            r0, r1 = r * 0.44, r * 0.56
        else:
            r0, r1 = r * 0.44, r * 0.78
        # The fibres: spokes from the pupil's edge to inside the rim, one short word each
        # (or an initial where none fits), spaced a word's height apart on the middle circle.
        n = max(6, int(round(2 * math.pi * (0.5 * (r0 + r1)) / (px_s * 1.05))))
        for j in range(n):
            a = -math.pi / 2 + 2 * math.pi * j / n
            # Read outward on the right half, inward on the left, so every word is upright.
            if math.cos(a) >= 0:
                P = np.array([[cx + r0 * math.cos(a), cy + r0 * math.sin(a)],
                              [cx + r1 * math.cos(a), cy + r1 * math.sin(a)]], np.float32)
            else:
                P = np.array([[cx + r1 * math.cos(a), cy + r1 * math.sin(a)],
                              [cx + r0 * math.cos(a), cy + r0 * math.sin(a)]], np.float32)
            _set_along(field, log, bm, P, words, px_s, 0.10, fit_len=(r1 - r0) * 1.05)
    return field, log


def raster(log: Sequence[Glyph], Wk: int, Hk: int, k: float, font_path: Optional[str]) -> np.ndarray:
    """The logged words drawn again at scale k on a Wk x Hk field: the font at px * k, the
    same angle, the centre scaled. At k = 1 this is the working field (up to the collision
    test, which the log already passed)."""
    field = np.zeros((Hk, Wk), np.float32)
    bm = _Bitmaps(font_path, k)
    for word, px, deg, cx, cy in log:
        _paste_max(field, bm.rotated(word, px, deg), cx * k, cy * k, False, 1.0)
    return field
