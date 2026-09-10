#!/usr/bin/env python3
"""
Prototype #4: proper EVENLY-SPACED STREAMLINE PLACEMENT (Jobard & Lefer's classic algorithm
for "fill a vector field with flow lines, no gaps, no crowding"), replacing the grid-seeding +
after-the-fact gap patching from glyphs_on_path_prototype.py.

Why the grid approach hit a wall: seeding on a fixed grid independent of the flow field doesn't
adapt to how much the field curves -- straight regions got seeds close enough together to
overlap (had to fight this with separation radii and multi-pass density tuning), while
sharply-curving regions left irregular gaps no amount of grid-density tuning closed cleanly. The
"guaranteed coverage" patch pass then had to cover whatever was left, and depending on how much
that was, it either stayed too sparse or became an illegible fine mesh -- there was no single
grid density that avoided both failure modes at once.

Jobard-Lefer's fix: don't seed independently of the lines you've already placed. GROW new
streamlines directly off the sides of existing ones, at a controlled perpendicular offset (the
target separation `sep_px`). This means spacing is uniform by construction -- tight where the
field is calm and a line can go far, naturally denser where it curves a lot and lines end sooner
-- rather than hoping a fixed grid happens to match the field's actual geometry.

Algorithm:
  1. Seed ONE streamline from the most trustworthy (highest-coherence) point in the mask, traced
     in BOTH directions from that point (not just forward) for a natural, complete first line.
  2. Walk that line at `sep_px` intervals; at each interval, propose two candidate seeds offset
     PERPENDICULAR to the line's local tangent, one on each side, by `sep_px`.
  3. A candidate is accepted only if it isn't already within `sep_px` of any EXISTING line
     (checked via a stamped occupancy grid, not recomputed geometrically -- fast enough for this
     resolution). Accepted candidates are traced bidirectionally and added to the pool; their own
     perpendiculars are enqueued the same way.
  4. A streamline in progress terminates early if it comes within a TIGHTER threshold
     (`test_px < sep_px`) of ANY other line -- so lines can approach each other more closely than
     a fresh seed is allowed to start, avoiding needlessly truncating a line that's just grazing
     a neighbor.
  5. Repeat until the queue of candidates is exhausted. The result is a field-adaptive tiling of
     streamlines with no large gaps and no pile-ups, by construction rather than by tuning.

Word/size assignment is a SEPARATE step after the geometry is decided: the few longest, highest-
coherence lines become HERO carriers (the name, biggest); the rest get the phrase stream at a
size that varies continuously with that line's own average coherence.

Usage:
    python3 glyphs_on_path_v2.py <image_path> [output_path]
"""
import os
import threading
import sys
import math
import random


import numpy as np
import cv2
from PIL import Image, ImageFont

from ..pet_proto import _foreground_mask, _weighted_stream, _phrases, _enhance_contrast, _edge_ink, _FONT
from .fur_flow import orientation_field
from .glyph_paths import (
    resolve_angle, clamp_turn, path_length, sample_path_at, render_word_bitmap,
)

DEFAULT_WORDS = ("LOYAL, GENTLE, SOUL, PLAYFUL, SWEET, KIND, HOME, JOY, WARM, CURIOUS, "
                 "FAITHFUL, BRAVE, WISE, FUNNY, CUDDLY, DEVOTED, PRECIOUS, BELOVED, "
                 "COMPANION, ADVENTUROUS, MISCHIEF, TENDER, SPIRITED, TRUE FRIEND, "
                 "FAMILY, GOOFY, SNUGGLES, BEST FRIEND, TREASURE, HAPPY")


class _RenderState(threading.local):
    """Per-render, per-THREAD state. This used to be a set of module globals (placements,
    stats, nose hint, ...). The service renders concurrently (RENDER_CONCURRENCY > 1): a
    print-quality pass still running while a preview re-render started for a background-color
    change had both threads appending to one placement log, and placement_report read the
    list twice, 6 words apart -- "boolean index did not match indexed array ... 8767 vs
    8761", surfaced to the customer as an alert. Every field here is now private to the
    thread doing the render; nothing about a render is shared except the bitmap cache."""

    def __init__(self):
        self.verbose = False
        # Every word actually placed this run: (x, y, font_px, chars, angle, bmp_w, bmp_h).
        # Reset per iteration; reported per anatomical zone at the end so "is this area
        # rendered with typography" is a count and a size, not an impression.
        self.placements = []
        # Parallel to placements: which pass placed each word ("feature", "struct", "hero",
        # "fill", "residual", "channel"). Lets the size hierarchy be measured per pass -- the
        # fills outnumber the structural words several to one, so a count-median hides what
        # the structural pass did.
        self.pass_tags = []
        self.pass_name = "feature"
        self.stats = {"glyph_px": 0, "overlap_px": 0, "core_px": 0, "core_overlap_px": 0}
        # Default 0.08: every claim metric was measured at this cap (letter-body collisions
        # ~1%). The first staging run shipped with 1.0 (no cap) and reported 6.71% -- the
        # callers' own tolerances (up to 0.24 in gap-fill) are placement heuristics, not a
        # collision policy.
        self.max_overlap_cap = float(os.environ.get("GOP_MAX_OVERLAP", "0.08") or 0.08)
        # Set from landmarks/GOP_LANDMARKS in render_v2; a real nose detection beats the scan.
        self.nose_hint = None
        self.fields = {}   # last render's size-rule fields (PET_V2_KEEP_FIELDS=1), measurement only


_TL = _RenderState()


def __getattr__(name):
    # Measurement scripts read the placement log / fields after a render under the old global
    # names; keep them working, resolved against the calling thread's state.
    _alias = {"_PLACEMENTS": "placements", "_PLACEMENT_PASS": "pass_tags", "_FIELDS": "fields",
              "_STATS": "stats", "_MAX_OVERLAP_CAP": "max_overlap_cap", "_NOSE_HINT": "nose_hint",
              "_VERBOSE": "verbose"}
    if name in _alias:
        return getattr(_TL, _alias[name])
    raise AttributeError(name)



def _gblur(src, ksize, sigmaX=0, **kwargs):
    """cv2.GaussianBlur with the same call shape, but large sigmas run on a downsampled copy.
    Profiled at the 1600px preview: 84 blur calls = 26 s of 88, dominated by the wide-field
    ones (sigma up to W*0.25 ~ 300px -> a 1800-tap kernel per axis). A Gaussian that wide
    is a low-pass; computing it at 1/f scale with sigma/f and resampling back is the same
    field to well below the quantization of anything that consumes it."""
    if not sigmaX or sigmaX < 12:
        return cv2.GaussianBlur(src, ksize, sigmaX=sigmaX, **kwargs)
    f = max(2, int(sigmaX / 6.0))
    H, W = src.shape[:2]
    sw, sh = max(4, W // f), max(4, H // f)
    small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigmaX / f, **kwargs)
    out = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return out.reshape(src.shape) if out.shape != src.shape else out


def _log(*args, **kwargs):
    if _TL.verbose:
        print(*args, **kwargs)


def upright_angle(angle):
    """A traced streamline's direction is only meaningful mod pi (fur has no inherent 'up' --
    resolve_angle elsewhere already treats it that way when following the field), so `angle`
    coming out of sample_path_at can just as easily point leftward as rightward. Rotating a
    word by a leftward angle draws it upside down -- readable letters need the SMALLER of
    {angle, angle + pi} in magnitude from horizontal, not whichever the trace happened to be
    going. This picks that one, so no word ever renders upside down."""
    a = math.atan2(math.sin(angle), math.cos(angle))   # normalize to (-pi, pi]
    if a > math.pi / 2:
        a -= math.pi
    elif a < -math.pi / 2:
        a += math.pi
    return a


def _bucket_words_by_length(words):
    """Split a word list into short/medium/long buckets by character count, for curvature-
    adaptive placement -- a tight curl needs a word that can physically turn with it, and
    PLAYFUL simply cannot bend through a doodle's curl the way LOYAL or SOUL can. Falls back to
    whatever's available if a bucket would otherwise be empty (a 3-word vocabulary shouldn't
    crash this), so it degrades gracefully rather than requiring a minimum vocabulary size."""
    short = [w for w in words if len(w) <= 5]
    medium = [w for w in words if 5 < len(w) <= 9]
    long_ = [w for w in words if len(w) > 9]
    if not short:
        short = medium or long_ or words
    if not medium:
        medium = short or long_ or words
    if not long_:
        long_ = medium or short or words
    return short, medium, long_


def words_for_curvature(stream, t_coh):
    """Pick the word bucket for an ENTIRE line, once, from its already-computed mean coherence
    (t_coh, 0=incoherent/curly, 1=coherent/calm) -- not per placement step. An earlier per-step
    version (sampling local curvature every few pixels while walking the path) was tried first and
    made things WORSE, not better: the fixed lookahead needed calibrating against this pipeline's
    own traced curvature distribution (median ~0.0158, capped near 0.033 by the tracer's own angle
    quantization) and even after fixing the thresholds, choosing a word's LENGTH based on the
    curvature immediately ahead of it created a feedback loop -- picking a long word consumes a
    long stretch of arc-length in one step, so the next curvature check often lands past a bend
    the word itself was drawn straight through, and a small long-word bucket (few phrases are
    genuinely long) got replayed so often on calm stretches that ONE word visually dominated whole
    photos (confirmed directly: "ADVENTUROUS" swallowed most of a render). Deciding once per line
    from t_coh -- the same signal already used to grade this line's font size -- avoids all of
    that: it's the same granularity the rest of this pipeline already trusts for size gradation,
    just applied to word length too."""
    short_words, medium_words, long_words = _bucket_words_by_length(stream)
    if t_coh < 0.35:
        return short_words
    elif t_coh < 0.65:
        return medium_words
    return long_words


def place_words_collision_aware(canvas, occupancy, pts, words, font, gap_px, alpha, max_overlap=0.15,
                                max_instances=None, size_at=None, get_font=None, gap_frac=None):
    """Same idea as place_words_along_path (v1), but checks each word's footprint against a
    SHARED occupancy map before committing it -- the real fix for overlapping letters, which
    streamline-level separation alone can't guarantee once font sizes vary (a big hero word can
    still reach into a neighboring lane even when the lane centerlines are properly spaced).
    A word whose own opaque pixels would land more than `max_overlap` on already-placed ink is
    skipped outright rather than composited anyway -- a visible gap there beats overlapping.
    Returns the number of ink pixels actually composited, so callers can track type-scale
    proportions (recommendation #6's micro/structural/hero share targets) without a second pass.

    max_instances: cap how many times a word gets placed along this ONE path before stopping.
    Needed for hero words specifically -- hero lines are picked for being LONG, and without this
    cap the same "LOYAL" repeats every gap_px along the entire line, producing a whole cluster of
    large words stacked together (confirmed directly: this is what the user's circled "too
    large" complaint was pointing at) instead of one deliberate emphasis word.

    Curvature-appropriate word length is now the CALLER's job (see words_for_curvature) -- pass
    in an already-length-appropriate `words` list for this line rather than the whole vocabulary.

    size_at(x, y) -> font px: when given (with get_font and gap_frac), every word is sized at ITS
    OWN position instead of one font for the whole path. One size per streamline averaged the
    size field along the line's whole length -- on a tight head-and-shoulders crop, where a
    single line runs from the cheek to the frame edge, every line averaged to the same middle
    value and the size hierarchy vanished (measured: 6px median in every distance band)."""
    total = path_length(pts)
    d = 0.0
    wi = 0
    H, W = occupancy.shape
    placed_px = 0
    placed_count = 0
    # Global collision ceiling (GOP_MAX_OVERLAP): every caller's tolerance is clamped to it, so
    # "no two words collide" is a property of the whole render that one number can attest to.
    max_overlap = min(max_overlap, _TL.max_overlap_cap)
    while d < total:
        if max_instances is not None and placed_count >= max_instances:
            break
        word = words[wi % len(words)]
        wi += 1
        samp = sample_path_at(pts, d)
        if samp is None:
            break
        x, y, angle = samp
        angle = upright_angle(angle)
        if size_at is not None:
            _fpx = float(size_at(x, y))
            font = get_font(_fpx)
            gap_px = _fpx * gap_frac
        # Rasterize + rotate once per (word, size, alpha, angle bin) and reuse. Profiled: 91,733
        # render/measure/rotate calls to place ~20,000 words -- 30 s of a 100 s render -- for a
        # vocabulary of 30 words at a few dozen sizes. 1-degree angle bins are invisible at
        # these sizes; the collision test still runs on the real footprint every time.
        # Quantized key: alpha is a continuous per-line value and sizes are jittered, so exact
        # keys never repeated (profiled: 93,696 renders with the cache in place). 16 alpha
        # levels and 3-degree bins are below what's visible at these sizes.
        key = (word, int(round(getattr(font, "size", 0))), (int(alpha) // 16) * 16,
               int(round(math.degrees(angle) / 3.0)) * 3)
        hit = _BITMAP_CACHE.get(key)
        if hit is None:
            bmp = render_word_bitmap(word, font, alpha=alpha)
            rot = bmp.rotate(-math.degrees(angle), expand=True, resample=Image.BICUBIC)
            hit = (bmp.width, bmp.height, rot, np.asarray(rot.split()[3], np.float32))
            if len(_BITMAP_CACHE) > 20000:
                _BITMAP_CACHE.clear()
            _BITMAP_CACHE[key] = hit
        bmp_w, bmp_h, rot, rot_alpha = hit
        d += bmp_w + gap_px
        px, py = int(round(x - rot.width / 2)), int(round(y - rot.height / 2))
        x0, y0 = max(0, px), max(0, py)
        x1, y1 = min(W, px + rot.width), min(H, py + rot.height)
        if x1 <= x0 or y1 <= y0:
            continue
        sub_alpha = rot_alpha[y0 - py:y1 - py, x0 - px:x1 - px]
        occ_roi = occupancy[y0:y1, x0:x1]
        glyph_mask = sub_alpha > 40
        if not glyph_mask.any():
            continue
        overlap_px = int((occ_roi[glyph_mask] > 40).sum())
        overlap_frac = overlap_px / float(glyph_mask.sum())
        if overlap_frac > max_overlap:
            continue   # would overlap too much -- leave a gap here rather than pile up
        # Stats are taken BEFORE the occupancy update: occ_roi is a view into occupancy, so
        # anything read after the np.maximum below includes this very word (measured 100%
        # "self-collision" before this was moved).
        core = sub_alpha > 128                 # letter bodies only, not antialiased halos
        core_overlap = int((occ_roi[core] > 128).sum())
        canvas.alpha_composite(rot, (px, py))
        occupancy[y0:y1, x0:x1] = np.maximum(occ_roi, sub_alpha)
        placed_px += int(glyph_mask.sum())
        placed_count += 1
        _TL.placements.append((x, y, float(getattr(font, "size", 0)), len(word), angle, bmp_w, bmp_h))
        _TL.pass_tags.append(_TL.pass_name)
        _TL.stats["glyph_px"] += int(glyph_mask.sum())
        _TL.stats["overlap_px"] += overlap_px     # this word's stroke pixels landing on prior ink
        _TL.stats["core_px"] += int(core.sum())
        _TL.stats["core_overlap_px"] += core_overlap
    return placed_px


_KEEP_FIELDS = os.environ.get("PET_V2_KEEP_FIELDS", "").strip().lower() not in ("", "0", "false", "off")
# (word, font px, alpha, angle deg) -> (w, h, rotated RGBA, its alpha array). Shared across
# threads on purpose: entries are immutable once built, and dict get/set are atomic in CPython.
_BITMAP_CACHE = {}


def render_channel_fill(canvas, occupancy, theta_s, mask, get_font, tone=None,
                        letters=("L", "S", "K", "J", "H", "G", "P", "W"), min_px=6, max_overlap=0.08,
                        size_cap=None):
    """Fill the free-space CHANNELS (the leading between lines of text, which at 2x is a lace of
    4-10px-wide corridors across the whole coat) along their own medial axis. The streamline
    tracer can't do this -- measured: 1 lane in a 107k-px lace region, because a path following
    the fur direction leaves a thin corridor within a few pixels and is discarded as too short.
    Instead: skeletonize the free space, and at every skeleton point whose local channel width
    (2 x distance-to-ink) fits a glyph, place one letter oriented along the local direction --
    widest channels first, the collision check deciding fit. Returns (letters placed, ink px)."""
    H, W = mask.shape
    ink = (occupancy > 40)
    free = ((mask > 0.5) & ~ink).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    skel = cv2.ximgproc.thinning(free * 255) > 0
    need = min_px * 1.3                                  # channel must be ~1.3 glyph heights wide
    ys, xs = np.nonzero(skel & (dist * 2.0 >= need))
    if len(ys) == 0:
        return 0, 0
    order = np.argsort(-dist[ys, xs])
    font = get_font(min_px)
    placed_px, placed_n = 0, 0
    half = min_px * 0.9
    for k, idx in enumerate(order):
        y, x = int(ys[idx]), int(xs[idx])
        if occupancy[y, x] > 40:
            continue
        # A letter takes the width of the corridor it sits in, up to the local structural size
        # (size_cap): a fixed 6px letter in a 20px corridor on the far body was a speck of lace
        # where the hierarchy asked for a real mark, and it outnumbered the structural words.
        if size_cap is not None:
            _px = float(np.clip(dist[y, x] * 2.0 / 1.3, min_px, max(min_px, float(size_cap[y, x]))))
            font = get_font(_px)
            half = _px * 0.9
        ang = float(theta_s[y, x])
        dx, dy = math.cos(ang), math.sin(ang)
        path = [(x - dx * half, y - dy * half, ang), (x + dx * half, y + dy * half, ang)]
        n0 = len(_TL.placements)
        # Tone-carrying alpha: a flat 205 filled every corridor with equally dark letters and
        # erased the density modeling of the typography-only panel (type-only likeness 0.43 ->
        # 0.35 measured). Same convention as the rest of the engine: more ink where the photo is
        # brighter (target_density = gray/255), so these letters model value instead of masking it.
        alpha = 205 if tone is None else int(70 + 185 * float(np.clip(tone[y, x], 0, 1)))
        placed_px += place_words_collision_aware(canvas, occupancy, path, [letters[k % len(letters)]], font,
                                                 gap_px=1.0, alpha=alpha, max_overlap=max_overlap, max_instances=1)
        placed_n += len(_TL.placements) - n0
    return placed_n, placed_px


def render_residual_fill(canvas, occupancy, theta_s, coherence_s, mask, base, get_font, rng,
                         tokens=("SOUL", "KIND", "HOME", "JOY", "WARM", "WISE", "LOVE"), size_field_px=None):
    """Fill every remaining free region larger than the smallest glyph with short tokens at the
    finest size, along the local orientation. Free space = inside the mask and not within half a
    glyph of existing ink (so nothing placed here can touch a neighbor). Repeats while a pass
    still gains footprint, so it stops when the gaps left are genuinely smaller than a glyph --
    the physical limit of "every exposed space has typography" at this resolution."""
    H, W = mask.shape
    min_px = 6
    r = max(1, int(round(min_px * 0.35)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    total_placed = 0
    # Rounds 0-2: short words. Round 3: single letters -- measured after the word rounds, ~67k px
    # of free area remained in ~360 regions whose bounding boxes were large but whose shapes were
    # slivers no 18px-long token fits; a single 6px glyph does. Still typography (the pet's
    # initials in production), and it's what "every exposed space" physically requires at the
    # font floor: any gap narrower than one glyph is unfillable at this resolution, by definition.
    letter_tokens = tuple(sorted({t[0] for t in tokens}))
    # Round 0 sizes each token to the LOCAL structural size (about half of it, like the fill
    # rounds), so a gap on the far body gets a medium word before the floor-size rounds mop up
    # what is left: with every residual token at 6px, the far body's count was dominated by
    # floor-size specks and the size hierarchy the structural pass built was buried.
    size_at = None
    if size_field_px is not None:
        def size_at(x, y):
            xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
            return max(min_px, float(size_field_px[yi, xi]) * 0.55)
    for _round in range(4):
        if _round == 3:
            tokens = letter_tokens
        _sz = size_at if _round == 0 else None
        occ_dil = cv2.dilate((occupancy > 40).astype(np.uint8), kernel)
        free = ((mask > 0.5) & (occ_dil == 0)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(free, 8)
        single = tokens == letter_tokens
        ext_min = min_px * (1.05 if single else 2.2)
        area_min = min_px * min_px * (0.8 if single else 1.5)
        big_ids = [i for i in range(1, n)
                   if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= ext_min
                   and stats[i, cv2.CC_STAT_AREA] >= area_min]
        free_area = int(sum(stats[i, cv2.CC_STAT_AREA] for i in big_ids))
        if free_area < min_px * min_px * 4:
            break
        # Per-region direct placement. The streamline tracer seeds ONCE globally and propagates
        # along its own lines, so on a mask of ~3,000 disconnected gaps it produced 1 line and 0
        # tokens (measured). Each gap instead gets a straight path through its own centroid along
        # the local fur direction (then the perpendicular if nothing fits); the collision check
        # decides whether a token actually fits.
        placed = 0
        n_before = len(_TL.placements)
        font = get_font(min_px)
        for i in big_ids:
            w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cx = stats[i, cv2.CC_STAT_LEFT] + w_ / 2.0
            cy = stats[i, cv2.CC_STAT_TOP] + h_ / 2.0
            # Large gaps get TILED (the tracer, confined to this one connected region, lays down
            # evenly spaced lanes); one straight token path through the centroid only ever fills a
            # single lane -- measured at 2x: 910k px still free after the rounds, mostly in big
            # regions that each got one token.
            if stats[i, cv2.CC_STAT_AREA] >= min_px * min_px * 40:
                x0_, y0_ = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                region = np.zeros((H, W), np.float32)
                region[y0_:y0_ + h_, x0_:x0_ + w_] = (labels[y0_:y0_ + h_, x0_:x0_ + w_] == i)
                # Lane budget scales with the region: the free space at 2x is a few enormous lace
                # networks (one can span the whole chest), and a flat 300-lane cap left 14% of the
                # animal glyph-fillable but unfilled (measured by channel width).
                _sep = max(3, int(min_px * 0.9))
                if _sz is not None:   # round 0: lanes as wide as the local token size
                    _sep = max(_sep, int(_sz(cx, cy) * 0.9))
                _area = int(stats[i, cv2.CC_STAT_AREA])
                lanes = evenly_spaced_streamlines(theta_s, coherence_s, region, _sep,
                                                  step=2.0, max_steps=max(300, int(math.hypot(w_, h_))),
                                                  min_coherence=0.0, max_turn=0.15,
                                                  max_lines=int(np.clip(_area / (_sep * _sep * 2), 300, 20000)))
                for lane in lanes:
                    placed += place_words_collision_aware(canvas, occupancy, lane, list(tokens), font,
                                                          gap_px=1.5, alpha=205, max_overlap=0.08,
                                                          size_at=_sz, get_font=get_font, gap_frac=0.25)
                continue
            ext = float(math.hypot(w_, h_))
            ang0 = float(theta_s[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))])
            got = 0
            for ang in (ang0, ang0 + math.pi / 2):
                dx, dy = math.cos(ang), math.sin(ang)
                path = [(cx + dx * t, cy + dy * t, ang) for t in np.linspace(-ext / 2, ext / 2, 9)]
                got = place_words_collision_aware(canvas, occupancy, path, list(tokens), font,
                                                  gap_px=1.5, alpha=205, max_overlap=0.08,
                                                  size_at=_sz, get_font=get_font, gap_frac=0.25)
                if got:
                    break
            placed += got
        total_placed += placed
        _log(f"    residual fill round {_round}: free area {free_area}px in {len(big_ids)} fillable regions, "
              f"{len(_TL.placements) - n_before} tokens placed ({placed}px)")
        if placed < min_px * min_px * 3 and _round < 3:
            tokens = letter_tokens      # words no longer fit anywhere -- go straight to letters
    return total_placed


def footprint_coverage(placements, mask):
    """Fraction of the animal's area lying inside some placed word's (rotated) box -- the number
    behind "comprised of typography." Stroke-pixel coverage saturates at ~35-45% even for solid
    text, so it can't support that claim; the footprint can."""
    H, W = mask.shape
    fp = np.zeros((H, W), np.uint8)
    for (x, y, fpx, chars, ang, bw, bh) in placements:
        rect = cv2.boxPoints(((float(x), float(y)), (float(bw), float(bh)), -math.degrees(ang)))
        cv2.fillPoly(fp, [np.round(rect).astype(np.int32)], 1)
    m = mask > 0.5
    return float((fp[m] > 0).mean()) if m.any() else 0.0


def placement_report(region_map, mask, extra_zones=None):
    if not _TL.placements:
        return {}
    H, W = region_map.shape
    pts = np.array([(p[0], p[1]) for p in _TL.placements])
    xs = np.clip(pts[:, 0].astype(int), 0, W - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, H - 1)
    sizes = np.array([p[2] for p in _TL.placements])
    zone_of = region_map[ys, xs]
    zones = {"eyes": (zone_of == 1) | (zone_of == 2), "muzzle": zone_of == 4, "rest": zone_of == 3}
    for name, z in (extra_zones or {}).items():
        zones[name] = z[ys, xs]
    out = {}
    for name, sel in zones.items():
        n = int(sel.sum())
        out[name] = (n, float(np.median(sizes[sel])) if n else 0.0, float(np.percentile(sizes[sel], 10)) if n else 0.0)
    return out


def ellipse_path(center, axes, angle_deg, arc_start=0.0, arc_end=2 * math.pi, n=64):
    """Generate a path around an ellipse as a list of (x, y, angle) samples -- the SAME format
    a traced fur-flow streamline produces, so it plugs directly into place_words_collision_aware/
    path_length/sample_path_at without any new placement code. `angle` at each sample is the
    ellipse's own tangent direction there, so text follows the curve naturally the same way it
    follows fur direction elsewhere."""
    cx, cy = center
    a, b = axes
    theta = math.radians(angle_deg)
    ct, st = math.cos(theta), math.sin(theta)
    pts = []
    for t in np.linspace(arc_start, arc_end, n):
        x0, y0 = a * math.cos(t), b * math.sin(t)
        x = cx + x0 * ct - y0 * st
        y = cy + x0 * st + y0 * ct
        dx0, dy0 = -a * math.sin(t), b * math.cos(t)
        dx = dx0 * ct - dy0 * st
        dy = dx0 * st + dy0 * ct
        pts.append((x, y, math.atan2(dy, dx)))
    return pts


def fit_dark_blob_ellipse(gray, mask, cx, cy, r, min_area_px=20):
    """Locate the real shape of a dark anatomical feature (pupil, nose leather) near (cx, cy):
    threshold the local neighborhood for its darkest ~35%, take the compact blob closest to the
    search center (not necessarily the largest -- a stray dark fur shadow nearby is often bigger
    than the actual feature), and fit an ellipse to its true contour. This is what makes the
    eye/nose renderers below actually track the photo's real feature shape and size rather than
    drawing a generic circle at a guessed point. Returns (center, (a, b), angle_deg) in full
    image coordinates, or None if nothing reliable is found (callers fall back to the generic
    treatment rather than draw a fabricated feature)."""
    H, W = gray.shape
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r))
    y0, y1 = max(0, int(cy - r)), min(H, int(cy + r))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    thresh = np.percentile(patch, 35)
    binary = (patch <= thresh).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return None
    pcx, pcy = patch.shape[1] / 2.0, patch.shape[0] / 2.0
    best, best_d = None, float("inf")
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        bcx, bcy = centroids[i]
        d = math.hypot(bcx - pcx, bcy - pcy)
        if d < best_d:
            best_d, best = d, i
    if best is None:
        return None
    blob = (labels == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (ecx, ecy), (minor, major), angle = cv2.fitEllipse(c)
    return (ecx + x0, ecy + y0), (major / 2.0, minor / 2.0), angle


def render_dense_disk(canvas, occupancy, center, axes, angle_deg, get_font, base, rng,
                      catchlight_offset=None, catchlight_frac=0.35):
    """A pupil or nostril: 'high typographic occupancy' per the recommendation -- dark, dense,
    genuinely typographic (several concentric micro-rings of tiny text), not a flat filled
    circle standing in for one. `axes` is (rx, ry) at `angle_deg` -- NOT forced circular, because
    a real pupil isn't always round: a cat's is a vertical slit, and forcing a circle erases that
    distinction the moment it's fitted from the photo (see render_eye_feature's second fit). A
    catchlight is real negative space, not a color: pre-seed the shared occupancy at the
    highlight's position BEFORE placing the rings, so the normal collision check naturally
    refuses to place any text there -- the void is a side effect of the same mechanism that
    prevents overlap everywhere else, not a special-cased erasure."""
    cx, cy = center
    rx, ry = axes
    rmax = max(rx, ry)
    if catchlight_offset is not None:
        theta = math.radians(angle_deg)
        ox, oy = catchlight_offset[0] * rx, catchlight_offset[1] * ry
        clx = cx + ox * math.cos(theta) - oy * math.sin(theta)
        cly = cy + ox * math.sin(theta) + oy * math.cos(theta)
        cv2.circle(occupancy, (int(round(clx)), int(round(cly))),
                  max(2, int(round(rmax * catchlight_frac))), 255.0, -1)
    for frac in np.linspace(0.2, 0.95, 5):
        ring = ellipse_path((cx, cy), (rx * frac, ry * frac), angle_deg, n=max(10, int(20 * frac)))
        font_px = max(4, base * 0.022 * (0.9 + 0.2 * rng.random()))
        place_words_collision_aware(canvas, occupancy, ring, ["LOYAL"], get_font(font_px),
                                    gap_px=font_px * 0.15, alpha=255, max_overlap=0.55)


def render_eye_feature(canvas, occupancy, gray, mask, base, center, get_font, rng):
    """Dedicated eye construction (recommendation #7): eyelid contour, iris rings, a dense
    pupil with a genuine negative-space catchlight -- built from the eye's OWN fitted shape,
    not a guessed circle. Returns True if a plausible eye was found and rendered, False if the
    fit looked unreliable (caller keeps the generic treatment there instead of drawing a
    fabricated feature over real fur)."""
    fit = fit_dark_blob_ellipse(gray, mask, center[0], center[1], base * 0.9)
    if fit is None:
        return False
    (ecx, ecy), (a_ax, b_ax), angle = fit
    if not (base * 0.12 < a_ax < base * 2.2 and base * 0.08 < b_ax < base * 2.2):
        return False   # implausible size for an eye at this photo's scale -- don't trust the fit
    # A real palpebral opening is never a thin sliver -- checked directly against a doodle render
    # where this fit grabbed a long, thin dark eyebrow-shadow crease (aspect ratio ~4.5:1) instead
    # of the actual eye, producing an oversized flattened "eye" that read as a black smudge, not
    # a feature. Reject implausibly elongated fits here rather than draw a fabricated shape.
    if max(a_ax, b_ax) / max(min(a_ax, b_ax), 1e-3) > 3.0:
        return False

    # Eyelid: one fine contour line right at the eye's own boundary.
    # Sizes cut (lid 0.085 -> 0.05, iris 0.055 -> 0.04 of base): the eye is where likeness needs
    # the finest, quietest type -- large words across a lid read as a label, not an eye.
    lid_path = ellipse_path((ecx, ecy), (a_ax * 1.08, b_ax * 1.08), angle, n=70)
    place_words_collision_aware(canvas, occupancy, lid_path, ["LOYAL"], get_font(base * 0.05),
                                gap_px=base * 0.015, alpha=200, max_overlap=0.25)
    # Iris: 2 concentric rings of small text between the pupil and the lid.
    for frac in (0.55, 0.82):
        ring = ellipse_path((ecx, ecy), (a_ax * frac, b_ax * frac), angle, n=56)
        place_words_collision_aware(canvas, occupancy, ring, ["LOYAL", "GENTLE", "SOUL"],
                                    get_font(base * 0.04), gap_px=base * 0.012, alpha=170, max_overlap=0.3)
    # Pupil: fit the pupil's OWN dark blob within the eye rather than assuming it's round --
    # a cat's pupil in bright light is a narrow vertical slit, not a circle, and that shape is
    # genuinely present in the photo's own pixels (the pupil is reliably the darkest thing inside
    # the eye). Search radius is capped to the eye's own extent so the fit can't wander into the
    # surrounding dark eyeliner fur and grab that instead.
    pupil_fit = fit_dark_blob_ellipse(gray, mask, ecx, ecy, min(a_ax, b_ax) * 0.85, min_area_px=6)
    if pupil_fit is not None:
        (pcx, pcy), (pa, pb), pangle = pupil_fit
        # Clip to a plausible pupil size -- a real pupil is a fraction of the whole eye, not most
        # of it. 0.6x was too permissive: checked directly against a doodle render where this let
        # the pupil disk swallow almost the whole eye, reading as a solid black smudge instead of
        # an eye with a visible surrounding iris band.
        pa = min(pa, a_ax * 0.42)
        pb = min(pb, b_ax * 0.42)
        if pa < base * 0.02 or pb < base * 0.02:
            pcx, pcy, pa, pb, pangle = ecx, ecy, min(a_ax, b_ax) * 0.42, min(a_ax, b_ax) * 0.42, 0.0
    else:
        pcx, pcy, pa, pb, pangle = ecx, ecy, min(a_ax, b_ax) * 0.42, min(a_ax, b_ax) * 0.42, 0.0
    render_dense_disk(canvas, occupancy, (pcx, pcy), (pa, pb), pangle, get_font, base, rng,
                      catchlight_offset=(-0.35, -0.4))
    return True




def locate_nose(gray, mask, eye_pts):
    """Find the nose leather below the eye midpoint. A single fixed eye-to-nose offset (as a
    fraction of eye separation) turned out NOT to be portable across photos: checked directly
    against all three reference photos by plotting candidate points at several ratios -- the cat
    photo's real nose sits at ~0.55-0.6x eye_sep below the midpoint, the dog's at ~0.7-0.85x, and
    the fluffy-faced doodle's at only ~0.15-0.2x (its face is foreshortened by the curly coat
    obscuring where the muzzle actually starts). One ratio cannot cover all three. Instead scan
    the whole plausible band and score each candidate fit by how nose-shaped it actually is
    (centered under the eye midpoint, a plausible size relative to eye_sep, not a thin sliver like
    a whisker crease) -- the photo itself decides where the nose is, not an assumed proportion.
    Returns (center, (a, b), angle_deg) or None."""
    if len(eye_pts) < 2:
        return None
    (x1, y1), (x2, y2) = eye_pts[0], eye_pts[1]
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    eye_sep = max(1.0, math.hypot(x2 - x1, y2 - y1))
    if _TL.nose_hint is not None:
        # A supplied nose landmark: fit the leather's real shape right there instead of scanning.
        hinted = fit_dark_blob_ellipse(gray, mask, _TL.nose_hint[0], _TL.nose_hint[1], eye_sep * 0.35, min_area_px=15)
        if hinted is not None:
            return hinted
    best, best_score = None, -1e9
    for ratio in np.linspace(0.12, 1.0, 15):
        cy = my + eye_sep * ratio
        fit = fit_dark_blob_ellipse(gray, mask, mx, cy, eye_sep * 0.45, min_area_px=15)
        if fit is None:
            continue
        (fcx, fcy), (fa, fb), fangle = fit
        if not (eye_sep * 0.08 < fa < eye_sep * 0.55):
            continue
        h_off = abs(fcx - mx) / eye_sep
        aspect = fa / max(fb, 1.0)
        score = 1.0 - h_off * 2.5 - max(0.0, aspect - 2.5) * 0.3
        if score > best_score:
            best_score, best = score, fit
    return best


def render_nose_feature(canvas, occupancy, gray, mask, base, eye_pts, get_font, rng):
    """Dedicated nose construction: locate the nose leather (via locate_nose, which scans for it
    rather than assuming a fixed offset -- see its docstring), outline its real fitted shape, add
    the central philtrum groove that's visible on every reference photo (dog, cat, and doodle all
    show a strong vertical seam splitting the nose in two), then fit each nostril AS ITS OWN dark
    blob rather than a symmetric pair of identical circles -- real nostrils are comma/W-shaped and
    not mirror-identical, and fitting each separately lets that real asymmetry come from the photo
    instead of being assumed away. Returns the fitted (center, axes, angle) on success so a caller
    can hand it to render_mouth_feature to search the correct place below the nose, or None if no
    reliable nose was found."""
    fit = locate_nose(gray, mask, eye_pts)
    if fit is None:
        return None
    (x1, y1), (x2, y2) = eye_pts[0], eye_pts[1]
    eye_sep = max(1.0, math.hypot(x2 - x1, y2 - y1))
    (ncx, ncy), (a_ax, b_ax), angle = fit

    outline = ellipse_path((ncx, ncy), (a_ax * 1.08, b_ax * 1.08), angle, n=64)
    place_words_collision_aware(canvas, occupancy, outline, ["LOYAL"], get_font(base * 0.05),
                                gap_px=base * 0.015, alpha=200, max_overlap=0.25)

    # Philtrum: a straight seam down the nose's own long axis, from just below the bridge to
    # just past the leather's lower edge -- every reference photo shows this groove clearly.
    theta = math.radians(angle)
    dx, dy = math.cos(theta), math.sin(theta)
    # The ellipse's "major" axis from fitEllipse isn't guaranteed vertical-in-photo, so use
    # whichever of the two axis directions is closer to vertical as the groove direction.
    perp_dx, perp_dy = -dy, dx
    groove_dir = (dx, dy) if abs(dy) > abs(perp_dy) else (perp_dx, perp_dy)
    glen = max(a_ax, b_ax) * 1.15
    groove_pts = []
    for t in np.linspace(-glen, glen, 24):
        gx = ncx + groove_dir[0] * t
        gy = ncy + groove_dir[1] * t
        groove_pts.append((gx, gy, math.atan2(groove_dir[1], groove_dir[0])))
    place_words_collision_aware(canvas, occupancy, groove_pts, ["LOYAL"], get_font(base * 0.035),
                                gap_px=base * 0.012, alpha=190, max_overlap=0.3)

    # Nostrils: fit each side's own dark blob (comma/W-shaped, not a generic circle) within a
    # tight local search window either side of the groove.
    side_r = min(a_ax, b_ax) * 0.85
    for side in (-1, 1):
        snx = ncx + side * a_ax * 0.42 * dx - side * 0 * dy
        sny = ncy + side * a_ax * 0.42 * dy
        nostril_fit = fit_dark_blob_ellipse(gray, mask, snx, sny, side_r, min_area_px=5)
        if nostril_fit is not None:
            (nx, ny), (na, nb), nangle = nostril_fit
            na = min(na, a_ax * 0.5)
            nb = min(nb, b_ax * 0.5)
            if na < base * 0.02 or nb < base * 0.02:
                nx, ny, na, nb, nangle = snx, sny, min(a_ax, b_ax) * 0.32, min(a_ax, b_ax) * 0.32, 0.0
        else:
            nx, ny, na, nb, nangle = snx, sny, min(a_ax, b_ax) * 0.32, min(a_ax, b_ax) * 0.32, 0.0
        render_dense_disk(canvas, occupancy, (nx, ny), (na, nb), nangle, get_font, base, rng,
                          catchlight_offset=None)
    return (ncx, ncy), (a_ax, b_ax), angle


def render_mouth_feature(canvas, occupancy, gray, mask, base, nose_fit, get_font, rng):
    """Dedicated mouth construction -- deliberately conservative. None of the three reference
    photos show an open mouth (teeth/tongue): the dog's closed jaw has NO visible seam at all
    below the nose (verified directly against the source crop -- fur just continues, no line),
    while the cat and doodle both show a genuine thin dark crease. Rather than assume a mouth
    line always exists, search the band directly below the fitted nose for the row with the
    strongest real darkness contrast against its neighbors, and only render if that contrast
    clears a real threshold -- so the dog photo correctly produces nothing (matching its source)
    instead of a fabricated line, while the cat/doodle's genuine crease gets traced and rendered
    with fine typography. Returns True/False."""
    if nose_fit is None:
        return False
    (ncx, ncy), (na, nb), nangle = nose_fit
    H, W = gray.shape
    half_w = na * 1.4
    band_top = ncy + nb * 1.1
    band_h = nb * 1.1
    x0, x1 = int(max(0, ncx - half_w)), int(min(W, ncx + half_w))
    y0, y1 = int(max(0, band_top - band_h * 0.15)), int(min(H, band_top + band_h))
    if x1 - x0 < 10 or y1 - y0 < 8:
        return False
    band = gray[y0:y1, x0:x1].astype(np.float32)
    row_mean = band.mean(axis=1)
    local_min = int(np.argmin(row_mean))
    others = np.delete(row_mean, local_min)
    if others.size == 0:
        return False
    contrast = float(others.mean() - row_mean[local_min])
    if contrast < 10.0:
        return False   # no genuine crease here -- don't draw a mouth that isn't in the photo

    mouth_row = y0 + local_min
    pts = []
    prev_y = mouth_row
    for cx_ in range(x0, x1, 2):
        lo, hi = max(y0, mouth_row - 5), min(y1, mouth_row + 6)
        col = gray[lo:hi, cx_].astype(np.float32)
        if col.size == 0:
            y_ = prev_y
        else:
            y_ = lo + int(np.argmin(col))
        pts.append((float(cx_), float(y_)))
        prev_y = y_
    if len(pts) < 6:
        return False
    path = []
    for i in range(len(pts) - 1):
        (x_, y_), (xn, yn) = pts[i], pts[i + 1]
        path.append((x_, y_, math.atan2(yn - y_, xn - x_)))
    path.append((pts[-1][0], pts[-1][1], path[-1][2] if path else 0.0))

    font_px = max(4, base * 0.035)
    place_words_collision_aware(canvas, occupancy, path, ["LOYAL"], get_font(font_px),
                                gap_px=font_px * 0.2, alpha=int(110 + min(70, contrast * 3)),
                                max_overlap=0.3)
    return True


def render_muzzle_topology(canvas, occupancy, gray, mask, base, nose_fit, eye_pts, stream, get_font, rng):
    """Dedicated muzzle construction -- "the muzzle requires its own renderer," not another patch
    of generic fur field. Two explicit radiating structures, both centered on the nose (the one
    anatomical point already reliably located): (1) muzzle spokes fanning down and out toward
    the cheeks/mouth corners -- the convergence-then-divergence pattern a real muzzle has and a
    generic streamline field has no reason to reproduce on its own; (2) whisker-pad spokes, a
    denser fan of much finer, shorter lines from two points lateral to the nose, standing in for
    the animal's actual whiskers as genuine typography -- tiny words along radiating lines that
    read as whisker texture from a distance and as words up close.

    The whisker-pad spokes are gated on real evidence of whiskers in the photo (a quick
    median-blur-residual check in the same spot, same technique as the whisker-suppression pass
    in main()) -- first version fired unconditionally "since both cats and dogs have a whisker
    pad," and on the short-haired dog photo (whose whiskers are real but visually negligible)
    that produced long bare grey lines shooting out past the cheek into open background with
    nothing there to justify them: a fabricated feature, not a constructed one. Confirmed exactly
    this failure by rendering it and looking at the result. Every spoke is a straight line (not
    curved) deliberately -- the point is an explicit, legible radiating structure, not one more
    thing chasing the fur field's own curvature. Returns True/False."""
    if nose_fit is None or len(eye_pts) < 2:
        return False
    (ncx, ncy), (na, nb), nangle = nose_fit
    (x1, y1), (x2, y2) = eye_pts[0], eye_pts[1]
    eye_sep = max(1.0, math.hypot(x2 - x1, y2 - y1))
    H, W = mask.shape

    def spoke_path(cx, cy, angle_deg, length, n=14):
        theta = math.radians(angle_deg)
        dx, dy = math.cos(theta), math.sin(theta)
        return [(cx + dx * t, cy + dy * t, theta) for t in np.linspace(nb * 0.3, length, n)]

    # ---- Muzzle spokes: nose -> cheeks/mouth-corner convergence, per the diagram -------------
    muzzle_len = eye_sep * 0.95
    for angle_deg in (55, 90, 125, 20, 160):
        path = spoke_path(ncx, ncy, angle_deg, muzzle_len, n=16)
        path = [(x, y, a) for (x, y, a) in path if 0 <= x < W and 0 <= y < H and mask[int(y), int(x)] > 0.5]
        if len(path) < 4:
            continue
        font_px = max(5, base * 0.04)
        place_words_collision_aware(canvas, occupancy, path, stream, get_font(font_px),
                                    gap_px=font_px * 0.3, alpha=180, max_overlap=0.3)

    # ---- Whisker-pad spokes: fine, short, dense fan standing in for real whiskers -----------
    # LENGTH is traced per-spoke from real evidence, not a fixed guess. First version used a
    # flat length (eye_sep*1.5) for every spoke on every photo -- confirmed directly on the
    # short-haired dog photo, whose whiskers are real but visually negligible: the result was
    # long bare grey lines shooting past the cheek into open background with nothing there to
    # justify them, a fabricated feature, not a constructed one. A patch-level "any evidence"
    # gate (tried next) didn't discriminate either -- ordinary fine fur texture near a muzzle
    # triggers the same residual signal as an actual whisker, so the dog patch scored HIGHER
    # than the doodle despite having no visible long whiskers at all. The fix: walk each
    # candidate spoke's OWN ray outward and stop it where the real thin-line signal (original
    # minus median-blurred, sampled along that specific direction) actually dies out, rather
    # than asking "is there whisker-like texture somewhere nearby" -- a real whisker persists in
    # a straight line for its own length; ordinary fur texture doesn't sustain that.
    residual_map = np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 7).astype(np.float32))
    pad_offset = na * 0.55
    pad_y = ncy + nb * 0.6
    max_reach = eye_sep * 1.8
    min_reach = eye_sep * 0.18   # a short stub even with zero evidence -- muzzle-fur continuity,
                                 # not a claim of a real long whisker
    placed_any = False
    for side in (-1, 1):
        pad_x = ncx + side * pad_offset
        base_angle = 0 if side > 0 else 180   # 0=right, 180=left (image-space atan2 convention)
        for offset_deg in (-28, -16, -6, 6, 16, 28):
            angle_deg = base_angle + offset_deg * (1 if side > 0 else -1)
            theta = math.radians(angle_deg)
            dx, dy = math.cos(theta), math.sin(theta)
            reach = min_reach
            miss_run = 0
            step = max(2.0, eye_sep * 0.03)
            t = min_reach
            while t <= max_reach:
                sx, sy = int(round(pad_x + dx * t)), int(round(pad_y + dy * t))
                if not (0 <= sx < W and 0 <= sy < H):
                    break
                # Outside the outline the threshold rises and no gap is forgiven: grass and
                # bokeh texture scored above 14 as readily as a whisker, so spokes on the tan
                # dog and the senior ran past the cheek as pale dashes in the backdrop. A real
                # whisker past the silhouette is one continuous bright line; texture is not.
                _outside = mask[sy, sx] < 0.3
                if residual_map[sy, sx] > (24.0 if _outside else 14.0):
                    reach = t
                    miss_run = 0
                else:
                    miss_run += 1
                    if miss_run > (1 if _outside else 5):   # the line has genuinely ended
                        break
                t += step
            path = spoke_path(pad_x, pad_y, angle_deg, reach, n=max(6, int(reach / step)))
            path = [(x, y, a) for (x, y, a) in path if 0 <= x < W and 0 <= y < H]
            if len(path) < 4:
                continue
            font_px = max(4, base * 0.026)
            placed = place_words_collision_aware(canvas, occupancy, path, ["LOYAL"], get_font(font_px),
                                                 gap_px=font_px * 0.25, alpha=170, max_overlap=0.4)
            placed_any = placed_any or placed > 0
    return placed_any


def find_fringe_points(mask, n_points, seed=11, gray=None, theta_s=None, base=None):
    """Locate candidate silhouette-fringe hairs from real evidence, not a guess: a segmentation
    mask's edge is antialiased, so pixels with PARTIAL coverage (neither solidly in nor solidly
    out) right at the boundary are the soft ghost of individual fur strands the segmenter could
    see well enough to register but not cleanly assign whole -- exactly the "irregular, furry,
    directional" edge quality the source photos have and this render's smooth cutout doesn't.
    Sampled with a FIXED, separate RNG (not the shared one main() uses) so the same points are
    selected every iteration regardless of how much randomness earlier code has already consumed
    that run -- the fringe should look like a stable feature of this animal's edge, not something
    that reshuffles iteration to iteration. Returns a list of (x, y) points."""
    partial = (mask > 0.06) & (mask < 0.55)
    ys, xs = np.where(partial)
    if len(xs) == 0:
        return []
    local_rng = random.Random(seed)
    if gray is None or theta_s is None or base is None:
        n = min(n_points, len(xs))
        idx = local_rng.sample(range(len(xs)), n)
        return [(float(xs[i]), float(ys[i])) for i in idx]
    # Evidence gate. The partial-coverage band exists on EVERY matte edge, smooth-coated or
    # not, so sampling it alone drew hairs off a short-haired dog's cheek as pale dashes in
    # the backdrop (staging, tan dog and senior). A hair that really leaves the outline is a
    # thin bright-or-dark line continuing outward from the point: walk the outward ray
    # 0.2 base past the edge and keep the point only if the thin-line residual (photo minus
    # its median) persists along it. Four times the candidates are drawn so a fluffy edge
    # still fills its quota; a smooth edge simply yields fewer hairs.
    H, W = mask.shape
    residual = np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 5).astype(np.float32))
    n_try = min(n_points * 4, len(xs))
    idx = local_rng.sample(range(len(xs)), n_try)
    out = []
    for i in idx:
        px, py = float(xs[i]), float(ys[i])
        dx, dy, _ = fringe_outward_dir(mask, theta_s, px, py)
        vals = []
        for t in np.linspace(2.0, base * 0.2, 6):
            sx, sy = int(round(px + dx * t)), int(round(py + dy * t))
            if 0 <= sx < W and 0 <= sy < H and mask[sy, sx] < 0.5:
                vals.append(residual[sy, sx])
        if len(vals) >= 3 and float(np.mean(vals)) > 9.0 and float(np.min(vals)) > 3.0:
            out.append((px, py))
            if len(out) >= n_points:
                break
    return out


def fringe_outward_dir(mask, theta_s, px, py):
    """The fur direction at a fringe point is mod-pi (no inherent direction), so pick whichever
    of its two signs actually points OUTWARD (toward lower mask coverage) rather than an
    arbitrary one -- shared by both the drawing pass and the compositing reveal so they always
    agree on which way each hair points."""
    H, W = mask.shape
    pxi, pyi = int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))
    angle = float(theta_s[pyi, pxi])
    dx, dy = math.cos(angle), math.sin(angle)
    fwd = mask[int(np.clip(pyi + dy * 4, 0, H - 1)), int(np.clip(pxi + dx * 4, 0, W - 1))]
    bwd = mask[int(np.clip(pyi - dy * 4, 0, H - 1)), int(np.clip(pxi - dx * 4, 0, W - 1))]
    if bwd < fwd:
        dx, dy = -dx, -dy
    return dx, dy, angle


def render_silhouette_fringe(canvas, occupancy, mask, theta_s, base, fringe_points, stream, get_font, rng):
    """Draw each fringe point (see find_fringe_points) as a short typographic hair following the
    LOCAL fur direction outward, breaking the silhouette's smooth cutout into an irregular,
    directional edge the way real fur does. Ink placed here extends past mask=0 in places, and
    the standard `a = ink_alpha * mask * ...` compositing would silently drop it there -- see
    build_fringe_zone, called separately at composite time, for how that ink is made visible."""
    H, W = mask.shape
    for (px, py) in fringe_points:
        dx, dy, angle = fringe_outward_dir(mask, theta_s, px, py)
        length = base * (0.14 + 0.22 * rng.random())
        n = max(4, int(round(length / 2.5)))
        path = [(px + dx * t, py + dy * t, angle) for t in np.linspace(0, length, n)]
        path = [(x, y, a) for (x, y, a) in path if 0 <= x < W and 0 <= y < H]
        if len(path) < 3:
            continue
        font_px = max(4, base * 0.028)
        place_words_collision_aware(canvas, occupancy, path, stream, get_font(font_px),
                                    gap_px=font_px * 0.2, alpha=150, max_overlap=0.4)


def build_fringe_zone(mask, theta_s, base, fringe_points):
    """Deterministic reveal-zone geometry for the fringe hairs above, computed independently at
    composite time (like main()'s whisker_zone) rather than reusing what render_silhouette_fringe
    drew during whichever iteration got picked as best -- that draw uses the shared per-iteration
    rng for length jitter, so its exact geometry isn't reproducible after the fact. Uses a single
    representative length instead (the middle of that jitter range) -- close enough for a reveal
    mask, which only needs to roughly cover where the ink might be, not match it pixel-for-pixel."""
    H, W = mask.shape
    zone = np.zeros((H, W), np.float32)
    length = base * 0.25
    thickness = max(1, int(round(base * 0.035)))
    for (px, py) in fringe_points:
        dx, dy, _ = fringe_outward_dir(mask, theta_s, px, py)
        ex, ey = int(round(px + dx * length)), int(round(py + dy * length))
        cv2.line(zone, (int(round(px)), int(round(py))), (ex, ey), 1.0, thickness=thickness)
    return zone


def region_coverage(ink_raw, mask, region_map, extra_zones=None):
    """Typography coverage per anatomical zone -- the number behind "areas not rendered with
    typography." Fraction of each zone's pixels under a letter (undilated canvas alpha > 0.5)."""
    m = mask > 0.5
    zones = {"eyes": m & ((region_map == 1) | (region_map == 2)),
             "muzzle": m & (region_map == 4),
             "rest": m & (region_map == 3)}
    for name, z in (extra_zones or {}).items():
        zones[name] = m & z
    out = {}
    for name, z in zones.items():
        n = int(z.sum())
        out[name] = (float((ink_raw[z] > 0.5).mean()) if n else 0.0, n)
    return out


def render_feature_microfill(canvas, occupancy, theta_s, coherence_s, mask, region_map, importance_norm,
                             base, stream, get_font, rng, zone_ids=(1, 2, 4)):
    """Fine, graduated typography inside the feature zones (eyes, muzzle/nose/mouth). The generic
    passes are region-confined and the dedicated renderers only claim their contours (lid,
    iris rings, nostrils, groove), so the iris interior, the leather between nostrils, and the
    muzzle between spokes were left with little or no type -- measured per zone by
    region_coverage before/after. Streamlines are grown INSIDE each zone (the zone is the mask)
    along the local orientation, and type size graduates by importance: finest (the font floor)
    at the pupil/nostril where importance peaks, growing outward -- so the words get smaller
    exactly where the detail they must describe gets finer. Collision-aware against everything
    already placed, so the constructed contours are never overwritten."""
    H, W = mask.shape
    placed_total = 0
    for zid in zone_ids:
        zone = ((region_map == zid) & (mask > 0.5)).astype(np.float32)
        if zone.sum() < 200:
            continue
        sep = max(4, int(round(base * 0.11)))
        lines = evenly_spaced_streamlines(theta_s, coherence_s, zone, sep, step=3.0, max_steps=400,
                                          min_coherence=0.0, max_turn=0.12,
                                          max_lines=int(800 * max(1.0, (W / 1000.0) ** 2)))
        for line in lines:
            imp = float(np.mean([importance_norm[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]
                                 for x, y, _ in line[::3]])) if line else 0.0
            font_px = base * (0.055 + 0.040 * (1.0 - imp))      # finest at peak importance
            alpha = 205                                          # fine, not faint (see structural pass)
            placed_total += place_words_collision_aware(canvas, occupancy, line, stream, get_font(font_px),
                                                        gap_px=font_px * 0.2, alpha=alpha, max_overlap=0.22)
    return placed_total


def multi_scale_orientation(gray, W):
    """Blend structure-tensor orientation across three scales (fine curls/individual hairs,
    medium fur clumps, macro head/ear volume), rather than one fixed sigma for the whole photo.

    Diagnosed directly on the Goldendoodle: its forehead rendered as long vertical columns of
    words that don't match the source's curls/whorls. A single medium-scale tensor genuinely
    can't represent both "a curl's own tight loop" and "the broad direction the forehead fur
    generally falls" -- averaging across a fixed neighborhood just washes the curls out into
    whatever the dominant coarse direction happens to be. (Worth noting: this isn't "the
    anatomical prior overpowering the structure tensor" -- there currently IS no active
    anatomical direction field; the swirl-based one was tried, made results worse, and was
    disabled earlier in this file's history. The striping is single-scale structure-tensor
    behavior, full stop.)

    Combines scales via circular statistics: each scale contributes a unit vector at angle
    2*theta (the standard trick for averaging directions that have no real 'up', since a line
    and its opposite are the same direction), weighted by that scale's OWN coherence squared --
    squared so a confidently-directional fine scale (a real curl) can dominate over a vague,
    low-confidence macro reading at the same pixel, not just contribute proportionally. The
    resultant vector's length after normalizing by total weight is itself a meaningful combined
    coherence: if fine and macro genuinely disagree, the vectors partially cancel and confidence
    drops, exactly as it should when scales conflict rather than reinforce."""
    sigmas = (max(1.0, W * 0.0035), max(2.0, W * 0.011), max(4.0, W * 0.032))
    cos2 = np.zeros(gray.shape, np.float32)
    sin2 = np.zeros(gray.shape, np.float32)
    weight_sum = np.zeros(gray.shape, np.float32)
    for sigma in sigmas:
        theta_i, coh_i = orientation_field(gray, sigma)
        # The confidence used as a WEIGHT is smoothed at that scale's own sigma before squaring
        # -- tested without this and got a real regression on a different photo's chest/leg fur:
        # a single noisy pixel at the finest scale (a JPEG artifact, a crisp individual hair)
        # can register spuriously high coherence in isolation, and squaring an UNsmoothed weight
        # let that one pixel's vote dominate its neighborhood, producing rigid, unnaturally
        # straight columns where the real fur direction is a gentle sweep. A genuine confident
        # curl agrees with its own neighbors and survives this smoothing; an isolated noise spike
        # doesn't. theta itself is left unsmoothed here -- only its vote's STRENGTH is.
        coh_w = _gblur(coh_i, (0, 0), sigmaX=sigma)
        w = coh_w * coh_w
        cos2 += np.cos(2 * theta_i) * w
        sin2 += np.sin(2 * theta_i) * w
        weight_sum += w
    theta = 0.5 * np.arctan2(sin2, cos2)
    coherence = np.divide(np.hypot(cos2, sin2), weight_sum, out=np.zeros(gray.shape, np.float32),
                          where=weight_sum > 1e-6)
    return theta, np.clip(coherence, 0, 1)


def ssim_map(img1, img2, sigma=7.0):
    """Local SSIM between two single-channel float images, same formula skimage uses, computed
    with Gaussian-weighted local windows via cv2.GaussianBlur instead of pulling in skimage."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: _gblur(x, (0, 0), sigmaX=sigma)
    mu1, mu2 = blur(img1), blur(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = blur(img1 * img1) - mu1_sq
    sigma2_sq = blur(img2 * img2) - mu2_sq
    sigma12 = blur(img1 * img2) - mu1_mu2
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return num / np.maximum(den, 1e-9)


def build_likeness_weight_map(mask, attractor_pts, base):
    """recommendation #14's face-weighted comparison, approximated with what we can actually
    detect: real per-part segmentation (nose vs. muzzle vs. cheeks) isn't available without a
    landmark model, so nose+muzzle are combined into one 'face center' region (their combined
    weight, 3+3=6) rather than faking a precision we don't have. Priority where regions overlap:
    eyes > face-center > silhouette > everywhere else in the mask (later paints over earlier)."""
    H, W = mask.shape
    m = mask > 0.5
    w = np.zeros((H, W), np.float32)
    w[m] = 1.0   # E_remaining
    edge_dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    w[m & (edge_dist < base * 0.5)] = 2.0   # E_silhouette
    if attractor_pts:
        ex = float(np.mean([p[0] for p in attractor_pts]))
        ey = float(np.mean([p[1] for p in attractor_pts]))
        eye_sep = max(1.0, math.hypot(*(np.subtract(attractor_pts[0], attractor_pts[1])))) \
            if len(attractor_pts) >= 2 else base
        face = np.zeros((H, W), np.uint8)
        cv2.ellipse(face, (int(ex), int(ey + eye_sep * 0.9)),
                    (int(eye_sep * 0.9), int(eye_sep * 1.1)), 0, 0, 360, 1, -1)
        w[m & (face > 0)] = 6.0             # E_nose + E_muzzle, combined
        for (ax, ay) in attractor_pts:
            eyes = np.zeros((H, W), np.uint8)
            cv2.circle(eyes, (int(ax), int(ay)), int(round(eye_sep * 0.35)), 1, -1)
            w[m & (eyes > 0)] = 4.0         # E_eyes
    return w


def build_region_map(mask, attractor_pts, base):
    """Semantic anatomical regions (recommendations #3/#4): eyes, muzzle/nose, and 'rest',
    confining streamlines so a forehead line can't wander through an eye into the muzzle. Real
    per-part segmentation (nose vs. left/right muzzle lobe vs. philtrum vs. chin) isn't available
    without a landmark model -- see build_likeness_weight_map's docstring for the same
    limitation -- so this is coarser than the ideal: two eye regions, one combined muzzle/nose
    region, and everything else as one 'rest' region, rather than the full 7-part model. Still a
    real, working version of the mechanism: each region gets independently seeded, so it
    generates its own typography instead of inheriting whatever a neighboring region's line
    happened to be doing. Softenable later (blend at the seams) -- deliberately not done yet, so
    the effect of hard barriers can be judged on its own first.

    Returns an (H, W) uint8 array: 0 = outside the mask, 1/2 = the two eyes (if found), 3 = rest,
    4 = muzzle/nose. Falls back to a single 'rest' region for the whole mask when no attractor
    pair was found (a region model built on a guessed eye position would be worse than none)."""
    H, W = mask.shape
    region = np.where(mask > 0.5, np.uint8(3), np.uint8(0))
    if not attractor_pts or len(attractor_pts) < 2:
        return region
    ex = float(np.mean([p[0] for p in attractor_pts]))
    ey = float(np.mean([p[1] for p in attractor_pts]))
    eye_sep = max(1.0, math.hypot(*(np.subtract(attractor_pts[0], attractor_pts[1]))))
    face = np.zeros((H, W), np.uint8)
    cv2.ellipse(face, (int(ex), int(ey + eye_sep * 0.9)),
                (int(eye_sep * 0.9), int(eye_sep * 1.1)), 0, 0, 360, 1, -1)
    region[(mask > 0.5) & (face > 0)] = 4
    for idx, (ax, ay) in enumerate(attractor_pts[:2]):
        eyes = np.zeros((H, W), np.uint8)
        cv2.circle(eyes, (int(ax), int(ay)), int(round(eye_sep * 0.4)), 1, -1)
        region[(mask > 0.5) & (eyes > 0)] = np.uint8(1 + idx)
    return region


def type_only_likeness(canvas, mask, bgr_source, attractor_pts, base, blur_sigma):
    """The test recommendation #14 asks for: blur the typography-only render and a similarly
    blurred grayscale source past the point where individual letters are legible, and compare
    what's left -- the macro tonal structure. If a change makes this number go up, the
    typography is doing more of the work of constructing the likeness, not just decorating a
    photo showing through underneath. Weighted per build_likeness_weight_map so eyes/face-center
    errors count far more than a chest patch, matching where humans actually judge likeness."""
    # SIGN FIX (matches the one in target_density, main()): this used to be `255 - ink_alpha`,
    # the ordinary "ink is a dark mark on white paper" reading. But the actual colored render
    # composites `dark_ground*(1-a) + photo*a` where `a` scales with ink_alpha -- more ink
    # reveals MORE of the photo (bright or dark, whatever it is), less ink reveals more of the
    # dark ground regardless of the photo. So "more ink" only means "matches a BRIGHT source
    # pixel" under this compositor, not a dark one. Scoring it the old way was measuring
    # agreement with the WRONG target and drove the correction loop to visibly darken every
    # render relative to its actual source colors -- confirmed directly by the user's side-by-
    # side comparison. Flipped so this metric rewards the same thing the real render needs.
    ink_alpha = np.asarray(canvas.split()[3], np.float32)   # 0..255, ink coverage only
    type_only_gray = ink_alpha                               # more ink -> brighter, matches
                                                              # a bright source pixel now
    source_gray = cv2.cvtColor(bgr_source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = _gblur(type_only_gray, (0, 0), sigmaX=blur_sigma)
    b = _gblur(source_gray, (0, 0), sigmaX=blur_sigma)
    smap = ssim_map(a, b, sigma=blur_sigma)
    weights = build_likeness_weight_map(mask, attractor_pts, base)
    m = mask > 0.5
    score = float((smap[m] * weights[m]).sum() / max(1e-6, weights[m].sum()))
    return score, a, b


def find_attractor_points(feat, mask, top_k=2):
    """Connected components of the feature field = proxy eye/nose locations. In a real
    integration this would come from pet_landmarks.py's actual RTMPose detections (already
    built and verified earlier tonight); this sandbox can't run that model (network-blocked,
    same limitation hit before), so the already-computed, already-validated photometric feat
    field stands in for it here -- same idea, a cheaper/less reliable source of the points.

    History, in order:
    1. First attempt used a wide, uncapped area range and found 18 "attractors" -- fur texture
       throws off plenty of small dark/light blobs that are NOT eyes or a nose, and blending a
       swirl field at all 18 made the overall flow noticeably MORE chaotic. Narrowed to the top
       few by size.
    2. That still picked the WRONG points -- the largest feat blobs turned out to be a dark
       chest-fur patch, not the eyes/nose, because a fur patch can simply have more raw pixel
       area than a small compact eye. Restricted candidates to a fixed fraction of the mask's
       own bounding-box height first.
    3. That fixed fraction silently assumed every photo is a head-and-chest crop. On a
       full-body standing dog it found zero attractors (an eye is a far smaller fraction of a
       full-body silhouette's area/height than of a head crop's).
    4. Two different geometric "find the neck/head-region" rewrites (a head-width rescaling,
       then a width-profile neck-narrowing detector) were both tried and both broke on real
       test photos in different ways -- overestimating on wide floppy ears, underestimating on
       a cat's pointed ones, and latching onto the crop's own bottom edge or the shoulders
       instead of the neck (each confirmed by drawing the computed cutoff on the actual photo,
       not assumed). Geometrically delimiting "the head region" first turned out to be the
       wrong foundation: it's fragile precisely because head/neck/ear shape varies so much
       across species and framing.
    5. Rebuilt around a different, far more universal signal instead: EYES COME IN A PAIR --
       similar size, similar height, meaningfully apart horizontally. That symmetry holds
       regardless of species, crop, or pose, so this now scores every pair of compact
       (non-elongated -- rejects fur streaks and ear-edge highlights) candidate blobs by how
       well they match that description and keeps the best pair, only using a generous
       (not precise) vertical band to keep obviously off-limits territory like paws/tail out of
       consideration at all.
    6. Pair-scoring alone still wasn't sufficient, found on two more real photos: (a) a curly
       Goldendoodle matched its best "pair" to the two OUTER EAR EDGES -- both compact, similar
       size, similar height, genuinely apart -- because a crisp silhouette edge against a plain
       background can out-contrast fuzzy real eyes; (b) a black Labrador matched an eyebrow
       crease and that SAME eye's own catchlight as its "pair" (uneven studio lighting made one
       eye's catchlight much fainter than the other's), i.e. two points near ONE eye rather than
       one point per eye. Fixed with two more checks: an eye sits well inside the silhouette, so
       reject any candidate too close to the mask boundary (screens out the ear-edge case); and
       a real pair must straddle the head's own horizontal center, one on each side (screens out
       the same-eye case) -- plus a minimum separation as a fraction of head WIDTH, not just of
       the blobs' own size, since two points near the same eye can still individually be
       "big enough apart" relative to a small blob's size without being real eye-to-eye distance.
    7. On the SAME doodle and Labrador photos, both eyes plus the brow ridge between them had
       actually merged into one oversized blob at feat's usual 0.5 cutoff (a soft shadow bridges
       the gap), leaving no second component to pair with, so the scorer's best REAL option was
       the wrong ear/eyebrow pair above. A sharper cutoff (0.7) does split that merge back into
       two correctly-placed eye blobs -- confirmed directly -- but every way of applying it that
       was tried also broke something else that was already working: applied globally, it erased
       an already-faint eye pair on a head-crop dog entirely, AND invented a brand new false
       pair on an unrelated leg marking on a different photo; restricted to only the single
       largest oversized blob's own bounding box, it missed a real case where the two eyes sat
       in two SEPARATE oversized diffuse regions (left half / right half of a solid-black head)
       rather than one shared merge; and widened further to the union of every oversized region,
       it started re-finding false positives again on the leg-marking photo (a solid-black dog's
       fur legitimately produces large diffuse blobs almost everywhere on the head, so "large
       diffuse blob near the head" doesn't reliably mean "worth re-splitting" the way it first
       appeared to). Reverted to the plain single-threshold version: it leaves the doodle's ear
       pair unfixed (a known, narrower miss) and the Labrador's eyes undetected (folds into the
       already-known dark-fur limitation) rather than risk a wrong hero placement somewhere
       unrelated on the body -- an honest "found nothing" is a safer failure than a confident,
       badly wrong point, and this stays true to that."""
    H, W = feat.shape
    ys, xs = np.nonzero(mask > 0.5)
    if len(ys) == 0:
        return []
    top, bottom = float(ys.min()), float(ys.max())
    head_cutoff = top + (bottom - top) * 0.65
    band_xs = xs[ys <= head_cutoff]
    if len(band_xs) == 0:
        return []
    band_center = (float(band_xs.min()) + float(band_xs.max())) / 2.0
    band_width = max(1.0, float(band_xs.max()) - float(band_xs.min()))
    edge_dist = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    total = float((mask > 0.5).sum()) or 1.0
    min_sep = band_width * 0.12   # a real eye-to-eye gap relative to head width, not blob size

    def best_pair(blobs):
        best, best_score = None, -1.0
        for i in range(len(blobs)):
            a1, x1, y1, s1 = blobs[i]
            for j in range(i + 1, len(blobs)):
                a2, x2, y2, s2 = blobs[j]
                if abs(x1 - x2) < max(min_sep, 0.5 * max(s1, s2)):
                    continue
                if (x1 - band_center) * (x2 - band_center) >= 0:
                    continue   # both on the same side of center -- not a left/right eye pair
                y_align = 1.0 - min(1.0, abs(y1 - y2) / (0.6 * max(s1, s2) + 1e-6))
                if y_align <= 0:
                    continue
                size_ratio = min(a1, a2) / max(a1, a2)
                score = size_ratio * y_align * math.sqrt(min(a1, a2))
                if score > best_score:
                    best_score, best = score, (i, j)
        return best

    binary = (feat > 0.5).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    good = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        frac = area / total
        cy = float(centroids[i][1])
        cx = float(centroids[i][0])
        bw, bh = float(stats[i, cv2.CC_STAT_WIDTH]), float(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        # Upper bound raised from 0.02 -- confirmed directly this was actively hiding the real
        # eyes on a test photo where local contrast enhancement merged each eye with its
        # eyebrow/surrounding dark fur into one much larger blob (0.067-0.09 here, previously
        # excluded entirely), leaving only small unrelated blobs (a cheek/chin marking) as
        # candidates, which is exactly what the pair-scorer then confidently picked. The
        # pair-symmetry checks below (straddles center, similar size, similar height, minimum
        # separation) are the real defense against false positives like a chest patch, not this
        # area cap -- being re-verified against the rest of the test set now that it's raised.
        if cy > head_cutoff or frac < 0.00015 or frac > 0.12 or aspect > 2.2:
            continue
        cxi = int(np.clip(round(cx), 0, W - 1))
        cyi = int(np.clip(round(cy), 0, H - 1))
        if edge_dist[cyi, cxi] < band_width * 0.03:
            continue
        good.append((area, cx, cy, math.sqrt(area)))

    pair = best_pair(good)
    if pair is None:
        # No candidate pair passed the symmetry test -- on a dark-furred face the eyes can
        # genuinely be near-invisible in a photometric contrast field (confirmed: a black
        # Border Collie's eyes barely register at all, since eye and surrounding fur are both
        # dark), or two real features can already be fused into one oversized blob (see history
        # point 7 above for why re-splitting that was tried and reverted). Falling back to
        # "biggest blob wins" here would be the original mis-picking bug re-entered through a
        # different door. Returning no attractors is the honest answer -- main() already has a
        # sensible fallback (image-center-ish) for when none are found.
        return []
    i, j = pair
    # Only the validated pair is returned by default (top_k=2) -- a "bonus" third point (an
    # attempt at the nose) was tried and, on the cat test photo, landed on an ear-tip texture
    # blob instead, since nothing validates it the way pair-symmetry validates the eyes. Better
    # to return two points we're confident in than three where the third is a guess.
    ordered = [good[i], good[j]] + [b for k, b in enumerate(good) if k not in pair]
    return [(x, y) for _a, x, y, _s in ordered[:top_k]]


def blend_attractor_field(theta, coherence, attractor_pts, radius, strength=0.85):
    """Blend a radial 'contour lines wrapping around a bump' swirl into theta near each
    attractor point, strongest close to the point and fading with distance. Angles here are
    mod pi (a line has no direction), so blending them the naive way breaks near the wrap-around
    -- the double-angle trick (blend (cos 2theta, sin 2theta) vectors, convert back after) is
    the correct way to average orientations, same reasoning as resolve_angle() elsewhere."""
    if not attractor_pts:
        return theta, coherence
    H, W = theta.shape
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    cos2, sin2 = np.cos(2 * theta), np.sin(2 * theta)
    weight_total = np.zeros((H, W), np.float32)
    for (cx, cy) in attractor_pts:
        dx, dy = xx - cx, yy - cy
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        w = strength * np.exp(-(dist / radius) ** 2)
        attractor_theta = np.arctan2(dy, dx) + np.pi / 2.0   # tangent to a circle around the point
        cos2 = cos2 * (1 - w) + np.cos(2 * attractor_theta) * w
        sin2 = sin2 * (1 - w) + np.sin(2 * attractor_theta) * w
        weight_total = np.maximum(weight_total, w)
    blended_theta = 0.5 * np.arctan2(sin2, cos2)
    # Attractor zones read as confidently directional (real fur genuinely does wrap tightly
    # around an eye), which also feeds the "place important words near the face" idea below --
    # a line that spent real distance in a high-coherence attractor zone scores higher for hero.
    blended_coherence = np.clip(coherence + weight_total * 0.5, 0, 1)
    return blended_theta, blended_coherence


def perpendicular(angle):
    return angle + math.pi / 2.0


def evenly_spaced_streamlines(theta, coherence, mask, sep_px, step=4.0, max_steps=4000,
                              min_coherence=0.05, max_turn=0.10, test_frac=0.55, max_lines=3000,
                              seed_region=None, region_map=None):
    """Returns a list of streamlines, each a list of (x, y, angle) points, tiling `mask`
    at roughly `sep_px` spacing everywhere -- see module docstring for the algorithm.

    seed_region: optional boolean/float mask restricting where the SINGLE initial seed (picked
    by highest coherence) may be chosen. Needed once the silhouette-tangent boundary blend
    (see main()) started boosting coherence right at the edge: the global coherence argmax then
    always landed ON the boundary, and since the boundary loop is a closed contour, that first
    streamline traced almost the entire perimeter in one line -- consuming the whole growth
    budget before the interior (fur texture) ever got a seed (measured directly: this collapsed
    growth from ~350-400 lines to 1). Pass a mask that excludes the near-edge band to force the
    first seed into real texture; growth from there still reaches the edge normally via the
    usual perpendicular-offset queueing, it just no longer STARTS there.

    `sep_px` may be a scalar (uniform spacing, the original behavior) OR an (H, W) array
    giving the LOCAL target separation at each point -- denser where detail should carry the
    likeness (features, silhouette), sparser where a few calm strokes read better (chest). A
    charcoal portrait doesn't put the same number of marks per square inch everywhere; classic
    Jobard-Lefer's constant separation can't express that on its own, so this generalizes the
    stamp/enqueue/accept radii to look the local value up wherever a field is given.

    (An earlier version of this function also accepted init_seed_covered/init_trace_covered/
    seed_points, to let a second call grow more streamlines seeded explicitly in leftover gaps.
    Tried and measured directly: gaps left by collision-aware word placement are WORD-placement
    gaps along a line's own geometry -- a hero word collided with a neighbor and got skipped --
    not untraced territory. A new trace seeded there re-enters that SAME line's own
    trace_covered stamp within a handful of steps and dies (measured: 16 seeded attempts
    produced lines 20-59px long, recovering essentially no coverage). Gap-fill now reuses the
    first pass's own `lines` with a second, smaller-font placement call instead -- see main().)

    region_map: optional (H, W) int array of anatomical-region IDs (see build_region_map).
    A traced line is confined to whichever region its OWN seed point started in -- the moment a
    step would land in a DIFFERENT region, that direction terminates, exactly the "a streamline
    shouldn't just continue because the vector field permits it" boundary the recommendations
    asked for. This is deliberately simple (no softened exceptions yet): each region ends up
    tiled by its own independently-seeded streamlines, so an eye gets its own typography instead
    of a passing forehead line wandering through it."""
    H, W = theta.shape
    is_field = isinstance(sep_px, np.ndarray)

    def sep_at(x, y):
        if not is_field:
            return sep_px
        xi = int(np.clip(round(x), 0, W - 1))
        yi = int(np.clip(round(y), 0, H - 1))
        return float(sep_px[yi, xi])

    seed_covered = np.zeros((H, W), np.uint8)
    trace_covered = np.zeros((H, W), np.uint8)

    def stamp(pts, arr, frac):
        for (x, y, _a) in pts:
            r = max(1, int(round(sep_at(x, y) * frac)))
            cv2.circle(arr, (int(round(x)), int(round(y))), r, 1, -1)

    def trace_one_dir(x0, y0, a0, seed_region_id):
        x, y, a = x0, y0, a0
        pts = [(x, y, a)]
        outside_run = 0
        for _ in range(max_steps):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H):
                break
            if mask[yi, xi] < 0.5 or coherence[yi, xi] < min_coherence:
                break
            if len(pts) > 2 and trace_covered[yi, xi]:
                break
            if region_map is not None:
                # A hard, single-pixel cutoff measurably hurt: terminating the instant a line
                # touches a different region fragments lines right at the seam, and that
                # fragmentation itself turned out to cost more macro tonal structure (measured
                # via the type-only likeness score) than the barrier gained. A few steps of
                # hysteresis -- require the line to be PERSISTENTLY outside its own region, not
                # just brush a single boundary pixel -- keeps the actual intent (an eye doesn't
                # get overrun by a long forehead line) while not fragmenting on a graze.
                if region_map[yi, xi] != seed_region_id:
                    outside_run += 1
                    if outside_run >= 3:
                        break
                else:
                    outside_run = 0
            raw = float(theta[yi, xi])
            a = clamp_turn(resolve_angle(raw, a), a, max_turn)
            x += step * math.cos(a)
            y += step * math.sin(a)
            pts.append((x, y, a))
        return pts

    def trace_bidirectional(x0, y0, a0):
        seed_region_id = region_map[int(round(y0)), int(round(x0))] if region_map is not None else 0
        fwd = trace_one_dir(x0, y0, a0, seed_region_id)
        bwd = trace_one_dir(x0, y0, a0 + math.pi, seed_region_id)
        bwd = list(reversed(bwd[1:]))   # drop the duplicated seed point, lead INTO fwd
        return bwd + fwd

    lines = []
    queue = []

    def enqueue_seeds_from(line):
        acc = 0.0
        for i in range(1, len(line)):
            x0, y0, _ = line[i - 1]
            x1, y1, a1 = line[i]
            acc += math.hypot(x1 - x0, y1 - y0)
            local_sep = sep_at(x1, y1)
            if acc >= local_sep:
                acc = 0.0
                perp = perpendicular(a1)
                for sgn in (1, -1):
                    queue.append((x1 + sgn * local_sep * math.cos(perp),
                                  y1 + sgn * local_sep * math.sin(perp), a1))

    seed_pool = mask > 0.5
    if seed_region is not None:
        restricted = seed_pool & (seed_region > 0.5)
        if restricted.any():   # fall back to the full mask if the region is somehow empty
            seed_pool = restricted
    idxs = np.argwhere(seed_pool)
    if len(idxs) == 0:
        return lines
    coh_at_mask = coherence[idxs[:, 0], idxs[:, 1]]
    y0, x0 = idxs[int(np.argmax(coh_at_mask))]
    first = trace_bidirectional(float(x0), float(y0), 0.0)
    if path_length(first) > sep_at(x0, y0):
        lines.append(first)
        stamp(first, seed_covered, 1.0)
        stamp(first, trace_covered, test_frac)
        enqueue_seeds_from(first)

    guard = 0
    while queue and len(lines) < max_lines:
        guard += 1
        if guard > max_lines * 8:
            break
        qx, qy, qa = queue.pop()
        xi, yi = int(round(qx)), int(round(qy))
        if not (0 <= xi < W and 0 <= yi < H) or mask[yi, xi] < 0.5 or seed_covered[yi, xi]:
            continue
        line = trace_bidirectional(qx, qy, qa)
        if path_length(line) < sep_at(qx, qy) * 1.1:
            continue
        lines.append(line)
        stamp(line, seed_covered, 1.0)
        stamp(line, trace_covered, test_frac)
        enqueue_seeds_from(line)

    return lines


def render_v2(bgr, words=None, *, mask=None, render_scale=None, max_overlap=None,
              landmarks=None, debug_dir=None, out_stem="render", verbose=False, backdrop_rgb=None,
              type_scale=None):
    """Render a typographic portrait of the pet in `bgr` (BGR uint8, already at the working
    resolution). Returns (rgb_uint8, metrics). `words`: the customer's comma-separated name +
    descriptors (the first entries weight highest; see _weighted_stream); None -> DEFAULT_WORDS.
    `mask`: optional precomputed foreground matte (e.g. after print-aspect fitting). `render_scale`:
    upsample factor applied here (None -> GOP_SCALE env, default 1). `max_overlap`: global
    collision cap (None -> GOP_MAX_OVERLAP env). `landmarks`: "x1,y1;x2,y2[;nx,ny]" eyes(+nose)
    from the production landmark model (None -> GOP_LANDMARKS env -> heuristic detector).
    `debug_dir`: when set, writes the A/B/C/D QA panels there as <out_stem>_*.jpg. `backdrop_rgb`:
    an (r, g, b) tuple for everything OUTSIDE the animal (the site's Gallery Dark / Gallery
    Gray choice); None keeps the photo-derived backdrop. The gap color between letters ON the
    animal is always derived from the coat -- it carries tone -- and is not affected.
    `type_scale`: the site's Small/Medium/Large slider (0.30 fine .. 0.56 bold, pet_proto's
    scale). It multiplies the micro and structural sizes TOGETHER, so the hierarchy between
    the fine face and the far body is the same at every setting; 0.30, the slider's default
    and what every staging judgment was made at, is 1.0x."""
    _TL.verbose = bool(verbose)
    _TL.nose_hint = None
    _TL.max_overlap_cap = float(max_overlap) if max_overlap is not None else \
        float(os.environ.get("GOP_MAX_OVERLAP", "0.08") or 0.08)
    out_path = os.path.join(debug_dir, out_stem + ".jpg") if debug_dir else None
    if render_scale is None:
        render_scale = float(os.environ.get("GOP_SCALE", "1") or 1)
    if render_scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_CUBIC)
        if mask is not None:
            mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        _log(f"render scale {render_scale:g}x -> {bgr.shape[1]}x{bgr.shape[0]}")
    H, W = bgr.shape[:2]
    if mask is None:
        mask = _foreground_mask(bgr)
    bgr_source = bgr.copy()   # kept for the type-only likeness test -- compare against the
                              # REAL photo, not our own contrast-enhanced version of it
    bgr = _enhance_contrast(bgr, mask)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Multi-scale (recommendation #11) replaces the old single-sigma tensor -- see
    # multi_scale_orientation's docstring for why (the doodle's vertical-striping complaint).
    theta, coherence = multi_scale_orientation(gray, W)
    theta_s = _gblur(theta, (0, 0), sigmaX=max(1.0, W * 0.006))
    coherence_s = _gblur(coherence, (0, 0), sigmaX=max(1.0, W * 0.006))

    base = max(16, int(round(W * 0.048)))

    # ---- Silhouette-tangent field near the boundary (recommendation #13) ------------------
    # As a streamline nears the silhouette, blend its direction toward the boundary's own
    # tangent -- the level sets of the mask's distance transform run parallel to the edge
    # everywhere, so the tangent to that level set IS the local silhouette tangent. This is a
    # different technique from the swirl-blend tried earlier for the eyes (which fought the
    # real fur signal and made things worse, per the user's own comparison): here the field
    # being blended toward is a genuine geometric property of the shape itself, not a synthetic
    # radial guess, and it only dominates in a narrow band right at the edge. Mod-pi angles
    # again require the double-angle trick to blend correctly (see blend_attractor_field).
    dist_to_edge = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    dist_blur = _gblur(dist_to_edge, (0, 0), sigmaX=max(1.5, W * 0.006))
    gy, gx = np.gradient(dist_blur)
    boundary_theta = np.arctan2(gy, gx) + math.pi / 2.0   # tangent = inward normal rotated 90 deg
    boundary_w = np.clip(1.0 - dist_to_edge / (base * 1.3), 0, 1) ** 1.5   # 1 at the edge, 0 inland
    cos2 = np.cos(2 * theta_s) * (1 - boundary_w) + np.cos(2 * boundary_theta) * boundary_w
    sin2 = np.sin(2 * theta_s) * (1 - boundary_w) + np.sin(2 * boundary_theta) * boundary_w
    theta_s = 0.5 * np.arctan2(sin2, cos2)
    coherence_s = np.clip(coherence_s + boundary_w * 0.35, 0, 1)

    # Feature field (eyes/nose proxy -- see find_attractor_points' docstring for why this
    # stands in for real pet_landmarks.py detections in this sandbox). Computed once, reused
    # for BOTH the attractor field below and the eyes/nose ink-protection at composite time.
    broad = _gblur(gray.astype(np.float32), (0, 0), sigmaX=max(1.0, W * 0.06))
    localdark = np.clip((broad - gray.astype(np.float32)) / 55.0, 0, 1)
    locallight = np.clip((gray.astype(np.float32) - broad) / 70.0, 0, 1)
    feat_raw = np.maximum(localdark, locallight) * (mask > 0.5)
    ok = int(max(3, round(W * 0.011))) | 1
    feat_raw = cv2.morphologyEx(feat_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)))
    feat = np.clip(_gblur(feat_raw, (0, 0), sigmaX=max(1.0, W * 0.012)) * 1.6, 0, 1)

    # Silhouette fringe points (recommendation: "edges are irregular, furry and directional" in
    # the source vs. "generally smooth/masked" in the render) -- picked once, from the mask's own
    # antialiased edge, so they're a stable feature of this animal's silhouette rather than
    # reshuffling per iteration. See find_fringe_points' docstring for why this counts as real
    # evidence rather than a fabricated guess.
    fringe_points = find_fringe_points(mask, n_points=max(40, int(W * 0.05)), gray=gray, theta_s=theta_s, base=base)
    _log(f"fringe: {len(fringe_points)} hairs with evidence of leaving the outline")

    attractor_radius = base * 2.2
    # Eye candidates come from the DARK half of the feature field only. `feat` (used below for
    # density/size gradation) is max(localdark, locallight), and on the dog the pair-scorer's
    # best pair was two bright tan fur patches on the forehead (measured: gray ~170 at both
    # attractor points vs ~30 at the real eyes, ~110px away). Eyes are dark; a bright anomaly
    # should never have been eligible.
    feat_dark_raw = cv2.morphologyEx(localdark * (mask > 0.5), cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)))
    feat_dark = np.clip(_gblur(feat_dark_raw, (0, 0), sigmaX=max(1.0, W * 0.012)) * 1.6, 0, 1)
    attractor_pts = find_attractor_points(feat_dark, mask)
    # Validate the dark-only pair by depth inside the silhouette, with the original detector as
    # fallback. Measured on the three test photos (edge distance as a fraction of head width):
    # every real eye >= 0.174 (dog 0.174/0.176, doodle 0.229/0.259, cat 0.274); the one false
    # positive -- the cat's second point landing on a dark ear edge -- 0.036. An eye is never at
    # the silhouette edge; 0.10 splits the two with margin either side. The original max(dark,
    # light) field found the cat's eyes correctly and the dog's/doodle's wrongly; the dark-only
    # field is the reverse, so the two together cover all three -- by measurement, not by hope.
    _m5 = mask > 0.5
    _ys, _xs = np.nonzero(_m5)
    if len(_ys) and len(attractor_pts) >= 2:
        _top, _bot = _ys.min(), _ys.max()
        _bx = _xs[_ys <= _top + (_bot - _top) * 0.65]
        _band_w = max(1.0, float(_bx.max() - _bx.min())) if len(_bx) else 1.0
        _ed = cv2.distanceTransform(_m5.astype(np.uint8), cv2.DIST_L2, 5)
        _depths = [float(_ed[int(np.clip(py, 0, H - 1)), int(np.clip(px, 0, W - 1))]) / _band_w
                   for (px, py) in attractor_pts]
        if min(_depths) < 0.10:
            fallback = find_attractor_points(feat, mask)
            _log(f"dark-only eye pair rejected (edge depth {[round(d, 3) for d in _depths]} < 0.10 of head width); "
                  f"falling back to combined-field pair {fallback}")
            attractor_pts = fallback
    # Landmark injection: GOP_LANDMARKS="x1,y1;x2,y2[;nx,ny]" (two eyes, optional nose). This is
    # the seam where production's pet_landmarks.py (RTMPose AP-10K) output enters; the pose
    # model's hosts are blocked from this sandbox, so known coordinates are supplied directly
    # instead. Overrides the heuristic above entirely -- a real detection beats a proxy.
    _lm_env = (landmarks or os.environ.get("GOP_LANDMARKS", "")).strip()
    if _lm_env:
        _pts = [tuple(float(v) for v in p.split(",")) for p in _lm_env.split(";") if p.strip()]
        if len(_pts) >= 2:
            attractor_pts = [_pts[0], _pts[1]]
        if len(_pts) >= 3:
            _TL.nose_hint = _pts[2]
        _log(f"landmarks injected from GOP_LANDMARKS: eyes={attractor_pts} nose={_pts[2] if len(_pts) >= 3 else None}")
    _log(f"anatomical attractors found (used for hero placement + size gradation only): "
          f"{len(attractor_pts)}  {attractor_pts}")
    # NOT blending these into theta/coherence anymore: two corrected attempts both made the
    # flow visibly more chaotic than the plain texture field, which already curves around the
    # eyes on its own (confirmed by comparing real renders side by side, not assumed) -- the
    # artificial circular swirl fought that real signal rather than reinforcing it. The points
    # are still useful for WHERE to anchor hero words and size gradation below, just not for
    # bending the flow direction itself.
    head_center = (float(np.mean([p[0] for p in attractor_pts])), float(np.mean([p[1] for p in attractor_pts]))) \
        if attractor_pts else (W / 2.0, H * 0.35)

    # Expanded from 3 phrases -- flagged directly as reading like "texture stamps" once density
    # got high enough to show the pattern clearly. _weighted_stream already front-loads whatever
    # comes first with more repetition (up to ~3.2x, decaying by position), so ordering this list
    # IS the hierarchy: short, plain words up front get seen often; the longer, more specific
    # tail appears but doesn't dominate. Deliberately mixes short/medium/long phrases throughout
    # (not grouped by length) so curvature_adaptive has real material in every bucket everywhere,
    # not just wherever the "short" words happen to sit in the list.
    full_words = words if (words and words.strip()) else DEFAULT_WORDS
    # Two vocabularies from the customer's text. _phrases splits on COMMAS, so a customer's
    # sentence ("MAGGIE LOSES HER MIND WHEN I COME IN THE DOOR") arrived as ONE 44-character
    # rigid token and was the only thing the engine had to place -- seen on staging: every
    # placement a banner, nothing short enough to follow a curl, fill a gap or build an eye.
    # The full phrases stay for the hero lines (a sentence across the brow reads well); the
    # streamline and fill passes get the individual WORDS, order preserved (name first keeps
    # its top weight in _weighted_stream), deduplicated, two-letter filler dropped.
    phrases = _phrases(full_words)
    seen, word_list = set(), []
    for ph in phrases:
        for w in ph.split():
            if len(w) >= 3 and w not in seen:
                seen.add(w); word_list.append(w)
    # Under ten distinct words -- a bare name, or a six-word sentence like "SHADOW SITS ON THE
    # WARM LAUNDRY" -- the same word repeats across every inch of the crown (seen on staging).
    # Pad with the brand vocabulary. The customer's words stay first, so _weighted_stream still
    # gives the name the top weight and the padding the least; the fills and the initials
    # (short_tokens, letter_tokens below) keep drawing on the customer's own words first.
    if len(word_list) < 10:
        for w in _phrases(DEFAULT_WORDS):
            for ww in w.split():
                if ww not in seen and len(word_list) < 14:
                    seen.add(ww); word_list.append(ww)
    stream = _weighted_stream(", ".join(word_list))
    hero_words = (phrases or stream)[:1]
    # Short tokens for the residual fill and initials for the channel fill come from the
    # customer's own words too, so the fine texture is theirs (the pet's name most of all).
    short_tokens = tuple(w for w in word_list if len(w) <= 5) or ("SOUL", "KIND", "HOME", "JOY")
    letter_tokens = tuple(dict.fromkeys(w[0] for w in word_list))

    # Baseline spacing tied to the SMALLER end of the size range that will be assigned
    # afterward, so fine/mid text fits the tiling without excessive overlap. Tightened from
    # 0.34x to 0.28x base along with the density-field tuning above, in response to "too
    # sparse" -- more streamlines everywhere, not just in the feature/edge zones.
    sep_px = max(6, int(round(base * 0.28)))

    # ---- Spatially variable separation (recommendation #4) --------------------------------
    # A charcoal portrait doesn't put the same number of marks per square inch everywhere;
    # neither should this. Denser (smaller separation -> more streamlines -> more typographic
    # material) near real features and right at the silhouette, where detail is what carries
    # the likeness; sparser (larger separation) over calm regions far from the head -- "longer,
    # calmer trajectories through the chest." This is now what carries TONE, not font size (see
    # recommendation #5/#6 below) -- density is the primary mechanism, size stays in three
    # controlled classes.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist_to_feat = np.full((H, W), 1e6, np.float32)
    for (ax, ay) in attractor_pts:
        dist_to_feat = np.minimum(dist_to_feat, np.hypot(xx - ax, yy - ay))
    feat_zone = np.clip(1.0 - dist_to_feat / (attractor_radius * 1.6), 0, 1)      # 1 at a feature
    edge_zone = np.clip(1.0 - dist_to_edge / (base * 1.2), 0, 1)                  # 1 at the silhouette
    # Tuned down from an earlier pass that read as "too sparse" overall (user feedback,
    # confirmed by comparing coverage: 46% with the old font-size-driven tone vs 32% with the
    # first density-driven version) -- the chest/background falloff was too aggressive (up to
    # 1.8x baseline separation) and the baseline itself was too loose. Density should still be
    # LOWEST far from the head, but the floor of that falloff needs to stay closer to the
    # baseline so calm regions read as "fewer, calmer strokes," not "empty."
    chest_zone = np.clip((yy - head_center[1]) / max(1.0, H * 0.35), 0, 1) * (1.0 - np.maximum(feat_zone, edge_zone))
    density_mult = 1.0 - 0.45 * feat_zone - 0.30 * edge_zone + 0.30 * chest_zone
    # This is now the STARTING point for the per-iteration density, not the final field --
    # sep_correction (below) adjusts it each round based on measured tonal error.
    sep_field_base = np.clip(sep_px * density_mult, sep_px * 0.45, sep_px * 1.3).astype(np.float32)

    # Semantic anatomical regions (recommendations #3/#4) -- see build_region_map's docstring
    # for what this can and can't distinguish given no landmark model. None when no attractor
    # pair was found, which evenly_spaced_streamlines treats as "no barriers" (old behavior).
    region_map = build_region_map(mask, attractor_pts, base)

    # ---- Anatomical importance map, reused as a RENDERING control, not just a scoring metric --
    # build_likeness_weight_map already encodes the right hierarchy (eyes=4, nose+muzzle=6,
    # silhouette=2, rest=1) because it was built to match how a viewer actually reads a face --
    # but until now it only ever fed the likeness score, never the render itself, so the eyes and
    # nose got the SAME ink boldness as a patch of neck fur. Reusing the identical map for both
    # means the portrait is now optimizing for the same thing it's being judged on. Normalized to
    # [0,1] and used below to scale ink alpha (visual priority/boldness), not density -- density's
    # own feat/edge/chest zones are already tuned and this avoids double-compounding two
    # overlapping "near a feature" signals into an unpredictable extreme.
    importance_map = build_likeness_weight_map(mask, attractor_pts, base)
    importance_norm = importance_map / max(1.0, float(importance_map.max()))

    def line_importance(line):
        vals = [importance_norm[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))] for x, y, _ in line[::4]]
        return float(np.mean(vals)) if vals else 0.0

    rng = random.Random(7)
    font_cache = {}

    def get_font(px):
        px = max(6, int(round(px)))
        f = font_cache.get(px)
        if f is None:
            f = ImageFont.truetype(_FONT, px) if _FONT else ImageFont.load_default()
            font_cache[px] = f
        return f

    def mean_coherence(line):
        vals = [coherence_s[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))] for x, y, _ in line[::4]]
        return float(np.mean(vals)) if vals else 0.0

    def min_dist_to_attractor(line):
        if not attractor_pts:
            return float("inf")
        pts = np.array([(x, y) for x, y, _ in line[::3]])
        best = float("inf")
        for (ax, ay) in attractor_pts:
            d = float(np.min(np.hypot(pts[:, 0] - ax, pts[:, 1] - ay)))
            best = min(best, d)
        return best

    # ---- Target tonal map for iterative error-correction ------------------------------------
    # The missing piece, diagnosed via the type-only likeness test: nothing before this point
    # ties ink density to the photo's actual LIGHT/DARK pattern -- density was driven by
    # distance-from-feature and fur coherence, never by "does this patch need to read as light
    # or dark to match the animal." That's why the typography-only panel never showed a face
    # when blurred (uniform gray texture where the source showed clear eye/nose blobs): its own
    # tonal map was never asked to resemble the source's. This target, and the loop below that
    # converges toward it, is that missing objective. Built from the same photo (`gray`, already
    # contrast-enhanced) and at the same kind of local-averaging scale the likeness test itself
    # uses, so what we optimize against here is the same thing that test measures.
    #
    # SIGN FIX: this was originally `1 - gray/255` (dark photo -> high target ink), copying
    # ordinary charcoal-on-white-paper logic where more ink makes a mark darker. That's the
    # wrong model for THIS compositor: the ground is a dark navy, and `a` (which scales directly
    # with ink density) controls how much of the actual PHOTO shows through it -- composited =
    # dark_ground*(1-a) + photo*a. More ink reveals MORE of the photo, whatever its color; less
    # ink reveals more of the dark ground regardless of what the photo actually looked like
    # there. So the inverted target was starving typography from every BRIGHT part of the coat
    # (the majority of most pets' faces) and pushing extra density into already-dark regions
    # that read as dark either way -- confirmed as the direct cause of "far too dark, doesn't
    # look like the source photo": the correction was correct arithmetic aimed at the wrong
    # goal. Flipped so bright photo -> high target ink -> that brightness gets revealed.
    target_density = np.clip(gray.astype(np.float32) / 255.0, 0, 1)
    target_density = 0.08 + 0.77 * target_density   # keep some paper AND some ink everywhere --
                                                     # a pure-black or pure-white target would
                                                     # erase texture at the tonal extremes
    corr_sigma = max(10.0, W * 0.02)
    target_density_blur = _gblur(target_density, (0, 0), sigmaX=corr_sigma)
    # Correct harder where likeness actually depends on it (eyes, muzzle/nose) than on generic
    # body texture -- reuses the SAME weighting the automated likeness test itself scores by.
    likeness_weights = build_likeness_weight_map(mask, attractor_pts, base)
    lr_map = 0.35 + 0.55 * np.clip(likeness_weights / 6.0, 0, 1)
    # First version of this loop kept streamline geometry FIXED across iterations and corrected
    # only word size/alpha on the existing lines. It converged (every one of 5 test photos
    # improved 20-145% in likeness) but plateaued well short of the target: collision-avoidance
    # caps how much MORE ink a given lane can hold before overlap-avoidance itself blocks further
    # placement, so the correction ran into a ceiling that had nothing to do with the tonal
    # target being wrong -- there was just nowhere left to put more ink within that geometry.
    # `sep_correction` fixes that by feeding back into the density FIELD itself: under-inked
    # regions get genuinely denser streamlines next round (more lanes to place words on, not
    # just bigger/darker words on the same lanes), over-inked regions get sparser ones.
    sep_correction = np.ones((H, W), np.float32)
    # Regrowing the streamline geometry every round (real Jobard-Lefer growth, not just a
    # placement re-run) is genuinely more expensive -- fewer rounds than the fixed-geometry
    # version, which could afford 4-7.
    # Was a fixed 5. Measured across the test photos, the best-scoring round was the 2nd-4th;
    # rounds after the score turns down were pure cost (each is a full grow + place). Cap at 4
    # (PET_V2_ITERS to override) and stop early on the first clear decrease -- see the loop tail.
    N_ITERS = int(os.environ.get("PET_V2_ITERS", "4") or 4)

    def line_mean_xy(line):
        pts = np.array([(x, y) for x, y, _ in line[::4]])
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    # ---- Continuous size gradation, tone still carried by density (recommendations #5, #6) --
    # An earlier version of this used two hard-switched sizes (MICRO / STRUCTURAL) based on a
    # single distance threshold -- fixed the old word-cloud problem, but the user flagged the
    # visible result as "jumps from small to medium to large without care": a line just inside
    # the threshold and one just outside it could differ suddenly with nothing in between.
    # Replaced with a smoothstep interpolation across the SAME distance-from-feature measure,
    # so every line in between gets a proportionate size -- a continuous gradient from fine
    # near the eyes/nose out to the (still modest, non-hero) structural size, rather than three
    # discrete steps. Tone is still primarily density (sep_field above), not size -- the low end
    # was also lowered (0.17x -> 0.10x base) per "smallest text should be finer."
    # Slider: Small 0.30 -> 1.00x (the tuned look), Medium 0.42 -> 1.29x, Large 0.56 -> 1.60x.
    # A 0.75 power so Large is bold and graphic without the far body outgrowing the frame.
    _tsk = (float(type_scale) / 0.30) ** 0.75 if type_scale else 1.0
    _tsk = min(2.0, max(0.7, _tsk))
    MICRO_PX = base * 0.10 * _tsk
    STRUCT_PX = base * 0.52 * _tsk   # widened from 0.40 so the far body genuinely reads larger (size_field)
    # Widened from 1.3x -- measured the ACTUAL micro/structural/hero split (recommendation #6's
    # target: 20-30% / 60-70% / 3-7% of ink area) and found micro was only 11-15%: at 1.3x, only
    # a small ring right around the eyes graded toward the fine end, so nearly the whole rest of
    # the portrait defaulted to full structural size regardless of how far it actually was.
    feat_close_radius = attractor_radius * 3.7

    def line_feat_dist(line):
        if not attractor_pts:
            return float("inf")
        lx, ly = line_mean_xy(line)
        return min(math.hypot(lx - ax, ly - ay) for ax, ay in attractor_pts)

    FILL_PX = MICRO_PX * 0.85

    # ---- Size rule for everything outside the features (a continuous field, one rule for every
    # photo) -----------------------------------------------------------------------------------
    # At preview size every zone measured a 6px median: outside the eyes/nose the whole animal
    # sat at one size and there was no hierarchy to read. Three signals, combined per pixel:
    #   * distance from the face -- fine near it, opening up down the neck and chest; hero words
    #     only far from it (see hero selection);
    #   * texture energy -- fine wherever the photo has fine detail (fur strands, folds), larger
    #     where the coat is smooth, so size follows what the fur needs to be described;
    #   * flow coherence -- a minor opener where the fur runs calm and straight (added per line).
    _gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    _gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    _energy = _gblur(np.hypot(_gx, _gy), (0, 0), sigmaX=max(2.0, base * 0.6))
    _p95 = float(np.percentile(_energy[mask > 0.5], 95)) if (mask > 0.5).any() else 1.0
    detail_field = np.clip(_energy / max(1e-6, _p95), 0, 1)
    if len(attractor_pts) >= 2:
        _es = max(1.0, math.hypot(attractor_pts[0][0] - attractor_pts[1][0], attractor_pts[0][1] - attractor_pts[1][1]))
    else:
        _es = base * 4.0

    # ---- The features: both eyes and the nose, as points and as a fine zone ------------------
    # locate_nose reads only the photo, so this is computed once here rather than per iteration.
    _nf = locate_nose(gray, mask, attractor_pts) if len(attractor_pts) >= 2 else None
    feature_pts = list(attractor_pts[:2])
    if _nf is not None:
        feature_pts.append((float(_nf[0][0]), float(_nf[0][1])))
    elif len(attractor_pts) >= 2:
        feature_pts.append((float(np.mean([p[0] for p in attractor_pts[:2]])),
                            float(np.mean([p[1] for p in attractor_pts[:2]])) + _es * 0.75))

    # Where type must stay fine: the eyes, nose and mouth THEMSELVES, graduating out. The cap
    # used to key on the likeness weight map (importance_norm > 0.45), whose "face-center"
    # ellipse is 1.8 eye-separations wide and 2.2 tall -- a scoring choice, not an anatomical
    # one; on a tight crop it covered half the animal and 80% of the structural words, and
    # everything was pinned to the micro size. Now: the eye discs (0.20 eye-sep: the eye, its
    # lids and the socket -- "the typography around and in the eyes should be fine"), the nose
    # leather locate_nose fits, and a mouth band below it are the fine zone; the cap eases off
    # over 0.20 eye-separations outside it. Fine ON the features, growing steadily away.
    fine_blend = np.ones((H, W), np.float32)
    if len(attractor_pts) >= 2:
        fine = np.zeros((H, W), np.uint8)
        for (ax, ay) in attractor_pts[:2]:
            cv2.circle(fine, (int(ax), int(ay)), int(round(_es * 0.20)), 1, -1)
        if _nf is not None:
            (ncx, ncy), (na, nb), nang = _nf
            cv2.ellipse(fine, (int(ncx), int(ncy)), (int(na * 1.0) + 1, int(nb * 1.0) + 1),
                        float(nang), 0, 360, 1, -1)
            nr = max(na, nb)
            cv2.ellipse(fine, (int(ncx), int(ncy + nr * 1.3)), (int(_es * 0.35), int(nr * 0.9) + 1),
                        0, 0, 360, 1, -1)
        else:
            mx, my = feature_pts[2]
            cv2.ellipse(fine, (int(mx), int(my)), (int(_es * 0.30), int(_es * 0.45)), 0, 0, 360, 1, -1)
        _out = cv2.distanceTransform((1 - fine).astype(np.uint8), cv2.DIST_L2, 5)
        fine_blend = np.clip(_out / max(1.0, _es * 0.20), 0, 1).astype(np.float32)

    # "Distance from the face" is the distance to the NEAREST feature -- eye, eye or nose --
    # not to the midpoint between the eyes. On a close-up the nose sits a full eye-separation
    # below that midpoint, so the old rule read the nose and muzzle as far from the face and
    # put the largest words in the frame on them (seen on staging, twice). Normalized by what
    # is in the frame: the farthest visible part of the animal gets the largest type whether
    # that is the chest bottom or, on a tight crop, the ear tips.
    if feature_pts:
        _dist = np.full((H, W), 1e9, np.float32)
        for (fx, fy) in feature_pts:
            _dist = np.minimum(_dist, np.hypot(xx - fx, yy - fy))
    else:
        _dist = np.hypot(xx - head_center[0], yy - head_center[1])
    _max_in_mask = float(np.percentile(_dist[mask > 0.5], 97)) if (mask > 0.5).any() else _es * 2.4
    # Two parts. NEAR is anatomical, in eye-separations: 0 at a feature, full at 1.2 eye-
    # separations (the ear tips), the same on every framing -- so a tight crop, where the
    # whole frame is close to the face, cannot stretch the fine region into medium type on
    # the muzzle bridge (a single frame-relative normalization did exactly that on staging).
    # FAR is frame-relative: whatever lies beyond 1.2 eye-separations grades up to the largest
    # type at the farthest visible part of the animal, chest bottom or ear tips alike.
    _d_es = _dist / _es
    _near = np.clip(_d_es / 1.2, 0, 1) ** 0.8
    _far_span = max(_es * 0.3, _max_in_mask - _es * 1.2)
    _far = np.clip((_dist - _es * 1.2) / _far_span, 0, 1)
    face_dist_field = (0.55 * _near + 0.45 * _far).astype(np.float32)
    _denom = _es * 1.2
    # Smoothness may only ENLARGE type away from the features. A whitened senior muzzle is
    # the smoothest, brightest patch on the head, and the ungated term opened type up right
    # there. It fades in between 0.45 and 0.85 eye-separations from the nearest feature, so
    # on the face itself size is distance alone.
    _smooth_gate = np.clip((_d_es - 0.45) / 0.40, 0, 1)
    size_field = np.clip(0.60 * face_dist_field + 0.40 * (1.0 - detail_field) * _smooth_gate, 0, 1).astype(np.float32)

    def line_size_t(line):
        vals = [size_field[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))] for x, y, _ in line[::4]]
        return float(np.mean(vals)) if vals else 0.0

    # The structural size every pixel would get (before per-line coherence/jitter): what the
    # lane spacing, the channel fill and the residual fill size themselves against.
    _cap = MICRO_PX * 1.05
    _raw_px = MICRO_PX + (STRUCT_PX - MICRO_PX) * np.clip(0.65 * size_field + 0.20, 0, 1)
    size_px_field = np.where(_raw_px > _cap, _cap + (_raw_px - _cap) * fine_blend, _raw_px).astype(np.float32)
    if _KEEP_FIELDS:   # ~100 MB of float32 planes at print size, so never retained in production
        _TL.fields.update(size_field=size_field, face_dist_field=face_dist_field, detail_field=detail_field,
                          importance_norm=importance_norm, micro_px=MICRO_PX, struct_px=STRUCT_PX, es=_es,
                          denom=_denom, max_in_mask=_max_in_mask, fine_blend=fine_blend,
                          size_px_field=size_px_field, feature_pts=feature_pts)

    # ---- The iterative loop: regrow geometry, rasterize, compare, correct DENSITY, regrow -----
    # This is the follow-up to the fixed-geometry version (which corrected only word size/alpha
    # and plateaued because collision-avoidance caps how much ink an EXISTING lane can hold).
    # Each round grows a fresh set of streamlines from the CURRENT sep_field, places the whole
    # portrait on them, measures the actual residual error against the target tone, and adjusts
    # sep_field itself for the next round -- genuinely denser lanes where still too light,
    # sparser where already too dark, not just bolder words on the same lanes.
    canvas = occupancy = lines = None
    best_score, best_canvas, best_tier_stats = -1.0, None, None
    best_placements = []
    best_pass = []
    best_stats = dict(_TL.stats)
    down_streak = 0
    for iteration in range(N_ITERS):
        sep_field = np.clip(sep_field_base * sep_correction, sep_px * 0.30, sep_px * 1.8).astype(np.float32)
        # Lanes must be at least as wide as the type they carry. The tonal spacing above was
        # tuned for the fine end (6-18px lanes); a 20-28px structural word on the far body
        # collided with its neighbors and was rejected, and the 6px channel letters took the
        # ground instead -- measured on a close-up: 120 structural words against 901 channel
        # letters in the far band. Where the size field asks for larger type, tone is carried
        # by opacity (tone_gain) in fewer, wider lanes rather than by lane count.
        sep_field = np.maximum(sep_field, size_px_field * 0.85).astype(np.float32)
        _log(f"iteration {iteration}: growing streamlines (sep {float(sep_field.min()):.1f}-"
              f"{float(sep_field.max()):.1f})...")
        # Line caps scale with pixel count: a fixed 3000 is fine at 1x (~1000 lines grown) but binds
        # at 2x, and everything downstream (fill, footprint) starves for lanes.
        px_scale = (W / 1000.0) ** 2
        lines = evenly_spaced_streamlines(theta_s, coherence_s, mask, sep_field, max_lines=int(3000 * max(1.0, px_scale)),
                                          step=max(3.0, base * 0.12), min_coherence=0.05, max_turn=0.10,
                                          seed_region=(edge_zone < 0.5), region_map=region_map)

        # ---- Fallback fill for genuinely bare pockets left by region confinement ------------
        # Confirmed directly on the Goldendoodle: a wrongly-placed attractor point (landing on
        # the ear, a known separate limitation) creates a small "eye zone" there via
        # build_region_map. The region barrier then correctly blocks surrounding streamlines
        # from crossing INTO it -- but nothing reliably seeds a real eye-sized zone's own
        # interior when it doesn't actually contain an eye, so that patch was left with no
        # streamline geometry AT ALL, not just skipped words (checked: a true gap, no line
        # passes anywhere near it). Detect any such untraced pocket after growth and fill it
        # with a small supplementary, UNCONFINED pass (region_map=None) -- the point of
        # confinement is to stop cross-contamination between anatomical regions, which doesn't
        # apply to territory nothing reached in the first place, so there's nothing to protect
        # by leaving it empty instead.
        traced = np.zeros((H, W), np.uint8)
        stamp_r = max(2, int(round(float(np.median(sep_field)) * 0.6)))
        for line in lines:
            for (x, y, _a) in line[::2]:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H:
                    cv2.circle(traced, (xi, yi), stamp_r, 1, -1)
        bare = (mask > 0.5) & (traced == 0)
        bare_u8 = bare.astype(np.uint8) * 255
        bk = max(3, int(round(base * 0.15))) | 1
        bare_u8 = cv2.morphologyEx(bare_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bk, bk)))
        n_bare, bare_labels, bare_stats, bare_centroids = cv2.connectedComponentsWithStats(bare_u8, 8)
        # First version of this used (sep_px*4)^2 and had NO bound on how far each fallback
        # call's own streamlines could travel -- measured directly, this found 11 "pockets" a
        # round (mostly ordinary small crevices, not real voids) and each unconfined,
        # unbounded call re-grew a full independent tiling of a large chunk of the mask,
        # exploding to 13,000+ streamlines and heavily re-covering ground the main pass had
        # already handled. Raised the area threshold to catch only genuinely substantial voids,
        # and capped each fallback call's own reach (max_steps, max_lines) so even a real one
        # stays a small local patch instead of a second full-mask regrowth.
        min_bare_area = (sep_px * 10) ** 2
        fallback_seeds = [(float(cx), float(cy)) for i, (cx, cy) in enumerate(bare_centroids)
                          if i > 0 and bare_stats[i, cv2.CC_STAT_AREA] >= min_bare_area]
        if fallback_seeds:
            _log(f"iteration {iteration}: {len(fallback_seeds)} bare pocket(s) found, "
                  f"filling with bounded local streamlines")
            for (sx, sy) in fallback_seeds:
                # evenly_spaced_streamlines only takes an explicit SEED POINT via seed_region --
                # restrict it to a small disk around this pocket's centroid so its one
                # (highest-coherence-within-the-disk) seed lands inside the actual bare area,
                # then let it grow LOCALLY (bounded steps/count) and unconfined (no region_map,
                # since confinement is exactly what left this territory untouched) from there.
                pocket_region_u8 = np.zeros((H, W), np.uint8)
                pr = max(3, int(round(base * 0.12)))
                cv2.circle(pocket_region_u8, (int(round(sx)), int(round(sy))), pr, 1, -1)
                local_steps = max(20, int(round(base * 3.0 / max(3.0, base * 0.12))))
                extra = evenly_spaced_streamlines(theta_s, coherence_s, mask, sep_field,
                                                  step=max(3.0, base * 0.12), min_coherence=0.0,
                                                  max_turn=0.10, seed_region=(pocket_region_u8 > 0),
                                                  max_steps=local_steps, max_lines=25)
                lines.extend(extra)

        scored = [(path_length(l), mean_coherence(l), min_dist_to_attractor(l), l) for l in lines]
        # Hero candidates are picked by PROXIMITY TO A REAL FEATURE first, length second -- "the
        # name reads near the eyes" rather than whichever streamline happens to be longest.
        # Hero lines go on the BODY, away from the features. They used to be chosen by proximity
        # to the eyes ("the name reads near the eyes"), which put the largest type across the
        # muzzle and eye sockets -- exactly where likeness needs the finest, quietest type.
        # Now: long, coherent lines that sit well outside the feature radius and in the
        # lowest-importance territory, longest first.
        hero_candidates = sorted(
            (s for s in scored if s[1] > 0.30 and s[0] > base * 2.0
             and s[2] > attractor_radius * 1.6 and line_importance(s[3]) < 0.35),
            key=lambda s: -s[0])
        hero_lines = set(id(s[3]) for s in hero_candidates[:5])
        scored.sort(key=lambda s: -s[0])
        non_hero_lines = [line for _l, _coh, _d, line in scored if id(line) not in hero_lines]
        line_coh = {id(line): coh for _l, coh, _d, line in scored}
        # Overlap tolerances tightened across the board -- flagged directly as "still overlaps
        # and collides, making the text unreadable." These were tuned earlier purely to
        # maximize ink coverage/likeness score, and 0.28-0.45 tolerance turns out to allow up to
        # nearly HALF of a word's own pixels to sit on top of already-placed ink before it's
        # rejected -- plenty to make letters genuinely hard to parse where they cross. A visible
        # gap is the correct outcome there, not a collision.
        fill_rounds = [
            (FILL_PX,        0.12, 0.18),   # (font size, max_overlap, gap_px as a fraction of size)
            (FILL_PX * 0.70, 0.16, 0.10),
            (FILL_PX * 0.55, 0.20, 0.06),
            (FILL_PX * 0.42, 0.24, 0.04),
        ]

        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        occupancy = np.zeros((H, W), np.float32)   # shared collision map for this iteration
        _TL.placements.clear()                        # per-iteration word placement log
        _TL.pass_tags.clear()
        _TL.pass_name = "feature"
        for _k in _TL.stats:
            _TL.stats[_k] = 0
        micro_px_area, struct_px_area, hero_px_area, fill_px_area = 0, 0, 0, 0
        MICRO_STRUCT_SPLIT = 0.5   # size_t below this counts toward micro, above -> structural

        # ---- Dedicated anatomical features (eyes, nose) -- claim their territory in occupancy
        # FIRST, so the generic structural/micro/gap-fill passes below naturally route around
        # them through the same collision mechanism that prevents overlap everywhere else. Only
        # attempted when a real attractor pair was found; otherwise falls back entirely to the
        # generic treatment (unchanged prior behavior for bcclean/blacklab).
        if len(attractor_pts) >= 2:
            for (ax, ay) in attractor_pts:
                render_eye_feature(canvas, occupancy, gray, mask, base, (ax, ay), get_font, rng)
            nose_fit = render_nose_feature(canvas, occupancy, gray, mask, base, attractor_pts, get_font, rng)
            render_mouth_feature(canvas, occupancy, gray, mask, base, nose_fit, get_font, rng)
            render_muzzle_topology(canvas, occupancy, gray, mask, base, nose_fit, attractor_pts, stream, get_font, rng)


        _TL.pass_name = "fringe"
        render_silhouette_fringe(canvas, occupancy, mask, theta_s, base, fringe_points, stream, get_font, rng)
        _TL.pass_name = "struct"

        # ---- Structural/micro pass -- everything EXCEPT hero lines (recommendation #6) -----
        # Size/alpha formula is back to plain (no correction multiplier) -- sep_field above is
        # now the ONLY tonal-correction lever, which is exactly the point: density genuinely
        # varies (more/fewer lanes), rather than the same lanes just getting bigger or darker.
        for length, coh, dist, line in scored:
            if id(line) in hero_lines:
                continue
            t = np.clip(line_feat_dist(line) / feat_close_radius, 0, 1)
            smooth = t * t * (3 - 2 * t)                    # smoothstep: eases in/out, no kink
            t_coh = np.clip(coh / 0.5, 0, 1)
            # Size rule (see size_field): distance-from-face and texture energy carry it,
            # feature proximity and flow coherence are minor terms.
            size_t = np.clip(0.65 * line_size_t(line) + 0.15 * smooth + 0.20 * t_coh, 0, 1)
            cls_px = MICRO_PX + (STRUCT_PX - MICRO_PX) * size_t
            font_px = cls_px * (0.92 + 0.16 * rng.random())   # tiny jitter for organic texture
            font = get_font(font_px)

            # Per-WORD size along the line (size_at): the line-level size_t above is only the
            # micro/struct bookkeeping split now. A streamline on a close-up runs from the
            # cheek to the frame edge; one size for all of it averaged the field to the middle
            # and every band measured the same 6px median. Each word reads the size field,
            # the importance cap and the jitter at its own position instead.
            def size_at(x, y, _smooth=smooth, _t_coh=t_coh):
                xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
                st = min(1.0, max(0.0, 0.65 * float(size_field[yi, xi]) + 0.15 * _smooth + 0.20 * _t_coh))
                px = (MICRO_PX + (STRUCT_PX - MICRO_PX) * st) * (0.92 + 0.16 * rng.random())
                cap = MICRO_PX * 1.05
                return cap + (px - cap) * float(fine_blend[yi, xi]) if px > cap else px
            # Raised from 150-210 -- flagged as a "gauzy haze": even where a stroke IS present,
            # alpha this low means the composite still shows more ground/wash than real photo
            # color, so ink strokes themselves read as translucent rather than crisp. Letters
            # still get real gradation from t_coh; they just don't max out below full ink anymore.
            # Features get FINER and QUIETER type, not bolder: the size cap on the eyes, nose
            # and mouth is applied per word in size_at (fine_blend), not per line here -- the
            # old line-level importance_norm > 0.45 test is what pinned half a close-up to
            # the micro size. The line-level font/font_px above only seed the placement call.
            # Opacity stays neutral in feature territory: a -18% ease-off there dropped the
            # muzzle's tonal mass (the brightest region on both test dogs, where tone wants the
            # MOST ink) -- muzzle SSIM 0.856 -> 0.746 measured. Fineness comes from size alone.
            alpha = int(min(255, 195 + 55 * t_coh))
            # words_for_curvature (line-mean-coherence bucketing) is defined above but NOT used
            # here -- tried it, and it made things worse, not better. See its docstring: this
            # pipeline's coherence field reads high almost everywhere on the doodle (a side effect
            # of the multi-scale smoothing that fixed an EARLIER "vertical striping" bug), so
            # nearly every line routed to the small "long" bucket and one or two long phrases
            # dominated the whole render -- confirmed directly, twice, with two different
            # thresholding schemes. Reverted to plain round-robin over the full vocabulary until
            # there's a real per-segment curvature signal to drive this with (see conversation).
            placed = place_words_collision_aware(canvas, occupancy, line, stream, font,
                                                 gap_px=font_px * 0.35, alpha=alpha,
                                                 size_at=size_at, get_font=get_font, gap_frac=0.35)
            if size_t < MICRO_STRUCT_SPLIT:
                micro_px_area += placed
            else:
                struct_px_area += placed

        # ---- Hero pass -- the final composition layer, not a density decision -------------
        _TL.pass_name = "hero"
        for length, coh, dist, line in scored:
            if id(line) not in hero_lines:
                continue
            # Was base*(0.95-1.20) -- 2.5-3x the structural max, a jump the user flagged as too
            # great. Scaled RELATIVE to STRUCT_PX instead, and NOT run through the error
            # correction -- hero is a deliberate compositional decision, not a tonal-fit one.
            font_px = STRUCT_PX * (1.25 + 0.25 * rng.random())   # STRUCT_PX widened; keep hero ~0.7x base
            font = get_font(font_px)
            # max_instances=1: hero lines are picked for being LONG, so without this the same
            # word repeats every gap_px along the whole line -- a cluster of large "LOYAL"s
            # stacked together, not one deliberate emphasis placement.
            # max_overlap tightened from 0.60 -- "a deliberate headline is allowed to sit boldly
            # through the finer texture" was true in principle, but 60% overlap tolerance was
            # producing genuinely illegible collisions, not a tasteful foreground/background
            # relationship. Hero is still allowed a bit more than structural text (it's meant to
            # stand out, and it has already-fixed absolute size/position going for it), just not
            # to the point of being unreadable.
            hero_px_area += place_words_collision_aware(canvas, occupancy, line, hero_words, font,
                                                         gap_px=font_px * 0.35, alpha=255, max_overlap=0.22,
                                                         max_instances=1)

        # ---- Gap-fill pass -- closes bare patches the collision budget above left behind ---
        pre_fill_coverage = float((occupancy[mask > 0.5] > 40).mean())
        prev_coverage = pre_fill_coverage
        _TL.pass_name = "fill"
        for round_px, round_overlap, gap_frac in fill_rounds:
            for line in non_hero_lines:
                # Fill words follow the same size field as the structural pass (0.7x .. 2.2x the
                # round size): measured with a fixed size, the fills outnumbered the structural
                # words 2:1 and dragged every band's median to the floor, hiding the hierarchy.
                # Relative to the LOCAL structural size (55% of it on round 0, stepping down by
                # round): near the face that's the floor; on the far body it's a real medium
                # word. Measured at 2400px with an absolute fill size, fills sat at 7-9px in
                # every band and buried a 38-44px far-body hierarchy in count.
                local_struct = MICRO_PX + (STRUCT_PX - MICRO_PX) * line_size_t(line)
                font_px = local_struct * (round_px / FILL_PX) * 0.55 * (0.90 + 0.20 * rng.random())
                font = get_font(font_px)
                t_coh = np.clip(line_coh[id(line)] / 0.5, 0, 1)
                alpha = int(min(255, 165 + 65 * t_coh))

                def fill_size_at(x, y, _ratio=(round_px / FILL_PX) * 0.55):
                    xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
                    ls = MICRO_PX + (STRUCT_PX - MICRO_PX) * float(size_field[yi, xi])
                    px = ls * _ratio * (0.90 + 0.20 * rng.random())
                    cap = MICRO_PX * 1.05
                    return cap + (px - cap) * float(fine_blend[yi, xi]) if px > cap else px
                fill_px_area += place_words_collision_aware(canvas, occupancy, line, stream, font,
                                                             gap_px=font_px * gap_frac, alpha=alpha,
                                                             max_overlap=round_overlap,
                                                             size_at=fill_size_at, get_font=get_font,
                                                             gap_frac=gap_frac)
            cov = float((occupancy[mask > 0.5] > 40).mean())
            gained = cov - prev_coverage
            prev_coverage = cov
            if gained < 0.003:
                break
        post_fill_coverage = prev_coverage

        # ---- Feature-zone micro-fill: fine, graduated type inside eyes/muzzle -----------------
        ink_pre = np.asarray(canvas.split()[3], np.float32) / 255.0
        cov_pre = region_coverage(ink_pre, mask, region_map)
        _TL.pass_name = "microfill"
        micro_px_area += render_feature_microfill(canvas, occupancy, theta_s, coherence_s, mask, region_map,
                                                  importance_norm, base, stream, get_font, rng)
        ink_post = np.asarray(canvas.split()[3], np.float32) / 255.0
        cov_post = region_coverage(ink_post, mask, region_map)
        _log(f"iteration {iteration}: zone coverage before->after micro-fill: "
              + "  ".join(f"{k} {cov_pre[k][0]:.0%}->{cov_post[k][0]:.0%}" for k in ("eyes", "muzzle", "rest")))

        # Residual + channel fill used to run here, every iteration -- 45 s of an 88 s 1600px
        # render, for canvases that were then discarded by keep-best. They now run ONCE, after
        # the loop, on the winning canvas (see below). The density loop's error signal is
        # measured on lanes + gap-fill, which is what it was tuned on before those fills existed.

        # ---- Measure the actual residual error and update the DENSITY field ---------------
        ink_now = np.asarray(canvas.split()[3], np.float32) / 255.0
        current_blur = _gblur(ink_now, (0, 0), sigmaX=corr_sigma)
        error = (target_density_blur - current_blur) * (mask > 0.5)   # + means still too light
        mean_abs_err = float(np.abs(error[mask > 0.5]).mean())
        score, _, _ = type_only_likeness(canvas, mask, bgr_source, attractor_pts, base, corr_sigma)
        _log(f"iteration {iteration}: grown {len(lines)} lines, gap-fill "
              f"{pre_fill_coverage:.1%}->{post_fill_coverage:.1%}  mean tonal error={mean_abs_err:.4f}  "
              f"likeness={score:.4f}")

        # Convergence isn't guaranteed monotonic (measured directly: a stronger/longer
        # fixed-geometry run once ended WORSE than an earlier round of itself) -- keep whichever
        # round actually scored best rather than assuming the last one always is.
        if score > best_score:
            best_score = score
            best_canvas = canvas.copy()
            best_occupancy = occupancy.copy()
            best_placements = list(_TL.placements)
            best_pass = list(_TL.pass_tags)
            best_stats = dict(_TL.stats)
            best_tier_stats = (micro_px_area, struct_px_area, hero_px_area, fill_px_area)
        elif iteration >= 1 and score < best_score - 0.005:
            # The score has turned down. One dip is not a verdict -- measured: a single-decrease
            # stop quit the 1050px dog at round 2 on a dip earlier runs recovered from, costing
            # 0.03 of likeness. Two consecutive decreases is: the density field has passed its
            # best fit, and the remaining rounds only cost time. Keep-best holds the winner.
            down_streak += 1
            if down_streak >= 2:
                _log(f"iteration {iteration}: likeness below best twice running ({score:.4f} < {best_score:.4f}); stopping early")
                break
        if score > best_score - 0.005:
            down_streak = 0

        if iteration < N_ITERS - 1:
            # error > 0 (still too light) -> shrink sep_correction -> denser lanes next round.
            # error < 0 (too dark already) -> grow sep_correction -> sparser lanes next round.
            sep_correction = np.clip(sep_correction * (1.0 - lr_map * error * 1.1), 0.35, 2.6)
            sep_correction = _gblur(sep_correction, (0, 0), sigmaX=max(4.0, base * 0.2))

    canvas = best_canvas
    occupancy = best_occupancy
    micro_px_area, struct_px_area, hero_px_area, fill_px_area = best_tier_stats
    _log(f"using iteration with best likeness ({best_score:.4f})")

    # ---- Final fills, once, on the winning canvas -------------------------------------------
    # Residual (short tokens in every free region a word fits) then channel (single letters
    # along the medial axis of the leading corridors). Placement log/stats are rewound to the
    # winning iteration first so the claim metrics describe exactly this canvas.
    _TL.placements[:] = best_placements
    _TL.pass_tags[:] = best_pass
    _TL.stats.update(best_stats)
    _TL.pass_name = "residual"
    micro_px_area += render_residual_fill(canvas, occupancy, theta_s, coherence_s, mask, base, get_font, rng,
                                          tokens=short_tokens, size_field_px=size_px_field)
    _TL.pass_name = "channel"
    ch_n, ch_px = render_channel_fill(canvas, occupancy, theta_s, mask, get_font, letters=letter_tokens,
                                      tone=gray.astype(np.float32) / 255.0, size_cap=size_px_field)
    micro_px_area += ch_px
    _log(f"final fills: channel fill placed {ch_n} letters ({ch_px}px)")
    best_placements = list(_TL.placements)
    best_pass = list(_TL.pass_tags)
    best_stats = dict(_TL.stats)

    # Gap-fill text is fine detail closing bare patches -- counts toward the micro share, same
    # spirit as the micro tier itself ("texture, fine tonal modeling, transitions").
    micro_px_area += fill_px_area
    total_tiered = max(1, micro_px_area + struct_px_area + hero_px_area)
    _log(f"type-scale proportions -- micro: {micro_px_area/total_tiered:.1%}  "
          f"structural: {struct_px_area/total_tiered:.1%}  hero: {hero_px_area/total_tiered:.1%}  "
          f"(targets: 20-30% / 60-70% / 3-7%)")

    # ---- Opacity carries tone, over ALL placed type ---------------------------------------
    # Measured on the black Lab: in the body zone (47% of the likeness weight) the typography
    # panel's tone had r = -0.02 with the photo -- the lanes are sparse there on purpose (dark
    # fur wants little ink) but the residual/channel fills then closed every gap at near-full
    # alpha, leaving uniform texture with no value structure. Scale the finished canvas's alpha
    # by the local photo value (blurred past letter scale) so density AND opacity follow tone.
    # Applied once, after the density loop, which keeps its own error signal unchanged; the
    # composite reads letter presence separately (see ink_soft), so legibility is unaffected.
    tone_blur = _gblur(gray.astype(np.float32) / 255.0, (0, 0), sigmaX=max(2.0, base * 0.15))
    tone_gain = (0.35 + 0.65 * np.clip(tone_blur, 0, 1)).astype(np.float32)
    _r, _g, _b, _a = canvas.split()
    # Letter PRESENCE, captured before the tone modulation: every measurement that means "is
    # there a letter here" (coverage, exposed space, letters-vs-gaps) reads this, not the dimmed
    # alpha -- otherwise a dark-fur letter at 35% opacity counts as empty space (measured: the
    # exposed-space figure jumped to 15.8% on the black Lab the moment tone modulation landed).
    ink_raw = np.asarray(_a, np.float32) / 255.0
    _a = Image.fromarray(np.clip(np.asarray(_a, np.float32) * tone_gain, 0, 255).astype(np.uint8))
    canvas = Image.merge("RGBA", (_r, _g, _b, _a))

    if debug_dir:
        out = Image.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, 255)), canvas).convert("RGB")
        out.save(out_path, quality=92)

    ink_alpha = np.asarray(canvas.split()[3], np.float32) / 255.0
    coverage = float((ink_alpha[mask > 0.5] > 0.12).mean())
    _log(f"ink coverage inside mask: {coverage:.1%}")

    dark_gate = np.clip((0.55 - np.clip(gray.astype(np.float32) / 255.0, 0, 1)) / 0.55, 0, 1)
    dark_gate = _gblur(dark_gate, (0, 0), sigmaX=max(1.0, W * 0.01))
    dk = max(3, int(round(W * 0.006))) | 1
    ink_dilated = cv2.dilate(ink_alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk)))
    ink_alpha = ink_alpha * (1.0 - dark_gate) + np.maximum(ink_alpha, ink_dilated) * dark_gate

    # ---- Real eye reveal, not a synthetic catchlight (recommendation #12, reworked) --------
    # Original approach: find the BRIGHTEST small spot near each attractor point (a proxy
    # catchlight) and hard-zero typography there. Flagged directly as "eyes look like black
    # holes, dead" -- checked against the actual source pixels and found two real bugs: (1) the
    # attractor point sits NEAR the eye but not precisely on the pupil (confirmed by drawing it
    # on the render), so a narrow search box often missed the eye and landed on cheek fur
    # instead; (2) "brightest pixel" is the wrong thing to search for in the first place -- a
    # small specular glint inside a dark iris is often dimmer in raw terms than ordinary sunlit
    # fur nearby, so that search reliably found fur, not the eye, even widened. The real fix
    # doesn't need to locate the catchlight precisely at all: find the pupil instead (the
    # DARKEST compact region near the attractor -- a much more reliable target, since an eye is
    # reliably darker than surrounding fur) and reveal the ACTUAL photo there directly --
    # whatever warm iris color and natural catchlight the source photo genuinely has (confirmed
    # directly: this dog's eye has real amber iris color and a visible glint) -- rather than
    # trying to synthesize a highlight from scratch.
    # Reveal the WHOLE eye, not an 11px disk at its darkest point. Probed at the dog's iris
    # catchlight with correct landmarks: eye_reveal was 0.000 there -- the disk (radius
    # base*0.22) covered the pupil center only, so the iris and glint got whatever reveal the
    # ink happened to give them, and the amber read as near-black. Use the same fitted ellipse
    # render_eye_feature builds the lid from, with the old disk as the fallback when no
    # plausible ellipse is found.
    eye_reveal = np.zeros((H, W), np.float32)
    for (ax, ay) in attractor_pts:
        fit = fit_dark_blob_ellipse(gray, mask, ax, ay, base * 0.9)
        ok_fit = (fit is not None
                  and base * 0.12 < fit[1][0] < base * 2.2 and base * 0.08 < fit[1][1] < base * 2.2
                  and max(fit[1]) / max(min(fit[1]), 1e-3) <= 3.0)
        if ok_fit:
            (ecx, ecy), (a_ax, b_ax), ang = fit
            cv2.ellipse(eye_reveal, (int(round(ecx)), int(round(ecy))),
                       (max(2, int(round(a_ax * 1.05))), max(2, int(round(b_ax * 1.05)))),
                       ang, 0, 360, 1.0, -1)
            continue
        r = int(round(base * 0.55))
        px0, px1 = max(0, int(ax) - r), min(W, int(ax) + r)
        py0, py1 = max(0, int(ay) - r), min(H, int(ay) + r)
        if px1 <= px0 or py1 <= py0:
            continue
        patch = gray[py0:py1, px0:px1].astype(np.float32)
        patch_blur = _gblur(patch, (0, 0), sigmaX=max(1.0, base * 0.03))
        by, bx = np.unravel_index(int(np.argmin(patch_blur)), patch_blur.shape)
        rad = max(3, int(round(base * 0.22)))
        cv2.circle(eye_reveal, (px0 + bx, py0 + by), rad, 1.0, -1)
    eye_reveal = _gblur(eye_reveal, (0, 0), sigmaX=max(1.0, base * 0.06))

    # ---- Real nose reveal -- same idea as eye_reveal, a real gap in the render's fidelity ----
    # The dedicated nose renderer above only claims the outline/groove/nostrils in `occupancy`;
    # the leather's own interior is left to the generic density pass, which doesn't specifically
    # boost reveal there. Checked directly against a render: the cat's actual coral-pink nose
    # leather came through as a flat, nearly colorless patch -- exactly what you'd expect, since
    # nothing was telling the compositor to reveal MORE photo there specifically. Fit the same
    # nose ellipse used by the dedicated renderer and reveal the real leather color within it.
    nose_reveal = np.zeros((H, W), np.float32)
    nose_fit_final = locate_nose(gray, mask, attractor_pts)
    if nose_fit_final is not None:
        (fncx, fncy), (fna, fnb), fnangle = nose_fit_final
        cv2.ellipse(nose_reveal, (int(round(fncx)), int(round(fncy))),
                   (max(1, int(round(fna * 0.95))), max(1, int(round(fnb * 0.95)))),
                   fnangle, 0, 360, 1.0, -1)
    nose_reveal = _gblur(nose_reveal, (0, 0), sigmaX=max(1.0, base * 0.06))

    # ---- Whisker zone -- makes the typographic whiskers from render_muzzle_topology actually
    # visible in the FINAL composite, not just the typography-only panel ------------------------
    # render_muzzle_topology draws its whisker spokes straight into `canvas`, deliberately
    # extending PAST the silhouette the way real whiskers do. But the whole compositing model is
    # `a = ink_alpha * mask * ...` -- outside the mask, `mask` is 0, so that ink's alpha never
    # translates into any visible reveal in the actual delivered image (photo_out), only in the
    # typography-only debug panel. Checked directly: without this fix the whisker spokes are
    # real, structured typography that's completely invisible in the piece anyone would actually
    # look at. Fixed by tracing the SAME spoke geometry into its own zone mask and, below, giving
    # ink there its own reveal against a fixed pale "whisker" color instead of trying to reveal
    # actual photo pixels that don't meaningfully exist for a hair drawn past the animal's edge.
    whisker_zone = np.zeros((H, W), np.float32)
    whisker_region_inside = np.zeros((H, W), np.float32)
    if nose_fit_final is not None and len(attractor_pts) >= 2:
        (wncx, wncy), (wna, wnb), _ = nose_fit_final
        (wx1, wy1), (wx2, wy2) = attractor_pts[0], attractor_pts[1]
        w_eye_sep = max(1.0, math.hypot(wx2 - wx1, wy2 - wy1))
        pad_offset = wna * 0.55
        pad_y = wncy + wnb * 0.6
        whisker_len = w_eye_sep * 1.5
        for side in (-1, 1):
            pad_x = wncx + side * pad_offset
            base_angle = 0 if side > 0 else 180
            for offset_deg in (-28, -16, -6, 6, 16, 28):
                angle_deg = base_angle + offset_deg * (1 if side > 0 else -1)
                theta = math.radians(angle_deg)
                ex = int(round(pad_x + math.cos(theta) * whisker_len))
                ey = int(round(pad_y + math.sin(theta) * whisker_len))
                cv2.line(whisker_zone, (int(round(pad_x)), int(round(pad_y))), (ex, ey),
                        1.0, thickness=max(2, int(round(base * 0.05))))
            # A wider band covering where REAL whiskers physically emerge, for the suppression
            # pass below -- broader than the thin spoke lines above, since a real whisker's base
            # and first stretch is well within the solid mask before it ever crosses the edge.
            # First pass at 0.9x/0.55x eye_sep was measured too small directly: the actual visible
            # whisker strands on the cat photo sweep out much farther than that (confirmed by
            # visualizing the detector's own signal against the source -- real whiskers extend
            # roughly 1.5-2x eye separation from the pad), so most of a whisker's visible length
            # fell outside the suppression band and remained fully visible.
            cv2.ellipse(whisker_region_inside, (int(round(pad_x)), int(round(pad_y))),
                       (int(round(w_eye_sep * 1.8)), int(round(w_eye_sep * 0.9))),
                       0, 0, 360, 1.0, -1)
    # Silhouette fringe hairs share the exact same problem as the whisker spokes -- ink drawn
    # past mask=0 that the standard compositing would otherwise drop -- so they share the same
    # fix: a reveal zone computed deterministically (build_fringe_zone) and merged into the same
    # outside-mask reveal pass below rather than building a second parallel mechanism.
    fringe_zone = build_fringe_zone(mask, theta_s, base, fringe_points)
    whisker_outside = np.maximum(whisker_zone, fringe_zone) * (mask <= 0.5)
    # Probed at the dog's iris catchlight: source L=157, ink_alpha 0.86, but a=0.051 -- this
    # region (scaled by eye separation, 545px on the dog -> a 980x490 ellipse) reached the
    # eyes, and a glint against a dark iris is precisely the thin-line signal the suppression
    # kills. Whiskers grow from the muzzle, below the eyes: clip the region to strictly below
    # the eye line so it can never touch an eye again, whatever the photo's proportions.
    if len(attractor_pts) >= 2:
        (ex1, ey1), (ex2, ey2) = attractor_pts[0], attractor_pts[1]
        eye_line_y = 0.5 * (ey1 + ey2)
        eye_sep_w = max(1.0, math.hypot(ex2 - ex1, ey2 - ey1))
        whisker_region_inside *= (yy > eye_line_y + 0.18 * eye_sep_w).astype(np.float32)
    whisker_region_inside = _gblur(whisker_region_inside, (0, 0), sigmaX=max(1.0, base * 0.1)) * mask

    # ---- Saturation-anomaly dampening (the open-mouth/tongue fix) --------------------------
    # A real defect found on the Border Collie render: a hero word happened to land right over
    # the open mouth, and because this compositing model reveals MORE of the actual photo
    # wherever ink is denser (a = ink_alpha * ..., composited = ground*(1-a) + photo*a), a
    # solid hero word sitting on the tongue became a vividly pink cutout -- jarring against the
    # otherwise muted, mostly-desaturated palette everywhere else on the animal. Fixed the same
    # way the eye catchlight is protected: detect the anomaly (here, saturation spiking well
    # above the LOCAL neighborhood's own saturation, not a fixed global threshold -- portable
    # across light and dark subjects) and dampen how much of the photo shows through there,
    # rather than special-casing "tongues." Any comparably saturated small anomaly gets the same
    # treatment.
    # Two real bugs found and fixed while tuning this against the ACTUAL tongue pixels, not
    # assumed: (1) a plain (unmasked) Gaussian blur for the "neighborhood average" leaks in
    # whatever is OUTSIDE the mask too (grass, background), and that leakage got WORSE, not
    # better, as the blur radius grew -- background saturation crept into the reference,
    # shrinking the measured anomaly right when a bigger radius was meant to fix it. Switched to
    # a properly mask-normalized blur (blur the masked values AND the mask itself, divide) so the
    # neighborhood average only ever reflects the subject's own fur. (2) The tongue itself has a
    # saturation GRADIENT (a vivid core fading into blended edges), so even the mask-normalized
    # version only averaged ~0.26-0.33 anomaly across the whole tongue despite peaking at 1.0 in
    # its core -- the first threshold/divisor pair (0.10 / 0.22) left the softer edges barely
    # dampened. Both tightened until the tongue's OWN measured average maps to a real reduction.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    mf = (mask > 0.5).astype(np.float32)
    sigma = max(1.0, W * 0.10)
    num = _gblur(sat * mf, (0, 0), sigmaX=sigma)
    den = _gblur(mf, (0, 0), sigmaX=sigma)
    broad_sat = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
    sat_anomaly = np.clip((sat - broad_sat - 0.05) / 0.10, 0, 1) * mf
    sat_anomaly = _gblur(sat_anomaly, (0, 0), sigmaX=max(1.0, base * 0.05))

    # `feat` was already computed once, up front, and reused for the attractor field above --
    # no need to recompute it here.
    # Lowered from 0.45 -- flagged directly as "eyes look like black holes, dead." This dampens
    # `a`, which controls how much of the ACTUAL photo shows through; suppressing it near the
    # eyes was hiding the real iris/pupil color and detail, not just thinning typography over
    # it (that protection belongs to the density fields elsewhere, not to hiding the photo).
    a = np.clip(ink_alpha * mask * (1.0 - 0.15 * feat) * (1.0 - 0.95 * sat_anomaly), 0, 1)
    # Boost reveal toward 1 within the real eye disk found above, so the actual iris color and
    # any natural catchlight show clearly regardless of how much ink happens to land there --
    # additive toward full reveal, not an override, so it still respects genuinely dense ink.
    a = np.clip(a + eye_reveal * (1.0 - a) * 0.92, 0, 1)
    # Same treatment for the nose leather -- slightly less than full reveal (0.85 vs 0.92) so a
    # trace of typographic texture still reads on top of it, unlike the eye where the priority
    # was eliminating a literal black hole.
    a = np.clip(a + nose_reveal * (1.0 - a) * 0.85, 0, 1)
    # ---- Photo wash: continuous faint color in the negative space, not pure paper -----------
    # The real ceiling on "muted fur": this technique only ever reveals the photo through thin
    # letter strokes, so even fully-saturated ink only covers ~40-45% of the silhouette at
    # legible density -- the rest is pure ground color, and the eye averages that in regardless
    # of how vivid the strokes themselves are. No amount of boosting the STROKE color fixes
    # that; the gaps themselves need to carry some of the real photo's color. Guarantee a
    # floor on `a` everywhere inside the mask (not an override -- `a` only goes UP where this
    # floor exceeds it, so real ink-driven reveal in dense areas is untouched) rather than
    # letting bare paper show at zero. Multiplied by the SAME anomaly suppression as the ink
    # reveal above, so this can't reopen the tongue-color bleed that fix was for -- a continuous
    # wash of a genuinely wrong color would be worse than the gap it's filling.
    # A FLAT 0.30 floor everywhere was the actual cause of "washed out": it lifts genuinely dark
    # regions -- the pupil, nostril, shadow creases -- toward showing SOME reveal even where the
    # real photo is supposed to read as deep shadow, flattening exactly the contrast that gives
    # a face its depth. Scaled by the photo's own brightness instead (dark photo -> little to no
    # wash, preserving real shadow; bright/mid fur -> a healthy wash, keeping the color-
    # continuity win) -- this fixes the mechanism, not just the symptom, so dropping the whole
    # thing isn't necessary: dark areas were never the ones that looked muted in the first place.
    # Flagged directly as a "gauzy haze over the majority of the image." Root cause: these
    # reference photos are deliberately bright/high-key (per the user's own reference shots), so
    # `brightness` is high across MOST of the silhouette, not just a few highlights -- meaning the
    # 0.40 peak floor was landing at ~0.30-0.38 almost everywhere, not just where genuinely needed.
    # With ink coverage around 40-45%, that means 60-70% of most pixels were the flat neutral
    # ground color bleeding through as a near-uniform veil -- exactly a haze, and exactly why it
    # didn't look like the "healthy wash in bright fur, little wash in shadow" it was designed to
    # be: on a bright photo there's barely any shadow left to tell the two cases apart. Cut the
    # peak substantially so the floor stays a light tint rather than a dominant blend -- letters
    # and dense typographic areas still carry the real ink-driven reveal untouched (this is only
    # a floor), but genuine gaps read as gaps again instead of a wash of ground color.
    # Down from 0.40: with the gap fill now a darkened copy of the photo (structure already
    # present in the gaps), a high wash floor only pulls gaps back up toward the letters and
    # erases the letter/gap distinction the typography depends on to be seen.
    wash_strength = 0.15
    brightness = np.clip(gray.astype(np.float32) / 255.0, 0, 1)
    wash = wash_strength * mask * brightness * (1.0 - 0.95 * sat_anomaly)
    a = np.clip(np.maximum(a, wash), 0, 1)

    # ---- Suppress REAL photographic whiskers within the mask ------------------------------
    # "Whiskers should also be typography... a high-end portrait should eventually have no
    # photographic whiskers." Even where mask=1, the wash/ink-reveal above naturally reveals
    # whatever's genuinely in the photo there -- including a cat's actual whisker hairs, thin
    # and high-contrast against duller surrounding fur. Confirmed directly: a render's cheek
    # showed real smooth grey whisker strands sitting on top of the typography once wash was
    # strong enough to reveal fur color at all. A median blur erases thin line structures like a
    # whisker while leaving broader fur texture alone, so the residual (original minus
    # median-blurred) isolates whisker-like anomalies specifically. Restricted to the muzzle/
    # cheek band computed above so this can't accidentally suppress unrelated fine detail
    # elsewhere on the animal.
    median_local = cv2.medianBlur(gray, 9).astype(np.float32)
    whisker_signal = np.abs(gray.astype(np.float32) - median_local)
    whisker_suppress = np.clip(whisker_signal / 18.0, 0, 1) * whisker_region_inside
    a = a * (1.0 - 0.92 * whisker_suppress)
    a = a[..., None]
    # ---- Ground color derived from the REAL photo background, not an arbitrary constant ----
    # This was a fixed navy-purple (26, 20, 40) regardless of what was actually behind the pet
    # -- grass, a wall, sky, whatever. "Truer colors" applies to the ground too: sample the
    # actual background pixels (masked-average blur, same normalization trick as sat_anomaly
    # above, so mask-interior pixels don't corrupt the average), then darken and desaturate so
    # it stays a recessive backdrop rather than a literal sharp photo -- the point is the HUE
    # now genuinely reflects the real scene (muted green for grass, muted warm gray for an
    # indoor wall, etc.), not that the background becomes a competing photographic element.
    # cv2.GaussianBlur silently squeezes a (H,W,1) array back down to (H,W) -- blur the 2D
    # weight map and re-add the channel axis explicitly rather than relying on it surviving.
    bg_weight2d = (mask <= 0.5).astype(np.float32)
    bg_sigma = max(20.0, W * 0.25)
    bg_num = _gblur(bgr_source.astype(np.float32) * bg_weight2d[..., None], (0, 0), sigmaX=bg_sigma)
    bg_den = _gblur(bg_weight2d, (0, 0), sigmaX=bg_sigma)[..., None]
    ground_bgr = np.divide(bg_num, bg_den, out=np.full_like(bg_num, 30.0), where=bg_den > 1e-6)
    ground_hsv = cv2.cvtColor(np.clip(ground_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    # Pushed much further -- the user supplied the actual source photos side by side with our
    # renders and the gap was obvious: these are bright, evenly, naturally lit photos (a sunlit
    # garden, a sunlit window, a light studio backdrop), and every version of this ground so far
    # has been some shade of dim. Brightened past the raw sampled average, not just toward it --
    # a blurred garden or a bright wall in real light reads brighter than its own pixel average
    # once vignetting/shadow falloff is removed, which is exactly what a real print of these
    # would look like.
    ground_hsv[..., 1] *= 0.80
    ground_hsv[..., 2] = np.clip(ground_hsv[..., 2] * 1.05, 0, 255)
    ground_bgr = cv2.cvtColor(np.clip(ground_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    outer_ground_rgb = cv2.cvtColor(ground_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    if backdrop_rgb is not None:
        # The customer chose a backdrop on the site. Honoring it here (and not in the gaps
        # between letters, which stay coat-derived) is what makes the choice do anything on
        # this engine: previously it was accepted and ignored, so switching to Gallery Gray
        # re-rendered 40 s of identical pixels.
        outer_ground_rgb = np.full((H, W, 3), np.asarray(backdrop_rgb, np.float32), np.float32)
    # SEGMENTATION FIX (still applies): one ground field can't serve both "outside the animal"
    # and "the gap between two letters ON the animal" -- they need different colors. But the
    # INNER one was set to a near-black tone for contrast, and with collision tolerances now
    # tightened for legibility, MORE of the subject falls into that gap (measured: coverage
    # dropped to 43-47%) -- so a large minority of the animal was rendering as near-black no
    # matter how bright the revealed ink areas got. Against these bright reference photos, that
    # reads as "too dark" overall even where the actual ink is vivid. Lightened substantially --
    # still a warm, slightly-recessive neutral so ink strokes read as the darker, crisper
    # element (the actual drawing), not the reverse, just nowhere near black.
    # First pass at this (150,132,112) overshot -- the render came back readable but flat,
    # washing out some of the eye/nose contrast the earlier eye-reveal fix had just recovered.
    # Backed off toward a middle ground: bright enough that it doesn't read as "dark," but not
    # so light that it competes with and flattens the actual revealed ink detail.
    # Still flagged as a "gauzy haze over the majority of the image" even after tuning the wash
    # floor. Root cause found by checking the actual pixel math: with ink coverage around 40-45%,
    # this flat (108,92,76) grey-brown is the DOMINANT color of most of the image regardless of
    # wash strength -- it's what's behind every gap. That's a fixed neutral, chosen once and used
    # for every pet regardless of actual coloring, so on a light cream doodle (or any coat whose
    # real color isn't already close to warm grey-brown) it reads as exactly what it is: a grey
    # film sitting over the animal's real color instead of a rest color the pet's own coat is
    # organically at rest. Fixed the same way outer_ground was fixed earlier -- derive it from the
    # REAL subject instead of a constant: a masked, heavily blurred average of the pet's own fur
    # (which also means it naturally varies across the coat -- lighter where the coat is lit,
    # cooler in shadow -- instead of being one dead-flat tone everywhere), then desaturated and
    # dimmed enough that ink strokes still read as the crisper, darker foreground element.
    # Measured directly why the haze complaint persisted even after this fix and the alpha-
    # ceiling fix: a generic fur patch's local grayscale contrast (std) was 18 in the render vs
    # 38 in the source -- HALF the real tonal variation, gone. Root cause: fur_sigma here was
    # W*0.12 (~120px on a 1000px-wide photo) -- far beyond "smooth away letter-level noise," it
    # averaged out essentially ALL of the fur's own broad light/shadow structure (a lit cheek vs
    # a shadowed ear) into one near-flat gradient, and that flat gradient is what shows in every
    # gap between letters across roughly half the image. A single flat fill under half the
    # picture reads as a haze/veil sitting over the real detail, regardless of how correct its hue
    # is. Cut drastically so this base layer keeps the photo's own broad tonal structure instead
    # of erasing it -- still blurred enough to hide letter-scale noise, not the whole face's shape.
    # Design decision, made explicit after the tonal fixes landed: once the letters revealed the
    # true photo AND the gaps showed a blurred near-copy of it, the composite converged on a
    # soft-focus photograph with faint text -- correct tone, no typography. The gap fill must be
    # the same picture at a clearly LOWER tone, so every letterform is brighter than its
    # surroundings and the image is legibly built of words while still carrying the photo's
    # structure in both layers. Lightly blurred only (letter-scale noise), not averaged away.
    fur_weight2d = (mask > 0.5).astype(np.float32)
    # ---- Edge decontamination: take the background back out of the silhouette band --------
    # A matte is never exact. Along the outline the photo's pixels are part fur, part whatever
    # was behind it: grass, sky, a sunlit rim. Revealed through the letters, and averaged into
    # the gap colour, that band rendered as a glow traced round the whole animal, green on the
    # collie, white-gold on the backlit shepherd (staging baseline dc242ea). Measured on a
    # loosened matte of the tan dog: the inner ring's chroma sat 9.2 units off the deep fur,
    # 40% of the way to the grass. Each pixel within 0.35 base of the edge is projected onto
    # the line from the local deep-fur colour F to the local background colour B, and the
    # background component is removed in proportion to how close to the edge it sits.
    _hard = (mask > 0.5).astype(np.float32)
    _d_in = cv2.distanceTransform(_hard.astype(np.uint8), cv2.DIST_L2, 5)
    # "Deep" fur starts a full base inside the outline when the subject is large enough to
    # allow it, half a base otherwise: the reference colour must not itself be contaminated.
    _mask_deep = (_d_in > base * 1.0).astype(np.float32)
    if float(_mask_deep.sum()) < 0.05 * float(_hard.sum()):
        _mask_deep = (_d_in > base * 0.5).astype(np.float32)
    if float(_mask_deep.sum()) < 100:          # a tiny subject: nothing deep enough to erode to
        _mask_deep = _hard
    _dec_sigma = max(4.0, base * 1.0)
    _src_f = bgr_source.astype(np.float32)
    _F = np.divide(_gblur(_src_f * _mask_deep[..., None], (0, 0), sigmaX=_dec_sigma),
                   _gblur(_mask_deep, (0, 0), sigmaX=_dec_sigma)[..., None] + 1e-4)
    _bgw = (mask <= 0.5).astype(np.float32)
    _B = np.divide(_gblur(_src_f * _bgw[..., None], (0, 0), sigmaX=_dec_sigma),
                   _gblur(_bgw, (0, 0), sigmaX=_dec_sigma)[..., None] + 1e-4)
    _BF = _B - _F
    # The background fraction is judged in a chroma-weighted LAB (L at 0.3): a plain RGB
    # projection read every LIGHTER patch of fur -- the white chin, the lit chest -- as
    # background, because backgrounds are usually brighter than deep fur (measured: 0.73
    # "background" in the cat's outer band with a tight matte). Grass against tan, sky
    # against white, are chroma differences; light fur against dark fur is not.
    # ... and the correction is applied to chroma ONLY. Subtracting the background vector in
    # RGB turned the collie's white paws violet: grass is brighter than white fur in green
    # alone, so "white minus (grass minus fur)" loses green and keeps magenta (staging,
    # 0322f32). Brightness is left as the photo has it; a sunlit rim stays a sunlit rim.
    def _lab(a):
        return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    _Pl, _Fl, _Bl = _lab(_src_f), _lab(_F), _lab(_B)
    _BFab = _Bl[..., 1:] - _Fl[..., 1:]
    _BF2 = np.maximum((_BFab * _BFab).sum(-1), 1e-6)
    _t = np.clip(((_Pl[..., 1:] - _Fl[..., 1:]) * _BFab).sum(-1) / _BF2, 0, 1)
    _t = np.where(np.sqrt(_BF2) < 6.0, 0.0, _t)       # background and fur alike here: nothing to remove
    # How deep does the contamination reach? Walk inward in 2px shells until the mean
    # background fraction falls under 0.12, then fade the correction out over that depth
    # (never less than 0.35 base, never more than 1.2). A fixed 0.35 base left a loosened
    # matte at 5.3 chroma units of drift; a matte twice as loose at 9.6 -- the band has to
    # be as wide as the matte is wrong, which only the photo can say.
    _depth = base * 0.35
    _dmax = base * 1.2
    _dd = 0.0
    while _dd < _dmax:
        _shell = (_d_in > _dd) & (_d_in <= _dd + 2.0)
        if _shell.sum() < 50 or float(_t[_shell].mean()) < 0.12:
            break
        _dd += 2.0
    _depth = float(np.clip(_dd + base * 0.15, base * 0.35, _dmax))
    # Full strength through the measured depth, then a short fade: a fade across the whole
    # band left the middle of it half-corrected (loose matte: 4.3 chroma units of drift).
    _band = np.clip(1.0 - (_d_in - _dd) / max(1.0, base * 0.2), 0, 1) * _hard
    _log(f"edge decontamination: background reaches {_dd:.0f}px in; band {_depth:.0f}px "
         f"(mean bg fraction in outer 0.2 base: {float(_t[(_d_in > 0) & (_d_in <= base * 0.2)].mean()):.2f})")
    # bgr_clean feeds the gap colour and the revealed photo. bgr_source stays the untouched
    # photo for the likeness score, so the score keeps one reference across builds.
    _Pl[..., 1:] -= (_band * _t)[..., None] * _BFab
    bgr_clean = cv2.cvtColor(np.clip(_Pl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    deep_fur_rgb = _F[..., ::-1].astype(np.float32)   # RGB, for the fringe hairs below

    fur_sigma = max(2.0, W * 0.006)
    fur_num = _gblur(bgr_clean.astype(np.float32) * fur_weight2d[..., None], (0, 0), sigmaX=fur_sigma)
    fur_den = _gblur(fur_weight2d, (0, 0), sigmaX=fur_sigma)[..., None]
    fur_bgr = np.divide(fur_num, fur_den, out=np.full_like(fur_num, 120.0), where=fur_den > 1e-6)
    fur_hsv = cv2.cvtColor(np.clip(fur_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    # Measured (dog, inside mask, LAB L percentiles): source 5th pct = 30, render = 56; source
    # 95th = 209, render = 191; median saturation 102 vs 72. Those two lines below were the direct
    # cause: a hard clip to [60, 200] on a layer behind ~half of every pixel forbids true blacks
    # and true whites outright, and sat*0.45 is the desaturation. Clamp removed, desaturation
    # eased; the real enforcement of the source's tonal range is the distribution match at the
    # end of the composite (see "tonal match" below), which this no longer fights.
    fur_hsv[..., 1] *= 0.85
    fur_hsv[..., 2] = np.clip(fur_hsv[..., 2] * 0.40, 0, 255)   # same picture, well below the letters
    inner_ground_bgr = cv2.cvtColor(np.clip(fur_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    inner_ground_rgb = cv2.cvtColor(inner_ground_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    ground_rgb = inner_ground_rgb * mask[..., None] + outer_ground_rgb * (1.0 - mask[..., None])
    # Vividness boost for the pet's own revealed colors -- flagged directly as "muted," and the
    # side-by-side reference photos confirmed it again at a brightness level, not just
    # saturation. Boost both explicitly, after all the tonal logic is settled, so this doesn't
    # fight the density correction above -- it only changes color vividness, not how much
    # ink/reveal there is. Brightness eased back from 1.38 (part of the same overshoot as the
    # ground above) to keep real contrast in the fur rather than flattening it.
    # Flagged directly: the doodle's nose leather (near-black with a faint warm undertone in the
    # source) rendered RED, not dark brown. Root cause -- a flat 1.55x saturation multiplier
    # applied everywhere, including near-black pixels where the nose_reveal fix above (added this
    # session) now shows close to the FULL boosted color instead of a ground-diluted blend. A
    # small saturation value on a near-zero-value pixel is barely visible normally, but boosted
    # 1.55x and then shown at ~90% reveal, that faint warm undertone becomes a dominant, visibly
    # red hue -- an artifact of the boost math, not a real color in the photo. Scale the
    # saturation boost by the pixel's own brightness (matches the wash fix's reasoning below):
    # bright fur gets the full 1.55x vividness push, while near-black leather/shadow/pupils keep
    # close to their real, mostly-neutral saturation instead of having a boosted color invented.
    # Measured directly against the doodle's actual nose pixels: the source averages S=97/255
    # (a real but modest warm brown, V=105), while `1.0 + 0.55*V` still pushed it to S=168 at
    # this V -- more than 1.7x -- because _enhance_contrast has ALREADY boosted saturation
    # upstream, so this multiplier was compounding on top of that, not starting from the raw
    # photo. A dark, desaturated warm hue reads as "brown"; the same hue boosted to high
    # saturation reads as "red" even though the hue angle barely moved -- that compounding is
    # exactly what turned the nose red. Steepened so dark pixels land near a NEUTRAL multiplier
    # (~1.0, preserving whatever _enhance_contrast already did) instead of still gaining nearly
    # 20%, while bright fur still reaches the full 1.55x vividness that fixed "muted" earlier.
    # All of the per-pixel color hacks that used to live here (contrast-enhanced source,
    # brightness-scaled saturation multiplier, V*1.22) are gone. Measured on the dog's iris:
    # source saturation 110, this pipeline's output 73 -- the "don't over-saturate dark pixels"
    # formula added for the doodle's nose was DESATURATING every dark saturated pixel, and an
    # amber iris is exactly that. With the end-of-composite distribution match now forcing the
    # output's luminance and saturation range onto the source's, there is no reason left to
    # pre-distort the colors at all: use the real photo, and let the match enforce the range.
    photo_rgb = cv2.cvtColor(bgr_clean, cv2.COLOR_BGR2RGB).astype(np.float32)
    composited = ground_rgb * (1.0 - a) + photo_rgb * a

    # Per-pixel stage probe: GOP_DEBUG_PT="x,y" prints each compositing stage's value at that
    # pixel, so a lost detail (a catchlight, a highlight) can be traced to the exact stage that
    # loses it instead of being guessed at from the final image.
    _dbg_pt = os.environ.get("GOP_DEBUG_PT")
    _dbg_pt = tuple(int(v) for v in _dbg_pt.split(",")) if _dbg_pt else None

    def _dbg(label, rgb):
        if _dbg_pt is None:
            return
        x, y = _dbg_pt
        px = np.clip(rgb[y:y + 1, x:x + 1], 0, 255).astype(np.uint8)
        Lv = int(cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0, 0])
        _log(f"  [probe {x},{y}] {label:<28s} L={Lv:3d}  rgb={tuple(int(v) for v in rgb[y, x])}")

    if _dbg_pt is not None:
        x, y = _dbg_pt
        _log(f"  [probe {x},{y}] a={float(a[y, x, 0]):.3f} ink_alpha={float(ink_alpha[y, x]):.3f} "
              f"eye_reveal={float(eye_reveal[y, x]):.3f} feat={float(feat[y, x]):.3f} "
              f"wash={float(wash[y, x]):.3f} sat_anom={float(sat_anomaly[y, x]):.3f}")
        _dbg("photo_rgb (source)", photo_rgb)
        _dbg("ground_rgb", ground_rgb)
        _dbg("composited (pre-match)", composited)

    # Reveal the whisker typography directly against a fixed pale "whisker" color rather than
    # through the mask-gated photo-reveal machinery -- there's no real photo pixel to reveal for
    # a hair drawn past the animal's own edge, so this gives it a color of its own instead.
    whisker_ink_alpha = ink_alpha * whisker_outside
    whisker_color = np.array([222.0, 218.0, 205.0], np.float32)
    # Silhouette fringe hairs used the same fixed pale as the whiskers. Against a dark
    # backdrop that reads as hair catching light; against Gallery Gray it read as a pale
    # outline traced around the animal. The fringe takes the coat's own colour so a brown dog
    # sheds brown hairs on any backdrop. Real whiskers (whisker_zone, cats) stay pale: they are.
    # The colour comes from DEEP fur (the matte eroded by half a base), not the edge band:
    # the edge average carried the background in, and the 10% lift on top of that made the
    # hairs a glow on every dark backdrop. Hair is the coat's own colour, no brighter.
    fur_edge_rgb = deep_fur_rgb
    _wz = np.clip(whisker_zone, 0, 1)[..., None]
    hair_color = whisker_color * _wz + fur_edge_rgb * (1.0 - _wz)
    composited = composited * (1.0 - whisker_ink_alpha[..., None]) + hair_color * whisker_ink_alpha[..., None]

    # edge_ink used to add a dark stroke at every strong internal edge -- a reasonable idea for a
    # moody, dark-ground piece, but against a brightened subject it was one more thing pulling
    # the average tone down. Lightened the color and roughly halved the blend strength.
    edge = (_edge_ink(gray.astype(np.uint8)) * mask)[..., None]
    edge_ink_color = np.array([70.0, 62.0, 52.0], np.float32)
    # Removed (ek was 0.30). Measured on the dog's eye: the catchlight is L=196 in the source and
    # survives the color pipeline at ~185, but a bright glint inside a dark iris is the strongest
    # internal edge in the image, so this blend pulled it 30% toward (70,62,52) -> predicted ~148,
    # measured 150. It was doing the same to every bright fine detail. Its purpose (contrast) is
    # now handled by the calibrated local-contrast step and the distribution match below.
    ek = 0.0

    # ---- Tonal match: force the composite's luminance/saturation distribution INSIDE the mask
    # to equal the source photo's, by quantile mapping ------------------------------------------
    # Every earlier haze fix tuned a constant toward the source and then eyeballed it. Measured
    # afterward, the render was still a tonal compressor: (dog, LAB L inside mask) 5th pct 56 vs
    # source 30, 95th pct 191 vs 209, median saturation 72 vs 102. Rather than keep guessing
    # constants, enforce the target directly: map the composite's L (and HSV S) so that its
    # quantiles inside the mask land exactly on the source's quantiles inside the mask. The map is
    # monotonic, so nothing structural changes -- a letter that was darker than its gap is still
    # darker than its gap -- but true blacks, true highlights, and the real saturation range come
    # back by construction, and the result is verified by re-measuring, not by looking.
    def _quantile_match(vals, ref_vals, x, n=256):
        q = np.linspace(0.0, 1.0, n)
        return np.interp(x, np.quantile(vals, q), np.quantile(ref_vals, q))

    def _local_std(L, m, blk=32):
        Hh, Ww = L.shape
        vals = [L[y:y + blk, x:x + blk].std()
                for y in range(0, Hh - blk, blk) for x in range(0, Ww - blk, blk)
                if m[y:y + blk, x:x + blk].mean() > 0.95]
        return float(np.mean(vals)) if vals else 0.0

    m_in = mask > 0.5
    src_lab = cv2.cvtColor(cv2.cvtColor(bgr_source, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2LAB).astype(np.float32)
    src_hsv = cv2.cvtColor(bgr_source, cv2.COLOR_BGR2HSV).astype(np.float32)
    comp_lab = cv2.cvtColor(np.clip(composited, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    L_before = comp_lab[..., 0][m_in].astype(np.float32)

    # Step 1 -- local (hair-scale) contrast, calibrated to the source's own measured value.
    # Measured as the mean L std over 32px blocks inside the mask: source 19.9, render 11.8 once
    # the raw photo colors are used. CLAHE recovers it but overshoots at any fixed setting (it
    # amplifies the letter/gap alternation into harshness), so instead of picking a strength,
    # blend toward a deliberately-strong CLAHE by exactly the fraction that lands the measured
    # local std on the source's: k = (target - current) / (clahe - current), clamped to [0, 1].
    # Runs BEFORE the distribution match so the match has the final say on the tonal range.
    L_cur = comp_lab[..., 0].astype(np.float32)
    target_ls = _local_std(src_lab[..., 0], m_in)
    cur_ls = _local_std(L_cur, m_in)
    # Escalate the CLAHE base until the target is actually reachable (a fixed base of 2.5 left
    # the blend clamped at k=1.0 and still 4 points short), then solve for the exact blend.
    clahe_clip, L_clahe, clahe_ls = None, L_cur, cur_ls
    for clip in (2.5, 4.0, 6.0, 9.0, 14.0):
        cand = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(comp_lab[..., 0]).astype(np.float32)
        cand_ls = _local_std(cand, m_in)
        clahe_clip, L_clahe, clahe_ls = clip, cand, cand_ls
        if cand_ls >= target_ls:
            break
    # Solve k against the POST-match result, not the pre-match blend: measured, calibrating
    # pre-match landed at 19.9 and the quantile match then pulled it to 16.6 (the map compresses
    # wherever the render's histogram is denser than the source's). Bisection on k over the
    # full blend -> L-match chain, so the number that's checked is the number that ships.
    src_L_in = src_lab[..., 0][m_in]
    # The UNdilated ink: `ink_alpha` was widened in dark regions earlier (dark_gate), which is
    # right for reveal but wrong for "is this pixel a letter" -- measured: with the dilated
    # mask the letter/gap delta read +1 while an external check with the typography panel
    # read +28. Use the canvas's own alpha for anything that means "letter vs gap."
    # ink_raw (pre-modulation presence) was captured above, before the tone gain was applied.
    # Presence, not opacity: any letter with alpha >= ~64 counts fully as a letter here, so the
    # tone modulation applied to the canvas above (dim type where the photo is dark) can't
    # weaken the letter/gap separation in the composite. Antialiased edges stay soft.
    ink_soft = np.clip(ink_raw / 0.25, 0, 1)
    letters_in = m_in & (ink_raw > 0.5)
    # Design parameter, explicit: gaps sit at this fraction of the letter tone at the same
    # spot. Letters are matched to the SOURCE (they carry its exact tonal range and color);
    # gaps are the same picture pulled down by this factor so every letterform is brighter
    # than its surroundings and the portrait is legibly made of words. A typographic portrait
    # with visible words cannot share the photo's global histogram -- this is where the two
    # goals are reconciled on purpose rather than fought over by a global match.
    GAP_TONE = 0.66      # dark/mid coats: letters carry the source tone, gaps sit at this fraction
    LETTER_TONE = 0.72   # light coats: gaps carry the source tone, letters sit at this fraction
    # Coat-aware: a fixed GAP_TONE made a cream doodle read as a brown dog (69% of its area is
    # gap). On a light coat the honest typographic look is dark words on the true light coat,
    # not light words on a darkened one. `light_mix` moves continuously between the two by the
    # source's own mean brightness inside the mask: ~0 for the dog/cat (mean L ~140-150),
    # ~1 for the doodle (~185). Either way the letter/gap separation is guaranteed.
    src_mean_L = float(src_L_in.mean())
    light_mix = float(np.clip((src_mean_L - 145.0) / 30.0, 0.0, 1.0))
    # Dark coats (chocolate/black; mean L under ~95): the mirror of the light-coat rule. Seen on
    # staging with a chocolate Lab: letters at the coat's own dark tone with gaps pushed darker
    # still is a dark photo with slightly-less-dark text on it. The honest typographic read on a
    # dark coat is LIGHT words on the true dark fur -- gaps carry the source, letters are lifted.
    # Ramp 120 -> 90: the staging chocolate Lab measured mean L 98 (dark_mix 0.73 here), the
    # black Lab test photo 45 (1.0); the tan dog (147) and tabby (141) stay out of it (0.0).
    dark_mix = float(np.clip((120.0 - src_mean_L) / 30.0, 0.0, 1.0))
    gaps_in = m_in & (ink_raw < 0.1)
    # Solve the two layer scales from two stated targets instead of fixed constants:
    #   fidelity  -- composite mean L inside the mask = TONE_FIDELITY x source mean L
    #                (measured cost of visible words; 0.66 fixed gave 0.76 on the dog, 0.85 on
    #                the doodle -- the rule makes it the same everywhere), and
    #   legibility -- |letter mean - gap mean| >= MIN_DELTA_L, which wins if the two conflict.
    # The layer carrying the source tone stays at 1.0; the other is solved from the letter
    # coverage; the two modes blend by light_mix. GAP_TONE/LETTER_TONE above are now the floors.
    TONE_FIDELITY, MIN_DELTA_L = 0.85, 40.0   # delta 29 read too photographic on the dog; 73 too dark
    c_eff = float(ink_soft[m_in].mean())                               # soft letter coverage
    # Solve from the MEASURED unscaled layer means (m_l, m_g), not from the assumption that a
    # matched layer's mean equals the source's -- it doesn't (measured ~139 vs 147 on the dog,
    # which left the delta at 31 against a 40 floor when solved analytically).
    _ref_l0 = L_cur[letters_in] if letters_in.sum() > 1000 else L_cur[m_in]
    _ref_g0 = L_cur[gaps_in] if gaps_in.sum() > 1000 else L_cur[m_in]
    m_l = float(_quantile_match(_ref_l0, src_L_in, L_cur)[letters_in].mean()) if letters_in.any() else src_mean_L
    m_g = float(_quantile_match(_ref_g0, src_L_in, L_cur)[gaps_in].mean()) if gaps_in.any() else src_mean_L
    target_mean = TONE_FIDELITY * src_mean_L
    # dark mode: letters at 1.0, gaps scaled -- fidelity target, then legibility floor wins
    g_dark = (target_mean - c_eff * m_l) / max(1e-6, (1.0 - c_eff) * m_g)
    g_dark = float(np.clip(min(g_dark, (m_l - MIN_DELTA_L) / max(1.0, m_g)), GAP_TONE, 1.0))
    # light mode: gaps at 1.0, letters scaled
    l_light = (target_mean - (1.0 - c_eff) * m_g) / max(1e-6, c_eff * m_l)
    l_light = float(np.clip(min(l_light, (m_g - MIN_DELTA_L) / max(1.0, m_l)), LETTER_TONE, 1.0))
    gap_scale = (1.0 - light_mix) * g_dark + light_mix * 1.0
    letter_scale = (1.0 - light_mix) * 1.0 + light_mix * l_light
    # Dark-coat blend: gaps -> source tone, letters lifted to clear the floor (capped so a
    # near-black coat can't turn its words white).
    if dark_mix > 0:
        l_dark = float(np.clip((m_g + MIN_DELTA_L) / max(1.0, m_l), 1.0, 2.2))
        gap_scale = (1.0 - dark_mix) * gap_scale + dark_mix * 1.0
        letter_scale = (1.0 - dark_mix) * letter_scale + dark_mix * l_dark
    # Enforce the legibility floor on the BLENDED scales (blending the two modes pulls the
    # layers back toward each other -- measured delta 31 against a 40 floor). Push the offset
    # layer of whichever mode dominates until the predicted delta meets the floor.
    pred_delta = m_l * letter_scale - m_g * gap_scale
    if abs(pred_delta) < MIN_DELTA_L:
        if dark_mix >= 0.5:
            letter_scale = min(2.2, (m_g * gap_scale + MIN_DELTA_L) / max(1.0, m_l))
        elif light_mix < 0.5:
            gap_scale = max(GAP_TONE, (m_l * letter_scale - MIN_DELTA_L) / max(1.0, m_g))
        else:
            letter_scale = max(LETTER_TONE, (m_g * gap_scale - MIN_DELTA_L) / max(1.0, m_l))

    def _chain(kk):
        Lw = (L_cur + kk * (L_clahe - L_cur)) * mask + L_cur * (1.0 - mask)
        # Each layer is matched to the source INDEPENDENTLY (a map built from gap pixels applied
        # to letter pixels sent the letters above the source range -- measured: delta collapsed
        # to +6 on the doodle). With both layers on the source's own range, the two scales set
        # the separation exactly: delta = source mean x (letter_scale - gap_scale).
        ref_l = Lw[letters_in] if letters_in.sum() > 1000 else Lw[m_in]
        ref_g = Lw[gaps_in] if gaps_in.sum() > 1000 else Lw[m_in]
        Lm_letters = _quantile_match(ref_l, src_L_in, Lw) * letter_scale
        Lm_gaps = _quantile_match(ref_g, src_L_in, Lw) * gap_scale
        # Absolute local floor on top of the ratio: a x0.70 step is 18 L in an iris at L~60 --
        # invisible -- which is exactly why eyes and nostrils still read as photo at 2x while the
        # fur around them reads as words (measured on the crop). Words in shadow need a fixed
        # minimum separation, not a proportional one.
        # Compared against the tone of the NEARBY letters (normalized blur of the letter layer
        # over a word-sized neighborhood), not the letter map evaluated at the gap pixel itself --
        # that value is systematically low at gap pixels and made the floor bind everywhere
        # (measured: whole-mask delta 37 -> 57, gaps L 98 -> 81), not just in shadow.
        LOCAL_FLOOR = 24.0
        _sig = max(2.0, base * 0.25)
        if dark_mix >= 0.5:
            # Dark coat: gaps hold the source; the floor lifts letters above the NEARBY gap tone.
            _gw = 1.0 - ink_soft
            _num = cv2.GaussianBlur(Lm_gaps * _gw, (0, 0), sigmaX=_sig)
            _den = cv2.GaussianBlur(_gw, (0, 0), sigmaX=_sig)
            local_gap_L = np.divide(_num, _den, out=Lm_gaps.copy(), where=_den > 1e-3)
            Lm_letters = np.maximum(Lm_letters, np.clip(local_gap_L + LOCAL_FLOOR, 0, 255))
        elif light_mix < 0.5:
            _num = _gblur(Lm_letters * ink_soft, (0, 0), sigmaX=_sig)
            _den = _gblur(ink_soft, (0, 0), sigmaX=_sig)
            local_letter_L = np.divide(_num, _den, out=Lm_letters.copy(), where=_den > 1e-3)
            Lm_gaps = np.minimum(Lm_gaps, np.clip(local_letter_L - LOCAL_FLOOR, 0, 255))
        else:
            _gw = 1.0 - ink_soft
            _num = _gblur(Lm_gaps * _gw, (0, 0), sigmaX=_sig)
            _den = _gblur(_gw, (0, 0), sigmaX=_sig)
            local_gap_L = np.divide(_num, _den, out=Lm_gaps.copy(), where=_den > 1e-3)
            Lm_letters = np.minimum(Lm_letters, np.clip(local_gap_L - LOCAL_FLOOR, 0, 255))
        Lm = ink_soft * Lm_letters + (1.0 - ink_soft) * Lm_gaps
        Lm = Lm * mask + Lw * (1.0 - mask)
        return Lw, Lm

    lo, hi = 0.0, 1.0
    _, Lm_hi = _chain(hi)
    if _local_std(Lm_hi, m_in) <= target_ls:
        k = hi
    else:
        for _ in range(7):
            mid = 0.5 * (lo + hi)
            _, Lm_mid = _chain(mid)
            if _local_std(Lm_mid, m_in) < target_ls:
                lo = mid
            else:
                hi = mid
        k = 0.5 * (lo + hi)
    L_work, L_matched = _chain(k)
    if _dbg_pt is not None:
        _log(f"  [probe {_dbg_pt[0]},{_dbg_pt[1]}] L before-CLAHE={L_cur[_dbg_pt[1], _dbg_pt[0]]:.0f} "
              f"CLAHE(clip {clahe_clip})={L_clahe[_dbg_pt[1], _dbg_pt[0]]:.0f} blended(k={k:.2f})={L_work[_dbg_pt[1], _dbg_pt[0]]:.0f}")

    # Step 2 -- luminance distribution match to the source (quantile map, monotonic), last so
    # the delivered image's tonal range inside the mask is the source's by construction.
    # (Already computed inside _chain above for the solved k.)
    comp_lab = comp_lab.astype(np.float32)
    comp_lab[..., 0] = L_matched * mask + L_work * (1.0 - mask)
    comp_rgb = cv2.cvtColor(np.clip(comp_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Step 3 -- saturation distribution match, same idea.
    comp_hsv = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    S_matched = _quantile_match(comp_hsv[..., 1][m_in], src_hsv[..., 1][m_in], comp_hsv[..., 1])
    comp_hsv[..., 1] = S_matched * mask + comp_hsv[..., 1] * (1.0 - mask)
    composited = cv2.cvtColor(np.clip(comp_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    _dbg("after L-match", comp_rgb.astype(np.float32))
    _dbg("final (after S-match)", composited)

    L_after = cv2.cvtColor(np.clip(composited, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    pcts = [5, 50, 95]
    _log(f"tonal match (LAB L inside mask, pct {pcts}): source={np.percentile(src_lab[..., 0][m_in], pcts).round(0)} "
          f"before={np.percentile(L_before, pcts).round(0)} after={np.percentile(L_after[m_in], pcts).round(0)}")
    _log(f"local contrast (mean 32px-block L std): source={target_ls:.1f} before={cur_ls:.1f} "
          f"after={_local_std(L_after, m_in):.1f}  (CLAHE clip={clahe_clip} blend k={k:.2f})")
    # Typography legibility in the delivered composite, as a number: mean L of letter pixels vs
    # gap pixels inside the mask. If these converge, the words have disappeared into the photo.
    gap_px = m_in & (ink_raw < 0.1)
    if letters_in.any() and gap_px.any():
        _log(f"letters-only L pct {pcts}: source={np.percentile(src_L_in, pcts).round(0)} "
              f"letters={np.percentile(L_after[letters_in], pcts).round(0)}")
        _log(f"letter/gap luminance: letters L={L_after[letters_in].mean():.0f}  gaps L={L_after[gap_px].mean():.0f}  "
              f"(delta {L_after[letters_in].mean() - L_after[gap_px].mean():+.0f}; light_mix={light_mix:.2f} "
              f"gap_scale={gap_scale:.2f} letter_scale={letter_scale:.2f}; letters cover {letters_in.sum() / m_in.sum():.0%})")
        _log(f"overall tone: composite mean L inside mask={L_after[m_in].mean():.0f} vs source {src_mean_L:.0f}")
    # Final typography coverage by anatomical zone (the "areas not rendered with typography" number).
    _extra = {}
    if nose_fit_final is not None:
        _nz = np.zeros((H, W), np.uint8)
        (fncx, fncy), (fna, fnb), fnangle = nose_fit_final
        cv2.ellipse(_nz, (int(round(fncx)), int(round(fncy))), (max(1, int(round(fna))), max(1, int(round(fnb)))),
                   fnangle, 0, 360, 1, -1)
        _extra["nose"] = _nz > 0
    cov_final = region_coverage(ink_raw, mask, region_map, _extra)
    _log("typography coverage by zone: " + "  ".join(f"{k} {v[0]:.0%}" for k, v in cov_final.items()))
    # Letter/gap separation where it matters most -- per zone, not just the whole mask.
    _zone_masks = {"eyes": (region_map == 1) | (region_map == 2), "rest": region_map == 3}
    _zone_masks.update(_extra)
    _parts = []
    for _zn, _zm in _zone_masks.items():
        _l, _g = m_in & _zm & (ink_raw > 0.5), m_in & _zm & (ink_raw < 0.1)
        if _l.sum() > 50 and _g.sum() > 50:
            _parts.append(f"{_zn} {L_after[_l].mean() - L_after[_g].mean():+.0f}")
    _log("letter/gap delta by zone: " + "  ".join(_parts))
    _TL.placements[:] = best_placements
    _TL.pass_tags[:] = best_pass
    rep = placement_report(region_map, mask, _extra)
    _log("words placed by zone (count / median px / 10th-pct px): "
          + "  ".join(f"{k} {v[0]} / {v[1]:.0f} / {v[2]:.0f}" for k, v in rep.items()))
    fp_cov = footprint_coverage(best_placements, mask)
    coll = best_stats["overlap_px"] / max(1, best_stats["glyph_px"])
    coll_core = best_stats["core_overlap_px"] / max(1, best_stats["core_px"])
    # "Exposed space": animal pixels farther than half a glyph (3px at the 6px floor) from any
    # typography -- the direct measure of "every exposed space has typography."
    # Free-space channel width via distance-to-ink: "exposed" = farther than half a glyph from any
    # typography, scaled with the render (3px at 1x); "fillable-but-unfilled" = free channels
    # wide enough for a glyph at the 6px floor (>=8px), the part of the deficit that's ours to fix.
    _dist = cv2.distanceTransform((ink_raw <= 0.3).astype(np.uint8), cv2.DIST_L2, 5)
    _thr = 3.0 * max(1.0, W / 1030.0)
    exposed = float(((_dist > _thr) & m_in)[m_in].mean())
    fillable = float(((_dist * 2 >= 8) & m_in)[m_in].mean())
    _log(f"CLAIM METRICS: typography footprint covers {fp_cov:.1%} of the animal; exposed space "
          f"(>{_thr:.0f}px from any glyph) {exposed:.1%}; glyph-fillable but unfilled {fillable:.1%}; "
          f"{len(best_placements)} words; collisions: letter BODIES overlapping {coll_core:.2%}, "
          f"any antialiased touch {coll:.2%}")

    composited_u8 = np.clip(composited, 0, 255).astype(np.uint8)
    if debug_dir:
        photo_out = out_path.rsplit(".", 1)[0] + "_on_photo.jpg"
        Image.fromarray(composited_u8).save(photo_out, quality=92)

    # ---- Recommendation #14: automatic type-only likeness test ----------------------------
    # Four panels: A (source), B (typography+color, already saved as photo_out), C (typography
    # only, already saved as out_path -- render_word_bitmap already fills near-black on white,
    # so it was monochrome by construction), D (typography-only blurred past legibility). Plus
    # a single face-weighted number so a future change can be judged against this run instead
    # of by eye. ~25px at this photo's resolution, per the recommendation's "~20-30px."
    blur_sigma = max(8.0, W * 0.022)
    score, blurred_type, blurred_source = type_only_likeness(
        canvas, mask, bgr_source, attractor_pts, base, blur_sigma)
    if debug_dir:
        stem = out_path.rsplit(".", 1)[0]
        Image.fromarray(np.clip(blurred_type, 0, 255).astype(np.uint8)).save(stem + "_D_blurred.jpg")
        Image.fromarray(np.clip(blurred_source, 0, 255).astype(np.uint8)).save(stem + "_D_blurred_source.jpg")
        cv2.imwrite(stem + "_A_source.jpg", bgr_source)
        _log(f"wrote {out_path}; wrote {photo_out}; wrote {stem}_A_source.jpg, {stem}_D_blurred.jpg")
    _log(f"type-only likeness score (face-weighted SSIM, blur sigma={blur_sigma:.1f}): {score:.4f}")
    metrics = {
        "footprint_coverage": fp_cov, "exposed_space": exposed, "fillable_unfilled": fillable,
        "elements": len(best_placements), "collision_body": coll_core, "collision_any": coll,
        "type_only_likeness": score, "zone_coverage": {k: v[0] for k, v in cov_final.items()},
        "words_by_zone": rep, "attractor_pts": attractor_pts, "size": (W, H),
        # How much of each final pixel is the OUTER ground (outside the animal, under no
        # fringe hair): lets a later request swap the backdrop color by arithmetic instead of
        # a re-render. Stored as uint8 to keep the render cache small.
        "outside_w": np.clip((1.0 - mask) * (1.0 - whisker_ink_alpha) * 255.0, 0, 255).astype(np.uint8),
        "backdrop_rgb": tuple(float(v) for v in backdrop_rgb) if backdrop_rgb is not None else None,
    }
    return composited_u8, metrics


# ---- Render cache for the site's entry point --------------------------------------------
# One photo produces three requests in normal use: the preview, the "view up close" loupe a
# moment later, and a re-render for every backdrop toggle. Each took a full 25-60 s render and
# they overlapped on the render threads, so the third routinely exceeded the browser's 120 s
# limit ("taking longer than usual"). The engine is deterministic, so the typography for a
# (photo, words, aspect) is rendered ONCE at the loupe resolution and everything else is
# derived: the preview is a resize, the backdrop swap is arithmetic on the outer-ground weight.
# A request already rendering for the same key is waited on, not duplicated.
_RENDER_CACHE = {}            # key -> (work_h, rgb_u8, outside_w_u8, ground_rgb)
_RENDER_CACHE_ORDER = []      # LRU, oldest first
_RENDER_CACHE_MAX = int(os.environ.get("PET_V2_CACHE_ENTRIES", "4") or 4)
_RENDER_INFLIGHT = {}         # key -> threading.Event
_RENDER_CACHE_LOCK = threading.Lock()
_PREVIEW_UNIFY_MAX = 1600     # preview and loupe both come from one render at this height


def _cache_put(key, value):
    with _RENDER_CACHE_LOCK:
        _RENDER_CACHE[key] = value
        if key in _RENDER_CACHE_ORDER:
            _RENDER_CACHE_ORDER.remove(key)
        _RENDER_CACHE_ORDER.append(key)
        while len(_RENDER_CACHE_ORDER) > _RENDER_CACHE_MAX:
            _RENDER_CACHE.pop(_RENDER_CACHE_ORDER.pop(0), None)


def _cache_get(key):
    with _RENDER_CACHE_LOCK:
        v = _RENDER_CACHE.get(key)
        if v is not None and key in _RENDER_CACHE_ORDER:
            _RENDER_CACHE_ORDER.remove(key)
            _RENDER_CACHE_ORDER.append(key)
        return v


def _with_backdrop(rgb_u8, outside_w_u8, old_rgb, new_rgb):
    """Swap the outer ground color of a finished render: exact where the pixel is pure
    backdrop, proportionate under the soft matte edge and the fringe hairs."""
    if old_rgb is None or tuple(old_rgb) == tuple(new_rgb):
        return rgb_u8
    w = outside_w_u8.astype(np.float32)[..., None] / 255.0
    delta = np.asarray(new_rgb, np.float32) - np.asarray(old_rgb, np.float32)
    return np.clip(rgb_u8.astype(np.float32) + delta * w, 0, 255).astype(np.uint8)


def render_pet_portrait_v2(image_bytes, words, ground="dark", height=900,
                           print_aspect=None, type_scale=None):
    """Drop-in for pet_proto.render_pet_portrait (same signature, PNG bytes out). `height` is
    the working resolution and therefore the typography fineness: previews ~1050-1600, print
    at the PET_V2_MAX_RENDER_PX cap (default 2400) then upscaled. `ground` is the site's
    backdrop choice (pet_proto.GROUNDS) and colors everything outside the animal; `type_scale`
    is the site's Small/Medium/Large slider (see render_v2)."""
    import hashlib
    from ..pet_proto import _fit_print_aspect, GROUNDS
    gb, gg, gr = GROUNDS.get((ground or "dark").strip().lower(), GROUNDS["dark"])
    ground_rgb = (float(gr), float(gg), float(gb))
    cap = int(os.environ.get("PET_V2_MAX_RENDER_PX", "2400") or 2400)
    want_h = min(int(height), cap) if height and height > 0 else 0
    # Preview-class requests (1050-1600) all render at 1600 so the loupe is a resize of the
    # preview's own render rather than a second one -- and preview and loupe then agree.
    work_h = _PREVIEW_UNIFY_MAX if 1050 <= want_h <= _PREVIEW_UNIFY_MAX else want_h
    _ts = round(float(type_scale), 3) if type_scale else 0.30
    key = (hashlib.sha1(image_bytes).hexdigest(), str(words or ""), float(print_aspect or 0.0), _ts)

    def _finish(entry):
        cached_h, rgb, outside_w, old_ground = entry
        rgb = _with_backdrop(rgb, outside_w, old_ground, ground_rgb)
        if height and height > 0 and rgb.shape[0] != int(height):
            out_w = int(round(rgb.shape[1] * height / rgb.shape[0]))
            rgb = cv2.resize(rgb, (out_w, int(height)),
                             interpolation=cv2.INTER_AREA if int(height) < rgb.shape[0] else cv2.INTER_CUBIC)
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encode failed")
        return png.tobytes()

    while True:
        entry = _cache_get(key)
        if entry is not None and entry[0] >= work_h:
            return _finish(entry)
        with _RENDER_CACHE_LOCK:
            ev = _RENDER_INFLIGHT.get(key)
            if ev is None:
                ev = threading.Event()
                _RENDER_INFLIGHT[key] = ev
                mine = True
            else:
                mine = False
        if not mine:
            ev.wait()          # someone is rendering this photo; take their result
            continue
        break

    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode image")
        if not work_h:
            work_h = bgr.shape[0]
        if bgr.shape[0] != work_h:
            bgr = cv2.resize(bgr, (max(1, int(bgr.shape[1] * work_h / bgr.shape[0])), work_h),
                             interpolation=cv2.INTER_AREA if work_h < bgr.shape[0] else cv2.INTER_CUBIC)
        mask = _foreground_mask(bgr)
        if print_aspect:
            bgr, mask = _fit_print_aspect(bgr, mask, float(print_aspect))
        rgb, metrics = render_v2(bgr, words, mask=mask, render_scale=1.0, backdrop_rgb=ground_rgb,
                                 type_scale=_ts,
                                 verbose=os.environ.get("PET_V2_VERBOSE", "") not in ("", "0"))
        entry = (work_h, rgb, metrics["outside_w"], ground_rgb)
        _cache_put(key, entry)
    finally:
        with _RENDER_CACHE_LOCK:
            _RENDER_INFLIGHT.pop(key, None)
        ev.set()
    return _finish(entry)


def main():
    """CLI for QA: python -m app.pet_v2.engine <photo> [out.jpg] -- writes the A/B/C/D panels
    beside out.jpg and prints every metric (GOP_SCALE, GOP_MAX_OVERLAP, GOP_LANDMARKS honored)."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    image_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "glyphs_v2.jpg"
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise SystemExit(f"could not read image: {image_path}")
    debug_dir = os.path.dirname(os.path.abspath(out_path))
    stem = os.path.splitext(os.path.basename(out_path))[0]
    render_v2(bgr, None, debug_dir=debug_dir, out_stem=stem, verbose=True)


if __name__ == "__main__":
    main()
