#!/usr/bin/env python3
"""
Throwaway prototype #3: genuine flow-following typography, done the way the two earlier
attempts proved you actually have to -- as RIGID objects (whole words, not raster pixels)
carried along a traced path, so each word stays perfectly crisp no matter how the path curves.

Why this is different from the failed raster-warp attempt (fur_flow_render_prototype.py):
that approach warped a continuous ink-density IMAGE, which meant the warp field had to be
smoothed coarser than a single glyph's width to avoid shearing individual letters apart -- and
once smoothed that coarse, the effect nearly vanished into what the existing luminance-drape
already does. This approach never touches pixels inside a word: it renders each word as its own
small bitmap ONCE, then rotates and pastes that whole rigid bitmap along a traced path. The word
can curve sharply from one placement to the next without any single word ever distorting.

Method:
  1. Structure-tensor orientation field (reused from fur_flow_prototype.py) gives a local flow
     ANGLE (mod pi -- a line has no direction, only orientation) and a COHERENCE score.
  2. trace_streamline(): walks forward from a seed point, at each step resolving the field's
     mod-pi angle against the path's current travel direction (so it doesn't randomly flip
     180 degrees step to step) and clamping the turn rate (so real fur noise doesn't make the
     path zigzag). Stops at the mask edge, canvas edge, or where coherence drops too low to
     trust (drifting into a flat/background area with no real grain to follow).
  3. place_words_along_path(): walks the traced path at word-spaced intervals, rendering each
     word as its own PIL image, rotating that whole image to the path's LOCAL tangent angle at
     that point, and pasting it centered there.
  4. Multiple streamlines are seeded from a grid so the frame fills up, the same way the
     existing engine tiles rows.

Kept deliberately simple for a first look: plain dark ink on white, no color/photo compositing,
no tiering -- the only question this answers is "does real flow-following, letters-intact
typography look organically grown from the coat, more than raster warping did." Coloring and
tiering are solved problems already in pet_proto.py if this looks worth pursuing further.

Usage:
    python3 glyphs_on_path_prototype.py <image_path> [output_path]
"""
import math
import random


import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from ..pet_proto import _foreground_mask, _weighted_stream, _phrases, _enhance_contrast, _edge_ink, _FONT
from .fur_flow import orientation_field


def resolve_angle(raw_mod_pi, prev_angle):
    """The structure tensor gives an orientation (a line has no direction, only tilt), so
    raw and raw+pi describe the SAME line. Pick whichever matches the path's current travel
    direction, so the trace doesn't flip 180 degrees at random from one step to the next."""
    cands = (raw_mod_pi, raw_mod_pi + math.pi, raw_mod_pi - math.pi)
    return min(cands, key=lambda a: abs(((a - prev_angle + math.pi) % (2 * math.pi)) - math.pi))


def clamp_turn(new_angle, prev_angle, max_turn):
    d = ((new_angle - prev_angle + math.pi) % (2 * math.pi)) - math.pi
    d = max(-max_turn, min(max_turn, d))
    return prev_angle + d


def trace_streamline(theta, coherence, mask, x0, y0, init_angle=0.0,
                      step=5.0, max_steps=300, min_coherence=0.15, max_turn=0.12,
                      covered=None, sep_px=0):
    """Walk forward from (x0,y0) following the local flow direction. Returns a list of
    (x, y, angle) samples. Traces LEFT-TO-RIGHT-ish (init_angle biases the starting travel
    direction toward normal reading direction; the field is then free to bend it from there).

    `covered`, if given, is a shared occupancy grid: the trace stops the moment it enters
    territory an EARLIER streamline already claimed, rather than overlapping it. Streamlines
    are a well-known case where naive multi-seeding converges onto the same ridge lines --
    without this, dense/chaotic pile-ups are the default outcome, not an edge case."""
    H, W = theta.shape
    x, y, angle = x0, y0, init_angle
    pts = [(x, y, angle)]
    for _ in range(max_steps):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            break
        if mask[yi, xi] < 0.5 or coherence[yi, xi] < min_coherence:
            break
        if covered is not None and len(pts) > 3 and covered[yi, xi]:
            break
        raw = float(theta[yi, xi])
        resolved = resolve_angle(raw, angle)
        angle = clamp_turn(resolved, angle, max_turn)
        x += step * math.cos(angle)
        y += step * math.sin(angle)
        pts.append((x, y, angle))
    if covered is not None:
        for (px, py, _) in pts:
            cv2.circle(covered, (int(round(px)), int(round(py))), sep_px, 1, -1)
    return pts


def path_length(pts):
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))


def sample_path_at(pts, target_dist):
    """Return (x, y, angle) at arc-length target_dist along the traced path, or None past the end."""
    acc = 0.0
    for i in range(1, len(pts)):
        x0, y0, _ = pts[i - 1]
        x1, y1, a1 = pts[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        if acc + seg >= target_dist:
            t = 0.0 if seg < 1e-6 else (target_dist - acc) / seg
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, a1)
        acc += seg
    return None


def render_word_bitmap(word, font, alpha=255):
    """A word as its own tight, upright RGBA bitmap -- rotated later as one rigid unit.
    `alpha` lets a finer tier draw as a quieter whisper of texture rather than fighting the
    hero tier for attention -- the real engine's tiers vary by SIZE only; here, since finer
    tiers exist mainly to fill gaps the hero tier's words left, a lighter touch keeps them
    reading as texture rather than as equally-important words."""
    tmp = Image.new("RGBA", (1, 1))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), word, font=font)
    w, h = bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((2 - bbox[0], 2 - bbox[1]), word, font=font, fill=(20, 20, 20, alpha))
    return im


def place_words_along_path(canvas, pts, words, font, gap_px, alpha=255):
    """Walk the path at word-spaced intervals; each word is rotated to the path's LOCAL
    tangent angle at its own position and pasted as a rigid block -- it never bends
    internally, only the sequence of placements curves."""
    total = path_length(pts)
    d = 0.0
    wi = 0
    while d < total:
        word = words[wi % len(words)]
        wi += 1
        bmp = render_word_bitmap(word, font, alpha=alpha)
        samp = sample_path_at(pts, d)
        if samp is None:
            break
        x, y, angle = samp
        rot = bmp.rotate(-math.degrees(angle), expand=True, resample=Image.BICUBIC)
        px, py = int(round(x - rot.width / 2)), int(round(y - rot.height / 2))
        canvas.alpha_composite(rot, (px, py))
        d += bmp.width + gap_px
