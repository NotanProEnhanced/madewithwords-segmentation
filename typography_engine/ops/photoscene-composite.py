#!/usr/bin/env python3
"""Bakes History in Words portraits directly into the Photo Tour panorama JPGs
(static/gallery/photoscene/*.jpg).

Rebuilt from zero after several narrower fixes in a row still didn't look right
together. What follows keeps the two ideas that had already proven out -- erase the
real frame rather than trying to fit a portrait into its shape, and perspective-warp
into its real detected corners rather than pasting a flat rectangle onto an angled
wall -- but replaces the weaker parts of the old pipeline:

  - Erasure was a hand-rolled 4-edge bilinear gradient fill. It removed the original
    art but the result was visibly a flat, faintly banded patch, not real wall. This
    uses cv2.inpaint (Telea) instead -- a real image-reconstruction algorithm that
    pulls in the surrounding wall texture, spotlight falloff and even nearby
    architectural lines (a pillar, a beam) as it fills the gap. Compared side by side
    against the old fill on the same regions, there's no contest.

  - The frame itself was a flat-filled rectangle -- correct proportions and correct
    perspective, but visually still reading as a graphic pasted on a photo rather than
    an object sitting in it. This version adds a soft blurred drop shadow (cast down
    and to the right, roughly matching the rooms' overhead spot lighting) and a subtle
    diagonal bevel shade on the gold border, and brightness-matches the finished frame
    to the actual local wall tone sampled from the freshly inpainted area, so a piece
    hung on a dim stretch of wall doesn't look like it's lit from somewhere else.

Always reads its source panoramas from ops/photoscene-originals/*.jpg -- the true,
never-baked photos as uploaded, kept permanently for exactly this purpose -- and
writes the result to static/gallery/photoscene/*.jpg, overwriting whatever's there.
Do NOT "fix" this to read from static/gallery/photoscene/ instead: those get
committed already-baked, so reading from there bakes on top of the previous bake
rather than the real photo. Needs Pillow, NumPy and OpenCV (opencv-python-headless):
    python3 ops/photoscene-composite.py

After running, paste the printed PHOTO_ROOMS block into gallery.html verbatim (it
already has the yaw correction applied -- see below) and bump PHOTO_BAKE_VERSION so
browsers with the old panoramas cached don't show stale baked positions.

IMPORTANT -- the yaw correction: THREE.js's SphereGeometry does NOT map texture u=0.5
to the camera's default forward direction (0,0,-1) the way naive spherical-coordinates
math would suggest. It was empirically calibrated (raycasting the live sphere mesh and
reading back its real UV at various camera angles) that texture u=0.75 sits at yaw=0.
This script's yaw_pitch() already applies that correction (yaw = PI/2 - naive_yaw), so
the yaw values it prints are ready to paste straight into gallery.html's PHOTO_ROOMS --
do NOT re-derive yaw from pixel position by the "obvious" formula without this
correction, or captions/click-targets will land near the right neighborhood but not on
the actual baked portrait, which is exactly what happened here before this was found.
"""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "static/gallery/art"
ORIGINALS_DIR = ROOT / "ops/photoscene-originals"   # true, never-baked source photos
OUT_DIR = ROOT / "static/gallery/photoscene"         # baked result served to visitors

ROOM_NAMES = {"entrance": "The Entrance", "main-room": "The Main Gallery", "rear-room": "The Rear Salon"}

# Outer pixel bounding boxes (x0,y0,x1,y1) of real frames in each 2048x1024
# equirectangular photo -- hand-measured by cropping+gridding the image and reading
# it off (see conversation history / ask Claude to redo this if the panoramas ever
# change). Used as a search window for find_frame_quad() and as the erase area; the
# box's own aspect ratio doesn't matter for either.
REGIONS = {
    "entrance": [
        ("edgar-allan-poe", (390, 320, 540, 515)),
        ("frederick-douglass", (780, 385, 885, 515)),
        ("harriet-tubman", (1020, 365, 1560, 540)),
        ("abraham-lincoln", (1800, 365, 1910, 540)),
        ("mark-twain", (1945, 250, 2048, 595)),
    ],
    "main-room": [
        ("susan-b-anthony", (120, 320, 280, 565)),
        ("walt-whitman", (420, 315, 595, 555)),
        ("sojourner-truth", (635, 355, 755, 540)),
        ("theodore-roosevelt", (1428, 290, 1618, 545)),
        ("nikola-tesla", (1645, 350, 1780, 530)),
        ("ulysses-s-grant", (1790, 290, 1970, 545)),
    ],
    "rear-room": [
        ("booker-t-washington", (0, 280, 268, 570)),
        ("sitting-bull", (295, 300, 405, 565)),
        ("geronimo", (620, 340, 788, 552)),
        ("clara-barton", (1085, 350, 1195, 545)),
        ("elizabeth-cady-stanton", (1215, 345, 1330, 545)),
        ("charles-darwin", (1370, 300, 1470, 545)),
        ("queen-victoria", (1605, 260, 1830, 590)),
        ("john-brown", (1900, 280, 2048, 590)),
    ],
}

W, H = 2048, 1024
RADIUS = 650       # matches PHOTO_FRAME_RADIUS in gallery.html
ERASE_MARGIN = 16  # px beyond the measured box, so the real frame's border/mat is fully gone
FRAME_GOLD = np.array([150, 116, 58])
FRAME_MAT = (239, 233, 220)
BORDER_PX, MAT_PX = 14, 8
SHADOW_PAD = 26
ART_ASPECT = 900 / 1125


# ---------------------------------------------------------------------------------
# Real-frame corner detection + perspective warp (kept from the previous version --
# this part held up; verify_perspective_math() below re-checks it before every run).
# ---------------------------------------------------------------------------------

def find_frame_quad(im_rgb, x0, y0, x1, y1, pad=35):
    """The real frame's actual four corners in the ORIGINAL (pre-erase) photo. Tries
    several thresholds (frame darkness varies by room lighting) and rejects anything
    that isn't a plausible, centered, correctly-oriented quadrilateral. Returns None
    (caller falls back to an axis-aligned box) if nothing clean is found -- this is a
    best-effort visual improvement, not something the bake should fail over."""
    exp_area = (x1 - x0) * (y1 - y0)
    ecx, ecy = (x0 + x1) / 2, (y0 + y1) / 2
    ix0, iy0 = max(0, x0 - pad), max(0, y0 - pad)
    ix1, iy1 = min(im_rgb.shape[1], x1 + pad), min(im_rgb.shape[0], y1 + pad)
    crop = im_rgb[iy0:iy1, ix0:ix1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    for thresh in (50, 60, 70, 80, 90, 100, 110):
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
            area = cv2.contourArea(c)
            if not (exp_area * 0.4 <= area <= exp_area * 1.8):
                continue
            peri = cv2.arcLength(c, True)
            for eps_frac in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07):
                approx = cv2.approxPolyDP(c, eps_frac * peri, True)
                if len(approx) != 4:
                    continue
                pts = approx.reshape(-1, 2).astype(float) + [ix0, iy0]
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                if not (abs(cx - ecx) < (x1 - x0) * 0.3 and abs(cy - ecy) < (y1 - y0) * 0.3):
                    continue
                quad = order_points(pts)
                tl, tr, br, bl = quad
                v = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
                h = (np.linalg.norm(tl - tr) + np.linalg.norm(bl - br)) / 2
                detected_aspect = h / max(v, 1e-6)
                expected_aspect = (x1 - x0) / max(y1 - y0, 1e-6)
                # Reject an orientation MISMATCH (expected portrait, detected
                # landscape or vice versa) -- real skew legitimately changes how
                # tall/wide a detected quad measures relative to the rough
                # hand-measured region, so matching degree, not just orientation,
                # rejected good detections too (see conversation history).
                if (expected_aspect > 1.05) != (detected_aspect > 1.05):
                    continue
                return quad
    return None


def order_points(pts):
    """Sort 4 arbitrary-order points into [top-left, top-right, bottom-right,
    bottom-left] -- the standard sum/difference trick (tl has the smallest x+y, br
    the largest; tr has the smallest y-x, bl the largest)."""
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def find_coeffs(dest_pts, src_pts):
    """The standard 8-parameter PIL perspective-transform recipe: coefficients that
    make Image.transform(..., PERSPECTIVE, coeffs) map each DEST pixel to the
    corresponding SRC pixel."""
    matrix = []
    for (x, y), (X, Y) in zip(dest_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=float)
    B = np.array(src_pts, dtype=float).reshape(8)
    return np.linalg.solve(A, B)


def verify_perspective_math():
    """Sanity-checks find_coeffs against a synthetic case with a known answer, so a
    subtly-backwards transform (this exact class of bug bit this feature once
    already) fails loudly here instead of silently warping every portrait wrong."""
    src = [(0, 0), (100, 0), (100, 100), (0, 100)]
    dest = [(10, 5), (90, 0), (95, 95), (5, 100)]
    coeffs = find_coeffs(dest, src)
    # Source: white square on a black background. Warp it into `dest`, then check
    # that a point well inside the quad (its centroid) samples white (inside the
    # warped shape) and a point well outside it (a canvas corner) samples black --
    # catches a transform that's inverted, transposed, or otherwise backwards, without
    # the edge-rounding fragility of testing right at the quad's own boundary.
    test_img = Image.new("L", (100, 100), 255)
    warped = test_img.transform((100, 100), Image.PERSPECTIVE, coeffs, Image.NEAREST, fillcolor=0)
    cx = round(sum(p[0] for p in dest) / 4)
    cy = round(sum(p[1] for p in dest) / 4)
    assert warped.getpixel((cx, cy)) > 200, "perspective warp math regressed (quad center came out empty)"
    assert warped.getpixel((99, 99)) < 50, "perspective warp math regressed (canvas corner should be outside the quad)"


# ---------------------------------------------------------------------------------
# Frame rendering: beveled gold border + soft drop shadow, brightness-matched to the
# real local wall tone.
# ---------------------------------------------------------------------------------

def build_framed_portrait(art):
    """The portrait in a freshly-drawn gold frame with a soft drop shadow, on its own
    RGBA canvas (padded for the shadow's blur radius). A subtle diagonal bevel on the
    gold gives it some sense of being a lit 3D object rather than a flat-filled
    rectangle; the shadow (rendered on this same canvas, so it warps together with
    the frame under perspective transform) grounds it against the wall."""
    pw, ph = 400, round(400 / ART_ASPECT)
    fw, fh = pw + 2 * (BORDER_PX + MAT_PX), ph + 2 * (BORDER_PX + MAT_PX)
    canvas = Image.new("RGBA", (fw + SHADOW_PAD * 2, fh + SHADOW_PAD * 2), (0, 0, 0, 0))

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sx0, sy0 = SHADOW_PAD + 5, SHADOW_PAD + 9
    sd.rectangle([sx0, sy0, sx0 + fw, sy0 + fh], fill=(0, 0, 0, 100))
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    canvas.alpha_composite(shadow)

    yy, xx = np.mgrid[0:fh, 0:fw]
    diag = (xx / fw + (fh - yy) / fh) / 2
    shade = 0.78 + 0.44 * diag
    gold = np.empty((fh, fw, 4), dtype=np.uint8)
    for c in range(3):
        gold[..., c] = np.clip(FRAME_GOLD[c] * shade, 0, 255).astype(np.uint8)
    gold[..., 3] = 255
    frame = Image.fromarray(gold, "RGBA")
    d = ImageDraw.Draw(frame)
    d.rectangle([BORDER_PX, BORDER_PX, fw - 1 - BORDER_PX, fh - 1 - BORDER_PX], fill=FRAME_MAT + (255,))
    resized = art.resize((pw, ph), Image.LANCZOS)
    frame.paste(resized, (BORDER_PX + MAT_PX, BORDER_PX + MAT_PX))
    canvas.alpha_composite(frame, (SHADOW_PAD, SHADOW_PAD))
    return canvas


def match_brightness(framed, wall_sample_rgb):
    """Scale the framed portrait's RGB toward the sampled local wall's brightness
    level, so a piece hung in a dim corner isn't lit like it's standing in a
    spotlight, and vice versa. Modest strength (0.35) -- enough to read as
    consistent with the room, not enough to wash out the art itself."""
    target = np.array(wall_sample_rgb, dtype=float).mean()
    arr = np.array(framed).astype(float)
    cur = arr[..., :3][arr[..., 3] > 0].mean() if (arr[..., 3] > 0).any() else 200
    if cur < 1:
        return framed
    factor = 1 + 0.35 * (target / cur - 1)
    factor = min(max(factor, 0.7), 1.4)
    arr[..., :3] = np.clip(arr[..., :3] * factor, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


# ---------------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------------

def composite_axis_aligned(im, framed, cx, cy, max_w, max_h):
    """No detected quad: center the framed portrait, scaled to sit comfortably in the
    erased area (not stretched or forced to fill it)."""
    fw, fh = framed.size
    content_w, content_h = fw - 2 * SHADOW_PAD, fh - 2 * SHADOW_PAD
    scale = min(1.0, max_w * 0.92 / content_w, max_h * 0.92 / content_h, 320 / content_h)
    fw2, fh2 = max(1, round(fw * scale)), max(1, round(fh * scale))
    framed2 = framed.resize((fw2, fh2), Image.LANCZOS)
    x0, y0 = round(cx - fw2 / 2), round(cy - fh2 / 2)
    im.paste(framed2, (x0, y0), framed2)
    pad2 = SHADOW_PAD * scale
    return (x0 + pad2, y0 + pad2, x0 + fw2 - pad2, y0 + fh2 - pad2)


def composite_perspective(im, framed, quad):
    """Warp the framed portrait (shadow included) into the real frame's own detected
    quadrilateral. The quad describes the FRAME's real edge, so it's expanded
    outward by the same proportion the canvas's shadow padding represents, keeping
    the shadow visible outside the warped frame rather than clipped at the quad."""
    fw, fh = framed.size
    content_w, content_h = fw - 2 * SHADOW_PAD, fh - 2 * SHADOW_PAD
    pad_frac_x, pad_frac_y = SHADOW_PAD / content_w, SHADOW_PAD / content_h
    center = quad.mean(axis=0)
    expanded = center + (quad - center) * [1 + 2 * pad_frac_x, 1 + 2 * pad_frac_y]

    xs, ys = expanded[:, 0], expanded[:, 1]
    ox0, oy0 = int(np.floor(xs.min())), int(np.floor(ys.min()))
    ox1, oy1 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
    local_quad = expanded - [ox0, oy0]
    src_corners = [(0, 0), (fw, 0), (fw, fh), (0, fh)]
    coeffs = find_coeffs(local_quad.tolist(), src_corners)
    warped = framed.transform((ox1 - ox0, oy1 - oy0), Image.PERSPECTIVE, coeffs,
                               Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    im.paste(warped, (ox0, oy0), warped)
    return (quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max())


def yaw_pitch(px, py):
    naive_yaw = (px / W - 0.5) * 2 * math.pi
    yaw = math.pi / 2 - naive_yaw               # see module docstring
    while yaw > math.pi: yaw -= 2 * math.pi
    while yaw <= -math.pi: yaw += 2 * math.pi
    pitch = (0.5 - py / H) * math.pi
    return yaw, pitch


def world_size(px0, py0, px1, py1):
    ang_w = (px1 - px0) / W * 2 * math.pi
    ang_h = (py1 - py0) / H * math.pi
    return 2 * RADIUS * math.tan(ang_w / 2), 2 * RADIUS * math.tan(ang_h / 2)


def main():
    verify_perspective_math()

    total = 0
    quad_hits = 0
    js_lines = ["const PHOTO_ROOMS = ["]
    for room, regions in REGIONS.items():
        im_rgb = Image.open(ORIGINALS_DIR / f"{room}.jpg").convert("RGB")
        arr = np.array(im_rgb)
        js_lines.append(f'  {{ name:"{ROOM_NAMES[room]}", src:"/static/gallery/photoscene/{room}.jpg", slots:[')
        for (it, (x0, y0, x1, y1)) in regions:
            total += 1
            quad = find_frame_quad(arr, x0, y0, x1, y1)
            if quad is not None:
                quad_hits += 1

            ex0, ey0 = max(0, x0 - ERASE_MARGIN), max(0, y0 - ERASE_MARGIN)
            ex1, ey1 = min(W, x1 + ERASE_MARGIN), min(H, y1 + ERASE_MARGIN)
            mask = np.zeros(arr.shape[:2], dtype=np.uint8)
            mask[ey0:ey1, ex0:ex1] = 255
            arr = cv2.inpaint(arr, mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)

            wall_sample = arr[ey0:ey1, ex0:ex1].reshape(-1, 3).mean(axis=0)
            framed = build_framed_portrait(Image.open(ART_DIR / f"{it}.png").convert("RGB"))
            framed = match_brightness(framed, wall_sample)

            im = Image.fromarray(arr).convert("RGBA")
            if quad is not None:
                try:
                    px0, py0, px1, py1 = composite_perspective(im, framed, quad)
                except np.linalg.LinAlgError:
                    print(f"  {it}: detected quad was degenerate, falling back to axis-aligned")
                    quad = None
            if quad is None:
                cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
                px0, py0, px1, py1 = composite_axis_aligned(im, framed, cx, cy, ex1 - ex0, ey1 - ey0)
            arr = np.array(im.convert("RGB"))

            yaw, pitch = yaw_pitch((px0 + px1) / 2, (py0 + py1) / 2)
            ww, wh = world_size(px0, py0, px1, py1)
            js_lines.append(
                f'    {{id:"{it}", yaw:{round(yaw, 4)}, pitch:{round(pitch, 4)}, '
                f'w:{round(ww, 1)}, h:{round(wh, 1)}}},'
            )
        js_lines.append("  ]},")
        Image.fromarray(arr).save(OUT_DIR / f"{room}.jpg", quality=90)
        print(f"{room}: baked {len(regions)} portraits")
    js_lines.append("];")

    print(f"\nTOTAL BAKED: {total}  (perspective-matched: {quad_hits}, axis-aligned fallback: {total - quad_hits})\n")
    print("\n".join(js_lines))


if __name__ == "__main__":
    main()
