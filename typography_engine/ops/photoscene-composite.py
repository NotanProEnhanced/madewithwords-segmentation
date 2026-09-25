#!/usr/bin/env python3
"""Bakes History in Words portraits directly into the Photo Tour panorama JPGs
(static/gallery/photoscene/*.jpg).

Earlier versions of this script tried to make our portraits sit inside the REAL
photographed frames in these photos -- pixel-replacing just the picture area so ours
used the real frame's own border/mat as its frame. That worked well for frames whose
real shape was already close to a portrait's, but for the frames that weren't (a wide
lake scene, a narrow tall strip), every way of reconciling a 4:5 portrait with a
differently-shaped hole looked wrong: stretched a face out of shape, left the original
art showing round the edges, or padded the gap with an obviously-pasted-on rectangle.
That's not a technique problem, it's a shape problem -- you cannot losslessly fit a
4:5 rectangle into a 3:1 one.

The version after that sidestepped the shape problem by erasing the original art
(gradient-filled from the real wall color) and drawing our own portrait, at its own
proportions, in a freshly-drawn AXIS-ALIGNED gold frame. That fixed the shape problem
but missed a different one: several of these frames sit on a wall seen at an angle in
the photo (especially near the edges of each panorama), where a real frame appears
skewed/trapezoidal, not a plain rectangle -- an axis-aligned frame pasted there reads
as flat and pasted-on rather than actually mounted on that angled wall.

This version fixes THAT: before erasing each frame, OpenCV detects its real four
corners (find_frame_quad(), thresholding + contour polygon approximation -- works for
most frames; the few it can't find a clean quad for fall back to an axis-aligned box,
same as before, not a regression). The portrait + frame is then perspective-warped
into that exact quadrilateral (find_coeffs() -- the standard 8-parameter PIL recipe,
verified against a synthetic test before use) before compositing, so it inherits the
same skew as whatever real frame used to occupy that spot.

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
from PIL import Image, ImageDraw

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
EDGE_SAMPLE = 10   # px thickness of the strip sampled just outside the erase box for its fill color
FRAME_GOLD = (138, 106, 52)
FRAME_MAT = (239, 233, 220)
BORDER_PX, MAT_PX = 10, 6
ART_ASPECT = 900 / 1125


def find_frame_quad(im_rgb, x0, y0, x1, y1, pad=35):
    """The real frame's actual four corners in the ORIGINAL (pre-erase) photo, via
    OpenCV: threshold for dark pixels (the frame's border reads much darker than the
    wall), take the largest contour whose area is plausible for this region, simplify
    it to a polygon, and require exactly 4 points centered near where we expect the
    frame to be. Tries several thresholds since frame darkness varies by room
    lighting. Returns None (caller falls back to an axis-aligned box) if nothing
    clean is found -- this is a best-effort visual improvement, not something the
    bake should fail over."""
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
                # Reject a quad whose aspect ratio is wildly different from the
                # region's own -- a sign order_points picked a bad point-to-corner
                # assignment (e.g. rotated ~90deg) rather than a genuinely different
                # frame shape. Caught exactly this on Sojourner Truth's frame: a
                # portrait-shaped region detected as landscape-shaped, which then
                # warped the portrait sideways.
                tl, tr, br, bl = quad
                v = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
                h = (np.linalg.norm(tl - tr) + np.linalg.norm(bl - br)) / 2
                detected_aspect = h / max(v, 1e-6)
                expected_aspect = (x1 - x0) / max(y1 - y0, 1e-6)
                # Only reject an orientation MISMATCH (expected roughly portrait,
                # detected roughly landscape, or vice versa) -- real perspective skew
                # legitimately changes how tall/wide a detected quad's edges measure
                # relative to the rough hand-measured region (see Mark Twain, whose
                # real frame IS unusually tall from this camera angle), so matching
                # degree, not just matching orientation, rejected good detections too.
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
    corresponding SRC pixel. Verified against a synthetic labeled-rectangle test
    before trusting it on real data (a transform this easy to get subtly backwards
    is exactly the kind of bug this session kept finding the hard way)."""
    matrix = []
    for (x, y), (X, Y) in zip(dest_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=float)
    B = np.array(src_pts, dtype=float).reshape(8)
    return np.linalg.solve(A, B)


def build_framed_portrait(art):
    """The portrait on its own RGBA canvas, in a freshly-drawn gold border + mat --
    since the real photographed frame is erased, every portrait gets the same
    deliberate framing rather than relying on whatever frame used to be there."""
    pw, ph = 400, round(400 / ART_ASPECT)
    fw, fh = pw + 2 * (BORDER_PX + MAT_PX), ph + 2 * (BORDER_PX + MAT_PX)
    canvas = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, fw - 1, fh - 1], fill=FRAME_GOLD + (255,))
    d.rectangle([BORDER_PX, BORDER_PX, fw - 1 - BORDER_PX, fh - 1 - BORDER_PX], fill=FRAME_MAT + (255,))
    resized = art.resize((pw, ph), Image.LANCZOS)
    canvas.paste(resized, (BORDER_PX + MAT_PX, BORDER_PX + MAT_PX))
    return canvas


def composite_axis_aligned(im, framed, cx, cy, max_w, max_h):
    """No detected quad: center the framed portrait, scaled to sit comfortably in the
    erased area (not stretched or forced to fill it -- see conversation history for
    why that looked wrong on oddly-shaped areas)."""
    fw, fh = framed.size
    scale = min(1.0, max_w * 0.92 / fw, max_h * 0.92 / fh, 320 / fh)
    fw2, fh2 = max(1, round(fw * scale)), max(1, round(fh * scale))
    framed2 = framed.resize((fw2, fh2), Image.LANCZOS)
    x0, y0 = round(cx - fw2 / 2), round(cy - fh2 / 2)
    im.paste(framed2, (x0, y0), framed2)
    return (x0, y0, x0 + fw2, y0 + fh2)


def composite_perspective(im, framed, quad):
    """Warp the framed portrait into the real frame's own detected quadrilateral."""
    xs, ys = quad[:, 0], quad[:, 1]
    ox0, oy0 = int(np.floor(xs.min())), int(np.floor(ys.min()))
    ox1, oy1 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
    local_quad = quad - [ox0, oy0]
    fw, fh = framed.size
    src_corners = [(0, 0), (fw, 0), (fw, fh), (0, fh)]
    coeffs = find_coeffs(local_quad.tolist(), src_corners)
    warped = framed.transform((ox1 - ox0, oy1 - oy0), Image.PERSPECTIVE, coeffs,
                               Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    im.paste(warped, (ox0, oy0), warped)
    return (ox0, oy0, ox1, oy1)


def erase_region(arr, x0, y0, x1, y1):
    """Gradient-fill the box in-place from the real wall color sampled just outside
    its own four edges (bilinear blend of the four edge means) -- removes the
    original photographed art without leaving any trace of a foreign patch, since the
    fill color comes from that exact spot's own real lighting, not a fixed value or a
    value borrowed from a different part of the room."""
    h, w = arr.shape[:2]
    ex0, ey0 = max(0, x0 - EDGE_SAMPLE), max(0, y0 - EDGE_SAMPLE)
    ex1, ey1 = min(w, x1 + EDGE_SAMPLE), min(h, y1 + EDGE_SAMPLE)
    top = arr[ey0:y0, x0:x1].reshape(-1, 3).mean(axis=0) if y0 > ey0 else arr[y0, x0:x1].mean(axis=0)
    bottom = arr[y1:ey1, x0:x1].reshape(-1, 3).mean(axis=0) if ey1 > y1 else arr[y1 - 1, x0:x1].mean(axis=0)
    left = arr[y0:y1, ex0:x0].reshape(-1, 3).mean(axis=0) if x0 > ex0 else arr[y0:y1, x0].mean(axis=0)
    right = arr[y0:y1, x1:ex1].reshape(-1, 3).mean(axis=0) if ex1 > x1 else arr[y0:y1, x1 - 1].mean(axis=0)

    bw, bh = x1 - x0, y1 - y0
    fx = np.linspace(0, 1, bw)[None, :, None]
    fy = np.linspace(0, 1, bh)[:, None, None]
    horiz = left[None, None, :] * (1 - fx) + right[None, None, :] * fx
    vert = top[None, None, :] * (1 - fy) + bottom[None, None, :] * fy
    fill = ((horiz + vert) / 2).astype(np.uint8)
    arr[y0:y1, x0:x1] = fill


def yaw_pitch(px, py):
    naive_yaw = (px / W - 0.5) * 2 * math.pi
    yaw = math.pi / 2 - naive_yaw               # see module docstring
    while yaw > math.pi: yaw -= 2 * math.pi
    while yaw <= -math.pi: yaw += 2 * math.pi
    pitch = (0.5 - py / H) * math.pi
    return yaw, pitch


def world_size(px0, py0, px1, py1):
    # Angular size derived from the drawn frame's actual on-screen bounding box, so
    # the invisible click target matches what's on screen either way (warped or not).
    ang_w = (px1 - px0) / W * 2 * math.pi
    ang_h = (py1 - py0) / H * math.pi
    return 2 * RADIUS * math.tan(ang_w / 2), 2 * RADIUS * math.tan(ang_h / 2)


def main():
    total = 0
    quad_hits = 0
    js_lines = ["const PHOTO_ROOMS = ["]
    for room, regions in REGIONS.items():
        im = Image.open(ORIGINALS_DIR / f"{room}.jpg").convert("RGB")
        arr = np.array(im)
        js_lines.append(f'  {{ name:"{ROOM_NAMES[room]}", src:"/static/gallery/photoscene/{room}.jpg", slots:[')
        for (it, (x0, y0, x1, y1)) in regions:
            total += 1
            quad = find_frame_quad(arr, x0, y0, x1, y1)
            if quad is not None:
                quad_hits += 1

            ex0, ey0 = max(0, x0 - ERASE_MARGIN), max(0, y0 - ERASE_MARGIN)
            ex1, ey1 = min(W, x1 + ERASE_MARGIN), min(H, y1 + ERASE_MARGIN)
            erase_region(arr, ex0, ey0, ex1, ey1)
            im = Image.fromarray(arr).convert("RGBA")

            framed = build_framed_portrait(Image.open(ART_DIR / f"{it}.png").convert("RGB"))
            if quad is not None:
                try:
                    px0, py0, px1, py1 = composite_perspective(im, framed, quad)
                except np.linalg.LinAlgError:
                    print(f"  {it}: detected quad was degenerate, falling back to axis-aligned")
                    quad = None
            if quad is None:
                cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
                px0, py0, px1, py1 = composite_axis_aligned(im, framed, cx, cy, ex1 - ex0, ey1 - ey0)

            im = im.convert("RGB")
            arr = np.array(im)  # keep arr in sync for the next find_frame_quad/erase_region call

            yaw, pitch = yaw_pitch((px0 + px1) / 2, (py0 + py1) / 2)
            ww, wh = world_size(px0, py0, px1, py1)
            js_lines.append(
                f'    {{id:"{it}", yaw:{round(yaw, 4)}, pitch:{round(pitch, 4)}, '
                f'w:{round(ww, 1)}, h:{round(wh, 1)}}},'
            )
        js_lines.append("  ]},")
        im.save(OUT_DIR / f"{room}.jpg", quality=90)
        print(f"{room}: baked {len(regions)} portraits")
    js_lines.append("];")

    print(f"\nTOTAL BAKED: {total}  (perspective-matched: {quad_hits}, axis-aligned fallback: {total - quad_hits})\n")
    print("\n".join(js_lines))


if __name__ == "__main__":
    main()
