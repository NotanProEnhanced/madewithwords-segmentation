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

This version sidesteps the whole problem: for each real frame ERASE the original
photographed art -- gradient-fill it from the real wall color sampled just outside its
own edges, removing it completely, not covering it with anything foreign -- and then
draw our own portrait, at its own correct proportions, in a freshly-drawn frame,
sized and centered in the erased area (not stretched or forced to any particular
size). Because the surrounding wall is now genuinely wall-colored (not a mismatched
patch), the portrait never needs to fill the whole erased area -- extra erased space
either side just reads as wall, the same as any other gap between frames in a real
gallery.

Always reads its source panoramas from ops/photoscene-originals/*.jpg -- the true,
never-baked photos as uploaded, kept permanently for exactly this purpose -- and
writes the result to static/gallery/photoscene/*.jpg, overwriting whatever's there.
Do NOT "fix" this to read from static/gallery/photoscene/ instead: those get
committed already-baked, so reading from there bakes on top of the previous bake
rather than the real photo. Needs Pillow and NumPy:
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
# change). Unlike earlier versions, the box's own aspect ratio no longer matters: it's
# erased, not fit into, so a wide or narrow real frame is exactly as usable as a
# square one now.
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
EDGE_SAMPLE = 10    # px thickness of the strip sampled just outside the erase box for its fill color
FRAME_GOLD = (138, 106, 52)
FRAME_MAT = (239, 233, 220)
BORDER_PX, MAT_PX = 10, 6


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


def frame_size(bw, bh):
    """A consistent-feeling portrait size for this erased area -- filling it fully is
    no longer the goal (see module docstring), just sitting comfortably inside it."""
    target_h = max(140, min(bh, 320))
    target_w = target_h * (900 / 1125)
    if target_w > bw * 0.92:
        target_w = bw * 0.92
        target_h = target_w / (900 / 1125)
    return round(target_w), round(target_h)


def draw_framed_portrait(im, art, cx, cy, pw, ph):
    """Paste the portrait centered at (cx,cy) inside a small drawn gold border + mat
    -- since the real photographed frame is gone (erased), every portrait gets the
    same deliberate framing rather than relying on whatever frame used to be there."""
    x0, y0 = round(cx - pw / 2), round(cy - ph / 2)
    d = ImageDraw.Draw(im)
    ox0, oy0 = x0 - MAT_PX - BORDER_PX, y0 - MAT_PX - BORDER_PX
    ox1, oy1 = x0 + pw + MAT_PX + BORDER_PX, y0 + ph + MAT_PX + BORDER_PX
    d.rectangle([ox0, oy0, ox1, oy1], fill=FRAME_GOLD)
    d.rectangle([x0 - MAT_PX, y0 - MAT_PX, x0 + pw + MAT_PX, y0 + ph + MAT_PX], fill=FRAME_MAT)
    resized = art.resize((pw, ph), Image.LANCZOS)
    im.paste(resized, (x0, y0))


def yaw_pitch(px, py):
    naive_yaw = (px / W - 0.5) * 2 * math.pi
    yaw = math.pi / 2 - naive_yaw               # see module docstring
    while yaw > math.pi: yaw -= 2 * math.pi
    while yaw <= -math.pi: yaw += 2 * math.pi
    pitch = (0.5 - py / H) * math.pi
    return yaw, pitch


def world_size(pw, ph, bx0, by0, bx1, by1):
    # Angular size derived from the DRAWN frame (including its border/mat), not the
    # raw portrait pixels, so the invisible click target matches what's on screen.
    fw, fh = pw + 2 * (BORDER_PX + MAT_PX), ph + 2 * (BORDER_PX + MAT_PX)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    ang_w = fw / W * 2 * math.pi
    ang_h = fh / H * math.pi
    return 2 * RADIUS * math.tan(ang_w / 2), 2 * RADIUS * math.tan(ang_h / 2)


def main():
    total = 0
    js_lines = ["const PHOTO_ROOMS = ["]
    for room, regions in REGIONS.items():
        im = Image.open(ORIGINALS_DIR / f"{room}.jpg").convert("RGB")
        arr = np.array(im)
        js_lines.append(f'  {{ name:"{ROOM_NAMES[room]}", src:"/static/gallery/photoscene/{room}.jpg", slots:[')
        for (it, (x0, y0, x1, y1)) in regions:
            total += 1
            ex0 = max(0, x0 - ERASE_MARGIN)
            ey0 = max(0, y0 - ERASE_MARGIN)
            ex1 = min(W, x1 + ERASE_MARGIN)
            ey1 = min(H, y1 + ERASE_MARGIN)
            erase_region(arr, ex0, ey0, ex1, ey1)
            im = Image.fromarray(arr)

            bw, bh = ex1 - ex0, ey1 - ey0
            pw, ph = frame_size(bw, bh)
            cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
            art = Image.open(ART_DIR / f"{it}.png").convert("RGB")
            draw_framed_portrait(im, art, cx, cy, pw, ph)
            arr = np.array(im)  # keep arr in sync for the next erase_region call

            yaw, pitch = yaw_pitch(cx, cy)
            ww, wh = world_size(pw, ph, ex0, ey0, ex1, ey1)
            js_lines.append(
                f'    {{id:"{it}", yaw:{round(yaw, 4)}, pitch:{round(pitch, 4)}, '
                f'w:{round(ww, 1)}, h:{round(wh, 1)}}},'
            )
        js_lines.append("  ]},")
        im.save(OUT_DIR / f"{room}.jpg", quality=90)
        print(f"{room}: baked {len(regions)} portraits")
    js_lines.append("];")

    print(f"\nTOTAL BAKED: {total}\n")
    print("\n".join(js_lines))


if __name__ == "__main__":
    main()
