#!/usr/bin/env python3
"""Bakes History in Words portraits directly into the Photo Tour panorama JPGs
(static/gallery/photoscene/*.jpg), replacing the real photographed artwork's picture
area pixel-for-pixel so ours sits inside the same real frame, mat and lighting as
whatever used to hang there -- see the "Photo Tour" comment block in gallery.html for
why (a first version overlaid separate 3D "cards" on top of the photos instead, which
read as flat stickers pasted onto a photograph, not art actually hanging on a wall).

Run this from a tree where static/gallery/photoscene/*.jpg are still the ORIGINAL,
un-baked panoramas (git checkout -- static/gallery/photoscene/ first if they've
already been baked once -- this script pastes destructively, not idempotently, and
running it twice bakes a portrait on top of an already-baked one). Needs Pillow:
    python3 ops/photoscene-composite.py

REGIONS below are outer pixel bounding boxes (x0,y0,x1,y1) of real frames in each
2048x1024 equirectangular photo, hand-measured by cropping+gridding the image and
reading it off (see conversation history / ask Claude to redo this if the panoramas
ever change). ITEMS assigns catalog items to regions in order, one list per room --
there are only as many usable regions as real frames worth pasting into, so a 23-item
catalog with only 19 good regions leaves the last 4 items unbaked (still sellable from
the grid and the walkthrough corridor, just without a natural spot in these photos).

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
import json
import math
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "static/gallery/art"
PANO_DIR = ROOT / "static/gallery/photoscene"

ITEMS = [
    "edgar-allan-poe", "frederick-douglass", "harriet-tubman", "abraham-lincoln", "mark-twain",
    "susan-b-anthony", "walt-whitman", "sojourner-truth", "theodore-roosevelt", "nikola-tesla",
    "ulysses-s-grant", "booker-t-washington", "sitting-bull", "geronimo", "clara-barton",
    "elizabeth-cady-stanton", "charles-darwin", "queen-victoria", "john-brown", "pt-barnum",
    "chief-joseph", "emily-dickinson", "robert-e-lee",
]

ROOM_NAMES = {"entrance": "The Entrance", "main-room": "The Main Gallery", "rear-room": "The Rear Salon"}

REGIONS = {
    "entrance": [
        (390, 320, 540, 515),
        (780, 385, 885, 515),
        (1020, 365, 1560, 540),
        (1800, 365, 1910, 540),
        (1945, 250, 2048, 595),
    ],
    "main-room": [
        (120, 320, 280, 565),
        (420, 315, 595, 555),
        (635, 355, 755, 540),
        (1428, 290, 1618, 545),
        (1645, 350, 1780, 530),
        (1790, 290, 1970, 545),
    ],
    "rear-room": [
        (0, 280, 268, 570),
        (295, 300, 405, 565),
        (620, 340, 788, 552),
        (1085, 350, 1195, 545),
        (1215, 345, 1330, 545),
        (1370, 300, 1470, 545),
        (1605, 260, 1830, 590),
        (1900, 280, 2048, 590),
    ],
}

W, H = 2048, 1024
RADIUS = 650  # matches PHOTO_FRAME_RADIUS in gallery.html


def inner_box(x0, y0, x1, y1):
    """A centered 4:5 (portrait) box within the region, filling most of it -- leaves a
    thin margin of the real frame's own mat/border visible, which is what makes the
    result read as "a picture in that frame" instead of a precisely-matched sticker."""
    avail_w, avail_h = (x1 - x0) * 0.94, (y1 - y0) * 0.9
    target_w = avail_h * 0.8
    if target_w > avail_w:
        target_w, target_h = avail_w, avail_w / 0.8
    else:
        target_h = avail_h
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - target_w / 2, cy - target_h / 2, cx + target_w / 2, cy + target_h / 2


def yaw_pitch(px, py):
    naive_yaw = (px / W - 0.5) * 2 * math.pi
    yaw = math.pi / 2 - naive_yaw               # see module docstring
    while yaw > math.pi: yaw -= 2 * math.pi
    while yaw <= -math.pi: yaw += 2 * math.pi
    pitch = (0.5 - py / H) * math.pi
    return yaw, pitch


def world_size(bx0, by0, bx1, by1):
    ang_w = (bx1 - bx0) / W * 2 * math.pi
    ang_h = (by1 - by0) / H * math.pi
    return 2 * RADIUS * math.tan(ang_w / 2), 2 * RADIUS * math.tan(ang_h / 2)


def main():
    idx = 0
    js_lines = ["const PHOTO_ROOMS = ["]
    for room, regions in REGIONS.items():
        im = Image.open(PANO_DIR / f"{room}.jpg").convert("RGB")
        js_lines.append(f'  {{ name:"{ROOM_NAMES[room]}", src:"/static/gallery/photoscene/{room}.jpg", slots:[')
        for (x0, y0, x1, y1) in regions:
            it = ITEMS[idx]; idx += 1
            bx0, by0, bx1, by1 = inner_box(x0, y0, x1, y1)
            bw, bh = int(round(bx1 - bx0)), int(round(by1 - by0))
            art = Image.open(ART_DIR / f"{it}.png").convert("RGB").resize((bw, bh), Image.LANCZOS)
            im.paste(art, (int(round(bx0)), int(round(by0))))
            yaw, pitch = yaw_pitch((bx0 + bx1) / 2, (by0 + by1) / 2)
            ww, wh = world_size(bx0, by0, bx1, by1)
            js_lines.append(
                f'    {{id:"{it}", yaw:{round(yaw, 4)}, pitch:{round(pitch, 4)}, '
                f'w:{round(ww, 1)}, h:{round(wh, 1)}}},'
            )
        js_lines.append("  ]},")
        im.save(PANO_DIR / f"{room}.jpg", quality=90)
        print(f"{room}: baked {len(regions)} portraits")
    js_lines.append("];")

    print(f"\nTOTAL ITEMS USED: {idx} of {len(ITEMS)}\n")
    print("\n".join(js_lines))


if __name__ == "__main__":
    main()
