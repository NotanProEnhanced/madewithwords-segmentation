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

from PIL import Image, ImageStat

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


ART_ASPECT = 900 / 1125  # w/h of the source portrait PNGs (0.8)
# How far a region's own aspect ratio may stretch the portrait before it's visibly
# distorted (a squished-narrow or smeared-wide face) rather than just mildly stretched.
ASPECT_BAND = (0.55, 1.5)


def inner_box(x0, y0, x1, y1):
    """Full-bleed: the region minus a hairline margin for the real frame's own inner
    edge. A first version tried to preserve the portrait's 4:5 aspect by fitting a
    smaller centered box inside the region -- for any region whose real aspect ratio
    wasn't close to 4:5 (most of them), that left a visible strip of the ORIGINAL
    photographed art uncovered along one or two edges. Stretching to fill the region
    completely eliminates that, and works fine for most regions -- but a handful are
    extreme (a very wide landscape frame, or a very tall narrow one), and stretching a
    portrait to fill THOSE smears or squishes the face into something unrecognizable.
    See clamp_box() for how those are handled instead."""
    margin_x, margin_y = (x1 - x0) * 0.03, (y1 - y0) * 0.03
    return x0 + margin_x, y0 + margin_y, x1 - margin_x, y1 - margin_y


def clamp_box(bx0, by0, bx1, by1):
    """For a region whose aspect ratio is too far from the portrait's to stretch
    without visible distortion, shrink to the nearest aspect within ASPECT_BAND
    (centered in the region) and report the leftover margin rectangles (0, 1 or 2 of
    them -- one pair if clamped, none if the region was already within band) so the
    caller can fill them separately instead of leaving the ORIGINAL art showing."""
    w, h = bx1 - bx0, by1 - by0
    aspect = w / h
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    if aspect > ASPECT_BAND[1]:              # region too wide -- clamp width
        new_w = h * ASPECT_BAND[1]
        px0, px1 = cx - new_w / 2, cx + new_w / 2
        margins = [(bx0, by0, px0, by1), (px1, by0, bx1, by1)]
        return (px0, by0, px1, by1), margins
    if aspect < ASPECT_BAND[0]:               # region too narrow/tall -- clamp height
        new_h = w / ASPECT_BAND[0]
        py0, py1 = cy - new_h / 2, cy + new_h / 2
        margins = [(bx0, by0, bx1, py0), (bx0, py1, bx1, by1)]
        return (bx0, py0, bx1, py1), margins
    return (bx0, by0, bx1, by1), []


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
            (px0, py0, px1, py1), margins = clamp_box(bx0, by0, bx1, by1)

            # Fill any leftover margin (only for regions whose aspect was clamped)
            # with a color sampled from that SAME margin strip in the original photo
            # -- close enough to the real local wall/mat tone to blend, unlike a single
            # fixed color used everywhere regardless of each region's own lighting.
            for (mx0, my0, mx1, my1) in margins:
                mx0i, my0i, mx1i, my1i = (int(round(v)) for v in (mx0, my0, mx1, my1))
                if mx1i <= mx0i or my1i <= my0i:
                    continue
                strip = im.crop((mx0i, my0i, mx1i, my1i))
                avg = tuple(int(c) for c in ImageStat.Stat(strip).mean[:3])
                im.paste(Image.new("RGB", (mx1i - mx0i, my1i - my0i), avg), (mx0i, my0i))

            pw, ph = int(round(px1 - px0)), int(round(py1 - py0))
            art = Image.open(ART_DIR / f"{it}.png").convert("RGB").resize((pw, ph), Image.LANCZOS)
            im.paste(art, (int(round(px0)), int(round(py0))))
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
