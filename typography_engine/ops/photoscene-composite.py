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


# The portrait PNGs are a bust on a plain, near-uniform backdrop -- sampled from a
# corner pixel, consistent across the whole catalog (checked several at (3,3)).
ART_BG = (231, 231, 231)


def inner_box(x0, y0, x1, y1):
    """The region minus a hairline margin for the real frame's own inner edge -- the
    area to fully replace."""
    margin_x, margin_y = (x1 - x0) * 0.03, (y1 - y0) * 0.03
    return x0 + margin_x, y0 + margin_y, x1 - margin_x, y1 - margin_y


def fit_on_own_background(art, bw, bh):
    """Resize the portrait to COVER the region at its own true aspect ratio (never
    stretched or cropped into the subject) and place it centered on a canvas of the
    art's own backdrop color, sized to the full region. Two earlier approaches both
    replaced less than the whole region: fitting a smaller aspect-correct box inside
    it left the ORIGINAL photographed art showing around the edges; stretching to fill
    the region completely fixed that but visibly distorted the face on any frame whose
    aspect ratio was far from the portrait's. This replaces the ENTIRE region every
    time, at the portrait's real proportions every time -- the "leftover" space (when
    the frame's own shape doesn't match a portrait) is filled with more of the
    portrait's OWN plain background, not the wall's, not a foreign texture, so it
    reads as this piece being mounted larger/smaller, never as a patch or a crop."""
    aw, ah = art.size
    scale = min(bw / aw, bh / ah)
    rw, rh = max(1, round(aw * scale)), max(1, round(ah * scale))
    resized = art.resize((rw, rh), Image.LANCZOS)
    canvas = Image.new("RGB", (bw, bh), ART_BG)
    canvas.paste(resized, ((bw - rw) // 2, (bh - rh) // 2))
    return canvas


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
            bx0i, by0i, bx1i, by1i = (int(round(v)) for v in (bx0, by0, bx1, by1))
            bw, bh = bx1i - bx0i, by1i - by0i

            art = Image.open(ART_DIR / f"{it}.png").convert("RGB")
            replacement = fit_on_own_background(art, bw, bh)
            im.paste(replacement, (bx0i, by0i))
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
