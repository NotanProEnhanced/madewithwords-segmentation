#!/usr/bin/env python3
"""Bakes History in Words portraits directly into the Photo Tour panorama JPGs
(static/gallery/photoscene/*.jpg), replacing the real photographed artwork's picture
area pixel-for-pixel so ours sits inside the same real frame, mat and lighting as
whatever used to hang there -- see the "Photo Tour" comment block in gallery.html for
why (a first version overlaid separate 3D "cards" on top of the photos instead, which
read as flat stickers pasted onto a photograph, not art actually hanging on a wall).

Always reads its source panoramas from ops/photoscene-originals/*.jpg -- the true,
never-baked photos as uploaded, kept permanently for exactly this purpose -- and
writes the result to static/gallery/photoscene/*.jpg, overwriting whatever's there.
Do NOT "fix" this to read from static/gallery/photoscene/ instead: those get
committed already-baked, so reading from there bakes a portrait on top of whatever
was baked there last time rather than on the real photo, and any REGIONS entry
removed since then silently keeps showing its stale previous bake instead of
reverting to the true original -- exactly the bug that prompted keeping a separate
pristine copy in the first place. Needs Pillow:
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

IMPORTANT -- why some real frames are skipped entirely: three earlier versions of this
script tried to make every real frame in these photos work for a portrait, however
poorly its own shape fit one, three different ways -- (1) fit a smaller aspect-correct
box inside the frame: left a border of the frame's ORIGINAL photographed art showing;
(2) stretch the portrait to fill the frame exactly: fixed that but visibly distorted
the face on any frame shaped very differently from a portrait; (3) fit the portrait at
its true aspect and pad the leftover with more of the portrait's own plain background:
no distortion and no original art showing, but on the more extreme frames the padding
was a large, flat, obviously-pasted-on rectangle -- still not the real answer. The real
fix is upstream of all three: REGIONS below only lists frames whose real aspect ratio
is already close to a portrait's (a tight tolerance band, checked in main()) -- a
handful of frames in these photos are genuinely landscape- or extreme-portrait-shaped
(a wide lake scene, a very tall narrow strip) and are simply never assigned a portrait,
left completely untouched. Trying to force a portrait into a landscape-shaped frame is
the wrong move regardless of technique; the corresponding catalog items just don't get
a baked position in these three photos (still sellable from the grid and the corridor).
"""
import math
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "static/gallery/art"
ORIGINALS_DIR = ROOT / "ops/photoscene-originals"   # true, never-baked source photos
OUT_DIR = ROOT / "static/gallery/photoscene"         # baked result served to visitors

ROOM_NAMES = {"entrance": "The Entrance", "main-room": "The Main Gallery", "rear-room": "The Rear Salon"}

# Outer pixel bounding boxes (x0,y0,x1,y1) of real frames in each 2048x1024
# equirectangular photo, hand-measured by cropping+gridding the image and reading it
# off (see conversation history / ask Claude to redo this if the panoramas ever
# change), each explicitly paired with the catalog item id that goes there -- NOT
# matched positionally against a separate items list, so dropping a region (see
# module docstring) can never silently shift every later item onto the wrong frame.
REGIONS = {
    "entrance": [
        ("edgar-allan-poe", (390, 320, 540, 515)),
        ("frederick-douglass", (780, 385, 885, 515)),
        ("abraham-lincoln", (1800, 365, 1910, 540)),
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
        ("geronimo", (620, 340, 788, 552)),
        ("clara-barton", (1085, 350, 1195, 545)),
        ("elizabeth-cady-stanton", (1215, 345, 1330, 545)),
        ("queen-victoria", (1605, 260, 1830, 590)),
    ],
}

W, H = 2048, 1024
RADIUS = 650  # matches PHOTO_FRAME_RADIUS in gallery.html
ART_ASPECT = 900 / 1125  # w/h of the source portrait PNGs (0.8)
# Regions must already be within this ratio of the portrait's own aspect (checked by
# an assertion in main(), not silently handled) -- see module docstring for why a
# region outside it is dropped from REGIONS instead of forced to fit some other way.
ASPECT_TOLERANCE = (0.55, 1.05)


def inner_box(x0, y0, x1, y1):
    """The region minus a hairline margin for the real frame's own inner edge -- the
    area to fully replace. A small, mild stretch to exactly this box is the ENTIRE
    story now that REGIONS only lists frames within ASPECT_TOLERANCE of a portrait's
    own shape -- no padding, no distortion, and the frame's real photographed border
    and mat (just outside this box) stay as the actual visible frame."""
    margin_x, margin_y = (x1 - x0) * 0.03, (y1 - y0) * 0.03
    return x0 + margin_x, y0 + margin_y, x1 - margin_x, y1 - margin_y


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
    total = 0
    js_lines = ["const PHOTO_ROOMS = ["]
    for room, regions in REGIONS.items():
        im = Image.open(ORIGINALS_DIR / f"{room}.jpg").convert("RGB")
        js_lines.append(f'  {{ name:"{ROOM_NAMES[room]}", src:"/static/gallery/photoscene/{room}.jpg", slots:[')
        for (it, (x0, y0, x1, y1)) in regions:
            aspect = (x1 - x0) / (y1 - y0)
            lo, hi = ASPECT_TOLERANCE
            assert lo <= aspect <= hi, (
                f"{room}/{it}: region aspect {aspect:.2f} outside tolerance {ASPECT_TOLERANCE} -- "
                f"drop this region from REGIONS rather than forcing a fit (see module docstring)"
            )
            total += 1
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
        im.save(OUT_DIR / f"{room}.jpg", quality=90)
        print(f"{room}: baked {len(regions)} portraits")
    js_lines.append("];")

    print(f"\nTOTAL BAKED: {total}\n")
    print("\n".join(js_lines))


if __name__ == "__main__":
    main()
