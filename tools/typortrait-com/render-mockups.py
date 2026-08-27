#!/usr/bin/env python3
"""Render the three framed mockups for typortrait.com's "See it..." section.

The June mockups were hand-composited wall shots and are inconsistent with each
other (960x1200, 594x744, 960x1200 against the 900x1125 the page declares).
This reproduces them from the product's own framed-on-a-desk scene -- the same
presentation customers see in the studio -- so they are repeatable and match the
use-case row above them.

The scene and its mat opening come from tools/reel_template.py (build_reel):

    OPX0, OPX1, OPY0, OPY1 = 0.219, 0.773, 0.206, 0.760

Both spans are 0.554, so the opening has the scene's own aspect: 4:5. The
portraits are therefore rendered at print_aspect=0.8 rather than square, so they
are composed for the frame instead of being cropped into it (a square crop would
lose ~20% off each side and clip the words).

Runs INSIDE the container, reading and writing through the bind-mounted data/
directory:

    docker exec typortrait-staging python /app/data/render-mockups.py

Writes mock-*-new.jpg for review; swaps nothing.
"""
import io
import os
import sys

sys.path.insert(0, "/app")

from PIL import Image, ImageOps                          # noqa: E402
from app.config import RenderConfig                      # noqa: E402
from app.pipeline.warnings import WarningCollector       # noqa: E402
from app.pipeline.analyze import analyze_image           # noqa: E402
from app.pipeline.displacement import render_displacement_portrait   # noqa: E402

SRC = "/app/data/marketing-src"
OUT = "/app/data/marketing-out"
SCENE = "/app/static/scene-desk-plain.jpg"

# Mat opening, as fractions of the scene -- lifted from build_reel so the two
# stay in step. If the scene art is ever replaced, update both places.
OPX0, OPX1, OPY0, OPY1 = 0.219, 0.773, 0.206, 0.760

# Page declares width="900" height="1125" for these images.
CANVAS = (900, 1125)

GROUND = "navy"
INK = "photo"
PRINT_ASPECT = 0.8          # 4:5, matching the mat opening
RENDER_WIDTH = 1400
# breathe=True adds a margin of pure ground around the composition. On a bare
# download that margin is doing real work, but inside a mat it shows as a
# lighter navy border around a darker navy field -- two blues where there should
# be one -- and the crop makes it asymmetric, so the print reads as misaligned.
# The mat already provides the breathing room here.
BREATHE = False

# Same words as the use-case renders -- these are the same three subjects, so the
# portrait in the frame should be the portrait on the card.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from render_usecases import JOBS as USECASE_JOBS       # noqa: E402
except Exception:                                          # noqa: BLE001
    USECASE_JOBS = None

JOB_FILES = [
    ("graduate.png", "mock-graduate-new.jpg"),
    ("son.png", "mock-children-new.jpg"),
    ("man.png", "mock-memorial-new.jpg"),
]


def words_for(src_name):
    if not USECASE_JOBS:
        raise SystemExit(
            "could not import render_usecases.py -- put it beside this script in "
            "data/ (as render_usecases.py, underscore not hyphen) so the mockups "
            "use the same words as the cards")
    for s, _out, words in USECASE_JOBS:
        if s == src_name:
            return words
    raise SystemExit("no words defined for %s in render_usecases.py" % src_name)


def detect_opening(scene_rgb, fallback):
    """Measure the print aperture from the scene itself.

    The scene photograph already contains a placeholder print with its own navy
    ground. The constants above come from reel_template.py and are close but not
    exact -- pasting to them leaves a rim of the scene's navy showing around the
    portrait, and the two navies do not match. Finding the placeholder's bounding
    box instead makes the paste flush by construction.

    Falls back to the constants if the detection looks implausible.
    """
    try:
        import numpy as np
    except Exception:                                      # noqa: BLE001
        print("numpy unavailable -- using constants")
        return fallback

    a = np.asarray(scene_rgb).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    navy = (b > r + 12) & (b > g + 6) & (b < 160) & (r < 130)
    ys, xs = np.nonzero(navy)
    if ys.size < 2000:
        print("no navy region found -- using constants")
        return fallback

    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    w, h = box[2] - box[0], box[3] - box[1]
    H, W = a.shape[:2]
    area = (w * h) / float(W * H)
    aspect = w / float(h)
    if not (0.10 < area < 0.60 and 0.70 < aspect < 0.92):
        print("detected region implausible (area %.2f aspect %.3f) -- using constants"
              % (area, aspect))
        return fallback
    print("detected opening %dx%d at (%d,%d)  aspect %.3f  area %.2f"
          % (w, h, box[0], box[1], aspect, area))
    return box


def trim_ground_pad(img, tol=14):
    """Trim a uniform flat-ground border off a rendered portrait.

    The render is padded with flat ground to reach the requested print aspect.
    Pasted into a mat, that pad shows as a lighter navy strip around a darker
    composed field -- two blues where there should be one -- and because the fit
    crops unevenly, the strip is wider on one side, so the print reads as
    misaligned rather than bordered.

    Rather than assume where the pad comes from, this measures it: take the
    corner colour as the pad reference and walk inward from each edge while
    every pixel in the row/column stays within `tol` of it. Trims nothing when
    there is no pad.
    """
    try:
        import numpy as np
    except Exception:                                      # noqa: BLE001
        return img, (0, 0, 0, 0)

    a = np.asarray(img).astype(np.int16)
    H, W = a.shape[:2]
    ref = a[2, 2].astype(np.int16)

    def flat_col(x):
        return int(np.abs(a[:, x] - ref).max()) <= tol

    def flat_row(y):
        return int(np.abs(a[y, :] - ref).max()) <= tol

    left = 0
    while left < W // 3 and flat_col(left):
        left += 1
    right = W
    while right > 2 * W // 3 and flat_col(right - 1):
        right -= 1
    top = 0
    while top < H // 3 and flat_row(top):
        top += 1
    bottom = H
    while bottom > 2 * H // 3 and flat_row(bottom - 1):
        bottom -= 1

    if left == 0 and top == 0 and right == W and bottom == H:
        return img, (0, 0, 0, 0)
    return img.crop((left, top, right, bottom)), (left, top, W - right, H - bottom)


def main():
    if not os.path.isfile(SCENE):
        raise SystemExit("scene not found: %s" % SCENE)
    if not os.path.isdir(SRC):
        raise SystemExit("no source directory: %s" % SRC)
    os.makedirs(OUT, exist_ok=True)

    missing = [s for s, _ in JOB_FILES if not os.path.isfile(os.path.join(SRC, s))]
    if missing:
        raise SystemExit("missing source files in %s: %s" % (SRC, ", ".join(missing)))

    W, H = CANVAS
    box = (int(round(OPX0 * W)), int(round(OPY0 * H)),
           int(round(OPX1 * W)), int(round(OPY1 * H)))
    bw, bh = box[2] - box[0], box[3] - box[1]
    print("canvas %dx%d   mat opening %dx%d at (%d,%d)  aspect %.3f"
          % (W, H, bw, bh, box[0], box[1], bw / float(bh)))

    scene = ImageOps.exif_transpose(Image.open(SCENE)).convert("RGB")
    print("scene  %dx%d  aspect %.3f" % (scene.size[0], scene.size[1],
                                         scene.size[0] / float(scene.size[1])))
    scene = scene.resize((W, H), Image.LANCZOS)

    # measure the real aperture rather than trusting the reel constants
    box = detect_opening(scene, box)
    bw, bh = box[2] - box[0], box[3] - box[1]

    for src_name, out_name in JOB_FILES:
        words = words_for(src_name)
        base = open(os.path.join(SRC, src_name), "rb").read()
        warns = WarningCollector()
        an = analyze_image(base, RenderConfig(), warns)
        png = render_displacement_portrait(
            an, words, ground=GROUND, out_width=RENDER_WIDTH, supersample=2,
            ink=INK, print_aspect=PRINT_ASPECT, breathe=BREATHE, graduate=True)
        portrait = Image.open(io.BytesIO(png)).convert("RGB")
        portrait, trimmed = trim_ground_pad(portrait)
        # cover-fit guards against any rounding drift between the render's aspect
        # and the opening's; with print_aspect=0.8 it is close to a straight resize.
        portrait = ImageOps.fit(portrait, (bw, bh), Image.LANCZOS)

        out = scene.copy()
        out.paste(portrait, (box[0], box[1]))
        out.save(os.path.join(OUT, out_name), quality=90, optimize=True)
        print("%-24s <- %-14s trimmed L%d T%d R%d B%d  %s"
              % (out_name, src_name, trimmed[0], trimmed[1], trimmed[2],
                 trimmed[3], warns.as_list()))

    print("\ndone -- review before swapping anything into /var/www")


if __name__ == "__main__":
    main()
