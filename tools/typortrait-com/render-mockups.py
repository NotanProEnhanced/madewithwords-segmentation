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

    for src_name, out_name in JOB_FILES:
        words = words_for(src_name)
        base = open(os.path.join(SRC, src_name), "rb").read()
        warns = WarningCollector()
        an = analyze_image(base, RenderConfig(), warns)
        png = render_displacement_portrait(
            an, words, ground=GROUND, out_width=RENDER_WIDTH, supersample=2,
            ink=INK, print_aspect=PRINT_ASPECT, breathe=True, graduate=True)
        portrait = Image.open(io.BytesIO(png)).convert("RGB")
        # cover-fit guards against any rounding drift between the render's aspect
        # and the opening's; with print_aspect=0.8 it is a straight resize.
        portrait = ImageOps.fit(portrait, (bw, bh), Image.LANCZOS)

        out = scene.copy()
        out.paste(portrait, (box[0], box[1]))
        out.save(os.path.join(OUT, out_name), quality=90, optimize=True)
        print("%-24s <- %-14s %s" % (out_name, src_name, warns.as_list()))

    print("\ndone -- review before swapping anything into /var/www")


if __name__ == "__main__":
    main()
