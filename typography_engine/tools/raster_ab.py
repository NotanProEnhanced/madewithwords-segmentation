"""A/B the SVG rasterizers (CairoSVG vs resvg) on a real portrait.

Renders the SAME photo + words BOTH ways in one process (only the rasterizer
differs -- layout/crop/compositing are identical), reports the speed of each,
and measures how different the two output images actually are. Use it to decide
whether resvg (much faster) is visually equivalent to CairoSVG for this pipeline.

Run it by piping this file into the container's python (no rebuild needed):

    docker exec -i typortrait-staging python - < tools/raster_ab.py

Optionally pass an explicit image path:

    docker exec -i typortrait-staging python - < tools/raster_ab.py  /path/in/container.jpg
"""
import glob
import io
import os
import sys
import time

import numpy as np
from PIL import Image

from app.config import RenderConfig
from app.main import _cached_analyze
from app.pipeline.tonal import render_layered_png
from app.pipeline.warnings import WarningCollector

WORDS = "GRACE KIND MOTHER FAITHFUL ALWAYS FAMILY GRACE KIND"
SKIP = ("preview", "thumb", "swatch", "mask", "overlay")


def _pick_image():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    pool = glob.glob("data/uploads/*.*") + glob.glob("tools/*.png") + glob.glob("tools/*.jpg")
    imgs = [c for c in pool
            if c.lower().endswith((".png", ".jpg", ".jpeg"))
            and not any(x in c.lower() for x in SKIP)]
    return imgs[0] if imgs else None


def _render(an, cfg, pref):
    os.environ["TYPO_RASTERIZER"] = pref
    t = time.time()
    png, *_ = render_layered_png(an, WORDS, "words", cfg, WarningCollector(),
                                 ink="sepia", remove_bg=True, out_width=900,
                                 render_w=1600, uppercase=True)
    return png, time.time() - t


def main():
    img = _pick_image()
    if not img:
        print("No test image found. Pass one explicitly:")
        print("  docker exec -i typortrait-staging python - < tools/raster_ab.py <path>")
        return
    cfg = RenderConfig()
    an = _cached_analyze(open(img, "rb").read(), cfg, WarningCollector())
    print("image=%s  faces=%d  (faces=0 => sparse, not representative)" % (img, len(an.faces)))

    pc, tc = _render(an, cfg, "cairosvg")
    pr, tr = _render(an, cfg, "resvg")

    A = Image.open(io.BytesIO(pc)).convert("RGB")
    B = Image.open(io.BytesIO(pr)).convert("RGB")
    if A.size != B.size:
        B = B.resize(A.size)
    d = np.abs(np.asarray(A, np.int16) - np.asarray(B, np.int16))

    print("")
    print("SPEED   cairosvg=%.2fs   resvg=%.2fs   speedup=%.1fx" % (tc, tr, tc / max(tr, 1e-6)))
    print("DIFF    mean_abs=%.3f / 255   max=%d   pixels_differing>8=%.3f%%"
          % (d.mean(), int(d.max()), 100.0 * (d.max(2) > 8).mean()))
    print("        (mean_abs < ~1-2 and pixels_differing ~0%% => visually identical)")
    A.save("/tmp/ab_cairosvg.png")
    B.save("/tmp/ab_resvg.png")
    print("IMAGES  /tmp/ab_cairosvg.png  and  /tmp/ab_resvg.png  (docker cp them out to eyeball)")


if __name__ == "__main__":
    main()
