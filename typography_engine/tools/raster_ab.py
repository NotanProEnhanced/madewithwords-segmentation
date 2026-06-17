"""Decompose a render's cost (and A/B the rasterizers if resvg is installed).

Finds a real face photo, renders it, and reports the FULL render time. The
'raster backend=... took=Xs' line the renderer logs is the rasterization
sub-time, so:  layout + composite = full - raster.  That tells us whether the
slow part is rasterizing (swap rasterizer) or the word-layout (optimize layout /
cut preview resolution). If resvg_py is installed it also A/Bs CairoSVG vs resvg.

Run (no rebuild needed):
    docker exec -i typortrait-staging python - < tools/raster_ab.py
    docker exec -i typortrait-staging python - < tools/raster_ab.py  data/private/<file>.png
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

WORDS = "GRACE KIND MOTHER FAITHFUL ALWAYS FAMILY GRACE KIND BRILLIANT MOM"
SKIP = ("preview", "thumb", "swatch", "mask", "overlay")


def _candidates():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return [sys.argv[1]]
    pool = []
    for base in ("data", "tools"):
        for ext in ("jpg", "jpeg", "png"):
            pool += glob.glob("%s/**/*.%s" % (base, ext), recursive=True)
    imgs = [c for c in pool if not any(x in c.lower() for x in SKIP)]
    imgs.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0, reverse=True)
    return imgs


def _render(an, cfg, pref):
    os.environ["TYPO_RASTERIZER"] = pref
    t = time.time()
    png, *_ = render_layered_png(an, WORDS, "words", cfg, WarningCollector(),
                                 ink="sepia", remove_bg=True, out_width=900,
                                 render_w=1600, uppercase=True)
    return png, time.time() - t


def main():
    cfg = RenderConfig()
    cands = _candidates()
    if not cands:
        print("No image found. Pass one explicitly: ... python - < tools/raster_ab.py <path>")
        return

    # Prefer an image with a detectable face (dense, representative of the slow case).
    chosen, an = None, None
    for c in cands[:10]:
        a = _cached_analyze(open(c, "rb").read(), cfg, WarningCollector())
        if len(a.faces) > 0:
            chosen, an = c, a
            break
    if chosen is None:
        chosen = cands[0]
        an = _cached_analyze(open(chosen, "rb").read(), cfg, WarningCollector())
    cov = float((an.silhouette.mask > 0).mean()) if an.silhouette.mask is not None else -1
    print("image=%s  faces=%d  sil_coverage=%.3f" % (chosen, len(an.faces), cov))
    if len(an.faces) == 0:
        print("  (no face among the candidates -> sparse render; the dense ~28s case may not reproduce here)")

    # CairoSVG = current production. The 'raster ... took=Xs' line printed above is
    # the rasterize sub-time; this is the FULL render time.
    pc, tc = _render(an, cfg, "cairosvg")
    print("FULL render (cairosvg) = %.2fs   [layout+composite = this MINUS the raster 'took=' line above]" % tc)
    Image.open(io.BytesIO(pc)).convert("RGB").save("/tmp/ab_cairosvg.png")
    print("wrote /tmp/ab_cairosvg.png")

    try:
        import resvg_py  # noqa: F401
    except Exception as e:
        print("resvg NOT installed here (%s) -> can't A/B it. (It's only the rasterizer; "
              "if rasterizing isn't the bottleneck, installing it won't help.)" % type(e).__name__)
        return

    pr, tr = _render(an, cfg, "resvg")
    A = Image.open("/tmp/ab_cairosvg.png").convert("RGB")
    B = Image.open(io.BytesIO(pr)).convert("RGB")
    if A.size != B.size:
        B = B.resize(A.size)
    d = np.abs(np.asarray(A, np.int16) - np.asarray(B, np.int16))
    print("FULL render (resvg)    = %.2fs   speedup=%.1fx" % (tr, tc / max(tr, 1e-6)))
    print("DIFF mean_abs=%.3f / 255   max=%d   pixels_differing>8=%.3f%%"
          % (d.mean(), int(d.max()), 100.0 * (d.max(2) > 8).mean()))
    B.save("/tmp/ab_resvg.png")
    print("wrote /tmp/ab_resvg.png")


if __name__ == "__main__":
    main()
