#!/usr/bin/env python3
"""Render-health canary -- a scheduled drift alarm for the portrait pipeline.

Runs a known, committed photo through the REAL renderers (Words + Sculpt) and
asserts the output is not silently degraded. It is the production guard for the
failure mode that hurts most: shipping a bad portrait of someone's loved one
without anyone noticing. Catches:

  * cairosvg silently falling back to resvg (tonal modulation dropped),
  * MediaPipe no longer locking onto a clear face (model missing / regressed),
  * a renderer producing a blank / uniform / wrong-sized image.

Exit code 0 = all checks pass; 1 = at least one failed (alert on this). Intended
to run on a schedule against the deployed image, e.g. daily cron:

    cd <engine-root> && python tools/render_canary.py || mail -s "Typortrait canary FAILED" you@example.com

Usage:
    python tools/render_canary.py [path/to/photo.jpg]
"""
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

# A clean, front-facing single portrait baked into the image alongside this
# script (tools/ ships in the container, marketing/ does not), so the canary
# always has a known-good face subject wherever the code runs.
DEFAULT_PHOTO = ENGINE_ROOT / "tools" / "assets" / "canary_portrait.png"
OUT_WIDTH = 768
WORDS = ["beloved", "mother", "always", "remembered"]

# Output-quality floors. A real single portrait clears these comfortably; a
# blank/uniform/failed render does not. Kept conservative to avoid false alarms.
MIN_DIM = 320           # px on the short side
MIN_STDDEV = 12.0       # grayscale std-dev: a blank image is ~0


def _decode(png_bytes: bytes):
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype="float64")
    return arr


def _quality(arr) -> dict:
    import numpy as np
    h, w = arr.shape[:2]
    gray = arr @ np.array([0.299, 0.587, 0.114])
    return {"w": int(w), "h": int(h), "stddev": float(gray.std())}


class Canary:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok), detail))
        return bool(ok)

    def fail(self, name: str, detail: str) -> None:
        self.results.append((name, False, detail))

    def report_and_exit(self) -> None:
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        print()
        for name, ok, detail in self.results:
            tag = "PASS" if ok else "FAIL"
            line = f"  [{tag}] {name}"
            if detail:
                line += f" -- {detail}"
            print(line)
        ok_all = passed == total
        print()
        print(f"RESULT: {'PASS' if ok_all else 'FAIL'} ({passed}/{total})")
        sys.exit(0 if ok_all else 1)


def main() -> None:
    photo = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PHOTO
    print("render-health canary")
    print(f"  engine: {ENGINE_ROOT}")
    print(f"  photo:  {photo}")

    cn = Canary()

    # --- 0. The photo exists -------------------------------------------------
    if not cn.check("canary photo present", photo.exists(), str(photo)):
        cn.report_and_exit()

    # --- 1. Rasterizer is full-fidelity (THE silent-degrade guard) -----------
    try:
        from app.pipeline.raster import cairosvg_available
        cn.check("cairosvg rasterizer usable (no silent resvg fallback)",
                 cairosvg_available(),
                 "resvg fallback drops tonal modulation" if not cairosvg_available() else "")
    except Exception as e:  # noqa: BLE001
        cn.fail("cairosvg rasterizer usable", f"probe raised: {e}")

    # --- Analyze once --------------------------------------------------------
    try:
        from app.config import RenderConfig
        from app.pipeline.analyze import analyze_image
        from app.pipeline.warnings import WarningCollector
        cfg = RenderConfig()
        warns = WarningCollector()
        an = analyze_image(photo.read_bytes(), cfg, warns)
    except Exception as e:  # noqa: BLE001
        cn.fail("analyze photo", f"{e}")
        traceback.print_exc()
        cn.report_and_exit()

    # --- 2. Face locked precisely via MediaPipe ------------------------------
    cn.check("face locked via mediapipe",
             getattr(an, "face_source", None) == "mediapipe",
             f"face_source={getattr(an, 'face_source', None)}")

    # --- 3. Words (layered) render is non-degenerate -------------------------
    try:
        from app.pipeline.tonal import render_layered_png
        png, *_ = render_layered_png(
            an, " ".join(WORDS), "words", cfg, WarningCollector(),
            ink="navy", remove_bg=False, light=False,
            out_width=OUT_WIDTH, render_w=1400, uppercase=False)
        q = _quality(_decode(png))
        ok = bool(png) and min(q["w"], q["h"]) >= MIN_DIM and q["stddev"] >= MIN_STDDEV
        cn.check("words render non-degenerate",
                 ok, f"{q['w']}x{q['h']}, stddev={q['stddev']:.1f}")
    except Exception as e:  # noqa: BLE001
        cn.fail("words render non-degenerate", f"{e}")
        traceback.print_exc()

    # --- 4. Sculpt (displacement) render is non-degenerate -------------------
    try:
        from app.pipeline.displacement import render_displacement_portrait
        png = render_displacement_portrait(
            an, WORDS, ground="navy", out_width=OUT_WIDTH, uppercase=False, ink="photo")
        q = _quality(_decode(png))
        ok = bool(png) and min(q["w"], q["h"]) >= MIN_DIM and q["stddev"] >= MIN_STDDEV
        cn.check("sculpt render non-degenerate",
                 ok, f"{q['w']}x{q['h']}, stddev={q['stddev']:.1f}")
    except Exception as e:  # noqa: BLE001
        cn.fail("sculpt render non-degenerate", f"{e}")
        traceback.print_exc()

    cn.report_and_exit()


if __name__ == "__main__":
    main()
