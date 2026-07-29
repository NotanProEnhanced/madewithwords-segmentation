#!/usr/bin/env python3
"""Batch re-render every gallery portrait at PRINT resolution for licensing / print
delivery.

Default 4800px wide = 300 DPI at 16x20in (~200 DPI at 24x36). Reads one source per
item as ``<id>-s.png`` from --src, pulls the word list from static/gallery/catalog.json,
and writes ``<id>.png`` masters to --out. The word typography is vector-crisp at any
size; photographic detail is capped by the source image resolution.

Does NOT touch the live site's masters/previews (that stays gallery_render.py's job) --
this writes only to the delivery folder.

    # one command -- render all catalog items at print size:
    python tools/render_print_masters.py

    # a couple, to test:
    python tools/render_print_masters.py --only sacred-heart-of-jesus,st-andrew
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=r"C:/Users/DELL/Desktop/FaithInWords-Sources",
                    help="folder of <id>-s.png source images")
    ap.add_argument("--out", default=r"C:/Users/DELL/Desktop/FaithInWords-Print-Masters",
                    help="where to write the <id>.png print masters")
    ap.add_argument("--out-width", type=int, default=4800,
                    help="master width in px (4800 = 300 DPI @16x20)")
    ap.add_argument("--only", default="", help="comma-separated ids to limit (for testing)")
    a = ap.parse_args()

    from app.config import RenderConfig
    from app.pipeline.analyze import analyze_image
    from app.pipeline.warnings import WarningCollector
    from app.pipeline.displacement import render_displacement_portrait

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cat = json.loads((ROOT / "static/gallery/catalog.json").read_text(encoding="utf-8"))
    items: dict = {}
    for c in cat["collections"]:
        for s in c["sections"]:
            for it in s["items"]:
                items.setdefault(it["id"], it)

    ids = [i.strip() for i in a.only.split(",") if i.strip()] or list(items)
    ok, fail = 0, []
    for n, iid in enumerate(ids, 1):
        it = items.get(iid)
        src = Path(a.src) / f"{iid}-s.png"
        words = (it.get("words") if it else "" or "").split()
        if not it or not src.exists() or not words:
            print(f"[{n}/{len(ids)}] SKIP {iid} (in_catalog={bool(it)} src={src.exists()} words={bool(words)})", flush=True)
            fail.append(iid); continue
        try:
            an = analyze_image(src.read_bytes(), RenderConfig(), WarningCollector())
            png = render_displacement_portrait(an, words, ground="navy", out_width=a.out_width,
                                               supersample=2, ink="photo", print_aspect=0.8)
            im = Image.open(io.BytesIO(png)); im.save(out / f"{iid}.png")
            print(f"[{n}/{len(ids)}] OK {iid} {im.size[0]}x{im.size[1]}", flush=True); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{n}/{len(ids)}] FAIL {iid}: {type(e).__name__}: {e}", flush=True)
            fail.append(iid)
    print(f"\nDONE {ok}/{len(ids)} print masters -> {out}\n  fail: {fail}", flush=True)


if __name__ == "__main__":
    main()
