"""Drop a FINISHED gallery master into place.

Use this when you already have a finished typortrait image (e.g. rendered in the
studio, or produced by gallery_render.py and edited). It writes the private
high-res master + the public storefront preview and clears the stale print cache.

    python tools/gallery_add.py good-shepherd path/to/finished.png
    python tools/gallery_add.py good-shepherd art.png --preview-width 1000

After this runs, the piece is purchasable immediately (digital now; prints when a
Printful token is configured).
"""
from __future__ import annotations

import argparse

from PIL import Image

from _gallery_lib import place_master


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item_id", help="catalog item id (see static/gallery/catalog.json)")
    ap.add_argument("art", help="path to the finished, high-res typortrait image")
    ap.add_argument("--preview-width", type=int, default=900,
                    help="downscaled width of the public storefront preview (default 900)")
    ap.add_argument("--allow-unlisted", action="store_true",
                    help="stage a piece not yet in the catalog")
    a = ap.parse_args()

    img = Image.open(a.art)
    master, preview, cleared = place_master(
        a.item_id, img, preview_width=a.preview_width, allow_unlisted=a.allow_unlisted)
    print(f"OK  master  -> {master}  ({img.size[0]}x{img.size[1]})")
    print(f"    preview -> {preview}")
    print(f"    print-cache files cleared: {cleared}")
    print(f"    storefront preview: /static/gallery/art/{a.item_id}.png")
    print(f"    it is now live at /gallery")


if __name__ == "__main__":
    main()
