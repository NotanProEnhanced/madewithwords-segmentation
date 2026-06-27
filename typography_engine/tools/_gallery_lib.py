"""Shared placement logic for the gallery production tools.

A gallery piece needs two files:
  - the PRINT MASTER  -> PRIVATE_DIR/gallery/<id>.png   (high-res, private, sold/printed)
  - the DISPLAY PREVIEW -> static/gallery/art/<id>.png   (downscaled, public storefront)

place_master() writes both from one finished image and clears the per-aspect print
cache so a re-run can't serve a stale padded file. Used by gallery_add.py (drop in
a finished master) and gallery_render.py (render a base artwork through the engine).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

# Make `app` importable when these tools are run from the engine root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import gallery_catalog as gc          # noqa: E402
from app.config import PRIVATE_DIR, STATIC_DIR  # noqa: E402

MASTER_DIR = PRIVATE_DIR / "gallery"
PREVIEW_DIR = STATIC_DIR / "gallery" / "art"
PRINT_CACHE = PRIVATE_DIR / "gallery" / "_print"


def place_master(item_id: str, img: Image.Image, preview_width: int = 900,
                 allow_unlisted: bool = False):
    """Write the private master + public preview for `item_id` from `img`.
    Returns (master_path, preview_path, cache_files_cleared)."""
    item_id = (item_id or "").strip()
    if not item_id:
        raise SystemExit("item_id is required")
    if not allow_unlisted and gc.get(item_id) is None:
        raise SystemExit(
            f"'{item_id}' is not in static/gallery/catalog.json. "
            f"Add it to the catalog first, or pass --allow-unlisted to stage it.")
    if img.mode != "RGB":
        img = img.convert("RGB")

    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    master_path = MASTER_DIR / f"{item_id}.png"
    img.save(master_path, "PNG")

    w, h = img.size
    if w > preview_width:
        prev = img.resize((preview_width, round(h * preview_width / w)), Image.LANCZOS)
    else:
        prev = img
    preview_path = PREVIEW_DIR / f"{item_id}.png"
    prev.save(preview_path, "PNG")

    cleared = 0
    if PRINT_CACHE.exists():
        for f in PRINT_CACHE.glob(f"{item_id}_*.png"):
            try:
                f.unlink(); cleared += 1
            except OSError:
                pass
    return master_path, preview_path, cleared
