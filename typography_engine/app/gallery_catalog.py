"""Server-side view of the gallery catalog.

The storefront (static/gallery.html) reads static/gallery/catalog.json directly
for display. The backend reads the SAME file here so checkout can validate that a
requested item id is real and locate its master art file. Single source of truth;
no item list is duplicated in code.

The gallery is intentionally DECOUPLED from the personalized render pipeline: a
gallery item is fixed, pre-rendered art (a master PNG), not a user `job`/recipe.
So gallery checkout/delivery never touches the studio's /checkout or compose path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from .config import PRIVATE_DIR, STATIC_DIR

_CATALOG_PATH = STATIC_DIR / "gallery" / "catalog.json"
_ART_DIR = STATIC_DIR / "gallery" / "art"
# Original base artwork (the un-typographic painting/photo) the item was rendered
# from. Kept so the admin studio can RE-RENDER an item with new words/ground/ink.
# Public is fine here because these seed bases are public-domain art; a private
# tracked dir would be needed before storing non-PD sources.
_BASE_DIR = STATIC_DIR / "gallery" / "base"
# Print-resolution masters live OUTSIDE the public /static mount so the high-res,
# sellable file isn't downloadable for free. Drop <id>.png here to go live; until
# then master_path() falls back to the public preview (fine for placeholder art).
_MASTER_DIR = PRIVATE_DIR / "gallery"


def _load_items() -> Dict[str, dict]:
    """Flatten the catalog to {item_id: item_dict}. Reads from disk each call so
    catalog edits are picked up without a restart (the file is tiny)."""
    out: Dict[str, dict] = {}
    try:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- missing/invalid catalog => empty gallery
        return out
    for col in data.get("collections", []):
        for sec in col.get("sections", []):
            for it in sec.get("items", []):
                iid = str(it.get("id") or "").strip()
                if iid:
                    out[iid] = it
    return out


def get(item_id: str) -> Optional[dict]:
    """Return the catalog item dict for a given id, or None if unknown."""
    return _load_items().get((item_id or "").strip())


def art_path(item_id: str) -> Optional[Path]:
    """Absolute path to an item's master art PNG, or None if the file is absent
    (e.g. a planned item whose final artwork hasn't been dropped in yet)."""
    iid = (item_id or "").strip()
    if not iid:
        return None
    p = _ART_DIR / f"{iid}.png"
    return p if p.exists() else None


def master_path(item_id: str) -> Optional[Path]:
    """Path to an item's PRINT-resolution master PNG (private). Falls back to the
    public preview when no private master is present yet, so the flow works with
    placeholder art and upgrades automatically when a real master is dropped in."""
    iid = (item_id or "").strip()
    if not iid:
        return None
    m = _MASTER_DIR / f"{iid}.png"
    if m.exists():
        return m
    return art_path(iid)


def base_path(item_id: str) -> Optional[Path]:
    """Absolute path to an item's ORIGINAL base artwork, or None if not stored.
    Only items with a stored base can be re-rendered by the admin studio."""
    iid = (item_id or "").strip()
    if not iid:
        return None
    p = _BASE_DIR / f"{iid}.png"
    return p if p.exists() else None


def set_render_params(item_id: str, params: dict) -> bool:
    """Persist the render recipe (words/ground/ink/breathe/sculpt) onto the item in
    catalog.json so a render is reproducible later. Atomic write (temp + os.replace);
    _load_items() reads fresh each call so there is no cache to invalidate. Returns
    True if the item was found and written."""
    iid = (item_id or "").strip()
    if not iid:
        return False
    try:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    found = False
    for col in data.get("collections", []):
        for sec in col.get("sections", []):
            for it in sec.get("items", []):
                if str(it.get("id") or "").strip() == iid:
                    it["render_params"] = params
                    found = True
    if not found:
        return False
    tmp = _CATALOG_PATH.with_name(_CATALOG_PATH.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, _CATALOG_PATH)
    return True
