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
from pathlib import Path
from typing import Dict, Optional

from .config import STATIC_DIR

_CATALOG_PATH = STATIC_DIR / "gallery" / "catalog.json"
_ART_DIR = STATIC_DIR / "gallery" / "art"


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
