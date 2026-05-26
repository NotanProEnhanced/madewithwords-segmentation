"""Diverse portrait fixtures for regression testing.

Sample images are downloaded on demand into ``cache/`` (gitignored) from the
URLs in ``portraits.json`` so we don't vendor third-party photos. Any extra
images you drop into ``cache/`` are picked up automatically by the regression
test.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Dict

_FIXTURE_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _FIXTURE_DIR / "cache"
_MANIFEST = _FIXTURE_DIR / "portraits.json"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def ensure_portraits() -> Dict[str, Path]:
    """Download any missing manifest images into cache/; return name -> path for
    every image present (manifest downloads plus any user-dropped files)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(_MANIFEST.read_text()).get("images", {})
    for name, url in manifest.items():
        ext = ".png" if url.lower().endswith(".png") else ".jpg"
        dst = _CACHE_DIR / f"{name}{ext}"
        if dst.exists() and dst.stat().st_size > 3000:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            data = urllib.request.urlopen(req, timeout=20).read()
            if len(data) > 3000:
                dst.write_bytes(data)
        except Exception:
            # Offline / blocked: skip; the test handles a possibly-empty set.
            continue
    found: Dict[str, Path] = {}
    for p in sorted(_CACHE_DIR.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and p.stat().st_size > 3000:
            found[p.stem] = p
    return found
