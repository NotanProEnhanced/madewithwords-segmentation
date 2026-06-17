"""SVG -> PNG rasterization. Prefers CairoSVG (production); falls back to
resvg-py when Cairo's native library is unavailable (e.g. local Windows).

CairoSVG and resvg are NOT pixel-identical: resvg does not honour every blend/
mask mode the layered renderer relies on, so a silent fallback quietly drops
tonal modulation -- a degraded portrait that looks "fine" until you compare it.
We therefore make the fallback *loud*: log it once, record the active backend,
and expose cairosvg_available() so the render path and the health canary can
warn on (or refuse to ship) a degraded rasterizer instead of shipping silently.
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

_log = logging.getLogger("typortrait.raster")

# Best-effort record of the backend used by the most recent rasterization:
# "cairosvg" (full fidelity) | "resvg" (degraded fallback) | None (none yet).
_last_backend: Optional[str] = None
_warned_fallback = False


def last_backend() -> Optional[str]:
    """The rasterizer backend used by the most recent svg_to_png_bytes call."""
    return _last_backend


@lru_cache(maxsize=1)
def cairosvg_available() -> bool:
    """True only if CairoSVG can ACTUALLY rasterize in this process -- it imports
    AND its native cairo library is present at call time. Import success alone is
    not enough (the native lib can be missing), so we do a 1x1 trial render.
    Cached: the answer cannot change without a process restart."""
    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>')
        return True
    except Exception:  # noqa: BLE001 -- ImportError or OSError (native lib missing)
        return False


def _note_fallback(reason: str) -> None:
    global _warned_fallback
    if not _warned_fallback:
        _warned_fallback = True
        _log.warning(
            "CairoSVG unavailable (%s) -- falling back to resvg. Rasterization is "
            "DEGRADED: blend/mask modulation may be dropped. Install cairo for "
            "full-fidelity output.", reason)


def _rasterizer_pref() -> str:
    """Which backend to use, from TYPO_RASTERIZER:
      'auto' (default) -- CairoSVG, fall back to resvg if cairo is unavailable;
      'resvg'          -- force resvg (MUCH faster; for the current solid-fill
                          text SVGs -- no blend/mask/gradient -- output matches
                          CairoSVG, so this is the fast path);
      'cairosvg'       -- force CairoSVG (full fidelity, slower).
    Env-driven so the backend can be flipped + A/B'd without a code change."""
    return os.environ.get("TYPO_RASTERIZER", "auto").strip().lower()


def svg_to_png_bytes(svg_text: str, output_width: Optional[int] = None) -> bytes:
    global _last_backend
    pref = _rasterizer_pref()
    t0 = time.time()
    if pref == "resvg":
        _last_backend = "resvg"
        data = _resvg_png_bytes(svg_text, output_width)
    else:
        try:
            import cairosvg
        except (ImportError, OSError) as e:
            if pref == "cairosvg":
                raise
            _note_fallback(f"import failed: {e}")
            _last_backend = "resvg"
            data = _resvg_png_bytes(svg_text, output_width)
        else:
            kwargs = {"output_width": int(output_width)} if output_width else {}
            try:
                data = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), **kwargs)
                _last_backend = "cairosvg"
            except OSError as e:
                # cairosvg imported but its native cairo library is missing at call time.
                if pref == "cairosvg":
                    raise
                _note_fallback(f"native lib missing: {e}")
                _last_backend = "resvg"
                data = _resvg_png_bytes(svg_text, output_width)
    dt = time.time() - t0
    if dt > 1.0:   # surface slow rasterizations (the usual render bottleneck)
        _log.warning("raster backend=%s out_width=%s svg_kb=%d took=%.2fs",
                     _last_backend, output_width, len(svg_text) // 1024, dt)
    return data


def _resvg_png_bytes(svg_text: str, output_width: Optional[int] = None) -> bytes:
    import resvg_py

    kwargs = {}
    if output_width:
        kwargs["width"] = int(output_width)
    return bytes(resvg_py.svg_to_bytes(svg_string=svg_text, **kwargs))


def write_png(svg_text: str, out_path: Path, output_width: Optional[int] = None) -> Path:
    data = svg_to_png_bytes(svg_text, output_width=output_width)
    out_path.write_bytes(data)
    return out_path
