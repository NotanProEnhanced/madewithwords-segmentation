"""FastAPI entrypoint for the isolated typography portrait engine.

Phase 1: project structure + health endpoint.
Phase 2: image upload -> silhouette / edge / landmark debug images.
Later phases add /render.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import admin as admin_mod
from . import orders as orders_db
from . import moderation
from . import suggest
from . import printful, products
from . import gallery_catalog
from .config import (
    BIOMETRIC_CONSENT_VERSION,
    BLOCKED_REGIONS,
    CONSENT_RETENTION_DAYS,
    CURRENCY,
    ENABLE_DEBUG_ENDPOINTS,
    GALLERY_ENABLED,
    RENDER_CONCURRENCY,
    DATA_DIR,
    DOWNLOAD_PNG_WIDTH,
    DOWNLOAD_PRICE_CENTS,
    GEO_COUNTRY_HEADER,
    GEO_REGION_HEADER,
    MAX_IMAGE_MP,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    OUTPUTS_DIR,
    PREVIEW_PNG_WIDTH,
    PRINTFUL_API_TOKEN,
    PRINTFUL_CONFIRM,
    PRIVATE_DIR,
    PUBLIC_BASE_URL,
    RETENTION_DAYS,
    STATIC_DIR,
    STRIPE_SECRET_KEY,
    STRIPE_WEBHOOK_SECRET,
    UMAMI_HOST,
    UMAMI_HOSTNAME,
    UMAMI_WEBSITE_ID,
    RenderConfig,
)
from .pipeline.analyze import analyze_image
from .pipeline.capabilities import probe
from .pipeline.debugviz import render_debug_set
from .pipeline.edges import detect_edges
from .pipeline.landmarks import detect_landmarks, haar_face_bbox
from .pipeline.pathgen import catmull_rom_to_bezier_d
from .pipeline.portrait import build_portrait
from .pipeline.preprocess import load_and_normalize
from .pipeline.quality import assess_portrait_input
from .pipeline.raster import write_png
from .pipeline.silhouette import extract_silhouette
from .pipeline.svgbuild import SvgDoc, validate_svg
from .pipeline.tonal import _PRINT_ASPECT
from .pipeline.warnings import WarningCollector


def _parse_words(words: Optional[str], words_json: Optional[str]) -> List[str]:
    """Accept words as a JSON array string, or split on commas, newlines AND
    spaces. Splitting on whitespace matters: a space-separated list like
    "grace brilliant mom" must become separate words, not one glued-together
    token that only fits the widest part of the silhouette."""
    if words_json:
        try:
            data = json.loads(words_json)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    if words:
        raw = words.replace(",", " ").replace("\n", " ")
        return [w for w in raw.split() if w]
    return []

# Stroke styling for region debug output (hex only).
_REGION_COLORS = {
    "silhouette": "#000000",
    "jaw_line": "#005f73",
    "brow_line": "#bb3e03",
    "lip_line": "#d00000",
}

app = FastAPI(title="Typography Portrait Engine", version=__version__)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Gather (crowd-sourced words) — ADDITIVE + FLAG-GATED. Mounted only when
# TYPO_GATHER_ENABLED is on, so it is entirely inert in production by default and
# cannot affect the studio/checkout paths. See app/gather.py + the spec.
try:
    from .config import GATHER_ENABLED
    if GATHER_ENABLED:
        from . import gather as _gather
        _gather.init_db()
        app.include_router(_gather.router)
except Exception as _e:  # never let an optional feature break app startup
    import logging as _logging
    _logging.getLogger("typortrait").warning("gather feature not mounted: %s", _e)

# Phase C: admin review dashboard + email notifications for consented reels.
app.include_router(admin_mod.router)


def _cleanup_old_files() -> int:
    """Delete previews and stored job inputs older than the retention window so
    the Privacy Policy's deletion statement stays accurate. Consent records are
    kept on a SEPARATE, much longer window (CONSENT_RETENTION_DAYS): they are the
    evidence we must retain to demonstrate consent + the age/guardian attestation,
    and they hold no biometric data (the photo they relate to is deleted on the
    short window). Returns count removed."""
    import time
    now = time.time()
    photo_cutoff = now - RETENTION_DAYS * 86400
    consent_cutoff = now - CONSENT_RETENTION_DAYS * 86400
    removed = 0
    for d in (OUTPUTS_DIR, PRIVATE_DIR):
        try:
            for f in d.iterdir():
                try:
                    if not f.is_file():
                        continue
                    is_consent = f.name.endswith(".consent.json") or f.name.endswith(".biometric_consent.json")
                    cutoff = consent_cutoff if is_consent else photo_cutoff
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    pass
        except OSError:
            pass
    return removed


@app.on_event("startup")
def _start_retention_sweeper() -> None:
    """Run the cleanup at startup, then every 12 hours, in a daemon thread."""
    import threading, time

    def loop():
        while True:
            try:
                _cleanup_old_files()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(12 * 3600)

    threading.Thread(target=loop, daemon=True).start()


@app.on_event("startup")
def _init_orders_db() -> None:
    """Create the orders table if missing. Safe to call repeatedly; needed for
    the print-on-demand order lifecycle (no-op for the digital-only path)."""
    try:
        orders_db.init_db()
    except Exception:  # noqa: BLE001
        pass


@app.on_event("startup")
def _warm_render_models() -> None:
    """Cold-load the MediaPipe face-mesh + selfie-segmentation models at boot, in a
    daemon thread, so the FIRST real user render doesn't pay the multi-second model
    load (which can blow the request timeout right after a deploy). Seeds the
    analysis cache with a throwaway synthetic image; harmless if it fails."""
    import threading

    def warm():
        try:
            import cv2
            import numpy as np
            from .pipeline.warnings import WarningCollector
            img = np.full((256, 256, 3), 210, np.uint8)
            cv2.circle(img, (128, 122), 66, (140, 140, 140), -1)   # rough head shape
            ok, buf = cv2.imencode(".png", img)
            if ok:
                _cached_analyze(buf.tobytes(), RenderConfig(), WarningCollector())
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=warm, daemon=True).start()


@app.on_event("startup")
def _start_admin_services() -> None:
    """Promote any legacy .queue markers to .review.json records, then start
    the background email scanner that notifies the admin of new queued reels."""
    try:
        admin_mod.migrate_legacy_queue_markers()
    except Exception:  # noqa: BLE001
        pass
    admin_mod.start_scanner()


@app.get("/")
def index(request: Request) -> RedirectResponse:
    # This deployment (VPS / app.typortrait.com) is the studio app. The marketing
    # landing page is served separately from a static host (typortrait.com).
    # Forward the query string so the pretty URL (e.g. /?brand=lovedinwords)
    # carries the brand through to the studio file instead of dropping it.
    q = request.url.query
    return RedirectResponse(url="/static/index.html" + (f"?{q}" if q else ""))


@app.get("/health")
def health() -> JSONResponse:
    caps = probe()
    return JSONResponse(
        {
            "ok": True,
            "service": "typography-portrait-engine",
            "version": __version__,
            "capabilities": caps,
            "render": {
                "concurrency_limit": RENDER_CONCURRENCY,
                "in_flight": _render_inflight,
                "queued": _render_waiting,
            },
        }
    )


@app.get("/health/upstream")
def health_upstream() -> JSONResponse:
    """Deep health: actually reach Stripe + Printful (cached ~60s). Kept separate
    from /health so uptime monitors can poll the fast, local check, and this one
    can be alarmed on independently. 503 if any CONFIGURED upstream is unhealthy."""
    from .upstream import check_all
    res = check_all()
    return JSONResponse(
        {
            "ok": res["ok"],
            "service": "typography-portrait-engine",
            "version": __version__,
            "upstream": res,
        },
        status_code=200 if res["ok"] else 503,
    )


@app.post("/debug/preprocess")
async def debug_preprocess(image: UploadFile = File(...)) -> JSONResponse:
    if not ENABLE_DEBUG_ENDPOINTS:   # ungated face processing -> disabled in prod
        raise HTTPException(status_code=404)
    warns = WarningCollector()
    img_bytes = await image.read()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "empty_upload"}, status_code=400)

    cfg = RenderConfig()
    try:
        img = load_and_normalize(img_bytes, cfg.work_max_dim, warns)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    # Face: MediaPipe landmarks if available, else Haar bbox fallback.
    landmarks = detect_landmarks(img, warns)
    face_bbox = None
    if landmarks is not None:
        face_bbox = landmarks.bbox
    else:
        face_bbox = haar_face_bbox(img, warns)

    sil = extract_silhouette(img, warns, face_bbox=face_bbox)
    edges = detect_edges(img, warns, cfg.canny_low, cfg.canny_high, mask=sil.mask)

    job_id = uuid.uuid4().hex[:12]
    files = render_debug_set(OUTPUTS_DIR, job_id, img, sil, edges, landmarks, face_bbox)
    file_urls = {k: f"/outputs/{v}" for k, v in files.items()}

    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "working_size": {"w": img.w, "h": img.h},
            "original_size": {"w": img.orig_w, "h": img.orig_h},
            "face_detected": face_bbox is not None,
            "face_source": "mediapipe" if landmarks is not None else ("haar" if face_bbox else "none"),
            "face_bbox": [float(v) for v in face_bbox] if face_bbox else None,
            "silhouette": {
                "bbox": [int(v) for v in sil.bbox],
                "coverage": round(sil.coverage, 4),
                "confidence": round(sil.confidence, 3),
            },
            "debug_images": file_urls,
            "warnings": warns.as_list(),
        }
    )


@app.post("/debug/regions")
async def debug_regions(image: UploadFile = File(...)) -> JSONResponse:
    """Phase 3: derive region paths and render them as a stroked debug SVG/PNG."""
    if not ENABLE_DEBUG_ENDPOINTS:   # ungated face processing -> disabled in prod
        raise HTTPException(status_code=404)
    warns = WarningCollector()
    img_bytes = await image.read()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "empty_upload"}, status_code=400)

    cfg = RenderConfig()
    try:
        an = analyze_image(img_bytes, cfg, warns)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    doc = SvgDoc(width=an.img.w, height=an.img.h, background="#ffffff")
    region_summary = []
    for i, rp in enumerate(an.regions.paths):
        d = catmull_rom_to_bezier_d(rp.points, rp.closed)
        color = _REGION_COLORS.get(rp.name, "#000000")
        sw = 2.4 if rp.kind == "primary" else 1.4
        doc.add_path(f"{rp.name}_{i}", d, stroke=color, fill="none", stroke_width=sw)
        region_summary.append({"name": rp.name, "kind": rp.kind, "points": int(len(rp.points)), "closed": rp.closed})

    svg_text = doc.to_svg()
    try:
        validate_svg(svg_text)
    except (ValueError, Exception) as e:  # noqa: BLE001
        warns.error("svg", "validation_failed", str(e))
        return JSONResponse({"ok": False, "error": "svg_invalid", "detail": str(e), "warnings": warns.as_list()}, status_code=500)

    job_id = uuid.uuid4().hex[:12]
    svg_path = OUTPUTS_DIR / f"{job_id}_regions.svg"
    svg_path.write_text(svg_text, encoding="utf-8")
    png_path = OUTPUTS_DIR / f"{job_id}_regions.png"
    write_png(svg_text, png_path)

    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "working_size": {"w": an.img.w, "h": an.img.h},
            "face_source": an.face_source,
            "regions": region_summary,
            "region_names": an.regions.names(),
            "svg": f"/outputs/{svg_path.name}",
            "png": f"/outputs/{png_path.name}",
            "warnings": warns.as_list(),
        }
    )


def _allowed_size_mf(face_frac):
    """Which word-size options (as min_font_px values) stay readable for a given
    subject framing. Small/fine type on a small or far face is unreadable, so the
    studio offers only sizes whose smallest tier still reads at this face size.
    Giant + Large are always offered; Medium needs a reasonably-sized face;
    Small needs a close-up. Thresholds tuned against real renders."""
    allow = [120.0, 57.0]                       # Giant, Large -- always
    if face_frac is None or face_frac >= 0.16:  # Medium
        allow.append(27.0)
    if face_frac is None or face_frac >= 0.28:  # Small (close-up only)
        allow.append(13.0)
    return sorted(allow, reverse=True)


# Faces filling >= this share of the frame are "close-ups": they carry bold
# Large type cleanly (fewer, more-readable words = less overlap). Looser/mid
# framings default to Medium for more words and finer detail. Tunable.
_CLOSEUP_FRAC = 0.42


def _recommended_size_mf(face_frac):
    """The DEFAULT word size to pre-select for this framing. We default every
    framing to Large (57): it's the most legible, lands the likeness cleanly, and
    avoids the wall-of-tiny-words look of smaller tiers. Users can still step down
    to Medium for more detail. Large is always an offered size."""
    allowed = _allowed_size_mf(face_frac)
    return 57.0 if 57.0 in allowed else allowed[0]   # Large (always offered)


def _image_too_large(data: bytes) -> bool:
    """True if the upload's pixel count exceeds MAX_IMAGE_PIXELS. We read only the
    header dimensions (no pixel decode), so this is cheap and -- crucially -- runs
    BEFORE any OpenCV/Pillow decode, so a 200+ MP photo is rejected with a friendly
    message instead of tripping Pillow's decompression-bomb guard (whose raw error
    used to leak to users) or ballooning the worker's memory. Fails OPEN (returns
    False) if dimensions can't be read -- the byte-size cap is the backstop."""
    try:
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            w, h = im.size
        return bool(int(w) * int(h) > MAX_IMAGE_PIXELS)
    except Exception:  # noqa: BLE001
        return False


def _apply_crop(data: bytes, crop: Optional[str]) -> bytes:
    """Crop image bytes to a rectangle given as four comma-separated fractions
    'x,y,w,h' (of the whole image), re-encoded as PNG. Returns the original bytes
    on absent/invalid crop or any failure. Done server-side so the crop applies to
    every format and resolution with no client-side canvas encoding to go wrong,
    and so the stored source (and every downstream render/recompose) is cropped."""
    if not crop:
        return data
    try:
        parts = [float(v) for v in str(crop).split(",")]
        if len(parts) != 4:
            return data
        fx, fy, fw, fh = parts
        if fw <= 0.0 or fh <= 0.0:
            return data
        import cv2 as _cv2
        import numpy as _np
        arr = _cv2.imdecode(_np.frombuffer(data, _np.uint8), _cv2.IMREAD_COLOR)
        if arr is None:
            return data
        h, w = arr.shape[:2]
        x0 = max(0, min(w - 1, int(round(fx * w))))
        y0 = max(0, min(h - 1, int(round(fy * h))))
        x1 = max(x0 + 1, min(w, int(round((fx + fw) * w))))
        y1 = max(y0 + 1, min(h, int(round((fy + fh) * h))))
        sub = arr[y0:y1, x0:x1]
        if sub.size == 0:
            return data
        ok, buf = _cv2.imencode(".png", sub)
        return buf.tobytes() if ok else data
    except Exception:  # noqa: BLE001
        return data


# --- Render concurrency: keep heavy CPU work OFF the event loop --------------
# A render (MediaPipe + OpenCV + rasterize) is blocking CPU work. Running it
# directly in an `async def` handler freezes the single event loop for its whole
# duration, so every other request (even /health) stalls behind it. We offload it
# into a worker thread via asyncio.to_thread (the loop stays responsive) and cap
# concurrency with a semaphore so a burst queues instead of thrashing CPU/RAM.
# Caveat: MediaPipe inference is likely GIL-bound, so this gives responsiveness +
# partial overlap, NOT linear CPU scaling -- real parallelism needs separate
# processes (uvicorn --workers / a render queue). See DEPLOY.md.
_render_sem = asyncio.Semaphore(RENDER_CONCURRENCY)
_render_inflight = 0    # renders currently executing (holding a permit)
_render_waiting = 0     # renders queued, waiting for a permit


async def _bounded_to_thread(fn, *args, **kwargs):
    """Run a blocking CPU-bound call in a worker thread, capped by _render_sem.
    Tracks in-flight / queued counts (surfaced on /health) so saturation is
    visible during load tests -> tells you when to move to worker processes."""
    global _render_inflight, _render_waiting
    _render_waiting += 1
    try:
        await _render_sem.acquire()
    finally:
        _render_waiting -= 1
    _render_inflight += 1
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    finally:
        _render_inflight -= 1
        _render_sem.release()


# --- Per-photo analysis cache ------------------------------------------------
# The result screen re-renders the SAME photo many times (every swatch colour,
# size, and the style-recommendation probes). Each /render used to re-run
# analyze_image -- MediaPipe face mesh + selfie segmentation, by far the most
# expensive step -- from scratch. But that analysis depends ONLY on the (cropped)
# image bytes and any manual mask, not on ink/style/size, so we cache it. The
# first render of a photo pays for MediaPipe; every later swatch/style/size render
# reuses it and skips straight to typography + rasterizing. We hand back a deep
# copy so a renderer can never mutate the cached, pristine analysis.
import copy as _copy
import hashlib as _hashlib
import threading as _threading
from collections import OrderedDict as _OrderedDict

_analysis_cache = _OrderedDict()         # key -> pristine Analysis
_analysis_lock = _threading.Lock()
_ANALYSIS_CACHE_MAX = 12                  # ~12 photos in flight; bounded memory


def _analysis_key(img_bytes: bytes, manual_mask) -> str:
    h = _hashlib.sha256(img_bytes).hexdigest()
    if manual_mask is not None:
        try:
            h += ":" + _hashlib.sha256(manual_mask.tobytes()).hexdigest()[:16]
        except Exception:  # noqa: BLE001
            h += ":mask"
    return h


def _cached_analyze(img_bytes, cfg, warns, manual_mask=None):
    """analyze_image with an LRU cache keyed on the image+mask. Returns a deep
    copy on a hit (so renderers can mutate freely). Degrades to a fresh analysis
    if cloning ever fails -- correctness never depends on the cache."""
    key = _analysis_key(img_bytes, manual_mask)
    with _analysis_lock:
        hit = _analysis_cache.get(key)
        if hit is not None:
            _analysis_cache.move_to_end(key)
    if hit is not None:
        try:
            return _copy.deepcopy(hit)
        except Exception:  # noqa: BLE001
            pass            # cached entry won't clone -> fall through to fresh
    an = analyze_image(img_bytes, cfg, warns, manual_mask=manual_mask)
    try:
        pristine = _copy.deepcopy(an)
        with _analysis_lock:
            _analysis_cache[key] = pristine
            _analysis_cache.move_to_end(key)
            while len(_analysis_cache) > _ANALYSIS_CACHE_MAX:
                _analysis_cache.popitem(last=False)
    except Exception:  # noqa: BLE001
        pass                # uncopyable -> just don't cache; still correct
    return an


# --- Privacy / biometric compliance gates -----------------------------------
# Face-mesh analysis is "biometric" (BIPA) / "special-category" (GDPR), so the
# two endpoints that touch a face (/measure, /render) must (1) refuse blocked
# regions and (2) require explicit consent BEFORE processing. See COMPLIANCE.md.
def _client_region(request: Request):
    """(country, region) upper-cased from the proxy's geo headers; '' if absent."""
    country = (request.headers.get(GEO_COUNTRY_HEADER) or "").strip().upper()
    region = (request.headers.get(GEO_REGION_HEADER) or "").strip().upper()
    return country, region


def _region_token(request: Request) -> str:
    c, r = _client_region(request)
    return "-".join(t for t in (c, r) if t)


def _blocked_region(request: Request):
    """The matched blocked-region token if this visitor is in a blocked region,
    else None. Fails OPEN (returns None) when the proxy gives no geo signal."""
    if not BLOCKED_REGIONS:
        return None
    c, r = _client_region(request)
    if not c and not r:
        return None
    candidates = {t for t in (c, r, (f"{c}-{r}" if c and r else "")) if t}
    hit = candidates & BLOCKED_REGIONS
    return next(iter(hit)) if hit else None


def _consent_given(val) -> bool:
    return (val or "").strip().lower() in ("on", "true", "1", "yes")


_GEO_BLOCK_DETAIL = ("To comply with the Illinois Biometric Information Privacy "
                     "Act (BIPA), Typortrait is not currently available to visitors "
                     "located in Illinois. We're sorry for the inconvenience.")
_BIO_CONSENT_DETAIL = ("Creating a portrait analyses facial geometry from your photo. "
                       "Please confirm the biometric-data consent to continue.")


def _compliance_gate(request: Request, biometric_consent: str):
    """Shared gate for the face-processing endpoints. Returns a JSONResponse to
    short-circuit (geo-block 451 or consent-required 400), or None to allow."""
    blocked = _blocked_region(request)
    if blocked:
        return JSONResponse({"ok": False, "error": "region_unavailable",
                             "detail": _GEO_BLOCK_DETAIL}, status_code=451)
    if not _consent_given(biometric_consent):
        return JSONResponse({"ok": False, "error": "biometric_consent_required",
                             "detail": _BIO_CONSENT_DETAIL}, status_code=400)
    return None


@app.post("/measure")
async def measure(request: Request, image: UploadFile = File(...), crop: Optional[str] = Form(None),
                  biometric_consent: str = Form("")) -> JSONResponse:
    """Lightweight pre-render check: detect the subject's face and return its
    width as a fraction of the photo, plus the word sizes that stay readable for
    that framing. The studio calls this on upload so it can grey out size options
    (e.g. Small) that would produce unreadable type for a far/small subject."""
    gate = _compliance_gate(request, biometric_consent)
    if gate is not None:
        return gate
    data = await image.read()
    if not data:
        return JSONResponse({"ok": False, "error": "empty_upload"}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"ok": False, "error": "file_too_large"}, status_code=413)
    # Surface the too-large case at upload (not after Create) so the user learns early.
    if _image_too_large(data):
        return JSONResponse({"ok": False, "error": "image_too_large"}, status_code=413)
    data = _apply_crop(data, crop)         # size-gate on the cropped framing when a crop is set
    warns = WarningCollector()
    try:
        from .pipeline.preprocess import load_and_normalize
        from .pipeline.landmarks import detect_faces, haar_face_bbox
        cfg = RenderConfig()
        img = load_and_normalize(data, cfg.work_max_dim, warns)
        faces = detect_faces(img, warns)
        bbox = faces[0].bbox if faces else haar_face_bbox(img, warns)
        w = float(img.gray.shape[1]) or 1.0
        h = float(img.gray.shape[0]) or 1.0
        face_frac = round(float(bbox[2]) / w, 4) if bbox else None
        # Face box as fractions of the photo (resolution-independent) so the studio
        # can open the crop editor framed on the subject -- useful when the upload
        # is a photo OF a framed picture and the person fills only part of it.
        face_box = ([round(float(bbox[0]) / w, 4), round(float(bbox[1]) / h, 4),
                     round(float(bbox[2]) / w, 4), round(float(bbox[3]) / h, 4)]
                    if bbox else None)
        n_faces = len(faces)            # 0/1 via haar fallback; >=2 only when MediaPipe sees a group
    except Exception as e:  # noqa: BLE001
        # On any failure, allow all sizes (the renderer's floor still protects).
        return JSONResponse({"ok": True, "face_frac": None, "faces": 0, "face_box": None,
                             "sizes": _allowed_size_mf(None), "default": _recommended_size_mf(None)})
    return JSONResponse({"ok": True, "face_frac": face_frac, "faces": n_faces, "face_box": face_box,
                         "sizes": _allowed_size_mf(face_frac), "default": _recommended_size_mf(face_frac)})


@app.post("/suggest-words")
async def suggest_words_endpoint(
    request: Request,
    story: str = Form(""),
    nickname: str = Form(""),
    sayings: str = Form(""),
    three_words: str = Form(""),
    hobby: str = Form(""),
) -> JSONResponse:
    """Memorial 'Share their story' step: take a pasted tribute/obituary and/or the
    optional structured details (nickname, sayings, three words, hobby) and return
    suggested portrait words. Convenience only -- fails soft to an empty list (no key
    / API error / empty input), so the studio falls back to manual entry and the flow
    never blocks on it. Only the words are returned, never the story or details. See
    suggest.py for the (billed) OpenAI call and caching."""
    text = (story or "").strip()
    details = {
        "nickname": (nickname or "").strip(),
        "sayings": (sayings or "").strip(),
        "three_words": (three_words or "").strip(),
        "hobby": (hobby or "").strip(),
    }
    if not text and not any(details.values()):
        return JSONResponse({"ok": True, "words": []})
    if not suggest.enabled():
        return JSONResponse({"ok": True, "words": [], "available": False})
    words = await asyncio.to_thread(suggest.suggest_words, text, details)
    return JSONResponse({"ok": True, "words": words, "available": True})


@app.get("/mask/{job}")
def auto_mask(job: str):
    """Return the AUTOMATIC background mask (PNG, white = subject) for a stored
    job, so the manual background editor can open pre-loaded with the auto cutout
    to refine. Computed from the stored (cropped) source; consent/geo were already
    cleared when the job was created."""
    if not _JOB_RE.match((job or "").strip().lower()):
        raise HTTPException(status_code=404)
    src = PRIVATE_DIR / f"{job}.src"
    if not src.exists():
        raise HTTPException(status_code=404)
    # Resume from the user's last saved manual mask if there is one, so re-opening
    # the editor continues their edit instead of resetting to the auto cutout.
    bgm = PRIVATE_DIR / f"{job}.bgmask.png"
    if bgm.exists():
        return Response(content=bgm.read_bytes(), media_type="image/png")
    try:
        import cv2
        from .pipeline.preprocess import load_and_normalize
        from .pipeline.landmarks import detect_faces, haar_face_bbox
        from .pipeline.silhouette import extract_silhouette
        warns = WarningCollector()
        img = load_and_normalize(src.read_bytes(), RenderConfig().work_max_dim, warns)
        faces = detect_faces(img, warns)
        fbox = faces[0].bbox if faces else haar_face_bbox(img, warns)
        sil = extract_silhouette(img, warns, face_bbox=fbox)
        ok_enc, buf = cv2.imencode(".png", sil.mask)
        if not ok_enc:
            raise HTTPException(status_code=500, detail="encode_failed")
        return Response(content=buf.tobytes(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/render")
async def render(
    request: Request,
    image: UploadFile = File(...),
    words: Optional[str] = Form(None),
    words_json: Optional[str] = Form(None),
    min_font_px: Optional[float] = Form(None),
    uppercase: bool = Form(True),
    background_hex: Optional[str] = Form(None),
    foreground_hex: Optional[str] = Form(None),
    ink: str = Form("navy"),
    ink_hex: Optional[str] = Form(None),   # for ink="custom": the user-picked colour
    style: str = Form("mosaic"),
    message: Optional[str] = Form(None),
    flow: Optional[str] = Form(None),   # displacement only: stream the text as a MESSAGE
                                        # (in order, looping) instead of a weighted word list
    poster: bool = Form(False),
    title: Optional[str] = Form(None),
    caption: Optional[str] = Form(None),
    png_width: int = Form(2000),
    render_w: int = Form(2600),
    remove_bg: bool = Form(True),
    light: bool = Form(False),
    ground: str = Form("navy"),
    ref: str = Form(""),
    brand: str = Form(""),
    crop: Optional[str] = Form(None),
    biometric_consent: str = Form(""),
    mask: Optional[str] = Form(None),
) -> JSONResponse:
    """Render a typographic portrait: validated SVG + PNG from approved words."""
    warns = WarningCollector()
    gate = _compliance_gate(request, biometric_consent)   # geo-block + biometric consent BEFORE any face processing
    if gate is not None:
        return gate
    ref_clean = re.sub(r"[^A-Za-z0-9_-]", "", ref or "")[:40]     # referral/source tag (persists)
    brand_clean = re.sub(r"[^A-Za-z0-9_-]", "", brand or "")[:40]  # ACTIVE brand skin (gates brand UX)
    img_bytes = await image.read()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "empty_upload"}, status_code=400)
    if len(img_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"ok": False, "error": "file_too_large",
             "detail": "That image is too large — please use one under 25 MB."},
            status_code=413,
        )
    # Reject extreme pixel counts BEFORE any decode (a small file can still be a
    # 200+ MP photo). Friendly message; the frontend maps the code too.
    if _image_too_large(img_bytes):
        return JSONResponse(
            {"ok": False, "error": "image_too_large",
             "detail": (f"This photo is very large (over {MAX_IMAGE_MP} megapixels) and can’t be "
                        "processed. Please use a smaller copy — on a phone, take a screenshot of "
                        "the photo and upload that; on a computer, resize or export it smaller.")},
            status_code=413,
        )
    # Crop to the studio's framing BEFORE anything else, so the stored source and
    # every downstream render/recompose use the cropped image (no client encoding).
    img_bytes = _apply_crop(img_bytes, crop)

    word_list = _parse_words(words, words_json)
    if not word_list:
        return JSONResponse({"ok": False, "error": "no_words"}, status_code=400)

    # Content moderation: the words/message/title/caption become the portrait, so
    # block hate / sexual / harassing text before rendering. Fails OPEN (allows)
    # when unconfigured or unavailable -- Terms + attestation are the backstop.
    mod_text = " ".join(t for t in (" ".join(word_list), message, title, caption) if t).strip()
    mod = await asyncio.to_thread(moderation.check_text, mod_text)
    if mod.get("flagged"):
        return JSONResponse(
            {"ok": False, "error": "content_flagged",
             "detail": "Your words include content that violates our Terms (hate, explicit, or "
                       "harassing content). Please revise them and try again."},
            status_code=422)

    cfg = RenderConfig()
    if min_font_px is not None:
        cfg.min_font_px = float(min_font_px)
    if background_hex:
        cfg.background_hex = background_hex
    if foreground_hex:
        cfg.foreground_hex = foreground_hex
    try:
        cfg.validate()
    except ValueError as e:
        return JSONResponse({"ok": False, "error": "bad_config", "detail": str(e)}, status_code=400)

    # Manual background mask (optional): a user-painted cutout that overrides auto
    # segmentation. A malformed mask just falls back to automatic.
    manual_mask_arr = None
    if mask:
        try:
            import base64
            import cv2
            import numpy as _np
            _b64 = mask.split(",", 1)[1] if "," in mask else mask
            _arr = cv2.imdecode(_np.frombuffer(base64.b64decode(_b64), _np.uint8), cv2.IMREAD_GRAYSCALE)
            if _arr is not None and _arr.size:
                manual_mask_arr = _arr
        except Exception:  # noqa: BLE001
            manual_mask_arr = None
    try:
        an = await _bounded_to_thread(_cached_analyze, img_bytes, cfg, warns, manual_mask=manual_mask_arr)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e), "warnings": warns.as_list()}, status_code=400)

    # Input-quality gate: give actionable feedback instead of a silently bad
    # portrait when the photo isn't a usable single head-and-shoulders shot.
    gate = assess_portrait_input(an)
    for issue in gate:
        (warns.error if issue.severity == "error" else warns.warn)("input", issue.code, issue.message)
    blocking = [i for i in gate if i.severity == "error"]
    if blocking:
        return JSONResponse(
            {"ok": False, "error": "unsuitable_image", "detail": blocking[0].message,
             "warnings": warns.as_list()},
            status_code=422,
        )

    from .pipeline.tonal import _PALETTES, _GRADIENTS, render_layered_png, custom_poster
    # "custom" = a user-picked colour (a (ground, ink) poster pair built from the
    # hex). Only the layered Words/Message renderer honours it; everything else
    # falls back. Invalid/absent hex -> not custom.
    custom_tuple = custom_poster(ink_hex) if (ink == "custom" and ink_hex) else None
    if ink in _PALETTES or ink in _GRADIENTS or ink in ("photo", "photo_paper"):
        ink_choice = ink
    elif custom_tuple:
        ink_choice = "custom"
    else:
        ink_choice = "navy"
    # User-facing styles: "displacement" = type-follows-the-form portrait (its own
    # raster renderer + ground choice); "message" = poster rows; anything else =
    # "words" (the scattered mosaic). Words/Passage share the layered renderer.
    is_displacement = (style == "displacement")
    disp_flow = is_displacement and str(flow or "").strip().lower() in ("1", "true", "yes", "on")
    ground_choice = ground if ground in ("paper", "navy", "black") else "navy"
    style_choice = "displacement" if is_displacement else (
        "message" if style in ("message", "poster", "story") else "words")
    render_w_eff = max(700, min(3000, int(render_w)))
    is_thumb = int(png_width) < 600         # small png_width => a swatch chip render
    if style_choice == "message":
        text = (message or "").strip() or " ".join(word_list)
    else:
        text = " ".join(word_list) or (message or "").strip()

    try:
        preview_w = min(int(png_width), PREVIEW_PNG_WIDTH)
        if is_displacement:
            from .pipeline.displacement import render_displacement_portrait
            disp_words = word_list or text.split()
            # The Sculpt renderer works at supersample x the ANALYSIS resolution
            # (independent of out_width): 5 full-canvas text layers + a warp. At
            # SS=2 that's ~2048px and ~20s. This endpoint only ever produces the
            # on-screen PREVIEW (the paid file is recomposed separately), so use
            # SS=1 for previews/thumbnails -- ~4x fewer pixels, the big memorial win.
            disp_ss = 1 if int(png_width) < 1200 else 2
            png_bytes = await _bounded_to_thread(
                render_displacement_portrait, an, disp_words, ground=ground_choice,
                out_width=max(320, preview_w), supersample=disp_ss, flow=disp_flow,
                uppercase=uppercase, ink=("photo" if ink_choice in ("custom", "photo_paper") else ink_choice))
            runs, ground_hex, mask_svg = [], None, None
        else:
            png_bytes, runs, ground_hex, mask_svg = await _bounded_to_thread(
                render_layered_png, an, text, style_choice, cfg, warns,
                ink=ink_choice, remove_bg=remove_bg, light=light,
                out_width=max(320, preview_w), render_w=render_w_eff, uppercase=uppercase,
                custom=custom_tuple)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e), "warnings": warns.as_list()}, status_code=400)
    except Exception as e:  # noqa: BLE001
        warns.error("render", "render_failed", str(e))
        return JSONResponse({"ok": False, "error": "render_failed", "detail": str(e), "warnings": warns.as_list()}, status_code=500)

    # Fail loud, never silent: the layered (Words/Message) renderer rasterizes
    # SVG, and a cairosvg->resvg fallback quietly drops tonal modulation. Surface
    # it as a warning everywhere; in production set TYPO_REQUIRE_CAIROSVG to make
    # it a hard block so a degraded portrait can never reach a paying customer.
    if not is_displacement:
        import os
        from .pipeline.raster import cairosvg_available
        if not cairosvg_available():
            msg = ("Rasterizer is in DEGRADED mode (resvg fallback; cairosvg "
                   "unavailable) -- tonal modulation may be dropped.")
            require = (os.getenv("TYPO_REQUIRE_CAIROSVG") or "").strip().lower() in ("1", "true", "yes", "on")
            (warns.error if require else warns.warn)("raster", "degraded_rasterizer", msg)

    if warns.has_errors() or not png_bytes:
        return JSONResponse({"ok": False, "error": "render_incomplete", "warnings": warns.as_list()}, status_code=422)

    job_id = uuid.uuid4().hex[:12]
    preview_path = OUTPUTS_DIR / f"{job_id}_preview.png"
    try:
        from .pipeline.watermark import add_watermark
        # Brand-aware mark: partner skins never stamp "typortrait.com" (would divert
        # the partner's referral). `brand` is the active skin from the request.
        preview_path.write_bytes(add_watermark(png_bytes, brand=brand))
    except Exception as e:  # noqa: BLE001
        warns.warn("render", "preview_failed", f"Watermark failed: {e}")
        preview_path.write_bytes(png_bytes)

    # Persist the inputs so the paid, high-res PNG can be recomposed once at
    # download (after payment). Skip for throwaway swatch-thumbnail renders.
    if not is_thumb:
        (PRIVATE_DIR / f"{job_id}.src").write_bytes(img_bytes)
        if manual_mask_arr is not None:
            # Persist the manual background mask so the paid high-res recompose
            # reproduces the same cutout the buyer saw in the preview.
            try:
                import cv2
                ok_enc, _buf = cv2.imencode(".png", manual_mask_arr)
                if ok_enc:
                    (PRIVATE_DIR / f"{job_id}.bgmask.png").write_bytes(_buf.tobytes())
            except Exception:  # noqa: BLE001
                pass
        # Only persist the layout mask for FULL-density renders. A light preview
        # (low render_w -- used for a fast on-screen preview) deliberately does NOT
        # store its mask, so the paid /download recomposes from the recipe at full
        # render_w=2600 (the no-mask path) -- preview is fast, paid file stays crisp.
        if mask_svg and render_w_eff >= 2000:
            (PRIVATE_DIR / f"{job_id}.mask.svg").write_text(mask_svg, encoding="utf-8")
        # Record the biometric-processing consent (BIPA written release / GDPR
        # explicit consent), versioned to the exact notice wording the user saw.
        import time as _t
        (PRIVATE_DIR / f"{job_id}.biometric_consent.json").write_text(json.dumps({
            "biometric_consent": True,
            # Folded into the same required checkbox (see consent_version wording):
            # user attests they are 13+ and, for a child's photo, the parent/guardian.
            "age_guardian_attested": True,
            "consent_version": BIOMETRIC_CONSENT_VERSION,
            "ts": int(_t.time()),
            "region": _region_token(request) or None,
        }), encoding="utf-8")
        (PRIVATE_DIR / f"{job_id}.json").write_text(json.dumps({
            "style": style_choice, "ink": ink_choice, "remove_bg": bool(remove_bg),
            "light": bool(light), "text": text, "uppercase": bool(uppercase),
            "min_font_px": float(cfg.min_font_px), "ground": ground_choice,
            "ink_hex": ink_hex if ink_choice == "custom" else None,   # rebuild the custom colour at download
            "flow": bool(disp_flow),   # displacement message-flow -> paid recompose must match
            "ref": ref_clean, "brand": brand_clean,
        }), encoding="utf-8")

    # Likeness score (how well this render preserves the face) -- the studio
    # compares scores across styles to recommend the best one for this photo.
    try:
        from .pipeline.score import likeness_score
        likeness = likeness_score(an, png_bytes)
    except Exception:  # noqa: BLE001
        likeness = None

    # When the upload was cropped, hand the studio a small JPEG of the CROPPED
    # source so the before/after slider's "before" matches the rendered "after"
    # (the client can't reliably crop the photo itself). Main render only.
    source_url = None
    if not is_thumb and crop:
        try:
            import cv2 as _cv2
            import numpy as _np
            sa = _cv2.imdecode(_np.frombuffer(img_bytes, _np.uint8), _cv2.IMREAD_COLOR)
            if sa is not None:
                sh, sw = sa.shape[:2]
                sc = 760.0 / max(1, sw)
                if sc < 1.0:
                    sa = _cv2.resize(sa, (int(sw * sc), int(sh * sc)), interpolation=_cv2.INTER_AREA)
                sp = OUTPUTS_DIR / f"{job_id}_src.jpg"
                if _cv2.imwrite(str(sp), sa, [int(_cv2.IMWRITE_JPEG_QUALITY), 86]):
                    source_url = f"/outputs/{sp.name}"
        except Exception:  # noqa: BLE001
            source_url = None

    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "job": job_id,
            "faces": len(an.faces),
            "likeness": likeness,
            "ink": ink_choice,
            "style": style_choice,
            "ground": ground_choice,
            "words_used": word_list,
            "text_runs": [
                {"region": r.region, "font_size": r.font_size, "kind": r.kind, "chars": len(r.text)}
                for r in runs
            ],
            "preview": f"/outputs/{preview_path.name}" if preview_path.exists() else None,
            "source": source_url,
            "price_cents": DOWNLOAD_PRICE_CENTS,
            "currency": CURRENCY,
            "warnings": warns.as_list(),
        }
    )


@app.get("/pricing")
def pricing() -> JSONResponse:
    """So the page can show the price up front (single source of truth = env)."""
    return JSONResponse({
        "price_cents": DOWNLOAD_PRICE_CENTS,
        "currency": CURRENCY,
        "configured": bool(STRIPE_SECRET_KEY),
    })


@app.api_route("/gallery", methods=["GET", "HEAD"], response_class=HTMLResponse)
def gallery_page() -> HTMLResponse:
    """The storefront gallery of pre-made typortraits (Bible Collection, Holidays,
    etc.). Static page; the catalog + art live under /static/gallery/."""
    if not GALLERY_ENABLED:
        raise HTTPException(status_code=404, detail="not found")
    page = STATIC_DIR / "gallery.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="gallery not found")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.post("/gallery/checkout")
def gallery_checkout(request: Request, item: str = Form(...), sku: str = Form("digital")) -> JSONResponse:
    """Create a Stripe Checkout session for a fixed gallery item. Decoupled from
    the personalized /checkout: a gallery item is pre-rendered art, validated
    against the catalog, with a master PNG on disk. Handles the digital download
    and single-variant prints (posters/canvas/framed); prints are fulfilled by
    Printful in the webhook via the item's master (no compose step)."""
    if not GALLERY_ENABLED:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": False, "error": "payments_unconfigured"}, status_code=503)
    it = gallery_catalog.get(item)
    if not it:
        return JSONResponse({"ok": False, "error": "unknown_item"}, status_code=404)
    if gallery_catalog.master_path(item) is None:
        # Item is in the catalog but its artwork isn't in place yet.
        return JSONResponse({"ok": False, "error": "art_missing"}, status_code=409)
    product = products.get(sku)
    if not product:
        return JSONResponse({"ok": False, "error": "unknown_product"}, status_code=400)
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    title = str(it.get("title") or "Typortrait")[:120]

    # --- Digital download ----------------------------------------------------
    if not product.physical:
        try:
            session = stripe.checkout.Session.create(
                mode="payment",
                line_items=[{
                    "quantity": 1,
                    "price_data": {
                        "currency": CURRENCY,
                        "unit_amount": DOWNLOAD_PRICE_CENTS,
                        "product_data": {"name": f"Typortrait — {title} (digital download)"},
                    },
                }],
                metadata={"gallery_item": item},
                success_url=f"{_req_base(request)}/gallery/download?item={item}&session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{_req_base(request)}/gallery?canceled=1",
            )
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": "stripe_error", "detail": str(e)}, status_code=502)
        return JSONResponse({"ok": True, "url": session.url})

    # --- Physical print ------------------------------------------------------
    if not PRINTFUL_API_TOKEN:
        return JSONResponse({"ok": False, "error": "fulfillment_unconfigured"}, status_code=503)
    # The gallery sells single-variant prints only (posters/canvas/framed). Sized
    # apparel, memorial-only bundles, etc. are out of scope for the storefront.
    if product.memorial_only or product.size_variants or product.bundle_items or not product.printful_variant_id:
        return JSONResponse({"ok": False, "error": "unsupported_product"}, status_code=409)
    variant_id = product.printful_variant_id
    eff_price, eff_ship = products.price_for(product, memorial=False)
    order_id = uuid.uuid4().hex[:16]
    line_items: List[dict] = [{
        "quantity": 1,
        "price_data": {
            "currency": CURRENCY,
            "unit_amount": eff_price,
            "product_data": {"name": f"Typortrait — {title} — {product.name}"},
        },
    }]
    if eff_ship > 0:
        line_items.append({
            "quantity": 1,
            "price_data": {"currency": CURRENCY, "unit_amount": eff_ship,
                           "product_data": {"name": "Shipping (USA)"}},
        })
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=line_items,
            metadata={"gallery_item": item, "sku": sku, "order_id": order_id, "ref": "gallery"},
            shipping_address_collection={"allowed_countries": ["US"]},
            phone_number_collection={"enabled": True},
            success_url=f"{_req_base(request)}/order/{order_id}?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{_req_base(request)}/gallery?canceled=1",
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "stripe_error", "detail": str(e)}, status_code=502)
    try:
        # job_id carries the gallery item with a sentinel prefix so the shared
        # fulfilment/order machinery can tell gallery orders from personalized ones.
        orders_db.create_pending(
            order_id=order_id, stripe_session_id=session.id, job_id=f"gallery:{item}",
            sku=sku, size=None, variant_id=variant_id, price_cents=eff_price,
            shipping_cents=eff_ship, currency=CURRENCY, ref="gallery",
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "order_persist_failed", "detail": str(e)}, status_code=500)
    return JSONResponse({"ok": True, "url": session.url, "order_id": order_id})


def _gallery_print_file(item: str, aspect: float = _PRINT_ASPECT) -> Optional[Path]:
    """Return a print-ready master for a gallery item, padded to `aspect` (the
    product's true width/height) with the art's own background so it's never
    cropped or stretched. Cached per (item, aspect). Mirrors what _ensure_clean_png
    does for personalized jobs, but the gallery master is fixed art (no compose)."""
    src = gallery_catalog.master_path(item)
    if src is None:
        return None
    cache_dir = PRIVATE_DIR / "gallery" / "_print"
    out = cache_dir / f"{item}_{aspect:.4f}.png"
    if out.exists():
        return out
    try:
        import cv2
        from .pipeline.tonal import _fit_print_canvas
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)   # BGR
        if img is None:
            return src
        ground = [int(c) for c in img[0, 0].tolist()]   # corner pixel = the art's ground (BGR)
        fitted = _fit_print_canvas(img, ground, aspect)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), fitted)
        return out
    except Exception:  # noqa: BLE001 -- fall back to the unpadded master
        return src


@app.api_route("/gallery/printful-fetch/{item}", methods=["GET", "HEAD"])
def gallery_printful_fetch(item: str, exp: int, sig: str, a: float = _PRINT_ASPECT):
    """Signed, time-limited URL Printful fetches a gallery item's print master
    from. Same HMAC scheme as /printful-fetch; `a` selects the product aspect."""
    if not GALLERY_ENABLED:
        raise HTTPException(status_code=404, detail="not found")
    if not printful.verify_signed_url(item, exp, sig):
        raise HTTPException(status_code=403, detail="invalid_or_expired_signature")
    aspect = min(2.0, max(0.4, float(a)))
    path = _gallery_print_file(item, aspect)
    if path is None:
        raise HTTPException(status_code=404, detail="unknown_item")
    return FileResponse(str(path), media_type="image/png")


@app.get("/gallery/download")
def gallery_download(item: str, session_id: str):
    """Serve a gallery item's master PNG only after verifying its paid Stripe
    session. Mirrors /download but reads the fixed master (no compose step)."""
    if not GALLERY_ENABLED:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": False, "error": "payments_unconfigured"}, status_code=503)
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        sess = stripe.checkout.Session.retrieve(session_id)
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "bad_session"}, status_code=400)
    paid = getattr(sess, "payment_status", None) == "paid"
    meta = getattr(sess, "metadata", None)
    meta_item = getattr(meta, "gallery_item", None) if meta is not None else None
    if not paid or meta_item != item:
        return JSONResponse({"ok": False, "error": "not_paid"}, status_code=402)
    path = gallery_catalog.master_path(item)
    if path is None:
        return JSONResponse({"ok": False, "error": "art_missing"}, status_code=404)
    return FileResponse(str(path), media_type="image/png", filename=f"typortrait-{item}.png")


@app.get("/products")
def list_products(brand: str = "") -> JSONResponse:
    """Storefront catalog (prices, shipping, sizes) — single source of truth so
    the UI never hardcodes prices. `brand` selects channel pricing (the memorial/
    EverLoved brand carries commission-adjusted prices). `fulfillment_configured`
    tells the front-end whether physical print-on-demand is live."""
    return JSONResponse({
        "ok": True,
        "currency": CURRENCY,
        "products": products.public_catalog(memorial=products.is_memorial_channel(brand)),
        "fulfillment_configured": bool(PRINTFUL_API_TOKEN),
    })


# Per-brand checkout presentation. A ?brand front-end (e.g. lovedinwords) sets the
# attribution source (stored as the recipe "ref"), which here drives the
# checkout/receipt line-item NAME so it matches the brand the customer came from.
# The card STATEMENT DESCRIPTOR is handled account-wide in Stripe (a single neutral
# descriptor that suits every brand), so we do NOT set a per-charge descriptor —
# a per-charge override would fight the account-wide value. Unknown/empty source
# falls back to the Typortrait name.
_CHECKOUT_BRANDS = {
    "lovedinwords": {"name": "Loved in Words"},
}


def _checkout_brand(ref: str) -> dict:
    return _CHECKOUT_BRANDS.get((ref or "").strip().lower(), {"name": "Typortrait"})


def _req_base(request: Request) -> str:
    """Host the user is actually on (from the proxy-passed Host header), so a checkout
    redirect never bounces app<->staging. Falls back to PUBLIC_BASE_URL. ONLY for
    user-facing redirects (Stripe success/cancel) -- server-to-server URLs that must be
    publicly reachable (Printful fetch, og:image, receipt emails) keep PUBLIC_BASE_URL."""
    host = request.headers.get("host")
    if not host:
        return PUBLIC_BASE_URL.rstrip("/")
    proto = (request.headers.get("x-forwarded-proto", "") or "https").split(",")[0].strip() or "https"
    return f"{proto}://{host}"


@app.post("/checkout")
def checkout(
    request: Request,
    job: str = Form(...),
    sku: str = Form("digital"),
    size: Optional[str] = Form(None),
    fmt: str = Form("png"),
) -> JSONResponse:
    """Create a Stripe Checkout session for either the digital download or a
    physical print. `sku` selects the product; the default "digital" preserves
    the original (and live) download flow exactly. Physical orders additionally
    collect a shipping address via Stripe and are fulfilled by Printful through
    the /webhook/stripe handler."""
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": False, "error": "payments_unconfigured"}, status_code=503)
    product = products.get(sku)
    if not product:
        return JSONResponse({"ok": False, "error": "unknown_product"}, status_code=400)
    # The job's inputs (recipe) are persisted at render time; the high-res PNG is
    # composed from them lazily at download / fulfillment time.
    if not (PRIVATE_DIR / f"{job}.json").exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    try:  # referral/source tag captured at render time (partner attribution)
        ref = str(json.loads((PRIVATE_DIR / f"{job}.json").read_text(encoding="utf-8")).get("ref", ""))[:40]
    except Exception:  # noqa: BLE001
        ref = ""
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    brand = _checkout_brand(ref)   # per-brand statement descriptor + line-item name
    eff_price, eff_ship = products.price_for(product, products.is_memorial_channel(ref))   # memorial/EverLoved channel = commission-adjusted price

    # --- Digital download: unchanged from the live synchronous-verify flow ----
    if not product.physical:
        try:
            session = stripe.checkout.Session.create(
                mode="payment",
                line_items=[{
                    "quantity": 1,
                    "price_data": {
                        "currency": CURRENCY,
                        "unit_amount": DOWNLOAD_PRICE_CENTS,
                        "product_data": {"name": f"{brand['name']} — high-resolution download"},
                    },
                }],
                metadata={"job": job, "ref": ref},
                success_url=f"{_req_base(request)}/success?job={job}&session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{_req_base(request)}/static/index.html?canceled=1",
            )
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": "stripe_error", "detail": str(e)}, status_code=502)
        return JSONResponse({"ok": True, "url": session.url})

    # --- Physical print: gated on a configured Printful token + valid size ----
    if not PRINTFUL_API_TOKEN:
        return JSONResponse({"ok": False, "error": "fulfillment_unconfigured"}, status_code=503)
    variant_id = products.resolve_variant_id(product, size)
    if variant_id is None and not product.bundle_items:
        return JSONResponse(
            {"ok": False, "error": "missing_or_invalid_size",
             "sizes": list(product.size_variants.keys()) if product.size_variants else []},
            status_code=400,
        )
    if product.bundle_items:
        variant_id = product.bundle_items[0][0]   # representative variant: keeps the order row + paid-gate happy (fulfilment expands the full bundle)
    order_id = uuid.uuid4().hex[:16]
    ext = "svg" if fmt == "svg" else "png"
    label_size = f" — {size}" if size else ""
    line_items: List[dict] = [{
        "quantity": 1,
        "price_data": {
            "currency": CURRENCY,
            "unit_amount": eff_price,
            "product_data": {"name": f"{brand['name']} — {product.name}{label_size}"},
        },
    }]
    if eff_ship > 0:
        line_items.append({
            "quantity": 1,
            "price_data": {
                "currency": CURRENCY,
                "unit_amount": eff_ship,
                "product_data": {"name": "Shipping (USA)"},
            },
        })
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=line_items,
            metadata={"job": job, "sku": sku, "size": size or "", "order_id": order_id, "fmt": ext, "ref": ref},
            shipping_address_collection={"allowed_countries": ["US"]},
            phone_number_collection={"enabled": True},
            success_url=f"{_req_base(request)}/order/{order_id}?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{_req_base(request)}/static/index.html?canceled=1",
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "stripe_error", "detail": str(e)}, status_code=502)
    try:
        orders_db.create_pending(
            order_id=order_id,
            stripe_session_id=session.id,
            job_id=job,
            sku=sku,
            size=size,
            variant_id=variant_id,
            price_cents=eff_price,
            shipping_cents=eff_ship,
            currency=CURRENCY,
            ref=ref,
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "order_persist_failed", "detail": str(e)},
                            status_code=500)
    return JSONResponse({"ok": True, "url": session.url, "order_id": order_id})


def _stripe_to_dict(obj):
    """Stripe SDK v15+ removed dict-style `.get()` from StripeObject; convert
    to a plain dict by round-tripping through JSON (Stripe objects' __str__
    emits canonical JSON) so the rest of the code can treat them as dicts."""
    if obj is None or isinstance(obj, (str, int, float, bool, list)):
        return obj
    if isinstance(obj, dict):
        return obj
    try:
        return json.loads(str(obj))
    except (ValueError, TypeError):
        return {}


def _ensure_clean_png(job: str, aspect: float = _PRINT_ASPECT) -> Optional[Path]:
    """Compose (once) and return the path to the clean print-resolution PNG for
    `job`, or None if the job's inputs are missing or composition fails.

    The expensive layout build is reused from the stored mask; only the photo is
    re-derived and composited at print resolution. Idempotent: returns the
    cached file if it already exists. Shared by the paid /download path and the
    signed /printful-fetch art URL so both produce byte-identical output.

    `aspect` is the print canvas aspect (width/height). It defaults to 4:5 (the
    digital download / on-screen proof, cached as `{job}.png`); a physical order
    passes its product's true aspect (e.g. 0.75 for an 18x24 poster) and the file
    is composed + cached per aspect (`{job}.a750.png`) so the art matches the
    physical size -- face centred, ground-padded, never cropped or stretched."""
    recipe_path = PRIVATE_DIR / f"{job}.json"
    src_path = PRIVATE_DIR / f"{job}.src"
    if not recipe_path.exists() or not src_path.exists():
        return None
    tag = "" if abs(aspect - _PRINT_ASPECT) < 0.005 else f".a{int(round(aspect * 1000))}"
    path = PRIVATE_DIR / f"{job}{tag}.png"
    if path.exists():
        return path
    try:
        from .pipeline.tonal import compose_layered, render_layered_png, custom_poster
        r = json.loads(recipe_path.read_text(encoding="utf-8"))
        # Rebuild the user-picked colour for a "custom" ink so the paid file matches
        # the preview.
        dl_custom = custom_poster(r.get("ink_hex")) if (r.get("ink") == "custom" and r.get("ink_hex")) else None
        warns2 = WarningCollector()
        _bgmask = None
        _bgm_path = PRIVATE_DIR / f"{job}.bgmask.png"
        if _bgm_path.exists():
            try:
                import cv2
                import numpy as _np
                _bgmask = cv2.imdecode(_np.frombuffer(_bgm_path.read_bytes(), _np.uint8), cv2.IMREAD_GRAYSCALE)
            except Exception:  # noqa: BLE001
                _bgmask = None
        an = analyze_image(src_path.read_bytes(), RenderConfig(), warns2, manual_mask=_bgmask)
        if r.get("style") == "displacement":
            from .pipeline.displacement import render_displacement_portrait
            png_bytes = render_displacement_portrait(
                an, (r.get("text", "") or "").split(),
                ground=r.get("ground", "navy"), out_width=DOWNLOAD_PNG_WIDTH,
                uppercase=bool(r.get("uppercase", True)), flow=bool(r.get("flow")),
                ink=("photo" if r.get("ink") == "custom" else r.get("ink")),
                print_aspect=aspect)
            if not png_bytes:
                return None
            path.write_bytes(png_bytes)
            return path
        mask_path = PRIVATE_DIR / f"{job}.mask.svg"
        if mask_path.exists():
            from .pipeline.tonal import _MSG_BOOST
            dl_boost = _MSG_BOOST if r.get("style") == "message" else 0.0
            png_bytes = compose_layered(
                mask_path.read_text(encoding="utf-8"), an,
                r.get("ink", "navy"), bool(r.get("remove_bg", True)), DOWNLOAD_PNG_WIDTH,
                light=bool(r.get("light", False)), boost=dl_boost, print_aspect=aspect,
                custom=dl_custom)
        else:  # no stored mask (e.g. light/engraving renders) or older jobs:
            # recompose at print size. Reuse the chosen word size so the paid
            # download matches the preview (light mode has no baked mask).
            cfg2 = RenderConfig()
            try:
                if r.get("min_font_px"):
                    cfg2.min_font_px = float(r["min_font_px"])
            except (TypeError, ValueError):
                pass
            png_bytes, _, _, _ = render_layered_png(
                an, r["text"], r.get("style", "words"), cfg2, warns2,
                ink=r.get("ink", "navy"), remove_bg=bool(r.get("remove_bg", True)),
                light=bool(r.get("light", False)), out_width=DOWNLOAD_PNG_WIDTH, render_w=2600,
                uppercase=bool(r.get("uppercase", True)), print_aspect=aspect, custom=dl_custom)
        if not png_bytes:
            return None
        path.write_bytes(png_bytes)
        return path
    except Exception:  # noqa: BLE001
        return None


@app.get("/download")
def download(job: str, session_id: str, fmt: str = "png"):
    """Serve the clean file only after verifying the Stripe payment for `job`."""
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": False, "error": "payments_unconfigured"}, status_code=503)
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        sess = stripe.checkout.Session.retrieve(session_id)
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "bad_session"}, status_code=400)
    # Stripe's Session object routes attribute access through __getattr__ and has
    # no dict .get(), so read fields via getattr (with safe defaults).
    paid = getattr(sess, "payment_status", None) == "paid"
    meta = getattr(sess, "metadata", None)
    meta_job = getattr(meta, "job", None) if meta is not None else None
    if not paid or meta_job != job:
        return JSONResponse({"ok": False, "error": "not_paid"}, status_code=402)
    recipe_path = PRIVATE_DIR / f"{job}.json"
    src_path = PRIVATE_DIR / f"{job}.src"
    if not recipe_path.exists() or not src_path.exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    # Compose the print-resolution layered PNG once (idempotent, cached on disk).
    path = _ensure_clean_png(job)
    if path is None:
        return JSONResponse({"ok": False, "error": "export_failed"}, status_code=500)
    # Drop a .paid marker so admin stats can count paid orders accurately
    # without re-querying Stripe per job. Idempotent.
    paid_marker = PRIVATE_DIR / f"{job}.paid"
    if not paid_marker.exists():
        try:
            import time as _time
            paid_marker.write_text(str(int(_time.time())), encoding="utf-8")
        except OSError:
            pass
    return FileResponse(str(path), media_type="image/png", filename=f"typortrait-{job}.png")


@app.get("/download/card")
def download_card(job: str, session_id: str, name: str = "", dates: str = "",
                  verse: str = "words", layout: str = "classic", custom_verse: str = ""):
    """Bonus print-ready memorial-card PDF (front/back), free with the digital
    download. Gated by the same Stripe verification as /download; the portrait's
    own words (from the recipe) fill the card back."""
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": False, "error": "payments_unconfigured"}, status_code=503)
    if not _session_paid(session_id, job):
        return JSONResponse({"ok": False, "error": "not_paid"}, status_code=402)
    recipe_path = PRIVATE_DIR / f"{job}.json"
    if not recipe_path.exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    clean = _ensure_clean_png(job)
    if clean is None:
        return JSONResponse({"ok": False, "error": "export_failed"}, status_code=500)
    try:
        words_raw = str(json.loads(recipe_path.read_text(encoding="utf-8")).get("text") or "")
    except Exception:  # noqa: BLE001
        words_raw = ""
    seen, words_list = set(), []
    for w in re.split(r"[\s,]+", words_raw):
        w = w.strip()
        if not w or w.lower() in seen:
            continue
        seen.add(w.lower()); words_list.append(w.title())
        if len(words_list) >= 6:
            break
    words = "  -  ".join(words_list) if words_list else "Beloved  -  Always"
    name = (name or "").strip()[:60] or "Beloved"
    dates = (dates or "").strip()[:40]
    layout = layout if layout in ("classic", "bleed", "elegant") else "classic"
    if verse == "custom":
        verse_arg = (custom_verse or "").strip()[:240]          # their own words (literal)
    elif verse in ("words", "psalm23", "light", "none"):
        verse_arg = verse                                       # a preset key
    else:
        verse_arg = "words"
    out_pdf = OUTPUTS_DIR / f"{job}_card.pdf"
    try:
        import memorial_card
        memorial_card.make_card(str(clean), name, dates, words, verse=verse_arg,
                                layout=layout, out_pdf=str(out_pdf))
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "card_failed"}, status_code=500)
    return FileResponse(str(out_pdf), media_type="application/pdf",
                        filename=f"memorial-card-{job}.pdf")


@app.get("/download/card-preview")
def card_preview(job: str, session_id: str, layout: str = "classic"):
    """Tiny live preview of the card FRONT in `layout`, built from the buyer's own
    portrait -- the layout-picker thumbnails on /success. Cached on disk; the page
    falls back to a stock thumbnail client-side if this can't be built."""
    layout = layout if layout in ("classic", "bleed", "elegant") else "classic"
    cache = OUTPUTS_DIR / f"{job}_cardprev_{layout}.png"
    if cache.exists():
        return FileResponse(str(cache), media_type="image/png")
    if not STRIPE_SECRET_KEY or not _session_paid(session_id, job):
        return JSONResponse({"ok": False, "error": "not_paid"}, status_code=402)
    if not (PRIVATE_DIR / f"{job}.json").exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    clean = _ensure_clean_png(job)
    if clean is None:
        return JSONResponse({"ok": False, "error": "export_failed"}, status_code=500)
    try:
        import memorial_card as mc
        import pypdfium2 as pdfium
        tmp = OUTPUTS_DIR / f"{job}_cardprev_{layout}.pdf"
        mc.make_card(str(clean), "", "", "Beloved", verse="none", layout=layout, out_pdf=str(tmp))
        # Rasterize the card front with PDFium (pypdfium2 -- permissive BSD-3-Clause/
        # Apache, self-contained wheel), replacing PyMuPDF/AGPL. PDF y is bottom-up, so
        # the trim-box crop flips through PAGE_H; then thumbnail for the layout picker.
        pdf = pdfium.PdfDocument(str(tmp))
        s = 150 / 72.0
        im = pdf[0].render(scale=s).to_pil().convert("RGB").crop(
            (int(mc.TX0 * s), int((mc.PAGE_H - mc.TY1) * s),
             int(mc.TX1 * s), int((mc.PAGE_H - mc.TY0) * s)))
        im.thumbnail((300, 300))
        im.save(str(cache))
        pdf.close()
        try:
            tmp.unlink()
        except OSError:
            pass
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "preview_failed"}, status_code=500)
    return FileResponse(str(cache), media_type="image/png")


@app.api_route("/printful-fetch/{job}", methods=["GET", "HEAD"])
def printful_fetch(job: str, exp: int, sig: str, a: float = _PRINT_ASPECT):
    """One-time signed URL Printful fetches the clean PNG from.

    The clean file lives in PRIVATE_DIR (paywalled, not statically mounted). For
    physical orders Printful must download the art; we expose it through this
    HMAC-signed, time-limited URL. The art is composed lazily here if it does not
    already exist (our model only stores the recipe + mask, not the big PNG).

    HEAD is supported (same validation) because Printful's fetcher sends a HEAD
    pre-flight to check the file before the GET; a 405 there can make stricter
    fetchers abandon the download. Starlette's FileResponse answers HEAD with the
    correct headers (Content-Type/Content-Length) and an empty body, and the
    lazy compose warms the file so the follow-up GET is served from cache."""
    if not printful.verify_signed_url(job, exp, sig):
        raise HTTPException(status_code=403, detail="invalid_or_expired_signature")
    # `a` is the print aspect (set by the order's product); clamp to a sane band so
    # a malformed value can't drive a runaway canvas. Unsigned -- it only changes
    # padding of the same already-paywalled art, so it carries no extra authority.
    aspect = min(2.0, max(0.4, float(a)))
    path = _ensure_clean_png(job, aspect)
    if path is None:
        raise HTTPException(status_code=404, detail="unknown_job")
    return FileResponse(str(path), media_type="image/png")


def _recipient_from_session(sess: dict) -> Optional[dict]:
    """Map a Stripe Checkout Session's shipping address to Printful's recipient
    schema. Stripe relocated the collected shipping address across API versions:
    2025+ nests it under ``collected_information.shipping_details`` while older
    versions exposed top-level ``shipping_details`` / ``shipping``. Try the new
    path first, then the legacy ones, then fall back to the billing address in
    ``customer_details``. Returns None when no usable street address was
    collected (e.g. digital orders)."""
    collected = sess.get("collected_information") or {}
    ship = (
        (collected.get("shipping_details") if isinstance(collected, dict) else None)
        or sess.get("shipping_details")
        or sess.get("shipping")
        or {}
    )
    if not isinstance(ship, dict):
        ship = {}
    customer = sess.get("customer_details") or {}
    addr = ship.get("address") or {}
    if not addr.get("line1"):
        # Last resort: the billing address Stripe collected on the customer.
        addr = (customer.get("address") or {}) if isinstance(customer, dict) else {}
    if not addr.get("line1"):
        return None
    return {
        "name": ship.get("name") or customer.get("name") or "",
        "address1": addr.get("line1") or "",
        "address2": addr.get("line2") or "",
        "city": addr.get("city") or "",
        "state_code": addr.get("state") or "",
        "country_code": addr.get("country") or "US",
        "zip": addr.get("postal_code") or "",
        "email": customer.get("email") or "",
        "phone": customer.get("phone") or "",
    }


def _fulfill_with_printful(order_id: str, recipient: dict) -> None:
    """Submit a paid physical order to Printful. Idempotent: noop unless the row
    is in 'paid' state with a variant. Caller has just transitioned it to paid."""
    o = orders_db.get(order_id)
    if not o:
        return
    if o["status"] != "paid" or not o["variant_id"]:
        return
    try:
        prod = products.get(str(o["sku"]))
        aspect = prod.print_aspect if prod else _PRINT_ASPECT
        # Compose the high-res print file FIRST so Printful's fetch hits a READY file.
        # (Was a background warm that RACED the order POST: Printful fetches the file
        # during order creation, and for a multi-item bundle the still-composing file
        # pushed the POST past its read timeout. Fulfilment runs off the webhook now,
        # so blocking ~30-45s here is fine.)
        job_id = o["job_id"]
        if str(job_id).startswith("gallery:"):
            # Gallery order: fixed art, no compose. Warm the aspect-padded master and
            # sign a URL to the gallery fetch route instead of the personalized one.
            item = str(job_id).split(":", 1)[1]
            _gallery_print_file(item, aspect)
            signed = printful.signed_print_url(item, aspect=aspect, path="/gallery/printful-fetch")
        else:
            _ensure_clean_png(job_id, aspect)
            signed = printful.signed_print_url(job_id, aspect=aspect)
        placement = "front" if str(o["sku"]).startswith("tshirt") else "default"
        common = dict(recipient=recipient, print_file_url=signed, external_id=order_id,
                      retail_price_cents=o["price_cents"], confirm=PRINTFUL_CONFIRM,
                      placement=placement)
        if prod and prod.bundle_items:
            res = printful.create_order(bundle=prod.bundle_items, **common)   # multi-item order
        else:
            res = printful.create_order(variant_id=o["variant_id"], **common)
        pf_id = res.get("id") if isinstance(res, dict) else None
        orders_db.mark_fulfilling(order_id=order_id, printful_order_id=int(pf_id or 0), raw=res)
    except Exception as e:  # noqa: BLE001
        orders_db.mark_error(order_id=order_id, error_message=str(e))


def _umami_event(name: str, data: Optional[dict] = None) -> None:
    """Fire a server-side Umami event (e.g. a confirmed purchase). Best-effort:
    swallows every error so analytics can never break a payment/fulfillment path.
    No-ops unless UMAMI_WEBSITE_ID is configured."""
    if not UMAMI_WEBSITE_ID:
        return
    try:
        import httpx
        httpx.post(
            f"{UMAMI_HOST}/api/send",
            json={
                "type": "event",
                "payload": {
                    "website": UMAMI_WEBSITE_ID,
                    "hostname": UMAMI_HOSTNAME,
                    "url": "/success",
                    "name": name,
                    "data": data or {},
                },
            },
            # Umami's collector requires a User-Agent or it rejects the event.
            headers={"User-Agent": "Typortrait-Server/1.0", "Content-Type": "application/json"},
            timeout=4.0,
        )
    except Exception:  # noqa: BLE001
        pass


# Persistent dedupe markers so the `purchase` conversion fires exactly once per
# Stripe session no matter which completion path confirms the sale first
# (webhook, the /order/{id} polling fallback, or the digital /success page) and
# no matter how many times those pages are reloaded. Lives under data/ (the
# volume-mounted dir) so it survives restarts.
_PURCHASE_TRACK_DIR = DATA_DIR / "purchase_events"
_SALE_NOTIFY_DIR = DATA_DIR / "sale_notified"


def _notify_sale(sess: dict, order: Optional[dict]) -> None:
    """On a paid checkout: (1) force the customer's Stripe receipt by setting
    receipt_email on the PaymentIntent (so it sends regardless of the dashboard
    setting), and (2) email the admin a 'you made a sale' alert. Deduped per
    session; the network I/O runs off the webhook response thread. Best-effort --
    never raises into the payment path."""
    sid = sess.get("id") or ""
    if not sid:
        return
    try:
        import hashlib
        _SALE_NOTIFY_DIR.mkdir(parents=True, exist_ok=True)
        marker = _SALE_NOTIFY_DIR / (hashlib.sha256(sid.encode("utf-8")).hexdigest()[:32] + ".txt")
        if marker.exists():
            return
        marker.write_text("1", encoding="utf-8")
    except Exception:  # noqa: BLE001
        return
    email = (sess.get("customer_details") or {}).get("email")
    pi = sess.get("payment_intent")
    meta = sess.get("metadata") or {}
    sku = meta.get("sku")
    prod = products.get(sku) if sku else None
    amt = sess.get("amount_total")
    summary = {
        "product": (prod.name if prod else (sku or "High-res digital download")),
        "amount": (f"{amt/100:.2f} {(sess.get('currency') or 'usd').upper()}"
                   if isinstance(amt, (int, float)) else ""),
        "email": email,
        "ref": meta.get("ref"),
        "order_id": (order or {}).get("id") or meta.get("order_id"),
        "shipping": _recipient_from_session(sess),
        "digital_included": bool(prod and prod.physical),
    }

    def _bg():
        if pi and email and STRIPE_SECRET_KEY:
            try:
                import stripe
                stripe.api_key = STRIPE_SECRET_KEY
                stripe.PaymentIntent.modify(pi, receipt_email=email)
            except Exception:  # noqa: BLE001
                pass
        try:
            admin_mod.send_sale_email(summary)
        except Exception:  # noqa: BLE001
            pass
    import threading
    threading.Thread(target=_bg, daemon=True).start()


def _fav_links(brand: str = "typortrait") -> str:
    """Favicon <link> tags for a server-rendered page. `brand` maps to the studio
    favicon set at /static/favicons/{brand}/; an unknown brand -> generic typortrait."""
    b = brand if brand in ("typortrait", "lovedinwords", "keeper", "everloved") else "typortrait"
    p = f"/static/favicons/{b}"
    return (f"<link rel='icon' type='image/svg+xml' href='{p}/favicon.svg'>"
            f"<link rel='icon' type='image/png' sizes='32x32' href='{p}/favicon-32.png'>"
            f"<link rel='apple-touch-icon' href='{p}/apple-touch-icon.png'>"
            f"<link rel='icon' href='{p}/favicon.ico' sizes='any'>")


def _make_another(job: str) -> tuple:
    """(href, label) for the post-purchase 'make another' link, preserving the
    buyer's brand: Loved in Words -> branded studio + 'Make another Tribute
    Portrait'; generic -> plain studio + 'Make another Typortrait'. The href is
    relative, so it stays on whatever host (staging/app) the buyer is on."""
    try:
        brand = str(json.loads((PRIVATE_DIR / f"{job}.json").read_text(encoding="utf-8")).get("brand") or "")
    except Exception:  # noqa: BLE001
        brand = ""
    if brand == "lovedinwords":
        return ("/static/index.html?brand=lovedinwords", "Make another Tribute Portrait")
    if brand:
        from urllib.parse import quote
        return (f"/static/index.html?brand={quote(brand, safe='')}", "Make another Typortrait")
    return ("/static/index.html", "Make another Typortrait")


def _track_purchase_once(
    session_id: str,
    *,
    sku: Optional[str] = None,
    amount_cents: Optional[int] = None,
    currency: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """Record a completed sale exactly once, deduped per Stripe session.

    Two things happen here, both idempotent via a single persisted marker:
      1. The sale is written to our own ledger (data/purchase_events/<hash>.txt as
         a small JSON record: ts, sku, amount, currency, source). This powers the
         admin "Referral sales" report — owned by us, independent of analytics.
      2. The ad-blocker-proof Umami `purchase` conversion is fired, if configured.

    Sales complete through several paths (webhook, /order poll, digital /success),
    so every path calls this; the marker guarantees a single record + event. Any
    missing detail is pulled from Stripe. Best-effort: never raises into a
    payment/fulfillment path."""
    if not session_id:
        return
    try:
        import hashlib
        import time as _time
        _PURCHASE_TRACK_DIR.mkdir(parents=True, exist_ok=True)
        marker = _PURCHASE_TRACK_DIR / (
            hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32] + ".txt"
        )
        if marker.exists():
            return
        # Fill in anything the caller didn't supply (e.g. the digital /success
        # page only has the session id) straight from Stripe.
        if amount_cents is None or sku is None or currency is None or source is None:
            try:
                import stripe
                stripe.api_key = STRIPE_SECRET_KEY
                sess = _stripe_to_dict(stripe.checkout.Session.retrieve(session_id))
                meta = sess.get("metadata") or {}
                if amount_cents is None:
                    amount_cents = sess.get("amount_total")
                if currency is None:
                    currency = sess.get("currency")
                if sku is None:
                    sku = meta.get("sku")
                if source is None:
                    source = meta.get("ref")        # referral/source tag (e.g. everloved)
            except Exception:  # noqa: BLE001
                pass
        rec = {
            "ts": int(_time.time()),
            "sku": sku or "digital",
            "amount_cents": int(amount_cents or 0),
            "currency": (currency or CURRENCY or "usd").lower(),
            "source": (source or "direct"),
        }
        # Claim the marker BEFORE emitting so a concurrent caller can't double-fire.
        # The JSON record doubles as the dedup marker (older markers held just "1").
        marker.write_text(json.dumps(rec), encoding="utf-8")
    except Exception:  # noqa: BLE001
        return
    if UMAMI_WEBSITE_ID:
        _umami_event("purchase", {
            "revenue": round(rec["amount_cents"] / 100.0, 2),
            "currency": rec["currency"],
            "sku": rec["sku"],
            "source": rec["source"],               # so referred sales (any sku) are countable
        })


@app.post("/webhook/stripe")
async def webhook_stripe(request: Request, stripe_signature: Optional[str] = Header(None)):
    """Stripe -> us. On checkout.session.completed for a physical order, mark the
    order paid (idempotent) and submit it to Printful. Signature verified via
    STRIPE_WEBHOOK_SECRET; without a secret the body is trusted (dev only)."""
    payload = await request.body()
    if STRIPE_WEBHOOK_SECRET:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        try:
            event = _stripe_to_dict(stripe.Webhook.construct_event(
                payload, stripe_signature or "", STRIPE_WEBHOOK_SECRET,
            ))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"bad_signature: {e}")
    else:
        try:
            event = json.loads(payload.decode("utf-8"))
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="bad_payload")

    if event.get("type") != "checkout.session.completed":
        return JSONResponse({"ok": True, "ignored": event.get("type")})

    sess = (event.get("data") or {}).get("object") or {}
    if sess.get("payment_status") != "paid":
        return JSONResponse({"ok": True, "ignored": "not_paid"})

    recipient = _recipient_from_session(sess)
    transitioned = orders_db.mark_paid(
        stripe_session_id=sess.get("id") or "",
        payment_intent=sess.get("payment_intent"),
        customer_email=(sess.get("customer_details") or {}).get("email"),
        recipient=recipient,
    )
    if transitioned and recipient and transitioned.get("variant_id"):
        import threading
        # Off the webhook: fulfilment composes a high-res file (~30-45s) which would
        # blow Stripe's ~10s webhook timeout. The webhook returns immediately.
        threading.Thread(target=_fulfill_with_printful,
                         args=(transitioned["id"], recipient), daemon=True).start()

    # Reliable, ad-blocker-proof conversion event to Umami. Deduped against the
    # synchronous /order and /success paths so the sale is counted exactly once.
    meta = sess.get("metadata") or {}
    _track_purchase_once(
        sess.get("id") or "",
        sku=meta.get("sku") or ("print" if (transitioned or {}).get("variant_id") else "digital"),
        amount_cents=sess.get("amount_total"),
        currency=sess.get("currency"),
        source=meta.get("ref"),
    )
    # Customer receipt (force Stripe to send one) + admin sale-notification email.
    _notify_sale(sess, transitioned)
    return JSONResponse({"ok": True})


def _reel_maker_block(job: str, session_id: str) -> str:
    """Self-contained 'Make a reel' card (HTML + scoped CSS + JS) for any paid
    job. Identical dual-consent flow and /reel call as the digital success
    page, so print buyers get the exact same reel ability and authorizations.
    Portable: carries its own styles so it can drop into any page chrome."""
    j = json.dumps(job)
    s = json.dumps(session_id)
    # Memorial brand (Loved in Words): tribute-video framing, and DROP the
    # "feature on our social channels" ask. Brand from the job's stored ref.
    _ref = ""
    try:
        _ref = str(json.loads((PRIVATE_DIR / f"{job}.json").read_text(encoding="utf-8")).get("brand") or "")
    except Exception:  # noqa: BLE001
        _ref = ""
    if _ref == "lovedinwords":
        reel_title = "Create a tribute video"
        reel_sub = "Turn the portrait into a short remembrance video — their face forming from your words — to keep or share with family."
        personal_label = "Yes, I’ll share this video for personal use."
        reel_btn = "Make the tribute video"
        ready_label = "Your tribute video is ready."
        share_label = "Share video"
        making_label = "Making the tribute video… (~30s)"
        ct_html = ""
    else:
        reel_title = "Make a reel from your portrait"
        reel_sub = "Turn it into a short 9:16 video — the dissolve, your words, and the finished portrait — perfect for sharing."
        personal_label = "Yes, I’ll share my reel for personal use."
        reel_btn = "Make my reel"
        ready_label = "Your reel is ready."
        share_label = "Share reel"
        making_label = "Making your reel… (~30s)"
        ct_html = ('<label class="rl-chk"><input type="checkbox" id="ct"> '
                   '<span>Optional: Allow Typortrait to feature this reel on our own social channels '
                   '(<a href="/terms" target="_blank" rel="noopener">terms</a>). You can revoke any time by emailing us.</span></label>')
    return (
        '<style>'
        '.rl-divider{height:1px;background:#ece9e3;margin:24px 0 18px}'
        '.rl-h{font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif;color:#0d1b3a;font-size:19px;margin:0 0 6px}'
        '.rl-sub{color:#6b7280;font-size:14px;line-height:1.5;margin:0 0 14px}'
        '.rl-chk{display:flex;gap:9px;align-items:flex-start;font-size:13.5px;color:#16203a;'
        'line-height:1.45;margin:10px 0;text-align:left;cursor:pointer}'
        '.rl-chk input{margin-top:2px;width:17px;height:17px;flex:0 0 auto;accent-color:#0d1b3a}'
        '.rl-chk a{color:#0d1b3a;font-weight:600}'
        '.btn.ghost{background:#fff;color:#0d1b3a;border:1.5px solid #0d1b3a}'
        '.btn:disabled{opacity:.7}'
        '#mk,#rlOut .btn{display:block;width:100%;text-align:center;margin-top:10px}'
        '.rl-spin{display:inline-block;width:14px;height:14px;margin-right:8px;border-radius:50%;'
        'border:2px solid rgba(255,255,255,.4);border-top-color:#fff;vertical-align:-2px;animation:rlsp .8s linear infinite}'
        '@keyframes rlsp{to{transform:rotate(360deg)}}'
        '</style>'
        '<div class="rl-divider"></div>'
        '<h2 class="rl-h">' + reel_title + '</h2>'
        '<p class="rl-sub">' + reel_sub + '</p>'
        '<label class="rl-chk"><input type="checkbox" id="cp"> '
        '<span>' + personal_label + '</span></label>'
        + ct_html +
        '<button class="btn" id="mk" disabled>' + reel_btn + '</button>'
        '<div id="rlOut" style="display:none">'
        '<p class="rl-sub" id="rlSub" style="margin-top:14px">' + ready_label + '</p>'
        '<a class="btn" id="rlMp4" download="typortrait-reel.mp4" style="display:none">Download MP4</a>'
        '<a class="btn ghost" id="rlGif" download="typortrait-reel.gif">Download GIF</a>'
        '<button class="btn ghost" id="rlSh" style="display:none">' + share_label + '</button>'
        '</div>'
        '<script>(function(){'
        'var job=' + j + ',sid=' + s + ',o=location.origin;'
        'var shareUrl=o+"/p/"+job;'
        'var cp=document.getElementById("cp"),ct=document.getElementById("ct"),mk=document.getElementById("mk");'
        'var rlOut=document.getElementById("rlOut"),rlSub=document.getElementById("rlSub");'
        'var rlGif=document.getElementById("rlGif"),rlMp4=document.getElementById("rlMp4"),rlSh=document.getElementById("rlSh");'
        'cp.onchange=function(){mk.disabled=!cp.checked;};'
        'mk.onclick=function(){if(!cp.checked)return;'
        'mk.disabled=true;mk.innerHTML=\'<span class="rl-spin"></span>' + making_label + '\';'
        'var fd=new FormData();fd.append("job",job);fd.append("session_id",sid);'
        'fd.append("personal_consent",cp.checked?"on":"");'
        'fd.append("typortrait_consent",(ct&&ct.checked)?"on":"");'
        'fetch("/reel",{method:"POST",body:fd}).then(function(r){return r.json();}).then(function(j){'
        'if(!j||!j.ok){mk.disabled=false;mk.innerHTML=' + json.dumps(reel_btn) + ';'
        'rlSub.textContent="Sorry, that didn\\u2019t work: "+((j&&(j.detail||j.error))||"please try again.");'
        'rlOut.style.display="block";return;}'
        'mk.style.display="none";rlOut.style.display="block";'
        'rlSub.textContent=j.typortrait_share?"Your reel is ready. Thanks \\u2014 we\\u2019ll review it before posting on our channels.":' + json.dumps(ready_label) + ';'
        'rlGif.href=j.gif_url;'
        'if(j.mp4_url){rlMp4.href=j.mp4_url;rlMp4.style.display="inline-block";}'
        'var mob=(window.matchMedia&&matchMedia("(pointer:coarse)").matches)||/Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent||"");'
        'if(mob&&navigator.canShare){rlSh.style.display="inline-block";rlSh.onclick=function(){'
        'var u=j.mp4_url||j.gif_url;var nm=j.mp4_url?"typortrait-reel.mp4":"typortrait-reel.gif";'
        'var mt=j.mp4_url?"video/mp4":"image/gif";'
        'fetch(u).then(function(r){return r.blob();}).then(function(bl){'
        'var f=new File([bl],nm,{type:mt});'
        'if(navigator.canShare({files:[f]}))return navigator.share({title:"Typortrait",text:"Made from our words \\u2014 with Typortrait.",url:shareUrl,files:[f]});'
        'throw 0;}).catch(function(){if(navigator.clipboard){navigator.clipboard.writeText(shareUrl);rlSh.textContent="Link copied";}});};}'
        '}).catch(function(){mk.disabled=false;mk.innerHTML=' + json.dumps(reel_btn) + ';'
        'rlSub.textContent="Network error \\u2014 please try again.";rlOut.style.display="block";});};'
        '})();</script>'
    )


@app.get("/order/{order_id}", response_class=HTMLResponse)
def order_status(order_id: str, session_id: Optional[str] = None):
    """Customer-facing order status page (physical print orders land here from
    Stripe's success_url)."""
    o = orders_db.get(order_id)
    if not o:
        raise HTTPException(status_code=404, detail="unknown_order")

    # If we arrived from Stripe success and the webhook hasn't fired yet, poll
    # Stripe directly so the customer sees a confirmed state instead of "pending
    # payment". (Webhooks are async and can lag a few seconds.)
    if session_id and o["status"] == "pending_payment" and STRIPE_SECRET_KEY:
        try:
            import stripe
            stripe.api_key = STRIPE_SECRET_KEY
            sess = _stripe_to_dict(stripe.checkout.Session.retrieve(session_id))
            if sess.get("payment_status") == "paid":
                recipient = _recipient_from_session(sess)
                transitioned = orders_db.mark_paid(
                    stripe_session_id=sess.get("id"),
                    payment_intent=sess.get("payment_intent"),
                    customer_email=(sess.get("customer_details") or {}).get("email"),
                    recipient=recipient,
                )
                if transitioned and recipient and transitioned.get("variant_id"):
                    import threading
                    threading.Thread(target=_fulfill_with_printful,
                                     args=(transitioned["id"], recipient), daemon=True).start()
                # Conversion event: this synchronous path is what completes most
                # physical sales when the webhook isn't delivered. Deduped so a
                # later webhook can't double-count.
                _track_purchase_once(
                    sess.get("id") or session_id or "",
                    sku=(sess.get("metadata") or {}).get("sku")
                    or ("print" if (transitioned or {}).get("variant_id") else "digital"),
                    amount_cents=sess.get("amount_total"),
                    currency=sess.get("currency"),
                    source=(sess.get("metadata") or {}).get("ref"),
                )
                o = orders_db.get(order_id)
        except Exception:  # noqa: BLE001
            pass

    product = products.get(o["sku"])
    name = product.name if product else o["sku"]
    # Gallery orders carry a "gallery:<item>" sentinel job_id (fixed art, no recipe),
    # so the personalized download/reel/'make another' don't apply — swap them out.
    is_gallery = str(o["job_id"]).startswith("gallery:")
    gallery_item = str(o["job_id"]).split(":", 1)[1] if is_gallery else ""
    if is_gallery:
        studio_href, studio_label = ("/gallery", "Browse the gallery")
    else:
        studio_href, studio_label = _make_another(o["job_id"])   # 'make another' keeps the buyer's brand + name
    status_msg = {
        "pending_payment": "Waiting for payment confirmation…",
        "paid": "Payment received. Preparing your order for fulfillment…",
        "fulfilling": "We've sent your Typortrait to the press. You'll get an email when it ships.",
        "shipped": "Your order has shipped!",
        "delivered": "Delivered. Thank you!",
        "error": "We hit a snag fulfilling this order. We'll be in touch — please reply to your receipt.",
    }.get(o["status"], o["status"])

    tracking_html = ""
    if o.get("tracking_url"):
        tracking_html = (
            f'<p><a class="btn" href="{o["tracking_url"]}" target="_blank" rel="noopener">'
            f'Track your shipment</a></p>'
        )

    download_html = ""
    if o["status"] in ("paid", "fulfilling", "shipped", "delivered") and session_id:
        dl = (f'/gallery/download?item={gallery_item}&session_id={session_id}' if is_gallery
              else f'/download?job={o["job_id"]}&fmt=png&session_id={session_id}')
        if o["sku"] == "digital":
            download_html = f'<p><a class="btn" href="{dl}">Download your PNG</a></p>'
        else:
            # Every print also comes with the high-res digital file — the same
            # clean PNG we send to the press. The /download paywall already
            # verifies this physical order's paid Stripe session, so no extra
            # checkout is needed.
            download_html = (
                '<div class="status" style="margin-top:14px">'
                '<strong>Your digital file is included — free.</strong><br>'
                '<span class="muted">The same high-resolution, watermark-free image '
                'we’re printing for you — yours to keep, reprint, or share.</span>'
                f'<p style="margin-bottom:0"><a class="btn" href="{dl}">'
                'Download your high-res file</a></p></div>'
            )

    # Print buyers get the same reel ability + authorizations as digital buyers.
    # /reel verifies this order's paid Stripe session, so the flow is identical.
    # (Gallery orders are fixed art with no source recipe -> no reel.)
    reel_html = ""
    if not is_gallery and o["status"] in ("paid", "fulfilling", "shipped", "delivered") and session_id:
        reel_html = _reel_maker_block(o["job_id"], session_id)

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" type="image/svg+xml" href="/static/favicons/typortrait/favicon.svg">
<link rel="icon" type="image/png" sizes="32x32" href="/static/favicons/typortrait/favicon-32.png">
<link rel="apple-touch-icon" href="/static/favicons/typortrait/apple-touch-icon.png">
<link rel="icon" href="/static/favicons/typortrait/favicon.ico" sizes="any">
<title>Typortrait — Order {order_id}</title>
<link rel="stylesheet" href="/static/inter.css">
<style>
  *{{box-sizing:border-box}}
  body{{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:#faf9f7;color:#16203a;margin:0;padding:24px;display:flex;justify-content:center}}
  .card{{background:#fff;border:1px solid #ece9e3;border-radius:20px;
        box-shadow:0 10px 40px rgba(20,30,60,.10);padding:28px;max-width:520px;width:100%}}
  h1{{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0d1b3a;margin:0 0 4px}}
  .muted{{color:#6b7280;font-size:14px}}
  dl{{margin:18px 0;display:grid;grid-template-columns:auto 1fr;gap:8px 16px;font-size:15px}}
  dt{{color:#6b7280}}
  .status{{background:#f3f5fa;border-radius:12px;padding:14px 16px;margin-top:18px}}
  .btn{{display:inline-block;background:#0d1b3a;color:#fff;text-decoration:none;
        border-radius:999px;padding:12px 18px;font-weight:600;margin-top:12px}}
</style></head><body>
<div class="card">
  <h1>Thank you</h1>
  <div class="muted">Order #{order_id}</div>
  <dl>
    <dt>Item</dt><dd>{name}{(' — ' + o['size']) if o.get('size') else ''}</dd>
    <dt>Total</dt><dd>${(o['price_cents'] + o['shipping_cents']) / 100:.2f} {o['currency'].upper()}</dd>
    <dt>Status</dt><dd>{o['status'].replace('_', ' ')}</dd>
  </dl>
  <div class="status">{status_msg}</div>
  {tracking_html}
  {download_html}
  {reel_html}
  <p class="muted" style="margin-top:24px">
    Bookmark this page to check on your order, or reply to your receipt email
    if anything looks off.
  </p>
  <p><a href="{studio_href}">{studio_label} →</a></p>
</div></body></html>"""
    return HTMLResponse(body)


def _session_paid(session_id: str, job: str) -> bool:
    """True if `session_id` is a paid Stripe session for `job`."""
    if not STRIPE_SECRET_KEY:
        return False
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        sess = stripe.checkout.Session.retrieve(session_id)
    except Exception:  # noqa: BLE001
        return False
    meta = getattr(sess, "metadata", None)
    return (getattr(sess, "payment_status", None) == "paid"
            and (getattr(meta, "job", None) if meta is not None else None) == job)


# Bumped whenever the consent copy on the success page materially changes,
# so every stored consent record points at the exact wording the user saw.
CONSENT_TEXT_VERSION = "v1-2026-05"


def _make_square_crop_jpeg(src_path, dst_path) -> bool:
    """Center-crop the source upload to a 1:1 JPEG for use as the reel's
    'before' frame. Honors EXIF rotation so phone photos aren't sideways."""
    try:
        from PIL import Image, ImageOps
        with Image.open(str(src_path)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            im = im.crop((left, top, left + s, top + s))
            im.thumbnail((1080, 1080), Image.LANCZOS)
            im.save(str(dst_path), "JPEG", quality=88, optimize=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@app.post("/reel")
def make_reel(
    job: str = Form(...),
    session_id: str = Form(...),
    personal_consent: str = Form(""),
    typortrait_consent: str = Form(""),
) -> JSONResponse:
    """Generate a shareable reel (GIF + MP4 when ffmpeg is present) from a
    paid job. Records the user's dual-consent decision.

    - `personal_consent` (required): the user confirms they will share the
      reel themselves for personal use.
    - `typortrait_consent` (optional, off by default): grants Typortrait a
      revocable license to feature the reel on its own social channels.
    """
    import os, sys, time
    if not _session_paid(session_id, job):
        return JSONResponse({"ok": False, "error": "not_paid"}, status_code=402)
    if (personal_consent or "").lower() not in ("on", "true", "1", "yes"):
        return JSONResponse({"ok": False, "error": "consent_required"}, status_code=400)

    recipe_path = PRIVATE_DIR / f"{job}.json"
    src_path = PRIVATE_DIR / f"{job}.src"
    portrait_path = PRIVATE_DIR / f"{job}.png"
    if not recipe_path.exists() or not src_path.exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    if not portrait_path.exists():
        # The high-res PNG is composed lazily on the first paid /download or
        # Printful fetch. A print buyer can reach the reel maker before either
        # has run, so compose it on demand here (idempotent). Only if that
        # still fails do we surface a clear retry.
        try:
            _ensure_clean_png(job)
        except Exception:  # noqa: BLE001
            pass
    if not portrait_path.exists():
        return JSONResponse(
            {"ok": False, "error": "portrait_not_ready",
             "detail": "Your portrait is still being prepared — please wait a few seconds and try again."},
            status_code=409,
        )

    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "bad_recipe"}, status_code=500)
    # The recipe persists the raw user-typed string in `text`; parse it the
    # same way the render endpoint does to recover the word list.
    words = _parse_words(recipe.get("text"), None)
    if not words:
        return JSONResponse({"ok": False, "error": "no_words"}, status_code=400)

    gif_path = OUTPUTS_DIR / f"{job}_reel.gif"
    mp4_path = OUTPUTS_DIR / f"{job}_reel.mp4"

    tools_dir = STATIC_DIR.parent / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        from reel_template import build_reel  # type: ignore
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "reel_import_failed", "detail": str(e)}, status_code=500)

    try:
        _memorial = str(recipe.get("brand") or "") == "lovedinwords"
        # Both brands present the portrait inside a real framed-on-a-desk scene,
        # tonally matched: Loved in Words gets the candle-lit memorial scene; every
        # other subject (pets, weddings, graduates) gets a brighter, candle-free
        # desk so the mood fits. Same fixed 4:5 mat opening; credit per brand.
        _scene = (STATIC_DIR / "everloved" / "scene-desk.jpg") if _memorial else (STATIC_DIR / "scene-desk-plain.jpg")
        build_reel({
            "before": str(src_path),       # original photo, kept at its real aspect ratio
            "after":  str(portrait_path),
            "words":  words,
            "out":    str(gif_path),
            "aspect": "source",            # personal reel preserves the original aspect ratio
            "minimal": True,               # no CTA / promo captions — just a discreet credit
            "scene":  str(_scene) if _scene.exists() else None,
            "credit": "lovedinwords.com" if _memorial else "Typortrait.com",
        })
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "reel_build_failed", "detail": str(e)}, status_code=500)

    # Convert GIF -> MP4 via the shared helper (same encoder as batch_reels.py).
    # If TYPO_REEL_AUDIO is set or tools/assets/reel_audio.mp3 exists, the MP4
    # ships with a music bed mixed at 0.45 volume with fade in/out, so the
    # silent-video downranking that TikTok / IG Reels apply isn't triggered.
    mp4_made = False
    try:
        from reel_template import convert_gif_to_mp4  # type: ignore
        audio_env = os.environ.get("TYPO_REEL_AUDIO", "").strip()
        if audio_env and Path(audio_env).exists():
            audio_path = audio_env
        else:
            default_audio = tools_dir / "assets" / "reel_audio.mp3"
            audio_path = str(default_audio) if default_audio.exists() else None
        mp4_made = convert_gif_to_mp4(str(gif_path), str(mp4_path), audio_path=audio_path)
    except Exception:  # noqa: BLE001
        mp4_made = False

    tp_consent = (typortrait_consent or "").lower() in ("on", "true", "1", "yes")
    try:
        (PRIVATE_DIR / f"{job}.consent.json").write_text(
            json.dumps({
                "job": job,
                "session_id": session_id,
                "personal_consent": True,
                "typortrait_share_consent": tp_consent,
                "consent_text_version": CONSENT_TEXT_VERSION,
                "ts": int(time.time()),
            }, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

    # When the user opts in to Typortrait sharing, register the reel with the
    # review pipeline. The background scanner in app.admin picks it up and
    # emails the admin; the dashboard at /admin/reels surfaces it for the
    # full lifecycle (approve, posted, revoke).
    if tp_consent:
        try:
            admin_mod.init_review(job)
        except Exception:  # noqa: BLE001
            pass

    return JSONResponse({
        "ok": True,
        "gif_url": f"/outputs/{job}_reel.gif",
        "mp4_url": f"/outputs/{job}_reel.mp4" if mp4_made else None,
        "share_url": f"/p/{job}",
        "typortrait_share": tp_consent,
    })


@app.get("/success", response_class=HTMLResponse)
def success(job: str, session_id: str):
    """Post-payment page: confirms the purchase and offers both the high-res PNG
    and the vector SVG (one purchase unlocks the job)."""
    import html, json as _json
    from urllib.parse import quote
    paid = _session_paid(session_id, job) and (PRIVATE_DIR / f"{job}.json").exists()
    fav_brand = "typortrait"   # favicon brand; set to the job's brand below (memorial skins)
    jq, sq = quote(job, safe=""), quote(session_id, safe="")
    png_url = f"/download?job={jq}&fmt=png&session_id={sq}"
    studio_href, studio_label = _make_another(job)   # 'make another' keeps the buyer's brand + name
    if paid:
        # Conversion event for digital sales (the primary revenue path, which
        # completes here rather than via the webhook). Deduped + revenue/sku
        # filled from Stripe, so a page refresh can't double-count.
        _track_purchase_once(session_id, sku="digital")
        # Memorial brand (Loved in Words): reframe the "reel" as a tribute video
        # and DROP the "feature on our social channels" ask -- marketing with a
        # grieving family's loved one is off-key. Brand is read from the job's
        # stored ref (set to the brand id when they came through that storefront).
        _ref = ""
        try:
            _ref = str(_json.loads((PRIVATE_DIR / f"{job}.json").read_text(encoding="utf-8")).get("brand") or "")
        except Exception:  # noqa: BLE001
            _ref = ""
        is_memorial = _ref == "lovedinwords"
        fav_brand = _ref or "typortrait"
        if is_memorial:
            reel_noun = "tribute video"
            reel_title = "Create a tribute video"
            reel_sub = "Turn the portrait into a short remembrance video — their face forming from your words — to keep or share with family."
            personal_label = "Yes, I’ll share this video for personal use."
            reel_btn = "Make the tribute video"
            ready_label = "Your tribute video is ready."
            share_label = "Share video"
            making_label = "Making the tribute video… (~30s)"
            ct_html = ""
        else:
            reel_noun = "reel"
            reel_title = "Make a reel from your portrait"
            reel_sub = "Turn it into a short 9:16 video — the dissolve, your words, and the finished portrait — perfect for sharing."
            personal_label = "Yes, I’ll share my reel for personal use."
            reel_btn = "Make my reel"
            ready_label = "Your reel is ready."
            share_label = "Share reel"
            making_label = "Making your reel… (~30s)"
            ct_html = ('<label class="chk"><input type="checkbox" id="ct"> '
                       '<span>Optional: Allow Typortrait to feature this reel on our own social channels '
                       '(<a href="/terms" target="_blank" rel="noopener">terms</a>). You can revoke any time by emailing us.</span></label>')
        # Memorial card (brand-aware): a free print-ready prayer-card PDF for the
        # memorial skins, generated from the portrait's own words (recipe `text`).
        show_card = _ref in ("lovedinwords", "keeper", "everloved")
        card_html = ""
        card_js = ""
        if show_card:
            card_html = (
                '<div class="divider"></div>'
                '<h2 class="reel-h">Make a memorial card</h2>'
                '<p class="sub reel-sub">A print-ready prayer card from their portrait &mdash; front &amp; back. Free with your download.</p>'
                '<div class="cdf">'
                '<input id="cdName" maxlength="60" placeholder="Full name" autocomplete="off" />'
                '<input id="cdDates" maxlength="40" placeholder="Dates (e.g. 1942 &ndash; 2026)" autocomplete="off" />'
                '</div>'
                '<div class="cdlab">Choose a layout</div>'
                '<div class="cdthumbs" id="cdThumbs">'
                '<button type="button" class="cdthumb sel" data-lay="classic"><img src="/download/card-preview?job=' + jq + '&amp;session_id=' + sq + '&amp;layout=classic" onerror="this.onerror=null;this.src=\'/static/card-previews/classic.png\'" alt="Classic layout"><span>Classic</span></button>'
                '<button type="button" class="cdthumb" data-lay="bleed"><img src="/download/card-preview?job=' + jq + '&amp;session_id=' + sq + '&amp;layout=bleed" onerror="this.onerror=null;this.src=\'/static/card-previews/bleed.png\'" alt="Full-portrait layout"><span>Full portrait</span></button>'
                '<button type="button" class="cdthumb" data-lay="elegant"><img src="/download/card-preview?job=' + jq + '&amp;session_id=' + sq + '&amp;layout=elegant" onerror="this.onerror=null;this.src=\'/static/card-previews/elegant.png\'" alt="Elegant layout"><span>Elegant</span></button>'
                '</div>'
                '<div class="cdlab">Verse (on the back)</div>'
                '<select id="cdVerse" class="cdsel">'
                '<option value="words">Tribute &mdash; made from their words</option>'
                '<option value="psalm23">Psalm 23</option>'
                '<option value="light">Helen Keller</option>'
                '<option value="custom">Write your own&hellip;</option>'
                '<option value="none">No verse</option>'
                '</select>'
                '<p class="cdvtext" id="cdVtext"></p>'
                '<textarea id="cdCustom" class="cdta" maxlength="240" rows="3" placeholder="Write a short verse or memory&hellip;" style="display:none"></textarea>'
                '<div class="cdhint" id="cdHint" style="display:none">Keep it short &mdash; a line or two reads best (up to 240 characters).</div>'
                '<button class="btn" id="cdDl">Download memorial card (PDF)</button>'
                '<a class="cdlink" href="/static/how-to-print.pdf" target="_blank" rel="noopener">How to print your card &mdash; a quick guide (PDF)</a>'
                '<p class="note" id="cdNote" style="display:none;color:#c0392b"></p>'
            )
            card_js = (
                'var cdLay="classic";'
                'var VT={words:"Their portrait was made from the words of everyone who loved them \\u2014 so their story stays close, always.",'
                'psalm23:"The Lord is my shepherd; I shall not want. He maketh me to lie down in green pastures.",'
                'light:"What we have once enjoyed we can never lose; all that we love deeply becomes a part of us.",custom:"",none:""};'
                'var thumbs=document.getElementById("cdThumbs");'
                'if(thumbs){thumbs.querySelectorAll(".cdthumb").forEach(function(b){b.onclick=function(){'
                'thumbs.querySelectorAll(".cdthumb").forEach(function(x){x.classList.remove("sel");});'
                'b.classList.add("sel");cdLay=b.getAttribute("data-lay");};});}'
                'var vsel=document.getElementById("cdVerse"),vtxt=document.getElementById("cdVtext"),cta=document.getElementById("cdCustom");'
                'var hint=document.getElementById("cdHint");'
                'function vupd(){var v=vsel.value;if(v==="custom"){cta.style.display="block";if(hint)hint.style.display="block";vtxt.style.display="none";}'
                'else{cta.style.display="none";if(hint)hint.style.display="none";vtxt.textContent=VT[v]||"";vtxt.style.display=VT[v]?"block":"none";}}'
                'if(vsel){vsel.onchange=vupd;vupd();}'
                'var cdDl=document.getElementById("cdDl");'
                'if(cdDl){cdDl.onclick=function(){'
                'var nm=(document.getElementById("cdName").value||"").trim();'
                'var dt=(document.getElementById("cdDates").value||"").trim();'
                'var vs=vsel.value,cv=(cta.value||"").trim();'
                'var note=document.getElementById("cdNote");note.style.display="none";'
                'var ot=cdDl.innerHTML;cdDl.disabled=true;cdDl.innerHTML=\'<span class="spin"></span>Making the card\\u2026\';'
                'var q="/download/card?job="+encodeURIComponent(job)+"&session_id="+encodeURIComponent(sid)'
                '+"&name="+encodeURIComponent(nm)+"&dates="+encodeURIComponent(dt)+"&layout="+cdLay+"&verse="+vs'
                '+"&custom_verse="+encodeURIComponent(cv);'
                'fetch(q).then(function(r){if(!r.ok)throw 0;return r.blob();}).then(function(b){'
                'var a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="memorial-card.pdf";'
                'document.body.appendChild(a);a.click();a.remove();cdDl.disabled=false;cdDl.innerHTML=ot;'
                '}).catch(function(){cdDl.disabled=false;cdDl.innerHTML=ot;'
                'note.style.display="block";note.textContent="Couldn\\u2019t make the card \\u2014 please try again.";});};}'
            )
        # Fetch the high-res file in the background (it's composed on first
        # request and can take a few seconds), showing a spinner, then hand the
        # user a ready, instant download instead of a hung button.
        inner = (
            '<div class="check">&#10003;</div>'
            '<h1>Your Typortrait is ready</h1>'
            '<p class="sub" id="sub">Preparing your high-resolution file — this can take up to a minute…</p>'
            '<button class="btn" id="dl" disabled><span class="spin"></span>Preparing…</button>'
            '<button class="btn ghost" id="sh">Share</button>'
            '<p class="note">Watermark-free, print-quality — ready to print or share.</p>'
            # ---- Phase B: optional reel / tribute-video card (brand-aware) ----
            '<div class="divider"></div>'
            '<h2 class="reel-h">' + reel_title + '</h2>'
            '<p class="sub reel-sub">' + reel_sub + '</p>'
            '<label class="chk"><input type="checkbox" id="cp"> '
            '<span>' + personal_label + '</span></label>'
            + ct_html +
            '<button class="btn" id="mk" disabled>' + reel_btn + '</button>'
            '<div class="link-wrap"><a class="link" href="' + studio_href + '">' + studio_label + '</a></div>'
            '<div id="rlOut" style="display:none">'
            '<p class="sub" id="rlSub">' + ready_label + '</p>'
            '<a class="btn" id="rlMp4" download="typortrait-reel.mp4" style="display:none">Download MP4</a>'
            '<a class="btn ghost" id="rlGif" download="typortrait-reel.gif">Download GIF</a>'
            '<button class="btn ghost" id="rlSh" style="display:none">' + share_label + '</button>'
            '</div>' + card_html +
            '<script>(function(){var url=' + _json.dumps(png_url) + ';'
            'var job=' + _json.dumps(job) + ',sid=' + _json.dumps(session_id) + ',o=location.origin;'
            'var shareUrl=o+"/p/"+job,prevUrl=o+"/outputs/"+job+"_preview.png";'
            'var btn=document.getElementById("dl"),sub=document.getElementById("sub"),sh=document.getElementById("sh");'
            'function save(b){var a=document.createElement("a");a.href=URL.createObjectURL(b);'
            'a.download="typortrait.png";document.body.appendChild(a);a.click();a.remove();}'
            'fetch(url).then(function(r){if(!r.ok)throw 0;return r.blob();}).then(function(b){'
            'btn.disabled=false;btn.innerHTML="Download your portrait";btn.onclick=function(){save(b);};'
            'sub.textContent="Done! Tap below to save it.";}).catch(function(){'
            'btn.disabled=false;btn.innerHTML="Download your portrait";btn.onclick=function(){location.href=url;};'
            'sub.textContent="Your portrait is ready.";});'
            'sh.onclick=function(){var t="Someone I love, made from our words — with Typortrait.";'
            'var mob=(window.matchMedia&&matchMedia("(pointer:coarse)").matches)||/Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent||"");'
            '(mob&&navigator.canShare?fetch(prevUrl).then(function(r){return r.blob();}).then(function(bl){'
            'var f=new File([bl],"typortrait.png",{type:"image/png"});'
            'if(navigator.canShare({files:[f]}))return navigator.share({title:"Typortrait",text:t,url:shareUrl,files:[f]});'
            'throw 0;}):Promise.reject())'
            '.catch(function(){if(navigator.clipboard){navigator.clipboard.writeText(shareUrl);sh.textContent="Link copied — paste anywhere";}else{prompt("Copy this link:",shareUrl);}});};'
            # ---- Reel maker JS ----
            'var cp=document.getElementById("cp"),ct=document.getElementById("ct"),mk=document.getElementById("mk");'
            'var rlOut=document.getElementById("rlOut"),rlSub=document.getElementById("rlSub");'
            'var rlGif=document.getElementById("rlGif"),rlMp4=document.getElementById("rlMp4"),rlSh=document.getElementById("rlSh");'
            'cp.onchange=function(){mk.disabled=!cp.checked;};'
            'mk.onclick=function(){if(!cp.checked)return;'
            'mk.disabled=true;mk.innerHTML=\'<span class="spin"></span>' + making_label + '\';'
            'var fd=new FormData();fd.append("job",job);fd.append("session_id",sid);'
            'fd.append("personal_consent",cp.checked?"on":"");'
            'fd.append("typortrait_consent",(ct&&ct.checked)?"on":"");'
            'fetch("/reel",{method:"POST",body:fd}).then(function(r){return r.json();}).then(function(j){'
            'if(!j||!j.ok){mk.disabled=false;mk.innerHTML=' + _json.dumps(reel_btn) + ';'
            'rlSub.textContent="Sorry, that didn\\u2019t work: "+((j&&(j.detail||j.error))||"please try again.");'
            'rlOut.style.display="block";return;}'
            'mk.style.display="none";rlOut.style.display="block";'
            'rlSub.textContent=j.typortrait_share?"Your reel is ready. Thanks \\u2014 we\\u2019ll review it before posting on our channels.":' + _json.dumps(ready_label) + ';'
            'rlGif.href=j.gif_url;'
            'if(j.mp4_url){rlMp4.href=j.mp4_url;rlMp4.style.display="inline-block";}'
            'var mob=(window.matchMedia&&matchMedia("(pointer:coarse)").matches)||/Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent||"");'
            'if(mob&&navigator.canShare){rlSh.style.display="inline-block";rlSh.onclick=function(){'
            'var u=j.mp4_url||j.gif_url;var nm=j.mp4_url?"typortrait-reel.mp4":"typortrait-reel.gif";'
            'var mt=j.mp4_url?"video/mp4":"image/gif";'
            'fetch(u).then(function(r){return r.blob();}).then(function(bl){'
            'var f=new File([bl],nm,{type:mt});'
            'if(navigator.canShare({files:[f]}))return navigator.share({title:"Typortrait",text:"Made from our words \\u2014 with Typortrait.",url:shareUrl,files:[f]});'
            'throw 0;}).catch(function(){if(navigator.clipboard){navigator.clipboard.writeText(shareUrl);rlSh.textContent="Link copied";}});};}'
            '}).catch(function(){mk.disabled=false;mk.innerHTML="Make my reel";'
            'rlSub.textContent="Network error \\u2014 please try again.";rlOut.style.display="block";});};'
            + card_js +
            '})();</script>'
        )
    else:
        inner = (
            '<h1>Finishing up&hellip;</h1>'
            '<p class="sub">If your payment just completed, your download will be ready in a moment. '
            'Refresh this page; if it doesn&rsquo;t appear, your card may not have been charged.</p>'
            f'<a class="btn" href="/success?job={html.escape(jq)}&amp;session_id={html.escape(sq)}">Refresh</a>'
            '<a class="link" href="' + studio_href + '">Back to the studio</a>'
        )
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>" + _fav_links(fav_brand) +
        "<title>Typortrait — your download</title><link rel='stylesheet' href='/static/inter.css'><style>"
        ":root{--navy:#0d1b3a;--muted:#6b7280;--line:#ece9e3}"
        "*{box-sizing:border-box}body{margin:0;min-height:100dvh;display:flex;align-items:center;justify-content:center;"
        "background:#faf9f7;color:#16203a;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;padding:24px}"
        ".card{background:#fff;border:1px solid var(--line);border-radius:20px;box-shadow:0 10px 40px rgba(20,30,60,.10);"
        "max-width:440px;width:100%;padding:34px 28px;text-align:center}"
        ".check{width:54px;height:54px;border-radius:50%;background:#0d1b3a;color:#fff;font-size:28px;line-height:54px;margin:0 auto 14px}"
        "h1{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--navy);font-size:26px;margin:6px 0 8px}"
        ".sub{color:var(--muted);font-size:15px;line-height:1.5;margin:0 0 22px}"
        ".btn{display:inline-block;min-width:240px;border-radius:999px;background:var(--navy);color:#fff;font-size:16px;font-weight:600;"
        "padding:14px 28px;margin:10px auto;text-decoration:none;border:none;cursor:pointer}"
        ".btn.ghost{background:#fff;color:var(--navy);border:1.5px solid var(--navy)}"
        ".note{color:var(--muted);font-size:13px;line-height:1.55;margin:18px 2px 0;text-align:left}"
        ".cdf{display:flex;flex-direction:column;gap:9px;margin:4px 0 2px;text-align:left}"
        ".cdf input,.cdf select{width:100%;border:1px solid var(--line);border-radius:11px;padding:11px 12px;font-size:15px;font-family:inherit;background:#fcfbf9;color:#16203a}"
        ".cdlab{font-size:13px;font-weight:600;color:var(--navy);text-align:left;margin:14px 2px 7px}"
        ".cdthumbs{display:flex;gap:8px}"
        ".cdthumb{flex:1;border:2px solid var(--line);border-radius:12px;background:#fff;padding:6px;cursor:pointer;display:flex;flex-direction:column;align-items:center;gap:5px}"
        ".cdthumb img{width:100%;border-radius:6px;display:block}"
        ".cdthumb span{font-size:11px;color:var(--muted)}"
        ".cdthumb.sel{border-color:var(--navy)}.cdthumb.sel span{color:var(--navy);font-weight:600}"
        ".cdsel{width:100%;border:1px solid var(--line);border-radius:11px;padding:11px 12px;font-size:15px;font-family:inherit;background:#fcfbf9;color:#16203a}"
        ".cdvtext{font-size:13px;line-height:1.5;color:var(--muted);font-style:italic;text-align:left;margin:8px 2px 2px}"
        ".cdta{width:100%;border:1px solid var(--line);border-radius:11px;padding:11px 12px;font-size:15px;font-family:inherit;background:#fcfbf9;color:#16203a;margin-top:8px;resize:vertical}"
        ".cdhint{font-size:12px;color:var(--muted);text-align:left;margin:7px 2px 0}"
        ".cdlink{display:inline-block;margin-top:11px;color:var(--navy);font-size:13px;text-decoration:underline}"
        ".link{display:inline-block;margin-top:18px;color:var(--muted);font-size:14px;text-decoration:none}"
        ".btn:disabled{opacity:.75}"
        ".spin{display:inline-block;width:14px;height:14px;margin-right:8px;border-radius:50%;"
        "border:2px solid rgba(255,255,255,.4);border-top-color:#fff;vertical-align:-2px;animation:sp .8s linear infinite}"
        "@keyframes sp{to{transform:rotate(360deg)}}"
        ".divider{height:1px;background:var(--line);margin:24px -4px 18px}"
        ".reel-h{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--navy);font-size:20px;margin:4px 0 6px}"
        ".reel-sub{margin-bottom:14px}"
        ".chk{display:flex;align-items:flex-start;gap:10px;text-align:left;color:#16203a;font-size:13.5px;line-height:1.45;margin:8px 4px}"
        ".chk input{margin-top:3px;flex-shrink:0}"
        ".chk a{color:var(--navy)}"
        ".link-wrap{text-align:center;margin-top:14px}"
        "</style></head><body><div class='card'>" + inner + "</div></body></html>"
    )
    return HTMLResponse(page)


@app.get("/p/{job}", response_class=HTMLResponse)
def share_page(job: str):
    """Public per-portrait page so a pasted/shared link unfurls with the portrait
    (og:image) and offers a 'make your own' CTA."""
    import html, re as _re
    job = _re.sub(r"[^a-zA-Z0-9]", "", job)[:40]
    prev = OUTPUTS_DIR / f"{job}_preview.png"
    img = (f"{PUBLIC_BASE_URL}/outputs/{job}_preview.png" if prev.exists()
           else "https://typortrait.com/og.png")
    title = "Someone you love, made from your words — Typortrait"
    desc = "A portrait made entirely of words. Create your own at Typortrait.com."
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>" + _fav_links() +
        f"<title>{html.escape(title)}</title>"
        f"<meta property='og:type' content='website'>"
        f"<meta property='og:title' content=\"{html.escape(title)}\">"
        f"<meta property='og:description' content=\"{html.escape(desc)}\">"
        f"<meta property='og:image' content=\"{html.escape(img)}\">"
        "<meta property='og:image:width' content='1200'>"
        "<meta property='og:image:height' content='1500'>"
        "<meta name='twitter:card' content='summary_large_image'>"
        f"<meta name='twitter:title' content=\"{html.escape(title)}\">"
        f"<meta name='twitter:description' content=\"{html.escape(desc)}\">"
        f"<meta name='twitter:image' content=\"{html.escape(img)}\">"
        "<link rel='stylesheet' href='/static/inter.css'><style>*{box-sizing:border-box}body{margin:0;min-height:100dvh;display:flex;align-items:center;"
        "justify-content:center;background:#0a0a0c;color:#f5f3ec;font-family:Inter,-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Roboto,Helvetica,Arial,sans-serif;padding:24px;text-align:center}"
        ".w{max-width:520px;width:100%}img{width:100%;border-radius:14px;box-shadow:0 16px 50px rgba(0,0,0,.5)}"
        "h1{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:24px;margin:22px 0 6px}"
        "p{color:#b9b6ae;font-size:15px;margin:0 0 20px}"
        "a{display:inline-block;background:#f5f3ec;color:#0a0a0c;font-weight:700;text-decoration:none;"
        "padding:14px 28px;border-radius:999px;font-size:16px}</style></head><body><div class='w'>"
        f"<img src=\"{html.escape(img)}\" alt=\"A Typortrait\"/>"
        "<h1>Someone you love, made from your words.</h1>"
        "<p>A portrait composed entirely of the words that matter.</p>"
        "<a href='https://typortrait.com'>Make your own &rarr;</a>"
        "</div></body></html>"
    )
    return HTMLResponse(page)


@app.post("/save-link")
async def save_link(email: str = Form(...), job: str = Form(...)) -> JSONResponse:
    """Exit-intent capture: email a user their private /p/{job} link so they can
    return to a portrait they previewed but didn't buy, before the ~RETENTION_DAYS
    auto-delete. Stores the email as an owned remarketing contact (sidecar) and
    sends the link via the existing SMTP path."""
    import re as _re
    job = _re.sub(r"[^a-zA-Z0-9]", "", job or "")[:40]
    email = (email or "").strip()[:200]
    # The portrait must actually exist (preview composed) for the link to work.
    if not job or not (OUTPUTS_DIR / f"{job}_preview.png").exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    if not _re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return JSONResponse({"ok": False, "error": "bad_email"}, status_code=400)
    sent = admin_mod.send_save_link_email(job, email)
    admin_mod.record_save_capture(job, email, emailed=sent)
    return JSONResponse({"ok": True, "emailed": sent})


def _resume_expired_page() -> str:
    """Shown when a saved link is opened after the portrait's files were purged
    by the retention sweep (~RETENTION_DAYS) or the job never existed."""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>" + _fav_links() +
        "<title>Portrait expired — Typortrait</title><link rel='stylesheet' href='/static/inter.css'><style>"
        "*{box-sizing:border-box}body{margin:0;min-height:100dvh;display:flex;align-items:center;"
        "justify-content:center;background:#faf9f7;color:#16203a;font-family:Inter,-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Roboto,Helvetica,Arial,sans-serif;padding:24px;text-align:center}.w{max-width:440px}"
        ".brand{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:26px;color:#0d1b3a;font-weight:700;margin:0 0 26px}"
        "h1{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0d1b3a;font-size:24px;margin:0 0 8px}"
        "p{color:#6b7280;font-size:15px;line-height:1.6;margin:0 0 22px}"
        "a{display:inline-block;background:#0d1b3a;color:#fff;font-weight:700;text-decoration:none;"
        "padding:14px 28px;border-radius:999px;font-size:16px}</style></head><body><div class='w'>"
        "<div class='brand'>Typortrait</div>"
        "<h1>This portrait has expired.</h1>"
        f"<p>To keep storage private, we automatically delete uploaded photos and generated files "
        f"after about {RETENTION_DAYS} days, so this saved link is no longer available. "
        "It only takes a minute to make a new one.</p>"
        "<a href='https://typortrait.com'>Make a new portrait &rarr;</a>"
        "</div></body></html>"
    )


_RESUME_TPL = """<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<link rel='icon' type='image/svg+xml' href='/static/favicons/typortrait/favicon.svg'>
<link rel='icon' type='image/png' sizes='32x32' href='/static/favicons/typortrait/favicon-32.png'>
<link rel='apple-touch-icon' href='/static/favicons/typortrait/apple-touch-icon.png'>
<link rel='icon' href='/static/favicons/typortrait/favicon.ico' sizes='any'>
<title>Your saved portrait — Typortrait</title>
<meta name='robots' content='noindex'>
<link rel="stylesheet" href="/static/inter.css">
<style>
:root{--navy:#0d1b3a;--ink:#16203a;--paper:#faf9f7;--card:#fff;--muted:#6b7280;--line:#ece9e3;
--serif:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;--sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
*{box-sizing:border-box}
html,body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased}
.wrap{max-width:560px;margin:0 auto;padding:28px 20px 64px}
.brand{font-family:var(--serif);font-size:26px;color:var(--navy);font-weight:700;text-align:center}
.sub{color:var(--muted);font-size:14px;text-align:center;margin:6px 0 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;box-shadow:0 10px 30px rgba(13,27,58,.06);padding:18px;margin-top:22px}
.shot{width:100%;border-radius:12px;display:block}
.wm{color:var(--muted);font-size:12px;text-align:center;margin:10px 0 0}
.t-lbl{font-weight:700;color:var(--navy);font-size:14px}
.t-sub{color:var(--muted);font-size:12.5px;margin:2px 0 10px}
.prod{display:flex;width:100%;text-align:left;gap:10px;align-items:center;justify-content:space-between;border:1.5px solid var(--line);background:#fff;border-radius:14px;padding:12px 14px;margin-top:8px;cursor:pointer;font-family:var(--sans)}
.prod.on{border-color:var(--navy);box-shadow:0 0 0 1px var(--navy) inset}
.pname{font-weight:700;color:var(--ink);font-size:14.5px}
.ptag{display:inline-block;background:#f1e7d2;color:#6b5a2a;font-size:10.5px;font-weight:700;border-radius:999px;padding:2px 8px;margin-left:6px;vertical-align:middle}
.pblurb{color:var(--muted);font-size:12.5px;margin-top:2px}
.pinc{color:var(--navy);font-size:11.5px;font-weight:600;margin-top:5px}
.pprice{color:var(--navy);font-weight:700;font-size:15px;white-space:nowrap;text-align:right}
.pship{color:var(--muted);font-size:11px;font-weight:500}
.sizeRow{display:none;align-items:center;gap:10px;margin-top:12px}
.sizeRow select{flex:1;border:1.5px solid var(--line);border-radius:10px;padding:10px;font-size:15px;font-family:var(--sans)}
.btn{display:block;width:100%;text-align:center;background:var(--navy);color:#fff;border:none;border-radius:999px;padding:15px;font-size:16px;font-weight:700;font-family:var(--sans);cursor:pointer;margin-top:16px}
.btn:disabled{opacity:.6;cursor:default}
.note{color:var(--muted);font-size:12.5px;text-align:center;margin-top:10px;min-height:1em}
.alt{display:block;text-align:center;color:var(--muted);font-size:13px;margin-top:16px;text-decoration:none}
.alt:hover{color:var(--navy)}
</style></head><body><div class="wrap">
<div class="brand">Typortrait</div>
<div class="sub">Welcome back &mdash; here&rsquo;s the portrait you saved.</div>
<div class="card">
<img class="shot" src="__IMG__" alt="Your saved portrait"/>
<p class="wm">Preview is watermarked &middot; your download and prints are clean, full-resolution.</p>
</div>
<div class="card" id="shop">
<div class="t-lbl" id="shopLbl" style="text-align:center;margin-bottom:12px">Choose your keepsake</div>
<div class="t-sub" id="shopSub">Every print comes with the high-res digital file, free &mdash; keep it, reprint it, share it.</div>
<div id="productList"></div>
<div class="sizeRow" id="sizeRow"><label for="sizeSel" class="t-lbl">Size</label><select id="sizeSel"></select></div>
<button class="btn" id="buy">Download</button>
<p class="note" id="note"></p>
</div>
<a class="alt" href="https://typortrait.com">Start a new portrait &rarr;</a>
</div>
<script>
const JOB="__JOB__";
let CAT=[], sku="digital", size=null, busy=false;
const buy=document.getElementById("buy"), note=document.getElementById("note"),
      list=document.getElementById("productList"), sizeRow=document.getElementById("sizeRow"),
      sizeSel=document.getElementById("sizeSel");
function money(c){ return "$"+(c/100).toFixed(2); }
function label(){ const p=CAT.find(x=>x.sku===sku); if(!p) return "Download"; return (p.physical?"Order — ":"Download — ")+money(p.price_cents); }
function pick(s){
  const p=CAT.find(x=>x.sku===s)||CAT.find(x=>x.sku==="digital"); if(!p) return;
  sku=p.sku;
  if(p.sizes && p.sizes.length){
    sizeSel.innerHTML=p.sizes.map(x=>"<option>"+x+"</option>").join("");
    if(p.sizes.indexOf(size)<0) size=p.sizes[0];
    sizeSel.value=size; sizeRow.style.display="flex";
  } else { size=null; sizeRow.style.display="none"; }
  document.querySelectorAll(".prod").forEach(el=>el.classList.toggle("on",el.dataset.sku===sku));
  buy.textContent=label();
}
sizeSel.addEventListener("change",e=>{ size=e.target.value; });
function render(data){
  CAT=(data.products||[]).map(p=>Object.assign({},p));
  if(!data.fulfillment_configured){
    document.getElementById("shopLbl").style.display="none";
    document.getElementById("shopSub").style.display="none";
    const d=CAT.find(x=>x.sku==="digital");
    if(d){ sku="digital"; buy.textContent="Download — "+money(d.price_cents); }
    return;
  }
  document.getElementById("shopLbl").style.display="";
  document.getElementById("shopSub").style.display="";
  const order=CAT.slice().sort((a,b)=>(a.physical?1:0)-(b.physical?1:0));
  list.innerHTML="";
  order.forEach(p=>{
    const el=document.createElement("button");
    el.type="button"; el.className="prod"+(p.sku===sku?" on":""); el.dataset.sku=p.sku;
    const tag=p.tag?'<span class="ptag">'+p.tag+'</span>':"";
    const ship=p.shipping_cents>0?'<div class="pship">+'+money(p.shipping_cents)+' ship</div>':"";
    const inc=p.physical?'<div class="pinc">&#10003; High-res digital file included, free</div>':"";
    el.innerHTML='<div><div class="pname">'+p.name+tag+'</div><div class="pblurb">'+p.blurb+'</div>'+inc+'</div>'+
      '<div class="pprice">'+money(p.price_cents)+ship+'</div>';
    el.addEventListener("click",()=>pick(p.sku));
    list.appendChild(el);
  });
  pick("digital");
}
fetch("/products").then(r=>r.json()).then(render).catch(()=>{ note.textContent="Couldn’t load options — refresh to try again."; });
buy.addEventListener("click", async ()=>{
  if(busy) return; busy=true; const old=buy.textContent;
  buy.disabled=true; buy.textContent="Opening secure checkout…"; note.textContent="";
  try{
    const fd=new FormData(); fd.append("job",JOB); fd.append("fmt","png");
    fd.append("sku",sku||"digital"); if(size) fd.append("size",size);
    const r=await fetch("/checkout",{method:"POST",body:fd}); const d=await r.json();
    if(d.ok && d.url){ location.href=d.url; return; }
    note.textContent=({payments_unconfigured:"Checkout isn’t set up yet.",fulfillment_unconfigured:"Print orders aren’t available right now — try the digital download.",missing_or_invalid_size:"Please pick a size first.",unknown_job:"This portrait has expired — start a new one."})[d.error]||"Couldn’t start checkout — please try again.";
  }catch(e){ note.textContent="Network error — please try again."; }
  busy=false; buy.disabled=false; buy.textContent=old;
});
</script></body></html>"""


@app.get("/resume/{job}", response_class=HTMLResponse)
def resume_page(job: str):
    """Reopen a saved portrait with working Download + print options, wired to
    the same /checkout contract as the studio. This is the target of the
    exit-intent 'save your portrait' email link, so a returning user can buy the
    exact portrait they previewed — without re-uploading or re-rendering. The
    job's recipe + preview persist until the retention sweep; past that we show a
    friendly expiry page."""
    import re as _re
    job = _re.sub(r"[^a-zA-Z0-9]", "", job or "")[:40]
    recipe = PRIVATE_DIR / f"{job}.json"
    preview = OUTPUTS_DIR / f"{job}_preview.png"
    if not job or not recipe.exists() or not preview.exists():
        return HTMLResponse(_resume_expired_page(), status_code=404)
    page = _RESUME_TPL.replace("__JOB__", job).replace("__IMG__", f"/outputs/{job}_preview.png")
    return HTMLResponse(page)


_POLICY_CONTACT = "support@typortrait.com"
_POLICY_UPDATED = "June 2026"


def _policy_page(title: str, blocks) -> HTMLResponse:
    body = "".join((f"<h2>{h}</h2>{p}" if h else p) for h, p in blocks)
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>" + _fav_links() +
        f"<title>{title} — Typortrait</title><link rel='stylesheet' href='/static/inter.css'><style>"
        "body{margin:0;background:#faf9f7;color:#16203a;font-family:Inter,-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Roboto,Helvetica,Arial,sans-serif;line-height:1.6}"
        ".wrap{max-width:760px;margin:0 auto;padding:40px 22px 80px}"
        "h1{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0d1b3a;font-size:30px;margin:0 0 4px}"
        "h2{font-size:18px;color:#0d1b3a;margin:28px 0 6px}.upd{color:#6b7280;font-size:14px;margin:0 0 8px}"
        "p,li{font-size:15px;color:#2b3550}ul{margin:6px 0 0;padding-left:20px}"
        "a{color:#0d1b3a}.back{display:inline-block;margin-bottom:18px;color:#6b7280;text-decoration:none;font-size:14px}"
        ".note{background:#fff;border:1px solid #ece9e3;border-left:4px solid #c9a24a;border-radius:10px;padding:12px 14px;font-size:13px;color:#6b5a2a}"
        "</style></head><body><div class='wrap'>"
        "<a class='back' href='/static/index.html'>&larr; Back to Typortrait&trade;</a>"
        f"<h1>{title}</h1><p class='upd'>Last updated {_POLICY_UPDATED}</p>"
        + body +
        f"<h2>Contact</h2><p>Questions? Email <a href='mailto:{_POLICY_CONTACT}'>{_POLICY_CONTACT}</a>.</p>"
        "</div></body></html>"
    )
    return HTMLResponse(page)


@app.get("/terms", response_class=HTMLResponse)
def terms():
    blocks = [
        ("1. Acceptance", "<p>By using Typortrait (the &ldquo;Service&rdquo;) you agree to these Terms of Use. If you do not agree, please do not use the Service. You must be at least 13 years old to use the Service, and if you are under 18 you must have your parent or legal guardian&rsquo;s permission. The Service is not for children under 13.</p>"),
        ("2. The Service", "<p>Typortrait turns a photo and words you provide into a typographic portrait, offered as a free watermarked preview and a paid, watermark-free digital download.</p>"),
        ("3. Your content and rights", "<p>For any image you upload, you confirm that:</p><ul>"
            "<li>you own it or have all rights and permissions necessary to use it;</li>"
            "<li>it does not depict copyrighted characters, brand logos, or celebrities/public figures without authorization;</li>"
            "<li>it contains no explicit, unlawful, hateful, defamatory, or infringing material;</li>"
            "<li>it does not depict a minor without the consent of their parent or guardian.</li></ul>"
            "<p>You are solely responsible for the images and words you submit. You grant Typortrait a limited license to process your image and text only to create and deliver your portrait.</p>"),
        ("4. Prohibited uses", "<p>Do not use the Service for unlawful, infringing, deceptive, or harmful purposes; do not upload other people&rsquo;s images without permission; do not resell or redistribute the Service itself.</p>"),
        ("5. Purchases and refunds", "<p>Prices are shown at checkout and processed securely by Stripe. The preview is free and watermarked; payment unlocks a watermark-free, high-resolution download for your personal and gift use. Digital files are delivered immediately: by purchasing you agree to that immediate delivery and, where applicable, acknowledge that you waive your 14-day right of withdrawal once the download begins. Personalised prints are made to order. If anything is wrong &mdash; a quality issue with your file, or a print that arrives damaged or not as ordered &mdash; <b>we&rsquo;ll fix it or refund it</b>. See our <a href=\"/refunds\">Refund Policy</a> for the details.</p>"),
        ("6. Intellectual property", "<p>You receive a license to use your generated portrait for personal, non-commercial purposes, including printing and gifting. Typortrait retains all rights in the Service, software, and brand.</p>"),
        ("6.5 Optional social-sharing license", "<p>After purchase you may, at your option, generate a short shareable &ldquo;reel&rdquo; version of your portrait. At that step you will see two checkboxes:</p><ul>"
            "<li><b>Personal use</b> &mdash; confirms that you will share the reel yourself. This does not transfer any rights to Typortrait.</li>"
            "<li><b>Allow Typortrait to feature this on our social channels</b> &mdash; this box is <b>off by default</b>. If you tick it, you grant Typortrait a revocable, non-exclusive, worldwide, royalty-free license to use the resulting reel (which includes your portrait, the words you provided, and the photo as used inside the reel) for promotional posts on Typortrait&rsquo;s own websites and social channels.</li></ul>"
            "<p>You can revoke the optional license at any time by emailing us, and we will remove the post within a reasonable period. Generating the reel is always optional; nothing is shared on our channels unless you explicitly tick the second box.</p>"),
        ("7. Disclaimer", "<p>The Service is provided &ldquo;as is&rdquo; without warranties of any kind. Results depend on the photo you provide.</p>"),
        ("8. Limitation of liability", "<p>To the maximum extent permitted by law, Typortrait is not liable for indirect or consequential damages, and total liability will not exceed the amount you paid.</p>"),
        ("9. Indemnification", "<p>You agree to indemnify Typortrait against claims arising from your content or your breach of these Terms.</p>"),
        ("10. Changes and governing law", "<p>We may update these Terms; continued use means you accept the changes. These Terms are governed by the laws of New York State, United States of America.</p>"),
    ]
    return _policy_page("Terms of Use", blocks)


@app.get("/refunds", response_class=HTMLResponse)
def refunds():
    blocks = [
        ("Our promise", "<p>We want you to love your portrait. If anything is wrong, tell us &mdash; <b>we&rsquo;ll fix it or refund it.</b> Email <a href='mailto:support@typortrait.com'>support@typortrait.com</a>.</p>"),
        ("Digital downloads", "<p>Your watermark-free download is delivered instantly, so purchases are generally final once delivered. But if the file has a quality problem &mdash; a poor likeness, a rendering glitch, or the wrong file &mdash; we will <b>re-create it or refund you in full</b>. Just contact us with your order details.</p>"),
        ("Prints &amp; physical products", "<p>Prints are made to order, so we don&rsquo;t accept &ldquo;changed my mind&rdquo; returns on a personalised item. But if your print arrives <b>damaged, defective, or not as ordered</b>, we&rsquo;ll <b>reprint or refund it</b> &mdash; send us a photo of the issue and we&rsquo;ll make it right.</p>"),
        ("How to request a refund", "<p>Email <a href='mailto:support@typortrait.com'>support@typortrait.com</a> with your order number (and a photo, for print issues). Approved refunds go back to your original payment method via Stripe, usually within a few business days.</p>"),
        ("EU/UK customers", "<p>For digital downloads you agree at checkout to immediate delivery and acknowledge that you lose your 14-day right of withdrawal once the download begins. Personalised prints are likewise exempt from the 14-day withdrawal right. None of this affects your statutory rights regarding faulty goods, which we always honour.</p>"),
    ]
    return _policy_page("Refund Policy", blocks)


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    blocks = [
        ("Overview", "<p>This Privacy Policy explains what Typortrait (&ldquo;we&rdquo;) collects, why, and your rights. We collect only what we need to create your portrait and process your order. For how we handle facial geometry specifically, see our <a href=\"/biometric-policy\">Biometric Data Policy</a>.</p>"),
        ("Who we are", f"<p>Typortrait (which also operates the <b>Loved in Words</b> memorial service) is the organization responsible for, and the data controller of, the information described here. Contact our privacy contact at <a href='mailto:{_POLICY_CONTACT}'>{_POLICY_CONTACT}</a> with any privacy question, to exercise your rights, or to raise a concern.</p>"),
        ("What we collect", "<ul>"
            "<li><b>Images you upload</b> and the <b>words/message</b> you enter.</li>"
            "<li><b>Facial geometry</b> derived from your photo, used transiently to place the type. We do <b>not</b> create or store a faceprint/biometric template &mdash; see the <a href=\"/biometric-policy\">Biometric Data Policy</a>.</li>"
            "<li><b>Payment information</b>, processed by Stripe. We do not see or store your full card details.</li>"
            "<li><b>Order and contact details</b> (e.g., email; shipping address for prints).</li>"
            "<li><b>Basic technical data</b> (e.g., server logs, approximate region) needed to operate and secure the Service.</li></ul>"),
        ("Legal bases (GDPR/UK GDPR)", "<p>Where the UK/EU GDPR applies, we rely on: <b>your explicit consent</b> to analyse facial geometry (a special category of data &mdash; we ask before processing); <b>performance of a contract</b> to create and deliver your portrait and fulfil orders; and our <b>legitimate interests</b> in operating, securing, and improving the Service. You can withdraw consent at any time, which stops further processing.</p>"),
        ("How we use it", "<p>To generate and deliver your portrait, process your payment and any print order, provide support, and operate, secure, and improve the Service. We do <b>not</b> use your photo to train face-recognition systems, and we do not use it to identify anyone.</p>"),
        ("Storage and retention", f"<p>Your uploaded photo and generated files are stored only to provide your preview and download, and are <b>automatically deleted after about {RETENTION_DAYS} days</b>. Facial geometry is computed in memory at render time and discarded immediately &mdash; no faceprint is retained. You can request earlier deletion at any time via our <a href=\"/data-request\">data request page</a>.</p>"
            "<p>We may retain limited order, payment, and transaction records for longer where required for tax, accounting, fraud-prevention, or other legal obligations. We also keep a minimal <b>record of your consent</b> (the fact that you consented and confirmed your age &mdash; no photo or biometric data) for longer, so we can demonstrate that consent was given.</p>"
            "<p>For resilience, we keep encrypted, off-site backups. After you delete your data (or it reaches the deletion period above), a copy may persist in those backups for a short additional period &mdash; about 35 days &mdash; until they rotate and it is purged. Deletions you request take effect in the live Service immediately.</p>"),
        ("Sharing and sub-processors", "<p>We share personal data only with the service providers (sub-processors) needed to run Typortrait:</p><ul><li><b>Stripe</b> &mdash; payment processing (we never see your full card details).</li><li><b>Printful</b> &mdash; print-on-demand fulfilment, physical orders only (receives your shipping details).</li><li><b>IONOS</b> &mdash; hosting and infrastructure (stores your photo and order data on our behalf).</li><li><b>OpenAI</b> &mdash; content moderation of the words you type (never your photo).</li><li><b>Umami</b> &mdash; privacy-first, cookieless analytics.</li></ul><p>We <b>do not sell or &ldquo;share&rdquo;</b> your personal information (as those terms are defined under U.S. state privacy laws), and we do not use it for cross-context behavioural advertising. We may disclose information if required by law. This list may change as our providers do; the &ldquo;last updated&rdquo; date above reflects the current set.</p>"
            "<p><b>Optional social feature.</b> After purchase you may generate a shareable &ldquo;reel&rdquo;; it appears on our channels only if you explicitly tick the optional box (off by default). See section 6.5 of the <a href=\"/terms\">Terms of Use</a>.</p>"),
        ("International transfers", "<p>We operate from the United States, so if you are outside the U.S. your information is transferred to and processed there. Where required, we rely on appropriate safeguards (such as the EU Standard Contractual Clauses) and/or your consent.</p>"),
        ("Your rights", "<p>Depending on where you live, you may have rights to access, correct, delete, port, or object to/restrict processing of your data, and to withdraw consent. To exercise them, use our <a href=\"/data-request\">data request page</a> or email us. We will not discriminate against you for exercising your rights, and we respond within the time limits the law requires.</p>"),
        ("California (CCPA/CPRA)", "<p>California residents have the rights to know, delete, and correct their personal information, and to limit the use of sensitive personal information. Your photo and facial geometry are treated as <b>sensitive personal information</b>, used only to provide the portrait you requested. We do not sell or share personal information. Submit a request via our <a href=\"/data-request\">data request page</a>.</p>"),
        ("Canada (PIPEDA)", "<p>If you are in Canada, we handle your personal information in accordance with the federal "
            "<b>Personal Information Protection and Electronic Documents Act (PIPEDA)</b> and substantially similar provincial "
            "laws, including <b>Quebec&rsquo;s Law&nbsp;25</b> and the personal-information laws of British Columbia and Alberta. "
            "We collect, use, and disclose personal information only for the purposes described in this policy, and we obtain "
            "your <b>express consent</b> before analysing the facial geometry in your photo &mdash; sensitive information that "
            "we process transiently and <b>never retain as a faceprint</b>. We do <b>not</b> create or maintain a biometric "
            "database (see our <a href=\"/biometric-policy\">Biometric Data Policy</a>). You may <b>access or correct</b> your "
            "information, <b>withdraw consent</b>, or request its deletion at any time through our "
            "<a href=\"/data-request\">data request page</a>. Because we operate from the United States, your information is "
            "transferred to and processed there and may be subject to access by U.S. authorities under U.S. law; we remain "
            "accountable for it and require our service providers to safeguard it. If you have a concern we cannot resolve, "
            "you may contact the <b>Office of the Privacy Commissioner of Canada</b> &mdash; and, in Quebec, the "
            "<b>Commission d&rsquo;acc&egrave;s &agrave; l&rsquo;information</b>.</p>"),
        ("Photos of other people", "<p>If you upload a photo of someone else, you confirm you have the right to do so (see the <a href=\"/terms\">Terms of Use</a>). If you believe your image was uploaded without your permission, contact us via the <a href=\"/data-request\">data request page</a> and we will remove it.</p>"),
        ("Children", "<p>The Service is not directed to children under 13, and you should not upload a minor&rsquo;s photo without the consent of their parent or guardian.</p>"),
        ("Security", "<p>We use reasonable technical and organisational measures to protect your data, but no method of transmission or storage is completely secure.</p>"),
        ("Changes", "<p>We may update this policy; the &ldquo;last updated&rdquo; date above will change accordingly.</p>"),
    ]
    return _policy_page("Privacy Policy", blocks)


@app.get("/biometric-policy", response_class=HTMLResponse)
def biometric_policy():
    blocks = [
        ("What this covers", "<p>This Biometric Data Policy explains how Typortrait handles facial geometry, including for the purposes of the Illinois Biometric Information Privacy Act (BIPA) and similar laws. It is our publicly available written policy for biometric data.</p>"),
        ("What we analyse and why", "<p>To turn your photo into a typographic portrait, our software analyses the geometry of the face in the image (the relative positions of facial features) so it can place the type along the contours of the face. This face-geometry analysis is the only biometric processing we perform, and its sole purpose is to render the artwork you requested.</p>"),
        ("We do not store a faceprint", "<p>The facial geometry is computed in memory at the moment of rendering and then discarded. We do <b>not</b> create, store, or maintain a biometric template or &ldquo;faceprint,&rdquo; and we do <b>not</b> use it to identify you or anyone else, for face recognition, for surveillance, or to train such systems.</p>"),
        ("Consent", "<p>Before we process your photo, we ask you to confirm that you understand this analysis and consent to it, and that you have the right to upload the image. We do not process the face until you give that consent, and you can withdraw it at any time.</p>"),
        ("Retention and destruction", f"<p>Because we keep no biometric template, there is nothing biometric to retain beyond the moment of rendering. The <b>source photo</b> you upload is stored only to provide your preview and download and is automatically deleted after about {RETENTION_DAYS} days, or sooner on request, and purged from our encrypted off-site backups as they rotate (within about 35 days). This is our written retention schedule and destruction guideline for biometric data.</p>"),
        ("No sale or disclosure", "<p>We do not sell, lease, trade, or otherwise profit from biometric data, and we do not disclose it to third parties except as strictly necessary to provide the Service or as required by law.</p>"),
        ("Canada (PIPEDA &amp; Quebec Law&nbsp;25)", "<p>For visitors in Canada, the facial-geometry analysis described here is "
            "performed transiently and immediately discarded: we do <b>not</b> create, store, or maintain a biometric "
            "database. We obtain your <b>express consent</b> before any analysis and honour withdrawal and deletion requests, "
            "consistent with <b>PIPEDA</b> and <b>Quebec&rsquo;s Law&nbsp;25</b>. You can withdraw consent and request deletion "
            "at any time via our <a href=\"/data-request\">data request page</a>.</p>"),
        ("Availability", "<p>To reduce risk, the Service may be unavailable to visitors in certain locations (for example, Illinois). Where it is available, the protections above apply.</p>"),
        ("Your choices", "<p>You can withdraw consent and request deletion of your photo at any time via our <a href=\"/data-request\">data request page</a>. See our <a href=\"/privacy\">Privacy Policy</a> for the full picture.</p>"),
    ]
    return _policy_page("Biometric Data Policy", blocks)


_JOB_RE = re.compile(r"^[a-f0-9]{12}$")


def _purge_job(job_id: str) -> int:
    """Delete the personal/biometric artifacts for a job (source photo, recipe,
    mask, previews, clean PNGs). The minimal CONSENT RECORD is intentionally KEPT
    (it holds no photo/biometric data) as proof that consent was given -- evidence
    we may need even after a deletion request. Returns the number of files removed."""
    if not _JOB_RE.match((job_id or "").strip().lower()):
        return 0
    jid = job_id.strip().lower()
    removed = 0
    for d, pattern in ((PRIVATE_DIR, f"{jid}.*"), (PRIVATE_DIR, f"{jid}_*"),
                       (OUTPUTS_DIR, f"{jid}_*"), (OUTPUTS_DIR, f"{jid}.*")):
        try:
            for f in d.glob(pattern):
                try:
                    if f.name.endswith(".consent.json") or f.name.endswith(".biometric_consent.json"):
                        continue   # retain consent evidence (no biometric data)
                    if f.is_file():
                        f.unlink()
                        removed += 1
                except OSError:
                    pass
        except OSError:
            pass
    return removed


_DATA_REQUEST_FORM = (
    "<h2>Request access to or deletion of your data</h2>"
    "<p>Use this form to ask us to delete the photo and files for a portrait you created, "
    "or to access, correct, or object to the data we hold. If you have your portrait link, "
    "paste its job ID for an immediate deletion; otherwise leave it blank and we&rsquo;ll "
    "action your request and reply by email.</p>"
    "<form method='post' action='/data-request' class='dr'>"
    "<label>Your email<br><input type='email' name='email' required placeholder='you@example.com'></label>"
    "<label>Request type<br><select name='request_type'>"
    "<option value='delete'>Delete my data</option>"
    "<option value='access'>Access my data</option>"
    "<option value='correct'>Correct my data</option>"
    "<option value='object'>Object to / restrict processing</option>"
    "<option value='takedown'>Remove a photo of me uploaded by someone else</option>"
    "<option value='report_abuse'>Report inappropriate or abusive content</option>"
    "</select></label>"
    "<label>Portrait job ID (optional, 12 characters)<br><input type='text' name='job_id' maxlength='12' placeholder='e.g. a1b2c3d4e5f6'></label>"
    "<label>Details (optional)<br><textarea name='details' rows='3'></textarea></label>"
    "<button type='submit'>Submit request</button>"
    "</form>"
    "<style>form.dr label{display:block;margin:14px 0}form.dr input,form.dr select,form.dr textarea"
    "{width:100%;max-width:420px;padding:8px;border:1px solid #d8d4cc;border-radius:8px;font:inherit;box-sizing:border-box}"
    "form.dr button{margin-top:16px;background:#0d1b3a;color:#fff;border:0;border-radius:8px;padding:10px 18px;font:inherit;cursor:pointer}</style>"
)


@app.get("/data-request", response_class=HTMLResponse)
def data_request_page():
    return _policy_page("Data Request", [("", _DATA_REQUEST_FORM)])


@app.post("/data-request", response_class=HTMLResponse)
async def data_request_submit(
    request: Request,
    email: str = Form(""),
    request_type: str = Form("delete"),
    job_id: str = Form(""),
    details: str = Form(""),
):
    """Self-serve data request. A valid job ID is deleted immediately; every
    request is logged server-side so the operator can fulfil non-job requests
    within the statutory window (see data/data_requests.log)."""
    jid = (job_id or "").strip().lower()
    deleted = _purge_job(jid) if jid else 0
    import time as _t, threading
    rec = {
        "ts": int(_t.time()),
        "email": (email or "").strip()[:200],
        "type": (request_type or "")[:40],
        "job_id": jid[:24],
        "deleted_files": deleted,
        "details": (details or "")[:2000],
        "region": _region_token(request) or None,
    }
    try:
        with (DATA_DIR / "data_requests.log").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001
        pass
    # Notify the admin so the statutory response window isn't missed (best-effort,
    # off-thread; no-op if SMTP isn't configured).
    try:
        threading.Thread(target=admin_mod.send_data_request_email, args=(rec,), daemon=True).start()
    except Exception:  # noqa: BLE001
        pass
    msg = ("Your portrait files were deleted." if deleted
           else "We&rsquo;ve received your request.")
    body = (f"<h2>Thank you</h2><p>{msg} We will action your request and contact you at the email "
            "you provided if anything further is needed. Most requests are completed within 30 days "
            "(sooner where the law requires).</p>"
            "<p><a href='/static/index.html'>&larr; Back to Typortrait</a></p>")
    return _policy_page("Data request received", [("", body)])


@app.get("/compliance/region")
def compliance_region(request: Request) -> JSONResponse:
    """Lets the studio disable upload up front for blocked regions. The /measure
    and /render gates still enforce this server-side regardless of the client."""
    blocked = _blocked_region(request)
    return JSONResponse({"blocked": bool(blocked),
                         "detail": _GEO_BLOCK_DETAIL if blocked else ""})