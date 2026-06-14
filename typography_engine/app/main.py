"""FastAPI entrypoint for the isolated typography portrait engine.

Phase 1: project structure + health endpoint.
Phase 2: image upload -> silhouette / edge / landmark debug images.
Later phases add /render.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import admin as admin_mod
from . import orders as orders_db
from . import printful, products
from .config import (
    CURRENCY,
    DATA_DIR,
    DOWNLOAD_PNG_WIDTH,
    DOWNLOAD_PRICE_CENTS,
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
    WATERMARK_URL,
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

# Phase C: admin review dashboard + email notifications for consented reels.
app.include_router(admin_mod.router)


def _cleanup_old_files() -> int:
    """Delete previews and stored job inputs older than the retention window so
    the Privacy Policy's deletion statement stays accurate. Returns count removed."""
    import time
    cutoff = time.time() - RETENTION_DAYS * 86400
    removed = 0
    for d in (OUTPUTS_DIR, PRIVATE_DIR):
        try:
            for f in d.iterdir():
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff:
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
def _start_admin_services() -> None:
    """Promote any legacy .queue markers to .review.json records, then start
    the background email scanner that notifies the admin of new queued reels."""
    try:
        admin_mod.migrate_legacy_queue_markers()
    except Exception:  # noqa: BLE001
        pass
    admin_mod.start_scanner()


@app.get("/")
def index() -> RedirectResponse:
    # This deployment (VPS / app.typortrait.com) is the studio app. The marketing
    # landing page is served separately from a static host (typortrait.com).
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health() -> JSONResponse:
    caps = probe()
    return JSONResponse(
        {
            "ok": True,
            "service": "typography-portrait-engine",
            "version": __version__,
            "capabilities": caps,
        }
    )


@app.post("/debug/preprocess")
async def debug_preprocess(image: UploadFile = File(...)) -> JSONResponse:
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


@app.post("/measure")
async def measure(image: UploadFile = File(...)) -> JSONResponse:
    """Lightweight pre-render check: detect the subject's face and return its
    width as a fraction of the photo, plus the word sizes that stay readable for
    that framing. The studio calls this on upload so it can grey out size options
    (e.g. Small) that would produce unreadable type for a far/small subject."""
    data = await image.read()
    if not data:
        return JSONResponse({"ok": False, "error": "empty_upload"}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"ok": False, "error": "file_too_large"}, status_code=413)
    warns = WarningCollector()
    try:
        from .pipeline.preprocess import load_and_normalize
        from .pipeline.landmarks import detect_faces, haar_face_bbox
        cfg = RenderConfig()
        img = load_and_normalize(data, cfg.work_max_dim, warns)
        faces = detect_faces(img, warns)
        bbox = faces[0].bbox if faces else haar_face_bbox(img, warns)
        w = float(img.gray.shape[1]) or 1.0
        face_frac = round(float(bbox[2]) / w, 4) if bbox else None
        n_faces = len(faces)            # 0/1 via haar fallback; >=2 only when MediaPipe sees a group
    except Exception as e:  # noqa: BLE001
        # On any failure, allow all sizes (the renderer's floor still protects).
        return JSONResponse({"ok": True, "face_frac": None, "faces": 0,
                             "sizes": _allowed_size_mf(None), "default": _recommended_size_mf(None)})
    return JSONResponse({"ok": True, "face_frac": face_frac, "faces": n_faces,
                         "sizes": _allowed_size_mf(face_frac), "default": _recommended_size_mf(face_frac)})


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    words: Optional[str] = Form(None),
    words_json: Optional[str] = Form(None),
    min_font_px: Optional[float] = Form(None),
    uppercase: bool = Form(True),
    background_hex: Optional[str] = Form(None),
    foreground_hex: Optional[str] = Form(None),
    ink: str = Form("navy"),
    style: str = Form("mosaic"),
    message: Optional[str] = Form(None),
    poster: bool = Form(False),
    title: Optional[str] = Form(None),
    caption: Optional[str] = Form(None),
    png_width: int = Form(2000),
    render_w: int = Form(2600),
    remove_bg: bool = Form(True),
    light: bool = Form(False),
    ground: str = Form("navy"),
    ref: str = Form(""),
) -> JSONResponse:
    """Render a typographic portrait: validated SVG + PNG from approved words."""
    warns = WarningCollector()
    ref_clean = re.sub(r"[^A-Za-z0-9_-]", "", ref or "")[:40]   # referral/source tag
    img_bytes = await image.read()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "empty_upload"}, status_code=400)
    if len(img_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"ok": False, "error": "file_too_large",
             "detail": "That image is too large — please use one under 25 MB."},
            status_code=413,
        )

    word_list = _parse_words(words, words_json)
    if not word_list:
        return JSONResponse({"ok": False, "error": "no_words"}, status_code=400)

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

    try:
        an = analyze_image(img_bytes, cfg, warns)
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

    from .pipeline.tonal import _PALETTES, _GRADIENTS, render_layered_png
    ink_choice = ink if (ink in _PALETTES or ink in _GRADIENTS or ink == "photo") else "navy"
    # User-facing styles: "displacement" = type-follows-the-form portrait (its own
    # raster renderer + ground choice); "message" = poster rows; anything else =
    # "words" (the scattered mosaic). Words/Passage share the layered renderer.
    is_displacement = (style == "displacement")
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
            png_bytes = render_displacement_portrait(
                an, disp_words, ground=ground_choice, out_width=max(320, preview_w),
                uppercase=uppercase, ink=ink_choice)
            runs, ground_hex, mask_svg = [], None, None
        else:
            png_bytes, runs, ground_hex, mask_svg = render_layered_png(
                an, text, style_choice, cfg, warns,
                ink=ink_choice, remove_bg=remove_bg, light=light,
                out_width=max(320, preview_w), render_w=render_w_eff, uppercase=uppercase)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e), "warnings": warns.as_list()}, status_code=400)
    except Exception as e:  # noqa: BLE001
        warns.error("render", "render_failed", str(e))
        return JSONResponse({"ok": False, "error": "render_failed", "detail": str(e), "warnings": warns.as_list()}, status_code=500)

    if warns.has_errors() or not png_bytes:
        return JSONResponse({"ok": False, "error": "render_incomplete", "warnings": warns.as_list()}, status_code=422)

    job_id = uuid.uuid4().hex[:12]
    preview_path = OUTPUTS_DIR / f"{job_id}_preview.png"
    try:
        from .pipeline.watermark import add_watermark
        preview_path.write_bytes(add_watermark(png_bytes, url=WATERMARK_URL))
    except Exception as e:  # noqa: BLE001
        warns.warn("render", "preview_failed", f"Watermark failed: {e}")
        preview_path.write_bytes(png_bytes)

    # Persist the inputs so the paid, high-res PNG can be recomposed once at
    # download (after payment). Skip for throwaway swatch-thumbnail renders.
    if not is_thumb:
        (PRIVATE_DIR / f"{job_id}.src").write_bytes(img_bytes)
        if mask_svg:
            (PRIVATE_DIR / f"{job_id}.mask.svg").write_text(mask_svg, encoding="utf-8")
        (PRIVATE_DIR / f"{job_id}.json").write_text(json.dumps({
            "style": style_choice, "ink": ink_choice, "remove_bg": bool(remove_bg),
            "light": bool(light), "text": text, "uppercase": bool(uppercase),
            "min_font_px": float(cfg.min_font_px), "ground": ground_choice,
            "ref": ref_clean,
        }), encoding="utf-8")

    # Likeness score (how well this render preserves the face) -- the studio
    # compares scores across styles to recommend the best one for this photo.
    try:
        from .pipeline.score import likeness_score
        likeness = likeness_score(an, png_bytes)
    except Exception:  # noqa: BLE001
        likeness = None

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


@app.get("/products")
def list_products() -> JSONResponse:
    """Storefront catalog (prices, shipping, sizes) — single source of truth so
    the UI never hardcodes prices. `fulfillment_configured` tells the front-end
    whether physical print-on-demand is live (a Printful token is present) or
    only the digital download should be offered."""
    return JSONResponse({
        "ok": True,
        "currency": CURRENCY,
        "products": products.public_catalog(),
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


@app.post("/checkout")
def checkout(
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
                success_url=f"{PUBLIC_BASE_URL}/success?job={job}&session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{PUBLIC_BASE_URL}/static/index.html?canceled=1",
            )
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": "stripe_error", "detail": str(e)}, status_code=502)
        return JSONResponse({"ok": True, "url": session.url})

    # --- Physical print: gated on a configured Printful token + valid size ----
    if not PRINTFUL_API_TOKEN:
        return JSONResponse({"ok": False, "error": "fulfillment_unconfigured"}, status_code=503)
    variant_id = products.resolve_variant_id(product, size)
    if variant_id is None:
        return JSONResponse(
            {"ok": False, "error": "missing_or_invalid_size",
             "sizes": list(product.size_variants.keys()) if product.size_variants else []},
            status_code=400,
        )
    order_id = uuid.uuid4().hex[:16]
    ext = "svg" if fmt == "svg" else "png"
    label_size = f" — {size}" if size else ""
    line_items: List[dict] = [{
        "quantity": 1,
        "price_data": {
            "currency": CURRENCY,
            "unit_amount": product.price_cents,
            "product_data": {"name": f"{brand['name']} — {product.name}{label_size}"},
        },
    }]
    if product.shipping_cents > 0:
        line_items.append({
            "quantity": 1,
            "price_data": {
                "currency": CURRENCY,
                "unit_amount": product.shipping_cents,
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
            success_url=f"{PUBLIC_BASE_URL}/order/{order_id}?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{PUBLIC_BASE_URL}/static/index.html?canceled=1",
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
            price_cents=product.price_cents,
            shipping_cents=product.shipping_cents,
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
        from .pipeline.tonal import compose_layered, render_layered_png
        r = json.loads(recipe_path.read_text(encoding="utf-8"))
        warns2 = WarningCollector()
        an = analyze_image(src_path.read_bytes(), RenderConfig(), warns2)
        if r.get("style") == "displacement":
            from .pipeline.displacement import render_displacement_portrait
            png_bytes = render_displacement_portrait(
                an, (r.get("text", "") or "").split(),
                ground=r.get("ground", "navy"), out_width=DOWNLOAD_PNG_WIDTH,
                uppercase=bool(r.get("uppercase", True)), ink=r.get("ink"),
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
                light=bool(r.get("light", False)), boost=dl_boost, print_aspect=aspect)
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
                uppercase=bool(r.get("uppercase", True)), print_aspect=aspect)
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
        # Warm the print file in the background so Printful's fetch is served from
        # cache: the high-res compose takes ~30-45s and we don't want it to land on
        # Printful's fetch timeout. Idempotent -- the signed URL still composes
        # lazily as a fallback if the warm hasn't finished by the time they fetch.
        import threading
        job_id = o["job_id"]
        threading.Thread(target=lambda: _ensure_clean_png(job_id, aspect), daemon=True).start()
        signed = printful.signed_print_url(job_id, aspect=aspect)
        placement = "front" if str(o["sku"]).startswith("tshirt") else "default"
        res = printful.create_order(
            recipient=recipient,
            variant_id=o["variant_id"],
            print_file_url=signed,
            external_id=order_id,
            retail_price_cents=o["price_cents"],
            confirm=PRINTFUL_CONFIRM,
            placement=placement,
        )
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
        _fulfill_with_printful(transitioned["id"], recipient)

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
    return JSONResponse({"ok": True})


def _reel_maker_block(job: str, session_id: str) -> str:
    """Self-contained 'Make a reel' card (HTML + scoped CSS + JS) for any paid
    job. Identical dual-consent flow and /reel call as the digital success
    page, so print buyers get the exact same reel ability and authorizations.
    Portable: carries its own styles so it can drop into any page chrome."""
    j = json.dumps(job)
    s = json.dumps(session_id)
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
        '<h2 class="rl-h">Make a reel from your portrait</h2>'
        '<p class="rl-sub">Turn it into a short 9:16 video — the dissolve, your words, and the '
        'finished portrait — perfect for sharing.</p>'
        '<label class="rl-chk"><input type="checkbox" id="cp"> '
        '<span>Yes, I’ll share my reel for personal use.</span></label>'
        '<label class="rl-chk"><input type="checkbox" id="ct"> '
        '<span>Optional: Allow Typortrait to feature this reel on our own social channels '
        '(<a href="/terms" target="_blank" rel="noopener">terms</a>). You can revoke any time by emailing us.</span></label>'
        '<button class="btn" id="mk" disabled>Make my reel</button>'
        '<div id="rlOut" style="display:none">'
        '<p class="rl-sub" id="rlSub" style="margin-top:14px">Your reel is ready.</p>'
        '<a class="btn" id="rlMp4" download="typortrait-reel.mp4" style="display:none">Download MP4</a>'
        '<a class="btn ghost" id="rlGif" download="typortrait-reel.gif">Download GIF</a>'
        '<button class="btn ghost" id="rlSh" style="display:none">Share reel</button>'
        '</div>'
        '<script>(function(){'
        'var job=' + j + ',sid=' + s + ',o=location.origin;'
        'var shareUrl=o+"/p/"+job;'
        'var cp=document.getElementById("cp"),ct=document.getElementById("ct"),mk=document.getElementById("mk");'
        'var rlOut=document.getElementById("rlOut"),rlSub=document.getElementById("rlSub");'
        'var rlGif=document.getElementById("rlGif"),rlMp4=document.getElementById("rlMp4"),rlSh=document.getElementById("rlSh");'
        'cp.onchange=function(){mk.disabled=!cp.checked;};'
        'mk.onclick=function(){if(!cp.checked)return;'
        'mk.disabled=true;mk.innerHTML=\'<span class="rl-spin"></span>Making your reel… (~30s)\';'
        'var fd=new FormData();fd.append("job",job);fd.append("session_id",sid);'
        'fd.append("personal_consent",cp.checked?"on":"");'
        'fd.append("typortrait_consent",ct.checked?"on":"");'
        'fetch("/reel",{method:"POST",body:fd}).then(function(r){return r.json();}).then(function(j){'
        'if(!j||!j.ok){mk.disabled=false;mk.innerHTML="Make my reel";'
        'rlSub.textContent="Sorry, that didn\\u2019t work: "+((j&&(j.detail||j.error))||"please try again.");'
        'rlOut.style.display="block";return;}'
        'mk.style.display="none";rlOut.style.display="block";'
        'rlSub.textContent=j.typortrait_share?"Your reel is ready. Thanks \\u2014 we\\u2019ll review it before posting on our channels.":"Your reel is ready.";'
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
                    _fulfill_with_printful(transitioned["id"], recipient)
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
        dl = f'/download?job={o["job_id"]}&fmt=png&session_id={session_id}'
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
    reel_html = ""
    if o["status"] in ("paid", "fulfilling", "shipped", "delivered") and session_id:
        reel_html = _reel_maker_block(o["job_id"], session_id)

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
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
  <p><a href="/static/index.html">Make another Typortrait →</a></p>
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
        build_reel({
            "before": str(src_path),       # original photo, kept at its real aspect ratio
            "after":  str(portrait_path),
            "words":  words,
            "out":    str(gif_path),
            "aspect": "source",            # personal reel preserves the original aspect ratio
            "minimal": True,               # no CTA / promo captions — just a discreet credit
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
    jq, sq = quote(job, safe=""), quote(session_id, safe="")
    png_url = f"/download?job={jq}&fmt=png&session_id={sq}"
    if paid:
        # Conversion event for digital sales (the primary revenue path, which
        # completes here rather than via the webhook). Deduped + revenue/sku
        # filled from Stripe, so a page refresh can't double-count.
        _track_purchase_once(session_id, sku="digital")
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
            # ---- Phase B: optional "Make a reel" card with dual consent ----
            '<div class="divider"></div>'
            '<h2 class="reel-h">Make a reel from your portrait</h2>'
            '<p class="sub reel-sub">Turn it into a short 9:16 video — the dissolve, your words, and the finished portrait — perfect for sharing.</p>'
            '<label class="chk"><input type="checkbox" id="cp"> '
            '<span>Yes, I’ll share my reel for personal use.</span></label>'
            '<label class="chk"><input type="checkbox" id="ct"> '
            '<span>Optional: Allow Typortrait to feature this reel on our own social channels '
            '(<a href="/terms" target="_blank" rel="noopener">terms</a>). You can revoke any time by emailing us.</span></label>'
            '<button class="btn" id="mk" disabled>Make my reel</button>'
            '<div class="link-wrap"><a class="link" href="/static/index.html">Create another portrait</a></div>'
            '<div id="rlOut" style="display:none">'
            '<p class="sub" id="rlSub">Your reel is ready.</p>'
            '<a class="btn" id="rlMp4" download="typortrait-reel.mp4" style="display:none">Download MP4</a>'
            '<a class="btn ghost" id="rlGif" download="typortrait-reel.gif">Download GIF</a>'
            '<button class="btn ghost" id="rlSh" style="display:none">Share reel</button>'
            '</div>'
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
            'mk.disabled=true;mk.innerHTML=\'<span class="spin"></span>Making your reel… (~30s)\';'
            'var fd=new FormData();fd.append("job",job);fd.append("session_id",sid);'
            'fd.append("personal_consent",cp.checked?"on":"");'
            'fd.append("typortrait_consent",ct.checked?"on":"");'
            'fetch("/reel",{method:"POST",body:fd}).then(function(r){return r.json();}).then(function(j){'
            'if(!j||!j.ok){mk.disabled=false;mk.innerHTML="Make my reel";'
            'rlSub.textContent="Sorry, that didn\\u2019t work: "+((j&&(j.detail||j.error))||"please try again.");'
            'rlOut.style.display="block";return;}'
            'mk.style.display="none";rlOut.style.display="block";'
            'rlSub.textContent=j.typortrait_share?"Your reel is ready. Thanks \\u2014 we\\u2019ll review it before posting on our channels.":"Your reel is ready.";'
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
            '})();</script>'
        )
    else:
        inner = (
            '<h1>Finishing up&hellip;</h1>'
            '<p class="sub">If your payment just completed, your download will be ready in a moment. '
            'Refresh this page; if it doesn&rsquo;t appear, your card may not have been charged.</p>'
            f'<a class="btn" href="/success?job={html.escape(jq)}&amp;session_id={html.escape(sq)}">Refresh</a>'
            '<a class="link" href="/static/index.html">Back to Typortrait</a>'
        )
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
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
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
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
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
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
_POLICY_UPDATED = "May 2026"


def _policy_page(title: str, blocks) -> HTMLResponse:
    body = "".join(f"<h2>{h}</h2>{p}" for h, p in blocks)
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
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
        "<a class='back' href='/static/index.html'>&larr; Back to Typortrait</a>"
        f"<h1>{title}</h1><p class='upd'>Last updated {_POLICY_UPDATED}</p>"
        + body +
        f"<h2>Contact</h2><p>Questions? Email <a href='mailto:{_POLICY_CONTACT}'>{_POLICY_CONTACT}</a>.</p>"
        "</div></body></html>"
    )
    return HTMLResponse(page)


@app.get("/terms", response_class=HTMLResponse)
def terms():
    blocks = [
        ("1. Acceptance", "<p>By using Typortrait (the &ldquo;Service&rdquo;) you agree to these Terms of Use. If you do not agree, please do not use the Service. You must be at least 18 years old, or have the consent of a parent or legal guardian.</p>"),
        ("2. The Service", "<p>Typortrait turns a photo and words you provide into a typographic portrait, offered as a free watermarked preview and a paid, watermark-free digital download.</p>"),
        ("3. Your content and rights", "<p>For any image you upload, you confirm that:</p><ul>"
            "<li>you own it or have all rights and permissions necessary to use it;</li>"
            "<li>it does not depict copyrighted characters, brand logos, or celebrities/public figures without authorization;</li>"
            "<li>it contains no explicit, unlawful, hateful, defamatory, or infringing material;</li>"
            "<li>it does not depict a minor without the consent of their parent or guardian.</li></ul>"
            "<p>You are solely responsible for the images and words you submit. You grant Typortrait a limited license to process your image and text only to create and deliver your portrait.</p>"),
        ("4. Prohibited uses", "<p>Do not use the Service for unlawful, infringing, deceptive, or harmful purposes; do not upload other people&rsquo;s images without permission; do not resell or redistribute the Service itself.</p>"),
        ("5. Purchases and downloads", "<p>Prices are shown at checkout and processed securely by Stripe. The preview is free and watermarked; payment unlocks a watermark-free, high-resolution download for your personal and gift use. Because files are delivered immediately, sales are final once the file is delivered &mdash; but if anything is wrong with your file, contact us and we&rsquo;ll make it right.</p>"),
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


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    blocks = [
        ("Overview", "<p>This Privacy Policy explains what information Typortrait handles and how. We aim to collect only what we need to create your portrait and process your order.</p>"),
        ("What we collect", "<ul>"
            "<li><b>Images you upload</b> and the <b>words/message</b> you enter.</li>"
            "<li><b>Payment information</b>, processed by Stripe. We do not see or store your full card details.</li>"
            "<li><b>Basic technical data</b> (e.g., server logs) needed to operate the Service.</li></ul>"),
        ("How we use it", "<p>To generate and deliver your portrait, process your payment, and operate and improve the Service.</p>"),
        ("Storage and retention", f"<p>Your uploaded photo and generated files are stored only to provide your preview and download, and are automatically deleted after about {RETENTION_DAYS} days. Email us to request earlier deletion.</p>"),
        ("Sharing", "<p>We share data only with the service providers needed to run Typortrait (for example, Stripe for payments and our hosting provider). We do not sell your data. We may disclose information if required by law.</p>"
            "<p><b>Optional social-channel feature.</b> After you purchase, you can choose to generate a short shareable &ldquo;reel&rdquo; of your portrait. Before doing so you are shown two checkboxes; only if you explicitly tick the second &mdash; &ldquo;Allow Typortrait to feature this on our social channels&rdquo; &mdash; will the reel be eligible to appear on Typortrait&rsquo;s own social channels. The default is off. You can revoke this at any time by emailing us; we will remove the post within a reasonable period. See section 6.5 of the <a href=\"/terms\">Terms of Use</a> for the full license language.</p>"),
        ("Your choices", "<p>You may request access to or deletion of your data by contacting us.</p>"),
        ("Children", "<p>The Service is not directed to children under 13, and you should not upload a minor&rsquo;s photo without parental consent.</p>"),
        ("Security", "<p>We use reasonable measures to protect your data, but no method of transmission or storage is completely secure.</p>"),
        ("Changes", "<p>We may update this policy; the &ldquo;last updated&rdquo; date above will change accordingly.</p>"),
    ]
    return _policy_page("Privacy Policy", blocks)