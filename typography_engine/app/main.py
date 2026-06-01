"""FastAPI entrypoint for the isolated typography portrait engine.

Phase 1: project structure + health endpoint.
Phase 2: image upload -> silhouette / edge / landmark debug images.
Phase E: Printful-fulfilled physical prints alongside the digital download.
"""
from __future__ import annotations

import base64
import json
import secrets
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import orders as orders_db
from . import printful, products
from .config import (
    ADMIN_PASSWORD,
    CURRENCY,
    DOWNLOAD_PRICE_CENTS,
    OUTPUTS_DIR,
    PRINTFUL_API_TOKEN,
    PRIVATE_DIR,
    PUBLIC_BASE_URL,
    STATIC_DIR,
    STRIPE_SECRET_KEY,
    STRIPE_WEBHOOK_SECRET,
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
from .pipeline.warnings import WarningCollector


def _parse_words(words: Optional[str], words_json: Optional[str]) -> List[str]:
    """Accept words as JSON array string or as a comma/newline separated string."""
    if words_json:
        try:
            data = json.loads(words_json)
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    if words:
        raw = words.replace("\n", ",")
        return [w for w in (s.strip() for s in raw.split(",")) if w]
    return []

# Stroke styling for region debug output (hex only).
_REGION_COLORS = {
    "silhouette": "#000000",
    "jaw_line": "#005f73",
    "brow_line": "#bb3e03",
    "lip_line": "#d00000",
}

app = FastAPI(title="Typography Portrait Engine", version=__version__)

# Initialize the orders DB once at import time (idempotent).
orders_db.init_db()


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


app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> RedirectResponse:
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


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    words: Optional[str] = Form(None),
    words_json: Optional[str] = Form(None),
    min_font_px: Optional[float] = Form(None),
    uppercase: bool = Form(True),
    background_hex: Optional[str] = Form(None),
    foreground_hex: Optional[str] = Form(None),
    ink: str = Form("gold_noir"),
    style: str = Form("mosaic"),
    message: Optional[str] = Form(None),
    poster: bool = Form(False),
    title: Optional[str] = Form(None),
    caption: Optional[str] = Form(None),
    png_width: int = Form(2000),
) -> JSONResponse:
    """Render a typographic portrait: validated SVG + PNG from approved words."""
    warns = WarningCollector()
    img_bytes = await image.read()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "empty_upload"}, status_code=400)

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

    from .pipeline.tonal import _PALETTES, _CALLIGRAM, _GRADIENTS, build_calligram
    from .pipeline.svgbuild import validate_svg as _validate
    ink_choice = ink if (ink in _PALETTES or ink in _GRADIENTS or ink in _CALLIGRAM or ink == "photo") else "gold_noir"
    style_choice = "story" if style == "story" else "mosaic"

    modulation_png_bytes = None
    try:
        if style_choice == "story":
            # Continuous-prose calligram from the user's message (falls back to
            # the approved words if no passage was supplied).
            passage = (message or "").strip() or " ".join(word_list)
            ink_hex, bg_hex = _CALLIGRAM.get(ink_choice, _CALLIGRAM["gold_noir"])
            svg, runs, modulation_png_bytes = build_calligram(
                an, passage, cfg, warns,
                ink_hex=ink_hex, bg_hex=bg_hex, subject_only=True,
            )
            if svg:
                _validate(svg)
            from .pipeline.portrait import PortraitResult
            result = PortraitResult(svg=svg, runs=runs)
        else:
            result = build_portrait(an, word_list, cfg, warns, uppercase=uppercase, ink=ink_choice)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e), "warnings": warns.as_list()}, status_code=400)
    except Exception as e:  # noqa: BLE001
        warns.error("render", "render_failed", str(e))
        return JSONResponse({"ok": False, "error": "render_failed", "detail": str(e), "warnings": warns.as_list()}, status_code=500)

    if warns.has_errors():
        return JSONResponse({"ok": False, "error": "render_incomplete", "warnings": warns.as_list()}, status_code=422)

    # Opt-in designed-composition layer: wrap the bare portrait into a titled poster.
    svg_out = result.svg
    composed = False
    if poster:
        from .pipeline.compose import compose_poster
        svg_out = compose_poster(result.svg, ink_choice, title=title, caption=caption)
        try:
            validate_svg(svg_out)
            composed = True
        except Exception as e:  # noqa: BLE001
            warns.warn("compose", "poster_failed", f"Poster composition failed: {e}")
            svg_out = result.svg

    job_id = uuid.uuid4().hex[:12]
    # Clean (paid) files go to the PRIVATE dir; only a watermarked preview is
    # served publicly. The clean art is never web-reachable without payment.
    (PRIVATE_DIR / f"{job_id}.svg").write_text(svg_out, encoding="utf-8")
    clean_png = PRIVATE_DIR / f"{job_id}.png"
    preview_path = OUTPUTS_DIR / f"{job_id}_preview.png"
    try:
        if modulation_png_bytes:
            # Story style with per-glyph photo modulation: bytes were pre-rendered
            # in build_calligram so the photo's tone runs THROUGH each letter shape.
            # Bypass cairosvg here -- write the bytes directly.
            clean_png.write_bytes(modulation_png_bytes)
        else:
            write_png(svg_out, clean_png, output_width=max(cfg.canvas_w, int(png_width)))
        from .pipeline.watermark import add_watermark
        preview_path.write_bytes(add_watermark(clean_png.read_bytes(), url=WATERMARK_URL))
    except Exception as e:  # noqa: BLE001
        warns.warn("render", "preview_failed", f"Preview export failed: {e}")

    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "job": job_id,
            "working_size": {"w": an.img.w, "h": an.img.h},
            "face_source": an.face_source,
            "faces": len(an.faces),
            "ink": ink_choice,
            "style": style_choice,
            "composed": composed,
            "words_used": word_list,
            "text_runs": [
                {"region": r.region, "font_size": r.font_size, "kind": r.kind, "chars": len(r.text)}
                for r in result.runs
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
    """Storefront catalog (prices, shipping, sizes). Single source of truth
    so the UI never hardcodes prices."""
    return JSONResponse({
        "ok": True,
        "currency": CURRENCY,
        "products": products.public_catalog(),
        "fulfillment_configured": bool(PRINTFUL_API_TOKEN),
    })


@app.post("/checkout")
def checkout(
    job: str = Form(...),
    sku: str = Form("digital"),
    size: Optional[str] = Form(None),
    fmt: str = Form("png"),
) -> JSONResponse:
    """Create a Stripe Checkout session for either a digital download or a
    physical print. The product (`sku`) determines which line items appear
    and whether a shipping address is collected."""
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": False, "error": "payments_unconfigured"}, status_code=503)

    product = products.get(sku)
    if not product:
        return JSONResponse({"ok": False, "error": "unknown_product"}, status_code=400)

    # The clean PNG (and SVG) must exist before we let anyone pay for them.
    if not (PRIVATE_DIR / f"{job}.png").exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)

    # For sized products, require a valid size.
    variant_id: Optional[int] = None
    if product.physical:
        if not PRINTFUL_API_TOKEN:
            return JSONResponse(
                {"ok": False, "error": "fulfillment_unconfigured"}, status_code=503,
            )
        variant_id = products.resolve_variant_id(product, size)
        if variant_id is None:
            return JSONResponse(
                {"ok": False, "error": "missing_or_invalid_size",
                 "sizes": list(product.size_variants.keys()) if product.size_variants else []},
                status_code=400,
            )

    import stripe
    stripe.api_key = STRIPE_SECRET_KEY

    order_id = uuid.uuid4().hex[:16]
    ext = "svg" if fmt == "svg" else "png"

    # Build line items: the product itself plus, for physical, a flat shipping line.
    label_size = f" — {size}" if size else ""
    line_items: List[dict] = [{
        "quantity": 1,
        "price_data": {
            "currency": CURRENCY,
            "unit_amount": product.price_cents,
            "product_data": {"name": f"Typortrait — {product.name}{label_size}"},
        },
    }]
    if product.physical and product.shipping_cents > 0:
        line_items.append({
            "quantity": 1,
            "price_data": {
                "currency": CURRENCY,
                "unit_amount": product.shipping_cents,
                "product_data": {"name": "Shipping (USA)"},
            },
        })

    session_kwargs: dict = {
        "mode": "payment",
        "line_items": line_items,
        "metadata": {
            "job": job, "sku": sku, "size": size or "", "order_id": order_id, "fmt": ext,
        },
        "cancel_url": f"{PUBLIC_BASE_URL}/static/index.html?canceled=1",
    }
    if product.physical:
        # Stripe collects the shipping address for us; webhook reads it.
        session_kwargs["shipping_address_collection"] = {"allowed_countries": ["US"]}
        session_kwargs["phone_number_collection"] = {"enabled": True}
        session_kwargs["success_url"] = (
            f"{PUBLIC_BASE_URL}/order/{order_id}?session_id={{CHECKOUT_SESSION_ID}}"
        )
    else:
        # Digital path keeps the existing /download flow.
        session_kwargs["success_url"] = (
            f"{PUBLIC_BASE_URL}/download?job={job}&fmt={ext}&session_id={{CHECKOUT_SESSION_ID}}"
        )

    try:
        session = stripe.checkout.Session.create(**session_kwargs)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "stripe_error", "detail": str(e)}, status_code=502)

    # Persist a pending order so the webhook has something to update.
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
        )
    except Exception as e:  # noqa: BLE001
        # If we can't persist, the order can still complete via Stripe but
        # the webhook won't find anything to fulfill. Surface the error.
        return JSONResponse({"ok": False, "error": "order_persist_failed", "detail": str(e)},
                            status_code=500)

    return JSONResponse({"ok": True, "url": session.url, "order_id": order_id})


@app.get("/download")
def download(job: str, session_id: str, fmt: str = "png"):
    """Serve the clean file only after verifying the Stripe payment for `job`."""
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": False, "error": "payments_unconfigured"}, status_code=503)
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        sess = _stripe_to_dict(stripe.checkout.Session.retrieve(session_id))
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "bad_session"}, status_code=400)
    if sess.get("payment_status") != "paid" or (sess.get("metadata") or {}).get("job") != job:
        return JSONResponse({"ok": False, "error": "not_paid"}, status_code=402)
    ext = "svg" if fmt == "svg" else "png"
    path = PRIVATE_DIR / f"{job}.{ext}"
    if not path.exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    media = "image/svg+xml" if ext == "svg" else "image/png"
    return FileResponse(str(path), media_type=media, filename=f"typortrait-{job}.{ext}")


# --- Print fulfillment (Phase E) -------------------------------------------

@app.get("/printful-fetch/{job}")
def printful_fetch(job: str, exp: int, sig: str):
    """One-time signed URL Printful fetches the clean PNG from.

    The clean file lives in PRIVATE_DIR (paywalled). For physical orders we
    need Printful to download it; we expose it via this signed URL instead
    of making PRIVATE_DIR publicly mounted."""
    if not printful.verify_signed_url(job, exp, sig):
        raise HTTPException(status_code=403, detail="invalid_or_expired_signature")
    path = PRIVATE_DIR / f"{job}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown_job")
    return FileResponse(str(path), media_type="image/png")


def _fulfill_with_printful(order_id: str, recipient: dict) -> None:
    """Submit a paid physical order to Printful. Idempotent: noop if already
    fulfilling. Caller has just transitioned the row to 'paid'."""
    o = orders_db.get(order_id)
    if not o:
        return
    if o["status"] != "paid" or not o["variant_id"]:
        return
    try:
        signed = printful.signed_print_url(o["job_id"])
        placement = "front" if o["sku"].startswith("tshirt") else "default"
        res = printful.create_order(
            recipient=recipient,
            variant_id=o["variant_id"],
            print_file_url=signed,
            external_id=order_id,
            retail_price_cents=o["price_cents"],
            confirm=True,
            placement=placement,
        )
        pf_id = res.get("id") if isinstance(res, dict) else None
        orders_db.mark_fulfilling(order_id=order_id, printful_order_id=int(pf_id or 0), raw=res)
    except Exception as e:  # noqa: BLE001
        orders_db.mark_error(order_id=order_id, error_message=str(e))


def _recipient_from_session(sess: dict) -> Optional[dict]:
    """Map a Stripe Checkout Session's shipping_details to Printful's
    recipient schema. Returns None for digital orders (no address)."""
    ship = sess.get("shipping_details") or sess.get("shipping") or {}
    addr = (ship.get("address") or {}) if isinstance(ship, dict) else {}
    if not addr:
        return None
    customer = sess.get("customer_details") or {}
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


@app.post("/webhook/stripe")
async def webhook_stripe(request: Request, stripe_signature: Optional[str] = Header(None)):
    """Stripe → us. On checkout.session.completed for a physical order,
    submit the order to Printful. Verified via STRIPE_WEBHOOK_SECRET."""
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
        # No secret configured -> trust the body (dev only). Don't run in prod.
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
    if transitioned and recipient:
        _fulfill_with_printful(transitioned["id"], recipient)
    return JSONResponse({"ok": True})


@app.get("/order/{order_id}", response_class=HTMLResponse)
def order_status(order_id: str, session_id: Optional[str] = None):
    """Customer-facing order status page."""
    o = orders_db.get(order_id)
    if not o:
        raise HTTPException(status_code=404, detail="unknown_order")

    # If we arrived from Stripe success and the webhook hasn't fired yet,
    # poll Stripe directly so the customer sees a confirmed state instead
    # of "pending payment". (Webhooks are async and can lag a few seconds.)
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
    if o["sku"] == "digital" and o["status"] in ("paid", "fulfilling"):
        # The /download path still requires a Stripe session id; if the
        # caller arrived here it likely came from /order/{id}?session_id=...
        if session_id:
            download_html = (
                f'<p><a class="btn" href="/download?job={o["job_id"]}&fmt=png'
                f'&session_id={session_id}">Download your PNG</a></p>'
            )

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Typortrait — Order {order_id}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:#faf9f7;color:#16203a;margin:0;padding:24px;display:flex;justify-content:center}}
  .card{{background:#fff;border:1px solid #ece9e3;border-radius:20px;
        box-shadow:0 10px 40px rgba(20,30,60,.10);padding:28px;max-width:520px;width:100%}}
  h1{{font-family:Georgia,serif;color:#0d1b3a;margin:0 0 4px}}
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
  <p class="muted" style="margin-top:24px">
    Bookmark this page to check on your order, or reply to your receipt email
    if anything looks off.
  </p>
  <p><a href="/static/index.html">Make another Typortrait →</a></p>
</div></body></html>"""
    return HTMLResponse(body)


# --- Admin -----------------------------------------------------------------

def _check_admin(authorization: Optional[str]) -> bool:
    if not ADMIN_PASSWORD:
        return False
    if not authorization or not authorization.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(authorization.split(" ", 1)[1]).decode("utf-8")
        _, _, pw = decoded.partition(":")
        return secrets.compare_digest(pw, ADMIN_PASSWORD)
    except Exception:  # noqa: BLE001
        return False


@app.get("/admin/orders", response_class=HTMLResponse)
def admin_orders(authorization: Optional[str] = Header(None)):
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="admin_unconfigured")
    if not _check_admin(authorization):
        return HTMLResponse(
            "Unauthorized", status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Typortrait admin"'},
        )
    rows = orders_db.list_recent(200)
    rows_html = []
    for o in rows:
        product = products.get(o["sku"])
        name = product.name if product else o["sku"]
        pf_id = o.get("printful_order_id") or "—"
        track = o.get("tracking_url")
        track_html = f'<a href="{track}" target="_blank">track</a>' if track else "—"
        err = (o.get("error_message") or "").replace("<", "&lt;")
        rows_html.append(
            f"<tr><td><a href='/order/{o['id']}'>{o['id']}</a></td>"
            f"<td>{name}{(' / ' + o['size']) if o.get('size') else ''}</td>"
            f"<td>${(o['price_cents'] + o['shipping_cents']) / 100:.2f}</td>"
            f"<td>{o['status']}</td>"
            f"<td>{o.get('customer_email') or '—'}</td>"
            f"<td>{pf_id}</td><td>{track_html}</td>"
            f"<td>{err}</td></tr>"
        )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Orders — admin</title>
<style>
  body{{font-family:-apple-system,sans-serif;margin:24px;background:#faf9f7;color:#16203a}}
  table{{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
  th,td{{border-bottom:1px solid #ece9e3;padding:10px 12px;text-align:left;font-size:14px;vertical-align:top}}
  th{{background:#f3f5fa;color:#6b7280;font-size:12px;letter-spacing:.06em;text-transform:uppercase}}
  td:last-child{{color:#7a2e2e;max-width:240px;word-break:break-word}}
</style></head><body>
<h1>Orders ({len(rows)})</h1>
<table>
<tr><th>Order</th><th>Item</th><th>Total</th><th>Status</th>
    <th>Email</th><th>Printful</th><th>Track</th><th>Error</th></tr>
{''.join(rows_html) or '<tr><td colspan=8>No orders yet.</td></tr>'}
</table></body></html>"""
    return HTMLResponse(body)
