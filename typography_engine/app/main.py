"""FastAPI entrypoint for the isolated typography portrait engine.

Phase 1: project structure + health endpoint.
Phase 2: image upload -> silhouette / edge / landmark debug images.
Later phases add /render.
"""
from __future__ import annotations

import json
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import (
    CURRENCY,
    DOWNLOAD_PNG_WIDTH,
    DOWNLOAD_PRICE_CENTS,
    OUTPUTS_DIR,
    PREVIEW_PNG_WIDTH,
    PRIVATE_DIR,
    PUBLIC_BASE_URL,
    STATIC_DIR,
    STRIPE_SECRET_KEY,
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
    ink: str = Form("navy"),
    style: str = Form("mosaic"),
    message: Optional[str] = Form(None),
    poster: bool = Form(False),
    title: Optional[str] = Form(None),
    caption: Optional[str] = Form(None),
    png_width: int = Form(2000),
    render_w: int = Form(2600),
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
    ink_choice = ink if (ink in _PALETTES or ink in _GRADIENTS or ink == "photo") else "navy"
    style_choice = "story" if style == "story" else "mosaic"
    # Internal working resolution. Smaller = faster (the tone pipeline scales with
    # pixel count): the front-end requests a low render_w for fast swatch
    # thumbnails. Clamped so it can't be abused or degrade the paid art.
    render_w_eff = max(700, min(3000, int(render_w)))

    try:
        if style_choice == "story":
            # Continuous-prose calligram from the user's message (falls back to
            # the approved words if no passage was supplied).
            passage = (message or "").strip() or " ".join(word_list)
            ink_hex, bg_hex = _CALLIGRAM.get(ink_choice, _CALLIGRAM["navy"])
            svg, runs = build_calligram(an, passage, cfg, warns, render_w=render_w_eff, ink_hex=ink_hex, bg_hex=bg_hex)
            if svg:
                _validate(svg)
            from .pipeline.portrait import PortraitResult
            result = PortraitResult(svg=svg, runs=runs)
        else:
            result = build_portrait(an, word_list, cfg, warns, uppercase=uppercase, ink=ink_choice, render_w=render_w_eff)
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
    # The clean SVG (vector, resolution-independent) is the master and the only
    # thing persisted at render time. The on-screen preview is a web-light
    # watermarked raster; the high-resolution clean PNG is rendered lazily at
    # download time (after payment), so the one expensive big raster runs once
    # per sale -- never on previews or swatch thumbnails. Clean files live in the
    # PRIVATE dir and are never web-reachable without payment.
    (PRIVATE_DIR / f"{job_id}.svg").write_text(svg_out, encoding="utf-8")
    preview_path = OUTPUTS_DIR / f"{job_id}_preview.png"
    try:
        from .pipeline.raster import svg_to_png_bytes
        from .pipeline.watermark import add_watermark
        preview_w = min(int(png_width), PREVIEW_PNG_WIDTH)
        clean_bytes = svg_to_png_bytes(svg_out, output_width=max(cfg.canvas_w, preview_w))
        preview_path.write_bytes(add_watermark(clean_bytes, url=WATERMARK_URL))
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


@app.post("/checkout")
def checkout(job: str = Form(...), fmt: str = Form("png")) -> JSONResponse:
    """Create a Stripe Checkout session to unlock the clean download of `job`."""
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": False, "error": "payments_unconfigured"}, status_code=503)
    ext = "svg" if fmt == "svg" else "png"
    # The SVG master is written at render time; the PNG is derived from it at
    # download, so validate the job against the SVG (not the per-format file).
    if not (PRIVATE_DIR / f"{job}.svg").exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{
                "quantity": 1,
                "price_data": {
                    "currency": CURRENCY,
                    "unit_amount": DOWNLOAD_PRICE_CENTS,
                    "product_data": {"name": "Typortrait — high-resolution download"},
                },
            }],
            metadata={"job": job, "fmt": ext},
            success_url=f"{PUBLIC_BASE_URL}/download?job={job}&fmt={ext}&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{PUBLIC_BASE_URL}/static/index.html?canceled=1",
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "stripe_error", "detail": str(e)}, status_code=502)
    return JSONResponse({"ok": True, "url": session.url})


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
    if sess.get("payment_status") != "paid" or (sess.get("metadata") or {}).get("job") != job:
        return JSONResponse({"ok": False, "error": "not_paid"}, status_code=402)
    ext = "svg" if fmt == "svg" else "png"
    svg_path = PRIVATE_DIR / f"{job}.svg"
    if not svg_path.exists():
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    path = PRIVATE_DIR / f"{job}.{ext}"
    if ext == "png" and not path.exists():
        # Print-resolution raster, produced once per paid job from the master SVG.
        try:
            write_png(svg_path.read_text(encoding="utf-8"), path, output_width=DOWNLOAD_PNG_WIDTH)
        except Exception:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": "export_failed"}, status_code=500)
    media = "image/svg+xml" if ext == "svg" else "image/png"
    return FileResponse(str(path), media_type=media, filename=f"typortrait-{job}.{ext}")
