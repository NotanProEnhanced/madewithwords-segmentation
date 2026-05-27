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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
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
    RETENTION_DAYS,
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
    remove_bg: bool = Form(True),
    light: bool = Form(False),
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

    from .pipeline.tonal import _PALETTES, _GRADIENTS, render_layered_png
    ink_choice = ink if (ink in _PALETTES or ink in _GRADIENTS or ink == "photo") else "navy"
    # Two user-facing styles: "message" = poster rows; anything else = "words"
    # (the scattered mosaic). Both use the layered photo-through-text renderer.
    style_choice = "message" if style in ("message", "poster", "story") else "words"
    render_w_eff = max(700, min(3000, int(render_w)))
    is_thumb = int(png_width) < 600         # small png_width => a swatch chip render
    if style_choice == "words":
        text = " ".join(word_list) or (message or "").strip()
    else:
        text = (message or "").strip() or " ".join(word_list)

    try:
        preview_w = min(int(png_width), PREVIEW_PNG_WIDTH)
        png_bytes, runs, ground_hex, mask_svg = render_layered_png(
            an, text, style_choice, cfg, warns,
            ink=ink_choice, remove_bg=remove_bg, light=light,
            out_width=max(320, preview_w), render_w=render_w_eff)
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
        }), encoding="utf-8")

    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "job": job_id,
            "faces": len(an.faces),
            "ink": ink_choice,
            "style": style_choice,
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


@app.post("/checkout")
def checkout(job: str = Form(...), fmt: str = Form("png")) -> JSONResponse:
    """Create a Stripe Checkout session to unlock the clean download of `job`."""
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": False, "error": "payments_unconfigured"}, status_code=503)
    # The job's inputs (recipe) are persisted at render time; the high-res PNG is
    # composed from them at download.
    if not (PRIVATE_DIR / f"{job}.json").exists():
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
            metadata={"job": job},
            success_url=f"{PUBLIC_BASE_URL}/success?job={job}&session_id={{CHECKOUT_SESSION_ID}}",
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
    path = PRIVATE_DIR / f"{job}.png"
    if not path.exists():
        # Compose the print-resolution layered PNG once. The layout build (the
        # costly part) is reused from the stored mask; only the photo is
        # re-derived and composited at print resolution.
        try:
            import json as _json
            from .pipeline.tonal import compose_layered, render_layered_png
            r = _json.loads(recipe_path.read_text(encoding="utf-8"))
            warns2 = WarningCollector()
            an = analyze_image(src_path.read_bytes(), RenderConfig(), warns2)
            mask_path = PRIVATE_DIR / f"{job}.mask.svg"
            if mask_path.exists():
                png_bytes = compose_layered(
                    mask_path.read_text(encoding="utf-8"), an,
                    r.get("ink", "navy"), bool(r.get("remove_bg", True)), DOWNLOAD_PNG_WIDTH,
                    light=bool(r.get("light", False)))
            else:  # older jobs without a stored mask: full recompose
                png_bytes, _, _, _ = render_layered_png(
                    an, r["text"], r.get("style", "words"), RenderConfig(), warns2,
                    ink=r.get("ink", "navy"), remove_bg=bool(r.get("remove_bg", True)),
                    light=bool(r.get("light", False)), out_width=DOWNLOAD_PNG_WIDTH, render_w=2600)
            if not png_bytes:
                return JSONResponse({"ok": False, "error": "export_failed"}, status_code=500)
            path.write_bytes(png_bytes)
        except Exception:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": "export_failed"}, status_code=500)
    return FileResponse(str(path), media_type="image/png", filename=f"typortrait-{job}.png")


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
        # Fetch the high-res file in the background (it's composed on first
        # request and can take a few seconds), showing a spinner, then hand the
        # user a ready, instant download instead of a hung button.
        inner = (
            '<div class="check">&#10003;</div>'
            '<h1>Your Typortrait is ready</h1>'
            '<p class="sub" id="sub">Preparing your high-resolution file — this can take a few seconds…</p>'
            '<button class="btn" id="dl" disabled><span class="spin"></span>Preparing…</button>'
            '<button class="btn ghost" id="sh">Share</button>'
            '<p class="note">Watermark-free, print-quality — ready to print or share.</p>'
            '<a class="link" href="/static/index.html">Create another portrait</a>'
            '<script>(function(){var url=' + _json.dumps(png_url) + ';'
            'var job=' + _json.dumps(job) + ',o=location.origin;'
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
        "<title>Typortrait — your download</title><style>"
        ":root{--navy:#0d1b3a;--muted:#6b7280;--line:#ece9e3}"
        "*{box-sizing:border-box}body{margin:0;min-height:100dvh;display:flex;align-items:center;justify-content:center;"
        "background:#faf9f7;color:#16203a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;padding:24px}"
        ".card{background:#fff;border:1px solid var(--line);border-radius:20px;box-shadow:0 10px 40px rgba(20,30,60,.10);"
        "max-width:440px;width:100%;padding:34px 28px;text-align:center}"
        ".check{width:54px;height:54px;border-radius:50%;background:#0d1b3a;color:#fff;font-size:28px;line-height:54px;margin:0 auto 14px}"
        "h1{font-family:Georgia,'Times New Roman',serif;color:var(--navy);font-size:26px;margin:6px 0 8px}"
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
        "<style>*{box-sizing:border-box}body{margin:0;min-height:100dvh;display:flex;align-items:center;"
        "justify-content:center;background:#0a0a0c;color:#f5f3ec;font-family:-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Roboto,Helvetica,Arial,sans-serif;padding:24px;text-align:center}"
        ".w{max-width:520px;width:100%}img{width:100%;border-radius:14px;box-shadow:0 16px 50px rgba(0,0,0,.5)}"
        "h1{font-family:Georgia,'Times New Roman',serif;font-size:24px;margin:22px 0 6px}"
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


_POLICY_CONTACT = "support@typortrait.com"
_POLICY_UPDATED = "May 2026"


def _policy_page(title: str, blocks) -> HTMLResponse:
    body = "".join(f"<h2>{h}</h2>{p}" for h, p in blocks)
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title} — Typortrait</title><style>"
        "body{margin:0;background:#faf9f7;color:#16203a;font-family:-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Roboto,Helvetica,Arial,sans-serif;line-height:1.6}"
        ".wrap{max-width:760px;margin:0 auto;padding:40px 22px 80px}"
        "h1{font-family:Georgia,'Times New Roman',serif;color:#0d1b3a;font-size:30px;margin:0 0 4px}"
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
        ("Sharing", "<p>We share data only with the service providers needed to run Typortrait (for example, Stripe for payments and our hosting provider). We do not sell your data. We may disclose information if required by law.</p>"),
        ("Your choices", "<p>You may request access to or deletion of your data by contacting us.</p>"),
        ("Children", "<p>The Service is not directed to children under 13, and you should not upload a minor&rsquo;s photo without parental consent.</p>"),
        ("Security", "<p>We use reasonable measures to protect your data, but no method of transmission or storage is completely secure.</p>"),
        ("Changes", "<p>We may update this policy; the &ldquo;last updated&rdquo; date above will change accordingly.</p>"),
    ]
    return _policy_page("Privacy Policy", blocks)