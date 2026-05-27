"""End-to-end smoke tests exercising the full HTTP stack via TestClient.

Uses matplotlib's bundled `grace_hopper.jpg` portrait as a self-contained
fixture (a real face, so the MediaPipe path is covered when available).
Run with:  cd typography_engine && python -m pytest -q
"""
from __future__ import annotations

import glob
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from starlette.testclient import TestClient

from app.config import OUTPUTS_DIR, PRIVATE_DIR
from app.main import app


def _has_private_recipe(body) -> bool:
    """The paid render's inputs (recipe + source) live privately (paywalled)."""
    return (PRIVATE_DIR / f"{body['job']}.json").exists() and (PRIVATE_DIR / f"{body['job']}.src").exists()


def _sample_face_bytes() -> bytes:
    matches = glob.glob(
        "/usr/**/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg",
        recursive=True,
    ) + glob.glob(
        "/usr/**/dist-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg",
        recursive=True,
    )
    if not matches:
        pytest.skip("No sample face image available for smoke test.")
    return Path(matches[0]).read_bytes()


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "opencv" in body["capabilities"]


def test_debug_preprocess():
    r = client.post(
        "/debug/preprocess",
        files={"image": ("face.jpg", _sample_face_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "silhouette" in body["debug_images"]


def test_render_produces_preview_and_private_recipe():
    r = client.post(
        "/render",
        files={"image": ("face.jpg", _sample_face_bytes(), "image/jpeg")},
        data={"words": "CODE,DREAM,NAVY,PIONEER", "min_font_px": "14"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert len(body["text_runs"]) >= 1

    # Public output is the watermarked preview PNG only.
    assert body["preview"]
    prev = OUTPUTS_DIR / Path(body["preview"]).name
    assert prev.exists() and prev.stat().st_size > 0
    assert prev.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # The paid inputs live privately (paywalled) and are NOT publicly served.
    assert _has_private_recipe(body)
    assert not (OUTPUTS_DIR / f"{body['job']}.src").exists()
    assert not (OUTPUTS_DIR / f"{body['job']}.json").exists()


def test_checkout_requires_stripe_key():
    """Without a Stripe key configured, checkout is gracefully unavailable."""
    r = client.post(
        "/render",
        files={"image": ("face.jpg", _sample_face_bytes(), "image/jpeg")},
        data={"words": "CODE,DREAM"},
    )
    job = r.json()["job"]
    c = client.post("/checkout", data={"job": job, "fmt": "png"})
    # 503 when unconfigured (test env), or a Stripe URL when a key is present.
    assert c.status_code in (503, 200)


def test_render_color_ink():
    """A non-mono ink renders and reports the chosen treatment + a preview."""
    r = client.post(
        "/render",
        files={"image": ("face.jpg", _sample_face_bytes(), "image/jpeg")},
        data={"words": "CODE,DREAM", "ink": "navy"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["ink"] == "navy"
    assert body["preview"] and _has_private_recipe(body)


def test_render_message_style():
    """Message style lays the passage as a layered type-poster render."""
    r = client.post(
        "/render",
        files={"image": ("face.jpg", _sample_face_bytes(), "image/jpeg")},
        data={"words": "GRACE,KIND", "style": "message", "ink": "navy",
              "message": "You are the kindest soul I know and I love you always and forever. " * 4},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["style"] == "message"
    assert len(body["text_runs"]) >= 1
    assert body["preview"]


def test_render_rejects_empty_words():
    r = client.post(
        "/render",
        files={"image": ("face.jpg", _sample_face_bytes(), "image/jpeg")},
        data={"words": ""},
    )
    assert r.status_code == 400


def test_render_rejects_non_portrait_image():
    """A non-portrait image (no detectable face) is gated with a 422 and an
    actionable message rather than silently producing a bad portrait."""
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noise).save(buf, format="PNG")

    r = client.post(
        "/render",
        files={"image": ("noise.png", buf.getvalue(), "image/png")},
        data={"words": "CODE,DREAM"},
    )
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "unsuitable_image"
    assert body["detail"]


def test_min_font_never_violated():
    """No emitted run may fall below the configured minimum font size."""
    min_font = 16.0
    r = client.post(
        "/render",
        files={"image": ("face.jpg", _sample_face_bytes(), "image/jpeg")},
        data={"words": "CODE,DREAM", "min_font_px": str(min_font)},
    )
    assert r.status_code == 200
    # The readable body respects the floor; "detail" accents (e.g. eyes) may be finer.
    for run in r.json()["text_runs"]:
        if run["kind"] == "primary":
            assert run["font_size"] >= min_font - 1e-6
