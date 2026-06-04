"""Central configuration for the typography portrait engine.

All colors MUST be hex strings (no rgba()/hsl()) so emitted SVG validates and
renders identically across SVG consumers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# MediaPipe face landmark model (optional capability).
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

SELFIE_SEGMENTER_MODEL = MODELS_DIR / "selfie_segmenter.tflite"
SELFIE_SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)


@dataclass
class RenderConfig:
    """Tunable parameters for a single render. Hex colors only."""

    canvas_w: int = 1000
    canvas_h: int = 1250
    background_hex: str = "#ffffff"
    foreground_hex: str = "#000000"

    # Readability guards.
    min_font_px: float = 20.0          # never emit text below this
    max_font_px: float = 120.0
    primary_font_family: str = "Arial, Helvetica, sans-serif"
    font_weight: str = "bold"
    letter_spacing_px: float = 0.0

    # Image preprocessing.
    work_max_dim: int = 1024           # longest side after resize for analysis

    # Silhouette / edge tuning.
    canny_low: int = 60
    canny_high: int = 160
    min_contour_points: int = 8

    def validate(self) -> None:
        for name in ("background_hex", "foreground_hex"):
            val = getattr(self, name)
            if not (isinstance(val, str) and val.startswith("#") and len(val) in (4, 7)):
                raise ValueError(f"{name} must be a hex color like #000000, got {val!r}")
        if self.min_font_px <= 0:
            raise ValueError("min_font_px must be positive")
        if self.min_font_px > self.max_font_px:
            raise ValueError("min_font_px cannot exceed max_font_px")


def env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


PORT = env_int("TYPO_PORT", 8077)

# --- Analytics (Umami Cloud) ------------------------------------------------
# Privacy-first, cookieless funnel analytics. The website ID is NOT secret (it
# is exposed in the browser tracking script); it lives here only so the
# server-side `purchase` conversion event (fired from the Stripe webhook, which
# ad-blockers can't suppress) can attribute to the same site. With no
# UMAMI_WEBSITE_ID set, the server-side event is silently skipped.
UMAMI_WEBSITE_ID = os.environ.get("UMAMI_WEBSITE_ID", "")
UMAMI_HOST = os.environ.get("UMAMI_HOST", "https://cloud.umami.is").rstrip("/")
# Hostname Umami records the event under (should match your tracked domain).
UMAMI_HOSTNAME = os.environ.get("UMAMI_HOSTNAME", "app.typortrait.com")

# --- Freemium / payments ---------------------------------------------------
# Clean (unwatermarked) renders are stored here and only served after payment;
# this directory is NOT mounted as static.
PRIVATE_DIR = BASE_DIR / "private"
PRIVATE_DIR.mkdir(exist_ok=True)

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
DOWNLOAD_PRICE_CENTS = env_int("TYPO_PRICE_CENTS", 1499)     # e.g. 1499 = $14.99
CURRENCY = os.environ.get("TYPO_CURRENCY", "usd")
# Public base URL of the app, used to build Stripe success/cancel redirects.
PUBLIC_BASE_URL = os.environ.get("TYPO_PUBLIC_URL", f"http://127.0.0.1:{PORT}")
WATERMARK_URL = os.environ.get("TYPO_WATERMARK_URL", "https://typortrait.com")

# --- Print-on-demand (Printful) --------------------------------------------
# All optional: with no PRINTFUL_API_TOKEN the catalog still renders and the
# digital download keeps working, but physical checkout is gated off. The
# Stripe webhook drives async fulfillment for physical orders (digital stays
# on the existing synchronous-verify path, so the live revenue path is intact).
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
PRINTFUL_API_TOKEN = os.environ.get("PRINTFUL_API_TOKEN", "")
PRINTFUL_STORE_ID = os.environ.get("PRINTFUL_STORE_ID", "")
PRINTFUL_API_BASE = os.environ.get("PRINTFUL_API_BASE", "https://api.printful.com")
# Secret used to sign the time-limited /printful-fetch art URLs Printful pulls
# from. Falls back to the Stripe webhook secret so a single secret can cover
# both; the final fallback is dev-only and must not be used in production.
PRINT_URL_SECRET = (os.environ.get("PRINT_URL_SECRET", "")
                    or STRIPE_WEBHOOK_SECRET or "dev-only-not-secure")
# Whether a paid physical order is auto-confirmed for fulfillment at Printful.
# True (default, production): the order is charged + printed + shipped.
# Set PRINTFUL_CONFIRM=false while testing so orders land as unconfirmed DRAFTS
# you can inspect (right product? right art?) and delete with no charge/print.
PRINTFUL_CONFIRM = (os.environ.get("PRINTFUL_CONFIRM", "true").strip().lower()
                    not in ("0", "false", "no", "off"))
# Order persistence (SQLite). Volume-mounted in docker-compose so it survives
# container rebuilds.
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
ORDERS_DB = DATA_DIR / "orders.db"
# The /admin/orders dashboard reuses the existing reel-admin auth
# (TYPO_ADMIN_PASSWORD + TYPO_SECRET_KEY, see app/admin.py), so there is no
# separate orders-admin password to configure.

# Rasterization sizes. The on-screen preview stays web-light; the paid PNG is
# rendered at print resolution (lazily, at download time) so the one expensive
# big raster runs once per sale, not on every preview/swatch.
PREVIEW_PNG_WIDTH = env_int("TYPO_PREVIEW_PX", 1400)
DOWNLOAD_PNG_WIDTH = env_int("TYPO_DOWNLOAD_PX", 2600)   # ~8.7in at 300dpi; fast to compose
# Uploaded photos, previews and generated files are auto-deleted after this many
# days so the Privacy Policy's retention statement stays accurate.
RETENTION_DAYS = env_int("TYPO_RETENTION_DAYS", 30)
