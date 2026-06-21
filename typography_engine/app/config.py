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
    # Ceiling for the largest (body) tier. Raised so the "Giant" word-size option
    # can produce very-large body letters; the per-region tiers (body->mid->face
    # ->eye) still scale down from min_font_px, so facial features stay finer.
    # Default renders (min_font_px=20 -> body ~44px) are well under this, so the
    # standard look is unchanged; only the large end benefits.
    max_font_px: float = 280.0
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

# Largest upload we accept. Normal phone photos are a few MB; this caps the
# pathological (40MP+ files) that would slow a render or exhaust memory on the
# single worker. The frontend rejects oversize before upload; the server
# enforces the same limit as defense-in-depth.
MAX_UPLOAD_BYTES = env_int("TYPO_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)   # 25 MB

# A file can be small on disk (heavily-compressed JPEG) yet enormous in pixels.
# Decoding a 200+ megapixel photo costs hundreds of MB of RAM and trips Pillow's
# decompression-bomb guard, whose raw error used to leak to users. We cap by
# pixel count and reject oversize photos early with a friendly message. 100 MP
# clears every mainstream phone (incl. 48/64 MP) and most high-res modes; only
# 200 MP shots and big scans are asked to be resized.
MAX_IMAGE_MP = env_int("TYPO_MAX_IMAGE_MP", 100)
MAX_IMAGE_PIXELS = MAX_IMAGE_MP * 1_000_000

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

# --- Gather (crowd-sourced words) — ADDITIVE, FLAG-GATED, DEFAULT OFF ----------
# A shareable link where a whole circle adds a word/memory and the portrait is
# woven from everyone's words. Fully self-contained (its own DB + routes + pages);
# the flag keeps every gather route inert in prod until deliberately enabled, so
# merging this branch can never affect the existing studio/checkout paths.
GATHER_ENABLED = (os.environ.get("TYPO_GATHER_ENABLED", "").strip().lower()
                  in ("1", "true", "yes", "on"))
GATHER_DIR = DATA_DIR / "gather"
GATHER_DB = GATHER_DIR / "gather.db"
# Throttle live re-renders so a busy gather can't hammer the single render slot:
# re-render at most this often, OR once this many new words have landed.
GATHER_RENDER_INTERVAL = env_int("TYPO_GATHER_RENDER_INTERVAL", 15)   # seconds
GATHER_RENDER_EVERY = env_int("TYPO_GATHER_RENDER_EVERY", 4)          # new words
GATHER_MAX_WORDS = env_int("TYPO_GATHER_MAX_WORDS", 64)               # distinct words carried
GATHER_WORD_MAXLEN = env_int("TYPO_GATHER_WORD_MAXLEN", 40)           # per submission
# The /admin/orders dashboard reuses the existing reel-admin auth
# (TYPO_ADMIN_PASSWORD + TYPO_SECRET_KEY, see app/admin.py), so there is no
# separate orders-admin password to configure.

# Rasterization sizes. The on-screen preview stays web-light; the paid PNG is
# rendered at print resolution (lazily, at download time) so the one expensive
# big raster runs once per sale, not on every preview/swatch.
PREVIEW_PNG_WIDTH = env_int("TYPO_PREVIEW_PX", 1400)
# Print/download long-edge width in px. 3600 clears Printful's 150-PPI floor with
# margin on every catalogued size: 225 PPI on 16x20, 200 PPI on 18x24 (the widest
# product) -- crisp paper prints, ~35s to compose (warmed in the background at
# fulfilment, behind a spinner for the digital download). Bump to 4320 for 240 PPI.
DOWNLOAD_PNG_WIDTH = env_int("TYPO_DOWNLOAD_PX", 3600)
# Uploaded photos, previews and generated files are auto-deleted after this many
# days so the Privacy Policy's retention statement stays accurate.
RETENTION_DAYS = env_int("TYPO_RETENTION_DAYS", 30)

# Consent records (proof of consent + the age/guardian attestation) are kept FAR
# longer than the photos they relate to: they're the evidence needed to
# demonstrate compliance, and they contain NO biometric data (the photo is
# deleted on the schedule above). Default ~7 years, covering BIPA's 5-year
# limitations period with margin. Confirm the period with counsel.
CONSENT_RETENTION_DAYS = env_int("TYPO_CONSENT_RETENTION_DAYS", 2555)

# --- Privacy / biometric compliance (GDPR special-category, BIPA, CCPA) ------
# We analyse facial geometry from the uploaded photo to place the type; that is
# "biometric" under BIPA and "special-category data" under GDPR. We store NO
# faceprint/template (the mesh is computed transiently and discarded), and the
# source photo auto-deletes per RETENTION_DAYS -- but we still need notice +
# consent before processing, a published retention policy, and a geo-gate.

# Versioned so each stored consent record points at the exact notice wording the
# user saw. Bump when the upload-time biometric notice copy materially changes.
# v2 adds the age (13+) + parent/guardian attestation folded into the same box.
BIOMETRIC_CONSENT_VERSION = os.environ.get("TYPO_BIO_CONSENT_VERSION", "bio-v3-2026-06")


def _env_region_set(key: str, default: str) -> set:
    raw = os.environ.get(key, default)
    return {p.strip().upper() for p in (raw or "").split(",") if p.strip()}


# BIPA geo-gate: refuse face processing for visitors in these regions. Tokens may
# be a country ("US"), a region/state ("IL"), or "COUNTRY-REGION" ("US-IL").
# Default blocks Illinois (BIPA's private right of action is the acute risk).
# Set TYPO_BLOCKED_REGIONS="" to disable. The visitor's location comes from the
# headers below, which the upstream proxy (nginx GeoIP2 / Cloudflare) must set;
# with NO geo signal the gate fails OPEN (allows) so the site never hard-breaks.
# See COMPLIANCE.md to wire a geo source and make the block effective.
BLOCKED_REGIONS = _env_region_set("TYPO_BLOCKED_REGIONS", "US-IL")
GEO_COUNTRY_HEADER = os.environ.get("TYPO_GEO_COUNTRY_HEADER", "X-Geo-Country")
GEO_REGION_HEADER = os.environ.get("TYPO_GEO_REGION_HEADER", "X-Geo-Region")

# The /debug/* endpoints run face detection WITHOUT the consent/geo gate, so they
# are disabled unless explicitly enabled (local development only). Never enable
# in production -- they would be an ungated biometric-processing bypass.
ENABLE_DEBUG_ENDPOINTS = os.environ.get("TYPO_ENABLE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

# --- Render concurrency -----------------------------------------------------
# Max heavy renders (MediaPipe + OpenCV + rasterize) allowed to run at once. The
# render is offloaded off the event loop into worker threads; this caps how many
# run concurrently so a burst queues instead of thrashing CPU/RAM on a single
# box. Unset/blank -> (cpu_count - 1), min 1. NOTE: threads do NOT give linear
# CPU scaling here -- MediaPipe inference is likely GIL-bound -- so this buys
# responsiveness + partial overlap; true parallelism needs multiple worker
# PROCESSES (raise uvicorn --workers / a render queue). See DEPLOY.md.
def _render_concurrency() -> int:
    raw = (os.environ.get("TYPO_RENDER_CONCURRENCY", "") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 2) - 1)


RENDER_CONCURRENCY = _render_concurrency()

# --- Content moderation -----------------------------------------------------
# The user's words/message BECOME the portrait, so we moderate that text (hate,
# sexual, harassment, etc.) via OpenAI's free moderation endpoint. With no key
# set, text moderation is OFF and the Terms + attestation are the only backstop.
# If the moderator errors, we FAIL OPEN (allow) so an outage can't take the paid
# product down. Uploaded-IMAGE NSFW is deliberately NOT auto-detected (Terms +
# human review + a report channel); CSAM is handled via policy + NCMEC reporting,
# not a classifier. See COMPLIANCE.md.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODERATION_MODEL = os.environ.get("TYPO_MODERATION_MODEL", "omni-moderation-latest")

# Word suggestion for the memorial "Share their story" step (paste a tribute ->
# suggested words). Reuses OPENAI_API_KEY. NOTE: unlike moderation, chat
# completions ARE billed -- tiny (gpt-4o-mini, a few hundred tokens, well under a
# cent per call), but not free; the usage cap on the key bounds it.
SUGGEST_MODEL = os.environ.get("TYPO_SUGGEST_MODEL", "gpt-4o-mini")
