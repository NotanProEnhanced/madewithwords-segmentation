"""Image loading and normalization (OpenCV)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from .warnings import WarningCollector

# Optional HEIC/HEIF support (iPhone's default photo format). The pillow-heif
# wheel bundles libheif, so it's self-contained; the import is guarded so its
# absence never breaks the app -- HEIC uploads just fall through to a clear
# "couldn't read that image" error instead of crashing.
try:  # pragma: no cover - optional dependency
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:
    pass


@dataclass
class LoadedImage:
    bgr: np.ndarray          # HxWx3 uint8, working resolution
    gray: np.ndarray         # HxW uint8
    orig_w: int
    orig_h: int
    scale: float             # working_dim / original_dim (uniform)
    # TRUE source dimensions, captured BEFORE enhance_source upscales a small photo.
    # orig_w/orig_h are post-enhance and would mask a low-res upload; src_w/src_h are
    # the honest file size used by the quality gate to judge real detail. Default 0
    # (falls back to working size) for any LoadedImage built without them.
    src_w: int = 0
    src_h: int = 0

    @property
    def h(self) -> int:
        return self.bgr.shape[0]

    @property
    def w(self) -> int:
        return self.bgr.shape[1]


def decode_image(img_bytes: bytes, warns: WarningCollector) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr
    # OpenCV can't decode every format -- notably HEIC/HEIF from iPhones, and a
    # few unusual TIFF/JPEG variants. Fall back to Pillow (which handles HEIC
    # when pillow-heif is registered above) and honor EXIF orientation so the
    # photo isn't composed sideways.
    try:
        from io import BytesIO

        from PIL import Image, ImageOps

        with Image.open(BytesIO(img_bytes)) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        if rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.size:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    raise ValueError("could_not_decode_image")


def _auto_expose(bgr: np.ndarray, warns: WarningCollector) -> np.ndarray:
    """Lift dark / underexposed photos toward a usable tonal range so the portrait
    isn't murky. ADAPTIVE: acceptably-lit photos (mean luminance >= 75) are returned
    untouched, so good inputs never regress. Only clearly-underexposed photos get a
    gamma lift (stronger the darker they are) toward a ~120 target mean, then a
    gentle luminance range-stretch. The same curve is applied to all channels, so
    colour is preserved."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean = float(g.mean())
    if mean >= 75.0:
        return bgr                                   # acceptably lit -> leave it
    out = bgr.astype(np.float32)
    m = max(8.0, mean)
    gamma = float(np.clip(np.log(m / 255.0) / np.log(120.0 / 255.0), 1.0, 2.6))
    if gamma > 1.01:
        out = np.power(out / 255.0, 1.0 / gamma) * 255.0
    g2 = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = (float(x) for x in np.percentile(g2, [1.0, 99.0]))
    if hi - lo > 1.0:
        out = (out - lo) * (255.0 / (hi - lo))
    if mean < 45.0:
        warns.warn("preprocess", "dark_photo",
                   "This photo is quite dark; a brighter, well-lit photo will render sharper.")
    return np.clip(out, 0, 255).astype(np.uint8)


def _auto_levels(bgr: np.ndarray, warns: WarningCollector) -> np.ndarray:
    """Recover LOW-CONTRAST / washed-out photos that ``_auto_expose`` misses -- it gates
    on mean brightness, so a flat-but-not-dark photo (hazy, underexposed-looking, low
    dynamic range) slips past it untouched. Gate on dynamic RANGE instead: below ~95 of
    255, apply a per-channel black/white-point stretch plus a median-normalising gamma,
    so contrast, midtones AND colour saturation come back (the engine's internal luma
    stretch alone can't restore colour). Well-exposed, full-range photos are a near-no-op,
    so this is safe even when enabled. Opt-in via the TYPO_AUTOLEVELS env flag."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    p2, p50, p98 = (float(x) for x in np.percentile(g, [2.0, 50.0, 98.0]))
    if p98 - p2 >= 95.0:                              # enough contrast already -> untouched
        return bgr
    denom = max(1.0, p98 - p2)
    norm = min(0.98, max(0.02, (p50 - p2) / denom))
    gamma = float(np.clip(np.log(0.5) / np.log(norm), 0.6, 1.6))
    out = np.clip((bgr.astype(np.float32) - p2) / denom, 0.0, 1.0)
    out = np.power(out, gamma) * 255.0
    if p98 - p2 < 70.0:
        warns.warn("preprocess", "flat_photo",
                   "This photo is very low-contrast; we've enhanced it, but a crisper "
                   "photo will render sharper.")
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# Vibrance: a post-render "give it life" pass — clarity (local contrast) + a soft
# highlight glow + a saturation nudge. Applied to the FINISHED render, so the words
# are already placed (typography untouched) and the dark ground stays dark (the glow
# rolls off at low luminance). Strength ~0.65 is a tasteful lift; 0 disables.
_VIBRANCE = 0.65


def apply_vibrance(img: np.ndarray, strength: float = _VIBRANCE, bgr: bool = False) -> np.ndarray:
    """Lift a rendered image's luminosity/clarity/saturation. `bgr` selects the
    channel order (renderers vary). Returns a uint8 array in the same order."""
    if strength is None or strength <= 0:
        return img
    s = float(strength)
    src = np.clip(img, 0, 255).astype(np.uint8)
    to_lab = cv2.COLOR_BGR2LAB if bgr else cv2.COLOR_RGB2LAB
    from_lab = cv2.COLOR_LAB2BGR if bgr else cv2.COLOR_LAB2RGB
    lab = cv2.cvtColor(src, to_lab).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=max(2.0, src.shape[1] * 0.012))
    L = L + (0.5 * s) * (L - blur)                              # clarity / local contrast
    n = np.clip(L, 0, 255) / 255.0
    L = np.clip(L, 0, 255) + (0.35 * s * 255.0) * (n * n) * (1.0 - n)   # highlight glow, soft roll-off
    A = 128.0 + (A - 128.0) * (1.0 + 0.30 * s)                  # saturation
    B = 128.0 + (B - 128.0) * (1.0 + 0.30 * s)
    out = np.stack([np.clip(L, 0, 255), np.clip(A, 0, 255), np.clip(B, 0, 255)], -1).astype(np.uint8)
    return cv2.cvtColor(out, from_lab)


def load_and_normalize(img_bytes: bytes, max_dim: int, warns: WarningCollector) -> LoadedImage:
    bgr_full = decode_image(img_bytes, warns)
    src_h, src_w = bgr_full.shape[:2]          # TRUE file size, before any upscaling
    # Faithful in-house enhancement (upscale small / denoise grainy) before any
    # downsizing, so imperfect source photos still give a clean tonal map. Never
    # alters identity; clean, well-sized photos pass through untouched.
    from .enhance import enhance_source
    bgr_full = enhance_source(bgr_full, warns)
    oh, ow = bgr_full.shape[:2]

    longest = max(oh, ow)
    if longest > max_dim:
        scale = max_dim / float(longest)
        new_w = max(1, int(round(ow * scale)))
        new_h = max(1, int(round(oh * scale)))
        bgr = cv2.resize(bgr_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        bgr = bgr_full

    if min(bgr.shape[:2]) < 64:
        warns.warn(
            "preprocess",
            "small_image",
            f"Working image is small ({bgr.shape[1]}x{bgr.shape[0]}); detail may be poor.",
        )

    bgr = _auto_expose(bgr, warns)   # lift dark photos; well-lit ones untouched
    if os.environ.get("TYPO_AUTOLEVELS", "").strip().lower() in ("1", "true", "on", "yes"):
        bgr = _auto_levels(bgr, warns)   # recover flat/low-contrast photos (opt-in; range-gated no-op on good ones)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    return LoadedImage(bgr=bgr, gray=gray, orig_w=ow, orig_h=oh, scale=scale,
                       src_w=src_w, src_h=src_h)
