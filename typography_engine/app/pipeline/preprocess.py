"""Image loading and normalization (OpenCV)."""
from __future__ import annotations

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


def load_and_normalize(img_bytes: bytes, max_dim: int, warns: WarningCollector) -> LoadedImage:
    bgr_full = decode_image(img_bytes, warns)
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

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    return LoadedImage(bgr=bgr, gray=gray, orig_w=ow, orig_h=oh, scale=scale)
