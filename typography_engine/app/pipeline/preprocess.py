"""Image loading and normalization (OpenCV)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .warnings import WarningCollector


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
    if bgr is None:
        raise ValueError("could_not_decode_image")
    return bgr


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

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    return LoadedImage(bgr=bgr, gray=gray, orig_w=ow, orig_h=oh, scale=scale)
