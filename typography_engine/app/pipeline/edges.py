"""Edge and contour extraction (OpenCV)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .preprocess import LoadedImage
from .warnings import WarningCollector


@dataclass
class EdgeResult:
    edges: np.ndarray              # HxW uint8 (0/255)
    contours: List[np.ndarray]     # list of Nx1x2 int contours, largest first


def detect_edges(
    img: LoadedImage,
    warns: WarningCollector,
    low: int,
    high: int,
    mask: np.ndarray | None = None,
) -> EdgeResult:
    gray = img.gray
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)

    if mask is not None:
        # Restrict edges to the subject so background clutter is ignored.
        m = (mask > 127).astype(np.uint8) * 255
        edges = cv2.bitwise_and(edges, m)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    if not cnts:
        warns.warn("edges", "no_contours", "No edge contours detected.")

    return EdgeResult(edges=edges, contours=cnts)


def silhouette_contour(mask: np.ndarray, warns: WarningCollector) -> np.ndarray | None:
    """Outer boundary contour of the silhouette mask (largest external)."""
    cnts, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        warns.warn("edges", "no_silhouette_contour", "Could not trace silhouette boundary.")
        return None
    return max(cnts, key=cv2.contourArea)
