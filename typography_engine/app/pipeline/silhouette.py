"""Foreground silhouette extraction via OpenCV GrabCut.

Deterministic and self-contained (no external model download required). Returns
a binary mask (255 = subject) plus a confidence score so callers can decide
whether to trust the silhouette or emit a warning.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import LoadedImage
from .warnings import WarningCollector


@dataclass
class Silhouette:
    mask: np.ndarray          # HxW uint8, 0/255
    bbox: tuple               # (x, y, w, h) in working coords
    coverage: float           # foreground fraction of frame
    confidence: float


def _bbox_of(mask: np.ndarray):
    ys, xs = np.where(mask > 127)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _confidence(mask: np.ndarray, bbox, coverage: float) -> float:
    h, w = mask.shape[:2]
    if coverage <= 1e-4:
        return 0.0
    size_score = min(1.0, coverage / 0.35)
    pen = 1.0
    if coverage < 0.05:
        pen *= 0.35
    if coverage > 0.85:
        pen *= 0.5
    x, y, bw, bh = bbox
    touch = 0
    if x <= 2:
        touch += 1
    if y <= 2:
        touch += 1
    if x + bw >= w - 3:
        touch += 1
    if y + bh >= h - 3:
        touch += 1
    border_pen = [1.0, 0.92, 0.8, 0.65, 0.5][touch]
    conf = (0.25 + 0.55 * size_score) * pen * border_pen
    return float(max(0.0, min(1.0, conf)))


def extract_silhouette(
    img: LoadedImage,
    warns: WarningCollector,
    face_bbox: tuple | None = None,
    iters: int = 5,
) -> Silhouette:
    h, w = img.bgr.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Thin border ring is almost always background in a portrait.
    b = max(1, int(min(w, h) * 0.02))
    gc_mask[:b, :] = cv2.GC_BGD
    gc_mask[-b:, :] = cv2.GC_BGD
    gc_mask[:, :b] = cv2.GC_BGD
    gc_mask[:, -b:] = cv2.GC_BGD

    if face_bbox is not None:
        # Build a head + shoulders foreground region anchored on the face box.
        fx, fy, fw, fh = [float(v) for v in face_bbox]
        # Probable-foreground head ellipse (a bit larger than the face, to include hair).
        head_cx = fx + fw / 2.0
        head_cy = fy + fh * 0.45
        head_rx = fw * 0.85
        head_ry = fh * 0.95
        cv2.ellipse(
            gc_mask, (int(head_cx), int(head_cy)), (int(head_rx), int(head_ry)),
            0, 0, 360, int(cv2.GC_PR_FGD), -1,
        )
        # Shoulders/torso: widening trapezoid below the face down to the frame bottom.
        neck_top = fy + fh * 0.85
        shoulder_w = fw * 1.9
        torso = np.array([
            [head_cx - fw * 0.45, neck_top],
            [head_cx + fw * 0.45, neck_top],
            [min(w, head_cx + shoulder_w / 2), h],
            [max(0, head_cx - shoulder_w / 2), h],
        ], dtype=np.int32)
        cv2.fillPoly(gc_mask, [torso], int(cv2.GC_PR_FGD))
        # Definite-foreground core of the face anchors the cut.
        cx0 = max(0, int(fx + fw * 0.25))
        cy0 = max(0, int(fy + fh * 0.25))
        cx1 = min(w, int(fx + fw * 0.75))
        cy1 = min(h, int(fy + fh * 0.75))
        if cx1 > cx0 and cy1 > cy0:
            gc_mask[cy0:cy1, cx0:cx1] = cv2.GC_FGD
        rect = (b, b, w - 2 * b, h - 2 * b)
    else:
        # No face: fall back to a centered probable-foreground oval.
        warns.warn("silhouette", "no_face_anchor", "No face box; silhouette seeded by centered region only.")
        cv2.ellipse(
            gc_mask, (w // 2, h // 2), (int(w * 0.32), int(h * 0.42)),
            0, 0, 360, int(cv2.GC_PR_FGD), -1,
        )
        rect = (b, b, w - 2 * b, h - 2 * b)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img.bgr, gc_mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        warns.error("silhouette", "grabcut_failed", f"GrabCut failed: {e}")
        return Silhouette(np.zeros((h, w), np.uint8), (0, 0, 0, 0), 0.0, 0.0)

    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Clean up: keep largest connected component, fill holes, smooth.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(labels == largest, 255, 0).astype(np.uint8)

    bbox = _bbox_of(fg)
    coverage = float((fg > 127).sum()) / float(h * w)
    conf = _confidence(fg, bbox, coverage)

    if conf < 0.35:
        warns.warn(
            "silhouette",
            "low_confidence",
            f"Silhouette confidence is low ({conf:.2f}); subject may not be cleanly separated.",
        )
    if coverage < 0.04:
        warns.warn(
            "silhouette",
            "tiny_subject",
            f"Subject covers only {coverage*100:.1f}% of frame.",
        )

    return Silhouette(mask=fg, bbox=bbox, coverage=coverage, confidence=conf)
