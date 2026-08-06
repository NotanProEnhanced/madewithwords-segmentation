"""Optional MediaPipe FaceLandmarker wrapper.

MediaPipe 0.10.x exposes only the Tasks API, which needs a downloadable
`face_landmarker.task` model plus system GL libs. All of that is optional: if
anything is missing we emit a warning and the caller falls back to OpenCV-only
heuristics. We never pretend landmarks exist when they don't.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

import numpy as np

from ..config import FACE_LANDMARKER_MODEL, FACE_LANDMARKER_URL
from .preprocess import LoadedImage
from .warnings import WarningCollector

_LOCK = Lock()
_LANDMARKER = None
_INIT_ERROR: Optional[str] = None


@dataclass
class FaceLandmarks:
    # Normalized (0..1) landmark coordinates, indexed by MediaPipe's 478-point mesh.
    points: np.ndarray              # Nx2 float in working-image pixel coords
    image_w: int
    image_h: int
    bbox: tuple                     # (x, y, w, h) in working coords


def ensure_model(warns: WarningCollector, allow_download: bool = True) -> bool:
    if FACE_LANDMARKER_MODEL.exists():
        return True
    if not allow_download:
        warns.warn("landmarks", "model_missing", "Face landmark model not present and download disabled.")
        return False
    try:
        FACE_LANDMARKER_MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
        return FACE_LANDMARKER_MODEL.exists()
    except Exception as e:
        warns.warn("landmarks", "model_download_failed", f"Could not fetch face model: {e}")
        return False


def _get_landmarker(warns: WarningCollector):
    global _LANDMARKER, _INIT_ERROR
    with _LOCK:
        if _LANDMARKER is not None:
            return _LANDMARKER
        if _INIT_ERROR is not None:
            return None
        if not ensure_model(warns):
            _INIT_ERROR = "model_unavailable"
            return None
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )

            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
                running_mode=RunningMode.IMAGE,
                num_faces=5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            _LANDMARKER = FaceLandmarker.create_from_options(opts)
            return _LANDMARKER
        except Exception as e:  # missing GL libs, etc.
            _INIT_ERROR = str(e)
            warns.warn("landmarks", "mediapipe_init_failed", f"MediaPipe unavailable: {e}")
            return None


def detect_faces(img: LoadedImage, warns: WarningCollector) -> List[FaceLandmarks]:
    """All faces found, largest first. Empty if MediaPipe is unavailable or finds
    no face."""
    landmarker = _get_landmarker(warns)
    if landmarker is None:
        return []
    import cv2
    import mediapipe as mp
    h, w = img.bgr.shape[:2]

    def _detect_pts(bgr):
        """Run the landmarker on a BGR (sub-)image; return landmark arrays in that
        image's own pixel coords ([] on error / no face)."""
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        except Exception as e:  # noqa: BLE001
            warns.warn("landmarks", "detect_failed", f"Landmark detection error: {e}")
            return []
        bh, bw = bgr.shape[:2]
        return [np.array([[lm.x * bw, lm.y * bh] for lm in lms], dtype=np.float32)
                for lms in (res.face_landmarks or [])]

    all_pts = _detect_pts(img.bgr)                      # full frame
    # A busy group / wide photo can hide a face from the full-frame detector even
    # though it resolves fine in a tighter crop (measured: a 4-person shot detected
    # 3, missing the 2nd-from-right; a tile crop found it). Recover missed faces by
    # re-detecting over overlapping horizontal tiles and merging (deduped below).
    # Skipped for the common single-subject portrait to avoid the extra passes.
    if len(all_pts) >= 2 or w > int(h * 1.2):
        i = 0
        while i * 0.30 < 1.0:
            x0 = int(i * 0.30 * w)
            x1 = int(min(1.0, i * 0.30 + 0.44) * w)
            if x1 - x0 >= 48:
                for pts in _detect_pts(img.bgr[:, x0:x1]):
                    pts = pts.copy()
                    pts[:, 0] += x0                     # tile-local -> full-frame coords
                    all_pts.append(pts)
            i += 1

    if not all_pts:
        warns.warn("landmarks", "no_face", "No face detected by MediaPipe.")
        return []

    faces: List[FaceLandmarks] = []
    for pts in all_pts:
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        faces.append(FaceLandmarks(points=pts, image_w=w, image_h=h,
                                   bbox=(x0, y0, x1 - x0, y1 - y0)))
    faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)

    # Deduplicate overlapping detections: a face at a tile seam appears in two tiles,
    # and MediaPipe can double-box one face. Keep the largest; drop later boxes that
    # overlap it (IoU) so each person is represented exactly once.
    def _iou(a, b):
        ax0, ay0, aw, ah = a
        bx0, by0, bw2, bh2 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax0 + aw, bx0 + bw2), min(ay0 + ah, by0 + bh2)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        return inter / (aw * ah + bw2 * bh2 - inter) if inter > 0 else 0.0

    kept: List[FaceLandmarks] = []
    for f in faces:
        if all(_iou(f.bbox, k.bbox) < 0.45 for k in kept):
            kept.append(f)
    return kept


def detect_landmarks(img: LoadedImage, warns: WarningCollector) -> Optional[FaceLandmarks]:
    """Primary (largest) face, for single-subject callers."""
    faces = detect_faces(img, warns)
    return faces[0] if faces else None


def haar_face_bbox(img: LoadedImage, warns: WarningCollector) -> Optional[tuple]:
    """OpenCV Haar-cascade face box fallback when MediaPipe is unavailable."""
    import cv2

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        warns.warn("landmarks", "haar_missing", "Haar cascade not found.")
        return None
    faces = cascade.detectMultiScale(img.gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        warns.warn("landmarks", "haar_no_face", "No face found via Haar fallback.")
        return None
    # Largest face.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x), int(y), int(w), int(h))
