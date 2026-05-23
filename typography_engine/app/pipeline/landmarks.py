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
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            _LANDMARKER = FaceLandmarker.create_from_options(opts)
            return _LANDMARKER
        except Exception as e:  # missing GL libs, etc.
            _INIT_ERROR = str(e)
            warns.warn("landmarks", "mediapipe_init_failed", f"MediaPipe unavailable: {e}")
            return None


def detect_landmarks(img: LoadedImage, warns: WarningCollector) -> Optional[FaceLandmarks]:
    landmarker = _get_landmarker(warns)
    if landmarker is None:
        return None
    try:
        import mediapipe as mp
        import cv2

        rgb = cv2.cvtColor(img.bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
    except Exception as e:
        warns.warn("landmarks", "detect_failed", f"Landmark detection error: {e}")
        return None

    if not result.face_landmarks:
        warns.warn("landmarks", "no_face", "No face detected by MediaPipe.")
        return None

    h, w = img.bgr.shape[:2]
    lms = result.face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    bbox = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
    return FaceLandmarks(points=pts, image_w=w, image_h=h, bbox=bbox)


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
