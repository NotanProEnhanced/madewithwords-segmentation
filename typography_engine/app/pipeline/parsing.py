"""Semantic regions of a person: hair, face skin, body skin, clothes, accessories.

WHY THIS EXISTS
  The displacement engine decides type size from a "detail field" built entirely out of
  MediaPipe face-mesh landmarks:

      df = np.clip(0.52 * face_norm + 0.70 * feat_norm, 0, 1)

  Both terms measure DISTANCE FROM THE FACE. Anything the mesh does not cover -- hair, a
  neck, a collar, a hat -- sits at df ~ 0, and df ~ 0 selects the LARGEST type tier. So
  those regions do not get a considered size; they get the coarsest one by default.

  The visible result is that the same region can be treated two opposite ways in one
  photograph: a man's short hair sits close to the face hull, the feather reaches it, and it
  renders in fine type -- while a woman's fuller hair stands away from the hull, drops to
  df ~ 0, and renders in the largest words on the picture. The engine is not choosing badly.
  It has no concept of hair, and the size it lands on is an accident of geometry.

  This module supplies the missing concept. With regions labeled, hair is hair wherever it
  is and however far from a landmark it happens to sit.

THE MODEL
  MediaPipe's multiclass selfie segmenter. 16MB, and MediaPipe is already a dependency --
  the engine uses its face mesh and selfie segmenter -- so this adds no new runtime.

  Measured on the ten-image test set: a hat comes back as 22.8% of the frame in the upper
  rows with hair collapsing to 1.2%, so hats are cleanly separable. Dark sunglasses land at
  5% across the eye band. Clear glasses do NOT separate (0.8% scattered) -- they read as
  face, so glasses handling cannot be promised on this model.

  It is trained on PEOPLE. On a photograph of a cat and a dog it returns 99.97% background,
  so `regions()` returns None there and every caller must fall back to its previous
  behavior rather than treating "no regions" as "no subject".

Fail-safe throughout: any failure returns None and the caller carries on unchanged.
"""
from __future__ import annotations

import os
import urllib.request
from threading import Lock
from typing import Optional

import cv2
import numpy as np

from .warnings import WarningCollector

# Class ids the model emits.
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, ACCESSORY = range(6)
NAMES = {BACKGROUND: "background", HAIR: "hair", BODY_SKIN: "body-skin",
         FACE_SKIN: "face-skin", CLOTHES: "clothes", ACCESSORY: "others/accessory"}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
MODEL_MIN_BYTES = 5_000_000

_LOCK = Lock()
_SEG = None
_FAILED_AT = 0.0
_RETRY_AFTER = 300.0


def _model_path() -> str:
    # data/ is bind-mounted, so the model survives an image rebuild. Putting it under
    # app/models would mean re-downloading it on every deploy.
    d = os.environ.get("TYPO_MODEL_DIR", "/app/data/models")
    return os.path.join(d, "selfie_multiclass_256x256.tflite")


def enabled() -> bool:
    return (os.environ.get("TYPO_PARSE", "").strip().lower()
            in ("1", "true", "on", "yes"))


def _ensure_model(warns: WarningCollector) -> Optional[str]:
    """Download to a temporary file and rename into place. The same atomic fetch as the
    other models here, for the same reason: a partly-written file at the final path gets
    loaded by whatever asks next, fails, and the failure can latch."""
    path = _model_path()
    if os.path.exists(path):
        if os.path.getsize(path) >= MODEL_MIN_BYTES:
            return path
        try:
            os.remove(path)
        except OSError:
            return None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if os.path.getsize(tmp) < MODEL_MIN_BYTES:
            os.remove(tmp)
            warns.warn("parsing", "model_too_small", "parsing model download was truncated")
            return None
        os.replace(tmp, path)
        return path
    except Exception as e:  # noqa: BLE001
        warns.warn("parsing", "model_download_failed", "could not fetch parsing model: %s" % e)
        return None


def _segmenter(warns: WarningCollector):
    """Cached segmenter, or None. Backs off rather than latching: one transient failure must
    not silently disable regions for the life of the process."""
    global _SEG, _FAILED_AT
    import time
    with _LOCK:
        if _SEG is not None:
            return _SEG
        if _FAILED_AT and (time.monotonic() - _FAILED_AT) < _RETRY_AFTER:
            return None
        try:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision

            path = _ensure_model(warns)
            if not path:
                _FAILED_AT = time.monotonic()
                return None
            _SEG = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=mpp.BaseOptions(model_asset_path=path),
                    output_category_mask=True))
            _FAILED_AT = 0.0
        except Exception as e:  # noqa: BLE001
            _FAILED_AT = time.monotonic()
            _SEG = None
            warns.warn("parsing", "init_failed", "parsing unavailable: %s" % e)
    return _SEG


def regions(bgr: np.ndarray, warns: WarningCollector,
            size: Optional[tuple] = None) -> Optional[np.ndarray]:
    """A label map (uint8, values 0..5) at `size` = (W, H), or None.

    None means "no usable regions" and callers must fall back, not assume an empty subject.
    That includes the case that matters most: an animal, where this model sees only
    background.
    """
    if not enabled():
        return None
    seg = _segmenter(warns)
    if seg is None:
        return None
    try:
        import mediapipe as mp

        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cat = seg.segment(img).category_mask.numpy_view().astype(np.uint8)
        # Almost nothing but background means this is not a person -- a pet, an object, a
        # photograph the model cannot read. Returning the mask anyway would let a caller
        # treat the whole frame as "not hair, not face", which is worse than no answer.
        if float((cat != BACKGROUND).mean()) < 0.05:
            return None
        if size is not None and (cat.shape[1], cat.shape[0]) != tuple(size):
            # NEAREST: these are labels, and interpolating between class 1 and class 3
            # would invent class 2 along every boundary.
            cat = cv2.resize(cat, tuple(size), interpolation=cv2.INTER_NEAREST)
        return cat
    except Exception as e:  # noqa: BLE001
        warns.warn("parsing", "segment_failed", "parsing failed: %s" % e)
        return None


def share(cat: np.ndarray, cls: int) -> float:
    """Fraction of the frame carrying one class -- for diagnostics and for deciding whether
    a region is substantial enough to act on."""
    return float((cat == cls).mean()) if cat is not None else 0.0
