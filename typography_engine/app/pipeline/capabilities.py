"""Runtime capability probing.

Reports which optional subsystems are actually usable in this environment so the
/health endpoint can be honest about degraded modes.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def probe() -> Dict[str, object]:
    caps: Dict[str, object] = {}

    try:
        import cv2  # noqa: F401
        caps["opencv"] = cv2.__version__
    except Exception as e:  # pragma: no cover - environment dependent
        caps["opencv"] = f"unavailable: {e}"

    try:
        import numpy as np  # noqa: F401
        caps["numpy"] = np.__version__
    except Exception as e:  # pragma: no cover
        caps["numpy"] = f"unavailable: {e}"

    try:
        import cairosvg  # noqa: F401
        caps["cairosvg"] = getattr(cairosvg, "__version__", "ok")
    except Exception as e:  # pragma: no cover
        caps["cairosvg"] = f"unavailable: {e}"

    # MediaPipe is optional. Importing + locating the model both matter, but we
    # only do a cheap import check here; full landmark init happens lazily.
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks.python import vision  # noqa: F401
        caps["mediapipe"] = mp.__version__
    except Exception as e:  # pragma: no cover
        caps["mediapipe"] = f"unavailable: {e}"

    from ..config import FACE_LANDMARKER_MODEL
    caps["face_landmarker_model_present"] = FACE_LANDMARKER_MODEL.exists()

    return caps
