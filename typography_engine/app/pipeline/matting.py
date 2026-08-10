"""Optional high-quality alpha matting (RobustVideoMatting, MIT-licensed).

Produces a true per-pixel alpha that resolves individual hair strands -- the fix for
the "cardboard cutout" / words-in-the-gaps look that a coarse segmentation mask can't
give. Entirely opt-in (env TYPO_MATTE_MODEL) and fail-safe: if onnxruntime isn't
installed, the model can't be fetched, or inference errors, `matte()` returns None and
callers fall back to the guided-filter matte on MediaPipe's mask. Nothing here can
break a render.

RVM ONNX I/O (per the model card):
  inputs : src (1,3,H,W float32, RGB 0..1), r1i..r4i (recurrent state, zeros to start),
           downsample_ratio (float32 scalar)
  outputs: fgr, pha (1,1,H,W alpha 0..1), r1o..r4o  -- we use `pha`.
"""
from __future__ import annotations

import os
import urllib.request
from threading import Lock
from typing import Optional

import cv2
import numpy as np

from ..config import MATTE_MODEL, MATTE_MODEL_URL
from .warnings import WarningCollector

_LOCK = Lock()
_SESSION = None
_INIT_ERROR: Optional[str] = None


def enabled() -> bool:
    return os.environ.get("TYPO_MATTE_MODEL", "").strip().lower() in ("1", "true", "on", "yes", "rvm")


def _ensure_model(warns: WarningCollector) -> bool:
    if MATTE_MODEL.exists() and MATTE_MODEL.stat().st_size > 0:
        return True
    try:
        MATTE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MATTE_MODEL_URL, MATTE_MODEL)
        return MATTE_MODEL.exists() and MATTE_MODEL.stat().st_size > 0
    except Exception as e:  # noqa: BLE001
        warns.warn("matte", "model_download_failed", f"Could not fetch matting model: {e}")
        return False


def _get_session(warns: WarningCollector):
    global _SESSION, _INIT_ERROR
    with _LOCK:
        if _SESSION is not None:
            return _SESSION
        if _INIT_ERROR is not None:
            return None
        if not _ensure_model(warns):
            _INIT_ERROR = "model_unavailable"
            return None
        try:
            import onnxruntime as ort  # type: ignore

            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
            _SESSION = ort.InferenceSession(
                str(MATTE_MODEL), sess_options=so, providers=["CPUExecutionProvider"]
            )
            return _SESSION
        except Exception as e:  # noqa: BLE001
            _INIT_ERROR = str(e)
            warns.warn("matte", "init_failed", f"onnxruntime/matting unavailable: {e}")
            return None


def matte(img_bgr: np.ndarray, warns: WarningCollector) -> Optional[np.ndarray]:
    """Return a float32 alpha (HxW, 0..1) at the input image's resolution, or None to
    fall back. Never raises."""
    if not enabled():
        return None
    sess = _get_session(warns)
    if sess is None:
        return None
    try:
        h, w = img_bgr.shape[:2]
        # Run at a bounded resolution for speed/memory; RVM alpha upsamples cleanly.
        long_side = max(h, w)
        cap = int(os.environ.get("TYPO_MATTE_MAXSIDE", "1280") or 1280)
        scale = min(1.0, cap / float(long_side))
        rw, rh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (rw, rh), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)   # 1,3,rh,rw
        # downsample_ratio so RVM's internal recurrent pass sees ~512px on the long side.
        dsr = float(np.clip(512.0 / float(max(rh, rw)), 0.25, 1.0))
        rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]
        feeds = {"src": src, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3],
                 "downsample_ratio": np.asarray([dsr], np.float32)}
        # Ask for the alpha output by name if present, else positional index 1.
        out_names = [o.name for o in sess.get_outputs()]
        want = "pha" if "pha" in out_names else out_names[min(1, len(out_names) - 1)]
        pha = sess.run([want], feeds)[0]
        alpha = np.squeeze(np.asarray(pha)).astype(np.float32)   # rh, rw
        if alpha.ndim != 2:
            return None
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0.0, 1.0)
    except Exception as e:  # noqa: BLE001 -- matting is best-effort
        warns.warn("matte", "infer_failed", f"Matting inference failed: {e}")
        return None
