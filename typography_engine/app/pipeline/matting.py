"""Optional high-quality alpha matting. Two backends, both opt-in and fail-safe.

Produces a true per-pixel alpha that resolves individual hair strands -- the fix for
the "cardboard cutout" / words-in-the-gaps look that a coarse segmentation mask can't
give. Entirely opt-in (env TYPO_MATTE_MODEL) and fail-safe: if onnxruntime isn't
installed, the model can't be fetched, or inference errors, `matte()` returns None and
callers fall back to the guided-filter matte on MediaPipe's mask. Nothing here can
break a render.

    TYPO_MATTE_MODEL=rvm     RobustVideoMatting (also 1/true/on/yes -- the original)
    TYPO_MATTE_MODEL=isnet   ISNet general-use
    unset / anything else    off; the guided-filter matte on MediaPipe's mask

WHY ISNET IS HERE
  The pet renderer (app/pet_proto.py) does not use this module at all -- it is a
  separate engine -- and it settled on isnet-general-use because it is "markedly
  better than u2net on fur edges and white-fur-on-white background". Hair is the
  same problem, and hair is exactly where the human pipeline fails: the segmenter
  cuts into it on tightly-cropped, frame-filling portraits, which reads as shadow on
  a dark backdrop and as a white hole on a light one. That is the defect behind the
  memorial brand's halo.

  So this offers the human pipeline the model that already demonstrably handles the
  harder version of the problem, rather than adding a third one. The model file is
  shared with the pet engine -- same path, same env var -- so enabling it here costs
  no extra download on a box where the pet brand already runs.

RVM ONNX I/O (per the model card):
  inputs : src (1,3,H,W float32, RGB 0..1), r1i..r4i (recurrent state, zeros to start),
           downsample_ratio (float32 scalar)
  outputs: fgr, pha (1,1,H,W alpha 0..1), r1o..r4o  -- we use `pha`.

ISNet ONNX I/O:
  input  : 1,3,1024,1024 float32, RGB, scaled by /max then centred (-0.5)
  output : saliency logits, min-max normalised here to 0..1
"""
from __future__ import annotations

import os
import tempfile
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

# Shared with app/pet_proto.py: same URL, same on-disk path, same env override, so the
# ~170MB file is fetched once and serves both engines.
_ISNET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
_ISNET_SIDE = 1024
_ISNET_LOCK = Lock()
_ISNET_SESSION = None
_ISNET_ERROR: Optional[str] = None


def _model_name() -> str:
    """'rvm', 'isnet', or '' for off. The legacy truthy values keep meaning RVM, so
    existing .env files that say TYPO_MATTE_MODEL=1 are unaffected."""
    v = os.environ.get("TYPO_MATTE_MODEL", "").strip().lower()
    if v == "isnet":
        return "isnet"
    if v in ("1", "true", "on", "yes", "rvm"):
        return "rvm"
    return ""


def enabled() -> bool:
    return _model_name() != ""


def _isnet_path() -> str:
    return os.path.join(os.environ.get("PET_MATTE_DIR", tempfile.gettempdir()), "isnet.onnx")


def _isnet_session(warns: WarningCollector):
    global _ISNET_SESSION, _ISNET_ERROR
    with _ISNET_LOCK:
        if _ISNET_SESSION is not None:
            return _ISNET_SESSION
        if _ISNET_ERROR is not None:
            return None
        path = _isnet_path()
        try:
            if not (os.path.exists(path) and os.path.getsize(path) > 1_000_000):
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                urllib.request.urlretrieve(_ISNET_URL, path)
            if not (os.path.exists(path) and os.path.getsize(path) > 1_000_000):
                _ISNET_ERROR = "model_unavailable"
                warns.warn("matte", "model_download_failed", "Could not fetch the isnet matting model")
                return None
            import onnxruntime as ort  # type: ignore

            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
            _ISNET_SESSION = ort.InferenceSession(
                path, sess_options=so, providers=["CPUExecutionProvider"]
            )
            return _ISNET_SESSION
        except Exception as e:  # noqa: BLE001
            _ISNET_ERROR = str(e)
            warns.warn("matte", "init_failed", f"isnet matting unavailable: {e}")
            return None


def _matte_isnet(img_bgr: np.ndarray, warns: WarningCollector) -> Optional[np.ndarray]:
    """Alpha from ISNet. Mirrors app/pet_proto.py's proven preprocessing rather than
    inventing a second one -- including its collapse guard, which catches the case where
    the model finds essentially no foreground and would otherwise erase the subject."""
    sess = _isnet_session(warns)
    if sess is None:
        return None
    try:
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        im = cv2.resize(rgb, (_ISNET_SIDE, _ISNET_SIDE), interpolation=cv2.INTER_AREA)
        im = im / max(float(im.max()), 1.0) - 0.5
        inp = np.transpose(im, (2, 0, 1))[None].astype(np.float32)
        pred = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0, 0]
        mn, mx = float(pred.min()), float(pred.max())
        pred = (pred - mn) / (mx - mn + 1e-8)
        alpha = cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(1.0, w * 0.0015))
        if float((alpha > 0.5).mean()) < 0.004:
            warns.warn("matte", "matte_collapsed", "isnet found almost no foreground; falling back")
            return None
        return np.clip(alpha, 0.0, 1.0)
    except Exception as e:  # noqa: BLE001 -- matting is best-effort
        warns.warn("matte", "infer_failed", f"isnet matting inference failed: {e}")
        return None


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
    which = _model_name()
    if not which:
        return None
    if which == "isnet":
        return _matte_isnet(img_bgr, warns)
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
