"""Optional pet FACE anchoring (RTMPose-m / AP-10K via rtmlib), used by pet_proto.py to
sharpen where fine typography lands, beyond the existing photometric `feat` field.

SCOPE, and why it stops at the face: this module was prototyped against real customer-style
photos (a tight head crop, a full-body stance, a multi-pet photo, a partial-body curled pose)
before writing any of this. Eyes/nose/neck were placed correctly in every single test, on both
dogs and cats, regardless of framing -- a solid signal. Body/limb keypoints (shoulders, hips,
paws, tail) were NOT: on a photo with no body in frame the model confidently invents plausible-
looking but wrong locations for them (a "wrong shoulder" scored 0.94, higher than some CORRECT
face points elsewhere), and a geometric consistency check (is the paw farther from the head than
the shoulder, as a real outstretched limb would be) does not reliably separate good chains from
bad ones either -- measured on the same test photos, two verified-CORRECT limb chains (a cat's
front legs, curled rather than outstretched) scored 0.87x and 1.03x on that ratio, statistically
indistinguishable from two verified-WRONG chains at 0.95x and 1.08x. Rather than ship a gate that
doesn't actually gate, body/limb anchoring is deferred. Only FACE points are exposed here.

Fully optional at every layer: if rtmlib/onnxruntime aren't installed, or the model files
weren't prefetched (see Dockerfile), every public function here returns None and pet_proto.py's
PET_LANDMARKS=0 default keeps rendering byte-identical to before this module existed. A load
failure backs off for PET_LM_RETRY_AFTER seconds rather than retrying on every single render, so
one bad request cannot turn into a standing per-render latency cost -- same pattern _u2net_session
already uses in pet_proto.py for the matting model.
"""
from __future__ import annotations

import os
import time
from threading import Lock

import numpy as np

try:
    import cv2
    from rtmlib import YOLOX, RTMPose
    _IMPORT_OK = True
except Exception:  # noqa: BLE001 -- optional dependency; must never crash the app at import time
    YOLOX = RTMPose = None
    _IMPORT_OK = False

# COCO detector class ids for the two species this pipeline serves.
_COCO_CAT = 15
_COCO_DOG = 16

_DET_MODEL_PATH = os.environ.get("PET_LM_DET_MODEL", "models/yolox_m.onnx")
_DET_MODEL_URL = os.environ.get(
    "PET_LM_DET_URL",
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx",
)
_POSE_MODEL_PATH = os.environ.get("PET_LM_POSE_MODEL", "models/rtmpose_ap10k_m.onnx")
# No fetch URL for the pose model: OpenMMLab distributes it as a .zip bundling config/pipeline
# JSON alongside the .onnx, not a bare file suitable for a direct download-in-place. Prefetched
# and unzipped once at image build time (see Dockerfile); this module never fetches it itself --
# if it isn't on disk, the feature is silently unavailable, exactly like a missing detector file.

# Verified-safe confidence floor: every FACE point (eyes, nose) across 4 test photos / 8 animal
# instances scored between 1.37 and 1.84. 0.7 leaves a wide margin without being paranoid, while
# still well above noise-level scores. (Body points are NOT gated by this alone -- see the module
# docstring for why they are not used here at all.)
_MIN_SCORE = float(os.environ.get("PET_LM_MIN_SCORE", "0.7") or 0.7)
_USE_NECK = (os.environ.get("PET_LM_USE_NECK", "1").strip().lower() not in ("0", "false", "off", "no"))

_DET_INPUT = (640, 640)
_POSE_INPUT = (256, 256)

# AP-10K's 17-keypoint order (mmpose configs/_base_/datasets/ap10k.py). Only the first four are
# ever read by this module; the rest exist so the index positions below stay self-documenting.
_L_EYE, _R_EYE, _NOSE, _NECK = 0, 1, 2, 3

_LOCK = Lock()
_STATE = {"det": None, "pose": None, "ready": None}   # ready: None=untried, True/False=cached result
_FAILED_AT = 0.0
_RETRY_AFTER = float(os.environ.get("PET_LM_RETRY_AFTER", "300") or 300.0)

# PET_LM_DEBUG=1: print exactly what happened and why, at every decision point. Every failure
# path below is a silent `return None` by design (a bad detection must never crash or visibly
# alter a paying customer's render) -- which also means, without this, there is NO way to tell
# "no pet detected" apart from "a real bug swallowed by the safety net" from outside the process.
_DEBUG = os.environ.get("PET_LM_DEBUG", "").strip().lower() not in ("", "0", "false", "off")


def _dbg(msg):
    if _DEBUG:
        print("[pet_landmarks] %s" % msg, flush=True)


def _load_models():
    """Return (det, pose) sessions, or (None, None) if unavailable. Cached; a failure backs off
    for _RETRY_AFTER seconds instead of re-attempting (and re-logging) on every render."""
    global _FAILED_AT
    with _LOCK:
        if _STATE["ready"] is True:
            return _STATE["det"], _STATE["pose"]
        if _STATE["ready"] is False and (time.monotonic() - _FAILED_AT) < _RETRY_AFTER:
            _dbg("skipped: a previous load failed within the last %.0fs" % _RETRY_AFTER)
            return None, None
        if not _IMPORT_OK:
            _dbg("unavailable: rtmlib/cv2 import failed at module load time")
            _STATE["ready"] = False
            _FAILED_AT = time.monotonic()
            return None, None
        _det_ok, _pose_ok = os.path.exists(_DET_MODEL_PATH), os.path.exists(_POSE_MODEL_PATH)
        if not (_det_ok and _pose_ok):
            _dbg("unavailable: model file(s) missing -- det(%s)=%s pose(%s)=%s"
                 % (_DET_MODEL_PATH, _det_ok, _POSE_MODEL_PATH, _pose_ok))
            _STATE["ready"] = False
            _FAILED_AT = time.monotonic()
            return None, None
        try:
            det = YOLOX(_DET_MODEL_PATH, det_mode="multiclass", model_input_size=_DET_INPUT)
            pose = RTMPose(_POSE_MODEL_PATH, model_input_size=_POSE_INPUT)
            _STATE["det"], _STATE["pose"], _STATE["ready"] = det, pose, True
            _dbg("models loaded OK")
            return det, pose
        except Exception as e:  # noqa: BLE001 -- any failure -> feature silently unavailable
            _dbg("model load FAILED: %r" % (e,))
            _STATE["ready"] = False
            _FAILED_AT = time.monotonic()
            return None, None


def face_landmarks(bgr, mask=None):
    """Return a dict with pixel coords for the verified-reliable face points, or None.

    Keys present when found and confident: "eye_l", "eye_r", "nose", and (if PET_LM_USE_NECK)
    "neck". Always returns "head_center" (mean of eye_l/eye_r/nose) alongside them when the eyes
    and nose were both found -- that trio is the only combination measured as reliable across
    every test photo, so it is the only case this function reports success for.

    If more than one dog/cat is detected, the LARGEST bbox is used (the render is a single-subject
    portrait; a smaller animal elsewhere in frame is treated as background, not the subject).

    Never raises. Returns None on: missing dependency/model files, no dog/cat detected, or the
    eyes/nose not clearing the confidence floor.
    """
    det, pose = _load_models()
    if det is None or pose is None:
        return None
    try:
        bboxes, classes = det(bgr)
        pet_boxes = [b for b, c in zip(bboxes, classes) if c in (_COCO_CAT, _COCO_DOG)]
        if not pet_boxes:
            _dbg("no dog/cat detected (detector found %d box(es) total, none cat/dog)" % len(bboxes))
            return None
        # Largest by area = the primary subject.
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in pet_boxes]
        box = pet_boxes[int(np.argmax(areas))]
        keypoints, scores = pose(bgr, bboxes=[box])
        kpts, scs = keypoints[0], scores[0]

        def pt(i, name):
            if scs[i] < _MIN_SCORE:
                _dbg("%s below confidence floor: %.3f < %.2f" % (name, scs[i], _MIN_SCORE))
                return None
            x, y = float(kpts[i][0]), float(kpts[i][1])
            if mask is not None:
                h, w = mask.shape[:2]
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < w and 0 <= yi < h) or mask[yi, xi] < 0.3:
                    _dbg("%s landed off-subject at (%d,%d), score %.3f -- discarded" % (name, xi, yi, scs[i]))
                    return None  # off-subject entirely -- discard rather than trust a stray point
            return (x, y)

        eye_l = pt(_L_EYE, "L_Eye")
        eye_r = pt(_R_EYE, "R_Eye")
        nose = pt(_NOSE, "Nose")
        neck = pt(_NECK, "Neck")
        if eye_l is None or eye_r is None or nose is None:
            _dbg("declining: need eyes+nose all confident, got eye_l=%s eye_r=%s nose=%s"
                 % (eye_l is not None, eye_r is not None, nose is not None))
            return None  # the one combination this module is confident in; anything less, decline
        out = {
            "eye_l": eye_l,
            "eye_r": eye_r,
            "nose": nose,
            "head_center": (
                (eye_l[0] + eye_r[0] + nose[0]) / 3.0,
                (eye_l[1] + eye_r[1] + nose[1]) / 3.0,
            ),
        }
        if _USE_NECK and neck is not None:
            out["neck"] = neck
        _dbg("OK: found %s at head_center=(%.0f,%.0f)"
             % (sorted(out.keys()), out["head_center"][0], out["head_center"][1]))
        return out
    except Exception as e:  # noqa: BLE001 -- a bad frame must degrade to "no landmarks", never crash a render
        _dbg("render-time FAILED: %r" % (e,))
        return None
