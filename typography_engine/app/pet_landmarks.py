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
from . import settings as _settings

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

_DEFAULT_DET_PATH = _settings.REGISTRY["PET_LM_DET_MODEL"].default
_DEFAULT_POSE_PATH = _settings.REGISTRY["PET_LM_POSE_MODEL"].default
# `or default`, not just a get()-default: docker-compose.yml passes these through as
# `${PET_LM_DET_MODEL:-}`, which always DEFINES the var in the container -- as an empty
# string when a tree's .env doesn't set it. get()'s own default only applies when the key is
# ABSENT, so an empty string silently wins over it. Confirmed live: without `or default` here,
# every render read PET_LM_DET_MODEL="" and PET_LM_POSE_MODEL="", os.path.exists("") is always
# False, and the feature was unreachable regardless of whether the real files were on disk.
_DET_MODEL_PATH = _settings.raw("PET_LM_DET_MODEL") or _DEFAULT_DET_PATH
_DET_MODEL_URL = _settings.raw("PET_LM_DET_URL")
_POSE_MODEL_PATH = _settings.raw("PET_LM_POSE_MODEL") or _DEFAULT_POSE_PATH
# No fetch URL for the pose model: OpenMMLab distributes it as a .zip bundling config/pipeline
# JSON alongside the .onnx, not a bare file suitable for a direct download-in-place. Prefetched
# and unzipped once at image build time (see Dockerfile); this module never fetches it itself --
# if it isn't on disk, the feature is silently unavailable, exactly like a missing detector file.

# Verified-safe confidence floor: every FACE point (eyes, nose) across 4 test photos / 8 animal
# instances scored between 1.37 and 1.84. 0.7 leaves a wide margin without being paranoid, while
# still well above noise-level scores. (Body points are NOT gated by this alone -- see the module
# docstring for why they are not used here at all.)
_MIN_SCORE = float(_settings.raw("PET_LM_MIN_SCORE") or 0.7)
# A second animal counts as a subject of the portrait when its box is at least this fraction
# of the largest one's area (see comparable_subjects).
_SECOND_MIN = float(_settings.raw("PET_LM_SECOND_MIN") or 0.25)
# Empty string (compose's `${PET_LM_USE_NECK:-}` default) happens to fall through to "on" here
# too, since "" isn't in the off-list below -- but explicit is better than relying on that,
# given the _DET_MODEL_PATH/_POSE_MODEL_PATH bug this exact "empty string from compose" shape
# just caused elsewhere in this file.
_USE_NECK = ((_settings.raw("PET_LM_USE_NECK").strip().lower() or "1")
             not in ("0", "false", "off", "no"))

_DET_INPUT = (640, 640)
_POSE_INPUT = (256, 256)

# AP-10K's 17-keypoint order (mmpose configs/_base_/datasets/ap10k.py). Only the first four are
# ever read by this module; the rest exist so the index positions below stay self-documenting.
_L_EYE, _R_EYE, _NOSE, _NECK = 0, 1, 2, 3

_LOCK = Lock()
_STATE = {"det": None, "pose": None, "ready": None}   # ready: None=untried, True/False=cached result
_FAILED_AT = 0.0
_RETRY_AFTER = float(_settings.raw("PET_LM_RETRY_AFTER") or 300.0)

# PET_LM_DEBUG=1: print exactly what happened and why, at every decision point. Every failure
# path below is a silent `return None` by design (a bad detection must never crash or visibly
# alter a paying customer's render) -- which also means, without this, there is NO way to tell
# "no pet detected" apart from "a real bug swallowed by the safety net" from outside the process.
_DEBUG = _settings.raw("PET_LM_DEBUG").strip().lower() not in ("", "0", "false", "off")


def _dbg(msg):
    if _DEBUG:
        print("[pet_landmarks] %s" % msg, flush=True)


def _no_arena(tool, path):
    """Re-open the tool's runtime session without a memory arena. rtmlib builds its sessions
    with the default options, and an onnxruntime arena keeps every buffer a run ever needed
    for the life of the process (measured on the matting model: 1.95 GB after two runs).
    Same model, same provider, same numbers; the run allocates and frees its own buffers."""
    try:
        import onnxruntime as ort
        if isinstance(getattr(tool, "session", None), ort.InferenceSession):
            so = ort.SessionOptions()
            so.enable_cpu_mem_arena = False
            tool.session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001 -- the arena is a memory matter, never a reason not to detect
        _dbg("could not rebuild session without arena: %r" % (e,))


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
            _no_arena(det, _DET_MODEL_PATH)
            _no_arena(pose, _POSE_MODEL_PATH)
            _STATE["det"], _STATE["pose"], _STATE["ready"] = det, pose, True
            _dbg("models loaded OK")
            return det, pose
        except Exception as e:  # noqa: BLE001 -- any failure -> feature silently unavailable
            _dbg("model load FAILED: %r" % (e,))
            _STATE["ready"] = False
            _FAILED_AT = time.monotonic()
            return None, None


def detect_subjects(bgr):
    """Every cat or dog the detector sees, largest first, as (x0, y0, x1, y1) pixel boxes.

    None when the detector is unavailable (so the caller cannot tell whether there is a pet);
    an empty list when it ran and found no cat or dog -- a person, a rabbit, an empty room.
    Never raises.
    """
    det, _pose = _load_models()
    if det is None:
        return None
    try:
        bboxes, classes = det(bgr)
        # The detector's own rows, untouched: the pose model's crop is computed from the box in
        # its own precision, and a converted copy could move a keypoint by a hair.
        pet_boxes = [b[:4] for b, c in zip(bboxes, classes) if c in (_COCO_CAT, _COCO_DOG)]
        if not pet_boxes:
            _dbg("no dog/cat detected (detector found %d box(es) total, none cat/dog)" % len(bboxes))
        pet_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return pet_boxes
    except Exception as e:  # noqa: BLE001 -- a bad frame must degrade to "nothing found", never crash a render
        _dbg("detector FAILED: %r" % (e,))
        return None


def comparable_subjects(boxes):
    """The boxes that count as subjects of the portrait: the largest, and any other whose area
    is at least PET_LM_SECOND_MIN (default 0.25) of it. Two dogs sitting together both pass; a
    cat asleep in the far corner of the room does not, and is background."""
    if not boxes:
        return []
    a0 = max(1.0, (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1]))
    return [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= _SECOND_MIN * a0]


def face_landmarks(bgr, mask=None):
    """Return a dict with pixel coords for the verified-reliable face points, or None.

    Keys present when found and confident: "eye_l", "eye_r", "nose", and (if PET_LM_USE_NECK)
    "neck". Always returns "head_center" (mean of eye_l/eye_r/nose) alongside them when the eyes
    and nose were both found -- that trio is the only combination measured as reliable across
    every test photo, so it is the only case this function reports success for.

    If more than one dog/cat is detected, the LARGEST bbox is used (the render is a single-subject
    portrait; a smaller animal elsewhere in frame is treated as background, not the subject).
    all_face_landmarks() is the multi-subject form.

    Never raises. Returns None on: missing dependency/model files, no dog/cat detected, or the
    eyes/nose not clearing the confidence floor.
    """
    boxes = detect_subjects(bgr)
    if not boxes:
        return None
    return face_in_box(bgr, boxes[0], mask)


def all_face_landmarks(bgr, boxes, mask=None):
    """One face reading per box, in the boxes' order: the dict face_landmarks() returns, or
    None where that animal's eyes and nose did not clear the confidence floor. Never raises."""
    return [face_in_box(bgr, b, mask) for b in (boxes or [])]


def human_faces(bgr, exclude_boxes=None, min_eye_sep=40.0):
    """People in the photo, largest first, from the MediaPipe face mesh the human brands use:
    {"eye_l", "eye_r", "nose", "mouth", "eye_sep"} in pixels (iris centres, nose tip, the
    lips' midpoint). MediaPipe will also call a dog's face a human face (measured: the golden
    retriever, eyes 256 px apart), so any face whose eyes lie inside one of `exclude_boxes`
    -- every cat or dog the detector found -- is dropped. Never raises; [] when the mesh is
    unavailable (it needs the GL libraries the image carries)."""
    try:
        import cv2
        from .pipeline.landmarks import detect_faces
        from .pipeline.preprocess import LoadedImage
        from .pipeline.warnings import WarningCollector
        img = LoadedImage(bgr=bgr, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                          orig_w=bgr.shape[1], orig_h=bgr.shape[0], scale=1.0)
        faces = detect_faces(img, WarningCollector())
    except Exception as e:  # noqa: BLE001 -- a person is a bonus subject, never a failed render
        _dbg("human face mesh unavailable: %r" % (e,))
        return []
    out = []
    for f in faces:
        p = f.points
        if len(p) < 478:
            continue
        eye_l, eye_r = (float(p[468][0]), float(p[468][1])), (float(p[473][0]), float(p[473][1]))
        sep = float(np.hypot(eye_l[0] - eye_r[0], eye_l[1] - eye_r[1]))
        if sep < min_eye_sep:
            continue
        mx, my = 0.5 * (eye_l[0] + eye_r[0]), 0.5 * (eye_l[1] + eye_r[1])
        if any(b[0] <= mx <= b[2] and b[1] <= my <= b[3] for b in (exclude_boxes or [])):
            _dbg("human face at (%d,%d) lies inside a pet box -- an animal, not a person" % (mx, my))
            continue
        out.append({"eye_l": eye_l, "eye_r": eye_r,
                    "nose": (float(p[1][0]), float(p[1][1])),
                    "mouth": (float((p[13][0] + p[14][0]) / 2), float((p[13][1] + p[14][1]) / 2)),
                    "eye_sep": sep, "points": np.asarray(p, np.float32)})
    out.sort(key=lambda d: d["eye_sep"], reverse=True)
    return out


def anatomy_field(points, H, W, eye_sep):
    """A tangent field from a face's own structure, for the streamlines to follow on skin.

    The engine's flow field comes from texture (fur, hair); skin has almost none, so on a
    face the lanes take their direction from noise. This builds the direction from the mesh
    instead: the face oval, both brows, both eye rings, the lips and the nose are traced as
    chains, and every pixel takes the tangent of the nearest point on the nearest chain, so
    the cheek runs concentric with the jaw, the forehead with the brow and the crown, and the
    type circles each eye and the mouth. Returns (theta, weight) at the frame's size, theta
    in radians, weight 1 on a contour falling to 0 by 1.5 eye-separations away, or None when
    the mesh is unusable. The weight is what the engine blends by; where it is 0 the texture
    field stands untouched, which is the hair and everything below the jaw."""
    try:
        import cv2
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
        from .pipeline.pathgen import order_edges_into_chains
    except Exception as e:  # noqa: BLE001
        _dbg("anatomy field unavailable: %r" % (e,))
        return None
    pts_all = np.asarray(points, np.float32)
    if pts_all.shape[0] < 478 or eye_sep <= 1:
        return None
    label = np.zeros((H, W), np.int32)
    tangents = [0.0]
    sid = 0
    for name in ("FACE_LANDMARKS_FACE_OVAL", "FACE_LANDMARKS_LEFT_EYEBROW", "FACE_LANDMARKS_RIGHT_EYEBROW",
                 "FACE_LANDMARKS_LIPS", "FACE_LANDMARKS_LEFT_EYE", "FACE_LANDMARKS_RIGHT_EYE", "FACE_LANDMARKS_NOSE"):
        conns = getattr(C, name, None)
        if not conns:
            continue
        for chain, closed in order_edges_into_chains([(e.start, e.end) for e in conns]):
            pts = pts_all[chain]
            if closed:
                pts = np.vstack([pts, pts[:1]])
            for i in range(len(pts) - 1):
                p, q = pts[i], pts[i + 1]
                d = q - p
                seg = float(np.hypot(d[0], d[1]))
                if seg < 1e-3:
                    continue
                ang = float(np.arctan2(d[1], d[0]))
                n = max(1, int(seg / 2.0))
                for t in np.linspace(0.0, 1.0, n, endpoint=False):
                    x, y = p + t * d
                    xi, yi = int(x), int(y)
                    if 0 <= xi < W and 0 <= yi < H:
                        sid += 1
                        tangents.append(ang)
                        label[yi, xi] = sid
    if sid == 0:
        return None
    src = (label == 0).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # DIST_LABEL_PIXEL numbers every contour pixel in scan order; map that number to the tangent.
    ys, xs = np.nonzero(label)
    lut = np.zeros(int(labels.max()) + 1, np.float32)
    lut[labels[ys, xs]] = np.asarray(tangents, np.float32)[label[ys, xs]]
    theta = lut[labels]
    sig = 0.6 * float(eye_sep)
    weight = np.exp(-(dist / sig) ** 2).astype(np.float32)
    weight[dist > 1.5 * eye_sep] = 0.0
    # Smooth in doubled-angle space so the seam between two chains' nearest regions is soft.
    k = max(1.0, 0.08 * float(eye_sep))
    c = cv2.GaussianBlur(np.cos(2 * theta) * weight, (0, 0), k)
    s = cv2.GaussianBlur(np.sin(2 * theta) * weight, (0, 0), k)
    theta = (0.5 * np.arctan2(s, c)).astype(np.float32)
    return theta, weight


def face_in_box(bgr, box, mask=None):
    """The face points of the animal inside `box` (see face_landmarks for the keys), or None."""
    _det, pose = _load_models()
    if pose is None:
        return None
    try:
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
