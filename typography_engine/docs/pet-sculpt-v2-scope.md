# Pet Sculpt v2 — Dog-Landmark Scope & Findings

Goal: make the premium **sculpt** (displacement) engine fire on **every dog**, not
just the retrievers the human 478-point face mesh happens to accept. Tonal already
covers all dogs (incl. pale/spotted coats after the `pet_subject` light-coat fix);
this is purely about unlocking the *premium* look universally.

## Status: render risk RETIRED (see proof below)

A **Dalmatian** — flat-faced, spotted, non-retriever, which the human mesh fails on
— sculpted at full premium quality through the **unchanged** engine, using only
hand-placed dog landmarks (`scratchpad/manual_sculpt_proof.py` →
`Desktop/DogSculpt-Dalmatian-manual-landmarks.png`). Quality is indistinguishable
from the Golden/Yellow-Lab sculpts.

Two findings from the proof:
1. **The engine sculpts any dog well given adequate landmarks.** No output-quality
   unknown remains. The problem reduces to producing ~15 rough dog keypoints.
2. **The engine is tolerant of imprecise landmarks.** The 3D drape is driven by
   photo luminance (`displacement.py` ~L395: `D = GaussianBlur(gray, ...)`), NOT by
   landmarks. Landmarks only steer the text-SIZE gradient + feature crispness. So a
   dog model needs to be only *roughly* right — which slashes both accuracy demands
   and annotation effort.

## Why the human mesh fails on dogs

`analyze_image` → `detect_faces` runs MediaPipe FaceLandmarker (human face model).
It false-fires on retriever morphology and misses pugs/dalmatians/collies. When it
returns None, `render_displacement_portrait` raises `displacement_needs_face`
(`displacement.py:181-183`).

## What the sculpt engine actually consumes (traced)

- **Shape / size gradient** — `fmh = convexHull(all points)` (L367) → face region;
  `_GROUPS` feature index sets (L78) → text fines on eyes/nose/mouth, forehead calms
  (L364-391). **The only dog-specific part.**
- **3D drape** — photo-luminance driven (L395-399). **Landmark-independent.**
- **Living eyes** — hard-coded to MediaPipe 478 indices: iris 468-477, EAR points
  33/160/158…, gated on `len(pts) >= 478` (L228-279). **Never fires for dogs; auto
  falls back.** Dogs use the **photo-eye overlay** instead (`_photo_eye_overlay`,
  imported at L659) — which project memory already validated as THE eye-realism lever
  (synthetic eyes were rejected).

## Adapter contract (proven end-to-end)

A dog-landmark source must supply only:
- **A face-covering point ring** → drives `convexHull` face region + size gradient.
- **Feature clusters** for `Leye / Reye / nose / lips / Lbrow / Rbrow`.
- **Eyes:** photo-overlay, not synthetic iris.

No 478 points, no iris rings, no EAR gating for dogs.

## Model landscape (researched Jul 2026)

- **No off-the-shelf pretrained dog-FACE landmark model exists.** DogFLW is
  dataset-only (Kaggle, **CC BY-NC 4.0 — non-commercial**): 4335 images, 46 points,
  no checkpoint/inference code. HuggingFace/Roboflow have human-face + animal-BODY
  pose (AP-10K, Dog-Pose) but nothing for dog faces. → must train our own.
- **Ultralytics YOLO is AGPL-3.0** — commercial use needs their paid license or a
  permissive detector.

## Decision: source of the dog-landmark model → **annotate our own small set**

Rationale: only path that is simultaneously commercially clean, cheap (proof shows
~150 rough-labeled faces / ~15 points likely suffice), and owned. DogFLW commercial
license is the fallback if a small self-set underperforms.

## Plan (lean)

1. **Scheme:** ~15 points mapped to the adapter contract (face ring 6-8, 2 eyes,
   nose, mouth, 2 brow).
2. **Label ~150 dog faces** with a simple click tool.
3. **Train** a lightweight keypoint head; evaluate on the test-dog set.
4. **Adapter** (dog points → crafted landmark array, per the proof) + wire
   `_photo_eye_overlay`, all **gated behind `pet_subject`** so typortrait stays
   byte-identical (verified: patched vs original render byte-for-byte equal).
5. **Detector/crop:** permissively-licensed dog detector (avoid AGPL in prod).

## Fallback / no regret

Any dog the sculpt can't handle drops to **tonal**, which now works on every coat.
v2 is pure upside with no regression floor.

## VPS / cost

PyTorch keypoint model is heavier than MediaPipe. Cheapest first: CPU nano model
(renders already 2-5s, fits); else serverless GPU invoked ONLY on the paid sculpt
path; add a GPU only if volume justifies it.
