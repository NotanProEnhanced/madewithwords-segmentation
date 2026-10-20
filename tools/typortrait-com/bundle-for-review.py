#!/usr/bin/env python3
"""Package the segmentation defect for outside review.

Collects the engine source, the exact failing input, the intermediate fields
that determine the composite, and a written briefing, into one zip served from
the review directory. Scans for credentials before writing.

    python3 bundle-for-review.py
    -> /var/www/typortrait.com/review/typortrait-mask-bundle.zip
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

TREE = "/root/typortrait-stg/typography_engine"
DATA = os.path.join(TREE, "data")
OUTDIR = "/var/www/typortrait.com/review"
ZIPNAME = "typortrait-mask-bundle.zip"

SOURCES = [
    "app/pipeline/analyze.py",
    "app/pipeline/silhouette.py",
    "app/pipeline/matting.py",
    "app/pipeline/displacement.py",
    "app/pipeline/tonal.py",
    "app/pipeline/preprocess.py",
    "app/config.py",
]

SECRET = re.compile(
    r"(sk_live|sk_test|rk_live|whsec_|BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16})")

BRIEFING = r"""# Typortrait - segmentation mask defect

## What the product does

Typortrait turns a photograph into a portrait built entirely from words. A
person's face is rendered as typography that follows the tone and form of the
original photo. The engine is Python: MediaPipe for face landmarks and person
segmentation, OpenCV for everything else.

## The pipeline, in the order that matters here

1. `analyze_image()` (app/pipeline/analyze.py)
   - `detect_faces()` -> 478-point MediaPipe mesh per face
   - `extract_silhouette()` (app/pipeline/silhouette.py) -> a `Silhouette` with
     - `mask`  HxW uint8, binary subject/background
     - `soft`  HxW uint8, feathered alpha (hair-preserving), may be None
   - Segmenter order in `extract_silhouette`:
     a. RobustVideoMatting ONNX, only if TYPO_MATTE_MODEL is set, and only
        accepted when 0.02 < coverage < 0.98
     b. MediaPipe selfie segmentation (`_selfie_mask`), refined by
        `_soft_matte()` -- a guided filter that snaps the soft alpha to image
        edges
     c. GrabCut fallback seeded from the face box

2. `render_displacement_portrait()` (app/pipeline/displacement.py)
   - `mask01`  binary mask at render resolution (used for density/geometry)
   - `soft01`  from `silhouette.soft` when present, then a knee:
        soft01 = clip((soft01 - TYPO_MATTE_FLOOR) / (1 - TYPO_MATTE_FLOOR))
        soft01 = soft01 ** TYPO_MATTE_GAMMA
     When `soft` is None, soft01 is a Gaussian blur of the binary instead.
   - Composite:  out = _base * (1 - al) + ink_col * al
        `_base` is the flat ground everywhere, EXCEPT inside the silhouette
        where TYPO_SUBJECT_BASE swaps in the dimmed source photograph.
        `al` is the ink alpha (where type is laid down).
   - Backdrop: when a backdrop is requested, the region OUTSIDE `soft01` is
     filled with that colour (`_fill`), blended by `1 - soft01`.

## The defect

On some photographs the segmenter classifies part of the SUBJECT as background.
Observed repeatedly:

  - a wedge across one man's hair, temple and cheek (upper left)
  - a smaller island in a bottom corner, below both faces
  - the top of a woman's head

Downstream, that region is painted with the ground / backdrop instead of the
subject. It reads as a blotch on the person's face. It is most obvious on
two-subject portraits and on tightly-cropped uploads, and it is NOT confined to
one photo -- several different images reproduce it.

The error is longstanding. It was not noticed until the backdrop began
defaulting to a light studio grey (#e6e6e6): on the previous navy ground the
same misclassified region rendered dark and passed as shadow. Rendering the same
failing input against the engine as of the commit BEFORE that default changed
shows the identical wedge, dark. So this is a visibility change, not a new bug.

## Reproduction

Inside the container, with the failing source included in this bundle as
`evidence/src.jpg`:

    from app.config import RenderConfig
    from app.pipeline.warnings import WarningCollector
    from app.pipeline.analyze import analyze_image
    from app.pipeline.displacement import render_displacement_portrait

    b = open('src.jpg','rb').read()
    a = analyze_image(b, RenderConfig(), WarningCollector())
    png = render_displacement_portrait(
        a, ['THE','THEMATIC','TE'], ground='navy', out_width=1400,
        supersample=2, ink='photo', print_aspect=1.25, breathe=True,
        graduate=True, backdrop='studio', flow=False, uppercase=True,
        variety=0.0)

Those arguments were captured from the live endpoint, not guessed.

IMPORTANT: the ORIGINAL uncropped photograph (`evidence/original.png`) renders
CLEANLY with the same settings. Only the app's cropped upload
(`evidence/src.jpg`) fails. The crop leaves the subject filling the frame, which
is where the segmenter's confidence collapses.

## Measurements

Uncropped original:   silhouette coverage 0.9697, faces 2
Cropped upload:       silhouette coverage 0.9380, faces 2

`evidence/` contains, for the FAILING render:

    src.jpg      the exact cropped input
    original.png the uncropped photo, which renders clean
    render.jpg   the failing output
    mask01.png   binary silhouette at render resolution
    soft01.png   the feathered alpha actually used for compositing
    alpha.png    ink alpha
    base.png     the base being composited against

In `soft01.png` the misclassified wedge is plainly black. In `base.png` the same
region carries the flat ground rather than the photograph.

## What has already been ruled out, by measurement

  - Teeth-clearing mask       fixed separately (per-face gate); not the cause
  - Lips anchor               a stroked polyline, cannot fill a region
  - TYPO_INK_LIFT saturation  no pixel in the output clips at 255
  - TYPO_HILIGHT_WASH         unchanged with it disabled
  - TYPO_EDGE_FALLOFF=0       unchanged
  - TYPO_MATTE_MODEL on/off   unchanged; RVM cuts the same wedge
  - TYPO_MATTE_FLOOR=0 with TYPO_MATTE_GAMMA=1.0   unchanged
  - Supersample 1 vs 2        both clean on the uncropped original
  - Render arguments          harness matches the endpoint exactly

## What was tried and rejected

  - Union the RVM and MediaPipe alphas (per-pixel max): added 0.09% coverage,
    did not close the hole.
  - Dilate mask and soft (TYPO_MASK_GROW): no measurable effect on this failure.
  - Force every detected face's convex hull to foreground, grown by a factor:
      grow 0.18 -> wedge remains
      grow 0.60 -> wedge closes, but on a NORMAL portrait (coverage ~0.63) it
                   adds ~9.6% of the frame and paints an oval of real background
                   around the head, which is worse.
      Scaling grow with coverage helped the trade-off but still left artefacts
      around the ears on single-subject portraits.
  - Fill small background islands when coverage is already high: closes the
    corner island, does not address the wedge.

All of these have been reverted. The engine in this bundle is the committed
state plus a per-face teeth gate and an inert field-dump hook.

## What we are asking

Either of these is a good answer:

1. How to make the person segmentation robust for tightly-cropped portraits with
   two adjacent faces, where the subject fills the frame -- ideally without
   swapping in a much heavier model, though a recommendation of a specific model
   and how to integrate it is welcome.

2. Failing that, how to composite so that a segmentation error cannot read as a
   hole in a face -- e.g. deriving the base from the photograph everywhere
   inside a generous subject region, or a confidence-weighted blend, so a wrong
   mask degrades gracefully instead of stamping the ground onto skin.

Constraints: Python, OpenCV, MediaPipe, optional onnxruntime already present.
Render budget is roughly 20 seconds per portrait on CPU; a second or two more is
acceptable, ten is not.

## Relevant environment values on the failing render

    TYPO_SUBJECT_BASE=1.0     replace ground with dimmed photo inside the mask
    TYPO_SUBJECT_DIM=0.30
    TYPO_MATTE_MODEL=1        RVM enabled
    TYPO_MATTE_FLOOR=0.12
    TYPO_MATTE_GAMMA=1.5
    TYPO_EDGE_FALLOFF=0.45
    TYPO_INK_LIFT=1.62
    TYPO_INK_LIFT_ADD=28
    TYPO_HILIGHT_WASH=0.5
    TYPO_BG_LIGHTEN=0
"""


def main():
    if not os.path.isdir(TREE):
        sys.exit("tree not found: %s" % TREE)
    stage = tempfile.mkdtemp(prefix="bundle-")
    src_dir = os.path.join(stage, "source")
    ev_dir = os.path.join(stage, "evidence")
    os.makedirs(src_dir)
    os.makedirs(ev_dir)

    # --- source ---
    missing = []
    for rel in SOURCES:
        p = os.path.join(TREE, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        dst = os.path.join(src_dir, rel.replace("/", "__"))
        shutil.copy2(p, dst)
    if missing:
        print("WARNING missing source files: %s" % ", ".join(missing))

    # --- evidence ---
    def grab(src, name):
        if src and os.path.isfile(src):
            shutil.copy2(src, os.path.join(ev_dir, name))
            return True
        print("missing evidence: %s" % name)
        return False

    outputs = os.path.join(DATA, "outputs")
    srcs = sorted((os.path.join(outputs, f) for f in os.listdir(outputs)
                   if f.endswith("_src.jpg")),
                  key=os.path.getmtime) if os.path.isdir(outputs) else []
    grab(srcs[-1] if srcs else None, "src.jpg")
    grab(os.path.join(DATA, "marketing-src", "couple.png"), "original.png")
    for f, n in (("render.jpg", "render.jpg"), ("mask01.png", "mask01.png"),
                 ("soft01.png", "soft01.png"), ("alpha.png", "alpha.png"),
                 ("base.png", "base.png")):
        grab(os.path.join(outputs, "realsrc", f), n)

    open(os.path.join(stage, "BRIEFING.md"), "w", encoding="utf-8").write(BRIEFING)

    # --- git context, no credentials involved ---
    try:
        log = subprocess.check_output(
            ["git", "-C", "/root/typortrait-stg", "log", "--oneline", "-15",
             "--", "typography_engine/app"], text=True)
        open(os.path.join(stage, "recent-commits.txt"), "w").write(log)
    except Exception as e:  # noqa: BLE001
        print("git log unavailable: %s" % e)

    # --- secret scan before anything is written out ---
    hits = []
    for root, _d, files in os.walk(stage):
        for f in files:
            p = os.path.join(root, f)
            if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg"):
                continue
            try:
                t = open(p, encoding="utf-8", errors="ignore").read()
            except Exception:  # noqa: BLE001
                continue
            if SECRET.search(t):
                hits.append(os.path.relpath(p, stage))
    if hits:
        sys.exit("ABORTED -- possible credentials in: %s" % ", ".join(hits))
    print("secret scan: clean")

    os.makedirs(OUTDIR, exist_ok=True)
    zpath = os.path.join(OUTDIR, ZIPNAME)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _d, files in os.walk(stage):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, stage))
    shutil.rmtree(stage, ignore_errors=True)
    print("wrote %s  (%.1f MB)" % (zpath, os.path.getsize(zpath) / 1e6))
    print("https://typortrait.com/review/%s" % ZIPNAME)


if __name__ == "__main__":
    main()
