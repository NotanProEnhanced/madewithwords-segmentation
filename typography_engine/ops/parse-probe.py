#!/usr/bin/env python3
"""What does MediaPipe's multiclass segmenter actually see in a photograph?

WHY
  The engine currently infers "where is the face, the hair, the clothing" from a Laplacian
  detail map -- a stand-in that needs five separate knobs to correct it (PET_HERO_CENTRE,
  PET_FEATURE_PROTECT, PET_DRAPE_DETAIL_DAMP, PET_FEATURE_SCOPE, PET_TIER_GAMMA). A model
  that labels those regions directly would replace the stand-in.

  Before building anything on that idea, this answers the only question that matters: does
  the model label THESE photographs usefully? It reports the share of each class and where
  each one sits, so a claim like "a hat is its own region" is checked rather than assumed.

  16MB, and MediaPipe is already a dependency -- the human engine uses its selfie segmenter
  and face mesh -- so this costs no new runtime.

CLASSES
  0 background   1 hair   2 body-skin   3 face-skin   4 clothes   5 others/accessory

  There is no explicit "hat" class. A hat lands in `others` or is absorbed into `hair`, and
  which one it is decides whether hats can be treated deliberately. That is what the
  per-image breakdown below is for.

WRITES NOTHING. Reads images and prints numbers.

Usage (inside the container, data/ is bind-mounted so copy it there first):
    cp typography_engine/ops/parse-probe.py /root/typortrait-stg/typography_engine/data/
    docker exec typortrait-staging python /app/data/parse-probe.py /root/typortrait-testset/src
    docker exec typortrait-staging python /app/data/parse-probe.py <one-image.jpg>
"""
import glob
import os
import sys
import urllib.request

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
MODEL = os.environ.get("PARSE_MODEL",
                       "/app/data/pet-models/selfie_multiclass_256x256.tflite")
NAMES = {0: "background", 1: "hair", 2: "body-skin", 3: "face-skin",
         4: "clothes", 5: "others/accessory"}


def ensure_model():
    """Same atomic fetch as the matte and depth models: a partly-written file must never be
    visible at the final path, or something will load it and fail in a way that latches."""
    if os.path.exists(MODEL) and os.path.getsize(MODEL) > 5_000_000:
        return MODEL
    d = os.path.dirname(MODEL) or "."
    os.makedirs(d, exist_ok=True)
    tmp = MODEL + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if os.path.getsize(tmp) < 5_000_000:
        os.remove(tmp)
        raise SystemExit("download too small -- got a error page rather than the model?")
    os.replace(tmp, MODEL)
    return MODEL


def main():
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    target = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-testset/src"
    files = sorted(glob.glob(os.path.join(target, "*.jpg")) +
                   glob.glob(os.path.join(target, "*.png"))) \
        if os.path.isdir(target) else [target]
    if not files:
        raise SystemExit("no images at %s" % target)

    path = ensure_model()
    print("model: %s (%.1f MB)\n" % (path, os.path.getsize(path) / 1e6))

    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mpp.BaseOptions(model_asset_path=path), output_category_mask=True))
    try:
        hdr = "%-22s" % "image" + "".join("%12s" % NAMES[k][:11] for k in range(6))
        print(hdr)
        for f in files:
            bgr = cv2.imread(f)
            if bgr is None:
                print("%-22s could not read" % os.path.basename(f))
                continue
            h = 900
            bgr = cv2.resize(bgr, (max(1, int(bgr.shape[1] * h / bgr.shape[0])), h),
                             interpolation=cv2.INTER_AREA)
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cat = seg.segment(img).category_mask.numpy_view()
            row = "%-22s" % os.path.basename(f)
            for k in range(6):
                row += "%11.2f%%" % (100.0 * float((cat == k).mean()))
            print(row)

            # Where a class sits matters as much as how much of it there is: an accessory
            # region at the TOP of the frame is plausibly a hat; the same share low down is
            # a collar or a necklace.
            for k in (5,):
                m = (cat == k)
                if m.mean() > 0.005:
                    ys = np.nonzero(m.any(axis=1))[0]
                    print("%-22s   %s spans rows %d-%d of %d (%s of the frame)"
                          % ("", NAMES[k], ys.min(), ys.max(), cat.shape[0],
                             "upper" if ys.mean() < cat.shape[0] * 0.4 else "lower/mixed"))
    finally:
        # Closed explicitly: letting the interpreter tear it down at exit raises a confusing
        # TypeError from MediaPipe's destructor that looks like a failure and is not.
        try:
            seg.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
