#!/usr/bin/env python3
"""Show WHERE the pet engine's matte finds the subject, band by band.

Written because a subject's neck was missing from renders and three rounds of
adjusting PET_MATTE_FILL changed nothing. Rather than keep guessing at a
threshold, this runs the real matte on the real photograph and prints what it
actually produced, so the question "is the neck in the mask at all?" has an
answer instead of a theory.

Reads the most recent stored source image, runs _foreground_mask() exactly as
render_pet_portrait() does, and reports for each horizontal band:

    coverage   fraction of the band the mask calls subject (alpha > 0.5)
    mean       mean alpha across the band
    max        the strongest alpha anywhere in the band

A neck that the model never saw shows near-zero across all three. A neck the
model saw but the threshold discarded shows a low mean with a max well above
it -- which would point at _solidify_matte rather than at the model.

It also prints the same bands for the RAW model output, before
_solidify_matte's largest-component selection and hole fill, so the two stages
can be told apart.

Reads one image and prints numbers. Writes nothing unless DUMP=<dir> is set, in
which case it also writes three pictures -- including one marking every pixel the
solidifier ADDED to the model's answer, which is where this stage can invent
subject out of background.

Usage (inside the container, data/ is bind-mounted so copy it there first):
    cp typography_engine/ops/pet-matte-probe.py /root/<tree>/typography_engine/data/
    docker exec <container> python /app/data/pet-matte-probe.py
    docker exec <container> python /app/data/pet-matte-probe.py /app/data/private/<job>.src
    docker exec -e DUMP=/app/data/mattedump <container> \
        python /app/data/pet-matte-probe.py /app/data/testset/05-couple.jpg
"""
import glob
import os
import sys

sys.path.insert(0, "/app")

import cv2                                     # noqa: E402
import numpy as np                             # noqa: E402

from app.pet_proto import (                    # noqa: E402
    _foreground_mask, _u2net_mask, _grabcut_mask, _MATTE_NAME, _MATTE_PATH,
)

BANDS = 12


def bands(m, label):
    h = m.shape[0]
    print("  %-22s %8s %8s %8s" % (label, "coverage", "mean", "max"))
    for i in range(BANDS):
        a, b = int(h * i / BANDS), int(h * (i + 1) / BANDS)
        seg = m[a:b]
        print("    rows %4d-%-4d %6.0f%%  %8.3f %8.3f %8.3f"
              % (a, b, 100.0 * (i + 0.5) / BANDS,
                 float((seg > 0.5).mean()), float(seg.mean()), float(seg.max())))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if not src:
        cands = sorted(glob.glob("/app/data/private/*.src"), key=os.path.getmtime, reverse=True)
        if not cands:
            raise SystemExit("no source images in /app/data/private")
        src = cands[0]
    print("source: %s" % src)

    bgr = cv2.imdecode(np.frombuffer(open(src, "rb").read(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit("could not decode %s" % src)
    print("size:   %dx%d" % (bgr.shape[1], bgr.shape[0]))

    # Match render_pet_portrait's working resolution, or the numbers describe a
    # different image than the one the renderer saw.
    work_h = int(os.environ.get("PROBE_H", "1050"))
    if bgr.shape[0] != work_h:
        bgr = cv2.resize(bgr, (max(1, int(bgr.shape[1] * work_h / bgr.shape[0])), work_h),
                         interpolation=cv2.INTER_AREA)
    print("worked: %dx%d   model=%s   present=%s"
          % (bgr.shape[1], bgr.shape[0], _MATTE_NAME, os.path.exists(_MATTE_PATH)))
    print("        PET_MATTE_FILL=%r" % os.environ.get("PET_MATTE_FILL"))
    print()

    raw = _u2net_mask(bgr)
    if raw is None:
        print("!! the model returned nothing -- this render used the GrabCut fallback,")
        print("   which is a rectangle-initialised approximation, not a matte.")
        raw = _grabcut_mask(bgr)
        bands(raw, "GrabCut fallback")
    else:
        bands(raw, "raw model output")
    print()
    solid = _foreground_mask(bgr)
    bands(solid, "after solidify")
    print()
    print("Read the lower bands: that is the neck and chest. Near-zero in BOTH tables")
    print("means the model never saw the neck. Low in the second but not the first")
    print("means _solidify_matte discarded it.")

    # DUMP=<dir> writes pictures as well as numbers. Bands answer "how much"; they cannot
    # answer "where", and the question that prompted this -- background between two people
    # coming out as subject -- is entirely about where. The overlay marks every pixel the
    # SOLIDIFIER added to the model's own answer, which is exactly the set of pixels that
    # can be wrong in that way: the largest-component pick, the hole fill, the torso fill.
    dd = os.environ.get("DUMP", "").strip()
    if dd:
        os.makedirs(dd, exist_ok=True)
        added = ((solid > 0.5) & (raw <= 0.5))
        cv2.imwrite(os.path.join(dd, "matte-raw.png"),
                    np.clip(raw * 255.0, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(dd, "matte-solid.png"),
                    np.clip(solid * 255.0, 0, 255).astype(np.uint8))
        ov = bgr.astype(np.float32).copy()
        ov[added] = ov[added] * 0.25 + np.array([60, 60, 235], np.float32) * 0.75
        cv2.imwrite(os.path.join(dd, "matte-added.png"),
                    np.clip(ov, 0, 255).astype(np.uint8))
        print()
        print("wrote %s" % dd)
        print("  matte-raw.png    what the model returned")
        print("  matte-solid.png  what the renderer used")
        print("  matte-added.png  RED = invented by _solidify_matte, not seen by the model")
        print("added %.2f%% of the frame   PET_MATTE_FILL=%r  PET_TORSO_FILL=%r"
              % (100.0 * float(added.mean()), os.environ.get("PET_MATTE_FILL"),
                 os.environ.get("PET_TORSO_FILL")))


if __name__ == "__main__":
    main()
