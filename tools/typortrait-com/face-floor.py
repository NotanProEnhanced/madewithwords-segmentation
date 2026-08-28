#!/usr/bin/env python3
"""A detected face can never be background.

THE BUG
  On a two-subject portrait the segmenter classified a wedge of one man's
  forehead, temple and cheek as background. The silhouette mask is white almost
  everywhere and black over that wedge, so the studio ground is painted across
  his face and he fades out -- the "white blotches" that survived every
  downstream explanation (teeth mask, ink lift, matte model, edge falloff,
  highlight wash). None of those were involved; the mask was already wrong
  before any of them ran.

THE FIX
  MediaPipe positively detected and landmarked that face -- the engine draws its
  eyes and lips from the same 478-point mesh it is ignoring here. So union the
  silhouette with each detected face's convex hull before anything downstream
  uses it.

  The hull is grown by TYPO_FACE_FLOOR_GROW (default 18%) because the mesh stops
  short of the forehead and under-jaw, which is exactly where this failure bit.

  This only ever ADDS foreground, and only where a face was positively found, so
  it cannot eat background or damage a photo that already segments correctly.
  A manual (user-painted) mask is left alone -- that is the user's own intent.

  TYPO_FACE_FLOOR=0 disables it without a rebuild.

Usage:  python3 face-floor.py <tree>/typography_engine
Idempotent; aborts without writing if the anchor is missing or a needed import
is absent (it reports which, rather than adding imports behind your back).
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
PATH = os.path.join(ROOT, "app/pipeline/analyze.py")

ANCHOR = "    edges = detect_edges(img, warns, cfg.canny_low, cfg.canny_high, mask=sil.mask)\n"

NEW = '''    # A detected face can never be background. The segmenter sometimes classifies
    # part of a face as background -- on a two-subject portrait it removed a wedge
    # of one man's forehead and cheek -- and the ground is then painted across it,
    # so the subject fades out. MediaPipe found and landmarked that face; the mesh
    # is trusted for eyes and lips, so trust it here too.
    #
    # Only ever ADDS foreground, and only inside a positively detected face. A
    # manual mask is left alone: that is the user's own intent.
    # TYPO_FACE_FLOOR=0 disables.
    if (faces and manual_mask is None
            and os.environ.get("TYPO_FACE_FLOOR", "1").strip().lower()
            not in ("0", "false", "off", "no", "")):
        _grow = float(os.environ.get("TYPO_FACE_FLOOR_GROW", "0.18") or 0.18)
        _hull = np.zeros((img.h, img.w), np.uint8)
        for _f in faces:
            _p = np.asarray(getattr(_f, "points", None), np.float32)
            if _p.ndim != 2 or _p.shape[0] < 3:
                continue
            _c = _p.mean(axis=0)
            _p = (_c + (_p - _c) * (1.0 + _grow)).astype(np.int32)
            cv2.fillConvexPoly(_hull, cv2.convexHull(_p), 255)
        if _hull.any():
            _m = np.maximum(sil.mask, _hull)
            _s = np.maximum(sil.soft, _hull) if sil.soft is not None else None
            _added = float((_m > 127).mean() - (sil.mask > 127).mean())
            if os.environ.get("TYPO_FACE_FLOOR_DEBUG", "").strip().lower() in ("1", "true", "on", "yes"):
                try:
                    print("[facefloor] faces=%d grow=%.2f added=%.3f%% of frame"
                          % (len(faces), _grow, 100.0 * _added))
                except Exception:
                    pass
            sil = Silhouette(mask=_m, bbox=sil.bbox,
                             coverage=float((_m > 127).mean()),
                             confidence=sil.confidence, soft=_s)
'''

NEEDED = ("import os", "import cv2", "import numpy as np")


def main():
    if not os.path.isfile(PATH):
        raise SystemExit("no such file: %s" % PATH)
    src = open(PATH, encoding="utf-8").read()

    if "TYPO_FACE_FLOOR" in src:
        print("already patched -- no change")
        return

    head = src.splitlines()[:40]
    missing = [m for m in NEEDED
               if not any(line.strip() == m or line.strip().startswith(m + " ")
                          for line in head)]
    if missing:
        raise SystemExit("ABORTED: analyze.py is missing module-scope imports: %s\n"
                         "Add them by hand -- a function-local import would rebind "
                         "the name for the whole function." % ", ".join(missing))

    if src.count(ANCHOR) != 1:
        raise SystemExit("ABORTED: anchor found %d times, expected 1" % src.count(ANCHOR))

    out = src.replace(ANCHOR, NEW + ANCHOR, 1)
    compile(out, PATH, "exec")
    shutil.copy2(PATH, PATH + ".bak-facefloor")
    open(PATH, "w", encoding="utf-8").write(out)
    print("patched OK   (backup: %s.bak-facefloor)" % PATH)
    print("SYNTAX OK")


if __name__ == "__main__":
    main()
