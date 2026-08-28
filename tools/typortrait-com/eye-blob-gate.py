#!/usr/bin/env python3
"""Stop fabricating dark eyeballs when the irises cannot be resolved.

THE DEFECT
  A group photo rendered with solid dark discs over every subject's eyes.

  Measured: with TYPO_EYE_DEBUG=1 the "[eye]" line never printed at all. That
  print sits AFTER the per-face iris loop, so its absence proves fewer than two
  irises resolved on every face and the loop `continue`d before any eye analysis
  ran. With `irises` empty, `if not irises:` fires and the legacy fallback paints
  an ink blob at each lid centroid -- the dark circles.

  The gate is `if ir >= 8.0 * _ssn`: an iris must measure at least 8px. Three
  people in one 1024x819 frame (upscaled from smaller, per the render warning)
  puts every iris under that.

  It is not the sunglasses path -- that only runs on an explicit toggle -- and
  not skin tone, though the artefact is most obvious on darker skin because the
  disc blends into the face less kindly.

TWO CHANGES

  TYPO_IRIS_MIN_PX   the 8.0px gate, exposed. Lowering it lets small faces
                     resolve real irises instead of falling back at all.

  TYPO_EYE_BLOB      whether the legacy blob is drawn when no iris resolved.
                     Default 1 = current behaviour. Set 0 and an unresolvable
                     eye simply renders as typography, which is what the rest of
                     the face does. A portrait with no drawn eyeball reads as a
                     stylistic choice; a dark disc over someone's eye reads as a
                     defect.

Usage:  python3 eye-blob-gate.py <tree>/typography_engine
Idempotent; aborts without writing if either anchor is not found exactly once.
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
PATH = os.path.join(ROOT, "app/pipeline/displacement.py")

IRIS_OLD = "            if ir >= 8.0 * _ssn:\n"
IRIS_NEW = ('            # Minimum iris radius for a face to get real eyes. Below this the\n'
            '            # face is skipped entirely and the legacy blob fallback takes over,\n'
            '            # which paints a dark disc. Small faces in a group photo sit under\n'
            '            # the original fixed 8.0. TYPO_IRIS_MIN_PX exposes it.\n'
            '            if ir >= float(os.environ.get("TYPO_IRIS_MIN_PX", "8.0") or 8.0) * _ssn:\n')

BLOB_OLD = "    if not irises:\n"
BLOB_NEW = ('    # Legacy blob fallback. When no iris resolved, this draws an ink disc at each\n'
            '    # lid centroid. On small faces that is a dark circle over a real eye -- worse\n'
            '    # than drawing nothing, since the rest of the face is already typography.\n'
            '    # TYPO_EYE_BLOB=0 renders those eyes as words instead. Default 1 = unchanged.\n'
            '    _eye_blob = os.environ.get("TYPO_EYE_BLOB", "1").strip().lower() \\\n'
            '        not in ("0", "false", "off", "no")\n'
            '    if not irises and _eye_blob:\n')


def die(msg):
    raise SystemExit("ABORTED (nothing written): " + msg)


def main():
    if not os.path.isfile(PATH):
        die("no such file: %s" % PATH)
    src = open(PATH, encoding="utf-8").read()

    if "TYPO_EYE_BLOB" in src:
        print("already patched -- no change")
        return
    for name, anchor in (("iris gate", IRIS_OLD), ("blob gate", BLOB_OLD)):
        if src.count(anchor) != 1:
            die("%s found %d times, expected 1" % (name, src.count(anchor)))

    out = src.replace(IRIS_OLD, IRIS_NEW, 1).replace(BLOB_OLD, BLOB_NEW, 1)
    compile(out, PATH, "exec")
    shutil.copy2(PATH, PATH + ".bak-eyeblob")
    open(PATH, "w", encoding="utf-8").write(out)
    print("patched OK   (backup: %s.bak-eyeblob)" % PATH)
    print("SYNTAX OK -- behaviour unchanged at default values")


if __name__ == "__main__":
    main()
