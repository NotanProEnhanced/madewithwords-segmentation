#!/usr/bin/env python3
"""Scale the face-hull floor with how much of the frame the subject occupies.

MEASURED
  A fixed hull cannot serve both cases:

    frame-filling crop (coverage 0.94)   grow 0.18 leaves the wedge
                                         grow 0.60 fixes it
    normal portrait   (coverage 0.63)    grow 0.60 adds 9.6% of the frame and
                                         paints a grey oval of real background
                                         around the head

  The failure only occurs when the subject already fills the frame -- that is
  where the segmenter's confidence collapses and where there is little genuine
  background to lose. A normal portrait needs almost no help and has plenty to
  lose.

SO
  grow interpolates on coverage:

    coverage <= TYPO_FACE_FLOOR_COV_LO (0.80)   ->  TYPO_FACE_FLOOR_GROW (0.18)
    coverage >= TYPO_FACE_FLOOR_COV_HI (0.95)   ->  TYPO_FACE_FLOOR_GROW_MAX (0.60)
    between                                     ->  linear

  Setting GROW_MAX equal to GROW restores fixed behaviour.

Usage:  python3 face-floor-adaptive.py <tree>/typography_engine
Idempotent; aborts without writing if the anchor is not found exactly once.
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
PATH = os.path.join(ROOT, "app/pipeline/analyze.py")

OLD = '        _grow = float(os.environ.get("TYPO_FACE_FLOOR_GROW", "0.18") or 0.18)\n'

NEW = '''        # Grow scales with coverage. A fixed value cannot serve both cases: a
        # frame-filling crop needs a generous hull to repair the segmenter's
        # collapse, while a normal portrait with real background gets a grey
        # oval of that background painted in as subject.
        _g_lo = float(os.environ.get("TYPO_FACE_FLOOR_GROW", "0.18") or 0.18)
        _g_hi = float(os.environ.get("TYPO_FACE_FLOOR_GROW_MAX", "0.60") or 0.60)
        _c_lo = float(os.environ.get("TYPO_FACE_FLOOR_COV_LO", "0.80") or 0.80)
        _c_hi = float(os.environ.get("TYPO_FACE_FLOOR_COV_HI", "0.95") or 0.95)
        _cov = float(sil.coverage)
        _t = 0.0 if _c_hi <= _c_lo else (_cov - _c_lo) / (_c_hi - _c_lo)
        _t = min(1.0, max(0.0, _t))
        _grow = _g_lo + (_g_hi - _g_lo) * _t
'''


def main():
    if not os.path.isfile(PATH):
        raise SystemExit("no such file: %s" % PATH)
    src = open(PATH, encoding="utf-8").read()

    if "TYPO_FACE_FLOOR_GROW_MAX" in src:
        print("already patched -- no change")
        return
    if src.count(OLD) != 1:
        raise SystemExit("ABORTED: anchor found %d times, expected 1" % src.count(OLD))

    out = src.replace(OLD, NEW, 1)
    # report the coverage that drove the choice
    out = out.replace(
        'print("[facefloor] faces=%d grow=%.2f added=%.3f%% of frame"\n'
        '                          % (len(faces), _grow, 100.0 * _added))',
        'print("[facefloor] faces=%d coverage=%.3f grow=%.2f added=%.3f%% of frame"\n'
        '                          % (len(faces), _cov, _grow, 100.0 * _added))', 1)
    compile(out, PATH, "exec")
    shutil.copy2(PATH, PATH + ".bak-adaptive")
    open(PATH, "w", encoding="utf-8").write(out)
    print("patched OK   (backup: %s.bak-adaptive)" % PATH)
    print("SYNTAX OK")


if __name__ == "__main__":
    main()
