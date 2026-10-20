#!/usr/bin/env python3
"""Judge each subject's mouth on its own pixels, not on the union of all mouths.

THE BUG
  Every face's inner-mouth mask is merged into one `teeth` field, and only then
  is the open/closed appearance test applied -- to the combined region. So one
  subject's dark lip crease pulls the union's p10 below the "dark cavity"
  threshold and validates "open mouth" for EVERY face in the photo. Both mouths
  are then cleared of type and render as flat pale shapes over closed lips.

  Measured on a two-subject portrait with both mouths shut:

      union            p10=54.0  -> KEPT      (blobs on both faces)
      same photo, my   p10=62.8  -> closed    (no blobs)
      harness framing

  The verdict flipped on framing alone, which is what a pooled statistic does.

THE FIX
  Evaluate the gate per face, inside the loop, before merging. A face whose own
  mouth reads closed is dropped and contributes nothing to the union. Identical
  behaviour for single-subject photos; only multi-subject renders change.

  The debug line now reports per face, so a future false positive can be traced
  to the subject that caused it.

Usage:  python3 teeth-per-face.py <tree>/typography_engine
Idempotent; aborts without writing if the anchors are not found exactly once.
Requires teeth-thresholds.py to have run first (it introduces the env vars).
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
DISP = os.path.join(ROOT, "app/pipeline/displacement.py")

START = "    teeth = None                       # union every subject's open-mouth teeth region\n"
END = "                teeth = None                           # no cavity, no teeth -> closed mouth\n"

NEW = '''    # Per-face gate. Merging first and testing the union let one subject's dark
    # lip crease validate "open mouth" for every face in the photo -- two closed
    # mouths rendered as two pale blobs. Each face is now judged on its own
    # pixels and dropped before it can contribute to the union.
    _tdark = float(os.environ.get("TYPO_TEETH_DARK", "60.0") or 60.0)
    _tbright = float(os.environ.get("TYPO_TEETH_BRIGHT", "205.0") or 205.0)
    _tdbg = os.environ.get("TYPO_TEETH_DEBUG", "").strip().lower() in ("1", "true", "on", "yes")
    teeth = None                       # union of the mouths that really are open
    for _fi, _fp in enumerate(all_pts):
        _tm = _teeth_mask(_fp, H, W)
        if _tm is None:
            continue
        # A REAL open mouth has a dark cavity OR genuinely bright teeth; a
        # falsely-detected one is uniform lip tone.
        _tpx = gray[_tm > 0.5]
        if _tpx.size > 10:
            _p10 = float(np.percentile(_tpx, 10))
            _p90 = float(np.percentile(_tpx, 90))
            _closed = (_p10 > _tdark and _p90 < _tbright)
            if _tdbg:
                try:
                    print("[teeth] face=%d p10=%.1f p90=%.1f dark<=%.1f bright>=%.1f -> %s"
                          % (_fi, _p10, _p90, _tdark, _tbright,
                             "closed" if _closed else "KEPT"))
                except Exception:
                    pass
            if _closed:
                continue
        teeth = _tm if teeth is None else np.maximum(teeth, _tm)
'''


def main():
    if not os.path.isfile(DISP):
        raise SystemExit("no such file: %s" % DISP)
    src = open(DISP, encoding="utf-8").read()

    if "face=%d p10" in src:
        print("already patched -- no change")
        return
    if "TYPO_TEETH_DARK" not in src:
        raise SystemExit("ABORTED: run teeth-thresholds.py first")
    for name, anchor in (("start", START), ("end", END)):
        if src.count(anchor) != 1:
            raise SystemExit("ABORTED: %s anchor found %d times, expected 1"
                             % (name, src.count(anchor)))

    i = src.index(START)
    j = src.index(END, i) + len(END)
    out = src[:i] + NEW + src[j:]
    compile(out, DISP, "exec")
    shutil.copy2(DISP, DISP + ".bak-perface")
    open(DISP, "w", encoding="utf-8").write(out)
    print("patched OK   (backup: %s.bak-perface)" % DISP)
    print("SYNTAX OK")


if __name__ == "__main__":
    main()
