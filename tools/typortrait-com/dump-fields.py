#!/usr/bin/env python3
"""Dump the three fields that fully determine the composite.

The render ends with:

    out = _base * (1 - al) + ink_col * al

so every pixel is decided by _base (the ground, or the source photo inside the
silhouette), al (ink alpha), and ink_col. Eleven hypotheses have been proposed
and eliminated by inference; these three images end that. Whichever field is
wrong in the pale regions is the cause, visibly.

Set TYPO_DUMP_FIELDS to a directory and the next render writes:

    mask01.png  binary silhouette at render resolution
    soft01.png  feathered alpha actually used for compositing
    alpha.png   ink alpha (al) -- where type is laid down
    base.png    the base being composited against

The contradiction to resolve: the mask reports 97% foreground, yet the pale
regions take their colour from the ground. Both cannot be true; one of these
four images will show which.

Usage:  python3 dump-fields.py <tree>/typography_engine
Idempotent; aborts without writing if the anchor is not found exactly once.
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
PATH = os.path.join(ROOT, "app/pipeline/displacement.py")

ANCHOR = "        out = _base * (1 - al) + ink_col * al\n"

NEW = '''        # Field dump (TYPO_DUMP_FIELDS=<dir>). The composite below is fully
        # determined by _base, al and ink_col, so when a render is wrong the
        # answer is in one of them -- no inference required.
        _dd = os.environ.get("TYPO_DUMP_FIELDS", "").strip()
        if _dd:
            try:
                os.makedirs(_dd, exist_ok=True)

                def _dump(_nm, _arr):
                    _a = np.asarray(_arr, np.float32)
                    if _a.ndim == 3 and _a.shape[2] == 1:
                        _a = _a[..., 0]
                    if float(_a.max()) <= 1.001:
                        _a = _a * 255.0
                    cv2.imwrite(os.path.join(_dd, _nm + ".png"),
                                np.clip(_a, 0, 255).astype(np.uint8))

                _dump("mask01", mask01)
                _dump("soft01", soft01)
                _dump("alpha", al)
                _dump("base", _base)
                _a1 = np.asarray(al, np.float32)
                if _a1.ndim == 3:
                    _a1 = _a1[..., 0]
                print("[dump] %s  alpha mean=%.3f p95=%.3f   soft01 mean=%.3f   "
                      "base mean=%.1f" % (_dd, float(_a1.mean()),
                                          float(np.percentile(_a1, 95)),
                                          float(np.asarray(soft01, np.float32).mean()),
                                          float(np.asarray(_base, np.float32).mean())))
            except Exception as _e:  # noqa: BLE001
                print("[dump] failed: %s" % _e)
'''


def main():
    if not os.path.isfile(PATH):
        raise SystemExit("no such file: %s" % PATH)
    src = open(PATH, encoding="utf-8").read()

    if "TYPO_DUMP_FIELDS" in src:
        print("already patched -- no change")
        return
    if src.count(ANCHOR) != 1:
        raise SystemExit("ABORTED: anchor found %d times, expected 1" % src.count(ANCHOR))

    out = src.replace(ANCHOR, NEW + ANCHOR, 1)
    compile(out, PATH, "exec")
    shutil.copy2(PATH, PATH + ".bak-dump")
    open(PATH, "w", encoding="utf-8").write(out)
    print("patched OK   (backup: %s.bak-dump)" % PATH)
    print("SYNTAX OK -- inert unless TYPO_DUMP_FIELDS is set")


if __name__ == "__main__":
    main()
