#!/usr/bin/env python3
"""Fill small background islands in the silhouette.

WHERE THIS CAME FROM
  A cropped two-subject upload segmented badly: a wedge over one man's hair and
  temple, and a smaller island in the bottom-left corner. Both were painted with
  the studio ground and read as pale blobs on the portrait.

  The face-hull floor (TYPO_FACE_FLOOR_GROW=0.6) fixes the wedge -- it is inside
  a detected face's neighbourhood. It cannot fix the corner island, which sits
  below both faces where no hull reaches.

THE RULE
  On a portrait whose subject fills the frame, a SMALL patch of background is
  almost certainly a segmentation error, wherever it is. Large regions are left
  alone -- those are real background.

    TYPO_MASK_FILL_SMALL   fill background components smaller than this
                           fraction of the frame (default 0 = off).
                           0.02 fills the corner island; a genuine background
                           is far larger and untouched.
    TYPO_MASK_FILL_MAXCOV  only apply when coverage already exceeds this
                           (default 0.80), so a normal portrait with real
                           background is never affected.

  Applied to mask and soft alike, after the face floor, so everything
  downstream sees one consistent silhouette.

Usage:  python3 fill-small-islands.py <tree>/typography_engine
Idempotent; aborts without writing if the anchor is not found exactly once.
Requires face-floor.py to have run first (it shares the same insertion point).
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
PATH = os.path.join(ROOT, "app/pipeline/analyze.py")

ANCHOR = "    edges = detect_edges(img, warns, cfg.canny_low, cfg.canny_high, mask=sil.mask)\n"

NEW = '''    # Small background islands inside a frame-filling subject are segmentation
    # errors, not background. The face-hull floor above cannot reach them -- the
    # one that prompted this sat in a corner below both faces -- and once the
    # backdrop is a light grey they read as pale blobs on the portrait.
    _fs = float(os.environ.get("TYPO_MASK_FILL_SMALL", "0") or 0.0)
    _fc = float(os.environ.get("TYPO_MASK_FILL_MAXCOV", "0.80") or 0.80)
    if _fs > 0.0 and manual_mask is None and float(sil.coverage) >= _fc:
        _bg = (sil.mask <= 127).astype(np.uint8)
        _n, _lab, _st, _ = cv2.connectedComponentsWithStats(_bg, 8)
        _frame = float(img.w * img.h)
        _fill = np.zeros_like(_bg)
        _filled = 0
        for _i in range(1, _n):
            if _st[_i][4] / _frame < _fs:
                _fill[_lab == _i] = 1
                _filled += 1
        if _filled:
            _m = np.maximum(sil.mask, _fill * 255)
            _s = np.maximum(sil.soft, _fill * 255) if sil.soft is not None else None
            if os.environ.get("TYPO_MASK_DEBUG", "").strip().lower() in ("1", "true", "on", "yes"):
                try:
                    print("[mask] filled %d island(s) < %.3f of frame; coverage %.4f -> %.4f"
                          % (_filled, _fs, float(sil.coverage), float((_m > 127).mean())))
                except Exception:  # noqa: BLE001
                    pass
            sil = Silhouette(mask=_m, bbox=sil.bbox,
                             coverage=float((_m > 127).mean()),
                             confidence=sil.confidence, soft=_s)
'''


def main():
    if not os.path.isfile(PATH):
        raise SystemExit("no such file: %s" % PATH)
    src = open(PATH, encoding="utf-8").read()

    if "TYPO_MASK_FILL_SMALL" in src:
        print("already patched -- no change")
        return
    if "TYPO_FACE_FLOOR" not in src:
        raise SystemExit("ABORTED: run face-floor.py first")
    if src.count(ANCHOR) != 1:
        raise SystemExit("ABORTED: anchor found %d times, expected 1" % src.count(ANCHOR))

    out = src.replace(ANCHOR, NEW + ANCHOR, 1)
    compile(out, PATH, "exec")
    shutil.copy2(PATH, PATH + ".bak-islands")
    open(PATH, "w", encoding="utf-8").write(out)
    print("patched OK   (backup: %s.bak-islands)" % PATH)
    print("SYNTAX OK -- off unless TYPO_MASK_FILL_SMALL is set")


if __name__ == "__main__":
    main()
