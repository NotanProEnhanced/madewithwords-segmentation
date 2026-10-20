#!/usr/bin/env python3
"""Combine both segmenters, and let the mask err on the side of the subject.

THE DEFECT
  The segmenter cuts into hair. On a two-subject portrait it removed a wedge of
  one man's temple and hairline; on another it took the top of a woman's head.
  Since the backdrop started defaulting to studio grey, those holes are painted
  light and read as blobs on the face. The mask is simply wrong there -- proven
  by dumping it -- so nothing downstream can recover it:

    TYPO_MATTE_FLOOR=0, TYPO_MATTE_GAMMA=1.0   no change
    TYPO_MATTE_MODEL on and off                no change
    TYPO_EDGE_FALLOFF=0                        no change
    face-hull floor                            wedge is outside the hull

TWO CHANGES, BOTH GATED, BOTH DEFAULT OFF

  TYPO_MATTE_UNION=1
    extract_silhouette returns the FIRST segmenter that works: RVM if enabled,
    else MediaPipe. It never combines them, though they fail in different
    places. This takes the per-pixel maximum of both, so a region either model
    is confident about survives.

  TYPO_MASK_GROW=<fraction of min(w,h)>
    Dilates the result. A mask that is slightly too generous shows a rim of the
    original background; one that is too tight puts a hole in a face. The first
    is a blemish, the second is unusable, so the bias should not be symmetric.
    0.004 is a sensible starting point.

Usage:  python3 matte-union.py <tree>/typography_engine
Idempotent; aborts without writing if either anchor is not found exactly once.
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
PATH = os.path.join(ROOT, "app/pipeline/silhouette.py")

HELPER = '''
def _mask_grow(binm, soft):
    """Dilate mask and soft alpha by TYPO_MASK_GROW (fraction of min(w,h)).

    An over-generous mask shows a rim of original background; a tight one puts a
    hole in a face. Those costs are not symmetric, so the bias is not either.
    0 (default) leaves both untouched.
    """
    try:
        g = float(os.environ.get("TYPO_MASK_GROW", "0") or 0.0)
    except Exception:  # noqa: BLE001
        return binm, soft
    if g <= 0.0:
        return binm, soft
    h, w = binm.shape[:2]
    r = max(1, int(round(min(w, h) * g)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    binm = cv2.dilate(binm, k)
    if soft is not None:
        soft = cv2.dilate(soft, k)
    if os.environ.get("TYPO_MASK_DEBUG", "").strip().lower() in ("1", "true", "on", "yes"):
        try:
            print("[mask] grow=%.4f radius=%dpx coverage=%.4f"
                  % (g, r, float((binm > 127).mean())))
        except Exception:  # noqa: BLE001
            pass
    return binm, soft


'''

RVM_OLD = """                bbox = _bbox_of(binm)
                conf = max(0.9, _confidence(binm, bbox, cov))
                return Silhouette(mask=binm, bbox=bbox, coverage=cov, confidence=conf, soft=soft)
"""

RVM_NEW = """                # The two segmenters fail in different places -- RVM and MediaPipe
                # each cut hair the other keeps. Returning whichever ran first
                # throws that away. Union recovers any region either is confident
                # about. TYPO_MATTE_UNION=1 enables; default is prior behaviour.
                if os.environ.get("TYPO_MATTE_UNION", "").strip().lower() in ("1", "true", "on", "yes"):
                    _mp_bin, _mp_soft = _selfie_mask(img, warns)
                    if _mp_bin is not None:
                        _before = float((binm > 127).mean())
                        binm = np.maximum(binm, _mp_bin)
                        if soft is not None and _mp_soft is not None:
                            soft = np.maximum(soft, _mp_soft)
                        elif soft is None:
                            soft = _mp_soft
                        cov = float((binm > 127).sum()) / float(h * w)
                        if os.environ.get("TYPO_MASK_DEBUG", "").strip().lower() in ("1", "true", "on", "yes"):
                            try:
                                print("[mask] union rvm=%.4f -> %.4f of frame" % (_before, cov))
                            except Exception:  # noqa: BLE001
                                pass
                binm, soft = _mask_grow(binm, soft)
                cov = float((binm > 127).sum()) / float(h * w)
                bbox = _bbox_of(binm)
                conf = max(0.9, _confidence(binm, bbox, cov))
                return Silhouette(mask=binm, bbox=bbox, coverage=cov, confidence=conf, soft=soft)
"""

MP_OLD = """    selfie, soft = _selfie_mask(img, warns)
    if selfie is not None:
        bbox = _bbox_of(selfie)
        coverage = float((selfie > 127).sum()) / float(h * w)
"""

MP_NEW = """    selfie, soft = _selfie_mask(img, warns)
    if selfie is not None:
        selfie, soft = _mask_grow(selfie, soft)
        bbox = _bbox_of(selfie)
        coverage = float((selfie > 127).sum()) / float(h * w)
"""

ANCHOR_DEF = "def extract_silhouette(\n"


def die(msg):
    raise SystemExit("ABORTED (nothing written): " + msg)


def main():
    if not os.path.isfile(PATH):
        die("no such file: %s" % PATH)
    src = open(PATH, encoding="utf-8").read()

    if "TYPO_MATTE_UNION" in src:
        print("already patched -- no change")
        return

    for name, anchor in (("rvm return", RVM_OLD), ("mediapipe return", MP_OLD),
                         ("extract_silhouette def", ANCHOR_DEF)):
        if src.count(anchor) != 1:
            die("%s found %d times, expected 1" % (name, src.count(anchor)))

    out = src.replace(ANCHOR_DEF, HELPER.lstrip("\n") + ANCHOR_DEF, 1)
    out = out.replace(RVM_OLD, RVM_NEW, 1)
    out = out.replace(MP_OLD, MP_NEW, 1)
    compile(out, PATH, "exec")
    shutil.copy2(PATH, PATH + ".bak-union")
    open(PATH, "w", encoding="utf-8").write(out)
    print("patched OK   (backup: %s.bak-union)" % PATH)
    print("SYNTAX OK -- both changes default OFF")


if __name__ == "__main__":
    main()
