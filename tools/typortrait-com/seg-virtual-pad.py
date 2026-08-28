#!/usr/bin/env python3
"""Give the segmenter back the context the crop removed.

THE EVIDENCE
  Same photograph, same renderer, same settings:

    uncropped original   -> clean silhouette
    app's cropped upload -> a wedge of the subject classified as background

  Both RVM and MediaPipe make the same error, so it is not a quirk of one model.
  Both are semantic portrait models: a frame-filling crop tells them "nearly
  everything here is person", which is exactly when their person-vs-environment
  prior is weakest.

THE FIX
  Segment a PADDED copy, then crop the alpha back. The uploaded image, the
  render geometry and the typography are untouched -- only the classifier's
  field of view changes. It invents nothing; it re-establishes the context the
  models were trained to rely on.

  The border is BORDER_REPLICATE and then heavily blurred, so the synthetic
  surround carries no sharp artificial edge for the model to latch onto. The
  real pixels are pasted back over the centre unblurred.

    TYPO_SEG_PAD    fraction of width/height to pad on each side.
                    0 (default) = off. Suggested starting value 0.22-0.25.

Usage:  python3 seg-virtual-pad.py <tree>/typography_engine
Idempotent; aborts without writing if any anchor is not found exactly once.
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
PATH = os.path.join(ROOT, "app/pipeline/silhouette.py")

HELPERS = '''
def _pad_for_segmentation(img, frac):
    """A padded copy of the image for SEGMENTATION ONLY.

    A tight crop starves a semantic portrait model of the context it needs to
    separate person from environment -- measured: the same photo segments
    cleanly uncropped and badly cropped, in two unrelated model families.
    Replicate-pad, blur the synthetic surround so it presents no false edge,
    then paste the real pixels back over the centre.

    Returns (padded_image, box) where box is (px, py, w, h) for _unpad.
    """
    h, w = img.bgr.shape[:2]
    px = max(1, int(round(w * frac)))
    py = max(1, int(round(h * frac)))
    canvas = cv2.copyMakeBorder(img.bgr, py, py, px, px, cv2.BORDER_REPLICATE)
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=max(3.0, max(w, h) * 0.025))
    canvas[py:py + h, px:px + w] = img.bgr
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    try:
        import dataclasses as _dc
        padded = _dc.replace(img, bgr=canvas, gray=gray)
    except Exception:  # noqa: BLE001 -- not a dataclass, or frozen oddly
        import copy as _copy
        padded = _copy.copy(img)
        try:
            padded.bgr = canvas
            padded.gray = gray
        except Exception:  # noqa: BLE001
            return img, None
    return padded, (px, py, w, h)


def _unpad(arr, box):
    """Crop a padded segmentation result back to the original frame."""
    if arr is None or box is None:
        return arr
    px, py, w, h = box
    if arr.shape[0] < py + h or arr.shape[1] < px + w:
        return arr
    return arr[py:py + h, px:px + w]


'''

DEF_ANCHOR = "def extract_silhouette(\n"

HEAD_OLD = """    h, w = img.bgr.shape[:2]

    # Best (opt-in): a real matting model -> true strand-level hair alpha. Only runs when
"""

HEAD_NEW = """    h, w = img.bgr.shape[:2]

    # Segmentation-only virtual uncrop. The uploaded image and the render are
    # untouched; this only widens what the classifier sees. Off unless set.
    _segpad = float(os.environ.get("TYPO_SEG_PAD", "0") or 0.0)
    _seg_img, _seg_box = (img, None)
    if _segpad > 0.0:
        _seg_img, _seg_box = _pad_for_segmentation(img, _segpad)
        if os.environ.get("TYPO_MASK_DEBUG", "").strip().lower() in ("1", "true", "on", "yes"):
            try:
                print("[segpad] frac=%.3f  %dx%d -> %dx%d"
                      % (_segpad, w, h, _seg_img.bgr.shape[1], _seg_img.bgr.shape[0]))
            except Exception:  # noqa: BLE001
                pass

    # Best (opt-in): a real matting model -> true strand-level hair alpha. Only runs when
"""

RVM_OLD = "        alpha = matting.matte(img.bgr, warns)\n"
RVM_NEW = ("        alpha = _unpad(matting.matte(_seg_img.bgr, warns), _seg_box)\n")

MP_OLD = "    selfie, soft = _selfie_mask(img, warns)\n"
MP_NEW = ("    selfie, soft = _selfie_mask(_seg_img, warns)\n"
          "    selfie, soft = _unpad(selfie, _seg_box), _unpad(soft, _seg_box)\n")


def die(msg):
    raise SystemExit("ABORTED (nothing written): " + msg)


def main():
    if not os.path.isfile(PATH):
        die("no such file: %s" % PATH)
    src = open(PATH, encoding="utf-8").read()

    if "TYPO_SEG_PAD" in src:
        print("already patched -- no change")
        return

    for name, anchor in (("extract_silhouette def", DEF_ANCHOR),
                         ("function head", HEAD_OLD),
                         ("rvm call", RVM_OLD),
                         ("mediapipe call", MP_OLD)):
        if src.count(anchor) != 1:
            die("%s found %d times, expected 1" % (name, src.count(anchor)))

    out = src.replace(DEF_ANCHOR, HELPERS.lstrip("\n") + DEF_ANCHOR, 1)
    out = out.replace(HEAD_OLD, HEAD_NEW, 1)
    out = out.replace(RVM_OLD, RVM_NEW, 1)
    out = out.replace(MP_OLD, MP_NEW, 1)
    compile(out, PATH, "exec")
    shutil.copy2(PATH, PATH + ".bak-segpad")
    open(PATH, "w", encoding="utf-8").write(out)
    print("patched OK   (backup: %s.bak-segpad)" % PATH)
    print("SYNTAX OK -- inert unless TYPO_SEG_PAD is set")


if __name__ == "__main__":
    main()
