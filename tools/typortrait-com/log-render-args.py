#!/usr/bin/env python3
"""Log the exact arguments the /render endpoint passes to the engine.

A direct call to render_displacement_portrait with the same photo, ground, ink,
aspect and backdrop produces a CLEAN image at both supersample 1 and 2. The app
produces blobs. So the app is passing something the harness is not, and listing
candidates has failed repeatedly.

This prints the real keyword arguments on each request when TYPO_ARGS_DEBUG is
set, so the harness can be made to match exactly -- and whichever argument
differs is the cause, by construction.

Usage:  python3 log-render-args.py <tree>/typography_engine
Idempotent; aborts without writing if the anchor is not found exactly once.
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
PATH = os.path.join(ROOT, "app/main.py")

ANCHOR = """            png_bytes = await _bounded_to_thread(
                render_displacement_portrait, an, disp_words, ground=ground_choice,
"""

NEW = """            # __import__ rather than `os`: this handler has a function-local
            # `import os` further down, which makes `os` local for the WHOLE
            # function -- referencing it here raises UnboundLocalError.
            if __import__("os").environ.get("TYPO_ARGS_DEBUG", "").strip().lower() in ("1", "true", "on", "yes"):
                try:
                    print("[args] ground=%r ink=%r backdrop=%r aspect=%r out_width=%r ss=%r "
                          "flow=%r uppercase=%r breathe=%r variety=%r sunglasses=%r "
                          "sunglass_faces=%r words=%d style=%r"
                          % (ground_choice,
                             ("photo" if ink_choice == "photo_paper" else ink_choice),
                             backdrop_choice, aspect_choice, max(320, preview_w), disp_ss,
                             disp_flow_eff, uppercase, STUDIO_BREATHE, disp_variety_eff,
                             sunglasses_on, sunglass_faces_sel, len(disp_words),
                             style_choice))
                except Exception as _e:  # noqa: BLE001
                    print("[args] failed: %s" % _e)
"""


def main():
    if not os.path.isfile(PATH):
        raise SystemExit("no such file: %s" % PATH)
    src = open(PATH, encoding="utf-8").read()

    if "TYPO_ARGS_DEBUG" in src:
        print("already patched -- no change")
        return
    if src.count(ANCHOR) != 1:
        raise SystemExit("ABORTED: anchor found %d times, expected 1" % src.count(ANCHOR))

    out = src.replace(ANCHOR, NEW + ANCHOR, 1)
    compile(out, PATH, "exec")
    shutil.copy2(PATH, PATH + ".bak-args")
    open(PATH, "w", encoding="utf-8").write(out)
    print("patched OK   (backup: %s.bak-args)" % PATH)
    print("SYNTAX OK -- inert unless TYPO_ARGS_DEBUG is set")


if __name__ == "__main__":
    main()
