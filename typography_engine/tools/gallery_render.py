"""Render a base artwork into a gallery master through the Typortrait engine.

Feed an original or public-domain image (a painting, an illustration you own) and
the engine weaves the piece's scripture/verse into it -- the same renderer the
studio uses. Words come from --words or, by default, tools/gallery_words/<id>.txt.

    # Lifelike (face-driven) -- the default; good for figures with a clear face:
    python tools/gallery_render.py good-shepherd base/good-shepherd.png --ground navy

    # explicit words + a lighter ground:
    python tools/gallery_render.py light-of-the-world base/hunt.jpg --ground paper \
        --words "I am the light of the world ..."

Style notes: 'displacement' (Lifelike) needs a detectable face; for word-only
pieces with no face (a hill, a landscape) use --style words. Writes the private
master + public preview via the same path as gallery_add.py.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image

from _gallery_lib import ROOT, place_master


def _load_words(item_id: str, words_arg: str | None) -> list[str]:
    text = words_arg
    if not text:
        wf = ROOT / "tools" / "gallery_words" / f"{item_id}.txt"
        if not wf.exists():
            raise SystemExit(f"No --words given and {wf} not found.")
        text = wf.read_text(encoding="utf-8")
    words = [w for w in text.replace("\n", " ").split() if w]
    if not words:
        raise SystemExit("No words to render.")
    return words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item_id")
    ap.add_argument("base", help="base artwork to weave the words into (you must own / it must be public domain)")
    ap.add_argument("--style", default="displacement", choices=["displacement", "words", "message"])
    ap.add_argument("--ground", default="navy", choices=["navy", "paper", "black"])
    ap.add_argument("--ink", default=None, help="ink/color key (default: style default)")
    ap.add_argument("--words", default=None, help="override words; else reads tools/gallery_words/<id>.txt")
    ap.add_argument("--out-width", type=int, default=1600)
    ap.add_argument("--supersample", type=int, default=2, help="displacement quality (2 = print)")
    ap.add_argument("--aspect", type=float, default=0.8, help="print canvas width/height (default 4:5)")
    ap.add_argument("--allow-unlisted", action="store_true")
    ap.add_argument("--breathe", action="store_true",
                    help="Phase-1 tonal 'breathing': deep-shadow negative space + specular relief (displacement only)")
    ap.add_argument("--discovery", action="store_true",
                    help="Phase-1 discovery layer: hide a small cross + IHS for close-inspection detail (displacement only)")
    ap.add_argument("--discovery-ref", default=None,
                    help="optional tiny scripture ref to hide too, e.g. 'JN 8:12' (implies --discovery)")
    a = ap.parse_args()

    discovery = None
    if a.discovery or a.discovery_ref:
        discovery = ["IHS"]
        if a.discovery_ref:
            discovery.append(a.discovery_ref)

    words = _load_words(a.item_id, a.words)
    img_bytes = Path(a.base).read_bytes()

    from app.config import RenderConfig
    from app.pipeline.analyze import analyze_image
    from app.pipeline.warnings import WarningCollector

    warns = WarningCollector()
    an = analyze_image(img_bytes, RenderConfig(), warns)

    if a.style == "displacement":
        from app.pipeline.displacement import render_displacement_portrait
        png = render_displacement_portrait(
            an, words, ground=a.ground, out_width=a.out_width,
            supersample=a.supersample, ink=a.ink, print_aspect=a.aspect,
            breathe=a.breathe, discovery=discovery)
    else:
        from app.pipeline.tonal import render_layered_png
        png, _runs, _g, _svg = render_layered_png(
            an, " ".join(words), a.style, RenderConfig(), warns,
            ink=(a.ink or "navy"), out_width=a.out_width)

    if not png:
        raise SystemExit("render produced no image; warnings: " + str(warns.as_list()))

    img = Image.open(io.BytesIO(png))
    master, preview, cleared = place_master(a.item_id, img, allow_unlisted=a.allow_unlisted)
    print(f"OK  rendered {a.style} ({img.size[0]}x{img.size[1]}) from {a.base}")
    print(f"    master  -> {master}")
    print(f"    preview -> {preview}   (print-cache cleared: {cleared})")
    print(f"    live at /gallery")


if __name__ == "__main__":
    main()
