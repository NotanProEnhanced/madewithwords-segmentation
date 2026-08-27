#!/usr/bin/env python3
"""Like-for-like render comparison across brand configurations.

The two screenshots that prompted this differed in BOTH the tuning and the word
list, so they could not settle which config is better. This renders one photo
with one word list, and is run inside each brand's own container -- so the only
variable is that container's environment.

    docker exec typortrait              python /app/data/render-compare.py typortrait
    docker exec typortrait-lovedinwords python /app/data/render-compare.py lovedinwords

Deliberately NOT done by setting os.environ inside a single process: several of
these values may be read at import time, so the only trustworthy way to compare
configurations is to run in the real container.

Writes compare-<tag>.jpg and prints the render-relevant environment so the image
and the settings that produced it stay together.
"""
import io
import os
import sys

sys.path.insert(0, "/app")

from PIL import Image                                    # noqa: E402
from app.config import RenderConfig                      # noqa: E402
from app.pipeline.warnings import WarningCollector       # noqa: E402
from app.pipeline.analyze import analyze_image           # noqa: E402
from app.pipeline.displacement import render_displacement_portrait   # noqa: E402

SRC_DEFAULT = "/app/data/marketing-src/man.png"   # override with argv[2]
OUT = "/app/data/marketing-out"

WORDS = [
    "MICHAEL", "MICHAEL", "MIKE", "DAD", "FATHER", "HUSBAND", "GRANDPA",
    "SON", "BROTHER", "FRIEND", "FAMILY", "HOME", "PROVIDER", "PROTECTOR",
    "MENTOR", "BEST FRIEND", "FAMILY FIRST", "SUNDAY DINNER", "BACKYARD",
    "GRILL MASTER", "MORNING COFFEE", "ROAD TRIPS", "VACATIONS", "GOLF",
    "FOOTBALL", "GARAGE", "OLD SONGS", "BAD JOKES", "GREAT STORIES",
    "HARD WORK", "ALWAYS THERE", "GOOD ADVICE", "STRONG", "DEPENDABLE",
    "LOYAL", "PATIENT", "GENEROUS", "FUNNY", "PROUD", "STEADFAST",
    "OUR ROCK", "ROLE MODEL", "LEGACY", "LOVE", "RESPECT", "FOREVER",
    "MICHAEL",
]

# The settings that plausibly explain a visible difference between the brands.
WATCH = [
    "TYPO_EYE_PHOTO", "TYPO_EYE_POP", "TYPO_EYE_SHARPEN", "TYPO_DARKSCLERA",
    "TYPO_IRIS_LIFT", "TYPO_BG_LIGHTEN", "TYPO_SHADOW_LIFT", "TYPO_VIBRANCE",
    "TYPO_LAYERED_PHOTO", "TYPO_HILIGHT_FINE", "TYPO_HILIGHT_WASH",
    "TYPO_MATTE_FLOOR", "TYPO_MATTE_MODEL", "TYPO_DEPOSTERIZE",
    "TYPO_INK_LIFT", "TYPO_INK_LIFT_ADD", "TYPO_SUBJECT_BASE",
    "TYPO_SUBJECT_DIM", "TYPO_EYE_PLAIN", "TYPO_WORD_VARIETY",
]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "untagged"
    src_path = sys.argv[2] if len(sys.argv) > 2 else SRC_DEFAULT
    if not os.path.isabs(src_path):
        src_path = os.path.join("/app/data/marketing-src", src_path)
    if not os.path.isfile(src_path):
        raise SystemExit("missing source: %s" % src_path)
    os.makedirs(OUT, exist_ok=True)

    print("=== %s ===" % tag)
    for k in WATCH:
        v = os.environ.get(k)
        print("  %-22s %s" % (k, "<unset>" if v is None else (v or "<empty>")))

    print("source %s" % src_path)
    base = open(src_path, "rb").read()
    warns = WarningCollector()
    an = analyze_image(base, RenderConfig(), warns)
    png = render_displacement_portrait(
        an, WORDS, ground="navy", out_width=1600, supersample=2,
        ink="photo", print_aspect=1.0, breathe=True, graduate=True,
        backdrop="studio")
    im = Image.open(io.BytesIO(png)).convert("RGB").resize((1100, 1100), Image.LANCZOS)
    out = os.path.join(OUT, "compare-%s.jpg" % tag)
    im.save(out, quality=92, optimize=True)
    print("wrote %s  %s" % (out, warns.as_list()))


if __name__ == "__main__":
    main()
