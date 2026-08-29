#!/usr/bin/env python3
"""Re-render sample-1 (Original) and sample-2 (Noir) for typortrait.com.

These are the two images in the "One photo. Two looks." section. They were made
by mksamples.py, which passed ground="navy" but no backdrop -- so they show the
subject on bare navy, while a real customer render puts them on the studio
backdrop (#e6e6e6, a light neutral wall). This reproduces exactly what the
studio posts: ground="navy", ink="photo"/"mono", backdrop="studio".

Words are the ones mksamples.py used. They are deliberately unattributed -- this
is the generic "same face, two looks" comparison, not a named subject like the
use-case cards.

Runs INSIDE the container, through the bind-mounted data/ directory:

    docker exec typortrait-staging python /app/data/render-samples.py

Writes sample-*-new2.jpg for review; swaps nothing. The -new2 suffix avoids
clobbering the -new.jpg files already sitting in /var/www from the earlier
(navy-backed) pass.
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

SRC = "/app/data/marketing-src"
OUT = "/app/data/marketing-out"
SOURCE = "woman.png"

WORDS = ["KIND", "PATIENT", "LAUGHS FIRST", "ALWAYS EARLY", "SUNDAY PANCAKES",
         "TERRIBLE JOKES", "STEADY HANDS", "MY FAVOURITE PERSON", "HOME"]

# (output file, ink) -- the two inks the studio still offers
JOBS = [("sample-1-new2.jpg", "photo"),      # Original
        ("sample-2-new2.jpg", "mono")]       # Noir

GROUND = "navy"
BACKDROP = "studio"        # the axis mksamples.py omitted
OUT_WIDTH = 1600
FINAL = 1100


def main():
    src = os.path.join(SRC, SOURCE)
    if not os.path.isfile(src):
        raise SystemExit("missing source: %s" % src)
    os.makedirs(OUT, exist_ok=True)

    base = open(src, "rb").read()
    for out_name, ink in JOBS:
        warns = WarningCollector()
        an = analyze_image(base, RenderConfig(), warns)
        png = render_displacement_portrait(
            an, WORDS, ground=GROUND, out_width=OUT_WIDTH, supersample=2,
            ink=ink, print_aspect=1.0, breathe=True, graduate=True,
            backdrop=BACKDROP)
        im = Image.open(io.BytesIO(png)).convert("RGB").resize(
            (FINAL, FINAL), Image.LANCZOS)
        im.save(os.path.join(OUT, out_name), quality=90, optimize=True)
        print("%-22s ink=%-6s %s" % (out_name, ink, warns.as_list()))

    print("\ndone -- review before swapping anything into /var/www")


if __name__ == "__main__":
    main()
