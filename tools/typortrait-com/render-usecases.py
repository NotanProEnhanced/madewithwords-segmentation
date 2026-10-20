#!/usr/bin/env python3
"""Render the three use-case portraits for typortrait.com's marketing page.

Runs INSIDE the container (it imports the engine from /app), the same way
mksamples.py produced samples 1 and 2. Reads and writes through the bind-mounted
data/ directory, so no docker cp is needed in either direction:

    host   /root/typortrait-stg/typography_engine/data/marketing-src/*.png
    inside /app/data/marketing-src/*.png

    host   /root/typortrait-stg/typography_engine/data/marketing-out/use-*-new.jpg
    inside /app/data/marketing-out/use-*-new.jpg

Settings match the samples exactly -- navy ground, Original ink ("photo"),
square print aspect, 1600px render downsampled to 1100 -- so the use-case row
and the examples row look like one family.

Run:
    docker exec typortrait-staging python /app/data/render-usecases.py

Edit WORDS below before running. Invented words render fine but read as filler
at loupe distance, and these images are doing the selling.
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

# (source file, output file, words)
# Words supplied by the owner. The repeated name at the head and tail of each
# list is deliberate -- it gives the engine the subject's name to place more
# than once, which is how the real product is used.
JOBS = [
    ("woman.png", "use-birthday-new.jpg", [
        "EMMA", "EMMA", "THIRTY", "WIFE", "DAUGHTER", "SISTER", "BEST FRIEND",
        "HOME", "SUNDAY MORNINGS", "STRONG COFFEE", "LONG WALKS",
        "OLD PLAYLISTS", "SECOND HELPINGS", "BOOKS EVERYWHERE",
        "TERRIBLE JOKES", "LAUGHS FIRST", "ALWAYS EARLY", "STEADY HANDS",
        "KIND", "PATIENT", "FUNNY", "FEARLESS", "GENEROUS", "CURIOUS",
        "WARM", "STUBBORN", "THOUGHTFUL", "LOYAL", "ANOTHER YEAR",
        "ANOTHER CHAPTER", "STILL COUNTING", "TEN YEARS", "US",
        "MY FAVOURITE PERSON", "HAPPY BIRTHDAY", "EMMA",
    ]),
    ("son.png", "use-children-new.jpg", [
        "LEO", "LEO", "SON", "GRANDSON", "BROTHER", "BUDDY", "LITTLE MAN",
        "MOMMY", "DADDY", "FAMILY", "HOME", "HUGS", "GIGGLES", "BIG SMILE",
        "BEDTIME STORIES", "SATURDAY MORNINGS", "CARTOONS", "LEGO",
        "DINOSAURS", "SUPERHEROES", "BIKE RIDES", "PLAYGROUND", "SOCCER",
        "ICE CREAM", "PIZZA", "PANCAKES", "BIRTHDAY CAKE",
        "CHRISTMAS MORNING", "BEST FRIENDS", "SILLY JOKES", "QUESTIONS",
        "MISCHIEF", "ADVENTURES", "CURIOUS", "FEARLESS", "SWEET", "FUNNY",
        "SMART", "KIND", "IMAGINATION", "OUR BOY", "OUR SUNSHINE",
        "LOVE YOU TO THE MOON", "LEO",
    ]),
    ("man.png", "use-memorial-new.jpg", [
        "MICHAEL", "MICHAEL", "MIKE", "DAD", "FATHER", "HUSBAND", "GRANDPA",
        "SON", "BROTHER", "FRIEND", "FAMILY", "HOME", "PROVIDER",
        "PROTECTOR", "MENTOR", "BEST FRIEND", "FAMILY FIRST",
        "SUNDAY DINNER", "BACKYARD", "GRILL MASTER", "MORNING COFFEE",
        "ROAD TRIPS", "VACATIONS", "GOLF", "FOOTBALL", "GARAGE", "OLD SONGS",
        "BAD JOKES", "GREAT STORIES", "HARD WORK", "ALWAYS THERE",
        "GOOD ADVICE", "STRONG", "DEPENDABLE", "LOYAL", "PATIENT",
        "GENEROUS", "FUNNY", "PROUD", "STEADFAST", "OUR ROCK", "ROLE MODEL",
        "LEGACY", "LOVE", "RESPECT", "FOREVER", "MICHAEL",
    ]),
]

GROUND = "navy"
INK = "photo"          # Original -- the only inks now offered are photo and mono
# The studio posts ground="navy" AND backdrop="studio" (variantPost in
# static/index.html). "backdrop" recolours only the segmented background to
# #e6e6e6, a light neutral wall. Without it the subject sits on bare navy --
# a background no customer ever gets.
BACKDROP = "studio"
OUT_WIDTH = 1600
FINAL = 1100


def main():
    if not os.path.isdir(SRC):
        raise SystemExit("no source directory: %s" % SRC)
    os.makedirs(OUT, exist_ok=True)

    missing = [j[0] for j in JOBS if not os.path.isfile(os.path.join(SRC, j[0]))]
    if missing:
        raise SystemExit("missing source files in %s: %s" % (SRC, ", ".join(missing)))

    for src_name, out_name, words in JOBS:
        base = open(os.path.join(SRC, src_name), "rb").read()
        warns = WarningCollector()
        an = analyze_image(base, RenderConfig(), warns)
        png = render_displacement_portrait(
            an, words, ground=GROUND, out_width=OUT_WIDTH, supersample=2,
            ink=INK, print_aspect=1.0, breathe=True, graduate=True,
            backdrop=BACKDROP)
        im = Image.open(io.BytesIO(png)).convert("RGB").resize(
            (FINAL, FINAL), Image.LANCZOS)
        im.save(os.path.join(OUT, out_name), quality=90, optimize=True)
        print("%-22s <- %-14s %s" % (out_name, src_name, warns.as_list()))

    print("\ndone -- review before swapping anything into /var/www")


if __name__ == "__main__":
    main()
