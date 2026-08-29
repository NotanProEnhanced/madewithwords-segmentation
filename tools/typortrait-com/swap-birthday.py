#!/usr/bin/env python3
"""Replace the Graduates card on typortrait.com with Birthdays, Anniversaries.

Two reasons the graduate had to go. The segmentation dropped his mortarboard --
a large dark object against a light backdrop -- so the portrait was missing the
one element that said "graduation". And graduation is a May/June purchase, so
the card was selling a season that is eight months away.

The woman takes that slot, and the card becomes year-round.

  1. card heading    Graduates -> Birthdays, Anniversaries...
  2. card body copy  rewritten for the new occasion
  3. image files     use-graduate.jpg -> use-birthday.jpg, same for mock-*
  4. alt / aria text rewritten
  5. ?v=6 on the new references

Aborts without writing if an anchor is missing. Idempotent.

Usage:  python3 swap-birthday.py [/var/www/typortrait.com] [source-dir]
"""
import os
import re
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/var/www/typortrait.com"
NEWDIR = (sys.argv[2] if len(sys.argv) > 2
          else "/root/typortrait-stg/typography_engine/data/marketing-out")
PAGE = os.path.join(ROOT, "index.html")

TITLE = "Birthdays, Anniversaries&hellip;"
WORDS_LINE = ("Words like <b>birthday &middot; anniversary &middot; together &middot; years &middot; always &middot; us</b>")
COPY = ("A gift that says more than a card: the words you would use about them, "
        "in their own portrait.")

TOKENS = [
    ("use-graduate", "use-birthday"),
    ("mock-graduate", "mock-birthday"),
    ("Graduate render", "Birthday render"),
    ("Graduate mockup", "Birthday mockup"),
    ("Enlarge a Graduate word-art portrait",
     "Enlarge a birthday and anniversary word-art portrait"),
    ("Enlarge a framed Graduate word-art print",
     "Enlarge a framed birthday and anniversary word-art print"),
    ("word-art portrait of a graduate",
     "word-art portrait for a birthday or anniversary"),
]


def die(msg):
    raise SystemExit("ABORTED (nothing written): " + msg)


def main():
    if not os.path.isfile(PAGE):
        die("no such file: %s" % PAGE)
    src = open(PAGE, encoding="utf-8").read()

    if TITLE in src:
        print("card already replaced")
    else:
        # The heading appears twice: the use-case card (prose body) and the
        # prints card (a "Words like ..." list). Different bodies, so replace
        # each with its own. Working by index avoids having to match the exact
        # typographic apostrophes in the old copy.
        h = "<h3>Graduates</h3>"
        if src.count(h) != 2:
            die("heading found %d times, expected 2" % src.count(h))

        bodies = ['<p>%s</p>' % COPY, '<p class="words">%s</p>' % WORDS_LINE]
        for body in bodies:
            i = src.index(h)
            j = src.index("</p>", i) + len("</p>")
            src = src[:i] + "<h3>%s</h3>%s" % (TITLE, body) + src[j:]
        print("both cards replaced (use-case prose, prints word list)")

        # All three mockups are now the same wood frame on a desk, so the old
        # per-image descriptions (light-oak on a wall, black frame on a desk)
        # no longer describe what a reader sees.
        n = len(re.findall(r'matted in [^"]*', src))
        src = re.sub(r'matted in [^"]*', 'in a wood frame on a desk', src)
        print("mockup alt text normalised on %d images" % n)

        for old, new in TOKENS:
            n = src.count(old)
            if n:
                src = src.replace(old, new)
            print("  %-42s %d" % (old, n))

    # cache-bust the renamed images
    hits = len(re.findall(r'/(?:use|mock)-birthday\.jpg\?v=\d', src))
    src = re.sub(r'(/(?:use|mock)-birthday\.jpg)\?v=\d', r'\1?v=6', src)
    print("cache-bust set on %d birthday references" % hits)

    shutil.copy2(PAGE, PAGE + ".bak-birthday")
    open(PAGE, "w", encoding="utf-8").write(src)
    print("index.html patched   (backup: %s.bak-birthday)" % PAGE)

    for kind in ("use", "mock"):
        new = os.path.join(NEWDIR, "%s-birthday-new.jpg" % kind)
        cur = os.path.join(ROOT, "%s-birthday.jpg" % kind)
        if not os.path.isfile(new):
            print("%s-birthday-new.jpg missing -- image not placed" % kind)
            continue
        shutil.copy2(new, cur)
        print("%s-birthday.jpg <- %s-birthday-new.jpg" % (kind, kind))

    # retire the graduate files rather than delete them
    for name in ("use-graduate.jpg", "mock-graduate.jpg"):
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            os.rename(p, p + ".retired")
            print("%s -> %s.retired" % (name, name))

    print("\ndone. reload https://typortrait.com/?v=8")


if __name__ == "__main__":
    main()
