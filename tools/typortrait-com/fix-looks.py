#!/usr/bin/env python3
"""Bring typortrait.com's marketing copy in line with what the studio actually offers.

The studio now offers exactly two word colours -- Original and Noir. The Custom
picker was retired (commit "Retire the Custom ink option") because a picked hex
rendered through the legacy flat single-ink sculpt path. The static marketing
site was never updated and still advertises a palette that no longer exists, in
five separate places including the JSON-LD that search engines index and quote.

This script fixes all of them in one pass and aborts WITHOUT writing if any
anchor does not appear exactly as many times as expected -- so a partial edit is
impossible.

  1. #examples section  -- three samples (Navy/Ivory/Ember) -> two (Original/Noir)
  2. ld+json ImageObject for sample-3.jpg -- removed
  3. ld+json captions for sample-1/2 -- "Navy"/"Ivory" -> "Original"/"Noir"
  4. Step 3 copy, visible + HowTo ld+json -- drops "navy or gold-on-black"
  5. WebApplication featureList -- drops the sepia/burgundy/forest/gold list
  6. FAQ answer, visible + FAQPage ld+json -- same
  7. meta description -- "multiple colours" -> "two word colours"

Also swaps in the re-rendered sample images (sample-N-new.jpg -> sample-N.jpg),
keeping the June originals as .bak-jun24.

Usage:  python3 fix-looks.py [/var/www/typortrait.com]
Idempotent: re-running on an already-patched file reports "already patched".
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/var/www/typortrait.com"
PAGE = os.path.join(ROOT, "index.html")

NEW_SECTION = '''<h2 class="statement">One photo. Two looks.</h2>
      <p class="support">The same face and the same words, rendered two ways &mdash; for anniversaries, memorials, weddings, birthdays and more.</p>
      <div class="cards" style="grid-template-columns:repeat(2,minmax(0,1fr));max-width:960px;margin-left:auto;margin-right:auto">
        <figure class="shot"><button type="button" class="shot-btn js-zoom" data-full="/sample-1.jpg?v=6" aria-label="Enlarge sample - Original"><img src="/sample-1.jpg?v=6" width="720" height="720" loading="lazy" alt="Portrait built from words, drawn in the colours of the photograph" /></button><figcaption>Original</figcaption></figure>
        <figure class="shot"><button type="button" class="shot-btn js-zoom" data-full="/sample-2.jpg?v=6" aria-label="Enlarge sample - Noir"><img src="/sample-2.jpg?v=6" width="720" height="720" loading="lazy" alt="The same portrait built from words, in monochrome" /></button><figcaption>Noir</figcaption></figure>
      </div>'''

HEADING = '<h2 class="statement">One photo, every look.</h2>'
CARDS = '<div class="cards">'
LDJSON_S3 = '{"@type": "ImageObject", "url": "https://typortrait.com/sample-3.jpg"'

REPLACEMENTS = [
    ("caption navy", 'Navy."', 'Original."', 1),
    ("caption ivory", 'Ivory."', 'Noir."', 1),
    ("step 3 copy",
     "Pick Words or Message, a colour like navy or gold-on-black.",
     "Pick Words or Message, then Original or Noir for the words themselves.", 2),
    ("featureList",
     '"Navy, sepia, burgundy, forest, gold-on-black and full-colour options"',
     '"Original and Noir word colours"', 1),
    ("faq colours",
     "plus a range of colours: classic black, navy, sepia, burgundy, forest, "
     "gold-on-black, vivid spectrum and aurora gradients, or full photo colour.",
     "plus two word colours: Original, which draws the words in the colours of "
     "your photograph, and Noir, which renders them in monochrome.", 2),
    ("meta description", "multiple colours", "two word colours", 1),
]


def die(msg):
    raise SystemExit("ABORTED (file unchanged): " + msg)


def main():
    if not os.path.isfile(PAGE):
        die("no such file: %s" % PAGE)
    src = open(PAGE, encoding="utf-8").read()

    if "One photo. Two looks." in src:
        print("already patched -- no change to index.html")
    else:
        log = []
        s = src

        # 1. the visible examples section: heading through the end of .cards
        if s.count(HEADING) != 1:
            die("heading found %d times, expected 1" % s.count(HEADING))
        i = s.index(HEADING)
        c = s.find(CARDS, i)
        if c < 0:
            die("no <div class=\"cards\"> after the examples heading")
        end = s.find("</div>", c)
        if end < 0:
            die("unterminated cards block")
        s = s[:i] + NEW_SECTION + s[end + len("</div>"):]
        log.append("%-22s 1" % "examples section")

        # 2. the sample-3 ImageObject inside the ld+json @graph
        if s.count(LDJSON_S3) != 1:
            die("sample-3 ld+json anchor found %d times, expected 1" % s.count(LDJSON_S3))
        k = s.index(LDJSON_S3)
        e = s.index("}", k) + 1
        pre = 2 if s[k - 2:k] == ", " else 0
        s = s[:k - pre] + s[e:]
        log.append("%-22s 1" % "sample-3 ld+json")

        # 3-7. plain string swaps, all count-checked before anything is written
        for name, old, new, expect in REPLACEMENTS:
            n = s.count(old)
            if n != expect:
                die("%s: found %d, expected %d" % (name, n, expect))
            s = s.replace(old, new)
            log.append("%-22s %d" % (name, n))

        shutil.copy2(PAGE, PAGE + ".bak-looks")
        open(PAGE, "w", encoding="utf-8").write(s)
        print("\n".join(log))
        print("index.html patched   (backup: %s.bak-looks)" % PAGE)

    # image swap -- separate from the HTML edit so either can be redone alone
    for n in (1, 2):
        new = os.path.join(ROOT, "sample-%d-new.jpg" % n)
        cur = os.path.join(ROOT, "sample-%d.jpg" % n)
        bak = cur + ".bak-jun24"
        if not os.path.isfile(new):
            print("sample-%d-new.jpg missing -- image not swapped" % n)
            continue
        if os.path.isfile(cur) and not os.path.isfile(bak):
            shutil.copy2(cur, bak)
        shutil.copy2(new, cur)
        print("sample-%d.jpg <- sample-%d-new.jpg" % (n, n))

    print("\ndone. reload https://typortrait.com/?v=6#examples")


if __name__ == "__main__":
    main()
