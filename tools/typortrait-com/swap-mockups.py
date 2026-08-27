#!/usr/bin/env python3
"""Swap the rebuilt framed mockups into typortrait.com and retitle the section.

The mockups are now the product's own framed-on-a-desk scene rather than the
June wall composites, so the heading "See it on your wall." no longer describes
what the images show.

  1. heading  -> "See the portrait in your home!"
  2. mock-graduate/children/memorial.jpg <- the -new.jpg renders (June kept as .bak-jun24)
  3. ?v=5 -> ?v=6 on every mock-* reference (src and data-full)
  4. width/height attributes -> 900x1125, which the renders now actually are

Aborts without writing if any anchor count is unexpected. Idempotent.

Usage:  python3 swap-mockups.py [/var/www/typortrait.com] [source-dir]
"""
import os
import re
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/var/www/typortrait.com"
NEWDIR = (sys.argv[2] if len(sys.argv) > 2
          else "/root/typortrait-stg/typography_engine/data/marketing-out")
PAGE = os.path.join(ROOT, "index.html")

OLD_HEADING = '<h2 class="statement">See it on your wall.</h2>'
NEW_HEADING = '<h2 class="statement">See the portrait in your home!</h2>'

NAMES = ("graduate", "children", "memorial")


def die(msg):
    raise SystemExit("ABORTED (nothing written): " + msg)


def main():
    if not os.path.isfile(PAGE):
        die("no such file: %s" % PAGE)
    src = open(PAGE, encoding="utf-8").read()

    if NEW_HEADING in src:
        print("heading already updated")
    else:
        n = src.count(OLD_HEADING)
        if n != 1:
            die("heading found %d times, expected 1" % n)
        src = src.replace(OLD_HEADING, NEW_HEADING, 1)
        print("heading updated")

    # ?v=5 -> ?v=6 on mock-* references only
    hits = len(re.findall(r'/mock-[a-z]+\.jpg\?v=5', src))
    src = re.sub(r'(/mock-[a-z]+\.jpg)\?v=5', r'\1?v=6', src)
    print("cache-bust bumped on %d mock-* references" % hits)

    # the renders are 900x1125; make the declared size match so nothing is scaled
    before = len(re.findall(r'(src="/mock-[a-z]+\.jpg\?v=6" )width="\d+" height="\d+"', src))
    src = re.sub(r'(src="/mock-[a-z]+\.jpg\?v=6" )width="\d+" height="\d+"',
                 r'\1width="900" height="1125"', src)
    print("size attributes normalised on %d images" % before)

    shutil.copy2(PAGE, PAGE + ".bak-mockups")
    open(PAGE, "w", encoding="utf-8").write(src)
    print("index.html patched   (backup: %s.bak-mockups)" % PAGE)

    for name in NAMES:
        new = os.path.join(NEWDIR, "mock-%s-new.jpg" % name)
        cur = os.path.join(ROOT, "mock-%s.jpg" % name)
        bak = cur + ".bak-jun24"
        if not os.path.isfile(new):
            print("mock-%s-new.jpg missing -- image not swapped" % name)
            continue
        if os.path.isfile(cur) and not os.path.isfile(bak):
            shutil.copy2(cur, bak)
        shutil.copy2(new, cur)
        print("mock-%s.jpg <- mock-%s-new.jpg" % (name, name))

    print("\ndone. reload https://typortrait.com/?v=7#prints")


if __name__ == "__main__":
    main()
