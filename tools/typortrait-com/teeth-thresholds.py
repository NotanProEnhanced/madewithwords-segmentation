#!/usr/bin/env python3
"""Make the teeth-detection thresholds visible and tunable.

A closed mouth is being read as open: the inner-lip ring passes the geometry
gate, the appearance gate lets it through, and the cleared region renders as a
flat white shape over the lips. On a two-subject portrait that is two blobs.

Nothing here changes behaviour. It exposes the three magic numbers as env vars
at their current values and adds a debug line so the real measurements can be
read off a failing photo instead of guessed at:

    TYPO_TEETH_MIN_RATIO   0.12    inner-lip height/width below which -> closed
    TYPO_TEETH_DARK        60.0    p10 at or below this -> a real dark cavity
    TYPO_TEETH_BRIGHT      205.0   p90 at or above this -> real bright teeth
    TYPO_TEETH_DEBUG               set to print the measurements per render

Note the debug flag is checked with an explicit truthy-value list, NOT bare
truthiness -- TYPO_EYE_DEBUG uses `.strip()` alone, which means setting it to
"0" turns it ON. Do not repeat that here.

Usage:  python3 teeth-thresholds.py <tree>/typography_engine
Idempotent. Aborts without writing if an anchor is missing or `os` is not
imported at module scope (a function-local import would rebind `os` for the
whole function and break other uses).
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
DISP = os.path.join(ROOT, "app/pipeline/displacement.py")
TONAL = os.path.join(ROOT, "app/pipeline/tonal.py")

MARKER = "TYPO_TEETH_MIN_RATIO"

TONAL_OLD = (
    "    if pw < 3.0 or ph < 2.0 or ph / pw < 0.12:    # lips together -> no teeth\n"
    "        return None\n"
)
TONAL_NEW = (
    "    # Geometry gate. A closed mouth still has a thin inner-lip ring, so this\n"
    "    # ratio is the first line of defence against a false positive.\n"
    "    _tr = float(os.environ.get(\"TYPO_TEETH_MIN_RATIO\", \"0.12\") or 0.12)\n"
    "    if os.environ.get(\"TYPO_TEETH_DEBUG\", \"\").strip().lower() in (\"1\", \"true\", \"on\", \"yes\"):\n"
    "        try:\n"
    "            print(\"[teeth] pw=%.1f ph=%.1f ratio=%.3f gate=%.3f -> %s\"\n"
    "                  % (pw, ph, (ph / pw if pw else 0.0), _tr,\n"
    "                     \"closed\" if (pw < 3.0 or ph < 2.0 or ph / pw < _tr) else \"open\"))\n"
    "        except Exception:\n"
    "            pass\n"
    "    if pw < 3.0 or ph < 2.0 or ph / pw < _tr:    # lips together -> no teeth\n"
    "        return None\n"
)

DISP_OLD = (
    "            if p10 > 60.0 and p90 < 205.0:\n"
    "                teeth = None                           # no cavity, no teeth -> closed mouth\n"
)
DISP_NEW = (
    "            _tdark = float(os.environ.get(\"TYPO_TEETH_DARK\", \"60.0\") or 60.0)\n"
    "            _tbright = float(os.environ.get(\"TYPO_TEETH_BRIGHT\", \"205.0\") or 205.0)\n"
    "            if os.environ.get(\"TYPO_TEETH_DEBUG\", \"\").strip().lower() in (\"1\", \"true\", \"on\", \"yes\"):\n"
    "                try:\n"
    "                    print(\"[teeth] p10=%.1f p90=%.1f dark<=%.1f bright>=%.1f -> %s\"\n"
    "                          % (p10, p90, _tdark, _tbright,\n"
    "                             \"closed\" if (p10 > _tdark and p90 < _tbright) else \"KEPT\"))\n"
    "                except Exception:\n"
    "                    pass\n"
    "            if p10 > _tdark and p90 < _tbright:\n"
    "                teeth = None                           # no cavity, no teeth -> closed mouth\n"
)


def die(msg):
    raise SystemExit("ABORTED (nothing written): " + msg)


def patch(path, old, new):
    src = open(path, encoding="utf-8").read()
    if MARKER in src or "TYPO_TEETH_DARK" in src:
        print("%s already patched" % os.path.basename(path))
        return
    if not any(line.strip() == "import os" for line in src.splitlines()[:60]):
        die("%s does not import os at module scope" % os.path.basename(path))
    if src.count(old) != 1:
        die("%s: anchor found %d times, expected 1" % (os.path.basename(path), src.count(old)))
    out = src.replace(old, new, 1)
    compile(out, path, "exec")
    shutil.copy2(path, path + ".bak-teeth")
    open(path, "w", encoding="utf-8").write(out)
    print("%s patched   (backup: %s.bak-teeth)" % (os.path.basename(path), path))


def main():
    for p in (DISP, TONAL):
        if not os.path.isfile(p):
            die("no such file: %s" % p)
    patch(TONAL, TONAL_OLD, TONAL_NEW)
    patch(DISP, DISP_OLD, DISP_NEW)
    print("SYNTAX OK -- behaviour unchanged at default values")


if __name__ == "__main__":
    main()
