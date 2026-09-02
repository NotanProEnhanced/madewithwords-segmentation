#!/usr/bin/env python3
"""Do the compose defaults and the code defaults agree?

WHY
  A setting has two defaults: the one in the code, `os.environ.get("X", "0")`, and the one
  in docker-compose.yml, `X=${X:-1}`. Compose wins -- it puts the variable in the container's
  environment, so the code's default is never reached.

  When they disagree, reading the code tells you one thing and the running container does
  another, and nothing announces it. That cost most of a day: a portrait rendered dark, and
  TYPO_POLARITY read as "0" in displacement.py while compose had been defaulting it to 1 in
  every container since it was written. Four separate explanations were proposed and
  discarded before anyone ran `printenv`.

  It also means `.env` is not the source of truth. Absence from `.env` does not mean a
  setting is off.

WHAT IT REPORTS
  * defaults that disagree -- the trap above
  * settings the code reads that compose never passes, so `.env` cannot reach them unless
    env_file happens to be declared
  * settings compose passes that no code reads -- dead configuration

WRITES NOTHING. Reads two files and prints.

    ./ops/env-default-audit.py
    ./ops/env-default-audit.py --quiet      only the disagreements (exit 1 if any)
"""
import collections
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
COMPOSE = HERE / "docker-compose.yml"
APP = HERE / "app"


def _norm(v):
    """Compare 1 with 1.0, and On with on, without treating 0.8 and 0.95 as equal."""
    if v is None:
        return None
    v = v.strip()
    try:
        return "%g" % float(v)
    except ValueError:
        return v.lower()


def main() -> int:
    quiet = "--quiet" in sys.argv
    compose = COMPOSE.read_text()

    passed = {}          # VAR -> compose default (None when ${VAR} with no default)
    for m in re.finditer(r'-\s*([A-Z][A-Z0-9_]*)=\$\{\1(?::-([^}]*))?\}', compose):
        passed[m.group(1)] = m.group(2)
    for m in re.finditer(r'-\s*([A-Z][A-Z0-9_]*)=([^\s${][^\n]*)', compose):
        passed.setdefault(m.group(1), m.group(2).strip())

    read = collections.defaultdict(set)      # VAR -> {(code default, where)}
    for p in sorted(APP.rglob("*.py")):
        txt = p.read_text()
        for m in re.finditer(r'os\.environ\.get\(\s*"([A-Z][A-Z0-9_]*)"\s*,\s*"([^"]*)"\s*\)', txt):
            read[m.group(1)].add((m.group(2), "%s:%d" % (p.relative_to(HERE),
                                                         txt[:m.start()].count("\n") + 1)))
        for m in re.finditer(r'os\.(?:environ\.get|getenv)\(\s*"([A-Z][A-Z0-9_]*)"\s*[,)]', txt):
            read[m.group(1)].add((None, "%s:%d" % (p.relative_to(HERE),
                                                   txt[:m.start()].count("\n") + 1)))
        # config.py wraps os.environ in typed helpers. Missing these made every setting they
        # own look like dead configuration -- including the retention period and the price.
        for m in re.finditer(r'env_(?:int|float|bool|str)\(\s*"([A-Z][A-Z0-9_]*)"\s*,\s*'
                             r'([^),]+)\s*[,)]', txt):
            read[m.group(1)].add((m.group(2).strip().strip('"\''),
                                  "%s:%d" % (p.relative_to(HERE),
                                             txt[:m.start()].count("\n") + 1)))

    # Duplicates. A variable listed twice in `environment:` takes the LAST entry silently.
    # TYPO_LAYERED_PHOTO was declared with :-0 and again with :-1; the file read as if the
    # feature were off while the container had it on, and neither line looked wrong alone.
    dupes = collections.Counter(
        m.group(1) for m in re.finditer(r'^\s+-\s*([A-Z][A-Z0-9_]*)=', compose, re.M))
    dupes = {k: n for k, n in dupes.items() if n > 1}
    if dupes:
        print("DECLARED MORE THAN ONCE  (the last entry wins, silently)")
        for k, n in sorted(dupes.items()):
            print("  %-30s %d times" % (k, n))
        print()

    bad = []
    for var, cdef in sorted(passed.items()):
        if cdef in (None, "") or var not in read:
            continue
        for kdef, where in sorted(read[var], key=lambda t: t[1]):
            if kdef is not None and _norm(cdef) != _norm(kdef):
                bad.append((var, cdef, kdef, where))

    print("DEFAULTS THAT DISAGREE  (compose wins; the code default is never reached)")
    for var, cdef, kdef, where in bad:
        print("  %-30s compose=%-16s code=%-10s %s" % (var, cdef, kdef, where))
    if not bad:
        print("  none")

    if not quiet:
        print("\nREAD BY THE CODE, NOT PASSED BY COMPOSE")
        missing = sorted(v for v in read if v not in passed)
        for var in missing:
            print("  %-30s %s" % (var, sorted(read[var], key=lambda t: t[1])[0][1]))
        if not missing:
            print("  none")

        print("\nPASSED BY COMPOSE, READ BY NOTHING  (dead configuration)")
        dead = sorted(v for v in passed if v not in read)
        for var in dead:
            print("  %-30s" % var)
        if not dead:
            print("  none")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
