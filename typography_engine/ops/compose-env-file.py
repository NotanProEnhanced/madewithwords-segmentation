#!/usr/bin/env python3
"""Make every variable in .env actually reach the container.

THE PROBLEM
  Each compose file hand-maintains an `environment:` list. Compose reads .env for
  ${VAR} interpolation, so a value there LOOKS configured -- but unless the
  variable is also listed under `environment:`, it never reaches the process.

  That cost three separate rounds of debugging in one day:
    TYPO_TEETH_DEBUG   set in .env, absent from compose -> no debug output, and
                       time spent looking for a bug in the code instead
    TYPO_SEG_PAD       set in four production .env files, absent from four
                       compose files -> the fix appeared deployed and was not
    TYPO_IRIS_MIN_PX   same, caught only because we checked the containers

  In every case .env read as configuration and behaved as a suggestion.

THE FIX
  Add `env_file: [.env]` to the service. Compose then passes the whole file into
  the container, so a variable in .env can never again be silently dropped.

  The existing `environment:` list is LEFT IN PLACE. It still supplies defaults
  via ${VAR:-default} for variables not present in .env, and an explicit
  environment entry takes precedence over env_file, so nothing changes for any
  variable that is already wired correctly. This only closes the hole.

  Note this does pass every .env key to the process, including credentials --
  which the service already needs and already receives by other means. Nothing
  new is exposed outside the container.

Usage:  python3 compose-env-file.py <tree>/typography_engine [--apply]
Dry run by default. Idempotent.
"""
import os
import re
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/typortrait-stg/typography_engine"
APPLY = "--apply" in sys.argv
PATH = os.path.join(ROOT, "docker-compose.yml")


def main():
    if not os.path.isfile(PATH):
        raise SystemExit("no such file: %s" % PATH)
    src = open(PATH, encoding="utf-8").read()

    if re.search(r"^\s*env_file:", src, re.M):
        print("%s: already has env_file -- no change" % PATH)
        return

    m = re.search(r"^(\s*)environment:\s*$", src, re.M)
    if not m:
        raise SystemExit("ABORTED: no 'environment:' key found in %s" % PATH)
    indent = m.group(1)

    block = ("%senv_file:\n"
             "%s  # Everything in .env reaches the container. Without this a value set\n"
             "%s  # in .env but missing from the environment list below is silently\n"
             "%s  # ignored -- which happened three times in one day.\n"
             "%s  - .env\n" % (indent, indent, indent, indent, indent))
    out = src[:m.start()] + block + src[m.start():]

    if not APPLY:
        print("DRY RUN -- would insert before 'environment:' in %s:\n" % PATH)
        print(block)
        print("Re-run with --apply to write.")
        return

    shutil.copy2(PATH, PATH + ".bak-envfile")
    open(PATH, "w", encoding="utf-8").write(out)
    print("patched %s   (backup: %s.bak-envfile)" % (PATH, PATH))


if __name__ == "__main__":
    main()
