#!/usr/bin/env python3
"""What actually differs between the five compose files.

Before replacing five hand-maintained files with one committed template, find out
how far apart they really are. If the only differences are the project name, the
image tag, the container name and the host port, the template is trivial and
safe. If dozens of environment lines diverge, it is a bigger job and the
divergence itself needs deciding on first.

Reports, per key:
  IDENTICAL   same in all five
  VARIES      differs -- shows the value per tree
  MISSING     absent from some trees

Values of anything that looks like a credential are masked; only presence and
whether it varies is shown.

    python3 compose-diff.py
"""
import os
import re
import sys

TREES = [
    "typortrait-stg",
    "typortrait-prod",
    "typortrait-faithinwords",
    "typortrait-lovedinwords",
    "typortrait-pawsinwords",
]

SECRET = re.compile(r"(KEY|SECRET|TOKEN|PASS|PWD|WEBHOOK|CRED|DSN|SIGNING)", re.I)
ENVLINE = re.compile(r"^\s*-\s+([A-Z][A-Z0-9_]*)=(.*)$")
TOPKEY = re.compile(r"^([a-z_]+):\s*(.*)$")


def parse(path):
    """Return (env_vars, top_level_keys, other_lines) for one compose file."""
    env, top, other = {}, {}, []
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError as e:
        return None, None, ["cannot read: %s" % e]
    for ln in lines:
        m = ENVLINE.match(ln)
        if m:
            env[m.group(1)] = m.group(2).strip()
            continue
        m = TOPKEY.match(ln)
        if m and m.group(1) in ("name", "services", "networks", "volumes"):
            top[m.group(1)] = m.group(2).strip()
            continue
        s = ln.strip()
        if s.startswith(("image:", "container_name:", "restart:", "- \"127.0.0.1:")):
            other.append(s)
    return env, top, other


def mask(k, v):
    return "<set>" if SECRET.search(k) and v else v


def main():
    data = {}
    for t in TREES:
        p = "/root/%s/typography_engine/docker-compose.yml" % t
        env, top, other = parse(p)
        if env is None:
            print("%s: %s" % (t, other[0]))
            continue
        data[t] = (env, top, other)

    if len(data) < 2:
        raise SystemExit("need at least two readable compose files")

    print("=== per-service settings ===")
    for t, (_e, _top, other) in data.items():
        print("%-26s %s" % (t, " | ".join(other) if other else "(none found)"))
    print()
    print("project name:")
    for t, (_e, top, _o) in data.items():
        print("  %-26s %s" % (t, top.get("name", "(derived from directory)")))

    keys = sorted({k for e, _t, _o in data.values() for k in e})
    identical, varies, missing = [], [], []
    for k in keys:
        present = {t: e.get(k) for t, (e, _t, _o) in data.items()}
        absent = [t for t, v in present.items() if v is None]
        vals = {v for v in present.values() if v is not None}
        if absent:
            missing.append((k, absent, vals))
        elif len(vals) == 1:
            identical.append(k)
        else:
            varies.append((k, present))

    print()
    print("=== environment keys: %d total ===" % len(keys))
    print("identical in all trees: %d" % len(identical))
    print()
    if varies:
        print("VARIES (%d):" % len(varies))
        for k, present in varies:
            print("  %s" % k)
            for t, v in present.items():
                print("      %-26s %s" % (t, mask(k, v)))
    if missing:
        print()
        print("MISSING from some trees (%d):" % len(missing))
        for k, absent, vals in missing:
            print("  %-28s absent from: %s" % (k, ", ".join(absent)))

    print()
    print("A template is straightforward if VARIES is only the project name,")
    print("image, container name and port. Anything else has to be a deliberate")
    print("per-brand override rather than drift.")


if __name__ == "__main__":
    main()
