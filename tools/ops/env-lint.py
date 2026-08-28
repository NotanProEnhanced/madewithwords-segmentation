#!/usr/bin/env python3
"""Report .env keys that never reach the running container.

The check compose cannot do for you: for each tree, compare the keys defined in
.env against the environment of the running container. Anything defined but not
present is configuration that reads as real and does nothing.

Values are never printed -- only key names, and only whether they arrived.

    python3 env-lint.py            all trees
    python3 env-lint.py <tree>     one tree
"""
import os
import re
import subprocess
import sys

TREES = [
    ("typortrait-stg", "typortrait-staging"),
    ("typortrait-staging", "typortrait"),
    ("typortrait-faithinwords", "typortrait-faithinwords"),
    ("typortrait-lovedinwords", "typortrait-lovedinwords"),
    ("typortrait-pawsinwords", "typortrait-pawsinwords"),
]

KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")


def env_keys(path):
    keys = []
    try:
        for line in open(path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = KEY.match(line)
            if m:
                keys.append(m.group(1))
    except OSError as e:
        print("  cannot read %s: %s" % (path, e))
    return keys


def container_keys(name):
    try:
        out = subprocess.check_output(["docker", "exec", name, "env"],
                                      text=True, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        return None
    return {l.split("=", 1)[0] for l in out.splitlines() if "=" in l}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    bad = 0
    for tree, container in TREES:
        if only and only not in (tree, container):
            continue
        print("== %s  (container %s)" % (tree, container))
        keys = env_keys("/root/%s/typography_engine/.env" % tree)
        have = container_keys(container)
        if have is None:
            print("  container not running -- skipped")
            continue
        missing = [k for k in keys if k not in have]
        print("  .env keys: %d   reaching the container: %d"
              % (len(keys), len(keys) - len(missing)))
        if missing:
            bad += len(missing)
            print("  NOT REACHING THE CONTAINER:")
            for k in missing:
                print("    %s" % k)
        else:
            print("  all keys reach the container")
    print()
    print("total unreachable keys: %d" % bad)
    if bad:
        print("Fix by adding env_file to the compose service "
              "(tools/ops/compose-env-file.py), then recreating the container.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
