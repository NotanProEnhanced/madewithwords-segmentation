#!/usr/bin/env python3
"""Report .env keys that never reach the running container, and unsafe settings.

The check compose cannot do for you: for each tree, compare the keys defined in
.env against the environment of the running container. Anything defined but not
present is configuration that reads as real and does nothing.

It also refuses one specific dangerous combination. Staging holds a real
Printful token so the physical purchase path can be tested end to end; the only
thing stopping it sending real print jobs to a real printer is
PRINTFUL_CONFIRM=false. Printful has no test mode, so that one word is the
entire safety margin, and it lives in a file that a future 'copy prod's .env
across' would silently overwrite. This makes that mistake visible in a second
rather than at the printer.

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
    ("typortrait-prod", "typortrait"),
    ("typortrait-faithinwords", "typortrait-faithinwords"),
    ("typortrait-lovedinwords", "typortrait-lovedinwords"),
    ("typortrait-pawsinwords", "typortrait-pawsinwords"),
]

KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")

# Trees that must never auto-confirm a Printful order. Production is absent
# deliberately -- confirming is exactly what it is for.
NO_CONFIRM = {"typortrait-stg"}


def env_values(path):
    """key -> value, for the few checks that need to look at a value. Nothing
    read here is ever printed."""
    out = {}
    try:
        for line in open(path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = KEY.match(line)
            if m:
                out[m.group(1)] = line[len(m.group(1)) + 1:].strip().strip('"\'')
    except OSError:
        pass
    return out


def check_confirm(tree, path):
    """Return a list of problems. A non-production tree with a working Printful
    token and confirm left on will send real orders to a real printer."""
    env = env_values(path)
    if tree not in NO_CONFIRM:
        return []
    token = env.get("PRINTFUL_API_TOKEN", "")
    confirm = env.get("PRINTFUL_CONFIRM", "true").strip().lower()
    if token and confirm not in ("0", "false", "no", "off"):
        return ["DANGER: PRINTFUL_API_TOKEN is set and PRINTFUL_CONFIRM=%s.\n"
                "    A test purchase on this tree will send a REAL order to the printer\n"
                "    and bill you. Printful has no test mode. Set PRINTFUL_CONFIRM=false\n"
                "    and recreate the container before testing anything."
                % (confirm or "(unset -> defaults to true)")]
    return []


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
    danger = 0
    for tree, container in TREES:
        if only and only not in (tree, container):
            continue
        print("== %s  (container %s)" % (tree, container))
        path = "/root/%s/typography_engine/.env" % tree
        keys = env_keys(path)
        for problem in check_confirm(tree, path):
            danger += 1
            print("  %s" % problem)
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
              "(ops/compose-env-file.py), then recreating the container.")
    if danger:
        print("UNSAFE SETTINGS: %d -- see DANGER above. Do not run a test purchase." % danger)
    sys.exit(1 if (bad or danger) else 0)


if __name__ == "__main__":
    main()
