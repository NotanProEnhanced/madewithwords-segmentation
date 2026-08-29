#!/usr/bin/env python3
"""Build one compose template from the five hand-maintained files.

WHY IT IS SAFE HERE
  compose-diff.py measured the five files: 101 of 136 environment keys identical,
  ZERO with differing values, and 35 present in some trees and not others. So the
  only genuine per-tree differences are the project name, image tag, container
  name and host port.

  With env_file in place, an entry like `- TYPO_FOO=${TYPO_FOO:-}` on a tree whose
  .env lacks TYPO_FOO yields an empty string -- which is exactly what "absent"
  already meant. Adding the union of keys to every tree is therefore a no-op for
  behaviour.

THE ONE HAZARD, AND HOW IT IS HANDLED
  A union entry with a NON-EMPTY default -- ${X:-0.45} -- added to a tree that did
  not have that line would newly impose 0.45 where the code's own default applied
  before. Those are listed under "REVIEW" and are NOT silently adopted: the script
  reports them and you decide before applying.

Usage:
    python3 compose-template.py            analyse and write the template to /tmp
    python3 compose-template.py --apply    also install it in every tree
                                           (backs up each existing file first)
"""
import os
import re
import shutil
import sys

TREES = [
    # (directory, project name, image tag, container name, host port)
    ("typortrait-stg", "typortrait-staging", "typortrait-staging:latest", "typortrait-staging", "8078"),
    ("typortrait-prod", "typortrait-prod", "typortrait:latest", "typortrait", "8077"),
    ("typortrait-faithinwords", "faithinwords", "typortrait-faithinwords:latest", "typortrait-faithinwords", "8079"),
    ("typortrait-lovedinwords", "lovedinwords", "typortrait-lovedinwords:latest", "typortrait-lovedinwords", "8080"),
    ("typortrait-pawsinwords", "pawsinwords", "typortrait-pawsinwords:latest", "typortrait-pawsinwords", "8081"),
]

ENVLINE = re.compile(r"^(\s*)-\s+([A-Z][A-Z0-9_]*)=(.*)$")
DEFAULTED = re.compile(r"^\$\{[A-Z0-9_]+:-(.+)\}$")

APPLY = "--apply" in sys.argv
OUT = "/tmp/docker-compose.template.yml"


def compose_path(tree):
    return "/root/%s/typography_engine/docker-compose.yml" % tree


def main():
    per_tree = {}
    for tree, *_rest in TREES:
        p = compose_path(tree)
        if not os.path.isfile(p):
            raise SystemExit("missing: %s" % p)
        env = {}
        for ln in open(p, encoding="utf-8"):
            m = ENVLINE.match(ln)
            if m:
                env[m.group(2)] = (m.group(1), m.group(3).strip())
        per_tree[tree] = env

    union = {}
    for tree, env in per_tree.items():
        for k, (indent, rhs) in env.items():
            union.setdefault(k, (indent, rhs))

    review = []
    for k, (_i, rhs) in sorted(union.items()):
        d = DEFAULTED.match(rhs)
        if not d:
            continue
        absent = [t for t, e in per_tree.items() if k not in e]
        if absent:
            review.append((k, d.group(1), absent))

    print("union of environment keys: %d" % len(union))
    print()
    if review:
        print("REVIEW -- these carry a non-empty compose default and would be ADDED")
        print("to trees that did not have them. Confirm the default matches what the")
        print("code already does, or the tree's behaviour changes:")
        for k, dflt, absent in review:
            print("  %-28s default=%-10s would newly apply to: %s"
                  % (k, dflt, ", ".join(absent)))
        print()
    else:
        print("No union key carries a non-empty default that is missing anywhere.")
        print("Adding the union to every tree is behaviour-neutral.")
        print()

    # the template body is taken from the staging file, with the per-service
    # values substituted and the environment list replaced by the union
    base = open(compose_path("typortrait-stg"), encoding="utf-8").read().splitlines(True)
    out, in_env, written = [], False, False
    for ln in base:
        m = ENVLINE.match(ln)
        if m:
            in_env = True
            if not written:
                indent = m.group(1)
                for k, (_i, rhs) in sorted(union.items()):
                    out.append("%s- %s=%s\n" % (indent, k, rhs))
                written = True
            continue
        if in_env and not m:
            in_env = False
        s = ln.strip()
        if s.startswith("name:"):
            out.append("name: ${COMPOSE_PROJECT}\n")
        elif s.startswith("image:"):
            out.append(ln.replace(s, "image: ${IMAGE_TAG}"))
        elif s.startswith("container_name:"):
            out.append(ln.replace(s, "container_name: ${CONTAINER_NAME}"))
        elif re.match(r'^-\s*"127\.0\.0\.1:\d+:8077"$', s):
            out.append(ln.replace(s, '- "127.0.0.1:${HOST_PORT}:8077"'))
        else:
            out.append(ln)

    open(OUT, "w", encoding="utf-8").write("".join(out))
    print("template written to %s (%d lines)" % (OUT, len(out)))
    print()
    print("Each tree's .env needs these four lines:")
    for tree, proj, image, cname, port in TREES:
        print("  %-26s COMPOSE_PROJECT=%s IMAGE_TAG=%s CONTAINER_NAME=%s HOST_PORT=%s"
              % (tree, proj, image, cname, port))

    if not APPLY:
        print()
        print("DRY RUN -- nothing installed. Re-run with --apply after reading REVIEW above.")
        return

    for tree, proj, image, cname, port in TREES:
        p = compose_path(tree)
        shutil.copy2(p, p + ".bak-template")
        shutil.copy2(OUT, p)
        e = "/root/%s/typography_engine/.env" % tree
        cur = open(e, encoding="utf-8").read()
        add = []
        for k, v in (("COMPOSE_PROJECT", proj), ("IMAGE_TAG", image),
                     ("CONTAINER_NAME", cname), ("HOST_PORT", port)):
            if not re.search(r"^%s=" % k, cur, re.M):
                add.append("%s=%s\n" % (k, v))
        if add:
            with open(e, "a", encoding="utf-8") as fh:
                fh.write("\n# per-tree identity, consumed by the shared compose template\n")
                fh.writelines(add)
        print("installed template in %s (backup .bak-template), .env +%d lines"
              % (tree, len(add)))
    print()
    print("Now, ONE TREE AT A TIME:")
    print("  cd /root/<tree>/typography_engine && docker compose config >/dev/null && docker compose up -d")
    print("Check `docker compose config` succeeds before `up` -- a bad substitution")
    print("shows up there rather than by taking a service down.")


if __name__ == "__main__":
    main()
