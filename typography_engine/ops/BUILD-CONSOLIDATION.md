# Build once, promote per tree (2026-09-03)

## Why

Each of the five trees used to run its own `docker compose up -d --build` — five
independent builds of what should be identical source, each keyed to whatever
happened to be checked out in that tree's own working directory at that moment.

That gap caused a real incident: four live brands sat nine commits behind while
`tt list`'s commit column looked current. `git pull` had updated each tree's working
tree; nothing had rebuilt the image running in the container. Git HEAD said one
thing, the served bytes said another, and only an independent check
(`tt verify "loupe/"`, which probes the live page rather than trusting the commit
hash) caught it.

## What changed

- `ops/build-image.sh` — builds ONE image, from a pinned commit via `git archive`
  (never from an ambient working tree, which might carry an uncommitted debug flag
  the way staging's `.env` once carried `TYPO_MASK_DEBUG`), tagged
  `typortrait:<short-sha>`.
- `ops/promote.sh` — points one tree at an already-built tag (sets `IMAGE_TAG` in
  that tree's `.env`, recreates the container) and then **verifies** the running
  container actually reports that image, rather than trusting the command's exit
  code. (`tt build` reporting success on zero matched trees is exactly the failure
  this guards against, one layer up.)
- All five `docker-compose*.yml` — `build: .` removed, `image:` now reads
  `${IMAGE_TAG}` everywhere (the base file already did this for prod; the other
  four previously hardcoded their own image name). Removing `build:` is
  deliberate: a stray `docker compose up -d --build` now fails loudly instead of
  silently rebuilding from whatever's on disk.
- `ops/tt build [tree...]` — now builds once and promotes the requested trees,
  instead of building each independently. Same command shape as before.
- `ops/tt promote <tree> [tag]` — new, thin wrapper around `promote.sh`.
- `ops/stg.sh restore` — its rebuild-if-the-saved-image-is-gone fallback now calls
  `build-image.sh` at the state's recorded commit instead of a bare
  `docker compose up -d --build`, which would fail with `build:` removed.

## What did NOT change

Five trees, five `.env` files, five containers, five ports — full isolation is
untouched. Brand skin is still resolved at request time from the `Host:` header,
unrelated to any of this. nginx is untouched. No registry is needed — one VPS,
one Docker daemon, image tags are shared automatically across all five
`docker compose` projects on that host.

## A bug this caught before it shipped

The first draft piped `git archive <sha>` straight into `docker build -`. `.git`
lives at the repo root, one level above `typography_engine/` — the same root that
holds `sites/` and everything else in the monorepo — so that would have built
successfully-looking images from the **wrong context**, with no `Dockerfile` at
the root `docker build` expects. Caught by extracting a real archive to a
directory and checking for `Dockerfile` before this ever ran against a live tree;
`build-image.sh` now does that check itself, every time, and refuses to build if
it fails.

**A second one, caught on the first real VPS run rather than in testing.** The
first version defaulted `SRC` to a hardcoded `/root/typortrait-prod`, regardless
of which tree's `./ops/build-image.sh` you actually ran. Run from staging right
after staging had just pulled the latest commit, it silently built from **prod's**
source instead — prod hadn't been pulled and was several commits behind — and
produced a real image, tagged and reported as "built" with no error, from stale
code. Caught only because the printed tag (`typortrait:af209dd`) didn't match the
commit that had just been pulled, and nobody had promoted it yet.

Fixed: `SRC` now defaults to the tree the invoked script itself lives in
(`dirname "$0"`, one level up — the same pattern `tt` already uses to find its
own path), not a hardcoded tree. `./ops/build-image.sh` run from any tree now
builds from that tree, always. `tt build` still builds from wherever `tt` itself
lives — prod's `ops/`, by convention — which is now stated explicitly in `tt`'s
own header rather than left as a silent assumption to rediscover the hard way.

## Rollback

Nothing here is one-way. To go back to per-tree builds on any file, add `build: .`
back next to its `image:` line — that's the whole revert. `ops/stg.sh` states
saved before this change restore exactly as before.

## Rolled out (2026-09-03)

Executed live against all five trees, backups taken first (`stg.sh save
pre-consolidation-<tree>` on each, plus a plain file copy of every `.env` and
compose file). Three real bugs found during the actual rollout, not in review:

1. `build-image.sh`'s git-repo detection (`[ -d "$TREE/.git" ]`) failed on the
   very first VPS run -- couldn't tell "inside a repo" from "at its root",
   and a worktree's `.git` is a file, not a directory. Replaced with
   `git rev-parse --show-toplevel`.
2. Under `set -euo pipefail`, a failing command inside a bare `VAR="$(cmd)"`
   assignment kills the script silently -- no message, just cmd's exit code.
   Found testing the fix above (a bad path exited 128 with nothing printed),
   then found the identical shape in `promote.sh`'s container-name lookup by
   testing for it specifically rather than assuming one instance was the
   only one.
3. The real near-miss: `build-image.sh` defaulted `SRC` to a hardcoded
   `/root/typortrait-prod`. Run from staging right after staging had just
   pulled, it silently built from **prod's** stale source instead --
   produced a real, successfully-tagged image from old code, no error
   anywhere. Caught only because the printed tag didn't match the commit
   just pulled, before anything was promoted with it. Fixed: `SRC` now
   defaults to the tree the invoked script itself lives in.

Final state: all five trees confirmed independently at three layers --
`promote.sh`'s own post-promotion check (`confirmed: container is running
<tag>`), `tt list`'s commit column, and `tt verify`'s live HTTP probe. All
five agreed: `a65d480`, `hits=2`, every tree.
