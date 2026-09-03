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

## Rollback

Nothing here is one-way. To go back to per-tree builds on any file, add `build: .`
back next to its `image:` line — that's the whole revert. `ops/stg.sh` states
saved before this change restore exactly as before.
