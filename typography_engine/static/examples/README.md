# /static/examples/<slug>/ — showcase image pairs

Drop-in convention for `/examples/<slug>` (see `app/examples_content.py` for
the category data and `ops/GALLERY-SOURCES.md` for the source prompts + word
lists). No code change or redeploy needed to add images — the page renders
with whichever pairs are present and skips the rest.

For each image id listed in a category's `images` list, `ops/render-examples.sh`
generates FOUR files (2026-09-04) — don't create these by hand, run the script:

```
static/examples/<slug>/<id>-before.jpg        full-resolution source photo
static/examples/<slug>/<id>-after.png         full-resolution rendered portrait (1600px)
static/examples/<slug>/<id>-before-card.jpg   resized/compressed source (max 800px, q82)
static/examples/<slug>/<id>-after-card.jpg    resized/compressed portrait (max 800px, q82)
```

The `-card.jpg` files are what the page actually displays (the grid, the hero
image, `og:image`/`twitter:image`) — the on-page card is only ~300-400px, so
serving the full 1600px/multi-MB render there wasted page weight and hurt
Core Web Vitals. The full-resolution originals are still used: the "View
detail" link opens the full `-after.png`, and the "Look closer" zoomed-detail
section deliberately keeps the full-res file too (it CSS-scales into a crop,
which needs real resolution the 800px card doesn't have).

A page requires **all four** files for an image before showing it — if only
the two originals exist (e.g. rendered before this feature existed), the
image is treated as not-ready until `ops/render-examples.sh` backfills its
cards, which it does automatically and without re-rendering (pure
resize+recompress, no render engine needed) the next time it's run.

Example: `couple-anniversary-portraits` needs H10, H19, H20 — 12 files total
(4 each), all produced by one `render-examples.sh` run.

The page itself renders as soon as the slug exists in `app/examples_content.py`
and the request is on the right brand host, even with zero pairs present yet
(it shows a "more examples on the way" note) — but it's only added to
`sitemap.xml` once at least one pair is fully ready, so an empty or
still-backfilling page is never what a crawler finds.
