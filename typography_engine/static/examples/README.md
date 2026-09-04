# /static/examples/<slug>/ — showcase image pairs

Drop-in convention for `/examples/<slug>` (see `app/examples_content.py` for
the category data and `ops/GALLERY-SOURCES.md` for the source prompts + word
lists). No code change or redeploy needed to add images — the page renders
with whichever pairs are present and skips the rest.

For each image id listed in a category's `images` list, add two files:

```
static/examples/<slug>/<id>-before.jpg   the source photo
static/examples/<slug>/<id>-after.png    the rendered typography portrait
```

Example: `couple-anniversary-portraits` needs H10, H19, H20 —

```
static/examples/couple-anniversary-portraits/H10-before.jpg
static/examples/couple-anniversary-portraits/H10-after.png
static/examples/couple-anniversary-portraits/H19-before.jpg
static/examples/couple-anniversary-portraits/H19-after.png
static/examples/couple-anniversary-portraits/H20-before.jpg
static/examples/couple-anniversary-portraits/H20-after.png
```

The page itself renders as soon as the slug exists in `app/examples_content.py`
and the request is on the right brand host, even with zero pairs present yet
(it shows a "more examples on the way" note) — but it's only added to
`sitemap.xml` once at least one pair is on disk, so an empty page is never
what a crawler finds.
