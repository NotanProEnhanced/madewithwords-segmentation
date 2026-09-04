#!/bin/bash
# Batch-render the /examples/<slug> gallery pairs from source photos you've already
# generated (via ChatGPT, see ops/GALLERY-SOURCES.md) and downloaded to the VPS.
#
# WHY
#   app/examples_content.py is the single source of truth for which image ids exist,
#   which category/slug each belongs to, and the exact word list to render onto each
#   one. This script reads that data directly (imported inside the running container,
#   never re-typed here) so there is exactly one place these facts can drift out of
#   sync, and it's a Python module the /examples page itself already depends on.
#
#   Renders CLEAN (no watermark) at web resolution, not full print quality -- these
#   are permanent public marketing images, not a customer's paid download, and the
#   public /render endpoint always stamps a preview watermark, so this calls the
#   render functions directly (same approach _studio_render_png already uses for the
#   existing pre-made storefront gallery) instead of going through /render.
#
#   Pet categories (brand: pawsinwords) use the landmark-free pet engine
#   (render_pet_portrait); human categories use the face-landmark engine
#   (analyze_image + render_displacement_portrait) -- running the wrong one on a
#   photo of a dog would either fail outright or silently produce nonsense, so the
#   script picks the engine from each category's own "brand" field, not a guess.
#
# USE
#   1. Generate the prompts in ops/GALLERY-SOURCES.md via ChatGPT, download each PNG/JPG.
#   2. Get them onto the VPS, named exactly <ID>.jpg (H04.jpg, P09.jpg, ...) inside one
#      directory -- e.g. scp them to ~/typortrait-examples-src/.
#   3. From a tree's typography_engine/ directory:
#        ./ops/render-examples.sh ~/typortrait-examples-src
#      or with a different container / tree:
#        SRC=~/typortrait-examples-src CONTAINER=typortrait-staging ./ops/render-examples.sh
#
#   Missing source files are skipped with a clear note -- safe to re-run as more
#   photos land; already-rendered pairs are NOT re-rendered unless FORCE=1.
set -uo pipefail

TREE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${SRC:-${1:-$HOME/typortrait-examples-src}}"
CONTAINER="${CONTAINER:-typortrait-staging}"
FORCE="${FORCE:-}"

[ -d "$SRC" ] || { echo "no source directory at $SRC"; exit 1; }
docker inspect "$CONTAINER" >/dev/null 2>&1 || { echo "container '$CONTAINER' not running -- set CONTAINER=<name>"; exit 1; }

echo "source photos: $SRC"
echo "container:     $CONTAINER"
echo "tree:          $TREE"
echo

# Stage 1 (host): copy every available <ID>.jpg into static/examples/<slug>/<ID>-before.jpg,
# and build the id->slug worklist for stage 2. Reads the category data straight out of the
# container so the id/slug/words list can never drift from what the page itself serves.
WORKLIST="$(docker exec -i "$CONTAINER" python3 - <<'PY'
import json
from app.examples_content import EXAMPLES
out = []
for slug, cat in EXAMPLES.items():
    for img in cat["images"]:
        out.append({"slug": slug, "id": img["id"], "brand": cat["brand"]})
print(json.dumps(out))
PY
)"
[ -n "$WORKLIST" ] || { echo "could not read app.examples_content.EXAMPLES from the container"; exit 1; }

copied=0; missing=0; skipped=0
TODO="[]"
TODO="$(python3 - "$WORKLIST" "$SRC" "$TREE" "$FORCE" <<'PY'
import json, os, shutil, sys
items = json.loads(sys.argv[1])
src_dir, tree, force = sys.argv[2], sys.argv[3], sys.argv[4]
todo = []
copied = missing = skipped = 0
for it in items:
    slug, iid = it["slug"], it["id"]
    src = os.path.join(src_dir, f"{iid}.jpg")
    dest_dir = os.path.join(tree, "static", "examples", slug)
    before = os.path.join(dest_dir, f"{iid}-before.jpg")
    after = os.path.join(dest_dir, f"{iid}-after.png")
    if not os.path.exists(src):
        missing += 1
        continue
    if os.path.exists(after) and force != "1":
        skipped += 1
        continue
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src, before)
    copied += 1
    todo.append(it)
print(f"copied {copied} before-photos, {missing} not found in {src_dir}, "
      f"{skipped} already rendered (FORCE=1 to redo)", file=sys.stderr)
print(json.dumps(todo))
PY
)"

TODO_COUNT="$(printf '%s' "$TODO" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
if [ "$TODO_COUNT" = "0" ]; then
    echo "nothing new to render"
    exit 0
fi
echo "rendering $TODO_COUNT pair(s) inside $CONTAINER ..."
echo

# Stage 2 (container): render each pending pair with the correct engine, clean (no
# watermark), to /app/outputs/examples/<slug>/ -- a writable, host-visible path
# (data/outputs is bind-mounted), since /app/static is read-only inside the container.
docker exec -i "$CONTAINER" python3 - "$TODO" <<'PY'
import json, pathlib, sys, traceback

todo = json.loads(sys.argv[1])
from app.config import RenderConfig
from app.pipeline.analyze import analyze_image
from app.pipeline.warnings import WarningCollector
from app.pipeline.displacement import render_displacement_portrait
from app.examples_content import EXAMPLES
from app.pet_proto import render_pet_portrait

OUT_ROOT = pathlib.Path("/app/outputs/examples")
STATIC_ROOT = pathlib.Path("/app/static/examples")
ok = fail = 0
for it in todo:
    slug, iid, brand = it["slug"], it["id"], it["brand"]
    img = next(i for i in EXAMPLES[slug]["images"] if i["id"] == iid)
    before = STATIC_ROOT / slug / f"{iid}-before.jpg"
    out_dir = OUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{iid}-after.png"
    try:
        photo_bytes = before.read_bytes()
        if brand == "pawsinwords":
            png = render_pet_portrait(photo_bytes, img["words"], ground="dark",
                                      height=1600, print_aspect=0.8)
        else:
            warns = WarningCollector()
            an = analyze_image(photo_bytes, RenderConfig(), warns)
            words_list = [w.strip() for w in img["words"].split(",") if w.strip()]
            png = render_displacement_portrait(an, words_list, ground="navy",
                                               out_width=1600, supersample=2,
                                               ink="photo", print_aspect=0.8)
        out_path.write_bytes(png)
        print(f"OK    {slug}/{iid}  ({len(png)} bytes)")
        ok += 1
    except Exception as e:  # noqa: BLE001 -- report and continue, one bad photo shouldn't stop the batch
        print(f"FAIL  {slug}/{iid}  {type(e).__name__}: {e}")
        traceback.print_exc()
        fail += 1
print(f"\n{ok} rendered, {fail} failed")
PY

echo
echo "copying rendered files back into static/examples/ ..."
python3 - "$TODO" "$TREE" <<'PY'
import json, os, shutil, sys
items = json.loads(sys.argv[1])
tree = sys.argv[2]
n = 0
for it in items:
    slug, iid = it["slug"], it["id"]
    src = os.path.join(tree, "data", "outputs", "examples", slug, f"{iid}-after.png")
    dest = os.path.join(tree, "static", "examples", slug, f"{iid}-after.png")
    if os.path.exists(src):
        shutil.copy2(src, dest)
        n += 1
print(f"{n} rendered pair(s) now live under static/examples/")
PY

echo
echo "done -- check a page now shows real images, e.g.:"
echo "  curl -s http://127.0.0.1:8078/examples/dog-portraits?brand=pawsinwords | grep -c 'P01-after.png'"
