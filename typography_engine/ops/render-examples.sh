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
#   Also generates a "-card" JPEG (max 800px wide, quality 82) alongside every
#   -before.jpg and -after.png (2026-09-04) -- the /examples page displays each pair
#   at ~300-400px, so serving the full 1600px/multi-MB render there (as an earlier
#   version of this script did) meant several MB of images just to show three
#   thumbnails, hurting page weight and Core Web Vitals. The full-resolution
#   -after.png stays exactly where it was: the "View detail" link and og:image both
#   still need real detail/quality, just not on the card itself. -before-card.jpg /
#   -after-card.jpg are what app/main.py's templates actually reference now.
#
#   Card generation is a pure resize+recompress -- no render engine, no container
#   needed for it. Backfilling cards for pairs that were already rendered before this
#   feature existed costs nothing but a moment of CPU, not a re-render.
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
#   photos land; already-rendered pairs are NOT re-rendered unless FORCE=1 (FORCE=1
#   also regenerates every card, not just missing ones).
set -uo pipefail

TREE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${SRC:-${1:-$HOME/typortrait-examples-src}}"
CONTAINER="${CONTAINER:-typortrait-staging}"
FORCE="${FORCE:-}"
CARD_MAX_W="${CARD_MAX_W:-800}"
CARD_QUALITY="${CARD_QUALITY:-82}"

[ -d "$SRC" ] || { echo "no source directory at $SRC"; exit 1; }
docker inspect "$CONTAINER" >/dev/null 2>&1 || { echo "container '$CONTAINER' not running -- set CONTAINER=<name>"; exit 1; }

echo "source photos: $SRC"
echo "container:     $CONTAINER"
echo "tree:          $TREE"
echo "card size:     max ${CARD_MAX_W}px wide, quality $CARD_QUALITY"
echo

# Stage 0 (host): read the id/slug/brand/words worklist straight out of the running
# container so it can never drift from what the page itself serves.
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

# Stage 1 (host): for every id, bring -before.jpg / -before-card.jpg / -after-card.jpg
# up to date from whatever's already on disk (pure resize+recompress, no render engine
# needed for any of this) and decide which ids genuinely need a full render (-after.png
# missing, or FORCE=1) -- those go in TODO for stage 2.
TODO="$(python3 - "$WORKLIST" "$SRC" "$TREE" "$FORCE" "$CARD_MAX_W" "$CARD_QUALITY" <<'PY'
import json, os, shutil, sys
import cv2

items = json.loads(sys.argv[1])
src_dir, tree, force = sys.argv[2], sys.argv[3], sys.argv[4]
card_max_w, card_quality = int(sys.argv[5]), int(sys.argv[6])
force = (force == "1")

def make_card(src_path, dst_path):
    img = cv2.imread(src_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    if w > card_max_w:
        scale = card_max_w / w
        img = cv2.resize(img, (card_max_w, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, card_quality]))

todo = []
copied = missing = skipped = cards_made = 0
for it in items:
    slug, iid = it["slug"], it["id"]
    dest_dir = os.path.join(tree, "static", "examples", slug)
    before = os.path.join(dest_dir, f"{iid}-before.jpg")
    before_card = os.path.join(dest_dir, f"{iid}-before-card.jpg")
    after = os.path.join(dest_dir, f"{iid}-after.png")
    after_card = os.path.join(dest_dir, f"{iid}-after-card.jpg")

    src = None
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        cand = os.path.join(src_dir, f"{iid}{ext}")
        if os.path.exists(cand):
            src = cand
            break

    have_before = os.path.exists(before) and not force
    if not have_before:
        if src is None:
            if not os.path.exists(before):
                missing += 1
                continue
            # before.jpg exists from a prior run but force=True and no src to redo it from --
            # keep the existing before.jpg, just don't re-copy it.
        else:
            os.makedirs(dest_dir, exist_ok=True)
            if src.lower().endswith((".jpg", ".jpeg")):
                shutil.copy2(src, before)
            else:
                img = cv2.imread(src)
                if img is None:
                    print(f"could not read {src}, skipping {iid}", file=sys.stderr)
                    missing += 1
                    continue
                cv2.imwrite(before, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            copied += 1

    if os.path.exists(before) and (force or not os.path.exists(before_card)):
        if make_card(before, before_card):
            cards_made += 1

    need_render = force or not os.path.exists(after)
    if os.path.exists(after) and (force or not os.path.exists(after_card)):
        if make_card(after, after_card):
            cards_made += 1

    if need_render:
        if os.path.exists(before):
            todo.append(it)
        # else: no before photo at all yet (src missing) -- already counted in `missing`
    else:
        skipped += 1

print(f"copied {copied} before-photos, {missing} not found in {src_dir}, "
      f"{cards_made} card(s) generated/refreshed, {skipped} already fully rendered "
      f"(FORCE=1 to redo everything)", file=sys.stderr)
print(json.dumps(todo))
PY
)"

TODO_COUNT="$(printf '%s' "$TODO" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
if [ "$TODO_COUNT" = "0" ]; then
    echo "nothing new to render (cards may still have been generated/refreshed above)"
    exit 0
fi
echo "rendering $TODO_COUNT pair(s) inside $CONTAINER ..."
echo

# Stage 2 (container): render each pending pair with the correct engine, clean (no
# watermark), to /app/outputs/examples/<slug>/ -- a writable, host-visible path
# (data/outputs is bind-mounted), since /app/static is read-only inside the container.
# Also writes the -card.jpg right here (cheap, and the full-res bytes are already in
# memory) so a fresh render never needs a separate backfill pass.
docker exec -i "$CONTAINER" python3 - "$TODO" "$CARD_MAX_W" "$CARD_QUALITY" <<'PY'
import json, pathlib, sys, traceback
import cv2
import numpy as np

todo = json.loads(sys.argv[1])
card_max_w, card_quality = int(sys.argv[2]), int(sys.argv[3])
from app.config import RenderConfig
from app.pipeline.analyze import analyze_image
from app.pipeline.warnings import WarningCollector
from app.pipeline.displacement import render_displacement_portrait
from app.examples_content import EXAMPLES
from app.pet_proto import render_pet_portrait

def write_card(png_bytes, dst_path):
    arr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    if w > card_max_w:
        scale = card_max_w / w
        img = cv2.resize(img, (card_max_w, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, card_quality]))

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
    card_path = out_dir / f"{iid}-after-card.jpg"
    try:
        photo_bytes = before.read_bytes()
        if brand == "pawsinwords":
            # type_scale=0.56 is the real "Large" preset a customer can already pick
            # (_PET_TYPE_SCALES in app/main.py: small=0.30, medium=0.42, large=0.56).
            # Left unset before (falls back to "small"/0.30), the typography read as
            # too fine/hard to read on a small on-page card -- not an arbitrary bump,
            # the same size option the product itself already offers.
            png = render_pet_portrait(photo_bytes, img["words"], ground="dark",
                                      height=1600, print_aspect=0.8, type_scale=0.56)
        else:
            warns = WarningCollector()
            an = analyze_image(photo_bytes, RenderConfig(), warns)
            words_list = [w.strip() for w in img["words"].split(",") if w.strip()]
            # backdrop="studio" matches the live product's own default (static/index.html's
            # state.backdrop:"studio" -> BACKDROPS["studio"]=(230,230,230), a light neutral
            # gray). Omitting it (as an earlier version of this script did) renders the raw
            # navy ground with no backdrop recolor at all -- not what a real customer's
            # default render actually looks like.
            #
            # word_scale=120/57 is the real "Large" preset (WORD_SIZES in static/index.html:
            # Small=27, Medium=57, Large=120 -- word_scale is that value divided by the 57
            # Medium baseline). Left at the default 1.0 (Medium) before, the typography read
            # too small/hard to read on the on-page card -- same size option a customer can
            # already choose, not a made-up number.
            png = render_displacement_portrait(an, words_list, ground="navy",
                                               out_width=1600, supersample=2, ink="photo",
                                               print_aspect=0.8, backdrop="studio",
                                               word_scale=120.0 / 57.0)
        out_path.write_bytes(png)
        write_card(png, card_path)
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
    for name in (f"{iid}-after.png", f"{iid}-after-card.jpg"):
        src = os.path.join(tree, "data", "outputs", "examples", slug, name)
        dest = os.path.join(tree, "static", "examples", slug, name)
        if os.path.exists(src):
            shutil.copy2(src, dest)
            n += 1
print(f"{n} file(s) now live under static/examples/")
PY

echo
echo "done -- check a page now shows real images, e.g.:"
echo "  curl -s http://127.0.0.1:8078/examples/dog-portraits?brand=pawsinwords | grep -c 'P01-after-card.jpg'"
