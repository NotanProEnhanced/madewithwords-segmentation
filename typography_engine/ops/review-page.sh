#!/bin/bash
# Build a side-by-side review page for two test-set runs, served from staging.
#
# WHY
#   compare-testset.sh answers "which images moved" exactly, and that is enough for most
#   changes: the engine is deterministic, so same bytes means nothing moved. But a change
#   like swapping the matte model moves ALL TEN by definition, and then the only question
#   left -- is each one BETTER -- has no byte answer. It has to be looked at.
#
#   Looking at it was the hard part. The renders live on the server, the eye is on a laptop,
#   and flipping between two directories over SFTP loses the comparison. This puts both runs
#   on one page, each pair adjacent, at the size they were rendered.
#
# USE
#   ./review-page.sh <run A> <run B>        e.g. ./review-page.sh e2d870d-dirty 7bb80e8-isnet
#   ./review-page.sh <run A> <run B> --rm   remove a previously built page and its copies
#
#   Then open the URL it prints. Staging is behind basic auth, so nothing is public --
#   but it IS served, so remove the page when the decision is made.
set -uo pipefail

SET="${SET:-/root/typortrait-testset}"
TREE="${TREE:-/root/typortrait-stg}"
DEST="$TREE/typography_engine/static/review"
BASE="${BASE:-https://staging.typortrait.com}"

A="${1:-}"; B="${2:-}"
if [ "${3:-}" = "--rm" ] || [ "${1:-}" = "--rm" ]; then
    rm -rf "$DEST"; echo "removed $DEST"; exit 0
fi
[ -n "$A" ] && [ -n "$B" ] || { echo "usage: $0 <run A> <run B>   (or --rm)"; exit 1; }
[ -d "$SET/out/$A" ] || { echo "no run at $SET/out/$A"; exit 1; }
[ -d "$SET/out/$B" ] || { echo "no run at $SET/out/$B"; exit 1; }

rm -rf "$DEST"; mkdir -p "$DEST/$A" "$DEST/$B"
cp "$SET/out/$A"/*.png "$DEST/$A/" 2>/dev/null
cp "$SET/out/$B"/*.png "$DEST/$B/" 2>/dev/null

{
cat <<HTML
<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>$A vs $B</title>
<style>
  :root{color-scheme:light dark}
  body{margin:0;padding:20px;font:14px/1.5 system-ui,sans-serif;background:#111;color:#eee}
  h1{font-size:16px;font-weight:600;margin:0 0 4px}
  p.note{margin:0 0 20px;color:#9aa}
  .row{margin:0 0 28px}
  .name{font:600 13px/1.4 ui-monospace,monospace;color:#cfd3da;margin:0 0 6px}
  .pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  figure{margin:0}
  figcaption{font:11px/1.4 ui-monospace,monospace;color:#8b93a1;padding:4px 0}
  img{width:100%;height:auto;display:block;background:#000;border:1px solid #2a2f3a}
  a.full{color:#9ec5ff;text-decoration:none;font-size:11px}
</style>
<h1>$A &nbsp;vs&nbsp; $B</h1>
<p class="note">Left is A, right is B. Click either to open it at full size. Judge the hard
cases first: dark-on-dark, white-hair, couple, hat.</p>
HTML
for f in $(ls -1 "$DEST/$A" | sort); do
    [ -f "$DEST/$B/$f" ] || continue
    printf '<div class="row"><div class="name">%s</div><div class="pair">' "$f"
    printf '<figure><a href="%s/%s"><img src="%s/%s" alt="%s in %s"></a><figcaption>A &middot; %s <a class="full" href="%s/%s">full size</a></figcaption></figure>' "$A" "$f" "$A" "$f" "$f" "$A" "$A" "$A" "$f"
    printf '<figure><a href="%s/%s"><img src="%s/%s" alt="%s in %s"></a><figcaption>B &middot; %s <a class="full" href="%s/%s">full size</a></figcaption></figure>' "$B" "$f" "$B" "$f" "$f" "$B" "$B" "$B" "$f"
    printf '</div></div>\n'
done
} > "$DEST/index.html"

n=$(ls -1 "$DEST/$A"/*.png 2>/dev/null | wc -l)
echo "$n pairs  ->  $BASE/static/review/"
echo
echo "Remove it when the decision is made -- staging is behind basic auth, but this is still served:"
echo "    $0 --rm"
