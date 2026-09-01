#!/bin/bash
# Compare two test-set runs image by image and say exactly which ones moved.
#
# WHY
#   The engine is deterministic, so "did this change affect anything I did not intend?" has
#   an exact answer: same bytes or different bytes. Reading that off `md5sum` by hand across
#   ten files invites the obvious mistake -- checking the image you were thinking about and
#   assuming the rest.
#
#   Most engine changes are supposed to touch a SUBSET. A per-face fix should move the
#   couple and nothing else. This prints which files moved so that claim is checked rather
#   than asserted.
#
# USE
#   ./compare-testset.sh 0f9d72d b8cf474        two commits
#   ./compare-testset.sh 0f9d72d               against the newest run
#   ./compare-testset.sh                       the two newest runs
#
#   Names are directory names under out/, so the parameter-suffixed ones work too:
#   ./compare-testset.sh b8cf474 db5a99b-png3600-rw2600  -- though comparing runs of
#   different SHAPE is meaningless and every file will differ.
set -uo pipefail

SET="${SET:-/root/typortrait-testset}"
OUTS="$SET/out"

[ -d "$OUTS" ] || { echo "no runs at $OUTS"; exit 1; }

_resolve() {   # accept a bare commit, a full directory name, or a path
    local n="$1"
    [ -d "$n" ] && { echo "$n"; return; }
    [ -d "$OUTS/$n" ] && { echo "$OUTS/$n"; return; }
    local hit
    hit=$(ls -1d "$OUTS"/"$n"*/ 2>/dev/null | head -1)
    [ -n "$hit" ] && { echo "${hit%/}"; return; }
    echo ""
}

if [ $# -ge 2 ]; then
    A=$(_resolve "$1"); B=$(_resolve "$2")
elif [ $# -eq 1 ]; then
    A=$(_resolve "$1"); B=$(ls -1dt "$OUTS"/*/ 2>/dev/null | head -1); B="${B%/}"
else
    A=$(ls -1dt "$OUTS"/*/ 2>/dev/null | sed -n 2p); A="${A%/}"
    B=$(ls -1dt "$OUTS"/*/ 2>/dev/null | head -1); B="${B%/}"
fi

[ -n "$A" ] && [ -d "$A" ] || { echo "cannot find first run"; exit 1; }
[ -n "$B" ] && [ -d "$B" ] || { echo "cannot find second run"; exit 1; }
[ "$A" != "$B" ] || { echo "both names resolve to $A"; exit 1; }

echo "A  $(basename "$A")"
echo "B  $(basename "$B")"
echo

same=0; diff=0; only=0
# Union of both sides: a file that exists in only one run is not "unchanged", and silently
# skipping it would hide a render that started failing.
for n in $( (ls -1 "$A"; ls -1 "$B") 2>/dev/null | sort -u); do
    fa="$A/$n"; fb="$B/$n"
    if [ ! -f "$fa" ]; then printf '  %-24s only in B\n' "$n"; only=$((only + 1)); continue; fi
    if [ ! -f "$fb" ]; then printf '  %-24s only in A\n' "$n"; only=$((only + 1)); continue; fi
    ha=$(md5sum "$fa" | cut -d' ' -f1)
    hb=$(md5sum "$fb" | cut -d' ' -f1)
    if [ "$ha" = "$hb" ]; then
        printf '  %-24s identical\n' "$n"; same=$((same + 1))
    else
        # Size alone does not prove how much changed, but a large swing is worth seeing
        # before opening the files.
        sa=$(stat -c%s "$fa"); sb=$(stat -c%s "$fb")
        printf '  %-24s DIFFERS   %s -> %s\n' "$n" \
               "$(numfmt --to=iec "$sa" 2>/dev/null || echo "$sa")" \
               "$(numfmt --to=iec "$sb" 2>/dev/null || echo "$sb")"
        diff=$((diff + 1))
    fi
done

echo
echo "$same identical, $diff differ, $only present in only one run"
[ "$diff" = "0" ] && [ "$only" = "0" ] && echo "The runs are the same render." || \
    echo "Open the ones marked DIFFERS and confirm each was meant to move."
