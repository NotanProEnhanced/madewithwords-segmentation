#!/bin/bash
# The pet regression gate: compare two render-petset.sh runs and say exactly what moved.
#
# WHY
#   The pet engine is deterministic, so "did this change touch anything I did not intend?"
#   has an exact answer per image: same bytes or not. And each render prints its own report
#   card (likeness, collisions, exposed space, footprint), which render-petset.sh stores
#   beside the image. This reads both and fails the gate when a number moves the wrong way:
#
#     likeness    down by more than 0.010 on any image        -> FAIL
#     collisions  above 1.20% on any image, or up by > 0.25   -> FAIL
#     exposed     above 3.0%  on any image, or up by > 0.50   -> FAIL
#
#   Bytes moving is not a failure by itself -- most engine changes are meant to move every
#   image -- but the list says which ones did, so "this only affects the eyes" is checked.
#
# USE
#   ./pet-gate.sh 12585cc c835642        two runs by commit (directory names under pets/out)
#   ./pet-gate.sh 12585cc                against the newest run
#   ./pet-gate.sh                        the two newest runs
#   Thresholds: LIKENESS_DROP=0.010 COLL_MAX=1.20 COLL_RISE=0.25 EXPOSED_MAX=3.0 EXPOSED_RISE=0.50
set -uo pipefail
SET="${SET:-/root/typortrait-testset/pets}"
OUTS="$SET/out"
LIKENESS_DROP="${LIKENESS_DROP:-0.010}"
COLL_MAX="${COLL_MAX:-1.20}"; COLL_RISE="${COLL_RISE:-0.25}"
EXPOSED_MAX="${EXPOSED_MAX:-3.0}"; EXPOSED_RISE="${EXPOSED_RISE:-0.50}"
[ -d "$OUTS" ] || { echo "no runs at $OUTS"; exit 1; }

_resolve() {
    local n="$1"
    [ -d "$n" ] && { echo "$n"; return; }
    [ -d "$OUTS/$n" ] && { echo "$OUTS/$n"; return; }
    local hit; hit=$(ls -1d "$OUTS"/"$n"*/ 2>/dev/null | head -1)
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
_missing() {
    echo "no run named '$1' under $OUTS"; echo "runs available:"
    ls -1dt "$OUTS"/*/ 2>/dev/null | sed 's|.*/out/||; s|/$||; s|^|  |' || echo "  (none)"
    exit 1
}
[ -n "$A" ] && [ -d "$A" ] || _missing "${1:-}"
[ -n "$B" ] && [ -d "$B" ] || _missing "${2:-}"
[ "$A" != "$B" ] || { echo "both names resolve to $A"; exit 1; }
echo "A  $(basename "$A")"
echo "B  $(basename "$B")"
echo

_get() { sed -n "s/^$2=//p" "$1" 2>/dev/null | head -1; }
_cmp() { awk -v a="$1" -v op="$2" -v b="$3" 'BEGIN { if (op == ">") exit !(a+0 > b+0); else exit !(a+0 < b+0) }'; }

moved=0; same=0; fails=0; missing=0
printf '  %-22s %-9s %9s %9s %9s %9s\n' "image" "bytes" "likeness" "coll%" "exposed%" "verdict"
for fb in "$B"/*.png; do
    [ -e "$fb" ] || continue
    name="$(basename "$fb" .png)"; fa="$A/$name.png"
    if [ ! -e "$fa" ]; then printf '  %-22s %-9s\n' "$name" "only in B"; missing=$((missing+1)); continue; fi
    if cmp -s "$fa" "$fb"; then bytes="same"; same=$((same+1)); else bytes="MOVED"; moved=$((moved+1)); fi
    ma="$A/$name.metrics"; mb="$B/$name.metrics"
    la=$(_get "$ma" likeness); lb=$(_get "$mb" likeness)
    ca=$(_get "$ma" collisions); cb=$(_get "$mb" collisions)
    ea=$(_get "$ma" exposed); eb=$(_get "$mb" exposed)
    verdict="ok"; why=""
    if [ -z "$lb" ] || [ -z "$cb" ] || [ -z "$eb" ]; then
        verdict="no metrics"; missing=$((missing+1))
    else
        if [ -n "$la" ] && _cmp "$(awk -v a="$la" -v b="$lb" 'BEGIN{print a-b}')" ">" "$LIKENESS_DROP"; then verdict="FAIL"; why="$why likeness-$(awk -v a="$la" -v b="$lb" 'BEGIN{printf "%.3f", a-b}')"; fi
        if _cmp "$cb" ">" "$COLL_MAX"; then verdict="FAIL"; why="$why collisions>$COLL_MAX"; fi
        if [ -n "$ca" ] && _cmp "$(awk -v a="$ca" -v b="$cb" 'BEGIN{print b-a}')" ">" "$COLL_RISE"; then verdict="FAIL"; why="$why collisions+$(awk -v a="$ca" -v b="$cb" 'BEGIN{printf "%.2f", b-a}')"; fi
        if _cmp "$eb" ">" "$EXPOSED_MAX"; then verdict="FAIL"; why="$why exposed>$EXPOSED_MAX"; fi
        if [ -n "$ea" ] && _cmp "$(awk -v a="$ea" -v b="$eb" 'BEGIN{print b-a}')" ">" "$EXPOSED_RISE"; then verdict="FAIL"; why="$why exposed+$(awk -v a="$ea" -v b="$eb" 'BEGIN{printf "%.2f", b-a}')"; fi
        [ "$verdict" = "FAIL" ] && fails=$((fails+1))
    fi
    printf '  %-22s %-9s %4s>%-4s %4s>%-4s %4s>%-4s %s%s\n' "$name" "$bytes" "${la:--}" "${lb:--}" "${ca:--}" "${cb:--}" "${ea:--}" "${eb:--}" "$verdict" "$why"
done
for fa in "$A"/*.png; do
    [ -e "$fa" ] || continue
    name="$(basename "$fa" .png)"
    [ -e "$B/$name.png" ] || { printf '  %-22s %-9s\n' "$name" "only in A"; missing=$((missing+1)); }
done
echo
echo "$moved moved, $same identical, $fails failed the thresholds, $missing missing"
[ "$fails" = "0" ] && [ "$missing" = "0" ]
