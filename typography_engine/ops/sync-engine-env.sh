#!/bin/bash
# Write ops/engine.env, the one engine settings file, into each tree's .env.
#
#   ./ops/sync-engine-env.sh typortrait-prod typortrait-lovedinwords ...    # named trees
#   ./ops/sync-engine-env.sh --all                                           # every /root/*/typography_engine/.env
#   DRY=1 ./ops/sync-engine-env.sh --all                                     # show what would change, touch nothing
#
# For every key engine.env owns (with or without a value), the tree's own line is removed;
# keys with a value are then appended as one block. Keys engine.env does not mention are
# left exactly as they were: identity, keys, ports, prices, a tree's feature switches.
# Each .env is backed up first (.env.bak-<stamp>), the same convention as promote.sh.
# The container is NOT recreated here: run `docker compose up -d` in the tree, or
# promote.sh, and then the parity render is the proof that the sites agree.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CANON="${CANON:-$HERE/engine.env}"
[ -f "$CANON" ] || { echo "no canonical file at $CANON"; exit 1; }

if [ "${1:-}" = "--all" ]; then
    TREES=()
    for e in /root/*/typography_engine/.env; do
        TREES+=("$(basename "$(dirname "$(dirname "$e")")")")
    done
else
    TREES=("$@")
fi
[ ${#TREES[@]} -gt 0 ] || { echo "usage: $0 <tree>... | --all"; exit 1; }

# The keys the canonical file owns, and the lines it sets.
OWNED=$(grep -E '^[A-Z0-9_]+=' "$CANON" | cut -d= -f1 | sort -u)
SET_LINES=$(grep -E '^[A-Z0-9_]+=.+' "$CANON")
STAMP=$(date +%Y%m%d-%H%M%S)
BEGIN="# ---- engine settings: written by ops/sync-engine-env.sh from ops/engine.env; edit THAT file, then sync ----"
END="# ---- end of engine settings ----"

for t in "${TREES[@]}"; do
    ENVF="/root/$t/typography_engine/.env"
    [ -f "$ENVF" ] || { echo "== $t: no .env at $ENVF, skipped"; continue; }
    # Drop the previous synced block (if any) and every line for an owned key.
    NEW=$(awk -v begin="$BEGIN" -v end="$END" '
        $0 == begin { skip = 1; next }
        $0 == end   { skip = 0; next }
        !skip { print }' "$ENVF")
    for k in $OWNED; do
        NEW=$(printf '%s\n' "$NEW" | grep -v -E "^${k}=" || true)
    done
    NEW=$(printf '%s\n' "$NEW" | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}')   # trim trailing blank lines
    OUT=$(printf '%s\n\n%s\n%s\n%s\n' "$NEW" "$BEGIN" "$SET_LINES" "$END")
    if [ -n "${DRY:-}" ]; then
        echo "== $t (dry run): lines that would change"
        diff <(sort "$ENVF") <(printf '%s\n' "$OUT" | sort) | grep -E '^[<>] [A-Z]' || echo "   (none)"
        continue
    fi
    cp -a "$ENVF" "$ENVF.bak-$STAMP"
    printf '%s\n' "$OUT" > "$ENVF"
    n_removed=$(diff <(sort "$ENVF.bak-$STAMP") <(sort "$ENVF") | grep -c '^< [A-Z]' || true)
    n_added=$(diff <(sort "$ENVF.bak-$STAMP") <(sort "$ENVF") | grep -c '^> [A-Z]' || true)
    echo "== $t: synced ($n_removed lines removed, $n_added added; backup $ENVF.bak-$STAMP)"
done
