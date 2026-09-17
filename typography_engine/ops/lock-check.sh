#!/bin/bash
# Prove the box-wide render lock: fire a heavy render at TWO containers at the same
# moment, watch the box's memory while they run, and read each container's /health
# `heavy_lock` afterwards. With the lock on, one of the two reports a wait of roughly a
# whole render (20-40 s) and the box's peak is one render plus the idle baselines. With
# the lock off (RENDER_LOCK_DIR empty), both run at once and the peak is two renders.
#
#   ./ops/lock-check.sh                       # pawsinwords (8081) and staging (8078)
#   A=8081 B=8078 PHOTO=/root/typortrait-testset/pets/src/01-tabby.png ./ops/lock-check.sh
#
# A and B are host ports of two containers; PHOTO is a pet photo both are sent. Staging
# must have PET_ENGINE=v2 (it does) for its render to be heavy.
set -uo pipefail
A="${A:-8081}"
B="${B:-8078}"
PHOTO="${PHOTO:-/root/typortrait-testset/pets/src/01-tabby.png}"
WORDS="${WORDS:-MILO, LOYAL, GENTLE, GOOFY, SOUL, KIND}"
[ -f "$PHOTO" ] || { echo "no photo at $PHOTO"; exit 1; }

_health() { curl -s --max-time 5 "http://127.0.0.1:$1/health" | python3 -c 'import sys,json; d=json.load(sys.stdin); h=d.get("heavy_lock",{}); print("port", sys.argv[1], "lock", "on" if h.get("enabled") else "OFF", "waits", h.get("waits"), "wait_max_s", h.get("wait_max_s"), "holding", h.get("holding"))' "$1" 2>/dev/null || echo "port $1: no health"; }
_render() {   # port -> seconds taken, one line
    local t0=$(date +%s.%N)
    local r=$(curl -s --max-time 600 -X POST "http://127.0.0.1:$1/render" -F "image=@$PHOTO" -F "words=$WORDS" \
        -F "pet=1" -F "pet_type=0.30" -F "ground=dark" -F "png_width=1400" -F "aspect=0.8" \
        -F "remove_bg=true" -F "uppercase=true" -F "brand=pawsinwords" -F "ref=pawsinwords" -F "biometric_consent=on" 2>&1)
    local ok=$(printf '%s' "$r" | grep -c '"preview"')
    printf 'port %s: %s in %.0fs\n' "$1" "$([ "$ok" = 1 ] && echo rendered || echo FAILED)" "$(echo "$(date +%s.%N) - $t0" | bc)"
}

echo "before:"; _health "$A"; _health "$B"
echo "firing both at $(date +%H:%M:%S) ..."
# Memory sampler: total used on the box every 2 s while the renders run.
( while true; do free -m | awk '/^Mem:/{print $3}'; sleep 2; done ) > /tmp/lock-check.mem &
SAMPLER=$!
_render "$A" & PA=$!
_render "$B" & PB=$!
wait $PA; wait $PB
kill $SAMPLER 2>/dev/null
PEAK=$(sort -n /tmp/lock-check.mem | tail -1); BASE=$(head -1 /tmp/lock-check.mem)
echo "box memory: ${BASE} MB before, peak ${PEAK} MB during (of $(free -m | awk '/^Mem:/{print $2}'))"
echo "after:"; _health "$A"; _health "$B"
echo
echo "Read: with the lock on, one port shows wait_max_s near a whole render and the peak is about one render above the baseline."
