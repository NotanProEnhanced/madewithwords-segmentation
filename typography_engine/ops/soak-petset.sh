#!/bin/bash
# Soak the pet engine on staging: every photo in the pet set through every setting the site
# offers, plus a burst of concurrent requests, with every failure, slow render and threshold
# breach listed at the end.
#
# WHY
#   The gate (render-petset.sh + pet-gate.sh) renders each photo once, at the site's default
#   settings, and compares builds. A customer does not stop at the defaults: they drag the
#   size slider, switch the backdrop, pick a square or landscape print, and the loupe and the
#   backdrop toggle arrive while the preview is still rendering. This runs that matrix so a
#   setting that breaks, times out or crosses a claim threshold is found here, not by a buyer.
#
# USE
#   ./soak-petset.sh              full matrix: 3 sizes x 3 aspects per photo (~1 h for 9 photos)
#   QUICK=1 ./soak-petset.sh      defaults only plus the backdrop toggle and concurrency (~10 min)
#   ./soak-petset.sh 05 07        just those photos
#
#   Backdrop toggles reuse the cached render, so they cost nothing and test the cache path.
#   Results: /root/typortrait-testset/pets/soak/<image-tag>-<date>/ with one line per request
#   in requests.log and the summary in summary.txt.
set -uo pipefail
SET="${SET:-/root/typortrait-testset/pets}"
PORT="${PORT:-8078}"
BASE="http://127.0.0.1:$PORT"
CONTAINER="${CONTAINER:-typortrait-staging}"
BRAND="${BRAND:-pawsinwords}"
SLOW_S="${SLOW_S:-90}"            # a preview slower than this is flagged (browser limit is 120)
COLL_MAX="${COLL_MAX:-1.20}"
EXPOSED_MAX="${EXPOSED_MAX:-3.0}"

[ -d "$SET/src" ] || { echo "no sources at $SET/src"; exit 1; }
[ -s "$SET/words.txt" ] || { echo "no default words at $SET/words.txt"; exit 1; }
DEF_WORDS="$(tr -d '\r\n' < "$SET/words.txt")"
IMAGE="$(docker inspect --format '{{.Config.Image}}' "$CONTAINER" 2>/dev/null || echo unknown)"
TAG="${IMAGE##*:}"
OUT="$SET/soak/$TAG-$(date -u +%Y%m%d-%H%M)"
mkdir -p "$OUT"
LOG="$OUT/requests.log"; : > "$LOG"

if [ -n "${QUICK:-}" ]; then
    SIZES=(0.30); ASPECTS=(0.8)
else
    SIZES=(0.30 0.42 0.56); ASPECTS=(0.8 1.0 1.25)
fi
GROUNDS=(dark mid)

_up() { curl -sf --max-time 5 -o /dev/null "$BASE/static/index.html" 2>/dev/null; }
_up || { echo "nothing answering on $BASE"; exit 1; }

# One request. Prints a log line: status, seconds, size, and the engine's report card.
_render() {   # $1 file  $2 words  $3 size  $4 ground  $5 aspect  $6 label
    local f="$1" words="$2" size="$3" ground="$4" aspect="$5" label="$6"
    local since t0 dt resp prev log claim like fp ex co lk status
    since=$(date -u +%Y-%m-%dT%H:%M:%S); t0=$(date +%s.%N)
    resp=$(curl -s --max-time 600 -X POST "$BASE/render" \
        -F "image=@$f" -F "words=$words" -F "pet=1" -F "pet_type=$size" -F "ground=$ground" \
        -F "png_width=1400" -F "aspect=$aspect" -F "remove_bg=true" -F "uppercase=true" \
        -F "brand=$BRAND" -F "ref=$BRAND" -F "biometric_consent=on" 2>&1)
    dt=$(printf '%.1f' "$(echo "$(date +%s.%N) - $t0" | bc)")
    prev=$(printf '%s' "$resp" | sed -n 's/.*"preview"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
    if [ -z "$prev" ]; then
        err=$(printf '%s' "$resp" | sed -n 's/.*"error"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
        status="FAIL ${err:-no preview}"
        printf '%-46s %-6s %6ss\n' "$label" "$status" "$dt" | tee -a "$LOG"
        return
    fi
    log=$(docker logs --since "$since" "$CONTAINER" 2>&1)
    claim=$(printf '%s\n' "$log" | grep 'CLAIM METRICS' | tail -1)
    like=$(printf '%s\n' "$log" | grep 'type-only likeness score' | tail -1)
    fp=$(printf '%s' "$claim" | sed -n 's/.*footprint covers \([0-9.]*\)%.*/\1/p')
    ex=$(printf '%s' "$claim" | sed -n 's/.*exposed space ([^)]*) \([0-9.]*\)%.*/\1/p')
    co=$(printf '%s' "$claim" | sed -n 's/.*BODIES overlapping \([0-9.]*\)%.*/\1/p')
    lk=$(printf '%s' "$like"  | sed -n 's/.*: \([0-9.]*\)$/\1/p')
    status="ok"
    [ -z "$claim" ] && status="ok(cached)"   # the backdrop toggle and repeats reuse the render
    printf '%-46s %-10s %6ss  footprint=%s exposed=%s collisions=%s likeness=%s\n' \
        "$label" "$status" "$dt" "${fp:--}" "${ex:--}" "${co:--}" "${lk:--}" | tee -a "$LOG"
}

want=("$@")
echo "soak of $IMAGE -> $OUT"; echo
for f in "$SET"/src/*.jpg "$SET"/src/*.jpeg "$SET"/src/*.png; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"; n="${b%%-*}"; stem="${b%.*}"
    if [ ${#want[@]} -gt 0 ]; then
        hit=0; for w in "${want[@]}"; do [ "$w" = "$n" ] && hit=1; done
        [ "$hit" = "1" ] || continue
    fi
    words="$DEF_WORDS"
    [ -s "$SET/src/$stem.words" ] && words="$(tr -d '\r\n' < "$SET/src/$stem.words")"
    for size in "${SIZES[@]}"; do
        for aspect in "${ASPECTS[@]}"; do
            for ground in "${GROUNDS[@]}"; do
                _render "$f" "$words" "$size" "$ground" "$aspect" "$stem size=$size aspect=$aspect $ground"
            done
        done
    done
done

# Concurrency: the loupe and a backdrop toggle arrive while the preview is still rendering.
# Two different photos at once, then the same photo twice at once (the second must wait for
# the first's render, not start its own).
echo; echo "--- concurrent requests ---"
first=$(ls "$SET"/src/*.png "$SET"/src/*.jpg 2>/dev/null | head -1)
second=$(ls "$SET"/src/*.png "$SET"/src/*.jpg 2>/dev/null | sed -n 2p)
if [ -n "$first" ] && [ -n "$second" ]; then
    _render "$first"  "$DEF_WORDS SOAK" 0.30 dark 0.8 "concurrent A $(basename "$first")" &
    _render "$second" "$DEF_WORDS SOAK" 0.30 dark 0.8 "concurrent B $(basename "$second")" &
    wait
    _render "$first" "$DEF_WORDS SOAK2" 0.30 dark 0.8 "same-photo 1 $(basename "$first")" &
    sleep 2
    _render "$first" "$DEF_WORDS SOAK2" 0.30 mid  0.8 "same-photo 2 $(basename "$first") (backdrop)" &
    wait
fi

# Summary: failures, slow renders, threshold breaches.
{
    echo "soak of $IMAGE, $(wc -l < "$LOG") requests"
    echo "failures:";        grep -c ' FAIL' "$LOG" | sed 's/^/  /'; grep ' FAIL' "$LOG" | sed 's/^/  /'
    echo "slower than ${SLOW_S}s:"
    awk -v s="$SLOW_S" '{ for (i=1;i<=NF;i++) if ($i ~ /s$/ && $i+0 > s) { print "  " $0; break } }' "$LOG"
    echo "collisions over ${COLL_MAX}%:"
    grep -o '^.*collisions=[0-9.]*' "$LOG" | awk -v m="$COLL_MAX" -F'collisions=' '$2+0 > m { print "  " $0 }'
    echo "exposed over ${EXPOSED_MAX}%:"
    grep -o '^.*exposed=[0-9.]*' "$LOG" | awk -v m="$EXPOSED_MAX" -F'exposed=' '$2+0 > m { print "  " $0 }'
    echo "render time, fresh renders only (s): min / median / max"
    grep ' ok ' "$LOG" | awk '{ for (i=1;i<=NF;i++) if ($i ~ /s$/) { sub(/s$/,"",$i); print $i+0; break } }' \
        | sort -n | awk '{ a[NR]=$1 } END { if (NR) printf "  %s / %s / %s\n", a[1], a[int((NR+1)/2)], a[NR] }'
} | tee "$OUT/summary.txt"
echo; echo "log: $LOG"
