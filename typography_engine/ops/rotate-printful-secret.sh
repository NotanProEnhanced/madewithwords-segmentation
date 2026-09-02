#!/bin/bash
# Rotate PRINTFUL_WEBHOOK_SECRET -- the shared secret guarding /webhook/printful.
#
# WHY THIS EXISTS
#   Printful's classic webhooks carry no signature, so main.py guards the endpoint with a
#   secret in the query string: the registered URL is .../webhook/printful?k=<secret>. That
#   means the secret lives in TWO places -- this tree's .env and the URL registered with
#   Printful -- and rotating it means changing both without leaving a gap.
#
#   There is no page in the Printful dashboard for this. Classic webhooks are registered
#   only through the API (POST /webhooks), and Printful keeps ONE webhook URL per store.
#   Nobody wrote that down, so the first rotation took an hour of hunting. Hence this.
#
# WHAT THE SECRET PROTECTS
#   Someone holding it can forge a `package_shipped` for an order whose id they already
#   know: that marks the order shipped and EMAILS THAT CUSTOMER a "your order has shipped"
#   message containing a tracking link of their choosing, from your domain. Order ids are
#   uuid4().hex[:12], so they cannot be guessed -- narrow, but rotate if it ever leaks.
#
# USE
#   ./rotate-printful-secret.sh              show what is registered now, change nothing
#   ./rotate-printful-secret.sh --rotate     generate a new secret, update .env, re-register
#
#   TREE=/root/typortrait-prod   the tree whose .env holds the secret (default)
#   CONTAINER=typortrait         the running container (prod's is NOT typortrait-prod)
#   PUBLIC_URL=https://app.typortrait.com
#
# SAFETY
#   POST /webhooks REPLACES the whole configuration -- it does not add to it. So the event
#   types are read from what is registered now and sent back unchanged; if none is
#   registered this refuses to invent them. The secret is never echoed, and the API token
#   is only ever read inside the container, so neither reaches your shell history.
set -uo pipefail

TREE="${TREE:-/root/typortrait-prod}"
CONTAINER="${CONTAINER:-typortrait}"
PUBLIC_URL="${PUBLIC_URL:-https://app.typortrait.com}"
ENVF="$TREE/typography_engine/.env"

[ -f "$ENVF" ] || { echo "no .env at $ENVF -- set TREE=/root/typortrait-<name>"; exit 1; }
docker inspect "$CONTAINER" >/dev/null 2>&1 || { echo "no container '$CONTAINER' -- set CONTAINER="; exit 1; }

_api() {   # run curl INSIDE the container so the token stays in its environment
    docker exec "$CONTAINER" sh -lc "curl -s $* -H \"Authorization: Bearer \$PRINTFUL_API_TOKEN\" https://api.printful.com/webhooks"
}

echo "tree      $TREE"
echo "container $CONTAINER"
echo
echo "== registered with Printful now"
CUR="$(_api)"
[ -n "$CUR" ] || { echo "no response from Printful -- is PRINTFUL_API_TOKEN set in this container?"; exit 1; }

# Show the shape without the secret: the URL is printed with everything after k= masked.
printf '%s\n' "$CUR" | sed 's/\(k=\)[^"&]*/\1<secret hidden>/g'
echo

TYPES="$(printf '%s' "$CUR" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print(""); raise SystemExit
r=(d or {}).get("result") or {}
t=r.get("types") or []
print(json.dumps(t) if t else "")
')"

if [ -z "$TYPES" ] || [ "$TYPES" = "[]" ]; then
    cat <<'EOF'
No webhook is registered for this store.

Nothing is calling /webhook/printful, so the secret guards a door nobody uses. Rather than
rotating it, remove it -- one less live-looking credential to leak:

    sed -i '/^PRINTFUL_WEBHOOK_SECRET=/d' <tree>/typography_engine/.env
    cd <tree>/typography_engine && docker compose up -d
EOF
    exit 0
fi
echo "event types currently registered: $TYPES"

if [ "${1:-}" != "--rotate" ]; then
    echo
    echo "Nothing changed. Re-run with --rotate to generate a new secret and re-register."
    exit 0
fi

command -v openssl >/dev/null || { echo "openssl not installed"; exit 1; }
K="$(openssl rand -hex 32)"

# .env first: the app must accept the new secret BEFORE Printful starts sending it. A
# forged-order window is worse than a few 403s, and Printful retries a non-200 anyway.
cp -a "$ENVF" "$ENVF.bak-$(date +%Y%m%d-%H%M%S)"
if grep -q '^PRINTFUL_WEBHOOK_SECRET=' "$ENVF"; then
    sed -i "s|^PRINTFUL_WEBHOOK_SECRET=.*|PRINTFUL_WEBHOOK_SECRET=$K|" "$ENVF"
else
    printf 'PRINTFUL_WEBHOOK_SECRET=%s\n' "$K" >> "$ENVF"
fi
echo "== .env updated (backup alongside it); restarting"
( cd "$TREE/typography_engine" && docker compose up -d ) || { echo "restart FAILED -- .env is changed, Printful is not"; exit 1; }

BODY="$(python3 -c '
import json,sys
print(json.dumps({"url": sys.argv[1] + "/webhook/printful?k=" + sys.argv[2],
                  "types": json.loads(sys.argv[3])}))' "$PUBLIC_URL" "$K" "$TYPES")"
printf '%s' "$BODY" | docker exec -i "$CONTAINER" sh -c 'cat > /tmp/pf.json'
echo "== registering the new URL with Printful"
_api "-X POST -H 'Content-Type: application/json' --data @/tmp/pf.json" | \
    sed 's/\(k=\)[^"&]*/\1<secret hidden>/g'
docker exec "$CONTAINER" rm -f /tmp/pf.json

echo
echo "== verifying"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$PUBLIC_URL/webhook/printful?k=deliberately-wrong")
echo "wrong secret -> HTTP $code   (want 403)"
code=$(docker exec "$CONTAINER" sh -lc "curl -s -o /dev/null -w '%{http_code}' -X POST 'http://127.0.0.1:8077/webhook/printful?k=$K'")
echo "new secret   -> HTTP $code   (want 400: accepted, then rejected the empty body)"
echo
echo "Done. The old secret is dead. Delete the .env backup once the next shipment arrives."
