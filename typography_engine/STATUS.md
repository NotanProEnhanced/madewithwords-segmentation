# Typortrait — STATUS / handoff (read me first)

Quick orientation for a new session. The app lives in `typography_engine/`.

## What's built (all merged to `main`)
- Render engine (`app/pipeline/tonal.py`): word-mosaic + **story calligram**; inks
  navy(default), photo, sepia, burgundy, forest, gold_noir, **spectrum**, **aurora**,
  mono; eyes (catchlight), skin gradation, multi-face, input-quality gate.
- Freemium + **Stripe**: free **watermarked** preview (Typortrait.com + QR); pay to
  download clean file. Clean art in PRIVATE dir; `/checkout` + `/download` (verifies
  payment). Price up front via `/pricing`, default **$14.99** (`TYPO_PRICE_CENTS`).
- Front-end (`static/index.html`): photo->words/message->reveal->**iOS-style live-thumbnail
  swatch carousel** (crossfade)->**unified product picker** (framed / canvas / poster /
  t-shirt / digital)->checkout. Mosaic/Story toggle, keepsake poster.
- Deploy: `Dockerfile`, `docker-compose.yml` (Stripe + Printful `.env` + ./data volumes),
  nginx, **GO-LIVE.md** (beginner, source of truth) + **DEPLOY.md**.
- Marketing static site: `typography_engine/marketing/` (index.html, og.png, robots,
  sitemap, favicons) — SEO/AEO/schema. Replace placeholder testimonials before launch.
- Tests: **33** (pytest) passing.

## Phase E (Printful POD) — branch `claude/printful-pod`
Physical print fulfillment for posters, framed prints, canvas, and t-shirts via
Printful. Customer pays you via Stripe; your server submits the order to Printful
on `checkout.session.completed`; Printful charges your card for wholesale + ships.

Files: `app/printful.py` (client), `app/orders.py` (SQLite at `data/orders.db`),
`app/products.py` (catalog). Endpoints added: `GET /products`, `POST /webhook/stripe`,
`GET /order/{id}`, `GET /printful-fetch/{job}` (signed URL Printful fetches the
clean PNG from), `GET /admin/orders` (HTTP basic auth, password in `ADMIN_PASSWORD`).

### Env vars to add to `.env` before deploy
```
STRIPE_WEBHOOK_SECRET=whsec_...        # Stripe dashboard → Webhooks → endpoint signing secret
PRINTFUL_API_TOKEN=...                 # https://developers.printful.com/ → Tokens
PRINTFUL_STORE_ID=...                  # numeric, from Printful dashboard URL
PRINT_URL_SECRET=<random 32+ chars>    # used to HMAC-sign the print-fetch URLs
ADMIN_PASSWORD=<your password>         # protects /admin/orders
```

### Stripe webhook setup
After deploy, in Stripe dashboard → Developers → Webhooks → Add endpoint:
- URL: `https://app.typortrait.com/webhook/stripe`
- Events: `checkout.session.completed`
- Copy the **signing secret** into `STRIPE_WEBHOOK_SECRET`, then restart the container.

### Printful product variant IDs
`products.py` references hard-coded variant IDs from Printful's catalog. **Verify
these IDs against `GET /products/<product_id>` on the live Printful API before
launch** — Printful occasionally renumbers SKUs. Current IDs target:
- `1320` (enhanced matte framed 16×20 black)
- `3618` (canvas 16×20, 1.25" thick)
- `4` (enhanced matte poster 18×24)
- `4012–4016` (Bella+Canvas 3001 unisex tee, black, S–2XL)

## Decisions
- Default ink **navy**; price **$14.99** digital; framed prints "coming soon".
- Architecture: **marketing on `typortrait.com`**, **app on `app.typortrait.com`**.
- Per-glyph assembly animation intentionally OFF in the live flow (would leak clean art).
- Contour/orientation styling and aggressive variable-size were tried and REVERTED
  (hurt likeness). Don't reintroduce as default.

## The VPS (74.208.113.203, Ubuntu 24.04) — current reality
- `typortrait.com` is ALREADY served by THIS VPS via nginx+TLS, docroot
  `/var/www/typortrait.com` (index.html). It 404s because **no index.html at that
  top level** -> fix = put marketing files there.
- It's a **busy, 1.8 GB box** with many other live sites/services (lineforge,
  portraitsinblack, color API, charcoal, an existing api.typortrait.com renderer,
  several uvicorn ports). Port **8077 free**, `app.typortrait.com` unused.
- Memory is too low to build/run our renderer as-is.

## NEXT STEPS (in progress)
1. **Marketing (do now):** from the PC folder with the 7 files,
   `scp index.html og.png robots.txt sitemap.xml favicon.ico favicon-32.png apple-touch-icon.png root@74.208.113.203:/var/www/typortrait.com/`
   then load https://typortrait.com. (Only adds files; doesn't touch existing subfolders/routes.)
2. **App memory — chosen path C:** free RAM by retiring DEAD old services, then add swap,
   then deploy. Identify dead ones by: a uvicorn/python whose port no ENABLED nginx site
   references is safe to stop (confirm with owner first). Do NOT stop anything serving
   the live sites above. Then increase swap to ~4 GB, then deploy app on 8077 +
   `app.typortrait.com` nginx + certbot (see GO-LIVE.md).

## Phase 2 (parked)
Daily FAL.ai source image -> render -> draft category landing page -> morning
Approve/Publish/Reject email. Human-approved, one quality page/day.
