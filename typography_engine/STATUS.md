# Typortrait — STATUS / handoff (read me first)

Quick orientation for a new session. The app lives in `typography_engine/`.

## What's built (all merged to `main`)
- Render engine (`app/pipeline/tonal.py`): word-mosaic + **story calligram**; inks
  navy(default), photo, sepia, burgundy, forest, gold_noir, **spectrum**, **aurora**,
  mono; eyes (catchlight), skin gradation, multi-face, input-quality gate.
- Freemium + **Stripe**: free **watermarked** preview (Typortrait.com + QR); pay to
  download clean file. Clean art in PRIVATE dir; `/checkout` + `/download` (verifies
  payment). Price up front via `/pricing`, default **$14.99** (`TYPO_PRICE_CENTS`).
- Front-end (`static/index.html`): photo→words/message→reveal→**iOS-style live-thumbnail
  swatch carousel** (crossfade)→download. Mosaic/Story toggle, keepsake poster.
- Deploy: `Dockerfile`, `docker-compose.yml` (Stripe `.env` + ./data volumes), nginx,
  **GO-LIVE.md** (beginner, source of truth) + **DEPLOY.md**.
- Marketing static site: `typography_engine/marketing/` (index.html, og.png, robots,
  sitemap, favicons) — SEO/AEO/schema. Replace placeholder testimonials before launch.
- Tests: 18 (pytest) passing.

## Decisions
- Default ink **navy**; price **$14.99** digital; framed prints "coming soon".
- Architecture: **marketing on `typortrait.com`**, **app on `app.typortrait.com`**.
- Per-glyph assembly animation intentionally OFF in the live flow (would leak clean art).
- Contour/orientation styling and aggressive variable-size were tried and REVERTED
  (hurt likeness). Don't reintroduce as default.

## The VPS (74.208.113.203, Ubuntu 24.04) — current reality
- `typortrait.com` is ALREADY served by THIS VPS via nginx+TLS, docroot
  `/var/www/typortrait.com` (index.html). It 404s because **no index.html at that
  top level** → fix = put marketing files there.
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
Daily FAL.ai source image → render → draft category landing page → morning
Approve/Publish/Reject email. Human-approved, one quality page/day.
