# Typortrait — Handoff

Pick‑up note for continuing this work from another session (cloud / phone / desktop).
The **code is the source of truth** — it's all on the branch below. This file is the map.

---

## Status

- **Branch:** `feat/displacement-style` (active dev branch). PROD runs `claude/printful-integration`,
  which on 2026-06-16 was fast-forwarded to `feat/displacement-style`'s tip `d543bfd` (clean FF —
  `feat/displacement-style` descends directly from prod's prior HEAD `18da690`, nothing dropped).
- **Latest commit:** `bb7459a` — *consent drop-zone lock UX* (atop the privacy-compliance work
  `640a9ec`; render-health, brand-preserving "make another" links, memorial gating, reels, purchase
  emails are all earlier on the same branch).
- **Deployed?** **PROD IS LIVE & COMPLIANCE-ENABLED** at `app.typortrait.com` on `bb7459a` (promoted
  2026-06-16 via `STAGING_BRANCH=feat/displacement-style ./promote.sh`). Verified on prod: canary 5/5,
  geo gate US-IL→blocked / US-CA→allowed, render completes. Two rollback tags exist:
  **`prod-pre-compliance-20260616`** (pre-compliance) and **`prod-pre-promote-20260616`** (= `18da690`,
  the original pre-everything state). Rollback = `git reset --hard <tag>` + `docker compose up -d --build`
  in `/root/typortrait/typography_engine`. Staging mirrors prod on port 8078.
- **App lives in:** `typography_engine/` (FastAPI app `app/main.py`, renderers in `app/pipeline/`, frontend `static/index.html`, reel builder `tools/reel_template.py`).

## The brand: "Loved in Words"

A memorial sub‑brand of Typortrait, same engine, different face. Activated by `?brand=lovedinwords`
(or the lovedinwords.com referrer). Driven by `BRAND_ID` in `static/index.html` (`BRANDS` registry).

**Critical rule learned the hard way:** the *brand experience* is gated on the **active brand**
(`BRAND_ID` → sent as the `brand` field → stored in the job recipe), **NOT** on `ref`. `ref` is the
referral/attribution tag and **persists in localStorage**, so it leaks across brands. Use `brand`
for any "is this the memorial experience?" decision; use `ref` only for sales attribution.

---

## What's done on this branch (newest → oldest, grouped)

**Eyes / teeth realism (renderers — `tonal.py` = Words/Message, `displacement.py` = Sculpt):**
- No typography on teeth or sclera; round pupils, limbal ring, deterministic white catchlights.
- Sclera/teeth take the photo's own pixels in Words; **Sculpt uses one neutral, dim tone for every
  ink** (the photo‑pixel version glowed on the solid ground). Photo‑ink eye/teeth are neutralised so
  they don't pick up the warm cast. Iris keeps the real eye color in Photo mode.

**Crop on upload (`f24ed27`, `25656f0`, `bd81171`):**
- Every upload opens a crop editor, auto‑framed on the detected face, locked to 4:5, draggable with
  mouse/touch/stylus. **Cropping happens server‑side** — the client records the crop rectangle
  (`state.crop`, fractions) and the server crops the source before render (client canvas export was
  unreliable). Before/after slider's "before" is the server‑cropped photo.

**Print sizing (`05c3409`, `9c16b13`):**
- Per‑product print aspect (face centered, ground‑padded, never cropped): 16×20 → 0.8, 18×24 → 0.75.
- 200–225 PPI on every size (`DOWNLOAD_PNG_WIDTH=3600`, env `TYPO_DOWNLOAD_PX`). Print file warmed
  in the background at fulfillment.

**Type scaling (`8acacfa`):** Sculpt type now scales gradually on tight crops (face‑relative feathering).

**Brand pass (`c9e5832`…`0bb4960`, plus fixes):**
- Loved in Words: opens on **Sculpt + Photo** (pinned), soft slate/ivory theme, **prints‑first**
  product order, **t‑shirt hidden**, landing page palette matched. Generic Typortrait unchanged.

**Reel / tribute video (`a44f943`…`bcd50c8`):**
- The personal reel plays the portrait inside a **real framed‑on‑a‑desk scene**.
  - Memorial → candle scene `static/everloved/scene-desk.jpg`, credit `lovedinwords.com`,
    success/order copy says **"Create a tribute video"**, and the **"feature on our socials" consent
    is removed**.
  - Generic → candle‑free scene `static/scene-desk-plain.jpg`, credit `Typortrait.com`, copy says
    **"Make a reel"** with the full consent.

**Purchase emails (`eff70bf`):** On a paid checkout the Stripe webhook (1) sets `receipt_email` on the
PaymentIntent so the customer always gets a Stripe receipt, and (2) emails `TYPO_ADMIN_EMAIL` a sale
alert. Deduped per session (`data/sale_notified`), off‑thread, best‑effort. Needs SMTP + webhook set.

**Render health — fail loud + canary (`1b48e34`, `10d1b42`):** Two guards against the worst failure
mode (silently shipping a degraded portrait of someone's loved one):
- **Fail-loud rasterizer.** `app/pipeline/raster.py` no longer silently falls back cairosvg→resvg
  (which drops tonal modulation). It logs loudly once, records the backend, and exposes
  `cairosvg_available()` (a real 1×1 trial render — catches "imports but native lib missing"). The
  Words/Message render path attaches a `degraded_rasterizer` warning; with **`TYPO_REQUIRE_CAIROSVG=1`**
  in `.env` it's a hard 422 so a degraded portrait can't reach a buyer. `/health` now reports
  `cairosvg_usable`.
- **Canary.** `tools/render_canary.py` runs a baked known-good portrait (`tools/assets/canary_portrait.png`)
  through BOTH real renderers and asserts: cairosvg full-fidelity, face locks via MediaPipe, output
  non-blank/correctly-sized. Exits non-zero on any failure.
- **Staging is wired:** `TYPO_REQUIRE_CAIROSVG=1` set, canary passes 5/5, and a daily cron runs it at
  07:00 UTC → `/var/log/typortrait-canary.log` (alerts on non-zero exit). Run by hand:
  `docker exec typortrait-staging python /app/tools/render_canary.py 2>/dev/null`.
- **Prod runs the SAME way as staging** (Docker — container `typortrait`, port 8077, compose
  `/root/typortrait/typography_engine/docker-compose.yml`). The pm2 `typortrait-render` is an unrelated
  legacy **node** service (`/var/www/typortrait-render/render-api.js`) — NOT the web app; leave it alone.
  Prod canary: `docker exec typortrait python /app/tools/render_canary.py 2>/dev/null`.
**Privacy compliance — GDPR / CCPA / BIPA (`640a9ec`, `bb7459a`) — LIVE on prod + staging:**
- **Biometric consent gate:** the studio grays out the photo drop zone until the `#bioConsent` box is
  ticked; `/measure` + `/render` enforce a `biometric_consent` field server-side (400 without it) and
  store a versioned consent record per job. No faceprint is ever stored (mesh is transient).
- **Illinois geo-block:** `/measure` + `/render` return 451 for blocked regions; studio disables upload
  up front via `GET /compliance/region`. Region from nginx GeoIP2 headers `X-Geo-Country`/`X-Geo-Region`
  (MaxMind GeoLite2-City at `/var/lib/GeoIP`, `geoip2.conf` + `proxy_set_header` in the server blocks).
  Default block `US-IL`; **fails OPEN** with no geo signal. Token format `COUNTRY-REGION` (avoids IL-state
  vs IL-country clash). Verified on prod: US-IL→blocked, US-CA→allowed.
- **Policies + DSAR:** expanded `/privacy`, new `/biometric-policy`, self-serve `/data-request`
  (job-id → immediate delete; all requests logged to `data/data_requests.log`). Debug face endpoints
  (`/debug/*`) now 404 unless `TYPO_ENABLE_DEBUG=1`. Full picture + lawyer punch list in `COMPLIANCE.md`.
- **Compose knobs ARE wired (`84a3cec`):** `TYPO_BLOCKED_REGIONS`, `TYPO_GEO_*_HEADER`,
  `TYPO_BIO_CONSENT_VERSION`, `TYPO_RETENTION_DAYS`, `TYPO_ENABLE_DEBUG`, `TYPO_REQUIRE_CAIROSVG` are now
  in both compose `environment:` blocks as `- VAR=${VAR:-default}`, so they're tunable from each env's
  `.env`. Defaults match the code (block `US-IL`, cairosvg warn, retention 30d). DSAR admin email +
  lawful-retention note added (`e86c106`).

**Upstream health — Stripe + Printful (`a35f993`, `8fb9755`) — LIVE on prod:**
- **`GET /health/upstream`** (separate from the fast local `/health`) pings Stripe (`/v1/account`) and
  Printful (`/orders`) with the configured keys — 8s timeout, 60s-cached. Returns `{ok, stripe, printful}`
  and **503** if any CONFIGURED upstream is unhealthy. A **403 is treated as healthy** (prod uses
  least-privilege keys that authenticate but can't read the probe endpoint); only 401 / 429 / 5xx /
  network fail. Exposes only coarse status (no keys/ids). Verified on prod: `ok:true`, Stripe live-key
  authenticates, Printful orders endpoint `200`. Wire into an uptime monitor or a `*/15` cron.

**`promote.sh` fix:** it used to prefer a **local** `feat/displacement-style` branch in
the prod tree, which `git fetch` does NOT update — so it once said "nothing to promote" and skipped the
rebuild while a new commit sat on origin. Now it always compares against **`origin/$STAGING_BRANCH`**.
Run it as `STAGING_BRANCH=feat/displacement-style ./promote.sh` from `/root/typortrait/typography_engine`.

---

## Next step

1. **Lawyer review of the compliance posture** (not legal sign-off yet) — punch list in `COMPLIANCE.md`
   (EU/UK Art. 27 rep, sub-processor DPAs, confirm the biometric stance, cookie audit). Also: whoever
   monitors `data/data_requests.log` for non-job DSARs.
2. **(Optional) Wire `/health/upstream` into monitoring** — uptime monitor or a `*/15` cron, so a
   revoked Stripe/Printful key or an outage pages you. Also schedule a **prod render-canary cron** if not
   already done (`docker exec typortrait …` → `typortrait-canary-prod.log`).
3. (Deferred) **Ever Loved partnership research** — a deep‑research run was started earlier; surface
   the findings + draft outreach when wanted. A trial = tracked referral link + code + one hero SKU,
   you as merchant of record (no deep integration); only being *visible on their site* needs their yes.

---

## How to deploy / test (run from the VPS — these are the human's commands)

```bash
cd ~/typortrait-staging/typography_engine && git pull --ff-only && ./staging.sh up -d --build
```

- **Static** (`static/*`, scene `.jpg`s) is **bind‑mounted** → live on `git pull`, no rebuild.
- **Python** (`app/`, `tools/`) is **baked into the image** → needs `--build` **and** a *fresh render*
  (renderer changes only show on a newly composed portrait, not a cached one).
- After deploy: **hard‑refresh** (Ctrl+Shift+R) the studio; static is browser‑cached.
- The brand experience is decided **at render time** by `brand`, so test with a **new** portrait:
  - Generic: `staging.typortrait.com/static/index.html` → reel + plain scene, "Make a reel".
  - Memorial: `…?brand=lovedinwords` → tribute video + candle scene.
- Verify the running image has new code: `docker exec typortrait-staging grep -c "<marker>" app/...`.

## Constraints (do not violate)

- **The human runs all VPS/SSH/deploy commands.** A cloud/phone session can edit + commit code but
  cannot deploy. Provide copy‑paste commands instead.
- **Never touch secrets.** Stripe keys, Printful token, webhook secret, admin/session/SMTP/FTP
  passwords are entered by the human; `.env` is git‑ignored. Verify secrets prefix‑only if ever needed.
- **Commit/push only when asked.** Don't merge to `main` or promote to prod without explicit go‑ahead.

## Useful pointers

- Brand registry + `BRAND_ID` resolution: `static/index.html` (search `BRANDS`, `BRAND_ID`).
- Reel builder + scenes: `tools/reel_template.py` (`build_reel`, the `scene` option).
- Memorial gates (use `brand`, not `ref`): `app/main.py` — `_reel_maker_block`, the `/reel` scene
  pick, the `/success` inline reel.
- Renderers: `app/pipeline/tonal.py` (Words/Message), `app/pipeline/displacement.py` (Sculpt).
