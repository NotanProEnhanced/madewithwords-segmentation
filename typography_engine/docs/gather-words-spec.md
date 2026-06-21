# Gather — crowd‑sourced words for a portrait

> **Status:** built on branch `feat/gather-words`, **flag‑gated OFF**
> (`TYPO_GATHER_ENABLED`). Inert in staging/prod until deliberately enabled.

A shareable link where a whole circle adds a word/memory, and the portrait is
woven from **everyone's** words — turning a solo tool into a collective ritual
and a viral, gift‑worthy keepsake. One initiator → many contributors → many buyers.

## Why isolation can't damage staging/prod
1. **Separate branch.** Staging tracks `feat/displacement-style`, prod tracks
   `claude/printful-integration`; neither sees this branch until merged.
2. **Feature flag.** `GATHER_ENABLED` (env `TYPO_GATHER_ENABLED`, default **off**).
   Every gather route mounts only when on (`app/main.py`). Even if merged, prod is
   unaffected until the env var is set.
3. **Self‑contained module.** Own DB file (`data/gather/gather.db`), own routes,
   own pages. It never imports into or mutates the studio/checkout/orders code.

## Timing model — the heart of it
`seed (instant) → live gather (throttled) → reveal (final edition) → optional living`

- **Seed (0 s).** The initiator creates a gather from a photo + their own first
  words; a finished portrait exists immediately. No one ever waits on an empty page.
- **Gather (hours–days).** Contributors add words via the share link. Each word
  lands in the **list instantly** (cheap DB write); the **portrait re‑renders on a
  throttle** — at most every `GATHER_RENDER_INTERVAL` s (45) **or** once
  `GATHER_RENDER_EVERY` (4) new words arrive — and only **one** gather render runs
  at a time (non‑blocking; excess triggers drop and retry on the next poll). This
  protects the single render slot the studio shares.
- **Reveal.** The initiator taps *Finish & reveal* (or a close date passes): a
  final edition renders and contributors can view + order their own. This is the
  purchase trigger and the shareable moment.
- **Living (optional).** A gather can stay open; new words → new editions on
  anniversaries (future: edition history + reprint).

## Data model (`app/gather.py`, SQLite `data/gather/gather.db`)
- **gathers**: `id, share_token, admin_token, title, photo_path, style, ink,
  ground, min_font, status('open'|'revealed'), created_at, close_at,
  last_render_at, words_at_render, has_portrait`.
- **contributions**: `id, gather_id, text, name, created_at, hidden`.

Tokens: `share_token` (public, short) in `/g/{share}`; `admin_token` (secret,
long, == bearer auth) in `/a/{admin}`. No accounts.

## Word handling
- A contribution is a word or short phrase (≤ `GATHER_WORD_MAXLEN` 40).
- For the portrait, all visible contributions are tokenised, uppercased,
  **frequency‑ranked** (consensus rises — "Kind" said 12× sorts first), de‑duped,
  and capped at `GATHER_MAX_WORDS` (64) so it stays legible.
- *(Future: pass weights so most‑said words render LARGER — needs renderer
  support; today ranking only orders the list.)*

## Endpoints (all under the flag)
Pages: `GET /g/{share}` (contribute), `GET /a/{admin}` (dashboard).
API:
- `POST /api/gather/create` — multipart `image,title,words(seed),style,ink,ground,min_font,close_at` → `{share_url, admin_url, gather_id}`
- `GET  /api/gather/{share}/state` — public state (+ opportunistic throttled render)
- `POST /api/gather/{share}/word` — `text,name` (rate‑limited per IP, length‑capped, profanity‑blocked)
- `GET  /api/gather/{share}/portrait.png`
- `GET  /api/gather/admin/{admin}/state` — incl. hidden words + share_url
- `POST /api/gather/admin/{admin}/word/{wid}/hide` — moderate
- `POST /api/gather/admin/{admin}/render` — force a render
- `POST /api/gather/admin/{admin}/reveal` — close + final edition

Render reuses the studio pipeline (`analyze_image` + `render_displacement_portrait`
/ `render_layered_png`) at preview resolution (SS=1, out_width 760) — same code,
no fork.

## Pages
- **`static/gather.html`** (contributor, conversion‑critical): evolving portrait,
  word count, one input + optional name, guided prompts, "you're part of it"
  confirmation, soft "get your own print" capture; polls `/state` every 6 s.
- **`static/gather-admin.html`** (initiator): portrait, live counts, copyable
  share link, word list with hide/show moderation, *Refresh portrait*, *Finish &
  reveal*.

## How to enable (when ready — NOT now)
1. Merge `feat/gather-words` → the target branch.
2. Set `TYPO_GATHER_ENABLED=1` in that environment's `.env`, rebuild.
3. Studio wiring (next increment): a "Make it the family's" button on the result
   screen posting the current job to `/api/gather/create`, then showing the
   share/admin links.

## Not built yet (next increments)
- Studio "Make it the family's" entry point (button on the result card).
- Weighted glyph sizing (most‑said = bigger).
- Email/notify at reveal; contributor "notify me" capture → list.
- Edition history + anniversary reprint; "the words & who said them" certificate.
- Retention/cleanup parity with `RETENTION_DAYS` (purge old gathers + photos).
