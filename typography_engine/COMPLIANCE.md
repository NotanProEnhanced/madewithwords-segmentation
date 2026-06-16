# Privacy compliance — GDPR / CCPA / BIPA

What the app now does, how to finish wiring it, and what still needs a lawyer.
**This is engineering documentation, not legal advice — have a privacy attorney
review before relying on it.** BIPA's private right of action ($1k–$5k per
violation) is the acute risk; everything here is aimed at lowering it.

---

## What's implemented (code)

**Strong starting posture (already true before this work):** facial geometry is
computed transiently at render and **never stored as a faceprint/template**;
only the source photo + render params persist, and the retention sweeper
auto-deletes them after `TYPO_RETENTION_DAYS` (default 30). We don't sell data.

**Added in this change:**
1. **Biometric notice + explicit consent before processing.** The studio shows a
   consent checkbox (`#bioConsent`) and will not call `/measure` (face detection)
   until it's ticked. `/measure` and `/render` both enforce it server-side
   (`biometric_consent` form field) and return **400** without it. A versioned
   consent record is stored per job (`<job>.biometric_consent.json`,
   `BIOMETRIC_CONSENT_VERSION`). Satisfies BIPA written-release + GDPR explicit
   consent for special-category data.
2. **Illinois geo-block.** `/measure` and `/render` return **451** for visitors
   in a blocked region; the studio also disables upload up front via
   `GET /compliance/region`. Region comes from proxy headers (see below).
   Configured by `TYPO_BLOCKED_REGIONS` (default `US-IL`). **Fails OPEN** when no
   geo signal is present — so it does nothing until you wire a geo source.
3. **Published policies.** `/privacy` expanded (legal bases, biometric
   disclosure, international transfers, GDPR + California rights, sub-processors,
   third-party-image takedown). New `/biometric-policy` is the BIPA public
   written policy. Footer links added.
4. **Self-serve data requests.** `/data-request` (GET form, POST handler):
   a valid 12-char job ID is **deleted immediately** (`_purge_job`); every
   request is logged to `data/data_requests.log` for fulfilment of
   access/correction/objection within the statutory window.
5. **Debug endpoints disabled in prod.** `/debug/preprocess` and `/debug/regions`
   ran face detection with no consent/geo gate; they now 404 unless
   `TYPO_ENABLE_DEBUG=1` (local dev only). Do **not** set it in production.
6. **Content moderation.** The user's words/message/title/caption (which become
   the portrait) are screened via OpenAI's moderation endpoint (`app/moderation.py`,
   key `OPENAI_API_KEY`) before render; a definitive flag returns 422. **Fails
   open** (allows) when unconfigured or on API error, so an outage can't take the
   product down -- Terms + the attestation are the backstop. **Uploaded-image NSFW
   is deliberately NOT auto-detected** (would mean shipping faces to a cloud or a
   model on a constrained box); instead: Terms prohibition + the reel admin-review
   for public content + a report path (`report_abuse` on `/data-request`).

### Config / env vars
| Var | Default | Purpose |
|-----|---------|---------|
| `TYPO_BLOCKED_REGIONS` | `US-IL` | Comma list of `COUNTRY-REGION` (or `COUNTRY`) tokens to block. `""` disables. |
| `TYPO_GEO_COUNTRY_HEADER` | `X-Geo-Country` | Header the proxy sets with ISO country (e.g. `US`). |
| `TYPO_GEO_REGION_HEADER` | `X-Geo-Region` | Header the proxy sets with the subdivision/state (e.g. `IL`). |
| `TYPO_BIO_CONSENT_VERSION` | `bio-v1-2026-06` | Bump when the consent wording materially changes. |
| `TYPO_RETENTION_DAYS` | `30` | Auto-delete window for photos / recipes / previews. |
| `TYPO_CONSENT_RETENTION_DAYS` | `2555` (~7y) | Separate, longer window for consent records (proof of consent + age/guardian attestation; no biometric data). Covers BIPA's 5-yr limitations period. |

---

## REQUIRED to make the geo-block real: wire a geo source

The block is inert until the proxy sets `X-Geo-Country` + `X-Geo-Region`. Use
nginx + the MaxMind GeoIP2 module (state-level needs the GeoLite2-City DB):

```nginx
# /etc/nginx/nginx.conf (http{} block)
geoip2 /usr/share/GeoIP/GeoLite2-City.mmdb {
    $geoip2_country_iso $country_iso_code;
    $geoip2_region_iso  subdivisions 0 iso_code;   # e.g. "IL" for Illinois
}
```
```nginx
# in the app.typortrait.com server{} / location that proxies to :8077
proxy_set_header X-Geo-Country $geoip2_country_iso;
proxy_set_header X-Geo-Region  $geoip2_region_iso;
```
Token format is `COUNTRY-REGION` (e.g. `US-IL`) — this is deliberate so the US
state `IL` is never confused with the ISO country `IL` (Israel). Both headers
must be set for state-level blocking to match.

**Verify after wiring:** from an Illinois IP, `GET /compliance/region` returns
`{"blocked": true}`, and `/measure` returns 451. From elsewhere, `{"blocked": false}`.

---

## Lawyer punch list (decisions / sign-off needed)

- [ ] **CSAM response plan (URGENT, operational/legal).** A face-upload app can be
      targeted. We do NOT (and cannot reliably) auto-detect CSAM. Required: a
      written response plan (preserve, do not redistribute, **report to NCMEC** per
      18 U.S.C. 2258A), a monitored abuse channel (`report_abuse` on `/data-request`
      feeds the admin email + log), and -- as volume grows -- enrol in **PhotoDNA**
      or a provider's CSAM scanning. Take this to counsel; it is not a code feature.
- [ ] **Confirm the biometric treatment.** We treat MediaPipe face-mesh as
      "biometric" (BIPA) / "special-category" (GDPR). Confirm this conservative
      stance and the consent + policy wording.
- [ ] **BIPA specifics:** confirm the written-release wording, the public
      retention/destruction policy at `/biometric-policy`, and that geo-blocking
      Illinois is the chosen risk posture (vs. also blocking TX/WA, or not).
- [ ] **GDPR:** appoint an **EU/UK Article 27 representative** if you have no
      EU/UK establishment and target those markets; confirm international-transfer
      mechanism (SCCs) and the legal-bases wording.
- [ ] **CCPA/CPRA:** confirm the California-rights section and that no activity
      counts as a "sale" or "share" (Umami is cookieless; confirm Stripe/Printful
      cookie behaviour — a cookie audit + possible banner for EU visitors).
- [ ] **Sub-processor DPAs:** execute/keep Data Processing Agreements with
      Stripe, Printful, the hosting provider, and Umami; maintain a record of
      processing (GDPR Art. 30).
- [ ] **Governing law / minors:** terms currently say NY law; confirm. The
      consent box now attests **age 13+** and, for a child's photo, parent/guardian
      status -- captured per job in `<job>.biometric_consent.json`
      (`age_guardian_attested`, `consent_version` bumped to `bio-v2-*`). Confirm
      the **13+ threshold** is the one you want (vs 18+), and that the attestation
      wording is sufficient for COPPA/BIPA on a minor's biometric data.
- [ ] **DSAR operations:** confirm who actions data requests and the SLA (GDPR 30
      days / CCPA 45 days). Every request now **emails `TYPO_ADMIN_EMAIL`**
      (`e86c106`) and is logged to `data/data_requests.log`; the human steps below
      still need an owner.
- [ ] **Consent-record retention:** photos delete at 30 days, but the minimal
      consent record (no biometric data) is **retained ~7 years** and is **kept
      even on a deletion request** (`_purge_job`) as proof that consent was given.
      Confirm this posture and the period (`TYPO_CONSENT_RETENTION_DAYS`) with
      counsel -- it trades full erasure for defensible proof-of-consent.

---

## How to handle a data request

When `/data-request` is submitted you get an email (`Data request · <type> · <email>`)
and a line in `data/data_requests.log`. The email is a **to-do**, not a "handled"
receipt — except a delete-by-job-ID, which is already done.

**Two rules for every request:** (1) respond within the window — ~30 days (GDPR) /
~45 days (CCPA); even a "done" reply counts. (2) **Verify the requester's identity
before disclosing or changing anything** (especially `access`); a delete-by-job-ID
is the exception (the job ID is a private token from their own receipt).

| Type | Auto-done? | Action |
|------|-----------|--------|
| `delete` + Job ID | ✅ files purged | Reply confirming. Optionally remove their order row too. |
| `delete` no Job ID | ❌ | Find by email (below), delete, reply. (Photos also auto-expire in ~30 days.) |
| `access` | ❌ | **Verify identity**, compile what we hold, send it, reply. |
| `correct` | ❌ | Verify identity, fix the field (e.g. name on an order), reply. |
| `object` / restrict | ❌ | Stop the named processing (e.g. revoke a reel feature opt-in), reply. |
| `takedown` (third party's photo) | ❌ | Locate + delete the job(s), reply confirming removal. |

**Where the data lives:** photos/renders → `private/` keyed by random job ID, and
**auto-deleted after ~30 days** (`TYPO_RETENTION_DAYS`); not searchable by email
unless they give the job ID, but they self-purge anyway. Order records (email +
shipping for prints) → `data/orders.db`, **searchable by email**. Sub-processors
(Stripe, Printful) hold copies for physical orders — a full erasure may require
asking them too.

**You may lawfully retain** order/payment/transaction records (tax, accounting,
fraud) even after a delete request — the privacy policy says so. So don't wipe
financial history; the obligation is to *respond*, and the sensitive (photo/
biometric) data is what gets deleted. Set an email filter on subject
`Data request ·` so these are never missed. Not legal advice — confirm the
identity-verification standard and retention periods with counsel.

---

## Deploy notes
Backend (`app/`) + frontend (`static/`) both changed → on prod:
`git pull` (static is live) **and** `docker compose up -d --build` (Python is baked).
After deploy, smoke-test: upload is blocked until `#bioConsent` is ticked;
`/privacy`, `/biometric-policy`, `/data-request` load; a data request with a job
ID deletes its files. Then wire the nginx geo headers and verify the 451 path.
