# ops — operating the fleet

Everything here runs on the VPS as root. Nothing here is imported by the
application; these are operator tools, safe to read before you run them.

Five deployments live on one box, each a full checkout of this repo:

| tree                       | brand / role            | port |
|----------------------------|-------------------------|------|
| `/root/typortrait-prod`    | app.typortrait.com      | 8077 |
| `/root/typortrait-stg`     | staging — testing only  | 8078 |
| `/root/typortrait-faithinwords` | faithinwords.com   | 8079 |
| `/root/typortrait-lovedinwords` | lovedinwords.com   | 8080 |
| `/root/typortrait-pawsinwords`  | pawsinwords.com    | 8081 |

> **The directory names lie about their roles, historically.** Until August 2026
> the *production* tree was called `typortrait-staging` and staging was
> `typortrait-stg` — one letter apart, opposite meanings. Production is now
> `typortrait-prod`. If you find a script, cron line or document still saying
> `/root/typortrait-staging`, it predates the rename and is pointing at nothing.

All five track branch `pet-engine-drape-tone-tiers` on remote **`github`**
(`git@github.com:NotanProEnhanced/madewithwords-segmentation.git`). Note the
remote is *not* called `origin` — `origin` is a local path to a sibling tree.

---

## There are two backup systems, and they do different jobs

This is the thing most likely to confuse whoever reads this next, so it is first.

### 1. `backup.sh` — restic, hourly, the full disaster-recovery set
Covers **one tree only** (`typortrait-prod`): both databases, the customer
source photos, consent records, `.env`, the live web roots, nginx and
Let's Encrypt. Deduplicated and encrypted, with two-tier retention so photos
roll off at ~35 days (matching the deletion promise) while consent records are
kept for the legal window. Setup: `BACKUP-SETUP.md`. Credentials:
`/root/.typortrait-backup.env`. Helper: `rc.sh`.

### 2. `backup-config.sh` — gpg + rclone, nightly, config and orders for **all five**
Covers what is not in git and cannot be rebuilt: each tree's `.env` and
`docker-compose.yml`, a consistent `orders.db` snapshot, the order/consent JSON,
plus the crontab and nginx config. Deliberately **excludes** customer source
photos. Pushed to `b2:typortrait-backups-jt/config`.
Verified by `restore-verify.sh`; re-verified monthly from B2 by
`verify-monthly.sh`.

### The gap between them
Four of the five trees (`faithinwords`, `lovedinwords`, `pawsinwords`, `stg`)
have their **config, orders and consent records** backed up nightly, but their
**customer source photos and `gather.db` are not backed up at all**. For photos
that is arguably correct — they are deleted at ~30 days by design. For
`gather.db` it is probably an oversight. Decide deliberately rather than by
accident; `backup.sh` is written for a single `APP_DIR` and would need a loop.

---

## The scripts

### Fleet
| script | what it does |
|---|---|
| `tt` | fleet tool — run a command across every tree, see status at a glance |
| `env-lint.py` | flags `.env` problems: short secrets, unreachable keys, missing values |
| `compose-diff.py` | compares the five compose files key by key |
| `compose-env-file.py` | checks each tree declares `env_file` (without it, `.env` is silently ignored) |
| `compose-template.py` | builds the one shared compose template; `--apply` installs it |

> `.env` values are silently ignored unless the service declares `env_file:`.
> A key can sit in `.env`, look correct, and do nothing. That cost three
> separate debugging sessions before `env_file` was added everywhere.

### Backup and recovery
| script | what it does |
|---|---|
| `backup.sh` | restic hourly backup (prod tree) |
| `restore.sh` | restic restore into a review directory — never overwrites live data |
| `rc.sh` | `source` it to load restic credentials into your shell |
| `backup-config.sh` | nightly encrypted config/orders backup, all five trees |
| `restore-verify.sh` | decrypt an archive and prove every tree is recoverable from it |
| `verify-monthly.sh` | pull the newest archive back from B2 and verify it; emails on failure |

### Site and orders
| script | what it does |
|---|---|
| `deploy-site.sh` | publish `sites/typortrait.com` from git to `/var/www` |
| `site-drift.sh` | detect the live site drifting from the repo |
| `close-stale-orders.py` | close orders stuck in a non-terminal status; dry run by default |
| `pf-check-orders.py` | read-only: look up orders at Printful with a chosen token |

### Documents
`BACKUP-SETUP.md` (one-time restic setup) · `RECOVERY-RUNBOOK.md` (rebuild,
command by command) · `INCIDENT-RESPONSE.md`.

---

## Scheduled work

```
0 * * * *   backup.sh          hourly restic  -- NOT SCHEDULED as of 2026-08-29
30 3 * * *  backup-config.sh   nightly config + orders to B2
0 4 1 * *   verify-monthly.sh  monthly restore proof
```

`backup.sh` was never added to cron. It ran three times between 1 July and 12
August, by hand. Two consequences, both fixed but worth understanding: there was
no hourly recovery point despite the setup guide promising one, and the
count-based `--keep-daily 35` retention never expired anything, so customer
source photos sat in the repository for 59 days against a ~30-day deletion
promise. Retention is now time-based — but it still only runs when a backup
runs, so **the cron line is part of the privacy commitment, not just the
recovery plan.**

Check what is *actually* scheduled with `crontab -l` rather than trusting this
table. A backup that silently stopped looks exactly like one that is fine, right
up until the day you need it — which is the entire reason `verify-monthly.sh`
exists and why it fails loudly when the newest archive is more than three days
old.

## Staging is a real test environment — and holds a real Printful token

Staging (`typortrait-stg`, port 8078, behind basic auth at
staging.typortrait.com) runs the full purchase path:

- **Stripe:** a test-mode restricted key (`rk_test_`) with exactly two scopes —
  Checkout Sessions: Write and PaymentIntents: Write. Those are every Stripe
  operation the app performs. Test card `4242 4242 4242 4242`.
- **Printful:** the **real** token and store id, because Printful has no test
  mode, with **`PRINTFUL_CONFIRM=false`** so orders land as unconfirmed drafts —
  visible in the dashboard, deletable, never printed, never billed.

> ⚠️ `PRINTFUL_CONFIRM=false` is the entire safety margin. Set it to true, or
> overwrite staging's `.env` with production's, and a staging test purchase
> sends a real job to a real printer. `env-lint.py` fails loudly on that
> combination — run it before any test purchase.

nginx exempts `/webhook/stripe` and `/printful-fetch/` from basic auth, because
Stripe and Printful cannot authenticate. Don't remove those exemptions.

Drafts land in the **production** Printful store, since staging uses the same
store id. Delete test drafts when you are done with them.

## Per-tree render settings that are NOT in git

`.env` holds credentials so it stays untracked, which means a setting made there
exists only on the box (the nightly config archive captures it; nothing else does).
Anything non-obvious belongs here.

**`PET_TORSO_FILL=0.55`** on `typortrait-lovedinwords` and `typortrait-faithinwords`.

ISNet's confidence fades toward the bottom of a frame and reaches literal zero: on a
head-and-shoulders portrait the matte measured 0.93 coverage at the face and 0.00 in
the bottom band, losing the neck, collar and shoulders — the "floating head".
`_solidify_matte()` cannot fix it, because it recovers only regions the image border
cannot reach and a neck runs off the bottom edge. This setting carries any column that
already extends past 55% of the frame height down to the bottom.

> Deliberately **not set on `typortrait-pawsinwords`.** Those renders are good today,
> and a pet photographed with floor visible below is exactly the case where extending
> columns downward would smear. It is off by default in code for the same reason.

Diagnose with `ops/pet-matte-probe.py` — copy it into a tree's `data/` and run it in
the container. It prints matte coverage per horizontal band before and after
`_solidify_matte`, which separates "the model never saw it" from "the threshold
discarded it". Three rounds of adjusting `PET_MATTE_FILL` were wasted before that
probe existed; the two tables answered it in one run.

## Conventions worth knowing

- **Dry run first.** `close-stale-orders.py`, `compose-template.py`,
  `deploy-site.sh` and `site-drift.sh` all report before they write and need an
  explicit `--apply`.
- **Secrets never appear in this repo, in arguments, or in output.** Tokens and
  passphrases are read from root-only files. Create them with `nano`, remove them
  with `shred -u`. Scripts print key *names* and lengths, never values.
- **Customer source photos are not copied into long-lived archives.** They are
  personal data covered by a ~30-day deletion promise, and an archive that keeps
  them for a year quietly breaks it.
- **`app/` is baked into the image; `static/` and `data/` are bind-mounted.**
  Changes under `app/` need `docker compose up -d --build`; `static/` changes
  take effect on reload.
