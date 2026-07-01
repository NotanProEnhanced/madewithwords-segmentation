# Off-site backup — one-time setup

Encrypted, incremental, off-site backups via **restic**. Run hourly → ~1-hour
recovery point. The scripts (`ops/backup.sh`, `ops/restore.sh`) contain **no
secrets**; all credentials live in a root-only file you create below.

> ⚠️ **The restic repo password is the one thing you must never lose.** Store a
> copy **offline** (password manager). If the server dies AND you don't have the
> password, the backups cannot be decrypted. This is the single most important
> step.

## 1. Install the tools (on the VPS)
```bash
apt-get update && apt-get install -y restic sqlite3 curl
```

## 2. Create the off-site bucket
Backblaze **B2** is cheapest and works natively with restic (AWS S3 also works).
- Create a **private** B2 bucket, e.g. `typortrait-backups`.
- Create an **application key** scoped to that bucket → note the *keyID* and *key*.

## 3. Create the repo password (keep a copy OFFLINE)
```bash
openssl rand -base64 32 > /root/.typortrait-restic-pass
chmod 600 /root/.typortrait-restic-pass
cat /root/.typortrait-restic-pass          # copy this into your password manager NOW
```

## 4. Create the creds file `/root/.typortrait-backup.env`
```bash
cat > /root/.typortrait-backup.env <<'ENV'
export RESTIC_REPOSITORY="b2:typortrait-backups:typortrait"
export RESTIC_PASSWORD_FILE="/root/.typortrait-restic-pass"
export B2_ACCOUNT_ID="YOUR_B2_KEY_ID"
export B2_ACCOUNT_KEY="YOUR_B2_APP_KEY"
ENV
chmod 600 /root/.typortrait-backup.env
```
*(For AWS S3 instead: set `RESTIC_REPOSITORY="s3:s3.amazonaws.com/your-bucket"`
and `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.)*

## 5. First run (initialises the repo + first backup)
```bash
chmod +x /root/typortrait-staging/typography_engine/ops/backup.sh
chmod +x /root/typortrait-staging/typography_engine/ops/restore.sh
/root/typortrait-staging/typography_engine/ops/backup.sh
tail -n 30 /var/log/typortrait-backup.log       # expect "backup ok"
```

## 6. Schedule it (hourly)
```bash
crontab -e
# add:
0 * * * * /root/typortrait-staging/typography_engine/ops/backup.sh
```

## 7. (Recommended) Dead-man's-switch alerting
Create a free check at healthchecks.io, set its ping URL as `HEALTHCHECK_URL`
at the top of `backup.sh`. It pings on success and on failure; if a run is ever
missed, healthchecks.io emails you.

## 8. Test the restore (do this quarterly)
```bash
/root/typortrait-staging/typography_engine/ops/restore.sh /root/test-restore
ls -R /root/test-restore        # confirm the DBs, private/, .env are present
rm -rf /root/test-restore
```

## What gets backed up
`data/orders.db`, `data/gather/gather.db` (consistent snapshots), `private/`
(recipes + consent records), `.env`, `/etc/nginx/sites-available`,
`/etc/letsencrypt`.

**Retention:** 24 hourly · 30 daily · 12 monthly · 7 yearly (the yearly line
covers the ~7-year consent-record retention).

## What is NOT backed up (by design — it's recoverable elsewhere)
Code (GitHub) · container images (rebuilt) · `outputs/` (regenerated from
recipes) · `models/` (re-downloaded) · payments (Stripe) · fulfilment (Printful).
