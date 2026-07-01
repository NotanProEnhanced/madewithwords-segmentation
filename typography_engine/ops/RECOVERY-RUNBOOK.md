# Recovery Runbook — step by step

Exact commands to recover Typortrait™ / Loved in Words™. Pair with the Disaster
Recovery Plan PDF (the "why"); this is the "how." Before you start, have your
**restic repo password** (from your password manager) and **B2 keys** ready — you
recreate the creds files first.

Repo: `git@github.com:NotanProEnhanced/madewithwords-segmentation.git` · deployed branch: `feat/displacement-style`

---

## A. Total server loss — full rebuild

### 1. Provision + install
Bring up a new Ubuntu server (same version), then:
```bash
apt-get update && apt-get install -y docker.io docker-compose-plugin nginx certbot restic sqlite3 git curl
```

### 2. Get the code
```bash
git clone git@github.com:NotanProEnhanced/madewithwords-segmentation.git /root/typortrait-staging
cd /root/typortrait-staging/typography_engine
git checkout feat/displacement-style
```

### 3. Recreate the backup credentials (from your offline copies)
```bash
printf '%s' 'YOUR_SAVED_RESTIC_PASSWORD' > /root/.typortrait-restic-pass && chmod 600 /root/.typortrait-restic-pass
nano /root/.typortrait-backup.env      # recreate the 4 export lines (repo + B2 keys)
chmod 600 /root/.typortrait-backup.env
```

### 4. Restore the data from B2
```bash
source ops/rc.sh                       # load restic env
restic snapshots                       # confirm the repo is reachable
./ops/restore.sh /root/typortrait-restore
```

### 5. Put the files back in place
```bash
R=/root/typortrait-restore
APP=/root/typortrait-staging/typography_engine
cp "$R$APP/.env" "$APP/.env" && chmod 600 "$APP/.env"
mkdir -p "$APP/data/gather"
cp "$R/root/.typortrait-backup-staging/orders.db" "$APP/data/orders.db"
cp "$R/root/.typortrait-backup-staging/gather.db" "$APP/data/gather/gather.db"
cp -a "$R$APP/data/private" "$APP/data/"
cp -a "$R/etc/nginx/sites-available/." /etc/nginx/sites-available/
cp -a "$R/etc/letsencrypt/." /etc/letsencrypt/
```

### 6. Bring up the app
```bash
cd "$APP"
docker compose up -d --build                                                       # prod
docker compose -p typortrait-staging -f docker-compose.staging.yml up -d --build   # staging (optional)
```

### 7. Restore nginx + TLS
```bash
for s in typortrait.com lovedinwords.com typortrait-app.conf staging.typortrait.com; do
  ln -sf /etc/nginx/sites-available/$s /etc/nginx/sites-enabled/ 2>/dev/null || true
done
nginx -t && systemctl reload nginx
# if certs didn't restore cleanly, re-issue:
# certbot --nginx -d typortrait.com -d www.typortrait.com -d app.typortrait.com -d lovedinwords.com
```

### 8. Repoint DNS
At your registrar, point the A records for `typortrait.com`, `www`, `app.typortrait.com`,
`lovedinwords.com` (and staging) to the **new server IP**. Wait for propagation, verify HTTPS.

### 9. Restore the marketing static sites (from the backup)
The live web roots are backed up to B2 and were restored under `$R/var/www` in
step 4/5, so put them back exactly as served — no repo copy or guesswork:
```bash
mkdir -p /var/www
cp -a "$R/var/www/typortrait.com"   /var/www/
cp -a "$R/var/www/lovedinwords.com" /var/www/
```
(FYI, in the repo `typortrait.com` is served from `marketing/_deploy/` and
`lovedinwords.com` from `marketing/lovedinwords/`; restoring the web roots from B2
also recovers the assets that aren't tracked in git.)

### 10. Reconcile + smoke test + re-arm backups
```bash
# reconcile in-flight orders against Stripe (payments) and Printful (fulfilment)
# smoke test: a render, a test checkout, /privacy, both marketing sites
(crontab -l 2>/dev/null; echo "0 * * * * /root/typortrait-staging/typography_engine/ops/backup.sh") | crontab -
```

---

## B. Restore a single lost/corrupted file (no full rebuild)
```bash
source ops/rc.sh
./ops/restore.sh /root/typortrait-restore            # 'latest'  (or add a snapshot id)
# copy just the affected file/dir into place, then reconcile vs Stripe/Printful.
```

## C. Security breach
1. Isolate the server (firewall off / stop containers).
2. Rotate **every** secret: Stripe, Printful, OpenAI, SMTP app password, `TYPO_ADMIN_PASSWORD`,
   `TYPO_SECRET_KEY`, and the B2 + restic backup credentials.
3. Rebuild on a clean server (Section A), restoring from a snapshot dated **before** the breach.
4. Notify per PIPEDA (OPC + affected individuals) if there's a real risk of significant harm; keep a breach record.

## Quarterly restore drill
```bash
source ops/rc.sh && ./ops/restore.sh /root/test-restore && ls -R /root/test-restore | head && rm -rf /root/test-restore
```
