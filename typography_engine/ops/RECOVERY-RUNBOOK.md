# Recovery Runbook — step by step

Exact commands to recover Typortrait™ / Loved in Words™. Pair with the Disaster
Recovery Plan PDF (the "why"); this is the "how." Before you start, have your
**restic repo password** (from your password manager) and **B2 keys** ready — you
recreate the creds files first.

Repo: `git@github.com:NotanProEnhanced/madewithwords-segmentation.git` · deployed
branch: `pet-engine-drape-tone-tiers` · on the VPS that remote is named
**`github`**, not `origin`.

There are **five** deployments, not one. Section A rebuilds production; Section
A-bis brings the other four back. See `ops/README.md` for the tree/port table.

---

## A. Total server loss — full rebuild

### 1. Provision + install
Bring up a new Ubuntu server (same version), then:
```bash
apt-get update && apt-get install -y docker.io docker-compose-plugin nginx certbot restic sqlite3 git curl
```

### 2. Get the code
```bash
git clone git@github.com:NotanProEnhanced/madewithwords-segmentation.git /root/typortrait-prod
cd /root/typortrait-prod/typography_engine
git checkout pet-engine-drape-tone-tiers
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
APP=/root/typortrait-prod/typography_engine
cp "$R$APP/.env" "$APP/.env" && chmod 600 "$APP/.env"
mkdir -p "$APP/data/gather"
# databases are staged per tree; check what the snapshot holds before copying
ls "$R/root/.typortrait-backup-staging/"
cp "$R/root/.typortrait-backup-staging/typortrait-prod/orders.db" "$APP/data/orders.db"
cp "$R/root/.typortrait-backup-staging/typortrait-prod/gather.db" "$APP/data/gather/gather.db"
cp -a "$R$APP/data/private" "$APP/data/"
cp -a "$R/etc/nginx/sites-available/." /etc/nginx/sites-available/
cp -a "$R/etc/letsencrypt/." /etc/letsencrypt/
```

### 6. Bring up the app
```bash
cd "$APP"
docker compose up -d --build
```
There is now ONE compose file, shared by all five trees. Per-tree identity
(`COMPOSE_PROJECT`, `IMAGE_TAG`, `CONTAINER_NAME`, `HOST_PORT`) comes from that
tree's `.env`, so the same command brings up whichever tree you are standing in.
`docker-compose.staging.yml` no longer exists.

Check `docker compose config` succeeds before `up`: a bad variable substitution
surfaces there rather than by taking a service down.

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

### 9. Restore the marketing static sites
`typortrait.com` is now **tracked in git** at `sites/typortrait.com/` and
published with `ops/deploy-site.sh`, so the repo is the source of truth:
```bash
./ops/deploy-site.sh --apply
./ops/site-drift.sh            # expect "live site matches the repo"
```
The other web roots are not in git. Restore those from the backup, where they
were placed under `$R/var/www` in steps 4–5:
```bash
mkdir -p /var/www
cp -a "$R/var/www/lovedinwords.com" /var/www/     # and any other roots present
ls "$R/var/www"                                    # check what the snapshot has
```

### 10. Reconcile + smoke test + re-arm backups
```bash
# reconcile in-flight orders against Stripe (payments) and Printful (fulfilment)
# smoke test: a render, a test checkout, /privacy, both marketing sites
(crontab -l 2>/dev/null; echo "0 * * * * /root/typortrait-prod/typography_engine/ops/backup.sh") | crontab -
```

---

## A-bis. The other four deployments

Section A rebuilds production only. Four more trees serve the other brands, and
a rebuild that stops at step 10 leaves them down.

```bash
for T in typortrait-stg typortrait-faithinwords typortrait-lovedinwords typortrait-pawsinwords; do
  git clone git@github.com:NotanProEnhanced/madewithwords-segmentation.git /root/$T
  git -C /root/$T checkout pet-engine-drape-tone-tiers
done
```

Each tree then needs its **own `.env`** — that file carries the brand, the
Stripe and Printful credentials, and the four identity keys the shared compose
template reads (`COMPOSE_PROJECT`, `IMAGE_TAG`, `CONTAINER_NAME`, `HOST_PORT`).
Without the right `.env` a tree will start, and serve the wrong brand on the
wrong port against the wrong store.

The `.env` files are **not** in the restic snapshot — that covers production
only. They are in the nightly config archive:

```bash
source /root/.backup-env                 # exports BACKUP_RCLONE
rclone lsf "$BACKUP_RCLONE" | sort | tail -1          # newest archive
rclone copy "$BACKUP_RCLONE/<archive>" /root/
gpg --decrypt --passphrase-file /root/.backup-pass /root/<archive> | tar -xzf - -C /root/restore-config
# then, per tree:
cp /root/restore-config/<tree>/.env /root/<tree>/typography_engine/.env
chmod 600 /root/<tree>/typography_engine/.env
cp /root/restore-config/<tree>/data/orders.db /root/<tree>/typography_engine/data/
cp -a /root/restore-config/<tree>/data/private/. /root/<tree>/typography_engine/data/private/
```

Bring each up and confirm the port matches the table in `ops/README.md`:
```bash
cd /root/<tree>/typography_engine && docker compose config >/dev/null && docker compose up -d --build
```

Then check the fleet with `./ops/env-lint.py` and `./ops/compose-diff.py` before
repointing DNS.

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
