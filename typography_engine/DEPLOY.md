# Deploying Typortrait to an Ubuntu VPS (typortrait.com)

> **Beginner?** Use **GO-LIVE.md** — it's the full click-by-click walkthrough
> (Windows/PowerShell) and is the source of truth, including the Stripe setup.
> This file is the condensed technical reference.

The engine is a Python/FastAPI server with native deps (Cairo, OpenCV,
MediaPipe). It needs the **VPS**, not shared hosting. Below is a Docker-based
deploy with nginx + Let's Encrypt TLS. Run everything as a sudo user.

**Payments:** before `docker compose up`, create a `.env` next to
`docker-compose.yml` so the freemium download flow works:
```
STRIPE_SECRET_KEY=your_stripe_secret_key
TYPO_PRICE_CENTS=1499
TYPO_CURRENCY=usd
TYPO_PUBLIC_URL=https://app.typortrait.com
```
Compose reads it automatically and persists previews + paid files via the
`./data` volumes. Without a key, the app still runs (free watermarked previews;
the Download button reports that checkout isn't configured).

## 0. DNS (do this first so TLS can verify)
Both names live on **this same VPS**. Point an A record for each at the VPS
public IP. The apex serves the **static marketing page**; the `app.` subdomain
serves the **FastAPI studio**:
```
A   typortrait.com       -> <VPS_IP>
A   www.typortrait.com   -> <VPS_IP>
A   app.typortrait.com   -> <VPS_IP>
```
Wait for them to resolve (`dig +short typortrait.com app.typortrait.com`).

## 1. Install Docker + nginx + certbot
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin nginx certbot python3-certbot-nginx git
sudo systemctl enable --now docker
```

## 2. Get the code and build/run the container
```bash
git clone <YOUR_REPO_URL> typortrait
cd typortrait/typography_engine          # build context is this directory
sudo docker compose up -d --build        # first build ~5-10 min (models baked in)
curl -s http://127.0.0.1:8077/health     # -> {"ok": true, ...}
```
The app now listens on 127.0.0.1:8077 and restarts automatically (reboot/crash).

## 3. Put nginx in front
The conf defines **two vhosts**: the apex `typortrait.com` as a static site
(root `/var/www/typortrait.com`) and `app.typortrait.com` as the reverse proxy
to the app on `127.0.0.1:8077`.
```bash
sudo mkdir -p /var/www/typortrait.com          # static marketing docroot
sudo cp deploy/nginx-typortrait.conf /etc/nginx/sites-available/typortrait.conf
sudo ln -sf /etc/nginx/sites-available/typortrait.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```
Visit http://app.typortrait.com — you should see the studio. The apex
http://typortrait.com serves the marketing page once you deploy it (step 6).

## 4. Enable HTTPS (free, auto-renewing)
Cover all three names (apex + www + app) in one cert:
```bash
sudo certbot --nginx -d typortrait.com -d www.typortrait.com -d app.typortrait.com
```
Certbot adds the 443 server blocks and HTTP->HTTPS redirects, and sets up
auto-renewal. Done — open **https://app.typortrait.com** (studio) and
**https://typortrait.com** (marketing) on phone or desktop.

## Updating after code changes
```bash
cd typortrait && git pull
cd typography_engine && sudo docker compose up -d --build
```

## 6. Deploy / update the marketing page (static, served by nginx)
The apex site is plain static files in `/var/www/typortrait.com/` — **not** the
app, and **not** any shared host. The canonical source is
`typography_engine/marketing/_deploy/` in this repo. To publish (or update) it,
copy that folder's contents into the docroot. From your **local machine**:
```powershell
# upload the staged files to the VPS
scp -r typography_engine/marketing/_deploy <user>@<VPS_IP>:~/
```
Then on the **VPS**:
```bash
# back up the current page (instant rollback), then swap in the new files
cp /var/www/typortrait.com/index.html /var/www/typortrait.com/index.backup.html
cp ~/_deploy/* /var/www/typortrait.com/
chown www-data:www-data /var/www/typortrait.com/*
```
Static files go live instantly — no nginx reload, no app rebuild. Roll back with
`cp /var/www/typortrait.com/index.backup.html /var/www/typortrait.com/index.html`.

## 5. Admin review dashboard + email notifications (Phase C)

When a buyer opts in to letting Typortrait feature their reel on social
channels, the reel lands in a review queue. You manage it from two
surfaces sharing the same state machine:

- **Dashboard** at `https://app.typortrait.com/admin/reels`
  (password-gated, lifecycle: queued → approved → posted, with reject /
  revoke that purge the reel files while keeping the consent record).
- **Email notifications** to `TYPO_ADMIN_EMAIL` with one-tap Approve /
  Reject links (signed, 14-day TTL). Works from a phone; no login needed.

Add these to `.env` next to `docker-compose.yml`:
```
TYPO_ADMIN_PASSWORD=pick-a-long-random-string
TYPO_SECRET_KEY=another-long-random-string
TYPO_ADMIN_EMAIL=you@example.com
TYPO_SMTP_HOST=smtp.gmail.com
TYPO_SMTP_PORT=587
TYPO_SMTP_USER=you@gmail.com
TYPO_SMTP_PASS=your-16-char-gmail-app-password
```

Generate two strong secrets:
```bash
openssl rand -hex 32   # use one value for TYPO_ADMIN_PASSWORD
openssl rand -hex 32   # use the other for TYPO_SECRET_KEY
```

### Getting a Gmail App Password (one-time, 5 minutes)

Gmail does not accept your normal account password over SMTP. You need a
16-character "App Password" tied to your Google account.

1. Sign in to the Gmail account you want to send from
   (e.g., `jjtokarz57@gmail.com`).
2. Go to **https://myaccount.google.com/security**.
3. Under **"How you sign in to Google"**, make sure **2-Step
   Verification** is **On**. App passwords are only available with 2FA on.
4. Open **https://myaccount.google.com/apppasswords**.
5. Name it something like `Typortrait SMTP` and click **Create**.
6. Google shows a **16-character password** (with spaces) — copy it now,
   you can't see it again.
7. Paste it into `.env` as `TYPO_SMTP_PASS` **without the spaces**
   (e.g. `abcdefghijklmnop`).
8. Set `TYPO_SMTP_USER` to the same Gmail address.

Apply with `sudo docker compose up -d --build`. The scanner runs every
60 seconds; the next queued reel will trigger an email.

If you skip the SMTP variables, the dashboard still works — you'll just
need to check it manually instead of being notified.

---

## Alternative: native install (no Docker), behind nginx
Prefer a lean install without Docker? This runs uvicorn under systemd; nginx +
certbot (steps 3–4 above) are identical.

```bash
# 1. System libraries the renderer needs (Cairo / OpenCV / MediaPipe + fonts)
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip nginx certbot python3-certbot-nginx git \
  libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi8 \
  libgl1 libgles2 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 fonts-dejavu-core fonts-liberation

# 2. Code + virtualenv
sudo git clone <YOUR_REPO_URL> /opt/typortrait
cd /opt/typortrait/typography_engine
sudo python3 -m venv .venv
sudo .venv/bin/pip install --upgrade pip
sudo .venv/bin/pip install -r requirements.txt
sudo chown -R www-data:www-data /opt/typortrait

# 3. Service (downloads the MediaPipe models on first request)
sudo cp deploy/typortrait.service /etc/systemd/system/typortrait.service
sudo systemctl daemon-reload && sudo systemctl enable --now typortrait
curl -s http://127.0.0.1:8077/health      # -> {"ok": true, ...}
```
Then do steps **3 (nginx)** and **4 (certbot)** above. Update later with:
`cd /opt/typortrait && sudo git pull && sudo systemctl restart typortrait`.

## Notes / sizing
- **RAM:** budget ~2 GB (MediaPipe + per-render image work). A 2 vCPU / 2-4 GB VPS is fine for low/moderate traffic.
- **Concurrency:** one uvicorn worker (a render is a few CPU-seconds). For more
  simultaneous users, raise `--workers` in the Dockerfile CMD and add RAM, or
  run multiple replicas behind nginx.
- **Disk:** rendered SVG/PNG accumulate in the container's `outputs/`. For long
  runs, add a cron to prune old files, or mount `outputs/` as a volume with a
  cleanup job.
- **Marketing page:** served as static files by nginx on this same VPS from
  `/var/www/typortrait.com/` (apex), while the app runs on `app.typortrait.com`.
  See step 6 for how to update it. (It does **not** live on IONOS shared hosting
  — the apex A record points straight at the VPS.)
