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
At your DNS provider, point the domain at the VPS public IP. With a separate
marketing page on the root, use a subdomain for the app:
```
A   app.typortrait.com   -> <VPS_IP>
```
Wait for it to resolve (`dig +short typortrait.com`).

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
```bash
sudo cp deploy/nginx-typortrait.conf /etc/nginx/sites-available/typortrait.conf
sudo ln -sf /etc/nginx/sites-available/typortrait.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```
Visit http://typortrait.com — you should see the app.

## 4. Enable HTTPS (free, auto-renewing)
```bash
sudo certbot --nginx -d typortrait.com -d www.typortrait.com
```
Certbot adds the 443 server block and an HTTP->HTTPS redirect, and sets up
auto-renewal. Done — open **https://typortrait.com** on phone or desktop.

## Updating after code changes
```bash
cd typortrait && git pull
cd typography_engine && sudo docker compose up -d --build
```

---

## Alternative: native install (no Docker), behind nginx
Prefer a lean install without Docker? This runs uvicorn under systemd; nginx +
certbot (steps 3–4 above) are identical.

```bash
# 1. System libraries the renderer needs (Cairo / OpenCV / MediaPipe + fonts)
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip nginx certbot python3-certbot-nginx git \
  libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi8 \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 fonts-dejavu-core fonts-liberation

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
- **Marketing page:** you can keep IONOS shared hosting for a static landing
  page on a different host/subdomain if you ever want to separate them.
