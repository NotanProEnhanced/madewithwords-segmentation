# Typortrait — Go-Live Guide (beginner, Windows/PowerShell)

You're setting up **two** things:

| What | Where it lives | Address |
|------|----------------|---------|
| **Marketing page** (the landing site) | IONOS **shared hosting** | `https://typortrait.com` |
| **The app** (makes the portraits) | IONOS **VPS** (Ubuntu) | `https://app.typortrait.com` |

The marketing page's buttons send people to the app. Do **Part 1** then **Part 2**.

Collect these first (write them down):
- **VPS public IP** — IONOS panel → your VPS/Cloud server → "IPv4" (looks like `203.0.113.45`).
- **Your code's Git URL** — e.g. `https://github.com/notanproenhanced/madewithwords-segmentation.git`, and the branch `claude/typography-portrait-engine-NBy2u`.
- Your IONOS login (for hosting File Manager + DNS).

---

# PART 1 — Marketing page on IONOS (no commands needed)

### 1a. Download the files
From this chat, save these into one folder on your PC (e.g. `Documents\typortrait-site`):
`index.html`, `og.png`, `robots.txt`, `sitemap.xml`, `favicon.ico`, `favicon-32.png`, `apple-touch-icon.png`.

### 1b. Edit two things (open `index.html` in Notepad)
- **Prices:** press Ctrl+H (Find/Replace) or scroll to the `<!-- EDIT PRICES -->` line. Change the `$29 / $89 / $119` and the feature text to your real offer.
- **Testimonials:** search for `Placeholder, replace me`. Replace all three quotes and names with **real** customer quotes (or delete that whole `<section id="love">…</section>` block until you have some — don't leave fake reviews live).
- Save the file.

### 1c. Upload to IONOS
1. Log in at ionos.com → **Hosting** → your package → **"Webspace"** or **"File Manager"** (sometimes called *Web Space Explorer*).
2. Open the website root folder (often named `/`, `htdocs`, or your domain name).
3. **Upload all 7 files** into that root.
4. Make sure `index.html` is in the root (it's the homepage).

### 1d. Point the domain at the hosting
If your domain `typortrait.com` and the hosting are in the **same IONOS account**, this is usually already done. To check: IONOS → **Domains** → `typortrait.com` → it should be "assigned" to your hosting package. If not, assign it.

### 1e. Test
Open `https://typortrait.com` on your computer and phone. You should see the page. (DNS changes can take up to a few hours the first time.)

✅ Part 1 done.

---

# PART 2 — The app on your Ubuntu VPS

### 2a. Open the firewall (in IONOS)
IONOS Cloud panel → your VPS → **Network / Firewall policies** → make sure **inbound ports 22 (SSH), 80 (HTTP), 443 (HTTPS)** are **allowed**. Save.

### 2b. Connect to the VPS from Windows PowerShell
1. Press **Start**, type **PowerShell**, open it.
2. Type this (replace the IP with your VPS IPv4):
   ```powershell
   ssh root@YOUR_VPS_IP
   ```
3. First time it asks *"Are you sure you want to continue connecting?"* — type `yes` and press Enter.
4. Enter the **root password** IONOS gave you (typing is invisible — that's normal). Press Enter.

You're now "inside" the VPS. Every command below is typed **in this same window**. Copy a whole block, right-click in PowerShell to paste, press Enter.

### 2c. Install the tools
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin nginx certbot python3-certbot-nginx git
sudo systemctl enable --now docker
```

### 2d. Get the code and start the app
Replace the URL/branch with yours:
```bash
git clone -b claude/typography-portrait-engine-NBy2u https://github.com/notanproenhanced/madewithwords-segmentation.git typortrait
cd typortrait/typography_engine
sudo docker compose up -d --build
```
The first build downloads everything and takes ~5–10 minutes. When it finishes, test it's alive:
```bash
curl -s http://127.0.0.1:8077/health
```
You should see `{"ok": true, ...}`.

> If `git clone` asks for a username/password, your repo is **private**. Easiest fix: make it public, or create a GitHub "personal access token" and clone with
> `https://YOUR_TOKEN@github.com/OWNER/REPO.git`.

### 2e. Put nginx in front (paste this whole block)
```bash
sudo tee /etc/nginx/sites-available/typortrait.conf >/dev/null <<'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name app.typortrait.com;
    client_max_body_size 25m;
    location / {
        proxy_pass http://127.0.0.1:8077;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/typortrait.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

### 2f. Point `app.typortrait.com` at the VPS (in IONOS DNS)
IONOS → **Domains** → `typortrait.com` → **DNS**. Add a record:
- **Type:** A
- **Host name:** `app`
- **Points to / Value:** your VPS IPv4
- Save. (Wait a few minutes; check by visiting `http://app.typortrait.com` — it should load the app over plain HTTP.)

### 2g. Turn on HTTPS (free, auto-renews)
Back in PowerShell (still on the VPS):
```bash
sudo certbot --nginx -d app.typortrait.com
```
Answer the prompts (enter your email, agree). When it finishes, open **https://app.typortrait.com** on your phone and desktop — the real app with upload + words + styles + download. 🎉

---

## Updating the app later (after I push changes)
```bash
ssh root@YOUR_VPS_IP
cd typortrait && git pull
cd typography_engine && sudo docker compose up -d --build
```

## If something's wrong
- App not responding: `sudo docker compose logs --tail=50` (run in `typortrait/typography_engine`).
- nginx error: `sudo nginx -t` shows the problem line.
- "site can't be reached": DNS not ready yet, or firewall ports 80/443 not open (step 2a).
- Restart everything: `sudo docker compose restart` and `sudo systemctl restart nginx`.

## To leave the VPS
Type `exit` and press Enter.
