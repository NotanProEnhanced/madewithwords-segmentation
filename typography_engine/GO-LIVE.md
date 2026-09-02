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

### 1a. Get the files
Upload the **entire contents of the `marketing/` folder** — all 12 files. If you
miss the hero/sample images the page loads but shows blank panels. The files are:
`index.html`, `og.png`, `robots.txt`, `sitemap.xml`, `favicon.ico`,
`favicon-32.png`, `apple-touch-icon.png`, `hero-after.png`, `hero-before.jpg`,
`sample-1.jpg`, `sample-2.jpg`, `sample-3.jpg`.

### 1b. (Optional) change the price
The page is already a clean single-price design at **$14.99** with no placeholder
text — you can upload it as-is. Only if you want a different price: open
`index.html`, find `$14.99` (there are two: the hero note and the pricing card),
change both, and **keep the app in sync** by setting `TYPO_PRICE_CENTS` in Part 2
to match (e.g. `$14.99` → `1499`, `$9.00` → `900`). Save.

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

### 2d. Get the code
```bash
git clone https://github.com/notanproenhanced/madewithwords-segmentation.git typortrait
cd typortrait/typography_engine
```

### 2d-i. Add your Stripe keys (so people can pay to download)
1. Create a free account at **stripe.com**. In the Stripe Dashboard go to
   **Developers → API keys** and copy your **Secret key** (starts with `sk_test_`
   while testing, `sk_live_` when you go live).
2. On the VPS, create a `.env` file next to docker-compose (paste this block,
   editing the values):
   ```bash
   cat > .env <<'EOF'
   STRIPE_SECRET_KEY=PASTE_YOUR_STRIPE_SECRET_KEY_HERE
   TYPO_PRICE_CENTS=1499
   TYPO_CURRENCY=usd
   TYPO_PUBLIC_URL=https://app.typortrait.com
   EOF
   ```
   - Replace `PASTE_YOUR_STRIPE_SECRET_KEY_HERE` with the key from Stripe
     (it begins with `sk_` — keep it secret).
   - `TYPO_PRICE_CENTS=1499` means **$14.99** per download — this **must match**
     the price shown on the marketing page (Part 1b). Change both together.
   - Keep `.env` private (never commit it).
   - No Stripe key yet? You can still launch — people will see the free
     watermarked preview; the Download button just says checkout isn't set up.

### 2d-ii. Build and start the app
```bash
sudo docker compose up -d --build      # first build ~5-10 min (models baked in)
curl -s http://127.0.0.1:8077/health   # -> {"ok": true, ...}
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

## How payments work (and testing)
- Anyone can create and view a portrait for free — it shows with a
  **"Typortrait.com" watermark + a QR code**.
- **Download** opens **Stripe Checkout**; after a successful payment the
  **clean, watermark-free file downloads automatically**.
- **Test it** (while using your `sk_test_` key) with Stripe's test card
  `4242 4242 4242 4242`, any future expiry, any CVC/ZIP — no real charge.
- **Go live:** in Stripe switch to **live mode**, copy the `sk_live_` key into
  `.env`, then `sudo docker compose up -d` to apply. Real cards now work.

## (Optional) Physical prints via Printful

The app can also sell **framed prints, canvas, posters, and t-shirts**, fulfilled
and shipped automatically by **Printful**. This is **fully optional**: with no
Printful token the storefront stays exactly as above (digital download only), and
the physical products simply don't appear. Turn it on whenever you're ready.

**How it works:** the customer picks a product in the studio → pays in Stripe
(Stripe also collects their shipping address) → Stripe calls our `/webhook/stripe`
→ the app submits the order to Printful, which prints and ships it. You watch
every order at **`https://app.typortrait.com/admin/orders`** (same password as the
reel admin — `TYPO_ADMIN_PASSWORD`).

### 3a. Get your Printful credentials
1. Create a store at **printful.com** and connect a payment method (Printful
   charges *you* wholesale when an order comes in; your customer already paid you
   via Stripe).
2. In Printful → **Settings → API** (or **Developers**), create an **API token**.
3. Note your **Store ID** (shown in the API/developer area).

### 3b. Register the Stripe webhook
1. Stripe Dashboard → **Developers → Webhooks → Add endpoint**.
2. **Endpoint URL:** `https://app.typortrait.com/webhook/stripe`
3. **Event to send:** `checkout.session.completed`
4. Save, then click the endpoint and copy its **Signing secret** (starts with
   `whsec_`). This lets the app trust that callbacks really came from Stripe.

### 3c. Add the keys to `.env` and restart
Add these lines to the same `.env` file from step 2d-i (keep the existing ones):
```bash
PRINTFUL_API_TOKEN=PASTE_YOUR_PRINTFUL_TOKEN
PRINTFUL_STORE_ID=PASTE_YOUR_STORE_ID
STRIPE_WEBHOOK_SECRET=whsec_PASTE_FROM_STRIPE
```
Then apply:
```bash
sudo docker compose up -d --build
```
The product picker now appears under the preview, and `/admin/orders` starts
logging orders. To turn physical prints back off, remove `PRINTFUL_API_TOKEN`
and restart — the digital flow is unaffected either way.

### 3d. Test a physical sale safely (draft mode) BEFORE going live
**Important:** Printful has no "test mode" — a confirmed order is really
printed, shipped, and billed to you. To rehearse a sale without any charge,
put the app in **draft mode** first:
1. While still using your Stripe **test** key (`sk_test_…`), add this line to
   `.env`, then `sudo docker compose up -d` to apply:
   ```bash
   PRINTFUL_CONFIRM=false
   ```
   In draft mode every paid physical order is created at Printful as an
   **unconfirmed draft** — never charged, never printed.
2. Buy a physical product through the studio using Stripe's test card
   `4242 4242 4242 4242` (any future expiry, any CVC/ZIP).
3. In your Printful dashboard, open the new draft order and check:
   - it's the **right product** (e.g. 16×20 framed poster), and
   - your portrait shows as the **print file** (Printful fetched the art).
   You can also watch it at `https://app.typortrait.com/admin/orders`.
4. **Delete the draft** in Printful when satisfied — no charge, no print.
5. **Go live:** remove `PRINTFUL_CONFIRM` from `.env` (or set it to `true`),
   switch Stripe to live mode + `sk_live_` key, then `sudo docker compose up -d`.
   Real orders now confirm and ship automatically.

> **Order history is saved.** Orders live in a small database at
> `typography_engine/data/orders.db`, which is volume-mounted, so it survives
> `docker compose up --build` and code updates. Don't delete the `data/` folder.

### 3c. The Printful webhook (shipment updates)

Printful's classic webhooks carry **no signature**, so `/webhook/printful` is guarded by a
secret in the query string: the registered URL is
`https://app.typortrait.com/webhook/printful?k=<PRINTFUL_WEBHOOK_SECRET>`.

**There is no page in the Printful dashboard for this.** Classic webhooks are registered
only through the API (`POST /webhooks`), and Printful keeps **one** webhook URL per store —
registering a new one replaces the old, including its event types. Settings → API in the
dashboard is only where the API token lives.

    # what is registered now (nothing is changed)
    /root/typortrait-prod/typography_engine/ops/rotate-printful-secret.sh

    # new secret, updated .env, re-registered URL, verified
    /root/typortrait-prod/typography_engine/ops/rotate-printful-secret.sh --rotate

Rotate if the secret is ever exposed. Someone holding it can forge a `package_shipped` for
an order whose id they already know, which marks it shipped and emails that customer a
"your order has shipped" message with a tracking link of their choosing, from your domain.
Order ids are `uuid4().hex[:12]` and cannot be guessed, so the risk is narrow — but the
email goes out under your name.

> **Verify the Printful variant IDs before your first real sale.** The product
> sizes/IDs are in `app/products.py`; if Printful changes a catalog ID, update it
> there. Run one test order end-to-end (Stripe test mode + a cheap product) to
> confirm it reaches Printful as "draft/pending" before going live.

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
