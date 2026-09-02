"""Phase C: admin surface for reviewing user-consented reels.

State machine: queued -> approved -> posted, with reject (admin decision)
or revoke (user-requested removal) terminal states that purge the reel
files from outputs/ while preserving the consent record for audit.

Two surfaces share this state:
  * Email notifications to TYPO_ADMIN_EMAIL with one-tap HMAC-signed
    Approve / Reject links (sent by a background thread).
  * Password-gated dashboard at /admin/reels for the full lifecycle,
    including marking posted with platform URLs and revocation.

Graceful degradation: missing TYPO_ADMIN_PASSWORD / TYPO_SECRET_KEY
shows a clear setup hint on /admin; missing SMTP creds disables the
background mailer but the dashboard still works.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import smtplib
import threading
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Cookie, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .config import DATA_DIR, OUTPUTS_DIR, PRIVATE_DIR, PUBLIC_BASE_URL, RETENTION_DAYS


# --- Configuration --------------------------------------------------------
ADMIN_PASSWORD = os.environ.get("TYPO_ADMIN_PASSWORD", "")
ADMIN_EMAIL    = os.environ.get("TYPO_ADMIN_EMAIL", "")
SECRET_KEY     = os.environ.get("TYPO_SECRET_KEY", "")
SMTP_HOST      = os.environ.get("TYPO_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.environ.get("TYPO_SMTP_PORT", "587") or "587")
SMTP_USER      = os.environ.get("TYPO_SMTP_USER", "")
SMTP_PASS      = os.environ.get("TYPO_SMTP_PASS", "")
# Optional brand display name for the From header (per-container), e.g. "Loved in
# Words". Recipients then see 'Loved in Words <account@…>' instead of a bare
# personal address — friendlier and less spam-prone. Unset => bare address (old behavior).
MAIL_FROM_NAME = os.environ.get("TYPO_MAIL_FROM_NAME", "").strip()
COOKIE_SECURE  = PUBLIC_BASE_URL.lower().startswith("https://")

REEL_STATES = ("queued", "approved", "posted", "rejected", "revoked")

router = APIRouter(prefix="/admin", tags=["admin"])

# The container runs a single uvicorn worker, so process-local locking
# is sufficient to serialize state writes.
_review_lock = threading.RLock()


# --- HMAC signing ---------------------------------------------------------

def _signing_key() -> bytes:
    if not SECRET_KEY:
        raise RuntimeError("TYPO_SECRET_KEY is not set")
    return SECRET_KEY.encode("utf-8")


def _sign(payload: str) -> str:
    return hmac.new(_signing_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def make_email_token(job: str, action: str, ttl_seconds: int = 14 * 86400) -> str:
    exp = int(time.time()) + ttl_seconds
    payload = f"{job}|{action}|{exp}"
    raw = f"{payload}|{_sign(payload)}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def verify_email_token(token: str):
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded).decode("utf-8")
        job, action, exp_s, sig = raw.split("|")
        if action not in ("approve", "reject"):
            return None
        if int(exp_s) < int(time.time()):
            return None
        if not hmac.compare_digest(_sign(f"{job}|{action}|{exp_s}"), sig):
            return None
        return job, action
    except Exception:  # noqa: BLE001
        return None


def make_session_cookie() -> str:
    payload = f"admin|{int(time.time())}"
    return f"{payload}|{_sign(payload)}"


def verify_session_cookie(cookie: Optional[str], max_age: int = 30 * 86400) -> bool:
    if not cookie or not SECRET_KEY:
        return False
    try:
        role, ts, sig = cookie.split("|")
        if role != "admin":
            return False
        if int(ts) < int(time.time()) - max_age:
            return False
        return hmac.compare_digest(_sign(f"{role}|{ts}"), sig)
    except Exception:  # noqa: BLE001
        return False


def _require_admin(session: Optional[str]):
    """Return a Response if the request is not authenticated, else None."""
    if not ADMIN_PASSWORD or not SECRET_KEY:
        return HTMLResponse(_admin_chrome(
            "Admin not configured",
            "<h1>Admin not configured</h1>"
            "<p>Set <code>TYPO_ADMIN_PASSWORD</code> and <code>TYPO_SECRET_KEY</code> "
            "in your <code>.env</code> file, then restart the container.</p>"
        ), status_code=503)
    if not verify_session_cookie(session):
        return RedirectResponse(url="/admin/login", status_code=302)
    return None


# --- Review record I/O ----------------------------------------------------

def review_path(job: str) -> Path:
    return PRIVATE_DIR / f"{job}.review.json"


def consent_path(job: str) -> Path:
    return PRIVATE_DIR / f"{job}.consent.json"


def read_review(job: str):
    p = review_path(job)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def write_review(job: str, **updates) -> dict:
    with _review_lock:
        cur = read_review(job) or {
            "job": job, "state": "queued", "ts_queued": int(time.time()),
            "emailed_at": None, "posted_links": [], "notes": "",
        }
        cur.update(updates)
        target = review_path(job)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
        tmp.replace(target)
        return cur


def init_review(job: str) -> dict:
    """Create a review record for a freshly-queued reel (called by /reel)."""
    with _review_lock:
        if not review_path(job).exists():
            return write_review(job, state="queued", ts_queued=int(time.time()),
                                emailed_at=None, posted_links=[], notes="")
        return read_review(job)


def purge_reel_files(job: str) -> int:
    """Remove reel artifacts from outputs/ for a rejected or revoked job.
    The consent + review records stay so the audit trail is preserved."""
    n = 0
    for name in (f"{job}_reel.gif", f"{job}_reel.mp4", f"{job}_before.jpg"):
        p = OUTPUTS_DIR / name
        if p.exists():
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    return n


def delete_reel(job: str) -> dict:
    """Delete a reel from the dashboard: always purge the reel artifacts and
    the review record so the job disappears from the admin surface.

    Compliance guard: if the reel was ever *posted* (its likeness used
    publicly), the consent record is PRESERVED as proof of authorization —
    you can still delete the video and de-list it, but the audit trail
    survives. For every other state (queued / rejected / test / personal),
    it's a full hard delete with no trace kept.

    Returns {"files": <count removed>, "consent_kept": <bool>}."""
    with _review_lock:
        rec = read_review(job) or {}
        posted = bool(rec.get("state") == "posted" or rec.get("ts_posted"))
        n = purge_reel_files(job)
        targets = [review_path(job)]
        if not posted:
            targets.append(consent_path(job))
        for p in targets:
            if p.exists():
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
    return {"files": n, "consent_kept": posted}


def transition(job: str, new_state: str, **extra) -> dict:
    if new_state not in REEL_STATES:
        raise ValueError(f"unknown state: {new_state}")
    updates = {"state": new_state, f"ts_{new_state}": int(time.time())}
    updates.update(extra)
    rec = write_review(job, **updates)
    if new_state in ("rejected", "revoked"):
        purge_reel_files(job)
    return rec


def list_reviews():
    out = []
    try:
        for p in sorted(PRIVATE_DIR.glob("*.review.json"),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
    except OSError:
        pass
    return out


# --- Stats / analytics ----------------------------------------------------

def compute_stats():
    """Aggregate counts from PRIVATE_DIR sidecars. Bounded by RETENTION_DAYS
    because the retention sweep removes older files; the UI surfaces that."""
    now = int(time.time())
    day_secs = 86400
    by_day = {}
    days_window = max(7, RETENTION_DAYS)
    for i in range(days_window):
        d = time.strftime("%Y-%m-%d", time.localtime(now - i * day_secs))
        by_day[d] = {"renders": 0, "paid": 0, "reels": 0, "consents": 0}

    stats = {
        "total_renders": 0,
        "paid_orders": 0,
        "reels_created": 0,
        "typortrait_consents": 0,
        "saves_captured": 0,
        "states": {s: 0 for s in REEL_STATES},
        "inks": {},
        "styles": {},
        "uppercase_count": 0,
        "remove_bg_count": 0,
        "light_count": 0,
        "word_count_buckets": {"1": 0, "2-3": 0, "4-6": 0, "7-12": 0, "13+": 0},
        "by_day": [],
        "retention_days": RETENTION_DAYS,
    }

    # Recipes (renders). Excludes the .consent.json / .review.json sidecars
    # by checking that the stem has no further dots.
    try:
        for p in PRIVATE_DIR.glob("*.json"):
            if "." in p.stem:  # skip <job>.consent and <job>.review
                continue
            stats["total_renders"] += 1
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
                ink = (r.get("ink") or "?").strip() or "?"
                style = (r.get("style") or "?").strip() or "?"
                stats["inks"][ink] = stats["inks"].get(ink, 0) + 1
                stats["styles"][style] = stats["styles"].get(style, 0) + 1
                if r.get("uppercase"):
                    stats["uppercase_count"] += 1
                if r.get("remove_bg"):
                    stats["remove_bg_count"] += 1
                if r.get("light"):
                    stats["light_count"] += 1
                text = r.get("text") or ""
                tokens = [w for w in text.replace(",", " ").replace("\n", " ").split() if w]
                n = len(tokens)
                if n == 1:
                    stats["word_count_buckets"]["1"] += 1
                elif n <= 3:
                    stats["word_count_buckets"]["2-3"] += 1
                elif n <= 6:
                    stats["word_count_buckets"]["4-6"] += 1
                elif n <= 12:
                    stats["word_count_buckets"]["7-12"] += 1
                else:
                    stats["word_count_buckets"]["13+"] += 1
            except Exception:  # noqa: BLE001
                pass
            d = time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))
            if d in by_day:
                by_day[d]["renders"] += 1
    except OSError:
        pass

    # Paid orders (.paid markers, dropped on first verified paid /download).
    try:
        for p in PRIVATE_DIR.glob("*.paid"):
            stats["paid_orders"] += 1
            d = time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))
            if d in by_day:
                by_day[d]["paid"] += 1
    except OSError:
        pass

    # Reels created (any paid user who clicked "Make my reel" leaves a
    # consent record, whether or not they opted in to Typortrait sharing).
    try:
        for p in PRIVATE_DIR.glob("*.consent.json"):
            stats["reels_created"] += 1
            d = time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))
            if d in by_day:
                by_day[d]["reels"] += 1
    except OSError:
        pass

    # Typortrait-share consents + lifecycle state distribution.
    try:
        for p in PRIVATE_DIR.glob("*.review.json"):
            stats["typortrait_consents"] += 1
            d = time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))
            if d in by_day:
                by_day[d]["consents"] += 1
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
                st = r.get("state", "queued")
                if st in stats["states"]:
                    stats["states"][st] += 1
            except Exception:  # noqa: BLE001
                pass
    except OSError:
        pass

    # Exit-intent "save my portrait" captures (owned remarketing contacts).
    try:
        for p in PRIVATE_DIR.glob("*.savecapture.json"):
            stats["saves_captured"] += 1
    except OSError:
        pass

    stats["by_day"] = [(d, by_day[d]) for d in sorted(by_day.keys(), reverse=True)]
    return stats


def referral_funnel(since_ts: int = 0):
    """Per-source funnel — renders -> sales -> revenue — grouped by the `?ref`
    tag, for events at or after `since_ts` (epoch seconds; 0 = all time).

    Renders come from the recipe sidecars (so they're bounded by the retention
    sweep); sales + revenue come from the purchase ledger (all-time). Empty/absent
    ref counts as "direct". Best-effort: never raises."""
    src = {}   # source -> {"renders": n, "sales": n, "revenue_cents": x}

    def slot(name: str) -> dict:
        return src.setdefault(name or "direct",
                              {"renders": 0, "sales": 0, "revenue_cents": 0})

    # Renders (recipe sidecars), filtered by file mtime.
    try:
        for p in PRIVATE_DIR.glob("*.json"):
            if "." in p.stem:                      # skip .consent / .review sidecars
                continue
            try:
                if since_ts and p.stat().st_mtime < since_ts:
                    continue
                r = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            slot(str(r.get("ref") or "direct"))["renders"] += 1
    except OSError:
        pass

    # Sales (purchase ledger), filtered by record ts. Older "1" markers (no
    # detail) won't parse as a dict and are skipped.
    try:
        for p in (DATA_DIR / "purchase_events").glob("*.txt"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(rec, dict):
                continue
            if since_ts and int(rec.get("ts") or 0) < since_ts:
                continue
            sl = slot(str(rec.get("source") or "direct"))
            sl["sales"] += 1
            sl["revenue_cents"] += int(rec.get("amount_cents") or 0)
    except OSError:
        pass

    return src


def _pct(num: float, den: float) -> str:
    if not den:
        return "—"
    return f"{(100.0 * num / den):.1f}%"


# --- Legacy migration -----------------------------------------------------

def migrate_legacy_queue_markers() -> int:
    """Promote any pre-Phase-C `<job>.queue` markers into `.review.json`
    so they show up on the dashboard. Idempotent."""
    n = 0
    try:
        for p in PRIVATE_DIR.glob("*.queue"):
            job = p.stem
            if not review_path(job).exists():
                try:
                    legacy = json.loads(p.read_text(encoding="utf-8"))
                    ts = int(legacy.get("queued_ts", time.time()))
                except Exception:  # noqa: BLE001
                    ts = int(time.time())
                write_review(job, state="queued", ts_queued=ts,
                             emailed_at=None, posted_links=[], notes="")
                n += 1
            try:
                p.unlink()
            except OSError:
                pass
    except OSError:
        pass
    return n


# --- Email --------------------------------------------------------------

def _from_header() -> str:
    """From header for outgoing mail. Prepends the brand display name
    (TYPO_MAIL_FROM_NAME) when set, so recipients see 'Loved in Words <account@…>'
    rather than a bare personal address; falls back to the bare address otherwise.
    The Name in the pair is what mail clients show, without changing the SMTP account."""
    if MAIL_FROM_NAME:
        # Wrap the name in an EmailMessage-safe form; commas/specials in a display
        # name must be quoted to stay RFC-5322 valid.
        safe = MAIL_FROM_NAME.replace('"', "").replace("\\", "")
        return f'"{safe}" <{SMTP_USER}>' if safe else SMTP_USER
    return SMTP_USER


def smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS and ADMIN_EMAIL and SECRET_KEY)


def send_email(to: str, subject: str, body_html: str, body_text: str) -> bool:
    """Send an email to an ARBITRARY recipient (e.g. a gather contributor who asked
    to be told when the portrait is ready) via the same Gmail SMTP path. Best-effort;
    needs only the SMTP creds (not ADMIN_EMAIL/SECRET_KEY). Returns True on send."""
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and to):
        _log("send_email skipped (smtp not configured / no recipient)")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = _from_header()
    msg["To"] = to
    msg.set_content(body_text or " ")
    if body_html:
        msg.add_alternative(body_html, subtype="html")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        _log(f"send_email ok to={to}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"send_email failed to={to}: {e}")
        return False


def _read_words_for_job(job: str):
    rp = PRIVATE_DIR / f"{job}.json"
    if not rp.exists():
        return []
    try:
        r = json.loads(rp.read_text(encoding="utf-8"))
        raw = (r.get("text") or "").replace(",", " ").replace("\n", " ")
        return [w for w in raw.split() if w]
    except Exception:  # noqa: BLE001
        return []


def send_sale_email(summary: dict) -> bool:
    """Email the admin a 'you made a sale' notice. Best-effort; reuses the same
    Gmail SMTP path. `summary` keys: product, amount (formatted str), email, ref,
    order_id, shipping (recipient dict|None), digital_included (bool)."""
    if not smtp_configured():
        _log("send_sale_email skipped (smtp not configured)")
        return False
    product = html.escape(str(summary.get("product") or "Order"))
    amount = html.escape(str(summary.get("amount") or ""))
    cust = html.escape(str(summary.get("email") or "—"))
    ref = html.escape(str(summary.get("ref") or ""))
    oid = html.escape(str(summary.get("order_id") or ""))
    ts_s = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    rows = [f"<li><b>Item:</b> {product}</li>", f"<li><b>Amount:</b> {amount}</li>",
            f"<li><b>Customer:</b> {cust}</li>"]
    if ref:
        rows.append(f"<li><b>Source:</b> {ref}</li>")
    if oid:
        rows.append(f"<li><b>Order:</b> {oid}</li>")
    if summary.get("digital_included"):
        rows.append("<li>High-res digital file included free.</li>")
    ship = summary.get("shipping") or None
    ship_html = ""
    if ship and ship.get("address1"):
        parts = [ship.get("name"), ship.get("address1"), ship.get("address2"),
                 ship.get("city"), ship.get("state_code"), ship.get("zip"), ship.get("country_code")]
        ship_html = "<p><b>Ship to:</b> " + html.escape(", ".join(p for p in parts if p)) + "</p>"
    dash = f"{PUBLIC_BASE_URL}/admin"
    body_html = f"""<div style="font-family:-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;color:#16203a">
<p>New sale.</p>
<ul>{''.join(rows)}</ul>
{ship_html}
<p style="color:#6b7280;font-size:13px">{ts_s} &middot; <a href="{dash}">admin dashboard</a></p>
</div>"""
    body_text = (f"New sale.\nItem: {summary.get('product')}\nAmount: {summary.get('amount')}\n"
                 f"Customer: {summary.get('email')}\nOrder: {summary.get('order_id') or '-'}\n")
    msg = EmailMessage()
    msg["Subject"] = f"New sale · {summary.get('product') or 'Order'} · {summary.get('amount') or ''}"
    msg["From"] = _from_header()
    msg["To"] = ADMIN_EMAIL
    msg.set_content(body_text)
    msg.add_alternative(body_html, subtype="html")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        _log(f"send_sale_email ok: order={summary.get('order_id')} to={ADMIN_EMAIL}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"send_sale_email FAILED: err={type(e).__name__}: {e}")
        return False


def send_data_request_email(rec: dict) -> bool:
    """Email the admin when a GDPR/CCPA data request arrives, so the statutory
    response window (GDPR ~30 days, CCPA ~45 days) isn't missed. Best-effort;
    reuses the Gmail SMTP path. `rec` keys: type, email, job_id, deleted_files,
    details, region, ts."""
    if not smtp_configured():
        _log("send_data_request_email skipped (smtp not configured)")
        return False
    rtype = html.escape(str(rec.get("type") or "request"))
    cust = html.escape(str(rec.get("email") or "—"))
    jid = html.escape(str(rec.get("job_id") or ""))
    deleted = int(rec.get("deleted_files") or 0)
    details = html.escape(str(rec.get("details") or ""))
    region = html.escape(str(rec.get("region") or ""))
    ts_s = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    rows = [f"<li><b>Type:</b> {rtype}</li>", f"<li><b>From:</b> {cust}</li>"]
    if jid:
        rows.append(f"<li><b>Job ID:</b> {jid} &mdash; {deleted} file(s) deleted automatically</li>")
    if region:
        rows.append(f"<li><b>Region:</b> {region}</li>")
    if details:
        rows.append(f"<li><b>Details:</b> {details}</li>")
    body_html = f"""<div style="font-family:-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;color:#16203a">
<p>New data request &mdash; please action within the statutory window (GDPR ~30 days, CCPA ~45 days). Verify the requester's identity before disclosing any data.</p>
<ul>{''.join(rows)}</ul>
<p style="color:#6b7280;font-size:13px">{ts_s} &middot; also logged to data/data_requests.log</p>
</div>"""
    body_text = (f"New data request.\nType: {rec.get('type')}\nFrom: {rec.get('email')}\n"
                 f"Job ID: {rec.get('job_id') or '-'} ({deleted} deleted)\nDetails: {rec.get('details') or '-'}\n")
    msg = EmailMessage()
    msg["Subject"] = f"Data request · {rec.get('type') or 'request'} · {rec.get('email') or ''}"
    msg["From"] = _from_header()
    msg["To"] = ADMIN_EMAIL
    msg.set_content(body_text)
    msg.add_alternative(body_html, subtype="html")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        _log(f"send_data_request_email ok: type={rec.get('type')} to={ADMIN_EMAIL}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"send_data_request_email FAILED: err={type(e).__name__}: {e}")
        return False


def send_review_email(job: str, words, consent: dict) -> bool:
    if not smtp_configured():
        _log(f"send_review_email skipped (smtp not configured): job={job}")
        return False
    approve = f"{PUBLIC_BASE_URL}/admin/email/approve?token={make_email_token(job, 'approve')}"
    reject  = f"{PUBLIC_BASE_URL}/admin/email/reject?token={make_email_token(job, 'reject')}"
    dash    = f"{PUBLIC_BASE_URL}/admin/reels/{job}"
    mp4_url = f"{PUBLIC_BASE_URL}/outputs/{job}_reel.mp4"
    gif_url = f"{PUBLIC_BASE_URL}/outputs/{job}_reel.gif"
    words_s = html.escape(" · ".join(words[:6]) if words else "")
    ts_s = time.strftime("%Y-%m-%d %H:%M:%S UTC",
                         time.gmtime(consent.get("ts", time.time())))
    body_html = f"""<div style="font-family:-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;color:#16203a">
<p>A user has consented to let Typortrait feature their reel on our social channels.</p>
<ul>
  <li><b>Job:</b> {html.escape(job)}</li>
  <li><b>Words:</b> {words_s}</li>
  <li><b>Consent recorded:</b> {ts_s} (text version {html.escape(str(consent.get('consent_text_version','')))})</li>
</ul>
<p><a href="{mp4_url}">Preview MP4</a> &middot; <a href="{gif_url}">Preview GIF</a></p>
<p>
  <a href="{approve}" style="background:#0d1b3a;color:#fff;padding:10px 22px;border-radius:999px;text-decoration:none;display:inline-block;font-weight:600">Approve</a>
  &nbsp;
  <a href="{reject}" style="background:#fff;color:#0d1b3a;border:1.5px solid #0d1b3a;padding:10px 22px;border-radius:999px;text-decoration:none;display:inline-block;font-weight:600">Reject</a>
</p>
<p style="color:#6b7280;font-size:13px">
  Or <a href="{dash}">open in dashboard</a> to mark it posted, revoke, or see the full record.
</p>
</div>"""
    body_text = (
        f"New reel awaiting review.\n"
        f"Job: {job}\n"
        f"Words: {' '.join(words[:6])}\n"
        f"Approve: {approve}\n"
        f"Reject:  {reject}\n"
        f"Dashboard: {dash}\n"
    )
    msg = EmailMessage()
    msg["Subject"] = f"Typortrait reel awaiting review · {job}"
    msg["From"] = _from_header()
    msg["To"] = ADMIN_EMAIL
    msg.set_content(body_text)
    msg.add_alternative(body_html, subtype="html")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        _log(f"send_review_email ok: job={job} to={ADMIN_EMAIL}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"send_review_email FAILED: job={job} err={type(e).__name__}: {e}")
        return False


def save_capture_path(job: str) -> Path:
    return PRIVATE_DIR / f"{job}.savecapture.json"


def record_save_capture(job: str, email: str, emailed: bool = False) -> dict:
    """Persist an exit-intent 'save my portrait' email capture as a sidecar.

    Feeds the admin funnel (a recovered/owned remarketing contact) and keeps
    a record of who asked us to hold their portrait. Idempotent per job: a
    repeat capture just refreshes the timestamp / emailed flag.

    The filename uses a dotted stem (`<job>.savecapture`) so compute_stats'
    render glob skips it, exactly like the consent/review sidecars."""
    rec = {"job": job, "email": email, "ts": int(time.time()), "emailed": bool(emailed)}
    with _review_lock:
        target = save_capture_path(job)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        tmp.replace(target)
    return rec


def send_save_link_email(job: str, to_email: str, brand_name: str = "Typortrait") -> bool:
    """Customer-facing email: deliver a private link back to a portrait the
    user previewed but didn't buy, so they can return before the ~RETENTION_DAYS
    auto-delete. Reuses the same Gmail SMTP path as the admin mailer. `brand_name`
    keeps the copy brand-specific (Paws in Words / Loved in Words / …)."""
    if not smtp_configured():
        _log(f"send_save_link_email skipped (smtp not configured): job={job}")
        return False
    # "Your Typortrait" reads fine for the generic brand; a sub-brand becomes "Your <Brand> portrait".
    saved_label = "Typortrait" if brand_name == "Typortrait" else f"{brand_name} portrait"
    link = f"{PUBLIC_BASE_URL}/resume/{html.escape(job)}"
    body_html = f"""<div style="font-family:-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;color:#16203a;max-width:520px">
<h2 style="font-family:Georgia,'Times New Roman',serif;color:#0d1b3a;margin:0 0 6px">Your {html.escape(saved_label)} is saved.</h2>
<p style="color:#2b3550;font-size:15px;line-height:1.6">Here's your private link — open it anytime to come back, download your high-res file, or order a print.</p>
<p style="margin:18px 0">
  <a href="{link}" style="background:#0d1b3a;color:#fff;padding:13px 28px;border-radius:999px;text-decoration:none;display:inline-block;font-weight:700">Open my portrait</a>
</p>
<p style="color:#6b7280;font-size:13px;line-height:1.6">A heads-up: we automatically delete uploaded photos and generated files after about {RETENTION_DAYS} days, so don't wait too long. If this wasn't you, just ignore this email.</p>
<p style="color:#9aa1ac;font-size:12px;line-height:1.6;margin-top:14px">You're in control of your data &mdash; <a href="{PUBLIC_BASE_URL}/data-request" style="color:#6b7280">manage or delete it anytime</a>.</p>
</div>"""
    body_text = (
        f"Your {saved_label} is saved.\n\n"
        f"Open it anytime: {link}\n\n"
        f"Heads-up: we auto-delete uploaded photos and generated files after about "
        f"{RETENTION_DAYS} days, so don't wait too long.\n\n"
        f"You're in control of your data -- manage or delete it anytime: {PUBLIC_BASE_URL}/data-request\n\n"
        "If this wasn't you, just ignore this email.\n"
    )
    msg = EmailMessage()
    msg["Subject"] = f"Your {saved_label} is saved — here's your link"
    msg["From"] = _from_header()
    msg["To"] = to_email
    msg.set_content(body_text)
    msg.add_alternative(body_html, subtype="html")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        _log(f"send_save_link_email ok: job={job} to={to_email}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"send_save_link_email FAILED: job={job} err={type(e).__name__}: {e}")
        return False


def keepsake_code_email_bodies(code_fmt: str, link: str, product_name: str = "", name: str = ""):
    """(html, text) for the buyer's 'here is your redemption code' email. SINGLE
    source of truth — the /admin/issue form and the CLI (tools/issue_code.py) both
    call this, so the email can never drift. `product_name` is accepted for call
    compatibility but not shown: technical SKU names ('High-res digital download')
    clash with the warm voice, so the copy says 'word portrait' for every product."""
    nm = (name or "").strip()
    hi = f"Hi {nm}," if nm else "Hello,"
    hi_h = html.escape(hi)
    lk = html.escape(link)
    cf = html.escape(code_fmt)
    img = f"{PUBLIC_BASE_URL.rstrip('/')}/static/lovedinwords/after.jpg"
    text = (
        f"{hi}\n\n"
        f"Thank you. Your word portrait is ready to personalize.\n\n"
        f"Your redemption code: {code_fmt}\n\n"
        f"Open this link to add a favorite photo and the words that describe your loved one,\n"
        f"see the portrait come together, and send it to us:\n\n"
        f"    {link}\n\n"
        f"Buying this as a gift? You can forward this email to the family — the link is\n"
        f"theirs to personalize, whenever they're ready.\n\n"
        f"Your code works once. Please keep it private — anyone with the code can redeem it.\n\n"
        f"With care,\nLoved in Words\n"
    )
    body_html = f"""<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#2b2f36;max-width:520px;line-height:1.6">
<p style="font-size:30px;line-height:1;color:#c85b57;margin:0 0 6px">&#9829;</p>
<h2 style="font-family:Georgia,'Times New Roman',serif;color:#42525f;font-size:22px;margin:0 0 12px">Your keepsake is ready to personalize</h2>
<p style="text-align:center;margin:0 0 18px">
  <img src="{img}" width="360" alt="A word portrait — a loved one's likeness formed from the words that describe them" style="width:100%;max-width:360px;height:auto;border-radius:14px;display:inline-block">
</p>
<p style="margin:0 0 4px">{hi_h}</p>
<p style="margin:0 0 16px">Thank you. Your <b>word portrait</b> is ready. Add a favorite photo and the words that describe your loved one, watch their portrait take shape, and send it to us.</p>
<p style="margin:0 0 6px;color:#7a756e;font-size:14px">Your redemption code</p>
<p style="font-size:22px;letter-spacing:.08em;font-weight:700;color:#42525f;background:#f4efe8;border-radius:12px;padding:14px;text-align:center;margin:0 0 18px">{cf}</p>
<p style="text-align:center;margin:0 0 18px">
  <a href="{lk}" style="background:#c85b57;color:#fff;padding:13px 30px;border-radius:999px;text-decoration:none;display:inline-block;font-weight:700">Personalize my keepsake</a>
</p>
<p style="margin:0 0 16px;color:#7a756e;font-size:14px;font-style:italic">Buying this as a gift? Forward this email to the family &mdash; the link is theirs to personalize, whenever they&rsquo;re ready.</p>
<p style="color:#9a948c;font-size:12.5px;margin:0">Your code works once. Please keep it private &mdash; anyone with the code can redeem it.</p>
<p style="color:#a49e95;font-size:12px;margin:18px 0 0">With care, Loved in Words &middot; powered by Typortrait.com</p>
</div>"""
    return body_html, text


def issue_redemption_code(sku: str, email: str, name: str = "", batch: Optional[str] = None) -> dict:
    """Generate ONE redemption code for `sku` and email it to the buyer. The
    programmatic twin of tools/issue_code.py, powering the /admin/issue form.
    Returns {ok, code, link, emailed, product} or {ok: False, error}."""
    from . import products, redemption
    product = products.get(sku)
    if not product:
        return {"ok": False, "error": f"Unknown SKU: {sku}"}
    email = (email or "").strip()
    if "@" not in email or " " in email or "." not in email.rsplit("@", 1)[-1]:
        return {"ok": False, "error": f"That doesn't look like a valid email address: {email!r}"}
    try:
        redemption.init_db()
        code_fmt = redemption.generate(sku, 1, batch=(batch or None),
                                       note=f"issued to {email} (admin form)")[0]
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Could not generate a code: {e}"}
    link = f"{PUBLIC_BASE_URL.rstrip('/')}/redeem?code={code_fmt}"
    body_html, body_text = keepsake_code_email_bodies(code_fmt, link, product.name, name)
    emailed = send_email(email, "Your keepsake is ready to personalize", body_html, body_text)
    return {"ok": True, "code": code_fmt, "link": link, "emailed": emailed,
            "product": product.name}


def _log(msg: str) -> None:
    """Surface admin/scanner events to docker compose logs typortrait."""
    import sys
    print(f"[admin] {msg}", file=sys.stderr, flush=True)


# --- Background scanner ---------------------------------------------------

def _scanner_loop() -> None:
    _log(f"scanner started (smtp_configured={smtp_configured()})")
    while True:
        try:
            if smtp_configured():
                for rec in list_reviews():
                    if rec.get("state") != "queued" or rec.get("emailed_at"):
                        continue
                    job = rec.get("job", "")
                    consent = {}
                    if consent_path(job).exists():
                        try:
                            consent = json.loads(consent_path(job).read_text(encoding="utf-8"))
                        except Exception:  # noqa: BLE001
                            pass
                    if send_review_email(job, _read_words_for_job(job), consent):
                        write_review(job, emailed_at=int(time.time()))
        except Exception as e:  # noqa: BLE001
            _log(f"scanner loop error: {type(e).__name__}: {e}")
        time.sleep(60)


def start_scanner() -> None:
    threading.Thread(target=_scanner_loop, daemon=True).start()


# --- HTML chrome ----------------------------------------------------------

def _admin_chrome(title: str, body: str) -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · Typortrait admin</title><link rel="stylesheet" href="/static/inter.css"><style>
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#faf9f7;color:#16203a;padding:24px}}
.wrap{{max-width:1100px;margin:0 auto}}h1{{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0d1b3a;font-size:24px;margin:0 0 16px}}h2{{font-size:18px;margin:24px 0 8px}}
nav{{display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap;align-items:center}}nav a{{color:#0d1b3a;text-decoration:none;padding:6px 12px;border:1px solid #ece9e3;border-radius:999px;font-size:13px;background:#fff}}
nav a.cur{{background:#0d1b3a;color:#fff;border-color:#0d1b3a}}nav .gap{{flex:1}}nav form{{display:inline}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #ece9e3;border-radius:10px;overflow:hidden}}
th,td{{padding:10px 12px;text-align:left;font-size:13px;border-bottom:1px solid #ece9e3;vertical-align:top}}th{{background:#f5f3ee;color:#16203a;font-weight:600}}
tr:last-child td{{border-bottom:0}}
.state{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}}
.s-queued{{background:#fef3c7;color:#92400e}}.s-approved{{background:#dbeafe;color:#1e40af}}.s-posted{{background:#d1fae5;color:#065f46}}.s-rejected{{background:#fee2e2;color:#991b1b}}.s-revoked{{background:#f3e8ff;color:#6b21a8}}
.btn{{display:inline-block;padding:8px 16px;border-radius:999px;font-size:13px;font-weight:600;text-decoration:none;border:none;cursor:pointer;margin:2px 4px 2px 0;background:#0d1b3a;color:#fff;font-family:inherit}}
.btn.ghost{{background:#fff;color:#0d1b3a;border:1.5px solid #0d1b3a}}.btn.danger{{background:#991b1b;color:#fff}}.btn.warn{{background:#92400e;color:#fff}}
.card{{background:#fff;border:1px solid #ece9e3;border-radius:12px;padding:18px;margin:12px 0}}
.field{{display:block;margin:8px 0}}.field label{{display:block;font-size:12px;color:#6b7280;margin-bottom:4px}}.field input,.field textarea{{width:100%;padding:8px 10px;border:1px solid #ece9e3;border-radius:8px;font-family:inherit;font-size:14px}}
pre{{background:#f5f3ee;padding:12px;border-radius:8px;font-size:12px;overflow:auto}}
video{{max-width:100%;max-height:60vh;border-radius:10px;background:#000;display:block;margin:0 auto}}
.msg{{padding:10px 14px;border-radius:8px;margin-bottom:14px;font-size:14px}}.msg.ok{{background:#d1fae5;color:#065f46}}.msg.err{{background:#fee2e2;color:#991b1b}}
img.thumb{{max-width:80px;border-radius:6px;display:block}}
a{{color:#0d1b3a}}
.kpis{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 18px}}
.kpi{{background:#fff;border:1px solid #ece9e3;border-radius:10px;padding:10px 14px;min-width:88px;flex:1}}
.kpi-v{{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:22px;color:#0d1b3a;line-height:1.1}}
.kpi-v a{{text-decoration:none}}
.kpi-l{{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em;margin-top:3px}}
.bar{{display:inline-block;height:10px;background:#0d1b3a;border-radius:4px;vertical-align:middle;margin-right:6px}}
.row-num{{text-align:right;font-variant-numeric:tabular-nums;color:#6b7280}}
.funnel-step{{display:flex;align-items:baseline;gap:14px;padding:8px 0;border-top:1px solid #f1eee8}}
.funnel-step:first-child{{border-top:0}}
.funnel-num{{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:20px;color:#0d1b3a;min-width:70px}}
.funnel-pct{{color:#6b7280;font-size:12px;min-width:80px}}
.muted{{color:#6b7280;font-size:12px}}
</style></head><body><div class="wrap">{body}</div></body></html>"""


def _admin_nav(current: str = "all", stats_active: bool = False,
               orders_active: bool = False, dr_active: bool = False,
               issue_active: bool = False, studio_active: bool = False) -> str:
    items = [
        ("all", "All", "/admin/reels"),
        ("queued", "Queued", "/admin/reels?filter=queued"),
        ("approved", "Approved", "/admin/reels?filter=approved"),
        ("posted", "Posted", "/admin/reels?filter=posted"),
        ("rejected", "Rejected", "/admin/reels?filter=rejected"),
        ("revoked", "Revoked", "/admin/reels?filter=revoked"),
    ]
    active_elsewhere = (stats_active or orders_active or dr_active or issue_active
                        or studio_active)
    bits = []
    for key, label, url in items:
        cls = ' class="cur"' if key == current and not active_elsewhere else ""
        bits.append(f'<a href="{url}"{cls}>{html.escape(label)}</a>')
    stats_cls = ' class="cur"' if stats_active else ""
    bits.append(f'<a href="/admin/stats"{stats_cls}>Stats</a>')
    orders_cls = ' class="cur"' if orders_active else ""
    bits.append(f'<a href="/admin/orders"{orders_cls}>Orders</a>')
    issue_cls = ' class="cur"' if issue_active else ""
    bits.append(f'<a href="/admin/issue"{issue_cls}>Issue code</a>')
    studio_cls = ' class="cur"' if studio_active else ""
    bits.append(f'<a href="/admin/studio"{studio_cls}>Studio</a>')
    dr_cls = ' class="cur"' if dr_active else ""
    bits.append(f'<a href="/admin/data-requests"{dr_cls}>Data requests</a>')
    bits.append('<span class="gap"></span>')
    bits.append('<form method="post" action="/admin/logout"><button class="btn ghost" type="submit">Log out</button></form>')
    return f'<nav>{"".join(bits)}</nav>'


def _kpi_strip(stats: dict) -> str:
    """Compact stats summary rendered above the reel list."""
    s = stats
    def k(label, value, link=None):
        v = f'<a href="{link}">{value}</a>' if link else str(value)
        return (f'<div class="kpi"><div class="kpi-v">{v}</div>'
                f'<div class="kpi-l">{html.escape(label)}</div></div>')
    items = [
        k("Renders", s["total_renders"]),
        k("Paid", s["paid_orders"]),
        k("Reels made", s["reels_created"]),
        k("Saved (exit)", s.get("saves_captured", 0)),
        k("Shared w/ us", s["typortrait_consents"]),
        k("Queued", s["states"]["queued"], "/admin/reels?filter=queued"),
        k("Approved", s["states"]["approved"], "/admin/reels?filter=approved"),
        k("Posted", s["states"]["posted"], "/admin/reels?filter=posted"),
    ]
    return f'<div class="kpis">{"".join(items)}</div>'


# --- Endpoints ------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
def admin_root():
    return RedirectResponse(url="/admin/reels", status_code=302)


@router.get("/login", response_class=HTMLResponse)
def admin_login_form(msg: str = ""):
    body = '<h1>Typortrait admin</h1>'
    if msg == "bad":
        body += '<div class="msg err">Wrong password.</div>'
    if not ADMIN_PASSWORD or not SECRET_KEY:
        body += ('<div class="msg err">Admin is not configured. Set '
                 '<code>TYPO_ADMIN_PASSWORD</code> and <code>TYPO_SECRET_KEY</code> '
                 'in <code>.env</code> and restart the container.</div>')
    body += ('<form method="post" action="/admin/login" class="card" style="max-width:380px">'
             '<div class="field"><label>Password</label>'
             '<input name="password" type="password" autofocus></div>'
             '<button class="btn" type="submit">Sign in</button></form>')
    return HTMLResponse(_admin_chrome("Sign in", body))


@router.post("/login")
def admin_login(password: str = Form(...)):
    if not ADMIN_PASSWORD or not SECRET_KEY:
        return HTMLResponse(_admin_chrome("Admin not configured",
            "<h1>Admin not configured</h1><p>Set <code>TYPO_ADMIN_PASSWORD</code> and "
            "<code>TYPO_SECRET_KEY</code> in <code>.env</code> and restart.</p>"),
            status_code=503)
    if not hmac.compare_digest(password, ADMIN_PASSWORD):
        return RedirectResponse(url="/admin/login?msg=bad", status_code=302)
    r = RedirectResponse(url="/admin/reels", status_code=302)
    r.set_cookie("admin_session", make_session_cookie(),
                 httponly=True, secure=COOKIE_SECURE, samesite="lax",
                 max_age=30 * 86400)
    return r


@router.post("/logout")
def admin_logout():
    r = RedirectResponse(url="/admin/login", status_code=302)
    r.delete_cookie("admin_session")
    return r


# Pill colors for the print-order lifecycle, reusing the .state chip style.
_ORDER_STATE_STYLE = {
    "pending_payment": "background:#fef3c7;color:#92400e",
    "paid":            "background:#dbeafe;color:#1e40af",
    "fulfilling":      "background:#e0e7ff;color:#3730a3",
    "shipped":         "background:#d1fae5;color:#065f46",
    "delivered":       "background:#d1fae5;color:#065f46",
    "error":           "background:#fee2e2;color:#991b1b",
}


@router.get("/orders", response_class=HTMLResponse)
def admin_orders(admin_session: Optional[str] = Cookie(None)):
    """Print-on-demand order dashboard. Shares the reel admin's cookie auth and
    chrome; lists recent physical/digital orders with their fulfillment state."""
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    # Imported here (not at module load) so the reel admin keeps working even if
    # the orders/catalog modules are absent on an older deploy.
    from . import orders as orders_db, products

    rows = orders_db.list_recent(200)
    body = '<h1>Print orders</h1>'
    body += _admin_nav(orders_active=True)
    if not rows:
        body += ('<div class="card"><p class="muted">No orders yet. Physical orders '
                 '(framed prints, canvas, posters, shirts) appear here once a customer '
                 'completes checkout.</p></div>')
        return HTMLResponse(_admin_chrome("Orders", body))

    trs = []
    for o in rows:
        product = products.get(o["sku"])
        name = product.name if product else o["sku"]
        item = html.escape(name) + (f' / {html.escape(o["size"])}' if o.get("size") else '')
        total = (o["price_cents"] + o["shipping_cents"]) / 100
        cur = html.escape((o.get("currency") or "usd").upper())
        status = o["status"]
        pill = (f'<span class="state" style="{_ORDER_STATE_STYLE.get(status, "")}">'
                f'{html.escape(status.replace("_", " "))}</span>')
        pf_id = o.get("printful_order_id") or "—"
        track = o.get("tracking_url")
        track_html = (f'<a href="{html.escape(track)}" target="_blank" rel="noopener">track</a>'
                      if track else "—")
        email = html.escape(o.get("customer_email") or "—")
        err = html.escape(o.get("error_message") or "")
        ref_tag = html.escape(o.get("ref") or "—")
        trs.append(
            f'<tr><td><a href="/order/{html.escape(o["id"])}">{html.escape(o["id"])}</a></td>'
            f'<td>{ref_tag}</td>'
            f'<td>{item}</td><td>${total:.2f} {cur}</td>'
            f'<td>{pill}</td><td>{email}</td><td>{pf_id}</td>'
            f'<td>{track_html}</td><td class="muted">{err}</td></tr>'
        )
    body += ('<table><tr><th>Order</th><th>Source</th><th>Item</th><th>Total</th><th>Status</th>'
             '<th>Email</th><th>Printful</th><th>Track</th><th>Error</th></tr>'
             + "".join(trs) + '</table>')
    return HTMLResponse(_admin_chrome("Orders", body))


@router.get("/issue", response_class=HTMLResponse)
def admin_issue_form(admin_session: Optional[str] = Cookie(None)):
    """Point-and-click equivalent of tools/issue_code.py: pick a product, enter the
    buyer's email, and issue + email a one-time redemption code."""
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    from . import products
    # EverLoved's three sellers first, then the rest of the catalog by name.
    front = ["digital", "framed_16x20", "bundle_family"]
    cat = sorted(products.CATALOG,
                 key=lambda p: (front.index(p.sku) if p.sku in front else len(front), p.name))
    opts = "".join(
        f'<option value="{html.escape(p.sku)}">{html.escape(p.name)} ({html.escape(p.sku)})</option>'
        for p in cat)
    default_batch = "everloved-" + time.strftime("%Y-%m")
    body = '<h1>Issue a redemption code</h1>'
    body += _admin_nav(issue_active=True)
    body += ('<p class="muted">From an EverLoved order: choose the product, enter the buyer&rsquo;s '
             'email, and click the button. We generate a one-time code and email the buyer their '
             'personalize link &mdash; no command line needed.</p>')
    body += (f'<form method="post" action="/admin/issue" class="card" style="max-width:540px">'
             f'<div class="field"><label>Product</label>'
             f'<select name="sku">{opts}</select></div>'
             f'<div class="field"><label>Buyer email</label>'
             f'<input name="email" type="email" required placeholder="buyer@example.com"></div>'
             f'<div class="field"><label>Buyer first name (optional)</label>'
             f'<input name="name" placeholder="Jane"></div>'
             f'<div class="field"><label>Batch label (optional)</label>'
             f'<input name="batch" value="{html.escape(default_batch)}"></div>'
             f'<button class="btn" type="submit">Issue &amp; email code</button></form>')
    return HTMLResponse(_admin_chrome("Issue code", body))


@router.post("/issue", response_class=HTMLResponse)
def admin_issue_submit(sku: str = Form(...), email: str = Form(...),
                       name: str = Form(""), batch: str = Form(""),
                       admin_session: Optional[str] = Cookie(None)):
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    result = issue_redemption_code(sku, email, name=name, batch=(batch.strip() or None))
    body = '<h1>Issue a redemption code</h1>'
    body += _admin_nav(issue_active=True)
    if not result.get("ok"):
        body += f'<div class="msg err">{html.escape(result.get("error") or "Could not issue the code.")}</div>'
        body += '<p><a class="btn ghost" href="/admin/issue">Try again</a></p>'
        return HTMLResponse(_admin_chrome("Issue code", body))
    code = html.escape(result["code"])
    link = html.escape(result["link"])
    prod = html.escape(result["product"])
    to = html.escape(email.strip())
    if result["emailed"]:
        body += f'<div class="msg ok">Code issued and emailed to <b>{to}</b>.</div>'
    else:
        body += ('<div class="msg err">Code issued, but the email failed to send (check SMTP). '
                 'Copy the code and link below and send them to the buyer manually.</div>')
    body += (f'<div class="card"><p><b>Product:</b> {prod}</p>'
             f'<p><b>Code:</b> <span style="font-size:20px;letter-spacing:.06em;font-weight:700">{code}</span></p>'
             f'<p><b>Redeem link:</b><br><a href="{link}">{link}</a></p></div>')
    body += ('<p><a class="btn" href="/admin/issue">Issue another</a> '
             '&nbsp;<a class="btn ghost" href="/admin/orders">View orders</a></p>')
    return HTMLResponse(_admin_chrome("Issue code", body))


_DR_TYPE_STYLE = {
    "delete":       "background:#fee2e2;color:#991b1b",
    "access":       "background:#dbeafe;color:#1e40af",
    "correct":      "background:#fef3c7;color:#92400e",
    "object":       "background:#f3e8ff;color:#6b21a8",
    "takedown":     "background:#fee2e2;color:#991b1b",
    "report_abuse": "background:#fee2e2;color:#991b1b",
}


@router.get("/data-requests", response_class=HTMLResponse)
def admin_data_requests(admin_session: Optional[str] = Cookie(None)):
    """GDPR/CCPA data requests submitted via /data-request, newest first. Read-only
    audit view over data/data_requests.log. A delete WITH a job ID was already purged
    at submit time; everything else is a to-do to action within the statutory window
    (GDPR ~30 days, CCPA ~45) -- and verify the requester's identity before disclosing
    or changing anything. No mutation here on purpose: deletions stay a deliberate,
    manual step (see COMPLIANCE.md)."""
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    body = '<h1>Data requests</h1>'
    body += _admin_nav(dr_active=True)

    recs = []
    try:
        with (DATA_DIR / "data_requests.log").open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except (ValueError, TypeError):
                    continue
    except OSError:
        pass
    recs.reverse()   # newest first

    if not recs:
        body += ('<div class="card"><p class="muted">No data requests yet. Submissions to the '
                 '<a href="/data-request">data request page</a> appear here (and email '
                 '<code>TYPO_ADMIN_EMAIL</code>). A delete with a job ID is purged automatically; '
                 'everything else is a to-do — verify identity, action within ~30 days, and reply.'
                 '</p></div>')
        return HTMLResponse(_admin_chrome("Data requests", body))

    now = int(time.time())
    def _auto_done(r):
        return r.get("type") == "delete" and bool(r.get("job_id"))
    todo = sum(1 for r in recs if not _auto_done(r))
    body += (f'<p class="muted">{len(recs)} total · {todo} need action · respond within '
             '~30 days (GDPR) / ~45 days (CCPA). <b>Verify the requester&rsquo;s identity</b> '
             'before disclosing or changing anything. Read-only — purge a job from the shell '
             '(see COMPLIANCE.md).</p>')

    trs = []
    for r in recs:
        ts = int(r.get("ts") or 0)
        when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts)) if ts else "—"
        age = (now - ts) / 86400.0 if ts else 0.0
        rtype = str(r.get("type") or "—")
        pill = (f'<span class="state" style="{_DR_TYPE_STYLE.get(rtype, "background:#eee;color:#333")}">'
                f'{html.escape(rtype)}</span>')
        email = html.escape(str(r.get("email") or "—"))
        jid = str(r.get("job_id") or "").strip()
        deleted = int(r.get("deleted_files") or 0)
        job_html = (f'<code>{html.escape(jid)}</code> · {deleted} purged'
                    if jid else '<span class="muted">—</span>')
        region = html.escape(str(r.get("region") or "—"))
        details = html.escape(str(r.get("details") or ""))
        if _auto_done(r):
            status = '<span class="state s-posted">auto-purged</span>'
        elif age > 30:
            status = '<span class="state s-rejected">overdue</span>'
        elif age > 23:
            status = '<span class="state s-queued">due soon</span>'
        else:
            status = '<span class="state s-approved">to do</span>'
        trs.append(
            f'<tr><td class="muted">{when}</td><td>{pill}</td><td>{email}</td>'
            f'<td>{job_html}</td><td>{region}</td><td>{status}</td>'
            f'<td class="muted">{details}</td></tr>')
    body += ('<table><tr><th>When</th><th>Type</th><th>Email</th><th>Job&nbsp;ID</th>'
             '<th>Region</th><th>Status</th><th>Details</th></tr>'
             + "".join(trs) + '</table>')
    return HTMLResponse(_admin_chrome("Data requests", body))


@router.get("/reels", response_class=HTMLResponse)
def admin_list(filter: str = "all", msg: str = "",
               admin_session: Optional[str] = Cookie(None)):
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    flt = filter if filter in REEL_STATES else "all"
    reviews = list_reviews()
    if flt != "all":
        reviews = [r for r in reviews if r.get("state") == flt]
    rows = []
    for r in reviews:
        job = r.get("job", "")
        st = r.get("state", "queued")
        ts = r.get(f"ts_{st}") or r.get("ts_queued") or 0
        when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)) if ts else ""
        je = html.escape(job)
        thumb = (f'<a href="/admin/reels/{je}">'
                 f'<img class="thumb" src="/outputs/{je}_before.jpg" '
                 f'onerror="this.style.display=\'none\'"></a>')
        actions = (f'<a class="btn ghost" href="/admin/reels/{je}">Open</a>'
                   f'<form method="post" action="/admin/reels/{je}/delete" style="display:inline" '
                   f'onsubmit="return confirm(\'Permanently delete this reel and ALL its records? This cannot be undone.\')">'
                   f'<button class="btn danger" type="submit">Delete</button></form>')
        rows.append(
            f'<tr><td>{thumb}</td>'
            f'<td><a href="/admin/reels/{je}">{je}</a></td>'
            f'<td>{when}</td>'
            f'<td><span class="state s-{st}">{st}</span></td>'
            f'<td>{actions}</td></tr>'
        )
    body = '<h1>Reels for review</h1>'
    body += _admin_nav(current=flt)
    if msg == "deleted":
        body += '<div class="msg ok">Reel permanently deleted — files and records removed.</div>'
    elif msg == "deleted_kept":
        body += ('<div class="msg ok">Reel deleted — video and review record removed. '
                 'The consent record was <b>kept</b> as proof of authorization '
                 '(this reel had been posted).</div>')
    try:
        body += _kpi_strip(compute_stats())
    except Exception as e:  # noqa: BLE001
        _log(f"kpi_strip failed: {e}")
    if not smtp_configured():
        body += ('<div class="msg err">SMTP is not fully configured — the '
                 'background mailer is paused. Queued reels still show here.</div>')
    if not rows:
        body += '<p class="card">No reels in this view.</p>'
    else:
        body += ('<table><thead><tr><th></th><th>Job</th><th>When</th>'
                 '<th>State</th><th></th></tr></thead><tbody>'
                 + "".join(rows) + '</tbody></table>')
    return HTMLResponse(_admin_chrome("Reels", body))


@router.get("/stats", response_class=HTMLResponse)
def admin_stats(window: str = "30", admin_session: Optional[str] = Cookie(None)):
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    s = compute_stats()
    body = '<h1>Stats &amp; signals</h1>'
    body += _admin_nav(stats_active=True)
    body += _kpi_strip(s)

    # --- Funnel -----------------------------------------------------------
    renders = s["total_renders"]
    paid = s["paid_orders"]
    reels = s["reels_created"]
    consents = s["typortrait_consents"]
    approved = s["states"]["approved"] + s["states"]["posted"] + s["states"]["revoked"]
    posted = s["states"]["posted"] + s["states"]["revoked"]
    funnel = [
        ("Renders",          renders,  None,     "Visitors who generated a watermarked preview."),
        ("Paid orders",      paid,     renders,  "Conversion: paid downloads / renders."),
        ("Reels made",       reels,    paid,     "Of paid buyers, how many built a shareable reel."),
        ("Shared with us",   consents, reels,    "Of reel makers, how many opted in to let us feature it."),
        ("Approved",         approved, consents, "Of opt-ins, how many you green-lit (approved + posted + revoked)."),
        ("Posted",           posted,   approved, "Of approved, how many you actually posted (or posted then revoked)."),
    ]
    body += '<div class="card"><h2>Funnel</h2>'
    for label, num, den, hint in funnel:
        body += (f'<div class="funnel-step">'
                 f'<span class="funnel-num">{num}</span>'
                 f'<span class="funnel-pct">{_pct(num, den) if den is not None else "—"}</span>'
                 f'<span><b>{html.escape(label)}</b><br>'
                 f'<span class="muted">{html.escape(hint)}</span></span></div>')
    body += f'<p class="muted" style="margin-top:14px">Window: last {s["retention_days"]} days (file retention).</p>'
    body += '</div>'

    # --- Referral funnel (renders -> sales -> revenue, per ?ref source) ---
    _WINDOWS = [("7", "7 days"), ("30", "30 days"), ("90", "90 days"),
                ("365", "1 year"), ("all", "All time")]
    sel = window if window in dict(_WINDOWS) else "30"
    days = None if sel == "all" else int(sel)
    since_ts = 0 if days is None else int(time.time()) - days * 86400
    fsrc = referral_funnel(since_ts)
    body += '<div class="card"><h2>Referral funnel</h2>'
    picks = [(f'<b>{html.escape(lbl)}</b>' if val == sel
              else f'<a href="/admin/stats?window={val}">{html.escape(lbl)}</a>')
             for val, lbl in _WINDOWS]
    body += '<p class="muted" style="margin-bottom:12px">Range: ' + " &middot; ".join(picks) + '</p>'
    if not fsrc:
        body += '<p class="muted">No activity in this range yet.</p>'
    else:
        def _rk(item):
            name, agg = item
            return (name == "direct", -agg["sales"], -agg["renders"])
        rows = []
        tot = {"renders": 0, "sales": 0, "revenue_cents": 0}
        for name, agg in sorted(fsrc.items(), key=_rk):
            for kk in tot:
                tot[kk] += agg[kk]
            partner = name != "direct"
            label = "Direct (no referral)" if name == "direct" else name
            disp = f"<b>{html.escape(label)}</b>" if partner else html.escape(label)
            conv = f'{100 * agg["sales"] / agg["renders"]:.0f}%' if agg["renders"] else "—"
            rows.append(f'<tr><td>{disp}</td><td class="row-num">{agg["renders"]}</td>'
                        f'<td class="row-num">{agg["sales"]}</td>'
                        f'<td class="row-num">{conv}</td>'
                        f'<td class="row-num">${agg["revenue_cents"] / 100:,.2f}</td></tr>')
        tconv = f'{100 * tot["sales"] / tot["renders"]:.0f}%' if tot["renders"] else "—"
        body += ('<table><thead><tr><th>Source</th><th class="row-num">Renders</th>'
                 '<th class="row-num">Sales</th><th class="row-num">Conv.</th>'
                 '<th class="row-num">Revenue</th></tr></thead><tbody>'
                 + "".join(rows)
                 + f'<tr><td><b>Total</b></td><td class="row-num"><b>{tot["renders"]}</b></td>'
                   f'<td class="row-num"><b>{tot["sales"]}</b></td>'
                   f'<td class="row-num"><b>{tconv}</b></td>'
                   f'<td class="row-num"><b>${tot["revenue_cents"] / 100:,.2f}</b></td></tr>'
                 '</tbody></table>')
    body += (f'<p class="muted" style="margin-top:10px">Source = the <code>?ref</code> tag captured at '
             f'render. Renders are limited to the {s["retention_days"]}-day retention window; sales + '
             f'revenue are recorded all-time. Conv. = sales &divide; renders.</p>')
    body += '</div>'

    # --- Choice popularity (style + ink + word count + toggles) -----------
    def _bar_table(title: str, counter: dict, total: int):
        if not counter:
            return f'<div class="card"><h2>{html.escape(title)}</h2><p class="muted">No data yet.</p></div>'
        rows = []
        items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        top = items[0][1] if items else 1
        for k, v in items:
            width = int(180 * (v / top)) if top else 0
            pct = (100 * v / total) if total else 0
            rows.append(
                f'<tr><td>{html.escape(str(k))}</td>'
                f'<td class="row-num">{v}</td>'
                f'<td class="row-num">{pct:.1f}%</td>'
                f'<td><span class="bar" style="width:{width}px"></span></td></tr>'
            )
        return (f'<div class="card"><h2>{html.escape(title)}</h2>'
                f'<table style="background:transparent;border:0">'
                f'<tbody>' + "".join(rows) + '</tbody></table></div>')

    body += _bar_table("Ink color popularity", s["inks"], renders)
    body += _bar_table("Style popularity", s["styles"], renders)
    body += _bar_table("Word-count distribution", s["word_count_buckets"], renders)

    # --- Render-option toggles -------------------------------------------
    toggles_total = renders or 1
    body += ('<div class="card"><h2>Render options</h2>'
             f'<p>Uppercase enabled: <b>{s["uppercase_count"]}</b> ({100*s["uppercase_count"]/toggles_total:.0f}%)</p>'
             f'<p>Background removed: <b>{s["remove_bg_count"]}</b> ({100*s["remove_bg_count"]/toggles_total:.0f}%)</p>'
             f'<p>Light mode: <b>{s["light_count"]}</b> ({100*s["light_count"]/toggles_total:.0f}%)</p>'
             '</div>')

    # --- Daily timeline ---------------------------------------------------
    body += '<div class="card"><h2>Last {0} days</h2>'.format(s["retention_days"])
    body += ('<table><thead><tr><th>Date</th><th class="row-num">Renders</th>'
             '<th class="row-num">Paid</th><th class="row-num">Reels</th>'
             '<th class="row-num">Shared w/ us</th></tr></thead><tbody>')
    for d, row in s["by_day"]:
        if row["renders"] == 0 and row["paid"] == 0 and row["reels"] == 0 and row["consents"] == 0:
            continue
        body += (f'<tr><td>{d}</td>'
                 f'<td class="row-num">{row["renders"]}</td>'
                 f'<td class="row-num">{row["paid"]}</td>'
                 f'<td class="row-num">{row["reels"]}</td>'
                 f'<td class="row-num">{row["consents"]}</td></tr>')
    body += '</tbody></table>'
    body += f'<p class="muted" style="margin-top:14px">Counts are bucketed by file modification time. Days with no activity hidden.</p>'
    body += '</div>'

    # --- Lifecycle state distribution -------------------------------------
    body += '<div class="card"><h2>Review states</h2>'
    body += '<table style="background:transparent;border:0"><tbody>'
    for st in REEL_STATES:
        n = s["states"][st]
        body += (f'<tr><td><span class="state s-{st}">{st}</span></td>'
                 f'<td class="row-num">{n}</td></tr>')
    body += '</tbody></table></div>'

    return HTMLResponse(_admin_chrome("Stats", body))


@router.get("/reels/{job}", response_class=HTMLResponse)
def admin_detail(job: str, msg: str = "",
                 admin_session: Optional[str] = Cookie(None)):
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    rec = read_review(job)
    if not rec:
        return HTMLResponse(_admin_chrome("Not found",
            f'<h1>Not found</h1><p>No review record for '
            f'<code>{html.escape(job)}</code>.</p>'
            f'<p><a class="btn ghost" href="/admin/reels">Back to list</a></p>'),
            status_code=404)
    consent = {}
    if consent_path(job).exists():
        try:
            consent = json.loads(consent_path(job).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    recipe = {}
    rp = PRIVATE_DIR / f"{job}.json"
    if rp.exists():
        try:
            recipe = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    st = rec.get("state", "queued")
    je = html.escape(job)
    body = f'<h1>Reel · {je}</h1>'
    body += _admin_nav()
    if msg == "approved":
        body += '<div class="msg ok">Marked as approved.</div>'
    elif msg == "rejected":
        body += '<div class="msg ok">Rejected — reel files purged. Consent record retained.</div>'
    elif msg == "posted":
        body += '<div class="msg ok">Marked as posted.</div>'
    elif msg == "revoked":
        body += '<div class="msg ok">Revoked — reel files purged. Posted record retained for audit.</div>'
    elif msg == "resent":
        body += '<div class="msg ok">Re-queued — the background scanner will email within ~60 seconds.</div>'
    body += f'<div class="card"><span class="state s-{st}">{st}</span></div>'
    body += (f'<div class="card"><h2>Preview</h2>'
             f'<video src="/outputs/{je}_reel.mp4" controls playsinline></video></div>')
    body += '<div class="card"><h2>Actions</h2>'
    if st == "queued":
        body += (f'<form method="post" action="/admin/reels/{je}/approve" style="display:inline">'
                 f'<button class="btn" type="submit">Approve</button></form>'
                 f'<form method="post" action="/admin/reels/{je}/reject" style="display:inline" '
                 f'onsubmit="return confirm(\'Reject this reel and delete its files?\')">'
                 f'<button class="btn danger" type="submit">Reject</button></form>')
    elif st == "approved":
        body += (f'<form method="post" action="/admin/reels/{je}/posted">'
                 f'<div class="field"><label>Posted URLs (one per line: Instagram, TikTok, etc.)</label>'
                 f'<textarea name="links" rows="3" placeholder="https://www.instagram.com/p/...&#10;https://www.tiktok.com/@..."></textarea></div>'
                 f'<button class="btn" type="submit">Mark as Posted</button></form>'
                 f'<form method="post" action="/admin/reels/{je}/reject" style="display:inline;margin-top:8px" '
                 f'onsubmit="return confirm(\'Reject this reel and delete its files?\')">'
                 f'<button class="btn danger" type="submit">Reject instead</button></form>'
                 f'<form method="post" action="/admin/reels/{je}/revoke" style="display:inline" '
                 f'onsubmit="return confirm(\'Revoke and purge files? Use this if the user requested removal before posting.\')">'
                 f'<button class="btn warn" type="submit">Revoke (user request)</button></form>')
    elif st == "posted":
        links = rec.get("posted_links") or []
        if links:
            body += ('<p><b>Posted to:</b><br>'
                     + "<br>".join(f'<a href="{html.escape(u)}" target="_blank" rel="noopener">'
                                   f'{html.escape(u)}</a>' for u in links)
                     + '</p>')
        body += (f'<form method="post" action="/admin/reels/{je}/revoke" '
                 f'onsubmit="return confirm(\'Revoke and purge files? Use this only after you have removed the post from your social channels.\')">'
                 f'<button class="btn warn" type="submit">Revoke (user requested removal)</button></form>')
    else:
        body += f'<p>No further actions available for state <b>{html.escape(st)}</b>.</p>'
    # Re-queue / re-send is available from any state so you can re-test the
    # email path or re-notify yourself without going through another checkout.
    body += (f'<form method="post" action="/admin/reels/{je}/resend" style="display:inline;margin-top:8px" '
             f'onsubmit="return confirm(\'Re-queue this reel and trigger another notification email?\')">'
             f'<button class="btn ghost" type="submit">Re-send notification email</button></form>')
    body += '</div>'
    # Permanent removal — separated into its own card so it reads as the
    # destructive, irreversible action it is. Compliance guard: posted reels
    # keep their consent record (proof the likeness use was authorized).
    posted_likeness = st == "posted" or bool(rec.get("ts_posted"))
    if posted_likeness:
        danger_note = ('Removes the video and review record so this reel leaves the '
                       'dashboard. Because it was posted, the <b>consent record is kept</b> '
                       'as proof you were authorized to use this likeness. This cannot be undone.')
    else:
        danger_note = ('Permanently removes this reel — deletes the video, consent record, '
                       'and review record. This cannot be undone and leaves no audit trail.')
    body += ('<div class="card"><h2>Danger zone</h2>'
             f'<p class="muted">{danger_note}</p>'
             f'<form method="post" action="/admin/reels/{je}/delete" style="display:inline" '
             f'onsubmit="return confirm(\'Permanently delete this reel? This cannot be undone.\')">'
             f'<button class="btn danger" type="submit">Delete permanently</button></form></div>')
    body += ('<div class="card"><h2>Consent record</h2><pre>'
             + html.escape(json.dumps(consent, indent=2)) + '</pre></div>')
    body += ('<div class="card"><h2>Recipe</h2><pre>'
             + html.escape(json.dumps(recipe, indent=2)) + '</pre></div>')
    body += ('<div class="card"><h2>Review record</h2><pre>'
             + html.escape(json.dumps(rec, indent=2)) + '</pre></div>')
    return HTMLResponse(_admin_chrome(f"Reel {job}", body))


@router.post("/reels/{job}/approve")
def admin_approve(job: str, admin_session: Optional[str] = Cookie(None)):
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    if not read_review(job):
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    transition(job, "approved")
    return RedirectResponse(url=f"/admin/reels/{job}?msg=approved", status_code=302)


@router.post("/reels/{job}/reject")
def admin_reject(job: str, admin_session: Optional[str] = Cookie(None)):
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    if not read_review(job):
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    transition(job, "rejected")
    return RedirectResponse(url=f"/admin/reels/{job}?msg=rejected", status_code=302)


@router.post("/reels/{job}/posted")
def admin_posted(job: str, links: str = Form(""),
                 admin_session: Optional[str] = Cookie(None)):
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    if not read_review(job):
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    url_list = [u.strip() for u in (links or "").splitlines() if u.strip()]
    transition(job, "posted", posted_links=url_list)
    return RedirectResponse(url=f"/admin/reels/{job}?msg=posted", status_code=302)


@router.post("/reels/{job}/revoke")
def admin_revoke(job: str, admin_session: Optional[str] = Cookie(None)):
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    if not read_review(job):
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    transition(job, "revoked")
    return RedirectResponse(url=f"/admin/reels/{job}?msg=revoked", status_code=302)


@router.post("/reels/{job}/resend")
def admin_resend(job: str, admin_session: Optional[str] = Cookie(None)):
    """Reset state to queued + clear emailed_at so the background scanner
    re-emails the notification. Useful for testing SMTP and for re-notifying
    yourself if you missed an email. Does not delete reel files."""
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    if not read_review(job):
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    write_review(job, state="queued", emailed_at=None)
    return RedirectResponse(url=f"/admin/reels/{job}?msg=resent", status_code=302)


@router.post("/reels/{job}/delete")
def admin_delete(job: str, admin_session: Optional[str] = Cookie(None)):
    """Delete a reel from the dashboard: purge its files and review record.
    Compliance guard — if the reel was ever posted, the consent record is
    kept as proof of authorization (see delete_reel). Redirects to the list
    since the detail page no longer exists."""
    guard = _require_admin(admin_session)
    if guard is not None:
        return guard
    if not read_review(job):
        return JSONResponse({"ok": False, "error": "unknown_job"}, status_code=404)
    res = delete_reel(job)
    msg = "deleted_kept" if res["consent_kept"] else "deleted"
    return RedirectResponse(url=f"/admin/reels?msg={msg}", status_code=302)


@router.get("/email/{action}")
def admin_email_action(action: str, token: str):
    """One-tap action from the notification email. Token-authenticated, so
    it works from a phone without typing a password. State transitions are
    naturally idempotent — clicking the same link twice is a no-op."""
    res = verify_email_token(token)
    if not res:
        return HTMLResponse(_admin_chrome("Link expired",
            '<h1>Link expired or invalid</h1>'
            '<p>Open the <a href="/admin/reels">dashboard</a> to act on this reel.</p>'),
            status_code=400)
    job, tok_action = res
    if tok_action != action:
        return HTMLResponse(_admin_chrome("Mismatched action",
            '<h1>Link mismatched</h1>'
            '<p>Open the <a href="/admin/reels">dashboard</a>.</p>'),
            status_code=400)
    if not read_review(job):
        return HTMLResponse(_admin_chrome("Not found",
            f'<h1>Not found</h1><p>No reel record for <b>{html.escape(job)}</b>.</p>'),
            status_code=404)
    if action == "approve":
        transition(job, "approved")
        msg = "Approved — open the dashboard to mark it posted once it goes live."
    elif action == "reject":
        transition(job, "rejected")
        msg = "Rejected — reel files have been purged."
    else:
        return HTMLResponse(_admin_chrome("Bad action", "<h1>Bad action</h1>"),
                            status_code=400)
    body = (f'<h1>Done.</h1><p>{msg}</p>'
            f'<p><a class="btn" href="/admin/reels/{html.escape(job)}">Open reel</a> '
            f'<a class="btn ghost" href="/admin/reels">Back to dashboard</a></p>')
    return HTMLResponse(_admin_chrome("Done", body))
