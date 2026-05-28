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

from .config import OUTPUTS_DIR, PRIVATE_DIR, PUBLIC_BASE_URL


# --- Configuration --------------------------------------------------------
ADMIN_PASSWORD = os.environ.get("TYPO_ADMIN_PASSWORD", "")
ADMIN_EMAIL    = os.environ.get("TYPO_ADMIN_EMAIL", "")
SECRET_KEY     = os.environ.get("TYPO_SECRET_KEY", "")
SMTP_HOST      = os.environ.get("TYPO_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.environ.get("TYPO_SMTP_PORT", "587") or "587")
SMTP_USER      = os.environ.get("TYPO_SMTP_USER", "")
SMTP_PASS      = os.environ.get("TYPO_SMTP_PASS", "")
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

def smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS and ADMIN_EMAIL and SECRET_KEY)


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


def send_review_email(job: str, words, consent: dict) -> bool:
    if not smtp_configured():
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
    msg["From"] = SMTP_USER
    msg["To"] = ADMIN_EMAIL
    msg.set_content(body_text)
    msg.add_alternative(body_html, subtype="html")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True
    except Exception:  # noqa: BLE001
        return False


# --- Background scanner ---------------------------------------------------

def _scanner_loop() -> None:
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
        except Exception:  # noqa: BLE001
            pass
        time.sleep(60)


def start_scanner() -> None:
    threading.Thread(target=_scanner_loop, daemon=True).start()


# --- HTML chrome ----------------------------------------------------------

def _admin_chrome(title: str, body: str) -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · Typortrait admin</title><style>
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#faf9f7;color:#16203a;padding:24px}}
.wrap{{max-width:1100px;margin:0 auto}}h1{{font-family:Georgia,serif;color:#0d1b3a;font-size:24px;margin:0 0 16px}}h2{{font-size:18px;margin:24px 0 8px}}
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
video{{max-width:100%;border-radius:10px;background:#000;display:block}}
.msg{{padding:10px 14px;border-radius:8px;margin-bottom:14px;font-size:14px}}.msg.ok{{background:#d1fae5;color:#065f46}}.msg.err{{background:#fee2e2;color:#991b1b}}
img.thumb{{max-width:80px;border-radius:6px;display:block}}
a{{color:#0d1b3a}}
</style></head><body><div class="wrap">{body}</div></body></html>"""


def _admin_nav(current: str = "all") -> str:
    items = [
        ("all", "All", "/admin/reels"),
        ("queued", "Queued", "/admin/reels?filter=queued"),
        ("approved", "Approved", "/admin/reels?filter=approved"),
        ("posted", "Posted", "/admin/reels?filter=posted"),
        ("rejected", "Rejected", "/admin/reels?filter=rejected"),
        ("revoked", "Revoked", "/admin/reels?filter=revoked"),
    ]
    bits = []
    for key, label, url in items:
        cls = ' class="cur"' if key == current else ""
        bits.append(f'<a href="{url}"{cls}>{html.escape(label)}</a>')
    bits.append('<span class="gap"></span>')
    bits.append('<form method="post" action="/admin/logout"><button class="btn ghost" type="submit">Log out</button></form>')
    return f'<nav>{"".join(bits)}</nav>'


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


@router.get("/reels", response_class=HTMLResponse)
def admin_list(filter: str = "all",
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
        actions = f'<a class="btn ghost" href="/admin/reels/{je}">Open</a>'
        rows.append(
            f'<tr><td>{thumb}</td>'
            f'<td><a href="/admin/reels/{je}">{je}</a></td>'
            f'<td>{when}</td>'
            f'<td><span class="state s-{st}">{st}</span></td>'
            f'<td>{actions}</td></tr>'
        )
    body = '<h1>Reels for review</h1>'
    body += _admin_nav(current=flt)
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
    body += '</div>'
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
