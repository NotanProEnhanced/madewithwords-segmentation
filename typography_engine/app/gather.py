"""Gather — crowd-sourced words for a portrait (ADDITIVE, FLAG-GATED).

A shareable link where a whole circle adds a word/memory, and the portrait is
woven from everyone's words. Self-contained on purpose: its OWN SQLite file, its
OWN routes, its OWN pages. Nothing here touches the studio render/checkout/orders
paths, and every route is mounted only when ``GATHER_ENABLED`` is true — so this
module is completely inert in production unless deliberately switched on.

Timing model (see docs/gather-words-spec.md):
  seed (instant) -> live gather (throttled re-render) -> reveal (final edition).

Render throttle: words land in the list INSTANTLY (cheap DB write); the actual
portrait re-renders at most every GATHER_RENDER_INTERVAL seconds OR once
GATHER_RENDER_EVERY new words have arrived, and only one gather render runs at a
time (non-blocking — excess triggers are dropped and retried on the next poll),
so a busy gather can never hammer the single render slot the studio shares.
"""
from __future__ import annotations

import io
import re
import secrets
import sqlite3
import threading
import time
from typing import List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response

from .config import (
    GATHER_DB, GATHER_DIR, GATHER_MAX_WORDS, GATHER_RENDER_EVERY,
    GATHER_RENDER_INTERVAL, GATHER_WORD_MAXLEN, PUBLIC_BASE_URL, STATIC_DIR,
)

router = APIRouter()

# --- storage -----------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(GATHER_DB), timeout=15)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """Create the gather tables + storage dir. Idempotent; called at startup
    only when the feature flag is on."""
    GATHER_DIR.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS gathers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              share_token TEXT UNIQUE NOT NULL,
              admin_token TEXT UNIQUE NOT NULL,
              title TEXT NOT NULL,
              photo_path TEXT NOT NULL,
              style TEXT NOT NULL DEFAULT 'displacement',
              ink TEXT NOT NULL DEFAULT 'photo',
              ground TEXT NOT NULL DEFAULT 'paper',
              min_font REAL NOT NULL DEFAULT 20,
              status TEXT NOT NULL DEFAULT 'open',          -- 'open' | 'revealed'
              created_at REAL NOT NULL,
              close_at REAL,                                 -- optional soft deadline
              last_render_at REAL NOT NULL DEFAULT 0,
              words_at_render INTEGER NOT NULL DEFAULT -1,
              has_portrait INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS contributions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              gather_id INTEGER NOT NULL,
              text TEXT NOT NULL,
              name TEXT,
              created_at REAL NOT NULL,
              hidden INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS ix_contrib_gather ON contributions(gather_id);
            """
        )


def _gather_by(col: str, val: str) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(f"SELECT * FROM gathers WHERE {col}=?", (val,)).fetchone()


# --- word handling -----------------------------------------------------------

# Tiny abuse guard. Not a content-policy engine -- the moderate-before-show /
# admin-hide controls are the real safety net; this just blocks the obvious.
_BLOCK = re.compile(r"\b(fuck|shit|cunt|nigger|faggot|bitch|asshole|whore)\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{0,%d}" % (GATHER_WORD_MAXLEN - 1))


def _clean_text(raw: str) -> str:
    t = " ".join((raw or "").split())[:GATHER_WORD_MAXLEN].strip()
    return t


def _tokens_for_render(rows: List[sqlite3.Row]) -> List[str]:
    """Visible contributions -> the word list the portrait is woven from.
    Frequency-ranked (consensus rises to the top) and capped so the portrait
    stays legible; near-duplicates collapse by uppercase form."""
    freq: dict = {}
    order: dict = {}
    n = 0
    for r in rows:
        if r["hidden"]:
            continue
        for m in _WORD_RE.findall(r["text"] or ""):
            w = m.upper()
            if len(w) < 2:
                continue
            freq[w] = freq.get(w, 0) + 1
            if w not in order:
                order[w] = n
                n += 1
    # rank: most-said first, stable by first-seen for ties
    ranked = sorted(freq.keys(), key=lambda w: (-freq[w], order[w]))
    return ranked[:GATHER_MAX_WORDS]


# --- throttled render --------------------------------------------------------

_render_lock = threading.Lock()       # only ONE gather render at a time
_inflight: set = set()                 # gather ids currently rendering


def _portrait_path(gid: int):
    return GATHER_DIR / str(gid) / "portrait.png"


def _do_render(gid: int) -> None:
    """Render the current word set onto the stored photo and save portrait.png.
    Runs in a worker thread; respects the single-render lock."""
    if not _render_lock.acquire(blocking=False):
        _inflight.discard(gid)         # slot busy -> free it so the NEXT poll retries
        return                         # (else the gather stays stuck in _inflight forever)
    try:
        g = _gather_by("id", str(gid))
        if not g:
            return
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM contributions WHERE gather_id=? ORDER BY id", (gid,)
            ).fetchall()
        words = _tokens_for_render(rows)
        if not words:
            words = ["BELOVED"]        # never render an empty portrait
        try:
            raw = open(g["photo_path"], "rb").read()
        except OSError:
            return
        png = _render_words(raw, words, g)
        if png is None:
            return
        outp = _portrait_path(gid)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_bytes(png)
        with _conn() as c:
            c.execute(
                "UPDATE gathers SET last_render_at=?, words_at_render=?, has_portrait=1 WHERE id=?",
                (time.time(), _visible_count(c, gid), gid),
            )
            c.commit()
    finally:
        _inflight.discard(gid)
        _render_lock.release()


def _render_words(raw: bytes, words: List[str], g: sqlite3.Row) -> Optional[bytes]:
    """Reuse the studio render pipeline (same as gallery_render): a LIGHT preview
    render of the gathered words on the source photo."""
    from .config import RenderConfig
    from .pipeline.warnings import WarningCollector
    from .pipeline.analyze import analyze_image
    cfg = RenderConfig()
    cfg.min_font_px = float(g["min_font"] or 20)
    warns = WarningCollector()
    try:
        an = analyze_image(raw, cfg, warns)
    except Exception:
        return None
    style = g["style"] or "displacement"
    try:
        if style == "displacement":
            from .pipeline.displacement import render_displacement_portrait
            ground = g["ground"] if g["ground"] in ("paper", "navy", "black") else "paper"
            return render_displacement_portrait(
                an, words, ground=ground,
                ink=("photo" if g["ink"] in ("photo", "photo_paper", "custom") else g["ink"]),
                out_width=760, supersample=1, uppercase=True,
            )
        from .pipeline.tonal import render_layered_png
        text = " ".join(words)
        png, *_ = render_layered_png(
            an, text, ("message" if style == "message" else "words"), cfg, warns,
            ink=(g["ink"] or "mono"), remove_bg=True, light=False,
            out_width=760, render_w=1100,
        )
        return png
    except Exception:
        return None


def _visible_count(c: sqlite3.Connection, gid: int) -> int:
    return c.execute(
        "SELECT COUNT(*) FROM contributions WHERE gather_id=? AND hidden=0", (gid,)
    ).fetchone()[0]


def _maybe_render(g: sqlite3.Row, force: bool = False) -> None:
    """Decide whether to (re)render now and kick off a worker if so."""
    gid = g["id"]
    if gid in _inflight:
        return
    with _conn() as c:
        vis = _visible_count(c, gid)
    new_words = vis - (g["words_at_render"] if g["words_at_render"] is not None else -1)
    elapsed = time.time() - (g["last_render_at"] or 0)
    due = (
        force
        or g["words_at_render"] is None or g["words_at_render"] < 0          # never rendered
        or (new_words >= GATHER_RENDER_EVERY)
        or (new_words > 0 and elapsed >= GATHER_RENDER_INTERVAL)
    )
    if not due:
        return
    _inflight.add(gid)
    threading.Thread(target=_do_render, args=(gid,), daemon=True).start()


# --- helpers -----------------------------------------------------------------

def _public_state(g: sqlite3.Row, include_hidden: bool = False) -> dict:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, text, name, created_at, hidden FROM contributions "
            "WHERE gather_id=? ORDER BY id DESC", (g["id"],)
        ).fetchall()
    words = [
        {"id": r["id"], "text": r["text"], "name": r["name"], "hidden": bool(r["hidden"])}
        for r in rows if include_hidden or not r["hidden"]
    ]
    visible = sum(1 for r in rows if not r["hidden"])
    contributors = len({(r["name"] or "").strip().lower() for r in rows if not r["hidden"] and (r["name"] or "").strip()})
    war = g["words_at_render"] if g["words_at_render"] is not None else -1
    return {
        "title": g["title"],
        "status": g["status"],
        "count": visible,
        "contributors": contributors,
        "close_at": g["close_at"],
        "has_portrait": bool(g["has_portrait"]),
        # True when words have arrived that the current portrait doesn't yet show
        # (a throttled re-render is pending) -> the pages show a quiet "updating…".
        "pending": visible != war,
        "portrait_url": f"/api/gather/{g['share_token']}/portrait.png?v={int(g['last_render_at'] or 0)}",
        "words": words,
    }


def _rl_ok(bucket: dict, key: str, limit: int, window: float = 60.0) -> bool:
    now = time.time()
    hist = [t for t in bucket.get(key, []) if now - t < window]
    if len(hist) >= limit:
        bucket[key] = hist
        return False
    hist.append(now)
    bucket[key] = hist
    return True


_rl_add: dict = {}     # per-IP word-add rate limit


# --- routes: pages -----------------------------------------------------------

@router.get("/g/{share}")
def page_contribute(share: str):
    if not _gather_by("share_token", share):
        raise HTTPException(404, "gather not found")
    return FileResponse(str(STATIC_DIR / "gather.html"))


@router.get("/a/{admin}")
def page_admin(admin: str):
    if not _gather_by("admin_token", admin):
        raise HTTPException(404, "gather not found")
    return FileResponse(str(STATIC_DIR / "gather-admin.html"))


# --- routes: API -------------------------------------------------------------

@router.post("/api/gather/create")
async def create(
    image: UploadFile = File(...),
    title: str = Form(...),
    words: str = Form(""),                 # seed words (comma/space separated)
    style: str = Form("displacement"),
    ink: str = Form("photo"),
    ground: str = Form("paper"),
    min_font: float = Form(20.0),
    close_at: Optional[float] = Form(None),
):
    title = _clean_text(title) or "In their words"
    share_token = secrets.token_urlsafe(7)
    admin_token = secrets.token_urlsafe(18)
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "no image")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO gathers(share_token,admin_token,title,photo_path,style,ink,ground,min_font,created_at,close_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (share_token, admin_token, title, "", style, ink, ground, float(min_font or 20),
             time.time(), close_at),
        )
        gid = cur.lastrowid
        c.commit()
    # store the source photo under the gather's dir, then backfill the path
    pdir = GATHER_DIR / str(gid)
    pdir.mkdir(parents=True, exist_ok=True)
    ppath = pdir / "source.jpg"
    ppath.write_bytes(raw)
    with _conn() as c:
        c.execute("UPDATE gathers SET photo_path=? WHERE id=?", (str(ppath), gid))
        c.commit()
    # seed words as the creator's first contribution(s)
    seed = [w for w in re.split(r"[,\n]+", words) if w.strip()]
    if seed:
        with _conn() as c:
            for s in seed:
                c.execute(
                    "INSERT INTO contributions(gather_id,text,name,created_at) VALUES(?,?,?,?)",
                    (gid, _clean_text(s), None, time.time()),
                )
            c.commit()
    _maybe_render(_gather_by("id", str(gid)), force=True)
    return {
        "ok": True,
        "gather_id": gid,
        "share_url": f"{PUBLIC_BASE_URL}/g/{share_token}",
        "admin_url": f"{PUBLIC_BASE_URL}/a/{admin_token}",
    }


@router.get("/api/gather/{share}/state")
def state(share: str):
    g = _gather_by("share_token", share)
    if not g:
        raise HTTPException(404, "not found")
    if g["status"] == "open":
        _maybe_render(g)                  # opportunistic throttled refresh
    return _public_state(g)


@router.post("/api/gather/{share}/word")
async def add_word(share: str, request: Request, text: str = Form(...), name: str = Form("")):
    g = _gather_by("share_token", share)
    if not g:
        raise HTTPException(404, "not found")
    if g["status"] != "open":
        return JSONResponse({"ok": False, "error": "closed"}, status_code=409)
    ip = (request.client.host if request.client else "?")
    if not _rl_ok(_rl_add, ip, limit=20):
        return JSONResponse({"ok": False, "error": "rate_limited"}, status_code=429)
    t = _clean_text(text)
    if not t or len(t) < 2:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    if _BLOCK.search(t):
        return JSONResponse({"ok": False, "error": "blocked"}, status_code=400)
    nm = _clean_text(name)[:24] or None
    with _conn() as c:
        c.execute(
            "INSERT INTO contributions(gather_id,text,name,created_at) VALUES(?,?,?,?)",
            (g["id"], t, nm, time.time()),
        )
        c.commit()
    _maybe_render(_gather_by("share_token", share))
    return {"ok": True}


@router.get("/api/gather/{share}/portrait.png")
def portrait(share: str):
    g = _gather_by("share_token", share)
    if not g:
        raise HTTPException(404, "not found")
    p = _portrait_path(g["id"])
    if not p.exists():
        raise HTTPException(404, "not rendered yet")
    return FileResponse(str(p), media_type="image/png")


# --- routes: admin (token in the path == bearer auth) ------------------------

@router.get("/api/gather/admin/{admin}/state")
def admin_state(admin: str):
    g = _gather_by("admin_token", admin)
    if not g:
        raise HTTPException(404, "not found")
    if g["status"] == "open":
        _maybe_render(g)               # dashboard polling also keeps the portrait fresh
    s = _public_state(g, include_hidden=True)
    s["share_url"] = f"{PUBLIC_BASE_URL}/g/{g['share_token']}"
    return s


@router.post("/api/gather/admin/{admin}/word/{wid}/hide")
def admin_hide(admin: str, wid: int, hidden: int = Form(1)):
    g = _gather_by("admin_token", admin)
    if not g:
        raise HTTPException(404, "not found")
    with _conn() as c:
        c.execute("UPDATE contributions SET hidden=? WHERE id=? AND gather_id=?",
                  (1 if hidden else 0, wid, g["id"]))
        c.commit()
    return {"ok": True}


@router.post("/api/gather/admin/{admin}/render")
def admin_render(admin: str):
    g = _gather_by("admin_token", admin)
    if not g:
        raise HTTPException(404, "not found")
    _maybe_render(g, force=True)
    return {"ok": True}


@router.post("/api/gather/admin/{admin}/reveal")
def admin_reveal(admin: str):
    g = _gather_by("admin_token", admin)
    if not g:
        raise HTTPException(404, "not found")
    with _conn() as c:
        c.execute("UPDATE gathers SET status='revealed' WHERE id=?", (g["id"],))
        c.commit()
    _maybe_render(_gather_by("admin_token", admin), force=True)   # final edition
    return {"ok": True, "share_url": f"{PUBLIC_BASE_URL}/g/{g['share_token']}"}
