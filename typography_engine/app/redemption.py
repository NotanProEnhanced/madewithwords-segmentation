"""Redemption codes — SQLite-backed one-time entitlement tokens.

A code is a prepaid ticket for exactly one product. A partner (EverLoved), a
gift buyer, or a pre-need arrangement sells/hands out a code; the recipient
enters it at /redeem, personalizes the portrait in the normal studio, and the
existing fulfillment path (digital download or Printful print) runs with NO
Stripe charge — the sale already happened off-platform.

Design mirrors app/orders.py: a single-file DB under data/ (volume-mounted so
it survives rebuilds), one row per code, a process-local lock (single uvicorn
worker), human-readable string statuses.

Lifecycle:  unused --begin_redeem--> redeeming --complete_redeem--> used
                                         |
                                         +--release (on failure) --> unused

`begin_redeem` is the atomic double-spend guard: it flips unused->redeeming in
one UPDATE and only the winning caller proceeds to fulfillment. If fulfillment
then fails, `release` returns the code to unused so the customer can retry. A
code left 'redeeming' by a crash is reclaimed after REDEEM_STALE_SECONDS.
"""
from __future__ import annotations

import secrets
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from .config import REDEEM_DB


_LOCK = threading.Lock()

# Unambiguous alphabet: no 0/O, 1/I/L — codes get read aloud and typed by
# grieving families, so every character must be unmistakable.
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_GROUPS = 3          # LIW-XXXX-XXXX-XXXX
_GROUP_LEN = 4
# A code stuck in 'redeeming' longer than this (server died mid-fulfillment) is
# considered abandoned and becomes redeemable again.
REDEEM_STALE_SECONDS = 15 * 60


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(REDEEM_DB), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    with _LOCK, _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS codes (
                code TEXT PRIMARY KEY,          -- canonical (no dashes, upper)
                sku TEXT NOT NULL,              -- product this code entitles
                batch TEXT,                     -- optional label (e.g. "everloved-2026-07")
                status TEXT NOT NULL,           -- unused | redeeming | used
                created_at REAL NOT NULL,
                redeeming_at REAL,              -- when begin_redeem last claimed it
                redeemed_at REAL,               -- when it was finally consumed
                job_id TEXT,                    -- the rendered artwork used
                order_id TEXT,                  -- physical order id (orders.db); null for digital
                customer_email TEXT,
                note TEXT                        -- freeform (who it was issued to, etc.)
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_codes_status ON codes(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_codes_batch ON codes(batch)")


# --- Code formatting ------------------------------------------------------

def _normalize(code: str) -> str:
    """Canonical form for storage/lookup: uppercase, drop the LIW- display prefix,
    then keep only alphabet chars. Forgiving of how a customer types it (spaces,
    dashes, lowercase, with or without the prefix). The prefix must be stripped
    BEFORE alphabet-filtering: 'W' is a valid code char, so a naive filter would
    leave a stray 'W' behind from 'LIW'."""
    if not code:
        return ""
    up = "".join(ch for ch in code.strip().upper() if ch.isalnum())
    if up.startswith("LIW"):
        up = up[3:]
    return "".join(ch for ch in up if ch in _ALPHABET)


def format_code(canonical: str) -> str:
    """Group a canonical code into LIW-XXXX-XXXX-XXXX for display."""
    canonical = _normalize(canonical)
    groups = [canonical[i:i + _GROUP_LEN] for i in range(0, len(canonical), _GROUP_LEN)]
    return "LIW-" + "-".join(groups) if groups else ""


def _random_code() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(_GROUPS * _GROUP_LEN))


# --- Generation -----------------------------------------------------------

def generate(sku: str, count: int, *, batch: Optional[str] = None,
             note: Optional[str] = None) -> List[str]:
    """Create `count` fresh unused codes for `sku`. Returns display-formatted
    codes (LIW-XXXX-XXXX-XXXX). Collisions are astronomically unlikely with this
    alphabet/length but are retried anyway."""
    if count < 1:
        return []
    now = time.time()
    out: List[str] = []
    with _LOCK, _conn() as c:
        for _ in range(count):
            for _attempt in range(8):
                canonical = _random_code()
                try:
                    c.execute(
                        "INSERT INTO codes (code, sku, batch, status, created_at, note) "
                        "VALUES (?, ?, ?, 'unused', ?, ?)",
                        (canonical, sku, batch or None, now, note or None),
                    )
                    out.append(format_code(canonical))
                    break
                except sqlite3.IntegrityError:
                    continue   # collision — try another
            else:
                raise RuntimeError("could not generate a unique code after 8 attempts")
    return out


# --- Lookup ---------------------------------------------------------------

def get(code: str) -> Optional[Dict[str, Any]]:
    canonical = _normalize(code)
    if not canonical:
        return None
    with _LOCK, _conn() as c:
        row = c.execute("SELECT * FROM codes WHERE code = ?", (canonical,)).fetchone()
    return dict(row) if row else None


def is_redeemable(code: str) -> bool:
    """True if the code exists and is available to redeem (unused, or a stale
    'redeeming' claim we'd reclaim). Read-only — used to validate at /redeem
    before committing the customer to the studio."""
    o = get(code)
    if not o:
        return False
    if o["status"] == "unused":
        return True
    if o["status"] == "redeeming":
        return (time.time() - (o.get("redeeming_at") or 0)) > REDEEM_STALE_SECONDS
    return False


# --- Atomic redeem --------------------------------------------------------

def begin_redeem(code: str) -> Optional[Dict[str, Any]]:
    """Atomically claim a code for fulfillment: flip unused->redeeming (or reclaim
    a stale redeeming) in a single UPDATE. Returns the row dict if THIS caller won
    the claim, else None (unknown code, already used, or a fresh in-flight claim).
    The winner must then call complete_redeem (success) or release (failure)."""
    canonical = _normalize(code)
    if not canonical:
        return None
    now = time.time()
    stale_before = now - REDEEM_STALE_SECONDS
    with _LOCK, _conn() as c:
        cur = c.execute(
            """
            UPDATE codes SET status='redeeming', redeeming_at=?
            WHERE code=? AND (
                status='unused'
                OR (status='redeeming' AND COALESCE(redeeming_at, 0) < ?)
            )
            """,
            (now, canonical, stale_before),
        )
        if cur.rowcount != 1:
            return None
        row = c.execute("SELECT * FROM codes WHERE code=?", (canonical,)).fetchone()
    return dict(row) if row else None


def complete_redeem(code: str, *, job_id: Optional[str],
                    order_id: Optional[str],
                    customer_email: Optional[str]) -> None:
    """Mark a claimed code fully consumed. Idempotent-safe: only a code currently
    in 'redeeming' is transitioned."""
    canonical = _normalize(code)
    now = time.time()
    with _LOCK, _conn() as c:
        c.execute(
            """
            UPDATE codes SET status='used', redeemed_at=?, job_id=?, order_id=?,
                             customer_email=?
            WHERE code=? AND status='redeeming'
            """,
            (now, job_id, order_id, customer_email, canonical),
        )


def release(code: str) -> None:
    """Return a claimed-but-failed code to 'unused' so the customer can retry.
    Only affects a code still in 'redeeming' (never un-consumes a used code)."""
    canonical = _normalize(code)
    with _LOCK, _conn() as c:
        c.execute(
            "UPDATE codes SET status='unused', redeeming_at=NULL "
            "WHERE code=? AND status='redeeming'",
            (canonical,),
        )


def void(code: str) -> bool:
    """Permanently retire a code so it can never be redeemed — for an orphaned or
    mistakenly-issued code. Only touches a code that hasn't been consumed yet
    ('unused' or 'redeeming'); a 'used' code is left alone (its redemption is
    history). 'void' is a terminal status that `is_redeemable`/`begin_redeem` never
    accept. Returns True if a code was voided."""
    canonical = _normalize(code)
    if not canonical:
        return False
    with _LOCK, _conn() as c:
        cur = c.execute(
            "UPDATE codes SET status='void', redeeming_at=NULL "
            "WHERE code=? AND status IN ('unused', 'redeeming')",
            (canonical,),
        )
        return cur.rowcount == 1


# --- Reporting ------------------------------------------------------------

def stats(batch: Optional[str] = None) -> Dict[str, int]:
    q = "SELECT status, COUNT(*) n FROM codes"
    args: tuple = ()
    if batch:
        q += " WHERE batch=?"
        args = (batch,)
    q += " GROUP BY status"
    counts = {"unused": 0, "redeeming": 0, "used": 0}
    with _LOCK, _conn() as c:
        for row in c.execute(q, args).fetchall():
            counts[row["status"]] = row["n"]
    counts["total"] = counts["unused"] + counts["redeeming"] + counts["used"]
    return counts


def list_codes(status: Optional[str] = None, batch: Optional[str] = None,
               limit: int = 500) -> List[Dict[str, Any]]:
    q = "SELECT * FROM codes"
    where, args = [], []
    if status:
        where.append("status=?"); args.append(status)
    if batch:
        where.append("batch=?"); args.append(batch)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with _LOCK, _conn() as c:
        rows = c.execute(q, tuple(args)).fetchall()
    return [dict(r) for r in rows]
