"""SQLite-backed order persistence.

Single-file DB at `data/orders.db`, volume-mounted in docker-compose so it
survives container rebuilds. Schema is small: one row per order, JSON blobs
for the parts we don't query on (recipient, raw Stripe/Printful responses).

Idempotency: orders are keyed by the Stripe Checkout Session ID. The
webhook handler can be called multiple times for the same session and we
will only create the Printful order on the first one.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ORDERS_DB


_LOCK = threading.Lock()

# Status values (kept as strings, not an enum, so the DB is human-readable):
#   pending_payment  - Stripe Checkout Session created, not yet paid
#   paid             - Stripe webhook received "checkout.session.completed"
#   fulfilling       - Printful order created (physical only)
#   shipped          - Printful pushed a shipment update
#   delivered        - Printful pushed a delivery confirmation
#   error            - something went wrong submitting to Printful


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(ORDERS_DB), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    with _LOCK, _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,            -- public order id (uuid hex slice)
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                stripe_session_id TEXT UNIQUE,
                stripe_payment_intent TEXT,
                job_id TEXT NOT NULL,           -- which rendered artwork
                sku TEXT NOT NULL,
                size TEXT,                      -- nullable; only for sized products
                variant_id INTEGER,             -- Printful variant; null for digital
                price_cents INTEGER NOT NULL,
                shipping_cents INTEGER NOT NULL DEFAULT 0,
                currency TEXT NOT NULL,
                status TEXT NOT NULL,
                customer_email TEXT,
                recipient_json TEXT,            -- shipping address as JSON
                printful_order_id INTEGER,
                printful_raw_json TEXT,         -- last response from Printful
                tracking_url TEXT,
                error_message TEXT,
                ref TEXT                        -- referral/source tag (partner attribution)
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at DESC)")
        # Migration: `ref` was added later, so existing DBs need the column appended.
        cols = {r[1] for r in c.execute("PRAGMA table_info(orders)").fetchall()}
        if "ref" not in cols:
            c.execute("ALTER TABLE orders ADD COLUMN ref TEXT")


def create_pending(
    *,
    order_id: str,
    stripe_session_id: str,
    job_id: str,
    sku: str,
    size: Optional[str],
    variant_id: Optional[int],
    price_cents: int,
    shipping_cents: int,
    currency: str,
    ref: Optional[str] = None,
) -> None:
    now = time.time()
    with _LOCK, _conn() as c:
        c.execute(
            """
            INSERT INTO orders (
                id, created_at, updated_at, stripe_session_id, job_id,
                sku, size, variant_id, price_cents, shipping_cents, currency, status, ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending_payment', ?)
            """,
            (order_id, now, now, stripe_session_id, job_id, sku, size,
             variant_id, price_cents, shipping_cents, currency, ref or None),
        )


def get(order_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK, _conn() as c:
        row = c.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    return _row_to_dict(row)


def get_by_session(session_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK, _conn() as c:
        row = c.execute(
            "SELECT * FROM orders WHERE stripe_session_id = ?", (session_id,)
        ).fetchone()
    return _row_to_dict(row)


def list_recent(limit: int = 100) -> List[Dict[str, Any]]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows if r]


def mark_paid(
    *,
    stripe_session_id: str,
    payment_intent: Optional[str],
    customer_email: Optional[str],
    recipient: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Idempotently mark an order paid. Returns the order dict if this call
    transitioned it (so the caller can submit to Printful), or None if it
    was already paid (so the caller skips Printful submission)."""
    now = time.time()
    with _LOCK, _conn() as c:
        row = c.execute(
            "SELECT * FROM orders WHERE stripe_session_id = ?", (stripe_session_id,)
        ).fetchone()
        if not row:
            return None
        if row["status"] != "pending_payment":
            return None
        c.execute(
            """
            UPDATE orders
            SET status='paid', updated_at=?, stripe_payment_intent=?,
                customer_email=?, recipient_json=?
            WHERE stripe_session_id=?
            """,
            (now, payment_intent, customer_email,
             json.dumps(recipient) if recipient else None,
             stripe_session_id),
        )
        new_row = c.execute(
            "SELECT * FROM orders WHERE stripe_session_id = ?", (stripe_session_id,)
        ).fetchone()
    return _row_to_dict(new_row)


def mark_fulfilling(*, order_id: str, printful_order_id: int, raw: Dict[str, Any]) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            """
            UPDATE orders
            SET status='fulfilling', updated_at=?, printful_order_id=?, printful_raw_json=?
            WHERE id=?
            """,
            (time.time(), printful_order_id, json.dumps(raw), order_id),
        )


def mark_shipped(*, order_id: str, tracking_url: Optional[str], raw: Dict[str, Any]) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            """
            UPDATE orders
            SET status='shipped', updated_at=?, tracking_url=?, printful_raw_json=?
            WHERE id=?
            """,
            (time.time(), tracking_url, json.dumps(raw), order_id),
        )


def mark_error(*, order_id: str, error_message: str) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "UPDATE orders SET status='error', updated_at=?, error_message=? WHERE id=?",
            (time.time(), error_message, order_id),
        )


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    for k in ("recipient_json", "printful_raw_json"):
        if d.get(k):
            try:
                d[k.removesuffix("_json")] = json.loads(d[k])
            except (json.JSONDecodeError, TypeError):
                d[k.removesuffix("_json")] = None
        else:
            d[k.removesuffix("_json")] = None
    return d
