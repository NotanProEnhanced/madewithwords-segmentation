#!/usr/bin/env python3
"""Read-only: look up every non-terminal order at Printful, with a chosen token.

WHY THIS EXISTS SEPARATELY
  Fifteen orders from June and July sit in 'fulfilling'. Queried with the token
  production currently holds they all return HTTP 404 -- but each has a Printful
  order ID, so Printful accepted them at the time. Printful's classic API tokens
  are store-scoped and ignore X-PF-Store-Id, so a token for one store returns 404
  for orders that live in another. These predate the store-routing fix, so they
  were most likely created under the previous store.

  This lets you query them with THAT store's token without editing any config or
  touching the running service.

TOKEN HANDLING
  The token is read from a file, never from the command line (which would put it
  in shell history) and never from this repo. Create it on the server:

      nano /root/.pf-token-oldstore     # paste the token, save
      chmod 600 /root/.pf-token-oldstore

  and shred it when you are done:

      shred -u /root/.pf-token-oldstore

  The token is never printed. Only order id, status and tracking are shown.

WRITES NOTHING. Not to the database, not to Printful.

Usage (inside the container):
    python /app/data/pf-check-orders.py /app/data/.pf-token
"""
import datetime
import json
import sqlite3
import sys
import urllib.error
import urllib.request

sys.path.insert(0, "/app")
from app.config import ORDERS_DB, PRINTFUL_API_BASE   # noqa: E402

NON_TERMINAL = ("fulfilling", "paid", "submitted", "pending_fulfillment")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: pf-check-orders.py <path-to-token-file>")
    try:
        token = open(sys.argv[1]).read().strip()
    except OSError as e:
        raise SystemExit("cannot read token file: %s" % e)
    if not token:
        raise SystemExit("token file is empty")

    conn = sqlite3.connect(str(ORDERS_DB))
    conn.row_factory = sqlite3.Row
    rows = [r for r in conn.execute("SELECT * FROM orders ORDER BY created_at")
            if r["status"] in NON_TERMINAL]
    print("orders to check: %d" % len(rows))
    print("%-11s %-18s %-11s %-14s %s"
          % ("ordered", "order_id", "printful", "pf_status", "tracking"))

    counts = {}
    for r in rows:
        pid = r["printful_order_id"]
        when = datetime.datetime.fromtimestamp(
            r["created_at"], datetime.timezone.utc).strftime("%Y-%m-%d")
        if not pid:
            print("%-11s %-18s %-11s %-14s" % (when, r["id"], "-", "NO_PF_ID"))
            counts["NO_PF_ID"] = counts.get("NO_PF_ID", 0) + 1
            continue
        req = urllib.request.Request(
            "%s/orders/%s" % (PRINTFUL_API_BASE.rstrip("/"), pid))
        req.add_header("Authorization", "Bearer " + token)
        try:
            res = json.loads(urllib.request.urlopen(req, timeout=25).read()).get("result") or {}
        except urllib.error.HTTPError as e:
            print("%-11s %-18s %-11s %-14s" % (when, r["id"], pid, "HTTP_%d" % e.code))
            counts["HTTP_%d" % e.code] = counts.get("HTTP_%d" % e.code, 0) + 1
            continue
        except Exception as e:  # noqa: BLE001
            print("%-11s %-18s %-11s %-14s %s" % (when, r["id"], pid, "FAILED", e))
            counts["FAILED"] = counts.get("FAILED", 0) + 1
            continue
        status = res.get("status") or "?"
        trk = ""
        for s in (res.get("shipments") or []):
            trk = s.get("tracking_url") or s.get("tracking_number") or trk
        print("%-11s %-18s %-11s %-14s %s" % (when, r["id"], pid, status, trk))
        counts[status] = counts.get(status, 0) + 1

    print("\nsummary:")
    for k in sorted(counts):
        print("  %-16s %d" % (k, counts[k]))
    print("\nnothing was modified.")


if __name__ == "__main__":
    main()
