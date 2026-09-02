#!/usr/bin/env python3
"""Close out orders that are stuck in a non-terminal status and are never moving.

WHAT THIS IS FOR
  Fifteen orders on production sit in 'fulfilling' from June and July. They
  carry Printful order IDs, so Printful accepted them at the time, but they
  return HTTP 404 under the token production now holds and do not appear in
  the store the dashboard shows. Nothing will ever transition them: the
  Printful webhook that would call mark_shipped() fires only for orders that
  Printful still knows about. They will sit in 'fulfilling' forever.

  FaithInWords and LovedInWords hold four orders each that were never checked;
  --all covers them too.

WHAT IT DOES *NOT* CLAIM
  The closing status is deliberately NOT 'shipped' or 'delivered'. We have no
  evidence these were fulfilled -- an empty store view is equally consistent
  with "never fulfilled" as with "fulfilled and forgotten". Writing 'delivered'
  would turn an open question into a false record, and would erase the only
  trace that fifteen people paid. The default status is 'closed_stale', which
  says exactly what we know: the record went stale and was closed by hand.

  The previous status, the closing date and your stated reason are written to a
  `closed_note` column, so the history survives the cleanup.

SAFETY
  * Dry run by default. Nothing is written without --apply.
  * Every database is copied with `sqlite3 .backup` into /root/backups/
    before a single row changes. To undo, copy the file back.
  * Only orders older than --older-than days (default 30) are eligible, so a
    live order placed this morning can never be swept up.
  * 'pending_payment' rows are abandoned checkouts, not stuck orders, and are
    left alone unless you pass --include-unpaid (they then close as
    'abandoned', not 'closed_stale' -- different thing, different word).
  * Reads and writes only the orders table. Touches no file in any tree, no
    running container, and nothing at Printful or Stripe.

ONE DISPLAY CONSEQUENCE, WORTH KNOWING BEFORE YOU RUN IT
  The customer order-status page maps known statuses to a sentence and falls
  back to printing the status itself, so a closed order shows "closed stale"
  rather than a friendly line. It also gates the artwork download link on
  status being one of paid/fulfilling/shipped/delivered, so closing removes
  that link. For these particular orders the rendered files are long past the
  ~30-day deletion window anyway, so the link would 404 either way -- but if
  you would rather keep the page tidy, add 'closed_stale' to the message map
  in main.py before running this.

Usage:
    python3 close-stale-orders.py                          # dry run, prod only
    python3 close-stale-orders.py --all                    # dry run, all trees
    python3 close-stale-orders.py --all --apply \
        --reason "not present in any Printful store; see billing check 2026-08"
"""
import argparse
import datetime
import os
import shutil
import sqlite3
import sys
import time

TREES = [
    "typortrait-prod",
    "typortrait-faithinwords",
    "typortrait-lovedinwords",
    "typortrait-pawsinwords",
    "typortrait-stg",
]

# Statuses that will never move on their own. 'error' is excluded: it is already
# terminal and already says what happened.
STUCK = ("fulfilling", "paid", "submitted", "pending_fulfillment")
UNPAID = ("pending_payment",)

BACKUP_ROOT = os.environ.get("BACKUP_ROOT", "/root/backups")
TREE_ROOT = os.environ.get("TREE_ROOT", "/root")


def db_path(tree):
    return "%s/%s/typography_engine/data/orders.db" % (TREE_ROOT, tree)


def ensure_note_column(conn):
    """Add closed_note if it is not there. Safe: the app writes named columns and
    reads with SELECT *, so an extra column is invisible to it."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(orders)").fetchall()}
    if "closed_note" not in cols:
        conn.execute("ALTER TABLE orders ADD COLUMN closed_note TEXT")
        return True
    return False


def candidates(conn, cutoff, include_unpaid):
    wanted = list(STUCK) + (list(UNPAID) if include_unpaid else [])
    qs = ",".join("?" * len(wanted))
    return conn.execute(
        "SELECT * FROM orders WHERE status IN (%s) AND created_at < ? "
        "ORDER BY created_at" % qs,
        wanted + [cutoff],
    ).fetchall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="append",
                    help="tree to act on (repeatable); default typortrait-prod")
    ap.add_argument("--all", action="store_true", help="every tree")
    ap.add_argument("--older-than", type=int, default=30, metavar="DAYS",
                    help="only orders older than this (default 30)")
    ap.add_argument("--status", default="closed_stale",
                    help="closing status for stuck orders (default closed_stale)")
    ap.add_argument("--include-unpaid", action="store_true",
                    help="also close pending_payment rows, as 'abandoned'")
    ap.add_argument("--reason", default="",
                    help="why, recorded in closed_note alongside the old status")
    ap.add_argument("--apply", action="store_true", help="actually write")
    args = ap.parse_args()

    if args.status in ("shipped", "delivered"):
        raise SystemExit(
            "refusing --status=%s: that asserts a fulfillment we have not verified.\n"
            "If you have proof they shipped (a Printful billing charge, a tracking\n"
            "number), that proof belongs in the record -- set the status by hand for\n"
            "those specific orders rather than sweeping all of them into it."
            % args.status)

    trees = TREES if args.all else (args.tree or ["typortrait-prod"])
    cutoff = time.time() - args.older_than * 86400
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    note_date = datetime.date.today().isoformat()

    backup_dir = os.path.join(BACKUP_ROOT, "pre-close-%s" % stamp)
    total = 0
    plan = []

    for tree in trees:
        p = db_path(tree)
        if not os.path.isfile(p):
            print("%-28s no orders.db" % tree)
            continue
        conn = sqlite3.connect(p, timeout=15.0)
        conn.row_factory = sqlite3.Row
        rows = candidates(conn, cutoff, args.include_unpaid)
        conn.close()
        print("%-28s %d order(s) older than %dd in a non-terminal status"
              % (tree, len(rows), args.older_than))
        if rows:
            print("    %-11s %-18s %-16s %-11s %s"
                  % ("ordered", "order_id", "status", "printful", "email"))
            for r in rows:
                when = datetime.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d")
                # customer addresses are personal data; show enough to identify a
                # row without printing the address into a terminal or a log
                em = (r["customer_email"] or "")
                em = (em.split("@")[0][:3] + "***@" + em.split("@")[-1]) if "@" in em else "-"
                print("    %-11s %-18s %-16s %-11s %s"
                      % (when, r["id"], r["status"], r["printful_order_id"] or "-", em))
            plan.append((tree, p, [dict(r) for r in rows]))
            total += len(rows)
        print()

    if not total:
        print("Nothing to close.")
        return

    if not args.apply:
        print("DRY RUN -- %d order(s) would be closed as '%s'"
              % (total, args.status))
        print("Re-run with --apply (and --reason \"...\") to write.")
        return

    os.makedirs(backup_dir, exist_ok=True)
    print("backing up %d database(s) to %s" % (len(plan), backup_dir))
    for tree, p, _rows in plan:
        dest = os.path.join(backup_dir, "%s-orders.db" % tree)
        src = sqlite3.connect(p, timeout=15.0)
        dst = sqlite3.connect(dest)
        with dst:
            src.backup(dst)      # consistent copy of a live WAL-mode database
        dst.close()
        src.close()
        os.chmod(dest, 0o600)
        print("  %s" % dest)
    print()

    now = time.time()
    for tree, p, rows in plan:
        conn = sqlite3.connect(p, timeout=15.0)
        with conn:
            added = ensure_note_column(conn)
            if added:
                print("%-28s added closed_note column" % tree)
            n = 0
            for r in rows:
                new_status = "abandoned" if r["status"] in UNPAID else args.status
                note = "was '%s'; closed %s by close-stale-orders.py" % (
                    r["status"], note_date)
                # --reason describes the stuck orders; an abandoned checkout is a
                # different thing and the status already explains itself
                if args.reason and new_status != "abandoned":
                    note += "; reason: %s" % args.reason
                prev = r.get("closed_note")
                if prev:
                    note = prev + " | " + note
                conn.execute(
                    "UPDATE orders SET status=?, updated_at=?, closed_note=? "
                    "WHERE id=? AND status=?",
                    (new_status, now, note, r["id"], r["status"]),
                )
                n += 1
            print("%-28s closed %d" % (tree, n))
        conn.close()

    print()
    print("Done. %d order(s) closed." % total)
    print("To undo: stop the container, copy the matching file from")
    print("  %s" % backup_dir)
    print("back over data/orders.db, and start it again.")
    print()
    print("This changed our records only. It did not cancel, refund or fulfill")
    print("anything, and it does not establish whether these customers received")
    print("a print.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
