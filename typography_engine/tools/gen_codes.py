#!/usr/bin/env python3
"""Generate & inspect redemption codes.

A redemption code is a prepaid ticket for one product: the recipient enters it
at /redeem and gets the full personalize + fulfill flow with no Stripe charge
(the sale happened off-platform — e.g. an EverLoved order, a gift, a pre-need
arrangement). Codes are one-time use.

Run from the engine directory (so it finds app/ and writes data/redemption.db):

    python tools/gen_codes.py generate <sku> <count> [--batch NAME] [--note "..."]
    python tools/gen_codes.py stats  [--batch NAME]
    python tools/gen_codes.py list   [--status unused|redeeming|used] [--batch NAME]

Examples:
    # 20 codes for the 16x20 framed portrait, tagged to this month's EverLoved batch
    python tools/gen_codes.py generate framed_16x20 20 --batch everloved-2026-07
    python tools/gen_codes.py stats --batch everloved-2026-07

Valid SKUs come from app/products.py (get one wrong and it's rejected). A code
carries its SKU, so generate a separate batch per product you sell.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `app` importable when invoked as `python tools/gen_codes.py` from the
# engine directory.
_ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from app import products, redemption   # noqa: E402


def _cmd_generate(args: argparse.Namespace) -> int:
    sku = args.sku
    product = products.get(sku)
    if not product:
        valid = ", ".join(sorted(p.sku for p in products.CATALOG))
        print(f"error: unknown sku {sku!r}. Valid SKUs: {valid}", file=sys.stderr)
        return 2
    if args.count < 1:
        print("error: count must be >= 1", file=sys.stderr)
        return 2

    redemption.init_db()
    codes = redemption.generate(sku, args.count, batch=args.batch, note=args.note)

    kind = "physical print" if product.physical else "digital download"
    print(f"# {len(codes)} redemption code(s) for {sku}  ({product.name} — {kind})")
    if args.batch:
        print(f"# batch: {args.batch}")
    print(f"# generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" + "-" * 40)
    for code in codes:
        print(code)

    if args.out:
        out = Path(args.out)
        header = (f"Loved in Words — redemption codes\n"
                  f"sku: {sku} ({product.name})\n"
                  f"batch: {args.batch or '-'}\n"
                  f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  + "-" * 40 + "\n")
        out.write_text(header + "\n".join(codes) + "\n", encoding="utf-8")
        print(f"#\n# wrote {len(codes)} code(s) to {out}", file=sys.stderr)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    redemption.init_db()
    s = redemption.stats(batch=args.batch)
    label = f" (batch: {args.batch})" if args.batch else ""
    print(f"Redemption codes{label}:")
    print(f"  unused    : {s['unused']}")
    print(f"  redeeming : {s['redeeming']}   (in-flight; reclaimed after "
          f"{redemption.REDEEM_STALE_SECONDS // 60} min if abandoned)")
    print(f"  used      : {s['used']}")
    print(f"  total     : {s['total']}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    redemption.init_db()
    rows = redemption.list_codes(status=args.status, batch=args.batch, limit=args.limit)
    if not rows:
        print("(no codes match)")
        return 0
    print(f"{'CODE':<20} {'SKU':<16} {'STATUS':<10} {'BATCH':<18} {'ORDER/JOB':<18} EMAIL")
    for r in rows:
        ref = r.get("order_id") or r.get("job_id") or "-"
        print(f"{redemption.format_code(r['code']):<20} {r['sku']:<16} "
              f"{r['status']:<10} {(r.get('batch') or '-'):<18} {str(ref):<18} "
              f"{r.get('customer_email') or ''}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate & inspect redemption codes.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create new codes for a SKU")
    g.add_argument("sku", help="product SKU (see app/products.py)")
    g.add_argument("count", type=int, help="how many codes to create")
    g.add_argument("--batch", help="label to group this batch (e.g. everloved-2026-07)")
    g.add_argument("--note", help="freeform note stored on each code")
    g.add_argument("--out", help="also write the codes to this text file")
    g.set_defaults(func=_cmd_generate)

    st = sub.add_parser("stats", help="counts by status")
    st.add_argument("--batch", help="restrict to a batch")
    st.set_defaults(func=_cmd_stats)

    ls = sub.add_parser("list", help="list codes")
    ls.add_argument("--status", choices=("unused", "redeeming", "used"))
    ls.add_argument("--batch", help="restrict to a batch")
    ls.add_argument("--limit", type=int, default=500)
    ls.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
