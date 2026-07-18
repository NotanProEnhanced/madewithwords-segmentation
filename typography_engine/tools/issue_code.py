#!/usr/bin/env python3
"""Issue a redemption code to a buyer in one step: generate a code AND email it to
them with the redeem link. This is the operational glue between an EverLoved sale
and the buyer receiving their code — run it once per order (from the order's buyer
email + the product they bought).

    python tools/issue_code.py <sku> <buyer_email> [--name "Jane"] [--batch everloved-2026-07]
    python tools/issue_code.py framed_16x20 buyer@example.com --name "Jane" --dry-run

--dry-run prints the code + the exact email that WOULD be sent, without sending.

Needs SMTP configured (TYPO_SMTP_HOST/USER/PASS) to actually send — the same creds
the app already uses. The redeem link is built from TYPO_PUBLIC_URL; override with
--link-base if the buyer should land on a specific host.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from app import products, redemption            # noqa: E402
from app import admin as admin_mod              # noqa: E402
from app.config import PUBLIC_BASE_URL          # noqa: E402

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _email_bodies(code_fmt: str, link: str, product_name: str, name: str):
    hi = f"Hi {name}," if name else "Hello,"
    text = (
        f"{hi}\n\n"
        f"Thank you. Your keepsake — {product_name} — is ready to personalize.\n\n"
        f"Your redemption code: {code_fmt}\n\n"
        f"Open this link to add a favorite photo and the words that describe your loved one,\n"
        f"see the portrait come together, and send it to us:\n\n"
        f"    {link}\n\n"
        f"Your code works once. Please keep it private — anyone with the code can redeem it.\n\n"
        f"With care,\nLoved in Words\n"
    )
    html = f"""<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#2b2f36;max-width:520px;line-height:1.6">
<p style="font-size:30px;line-height:1;color:#c85b57;margin:0 0 6px">&#9829;</p>
<h2 style="font-family:Georgia,'Times New Roman',serif;color:#42525f;font-size:22px;margin:0 0 10px">Your keepsake is ready to personalize</h2>
<p style="margin:0 0 4px">{hi}</p>
<p style="margin:0 0 16px">Thank you. Your <b>{product_name}</b> is ready. Add a favorite photo and the words that describe your loved one, watch their portrait take shape, and send it to us.</p>
<p style="margin:0 0 6px;color:#7a756e;font-size:14px">Your redemption code</p>
<p style="font-size:22px;letter-spacing:.08em;font-weight:700;color:#42525f;background:#f4efe8;border-radius:12px;padding:14px;text-align:center;margin:0 0 18px">{code_fmt}</p>
<p style="text-align:center;margin:0 0 18px">
  <a href="{link}" style="background:#c85b57;color:#fff;padding:13px 30px;border-radius:999px;text-decoration:none;display:inline-block;font-weight:700">Personalize my keepsake</a>
</p>
<p style="color:#9a948c;font-size:12.5px;margin:0">Your code works once. Please keep it private &mdash; anyone with the code can redeem it.</p>
<p style="color:#a49e95;font-size:12px;margin:18px 0 0">With care, Loved in Words &middot; powered by Typortrait.com</p>
</div>"""
    return html, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Issue a redemption code to a buyer (generate + email).")
    ap.add_argument("sku", help="product SKU the buyer purchased (see app/products.py)")
    ap.add_argument("email", help="buyer's email address")
    ap.add_argument("--name", default="", help="buyer's first name (for the greeting)")
    ap.add_argument("--batch", help="batch label (e.g. everloved-2026-07)")
    ap.add_argument("--link-base", default=PUBLIC_BASE_URL,
                    help="base URL for the redeem link (default: TYPO_PUBLIC_URL)")
    ap.add_argument("--dry-run", action="store_true", help="print the code + email, don't send")
    args = ap.parse_args(argv)

    product = products.get(args.sku)
    if not product:
        valid = ", ".join(sorted(p.sku for p in products.CATALOG))
        print(f"error: unknown sku {args.sku!r}. Valid: {valid}", file=sys.stderr)
        return 2
    if not _EMAIL_RE.match(args.email.strip()):
        print(f"error: {args.email!r} doesn't look like an email address", file=sys.stderr)
        return 2

    redemption.init_db()
    note = f"issued to {args.email.strip()}"
    code_fmt = redemption.generate(args.sku, 1, batch=args.batch, note=note)[0]
    base = args.link_base.rstrip("/")
    # The redeem page pre-fills ?code=, so the buyer just clicks Continue.
    link = f"{base}/redeem?code={code_fmt}"
    html, text = _email_bodies(code_fmt, link, product.name, args.name.strip())
    subject = "Your keepsake is ready to personalize"

    if args.dry_run:
        print(f"[dry-run] code: {code_fmt}\n[dry-run] to: {args.email}\n[dry-run] link: {link}\n"
              f"[dry-run] subject: {subject}\n---- text body ----\n{text}")
        return 0

    ok = admin_mod.send_email(args.email.strip(), subject, html, text)
    if ok:
        print(f"issued {code_fmt} for {args.sku} -> emailed {args.email}")
        return 0
    print(f"code {code_fmt} was generated, but the email FAILED to send "
          f"(check SMTP config). Send the code + link manually:\n  {link}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
