"""Publish the Admin Studio's output into the live gallery as the Sacred Collection.

Reads tools/_gallery_out/ (produced by admin_studio.py batch) and:
  - copies previews  -> static/gallery/art/<id>.png     (public storefront thumb)
  - copies masters   -> private/gallery/<id>.png         (private, sellable print file)
  - merges a collection into static/gallery/catalog.json, grouping items into
    sections by their `category`, carrying each item's title, render words + price.

Idempotent: re-running replaces the collection (id `sacred` by default) in place and
leaves your other collections (Bible, Holidays) untouched. Nothing here deploys --
it only writes local files; you review /gallery locally, then deploy deliberately.

    python tools/gallery_publish.py                 # publish everything in _gallery_out
    python tools/gallery_publish.py --dry-run       # show what would happen, write nothing
    python tools/gallery_publish.py --collection-title "Sacred Art"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import PRIVATE_DIR, STATIC_DIR  # noqa: E402

OUT = ROOT / "tools" / "_gallery_out"
ART_DIR = STATIC_DIR / "gallery" / "art"
MASTER_DIR = PRIVATE_DIR / "gallery"
CATALOG = STATIC_DIR / "gallery" / "catalog.json"


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-") or "misc"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection-id", default="sacred")
    ap.add_argument("--collection-title", default="The Sacred Collection")
    ap.add_argument("--blurb", default="Sacred figures, each woven entirely from the words that belong to them.")
    ap.add_argument("--default-category", default="Sacred Figures")
    ap.add_argument("--all", action="store_true",
                    help="publish EVERY rendered item, ignoring the approval gate")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = OUT / "catalog.json"
    if not src.exists():
        sys.exit(f"No {src} — run an Admin Studio batch first.")
    items = json.loads(src.read_text(encoding="utf-8"))
    if not items:
        sys.exit("catalog.json is empty — nothing to publish.")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    # Approval gate: publish only items marked "approved" in the Admin Studio,
    # unless --all is passed. Keeps a rejected/pending render out of the store.
    approvals: dict = {}
    ap_file = OUT / "approvals.json"
    if ap_file.exists():
        try:
            approvals = json.loads(ap_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            approvals = {}

    # Group into sections by category, preserving first-seen order.
    sections: dict[str, dict] = {}
    copied, missing, unapproved = 0, [], []
    for it in items:
        iid = str(it.get("id") or "").strip()
        prev = OUT / "previews" / f"{iid}.png"
        mast = OUT / "masters" / f"{iid}.png"
        if not iid or not prev.exists() or not mast.exists():
            missing.append(iid or "(no id)")
            continue
        status = approvals.get(iid, "pending")
        if not a.all and status != "approved":
            unapproved.append(f"{iid} ({status})")
            continue
        if not a.dry_run:
            shutil.copyfile(prev, ART_DIR / f"{iid}.png")
            shutil.copyfile(mast, MASTER_DIR / f"{iid}.png")
        copied += 1
        cat = (it.get("category") or a.default_category).strip() or a.default_category
        sec = sections.setdefault(_slug(cat), {"id": _slug(cat), "title": cat, "items": []})
        entry = {"id": iid, "title": (it.get("title") or iid).strip(),
                 "subject": (it.get("subject") or it.get("title") or iid).strip(),
                 "words": (it.get("words") or "").strip(),
                 "art": "original (AI-generated)", "text": ""}
        price = str(it.get("price") or "").strip()
        if price:
            entry["price"] = price          # per-item price (dollars); storefront + checkout honor it
        sec["items"].append(entry)

    if copied == 0 and not a.dry_run:
        print(f"Nothing to publish -- held back {len(unapproved)} not approved, "
              f"{len(missing)} missing art. Catalog left unchanged.")
        print("  Approve items in the Admin Studio Output tab, or re-run with --all.")
        return

    collection = {"id": a.collection_id, "title": a.collection_title,
                  "blurb": a.blurb, "sections": list(sections.values())}

    cat_data = {"collections": []}
    if CATALOG.exists():
        try:
            cat_data = json.loads(CATALOG.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Existing catalog.json is invalid, refusing to overwrite: {e}")
    cols = cat_data.setdefault("collections", [])
    cols[:] = [c for c in cols if c.get("id") != a.collection_id] + [collection]

    if not a.dry_run:
        CATALOG.write_text(json.dumps(cat_data, indent=2, ensure_ascii=False), encoding="utf-8")

    tag = "DRY-RUN — nothing written\n" if a.dry_run else ""
    print(f"{tag}published {copied} items into '{a.collection_title}' "
          f"({len(sections)} sections: {', '.join(s['title'] for s in sections.values())})")
    if unapproved:
        print(f"  held back {len(unapproved)} not approved (use --all to override): {unapproved[:8]}")
    if missing:
        print(f"  skipped {len(missing)} with missing art: {missing[:12]}")
    print(f"  art     -> {ART_DIR}")
    print(f"  masters -> {MASTER_DIR}")
    print(f"  catalog -> {CATALOG}")
    if not a.dry_run:
        print("\n  Review locally at /gallery, then to go live:")
        print("   1) commit + push static/gallery/ (art + catalog.json)")
        print("   2) copy the new private/gallery/*.png masters to the server")
        print("   3) deploy (git pull on the server; static is bind-mounted)")


if __name__ == "__main__":
    main()
