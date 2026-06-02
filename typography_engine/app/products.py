"""Product catalog for the print-on-demand offering.

Single source of truth for what we sell, what it costs the customer, and which
Printful catalog variant fulfills it. Retail prices are set to give a healthy
margin over Printful's wholesale + estimated US shipping; revisit if Printful's
pricing changes.

The `printful_variant_id` values are stable IDs from Printful's catalog API
(GET /products/<product_id>). They identify the SKU+size+color combination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Product:
    sku: str                      # internal SKU (used in URLs, metadata)
    name: str                     # customer-facing name
    blurb: str                    # short marketing line shown under the name
    price_cents: int              # retail price the customer pays (excl. shipping)
    shipping_cents: int           # flat US shipping the customer pays
    physical: bool                # True if Printful-fulfilled; False = digital
    printful_variant_id: Optional[int] = None  # required if physical
    # For products that need a size variant (t-shirts), this lists the
    # selectable sizes mapped to their Printful variant IDs.
    size_variants: Optional[Dict[str, int]] = None
    # Tag shown on the product card. Optional.
    tag: Optional[str] = None


# Printful variant IDs for the t-shirt (Bella+Canvas 3001, unisex, BLACK).
# Verified against Printful's live catalog (GET /products/71) 2026-06-02.
# If a variant ID changes upstream, only this map needs updating.
_TSHIRT_SIZES: Dict[str, int] = {
    "S": 4016,
    "M": 4017,
    "L": 4018,
    "XL": 4019,
    "2XL": 4020,
}


CATALOG: List[Product] = [
    Product(
        sku="framed_16x20",
        name="16×20 framed print",
        blurb="Museum-grade matte paper, solid wood frame. Ready to hang.",
        price_cents=6900,
        shipping_cents=1500,
        physical=True,
        printful_variant_id=4399,   # Enhanced Matte Paper Framed Poster 16×20, Black frame (product 2) — verified 2026-06-02
        tag="Best gift",
    ),
    Product(
        sku="canvas_16x20",
        name="16×20 gallery canvas",
        blurb="1.25\" stretched canvas, gallery-wrapped edges.",
        price_cents=4900,
        shipping_cents=1200,
        physical=True,
        printful_variant_id=6,      # Canvas (in) 16×20, 1.25" thick (product 3) — verified 2026-06-02
    ),
    Product(
        sku="poster_18x24",
        name="18×24 archival poster",
        blurb="Heavy matte paper, true colors. Frame separately.",
        price_cents=2900,
        shipping_cents=800,
        physical=True,
        printful_variant_id=1,      # Enhanced Matte Paper Poster (in) 18×24 (product 1) — verified 2026-06-02
    ),
    Product(
        sku="tshirt_unisex",
        name="Unisex t-shirt",
        blurb="Bella+Canvas 3001, soft cotton, front print.",
        price_cents=3400,
        shipping_cents=500,
        physical=True,
        size_variants=_TSHIRT_SIZES,
    ),
    Product(
        sku="digital",
        name="High-res digital download",
        blurb="Full-resolution PNG, no watermark. Print anywhere.",
        price_cents=1499,
        shipping_cents=0,
        physical=False,
    ),
]


_BY_SKU: Dict[str, Product] = {p.sku: p for p in CATALOG}


def get(sku: str) -> Optional[Product]:
    return _BY_SKU.get(sku)


def resolve_variant_id(product: Product, size: Optional[str]) -> Optional[int]:
    """Return the Printful variant ID for a product + optional size choice."""
    if product.size_variants:
        if size and size in product.size_variants:
            return product.size_variants[size]
        return None
    return product.printful_variant_id


def public_catalog() -> List[dict]:
    """JSON-safe catalog payload for the storefront UI."""
    out = []
    for p in CATALOG:
        out.append({
            "sku": p.sku,
            "name": p.name,
            "blurb": p.blurb,
            "price_cents": p.price_cents,
            "shipping_cents": p.shipping_cents,
            "physical": p.physical,
            "sizes": list(p.size_variants.keys()) if p.size_variants else [],
            "tag": p.tag,
        })
    return out
