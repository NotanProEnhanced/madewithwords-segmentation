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
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Product:
    sku: str                      # internal SKU (used in URLs, metadata)
    name: str                     # customer-facing name
    blurb: str                    # short marketing line shown under the name
    price_cents: int              # retail price the customer pays (excl. shipping)
    shipping_cents: int           # flat US shipping the customer pays
    physical: bool                # True if Printful-fulfilled; False = digital
    # Print canvas aspect (width / height) the fulfilment file is rendered to, so
    # the art matches the physical size with the face centred and ground-padded
    # (never cropped or stretched). 4:5 = 0.8 for 16x20 / digital; 18x24 = 0.75.
    print_aspect: float = 0.8
    printful_variant_id: Optional[int] = None  # required if physical
    # For products that need a size variant (t-shirts), this lists the
    # selectable sizes mapped to their Printful variant IDs.
    size_variants: Optional[Dict[str, int]] = None
    # Tag shown on the product card. Optional.
    tag: Optional[str] = None
    # Bundle: several Printful line items in ONE order, as (variant_id, qty). When
    # set, printful_variant_id is None and fulfilment expands these. All items must
    # share one print aspect, so a single render fulfils the whole order.
    bundle_items: Optional[List[Tuple[int, int]]] = None
    # Show only on the memorial (lovedinwords) brand -- e.g. grief-framed bundles.
    memorial_only: bool = False


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
        print_aspect=0.75,          # 18/24 = 3:4, not 4:5 -- render the print file to match
        printful_variant_id=1,      # Enhanced Matte Paper Poster (in) 18×24 (product 1) — verified 2026-06-02
    ),
    # --- Smaller, affordable sizes: a print everyone who loved them can keep.
    # Variant IDs + base costs verified 2026-06-22 via the Printful catalog API.
    Product(
        sku="framed_8x10",
        name="8×10 framed keepsake",
        blurb="A smaller framed portrait — for a desk, a shelf, or to give.",
        price_cents=3900,
        shipping_cents=900,
        physical=True,
        printful_variant_id=4651,   # Enhanced Matte Framed Poster 8×10, Black (product 2) — base $20.35
        tag="For the family",
    ),
    Product(
        sku="canvas_8x10",
        name="8×10 gallery canvas",
        blurb="A small gallery-wrapped canvas — ready to stand or hang.",
        price_cents=3400,
        shipping_cents=900,
        physical=True,
        printful_variant_id=19293,  # Canvas (in) 8×10 (product 3) — base $15.81
    ),
    Product(
        sku="poster_8x10",
        name="8×10 archival print",
        blurb="Heavy matte paper, desk-sized. Frame separately.",
        price_cents=1900,
        shipping_cents=600,
        physical=True,
        printful_variant_id=4463,   # Enhanced Matte Paper Poster (in) 8×10 (product 1) — base $6.89
    ),
    Product(
        sku="print_5x7",
        name="5×7 keepsake print",
        blurb="A small print — for a nightstand, a desk, or to share.",
        price_cents=1500,
        shipping_cents=500,
        physical=True,
        print_aspect=0.714,         # 5/7 ≈ 0.714 -- render the print file to match
        printful_variant_id=16364,  # Enhanced Matte Paper Poster (in) 5×7 (product 1) — base $5.39
    ),
    # --- Bundles (memorial brand only). Single-aspect, so one render fulfils every
    # item; variant IDs are the same verified Printful catalog IDs used above.
    Product(
        sku="bundle_family",
        name="“Keep them close” — family set",
        blurb="One framed portrait for the home, and four prints for those who loved them.",
        price_cents=14900,
        shipping_cents=2100,                   # Printful's real flat rate for this 5-item set is $20.18
        physical=True,
        printful_variant_id=None,
        bundle_items=[(4399, 1), (4463, 4)],   # 16×20 framed + 4× 8×10 poster (all 4:5)
        memorial_only=True,
        tag="Most loved",
    ),
    Product(
        sku="bundle_share6",
        name="“A copy for everyone” — six 5×7 prints",
        blurb="Six small keepsakes to give to the family and friends who loved them.",
        price_cents=6700,
        shipping_cents=1400,
        physical=True,
        print_aspect=0.714,
        printful_variant_id=None,
        bundle_items=[(16364, 6)],             # 6× 5×7 (all 0.714)
        memorial_only=True,
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


# Memorial (EverLoved) channel price overrides (cents). Higher than the standard
# price to absorb the 20-30% partner commission; SHIPPING is unchanged. SKUs not
# listed keep their standard price. Bundles are memorial-only -> priced on the Product.
_MEMORIAL_PRICE: Dict[str, int] = {
    "framed_16x20": 8900,
    "canvas_16x20": 6400,
    "poster_18x24": 3400,
    "framed_8x10":  4900,
    "canvas_8x10":  3900,
    "poster_8x10":  2300,
    "print_5x7":    1600,
}
_MEMORIAL_REFS = {"lovedinwords", "everloved", "keeper"}


def is_memorial_channel(ref: Optional[str]) -> bool:
    """True for the EverLoved/memorial brand, where the commission-adjusted prices apply."""
    return (ref or "").strip().lower() in _MEMORIAL_REFS


def price_for(p: Product, memorial: bool) -> Tuple[int, int]:
    """(price_cents, shipping_cents) for the channel. Memorial uses the higher override
    where one exists; shipping is identical on every channel."""
    if memorial and p.sku in _MEMORIAL_PRICE:
        return _MEMORIAL_PRICE[p.sku], p.shipping_cents
    return p.price_cents, p.shipping_cents


def public_catalog(memorial: bool = False) -> List[dict]:
    """JSON-safe catalog payload for the storefront UI (brand-priced when memorial)."""
    out = []
    for p in CATALOG:
        price, ship = price_for(p, memorial)
        out.append({
            "sku": p.sku,
            "name": p.name,
            "blurb": p.blurb,
            "price_cents": price,
            "shipping_cents": ship,
            "physical": p.physical,
            "sizes": list(p.size_variants.keys()) if p.size_variants else [],
            "tag": p.tag,
            "memorial_only": p.memorial_only,
        })
    return out
