"""Tests for the Printful POD layer (Phase E).

Stripe / Printful HTTP calls are NOT exercised — these tests cover the
deterministic local pieces: catalog correctness, signed-URL signing,
order DB transitions (esp. idempotency), and the storefront endpoint.

Run with:  cd typography_engine && python -m pytest -q tests/test_phase_e.py
"""
from __future__ import annotations

import time
import uuid

import pytest
from starlette.testclient import TestClient

from app import orders as orders_db
from app import printful, products
from app.main import app


client = TestClient(app)


# ---------- product catalog -------------------------------------------------

def test_catalog_has_expected_skus():
    skus = {p.sku for p in products.CATALOG}
    # Phase E launch set: 4 physical + 1 digital.
    assert {"framed_16x20", "canvas_16x20", "poster_18x24", "tshirt_unisex", "digital"} <= skus


def test_resolve_variant_id_digital_is_none():
    p = products.get("digital")
    assert p is not None
    assert products.resolve_variant_id(p, None) is None


def test_resolve_variant_id_sized_product_requires_size():
    p = products.get("tshirt_unisex")
    assert p is not None and p.size_variants
    assert products.resolve_variant_id(p, None) is None
    assert products.resolve_variant_id(p, "bogus") is None
    a_size = next(iter(p.size_variants))
    assert products.resolve_variant_id(p, a_size) == p.size_variants[a_size]


def test_resolve_variant_id_fixed_product_returns_id():
    p = products.get("poster_18x24")
    assert p is not None
    assert products.resolve_variant_id(p, None) == p.printful_variant_id


def test_public_catalog_payload_is_json_safe():
    payload = products.public_catalog()
    # Must be a list of plain dicts the JSON encoder can handle.
    assert isinstance(payload, list) and payload
    sample = payload[0]
    for key in ("sku", "name", "blurb", "price_cents", "physical", "sizes"):
        assert key in sample


def test_products_endpoint_returns_catalog():
    r = client.get("/products")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["currency"]
    assert any(p["sku"] == "digital" for p in body["products"])
    assert any(p["sku"] == "framed_16x20" for p in body["products"])


# ---------- signed print URLs ----------------------------------------------

def test_signed_url_round_trip():
    url = printful.signed_print_url("abc123def456", ttl_seconds=120)
    assert "exp=" in url and "sig=" in url
    # Pull exp + sig back out
    qs = url.split("?", 1)[1]
    parts = dict(kv.split("=", 1) for kv in qs.split("&"))
    assert printful.verify_signed_url("abc123def456", int(parts["exp"]), parts["sig"]) is True


def test_signed_url_rejects_tampered_job():
    url = printful.signed_print_url("orig", ttl_seconds=120)
    qs = url.split("?", 1)[1]
    parts = dict(kv.split("=", 1) for kv in qs.split("&"))
    assert printful.verify_signed_url("evil", int(parts["exp"]), parts["sig"]) is False


def test_signed_url_rejects_expired():
    url = printful.signed_print_url("abc", ttl_seconds=-5)
    qs = url.split("?", 1)[1]
    parts = dict(kv.split("=", 1) for kv in qs.split("&"))
    assert printful.verify_signed_url("abc", int(parts["exp"]), parts["sig"]) is False


def test_printful_fetch_rejects_bad_signature():
    r = client.get("/printful-fetch/abc?exp=9999999999&sig=deadbeef")
    assert r.status_code == 403


# ---------- orders DB ------------------------------------------------------

def _new_order_args(**overrides):
    base = dict(
        order_id=uuid.uuid4().hex[:16],
        stripe_session_id=f"cs_test_{uuid.uuid4().hex[:24]}",
        job_id="job_" + uuid.uuid4().hex[:8],
        sku="poster_18x24",
        size=None,
        variant_id=4,
        price_cents=2900,
        shipping_cents=800,
        currency="usd",
    )
    base.update(overrides)
    return base


def test_orders_create_and_fetch():
    args = _new_order_args()
    orders_db.create_pending(**args)
    row = orders_db.get(args["order_id"])
    assert row is not None
    assert row["status"] == "pending_payment"
    assert row["sku"] == "poster_18x24"
    assert row["price_cents"] == 2900


def test_mark_paid_is_idempotent():
    args = _new_order_args()
    orders_db.create_pending(**args)
    recipient = {"name": "Jane Doe", "address1": "1 Main", "city": "X",
                 "state_code": "CA", "country_code": "US", "zip": "94000"}
    first = orders_db.mark_paid(
        stripe_session_id=args["stripe_session_id"],
        payment_intent="pi_1",
        customer_email="jane@example.com",
        recipient=recipient,
    )
    assert first is not None
    assert first["status"] == "paid"
    # Second call must NOT re-transition (so Printful order isn't created twice).
    second = orders_db.mark_paid(
        stripe_session_id=args["stripe_session_id"],
        payment_intent="pi_1",
        customer_email="jane@example.com",
        recipient=recipient,
    )
    assert second is None


def test_mark_paid_unknown_session_returns_none():
    out = orders_db.mark_paid(
        stripe_session_id="cs_test_does_not_exist",
        payment_intent=None,
        customer_email=None,
        recipient=None,
    )
    assert out is None


def test_mark_fulfilling_then_shipped():
    args = _new_order_args()
    orders_db.create_pending(**args)
    orders_db.mark_paid(
        stripe_session_id=args["stripe_session_id"],
        payment_intent="pi_2", customer_email="a@b.co",
        recipient={"name": "A", "address1": "X", "city": "Y",
                   "state_code": "CA", "country_code": "US", "zip": "94000"},
    )
    orders_db.mark_fulfilling(order_id=args["order_id"], printful_order_id=12345,
                              raw={"id": 12345})
    row = orders_db.get(args["order_id"])
    assert row["status"] == "fulfilling"
    assert row["printful_order_id"] == 12345
    orders_db.mark_shipped(order_id=args["order_id"],
                           tracking_url="https://example.com/track/abc",
                           raw={"shipments": [{}]})
    row = orders_db.get(args["order_id"])
    assert row["status"] == "shipped"
    assert row["tracking_url"].startswith("https://")


# ---------- checkout endpoint guards ---------------------------------------

def test_checkout_unknown_product_400():
    r = client.post("/checkout", data={"job": "anyjob", "sku": "nonsense"})
    # Could be 400 (unknown_product) OR 503 (payments_unconfigured) depending
    # on whether Stripe is set in this test env. Either is "correctly rejected".
    assert r.status_code in (400, 503)


def test_checkout_unknown_job_404():
    # Skip if Stripe isn't configured: endpoint short-circuits with 503 first.
    from app.config import STRIPE_SECRET_KEY
    if not STRIPE_SECRET_KEY:
        pytest.skip("Stripe not configured in this test environment.")
    r = client.post("/checkout", data={"job": "nonexistent_job_id", "sku": "digital"})
    assert r.status_code == 404
