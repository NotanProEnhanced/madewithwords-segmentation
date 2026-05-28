"""Minimal Printful REST client for order fulfillment.

Uses Printful's v1 API (https://developers.printful.com/docs/) — stable and
well-documented. v2 is available but offers no advantage for our flow.

Auth: personal access token from https://developers.printful.com/ →
`Authorization: Bearer <token>`. The token is store-scoped, so we pass the
store ID in `X-PF-Store-Id` to make orders go to the right store when an
account has more than one.

The art file is referenced by a public URL that Printful fetches. We expose
a signed, time-limited URL (`/printful-fetch/{job}?...`) so the clean PNG
remains otherwise private. See `signed_print_url()`.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional

import httpx

from .config import (
    PRINT_URL_SECRET,
    PRINTFUL_API_BASE,
    PRINTFUL_API_TOKEN,
    PRINTFUL_STORE_ID,
    PUBLIC_BASE_URL,
)


class PrintfulError(Exception):
    def __init__(self, message: str, status: int = 0, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


def _headers() -> Dict[str, str]:
    if not PRINTFUL_API_TOKEN:
        raise PrintfulError("PRINTFUL_API_TOKEN not configured")
    h = {
        "Authorization": f"Bearer {PRINTFUL_API_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if PRINTFUL_STORE_ID:
        h["X-PF-Store-Id"] = str(PRINTFUL_STORE_ID)
    return h


def _request(method: str, path: str, *, json: Any = None, timeout: float = 30.0) -> Dict[str, Any]:
    url = f"{PRINTFUL_API_BASE.rstrip('/')}{path}"
    try:
        r = httpx.request(method, url, headers=_headers(), json=json, timeout=timeout)
    except httpx.HTTPError as e:
        raise PrintfulError(f"network_error: {e}") from e
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text}
    if r.status_code >= 400:
        msg = (body.get("error") or {}).get("message") if isinstance(body, dict) else None
        raise PrintfulError(msg or f"http_{r.status_code}", status=r.status_code, payload=body)
    if isinstance(body, dict):
        return body.get("result", body)
    return {"result": body}


# ---------- signed print URLs ------------------------------------------------

def signed_print_url(job: str, ttl_seconds: int = 86_400) -> str:
    """Return a public URL Printful can fetch the clean PNG from.

    The URL embeds an HMAC signature + expiry. Without the secret an attacker
    cannot forge a working URL; with a valid signature anyone can fetch the
    art (so don't share these URLs publicly — they go straight to Printful).
    """
    expires = int(time.time()) + ttl_seconds
    payload = f"{job}:{expires}".encode()
    sig = hmac.new(PRINT_URL_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    base = PUBLIC_BASE_URL.rstrip("/")
    return f"{base}/printful-fetch/{job}?exp={expires}&sig={sig}"


def verify_signed_url(job: str, exp: int, sig: str) -> bool:
    if exp < int(time.time()):
        return False
    payload = f"{job}:{exp}".encode()
    expected = hmac.new(PRINT_URL_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


# ---------- orders -----------------------------------------------------------

def create_order(
    *,
    recipient: Dict[str, Any],
    variant_id: int,
    print_file_url: str,
    quantity: int = 1,
    external_id: Optional[str] = None,
    retail_price_cents: Optional[int] = None,
    confirm: bool = True,
    placement: str = "default",
) -> Dict[str, Any]:
    """Create a Printful order (and confirm it for fulfillment by default).

    `recipient` follows Printful's address schema: name, address1, city,
    state_code, country_code, zip, email, phone (optional).
    `placement` is "default" for posters/canvas/framed, "front" for apparel.
    """
    item: Dict[str, Any] = {
        "variant_id": variant_id,
        "quantity": quantity,
        "files": [{"url": print_file_url, "type": placement}],
    }
    if retail_price_cents is not None:
        item["retail_price"] = f"{retail_price_cents / 100:.2f}"
    payload: Dict[str, Any] = {
        "recipient": recipient,
        "items": [item],
    }
    if external_id:
        payload["external_id"] = external_id
    qs = "?confirm=true" if confirm else ""
    return _request("POST", f"/orders{qs}", json=payload)


def get_order(order_id: int) -> Dict[str, Any]:
    return _request("GET", f"/orders/{order_id}")


def list_orders(limit: int = 20) -> List[Dict[str, Any]]:
    res = _request("GET", f"/orders?limit={int(limit)}")
    if isinstance(res, list):
        return res
    return res.get("items", []) if isinstance(res, dict) else []


def shipping_rates(
    *,
    recipient: Dict[str, Any],
    variant_id: int,
    quantity: int = 1,
) -> List[Dict[str, Any]]:
    """Live shipping quotes for a recipient + product. Returns a list of
    {id, name, rate, currency, minDeliveryDays, maxDeliveryDays}.

    We don't use this in the live checkout path (we use flat rates from
    products.py for predictable pricing), but it's useful for admin tooling
    and for sanity-checking the flat rates against real costs.
    """
    payload = {
        "recipient": recipient,
        "items": [{"variant_id": variant_id, "quantity": quantity}],
    }
    res = _request("POST", "/shipping/rates", json=payload)
    if isinstance(res, list):
        return res
    return res.get("items", []) if isinstance(res, dict) else []
