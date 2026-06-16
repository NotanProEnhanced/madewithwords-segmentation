"""Upstream dependency health checks (Stripe, Printful).

The basic /health endpoint reports local capabilities only. This adds a deeper,
network check of the two external services that take payment and fulfil orders,
so a bad/expired key or an upstream outage is visible to monitoring BEFORE a
customer hits it at checkout.

Each check is best-effort, short-timeout, and cached briefly so polling the
health endpoint can't hammer the upstream APIs. Responses expose only coarse
status (ok / latency / reason) -- never keys, account ids, or store names -- so
the endpoint stays safe to leave unauthenticated for uptime monitors.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import httpx

from .config import (
    PRINTFUL_API_BASE,
    PRINTFUL_API_TOKEN,
    PRINTFUL_STORE_ID,
    STRIPE_SECRET_KEY,
)

_TIMEOUT = 8.0
_CACHE_TTL = 60.0
_cache: Dict[str, tuple] = {}   # name -> (expires_ts, result)


def _cached(name: str, fn: Callable[[], dict]) -> dict:
    now = time.time()
    hit = _cache.get(name)
    if hit and hit[0] > now:
        return {**hit[1], "cached": True}
    res = fn()
    _cache[name] = (now + _CACHE_TTL, res)
    return res


def _reason(status: int) -> str:
    if status in (401, 403):
        return "auth_failed"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "upstream_error"
    return f"http_{status}"


def _check_stripe() -> dict:
    if not STRIPE_SECRET_KEY:
        return {"configured": False, "ok": None, "detail": "STRIPE_SECRET_KEY not set"}
    mode = "test" if STRIPE_SECRET_KEY.startswith("sk_test") else "live"
    t0 = time.time()
    try:
        r = httpx.get("https://api.stripe.com/v1/account",
                      headers={"Authorization": f"Bearer {STRIPE_SECRET_KEY}"},
                      timeout=_TIMEOUT)
    except httpx.HTTPError as e:
        return {"configured": True, "ok": False, "mode": mode, "detail": f"network_error: {type(e).__name__}"}
    ms = round((time.time() - t0) * 1000)
    if r.status_code == 200:
        return {"configured": True, "ok": True, "mode": mode, "latency_ms": ms}
    return {"configured": True, "ok": False, "mode": mode, "latency_ms": ms,
            "status": r.status_code, "detail": _reason(r.status_code)}


def _check_printful() -> dict:
    if not PRINTFUL_API_TOKEN:
        return {"configured": False, "ok": None, "detail": "PRINTFUL_API_TOKEN not set (digital-only mode)"}
    headers = {"Authorization": f"Bearer {PRINTFUL_API_TOKEN}", "Accept": "application/json"}
    if PRINTFUL_STORE_ID:
        headers["X-PF-Store-Id"] = str(PRINTFUL_STORE_ID)
    t0 = time.time()
    try:
        r = httpx.get(f"{PRINTFUL_API_BASE.rstrip('/')}/store", headers=headers, timeout=_TIMEOUT)
    except httpx.HTTPError as e:
        return {"configured": True, "ok": False, "detail": f"network_error: {type(e).__name__}"}
    ms = round((time.time() - t0) * 1000)
    if r.status_code == 200:
        return {"configured": True, "ok": True, "latency_ms": ms}
    return {"configured": True, "ok": False, "latency_ms": ms,
            "status": r.status_code, "detail": _reason(r.status_code)}


def check_stripe() -> dict:
    return _cached("stripe", _check_stripe)


def check_printful() -> dict:
    return _cached("printful", _check_printful)


def check_all() -> dict:
    """{ok, stripe, printful}. `ok` is True iff every CONFIGURED upstream is
    healthy; an unconfigured upstream (e.g. Printful in digital-only mode) does
    not fail the overall check."""
    stripe_r = check_stripe()
    printful_r = check_printful()
    configured = [r for r in (stripe_r, printful_r) if r.get("configured")]
    overall = all(r.get("ok") for r in configured) if configured else True
    return {"ok": overall, "stripe": stripe_r, "printful": printful_r}
