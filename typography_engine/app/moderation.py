"""Text content moderation for the user's words/message.

The words a user types BECOME the portrait, so a "portrait composed of slurs / a
hateful message" is a direct abuse vector. We screen that text with OpenAI's free
moderation endpoint (hate, sexual, harassment, violence, self-harm, ...).

Design choices:
- FAIL OPEN: if no key is configured or the API errors/times out, we ALLOW the
  render. Moderation must never take the paid product down; the Terms +
  attestation are the backstop, and only a *definitive* flag blocks.
- CACHED by text hash: the same words are re-rendered many times per session
  (swatches, reco, thumbnails), so we check once and reuse the verdict.
- Result exposes only categories, never the text.
"""
from __future__ import annotations

import hashlib
import time
from typing import Dict

import httpx

from .config import MODERATION_MODEL, OPENAI_API_KEY

_TIMEOUT = 6.0
_CACHE_TTL = 3600.0
_CACHE_MAX = 2000
_cache: Dict[str, tuple] = {}   # sha256(text) -> (expires_ts, result)


def enabled() -> bool:
    return bool(OPENAI_API_KEY)


def check_text(text: str) -> dict:
    """Return {flagged, checked, categories?, reason?}. flagged=True means BLOCK.
    checked=False means we couldn't moderate (no key / error) and allowed it."""
    text = (text or "").strip()
    if not text:
        return {"flagged": False, "checked": False, "reason": "empty"}
    if not OPENAI_API_KEY:
        return {"flagged": False, "checked": False, "reason": "not_configured"}

    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    now = time.time()
    hit = _cache.get(h)
    if hit and hit[0] > now:
        return hit[1]

    try:
        r = httpx.post(
            "https://api.openai.com/v1/moderations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": MODERATION_MODEL, "input": text},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            res = {"flagged": False, "checked": False, "reason": f"http_{r.status_code}"}
        else:
            data = (r.json().get("results") or [{}])[0]
            cats = sorted(k for k, v in (data.get("categories") or {}).items() if v)
            res = {"flagged": bool(data.get("flagged")), "checked": True, "categories": cats}
    except httpx.HTTPError as e:  # network / timeout -> fail open
        res = {"flagged": False, "checked": False, "reason": f"network_error: {type(e).__name__}"}

    if len(_cache) > _CACHE_MAX:
        _cache.clear()
    _cache[h] = (now + _CACHE_TTL, res)
    return res
