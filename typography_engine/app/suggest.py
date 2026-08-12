"""Word suggestion for the memorial "Share their story" step.

A grieving person shouldn't have to stare at an empty "type words, separate with
commas" box. Instead they paste an obituary, eulogy, or a few memories, and we
suggest meaningful words (the person's name, their roles, their qualities) which
they then edit before previewing.

Design choices mirror moderation.py:
- FAIL SOFT: no key / API error / timeout -> return an empty list. The UI falls
  back to manual entry, so suggestion is a convenience, never a hard dependency.
- CACHED by text hash: the same pasted story may be re-submitted (back/forward).
- Only the suggested words leave this module, never the raw story.

NOTE: unlike moderation, chat completions are BILLED. The call is tiny
(gpt-4o-mini, story capped, ~120 output tokens) and the key's usage cap bounds it.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Dict, List

import httpx

from .config import OPENAI_API_KEY, SUGGEST_MODEL

_TIMEOUT = 20.0
_CACHE_TTL = 3600.0
_CACHE_MAX = 500
_STORY_MAX_CHARS = 4000      # cap input to bound cost
_MAX_WORDS = 12
_WORD_MAX_LEN = 22
_cache: Dict[str, tuple] = {}   # sha256(story) -> (expires_ts, words)

_DETAIL_MAX = 200            # cap each structured detail field
_SYSTEM = (
    "You help build a memorial word-portrait: an image of a person made entirely "
    "of words that describe them. From what the family shares -- a tribute, an "
    "obituary, memories, or a few details -- extract the most meaningful words to "
    "picture them by: their first name or nickname, their roles (MOTHER, "
    "GRANDFATHER, TEACHER, VETERAN), and their qualities (KIND, FAITHFUL, FUNNY, "
    "GENEROUS). Turn a hobby or passion into a word (gardening -> GARDENER, fishing "
    "-> FISHERMAN, music -> MUSIC). If they give a nickname or three chosen words, be "
    "sure to include them. Prefer single words; two-word phrases only when essential. "
    "Return ONLY a JSON array of 6 to 12 UPPERCASE strings, most important first. "
    "No commentary, no markdown, no punctuation inside the words. "
    'Example: ["GRACE","MOTHER","GRANDMOTHER","KIND","FAITHFUL","GARDENER"]'
)

# Structured detail fields the studio may add alongside the free-text story.
_DETAIL_LABELS = (
    ("nickname", "What people called them (names/nicknames)"),
    ("role", "Who they were to their family (roles/relationships)"),
    ("three_words", "A few words that described them"),
    ("hobby", "What they loved (hobbies, passions, teams, faith)"),
    ("sayings", "Something they always said"),
    ("known", "What they were known for"),
    ("legacy", "What they left the family with (values they passed on)"),
)


def _compose(story: str, details: dict) -> str:
    """Merge the free-text story with any structured details into one prompt body."""
    parts: List[str] = []
    story = (story or "").strip()[:_STORY_MAX_CHARS]
    if story:
        parts.append(story)
    hints: List[str] = []
    for key, label in _DETAIL_LABELS:
        v = str((details or {}).get(key, "") or "").strip()[:_DETAIL_MAX]
        if v:
            hints.append(f"{label}: {v}")
    if hints:
        parts.append("Details the family added:\n" + "\n".join(hints))
    return "\n\n".join(parts).strip()


def enabled() -> bool:
    return bool(OPENAI_API_KEY)


def _clean_word(w: str) -> str:
    # Keep letters, spaces, hyphens, apostrophes; collapse whitespace; uppercase.
    w = re.sub(r"[^A-Za-z'\- ]+", "", str(w)).strip()
    w = re.sub(r"\s+", " ", w)
    return w.upper()[:_WORD_MAX_LEN].strip()


def _parse_words(content: str) -> List[str]:
    """Pull a list of words out of the model's reply, tolerant of stray prose or
    code fences around the JSON array."""
    raw: List = []
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                raw = parsed
        except (ValueError, TypeError):
            raw = []
    if not raw:
        # Fallback: comma / newline separated.
        raw = re.split(r"[,\n]+", content)

    out: List[str] = []
    seen = set()
    for item in raw:
        w = _clean_word(item)
        if not w or len(w) < 2:
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= _MAX_WORDS:
            break
    return out


def suggest_words(story: str, details: dict | None = None) -> List[str]:
    """Return a list of suggested UPPERCASE words from a pasted tribute and/or the
    optional structured details. Returns [] on no key / error / empty input -- the
    caller falls back to manual entry."""
    combined = _compose(story, details or {})
    if not combined or not OPENAI_API_KEY:
        return []

    h = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    now = time.time()
    hit = _cache.get(h)
    if hit and hit[0] > now:
        return list(hit[1])

    words: List[str] = []
    try:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": SUGGEST_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": combined},
                ],
                "temperature": 0.3,
                "max_tokens": 160,
            },
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            content = (
                (r.json().get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            words = _parse_words(content or "")
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        words = []  # fail soft -> manual entry

    if len(_cache) > _CACHE_MAX:
        _cache.clear()
    _cache[h] = (now + _CACHE_TTL, words)
    return words
