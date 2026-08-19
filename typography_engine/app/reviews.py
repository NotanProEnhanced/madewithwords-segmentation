"""Customer review / testimonial store.

For a memorial brand, other families' words are the marketing -- so capturing a
heartfelt reaction the moment it happens, and being able to feature it later, is
the highest-leverage growth loop. This is the durable store behind that.

Reviews are captured as PENDING and never shown publicly until a human approves
them (a grief brand must moderate -- spam, or an inappropriate entry, next to a
customer's late mother is unacceptable). Featuring is a second, explicit flag.

Filesystem-backed under PRIVATE_DIR so it survives restarts and is trivially
moderated by hand if needed. Pure stdlib; imports nothing app-specific.

Layout (under the base dir):
    <id>.json   one review: rating, text, name, consent, status, timestamps
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class ReviewStore:
    def __init__(self, base_dir: str):
        self.base = base_dir
        os.makedirs(self.base, exist_ok=True)

    def _path(self, rid: str) -> str:
        return os.path.join(self.base, rid + ".json")

    def _new_id(self) -> str:
        return "%d-%s" % (int(time.time()), os.urandom(4).hex())

    def submit(self, *, job: str = "", rating: int = 0, text: str = "",
               name: str = "", brand: str = "", consent: bool = False,
               email: str = "") -> str:
        """Persist a new review as PENDING. Returns the review id."""
        rid = self._new_id()
        rec = {
            "id": rid,
            "job": (job or "")[:64],
            "rating": max(0, min(5, int(rating or 0))),
            "text": (text or "").strip()[:2000],
            "name": (name or "").strip()[:120],
            "brand": (brand or "").strip()[:40],
            "email": (email or "").strip()[:200],
            "consent": bool(consent),          # may we feature it publicly?
            "status": "pending",               # pending | approved | hidden
            "featured": False,                 # show on the landing / marketing
            "created_at": int(time.time()),
        }
        tmp = self._path(rid) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, self._path(rid))
        return rid

    def _read(self, fn: str) -> Optional[Dict[str, Any]]:
        try:
            with open(os.path.join(self.base, fn)) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return None

    def list_all(self) -> List[Dict[str, Any]]:
        """Every review, newest first -- for moderation."""
        out = []
        try:
            names = [n for n in os.listdir(self.base) if n.endswith(".json")]
        except FileNotFoundError:
            return []
        for n in names:
            r = self._read(n)
            if r:
                out.append(r)
        out.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return out

    def list_public(self, brand: str = "", limit: int = 24) -> List[Dict[str, Any]]:
        """Approved + featured + consented reviews for public display."""
        rows = [r for r in self.list_all()
                if r.get("status") == "approved" and r.get("featured") and r.get("consent")]
        if brand:
            rows = [r for r in rows if not r.get("brand") or r.get("brand") == brand]
        return rows[:max(0, limit)]

    def set_status(self, rid: str, status: str = None, featured: bool = None) -> bool:
        """Moderator action: approve/hide and/or feature a review."""
        rec = self._read(rid + ".json")
        if not rec:
            return False
        if status in ("pending", "approved", "hidden"):
            rec["status"] = status
        if featured is not None:
            rec["featured"] = bool(featured)
        rec["moderated_at"] = int(time.time())
        tmp = self._path(rid) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.rename(tmp, self._path(rid))
        return True

    def stats(self) -> Dict[str, Any]:
        rows = self.list_all()
        rated = [r["rating"] for r in rows if r.get("rating")]
        return {
            "total": len(rows),
            "pending": sum(1 for r in rows if r.get("status") == "pending"),
            "approved": sum(1 for r in rows if r.get("status") == "approved"),
            "featured": sum(1 for r in rows if r.get("featured")),
            "avg_rating": round(sum(rated) / len(rated), 2) if rated else None,
        }
