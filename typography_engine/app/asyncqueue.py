"""Filesystem-backed render job queue (cross-process, restart-safe).

The render tier is CPU-bound and tops out at a fixed renders/min. Under a traffic
spike we don't want visitors holding a connection while they wait -- we want to
ACCEPT the job instantly, render it in the background as capacity frees, and
deliver by email. This module is the durable queue behind that flow.

Why filesystem (not an in-process asyncio queue): with multiple uvicorn worker
PROCESSES an in-process queue isn't shared, and a restart would drop jobs. A
directory on the bind-mounted PRIVATE_DIR is shared by every worker and survives
restarts. Claims are atomic via os.rename (two workers race; exactly one wins),
so no locks or extra dependencies are needed.

Layout (under the queue base dir):
    blobs/<id>.bin       the source image bytes (written BEFORE the json)
    pending/<id>.json    queued job metadata  (presence here == "waiting")
    working/<id>.json    claimed by a worker
    done/<id>.json       rendered + delivered
    failed/<id>.json     gave up (metadata carries the error)

Job ids are `<seconds>-<rand>` so a plain filename sort is FIFO by enqueue time.
Nothing here imports the render pipeline or models -- it's pure stdlib, unit-
testable on its own, and safe to import anywhere.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_SUBDIRS = ("blobs", "pending", "working", "done", "failed")


@dataclass
class Job:
    id: str
    meta: Dict[str, Any]
    blob_path: str          # path to the source image bytes
    working_path: str       # path to the claimed json (in working/)


class RenderQueue:
    def __init__(self, base_dir: str):
        self.base = base_dir
        for sub in _SUBDIRS:
            os.makedirs(os.path.join(self.base, sub), exist_ok=True)

    # -- paths -------------------------------------------------------------
    def _p(self, sub: str, name: str) -> str:
        return os.path.join(self.base, sub, name)

    def _new_id(self) -> str:
        # seconds-resolution prefix keeps the filename sort FIFO; the random
        # suffix avoids collisions within the same second across processes.
        return "%d-%s" % (int(time.time()), os.urandom(4).hex())

    # -- producer ----------------------------------------------------------
    def enqueue(self, meta: Dict[str, Any], image: bytes) -> str:
        """Persist a job and mark it pending. The blob is written and fsync-renamed
        into place BEFORE the pending json appears, so a claimer never sees a job
        whose image isn't fully on disk."""
        jid = self._new_id()
        meta = dict(meta)
        meta.setdefault("id", jid)
        meta.setdefault("enqueued_at", int(time.time()))
        # blob first (atomic rename from a tmp in the same dir -> no partial reads)
        blob = self._p("blobs", jid + ".bin")
        tmp_blob = blob + ".tmp"
        with open(tmp_blob, "wb") as f:
            f.write(image)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_blob, blob)
        # then the pending json (its appearance == the job is ready to claim)
        pend = self._p("pending", jid + ".json")
        tmp_json = pend + ".tmp"
        with open(tmp_json, "w") as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_json, pend)
        return jid

    # -- consumer ----------------------------------------------------------
    def claim(self) -> Optional[Job]:
        """Atomically take the oldest pending job, or None if the queue is empty.
        Safe across processes: the os.rename either succeeds (we own it) or raises
        FileNotFoundError (another worker beat us -> try the next)."""
        try:
            names = sorted(n for n in os.listdir(self._p("pending", "")) if n.endswith(".json"))
        except FileNotFoundError:
            return None
        for name in names:
            src = self._p("pending", name)
            dst = self._p("working", name)
            try:
                os.rename(src, dst)          # atomic claim
            except (FileNotFoundError, OSError):
                continue                     # lost the race -> next candidate
            jid = name[:-5]
            try:
                with open(dst) as f:
                    meta = json.load(f)
            except Exception:                # noqa: BLE001  corrupt -> fail it, keep draining
                self._move(dst, "failed", {"error": "unreadable_meta"})
                continue
            return Job(id=jid, meta=meta, blob_path=self._p("blobs", jid + ".bin"), working_path=dst)
        return None

    def mark_done(self, job: Job, result: Optional[Dict[str, Any]] = None) -> None:
        meta = dict(job.meta)
        meta["done_at"] = int(time.time())
        if result:
            meta["result"] = result
        self._move(job.working_path, "done", meta)
        self._rm(job.blob_path)              # source no longer needed

    def mark_failed(self, job: Job, error: str) -> None:
        meta = dict(job.meta)
        meta["failed_at"] = int(time.time())
        meta["error"] = error
        meta["attempts"] = int(meta.get("attempts", 0)) + 1
        self._move(job.working_path, "failed", meta)
        self._rm(job.blob_path)

    def requeue(self, job: Job) -> None:
        """Return a claimed job to pending (transient failure, or worker shutting
        down). Bumps the attempt count so a poison job can't loop forever."""
        meta = dict(job.meta)
        meta["attempts"] = int(meta.get("attempts", 0)) + 1
        pend = self._p("pending", job.id + ".json")
        self._write_json(pend, meta)
        self._rm(job.working_path)

    # -- maintenance -------------------------------------------------------
    def requeue_stale(self, older_than_s: float = 900.0) -> int:
        """A worker that died mid-render leaves a working/ json behind. Return any
        older than the timeout to pending so the job isn't lost. Returns count moved."""
        moved = 0
        wdir = self._p("working", "")
        now = time.time()
        try:
            names = [n for n in os.listdir(wdir) if n.endswith(".json")]
        except FileNotFoundError:
            return 0
        for name in names:
            path = self._p("working", name)
            try:
                if now - os.path.getmtime(path) < older_than_s:
                    continue
                with open(path) as f:
                    meta = json.load(f)
            except (FileNotFoundError, OSError, ValueError):
                continue
            meta["attempts"] = int(meta.get("attempts", 0)) + 1
            self._write_json(self._p("pending", name), meta)
            self._rm(path)
            moved += 1
        return moved

    def depth(self) -> int:
        """Number of jobs waiting to be rendered (what the adaptive switch reads)."""
        try:
            return sum(1 for n in os.listdir(self._p("pending", "")) if n.endswith(".json"))
        except FileNotFoundError:
            return 0

    def stats(self) -> Dict[str, int]:
        out = {}
        for sub in ("pending", "working", "done", "failed"):
            try:
                out[sub] = sum(1 for n in os.listdir(self._p(sub, "")) if n.endswith(".json"))
            except FileNotFoundError:
                out[sub] = 0
        return out

    # -- internals ---------------------------------------------------------
    def _write_json(self, path: str, meta: Dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)

    def _move(self, src_json: str, sub: str, meta: Dict[str, Any]) -> None:
        name = os.path.basename(src_json)
        self._write_json(self._p(sub, name), meta)
        self._rm(src_json)

    def _rm(self, path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass
