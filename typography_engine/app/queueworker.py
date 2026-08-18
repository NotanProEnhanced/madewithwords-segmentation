"""Background worker that drains the async render queue.

Design choice: a queued job is rendered by REPLAYING it against the app's own
`/render` endpoint over loopback, rather than calling the render pipeline
directly. That reuses the exact production render path (compliance gate, analysis
cache, watermark, persistence) with zero duplicated logic that could drift, and
it naturally shares the render semaphore with live traffic. When the portrait is
ready we email the visitor a private link.

Started once per uvicorn worker process (see main.py startup). Multiple processes
draining the same filesystem queue is fine and desirable -- claims are atomic, so
each job is rendered exactly once, and N processes drain N-in-parallel.

Entirely inert unless TYPO_ASYNC_QUEUE is enabled.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import httpx

from .asyncqueue import Job, RenderQueue
from .config import (
    ASYNC_MAX_ATTEMPTS,
    ASYNC_QUEUE_DIR,
    ASYNC_QUEUE_ENABLED,
    ASYNC_STALE_SECONDS,
    PORT,
)

_queue: Optional[RenderQueue] = None


def _log(msg: str) -> None:
    print(f"[asyncqueue] {msg}", flush=True)


def get_queue() -> RenderQueue:
    """Process-local handle to the shared on-disk queue."""
    global _queue
    if _queue is None:
        _queue = RenderQueue(ASYNC_QUEUE_DIR)
    return _queue


def enqueue_render(fields: Dict[str, Any], image: bytes, email: str,
                   brand_name: str = "Typortrait") -> str:
    """Accept a render job for background processing. `fields` are the exact
    /render form fields (minus the image); they're replayed verbatim later."""
    meta = {"fields": fields, "email": email, "brand_name": brand_name}
    return get_queue().enqueue(meta, image)


async def _process_job(client: httpx.AsyncClient, base_url: str, q: RenderQueue, job: Job) -> None:
    with open(job.blob_path, "rb") as f:
        image = f.read()
    fields = dict(job.meta.get("fields") or {})
    data = {k: ("" if v is None else str(v)) for k, v in fields.items()}
    files = {"image": ("upload.jpg", image, "image/jpeg")}
    r = await client.post(base_url + "/render", data=data, files=files)
    rendered_job = None
    try:
        body = r.json()
        if body.get("ok"):
            rendered_job = body.get("job")
    except Exception:  # noqa: BLE001  non-JSON / error body
        pass
    if not rendered_job:
        raise RuntimeError(f"render replay failed (status={r.status_code})")
    # Portrait is rendered + persisted. Email delivery is best-effort: a mail hiccup
    # must NOT fail the job (the portrait exists and is reachable at /resume/<job>).
    email = job.meta.get("email")
    if email:
        try:
            from .admin import send_ready_email
            send_ready_email(rendered_job, email, job.meta.get("brand_name", "Typortrait"))
        except Exception as e:  # noqa: BLE001
            _log(f"delivery email failed job={rendered_job}: {type(e).__name__}: {e}")
    q.mark_done(job, {"render_job": rendered_job})
    _log(f"done job={job.id} -> render={rendered_job}")


async def worker_loop() -> None:
    """Claim -> replay -> email, forever. Resilient: a bad job is retried up to
    ASYNC_MAX_ATTEMPTS then parked in failed/, and no exception can kill the loop."""
    if not ASYNC_QUEUE_ENABLED:
        return
    q = get_queue()
    base_url = f"http://127.0.0.1:{PORT}"
    _log(f"worker starting (dir={ASYNC_QUEUE_DIR}, target={base_url})")
    await asyncio.sleep(3)      # let uvicorn start accepting connections first
    last_sweep = 0.0
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        while True:
            try:
                now = time.time()
                if now - last_sweep > 120:      # reclaim jobs from any dead worker
                    moved = q.requeue_stale(ASYNC_STALE_SECONDS)
                    if moved:
                        _log(f"requeued {moved} stale job(s)")
                    last_sweep = now
                job = q.claim()
                if job is None:
                    await asyncio.sleep(1.0)
                    continue
                try:
                    await _process_job(client, base_url, q, job)
                except Exception as e:  # noqa: BLE001
                    if int(job.meta.get("attempts", 0)) + 1 < ASYNC_MAX_ATTEMPTS:
                        q.requeue(job)
                        _log(f"retry job={job.id}: {type(e).__name__}: {e}")
                        await asyncio.sleep(2.0)
                    else:
                        q.mark_failed(job, f"{type(e).__name__}: {e}"[:200])
                        _log(f"gave up job={job.id}: {e}")
            except Exception as e:  # noqa: BLE001  loop must never die
                _log(f"loop error: {type(e).__name__}: {e}")
                await asyncio.sleep(2.0)
