"""In-memory render-job registry: thread-safe per-job progress + SSE.

Every /render/start spawns a worker thread that runs the actual pipeline.
The thread is handed a `ProgressReporter` which is called from inside the
pipeline at real work boundaries (one call per row placed, one call when
the rasterise step starts, etc). The reporter pushes events onto a
per-job queue that the SSE endpoint drains.

Honesty rules (the user said: "Best practice! Nothing fake."):
  - Events are emitted only from inside the pipeline. No cosmetic timer
    ticks, no fixed-interval rotators.
  - `frac` for a stage is the real fraction of work units completed in
    that stage (rows-placed / total-rows), never wall-clock-derived.
  - Overall progress is computed from a rolling-average of per-stage
    durations across past completed jobs; until we have at least one
    completed run we weight every stage equally rather than invent a
    distribution.
  - ETA is reported only when every stage has at least one historical
    timing; before that the SSE event carries `eta_seconds=None` and the
    UI shows "calculating" -- we never guess.
  - Cancellation is real: the reporter raises Cancelled inside the
    pipeline call site, so a cancelled job stops mid-row, not after the
    whole pipeline finishes.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable, Deque, Dict, List, Optional


# Stages in the order they actually run inside the /render worker.
# Renamed from the previous fake "Finding the face… / Placing your words…"
# rotation so each label corresponds to a function the worker actually calls.
STAGES = (
    "analyze",
    "layout_body",
    "layout_mid",
    "layout_face",
    "layout_eyes",
    "rasterize",
    "composite",
    "watermark",
    "persist",
)

STAGE_LABELS = {
    "analyze":     "Reading your photo",
    "layout_body": "Laying the outer words",
    "layout_mid":  "Tightening around the face",
    "layout_face": "Resolving the face",
    "layout_eyes": "Sharpening the eyes",
    "rasterize":   "Rasterising the type",
    "composite":   "Compositing the photo through the type",
    "watermark":   "Watermarking the preview",
    "persist":     "Saving",
}


class Cancelled(Exception):
    """Raised inside the pipeline when the user has cancelled the job."""


@dataclass
class JobEvent:
    """One event in the SSE stream. Serialised verbatim to JSON."""
    type: str                       # progress | stage_done | done | error | cancelled
    stage: str = ""
    stage_label: str = ""
    stage_index: int = 0
    stage_count: int = len(STAGES)
    frac: float = 0.0
    overall: float = 0.0
    eta_seconds: Optional[float] = None
    detail: str = ""
    result: Optional[Dict[str, Any]] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "stage_index": self.stage_index,
            "stage_count": self.stage_count,
            "frac": round(self.frac, 4),
            "overall": round(self.overall, 4),
            "detail": self.detail,
        }
        if self.eta_seconds is not None:
            d["eta_seconds"] = round(self.eta_seconds, 2)
        if self.result is not None:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class JobRecord:
    job_id: str
    started_at: float
    queue: "Queue[JobEvent]" = field(default_factory=Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    stage_times: Dict[str, float] = field(default_factory=dict)
    current_stage: str = "analyze"
    current_stage_start: float = 0.0
    ended_at: Optional[float] = None
    final_status: str = "running"   # running | done | error | cancelled
    final_event: Optional[JobEvent] = None


# Rolling-average history of stage durations across completed jobs.
# Used only to weight the overall progress bar + compute ETA; never used
# to fake intra-stage motion.
_STAGE_HISTORY: Dict[str, Deque[float]] = {s: deque(maxlen=20) for s in STAGES}
_HISTORY_LOCK = threading.Lock()


def _stage_avg(stage: str) -> Optional[float]:
    with _HISTORY_LOCK:
        h = _STAGE_HISTORY.get(stage)
        if not h:
            return None
        return sum(h) / len(h)


def _record_stage_duration(stage: str, secs: float) -> None:
    # Drop obvious junk: a negative dt from clock skew, or a runaway >2min
    # value that would poison the moving average. Real stages finish well
    # under that on the configured working sizes.
    if secs <= 0 or secs > 120:
        return
    with _HISTORY_LOCK:
        h = _STAGE_HISTORY.get(stage)
        if h is not None:
            h.append(secs)


def _compute_weights() -> List[float]:
    """Weight each stage by its rolling-average duration. Flat (1/N) until
    every stage has at least one historical sample -- we don't invent a
    distribution before we've measured one."""
    avgs = [_stage_avg(s) for s in STAGES]
    if any(a is None for a in avgs):
        return [1.0 / len(STAGES)] * len(STAGES)
    total = sum(avgs)  # type: ignore[arg-type]
    if total <= 0:
        return [1.0 / len(STAGES)] * len(STAGES)
    return [float(a) / total for a in avgs]  # type: ignore[union-attr]


class JobRegistry:
    """Process-wide in-memory registry of in-flight jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def new(self) -> JobRecord:
        jid = uuid.uuid4().hex[:12]
        now = time.time()
        rec = JobRecord(job_id=jid, started_at=now, current_stage_start=now)
        with self._lock:
            self._jobs[jid] = rec
        return rec

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def reap(self, older_than_seconds: float = 600) -> None:
        cutoff = time.time() - older_than_seconds
        with self._lock:
            stale = [k for k, v in self._jobs.items()
                     if v.ended_at is not None and v.ended_at < cutoff]
            for k in stale:
                self._jobs.pop(k, None)


registry = JobRegistry()


class ProgressReporter:
    """The object the pipeline calls. Closes over a JobRecord; safe to
    pass from the request thread into the worker thread."""

    def __init__(self, rec: JobRecord) -> None:
        self.rec = rec
        self.weights = _compute_weights()

    # ---- internal helpers ---------------------------------------------------

    def _emit(self, ev: JobEvent) -> None:
        try:
            self.rec.queue.put_nowait(ev)
        except Exception:  # noqa: BLE001 -- never let telemetry kill the job
            pass

    def _overall(self, stage: str, frac: float) -> float:
        if stage not in STAGES:
            return 0.0
        idx = STAGES.index(stage)
        before = sum(self.weights[:idx])
        cur = self.weights[idx]
        return min(1.0, before + cur * max(0.0, min(1.0, frac)))

    def _eta(self, stage: str, frac: float) -> Optional[float]:
        avgs = [_stage_avg(s) for s in STAGES]
        if any(a is None for a in avgs):
            return None
        total = sum(avgs)  # type: ignore[arg-type]
        idx = STAGES.index(stage) if stage in STAGES else 0
        elapsed_in_stage = avgs[idx] * max(0.0, min(1.0, frac))  # type: ignore[operator]
        elapsed = sum(avgs[:idx]) + elapsed_in_stage  # type: ignore[arg-type]
        return max(0.0, float(total) - float(elapsed))  # type: ignore[arg-type]

    def _close_prev_stage(self, new_stage: str) -> None:
        now = time.time()
        prev = self.rec.current_stage
        if prev == new_stage:
            return
        dt = now - self.rec.current_stage_start
        self.rec.stage_times[prev] = dt
        _record_stage_duration(prev, dt)
        self.rec.current_stage = new_stage
        self.rec.current_stage_start = now
        self._emit(JobEvent(
            type="stage_done",
            stage=prev,
            stage_label=STAGE_LABELS.get(prev, prev),
            stage_index=STAGES.index(prev) if prev in STAGES else 0,
            frac=1.0,
            overall=self._overall(prev, 1.0),
        ))

    # ---- pipeline-facing surface --------------------------------------------

    def __call__(self, stage: str, frac: float, detail: str = "") -> None:
        """Pipeline calls this at every work boundary. Raises Cancelled if
        the user has hit the cancel button -- the pipeline propagates it
        up and the worker thread emits a 'cancelled' event."""
        if self.rec.cancel.is_set():
            raise Cancelled()
        if stage not in STAGES:
            return
        if stage != self.rec.current_stage:
            self._close_prev_stage(stage)
        idx = STAGES.index(stage)
        self._emit(JobEvent(
            type="progress",
            stage=stage,
            stage_label=STAGE_LABELS.get(stage, stage),
            stage_index=idx,
            frac=max(0.0, min(1.0, frac)),
            overall=self._overall(stage, frac),
            eta_seconds=self._eta(stage, frac),
            detail=detail,
        ))

    # ---- terminal events ----------------------------------------------------

    def done(self, result: Dict[str, Any]) -> None:
        now = time.time()
        prev = self.rec.current_stage
        dt = now - self.rec.current_stage_start
        self.rec.stage_times[prev] = dt
        _record_stage_duration(prev, dt)
        self.rec.ended_at = now
        self.rec.final_status = "done"
        ev = JobEvent(
            type="done", stage=prev, stage_label=STAGE_LABELS.get(prev, prev),
            stage_index=len(STAGES) - 1, frac=1.0, overall=1.0,
            eta_seconds=0.0, result=result,
        )
        self.rec.final_event = ev
        self._emit(ev)
        self.rec.done.set()

    def error(self, message: str) -> None:
        self.rec.ended_at = time.time()
        self.rec.final_status = "error"
        ev = JobEvent(type="error", error=message)
        self.rec.final_event = ev
        self._emit(ev)
        self.rec.done.set()

    def cancelled(self) -> None:
        self.rec.ended_at = time.time()
        self.rec.final_status = "cancelled"
        ev = JobEvent(type="cancelled")
        self.rec.final_event = ev
        self._emit(ev)
        self.rec.done.set()
