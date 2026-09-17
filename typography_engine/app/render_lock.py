"""One heavy render at a time across the whole box.

The v2 engine (PawsInWords, and Natural on the human sites) holds up to 3 GB while it
renders. Five brand containers share an 8 GB box, and until now they knew nothing of each
other: two heavy renders on two brands at the same moment was the state that locked the
box up in September. This is a lock on a file in a directory every container bind-mounts
from the same host path (docker-compose: RENDER_LOCK_HOST_DIR -> /app/locks), taken with
flock, which the kernel enforces across containers because they share the inode. A heavy
render waits its turn; Lifelike renders are not heavy and never wait.

Off when RENDER_LOCK_DIR is unset or empty: the engine renders exactly as it did, and the
gates stay byte-identical either way, because waiting changes nothing about the pixels.

/health reports `heavy_lock`: whether it is on, how many renders in this process are
waiting or holding, and the longest wait seen since start -- the number that says when
the box needs a second queue (more memory and cores), see the journal for the rule.
"""
from __future__ import annotations

import os
import threading
import time

try:
    import fcntl
except ImportError:  # pragma: no cover -- not a Linux box; the lock is simply off
    fcntl = None

_DIR = (os.environ.get("RENDER_LOCK_DIR", "") or "").strip()
_NAME = "heavy-render.lock"
_STATE = threading.Lock()
_waiting = 0
_holding = 0
_waits = 0
_wait_max_s = 0.0
_wait_total_s = 0.0


def enabled() -> bool:
    return bool(_DIR) and fcntl is not None


class _Held:
    """What acquire() returns: release() once, safely, from any thread."""

    def __init__(self, fd, waited_s):
        self._fd = fd
        self.waited_s = waited_s
        self._done = False

    def release(self) -> None:
        global _holding
        if self._done:
            return
        self._done = True
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            with _STATE:
                _holding -= 1


def acquire() -> _Held:
    """Block until no other heavy render on this box holds the lock, then hold it. With
    the lock off, returns at once and release() is a no-op. Never raises for a lock
    problem: a directory that cannot be opened means the lock is off for this render,
    logged once per process, rather than a failed portrait."""
    global _waiting, _holding, _waits, _wait_max_s, _wait_total_s
    if not enabled():
        return _Held(None, 0.0)
    try:
        os.makedirs(_DIR, exist_ok=True)
        fd = os.open(os.path.join(_DIR, _NAME), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:  # noqa: BLE001
        _warn_once(f"render lock unavailable ({e!r}); rendering without it")
        return _Held(None, 0.0)
    t0 = time.monotonic()
    with _STATE:
        _waiting += 1
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as e:  # noqa: BLE001
        os.close(fd)
        with _STATE:
            _waiting -= 1
        _warn_once(f"render lock failed ({e!r}); rendering without it")
        return _Held(None, 0.0)
    waited = time.monotonic() - t0
    with _STATE:
        _waiting -= 1
        _holding += 1
        _waits += 1
        _wait_total_s += waited
        if waited > _wait_max_s:
            _wait_max_s = waited
    if waited >= 1.0:
        print(f"[render-lock] waited {waited:.1f}s for the box", flush=True)
    return _Held(fd, waited)


def status() -> dict:
    with _STATE:
        return {
            "enabled": enabled(),
            "dir": _DIR or None,
            "waiting": _waiting,
            "holding": _holding,
            "waits": _waits,
            "wait_max_s": round(_wait_max_s, 1),
            "wait_mean_s": round(_wait_total_s / _waits, 1) if _waits else 0.0,
        }


_warned = set()


def _warn_once(msg: str) -> None:
    if msg in _warned:
        return
    _warned.add(msg)
    print(f"[render-lock] {msg}", flush=True)
