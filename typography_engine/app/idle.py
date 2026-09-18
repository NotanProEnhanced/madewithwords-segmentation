"""Idle unloading: give the models and caches back when nobody has rendered for a while.

A container that has rendered once holds, for the life of the process, every ONNX
matting session it opened (ISNet for pets, ISNet and RVM for people: about 170 MB each)
and the glyph caches the pet engine fills (measured 2026-09-18, locally: 77 MB after two
renders). With four brands on one box that is the difference between four containers
idling at 340 MB each and four idling at a gigabyte each, which is the margin one
Natural render needs.

Each module that holds something registers an unloader here. A daemon thread checks
every 30 s; when nothing has touched a model for TYPO_IDLE_UNLOAD_S (default 600; 0
turns the whole thing off) it runs every unloader, then hands the freed heap to the OS.
The next render reloads what it needs (an ISNet session opens in about a second) and
produces the same pixels: a session is a session. Only references are dropped, never
objects in use -- a render mid-inference keeps its own reference to the session and
the glyph caches are plain dicts whose clear() is atomic under the GIL, so a lookup
that misses simply rasterises again.

Fail-safe throughout: an unloader that raises is logged and skipped; the thread never
takes the process down.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from typing import Callable, Dict, List, Tuple

_LOCK = threading.Lock()
_UNLOADERS: List[Tuple[str, Callable[[], None]]] = []
_last_touch = time.monotonic()
_unloads = 0
_last_unload_at = 0.0
_last_unload_ran: List[str] = []
_thread = None


def seconds() -> float:
    v = os.environ.get("TYPO_IDLE_UNLOAD_S", "600").strip()
    try:
        return max(0.0, float(v or 0))
    except ValueError:
        return 600.0


def enabled() -> bool:
    return seconds() > 0


def register(name: str, fn: Callable[[], None]) -> None:
    with _LOCK:
        if all(n != name for n, _ in _UNLOADERS):
            _UNLOADERS.append((name, fn))


def touch() -> None:
    """Call whenever a model or cache is used. Cheap; no lock."""
    global _last_touch
    _last_touch = time.monotonic()


def idle_seconds() -> float:
    return time.monotonic() - _last_touch


def trim() -> None:
    """Hand freed heap back to the OS. gc first, so what Python just dropped is freed too."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 -- not glibc: nothing to do
        pass


def unload_now(reason: str = "manual") -> List[str]:
    """Run every unloader once. Returns the names that ran."""
    global _unloads, _last_unload_at, _last_unload_ran
    ran: List[str] = []
    with _LOCK:
        todo = list(_UNLOADERS)
    for name, fn in todo:
        try:
            fn()
            ran.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[idle] unloader {name} failed: {e}")
    trim()
    _unloads += 1
    _last_unload_at = time.time()
    _last_unload_ran = ran
    print(f"[idle] unloaded ({reason}): {', '.join(ran) or 'nothing registered'}")
    return ran


def _loop() -> None:
    unloaded_for_this_idle = False
    while True:
        time.sleep(30)
        try:
            t = seconds()
            if t <= 0:
                continue
            if idle_seconds() >= t:
                if not unloaded_for_this_idle:
                    unload_now(reason=f"idle {int(idle_seconds())}s")
                    unloaded_for_this_idle = True
            else:
                unloaded_for_this_idle = False
        except Exception as e:  # noqa: BLE001
            print(f"[idle] loop error: {e}")


def start() -> None:
    global _thread
    with _LOCK:
        if _thread is not None:
            return
        _thread = threading.Thread(target=_loop, name="idle-unload", daemon=True)
        _thread.start()


def status() -> Dict[str, object]:
    return {
        "enabled": enabled(),
        "after_s": int(seconds()),
        "idle_s": int(idle_seconds()),
        "unloads": _unloads,
        "last_unload": int(_last_unload_at),
        "last_ran": list(_last_unload_ran),
        "registered": [n for n, _ in _UNLOADERS],
    }
