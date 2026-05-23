"""FastAPI entrypoint for the isolated typography portrait engine.

Phase 1: project structure + health endpoint.
Later phases add /debug/preprocess and /render.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import __version__
from .pipeline.capabilities import probe

app = FastAPI(title="Typography Portrait Engine", version=__version__)


@app.get("/health")
def health() -> JSONResponse:
    caps = probe()
    return JSONResponse(
        {
            "ok": True,
            "service": "typography-portrait-engine",
            "version": __version__,
            "capabilities": caps,
        }
    )
