#!/usr/bin/env bash
# Launch the isolated typography portrait engine.
# Does NOT touch the live segmentation app (app.py at repo root).
set -euo pipefail
cd "$(dirname "$0")"
PORT="${TYPO_PORT:-8077}"
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" "$@"
