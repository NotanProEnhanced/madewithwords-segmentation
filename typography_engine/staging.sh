#!/usr/bin/env bash
# Convenience wrapper so the staging mirror ALWAYS runs with its own standalone
# compose file + its own project name (so it can never collide with prod). Run
# from the staging working tree (the one checked out to the `staging` branch).
#
# Examples:
#   ./staging.sh up -d --build     # build + start staging on 127.0.0.1:8078
#   ./staging.sh ps                # status
#   ./staging.sh logs -f           # tail logs
#   ./staging.sh down              # stop + remove the staging container
set -euo pipefail
exec docker compose \
  -f docker-compose.staging.yml \
  -p typortrait-staging \
  "$@"
