#!/usr/bin/env bash
# Convenience wrapper so the staging mirror ALWAYS runs with the override file +
# its own project name (so it never collides with the production stack). Run from
# the staging working tree (the one checked out to the `staging` branch).
#
# Examples:
#   ./staging.sh up -d --build     # build + start staging on 127.0.0.1:8078
#   ./staging.sh ps                # status
#   ./staging.sh logs -f           # tail logs
#   ./staging.sh down              # stop + remove the staging container
set -euo pipefail
exec docker compose \
  -f docker-compose.yml \
  -f docker-compose.staging.yml \
  -p typortrait-staging \
  "$@"
