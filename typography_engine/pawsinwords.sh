#!/usr/bin/env bash
# Convenience wrapper so the PawsInWords studio ALWAYS runs with its own standalone
# compose file + its own project name (so it can never collide with prod, staging,
# faithinwords, or lovedinwords). Run from the PawsInWords working tree
# (~/typortrait-pawsinwords).
#
# Examples:
#   ./pawsinwords.sh up -d --build   # build + start on 127.0.0.1:8081
#   ./pawsinwords.sh ps              # status
#   ./pawsinwords.sh logs -f         # tail logs
#   ./pawsinwords.sh down            # stop + remove the container
set -euo pipefail
exec docker compose \
  -f docker-compose.pawsinwords.yml \
  -p typortrait-pawsinwords \
  "$@"
