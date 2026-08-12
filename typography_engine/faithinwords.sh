#!/usr/bin/env bash
# Convenience wrapper so the FaithInWords storefront ALWAYS runs with its own
# standalone compose file + its own project name (so it can never collide with
# prod or staging). Run from the FaithInWords working tree.
#
# Examples:
#   ./faithinwords.sh up -d --build   # build + start on 127.0.0.1:8079
#   ./faithinwords.sh ps              # status
#   ./faithinwords.sh logs -f         # tail logs
#   ./faithinwords.sh down            # stop + remove the container
set -euo pipefail
exec docker compose \
  -f docker-compose.faithinwords.yml \
  -p typortrait-faithinwords \
  "$@"
