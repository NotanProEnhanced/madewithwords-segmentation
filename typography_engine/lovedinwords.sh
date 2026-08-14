#!/usr/bin/env bash
# Convenience wrapper so the LovedInWords store ALWAYS runs with its own standalone
# compose file + its own project name (so it can never collide with prod, staging, or
# faithinwords). Run from the LovedInWords working tree (~/typortrait-lovedinwords).
#
# Examples:
#   ./lovedinwords.sh up -d --build   # build + start on 127.0.0.1:8080
#   ./lovedinwords.sh ps              # status
#   ./lovedinwords.sh logs -f         # tail logs
#   ./lovedinwords.sh down            # stop + remove the container
set -uo pipefail
docker compose \
  -f docker-compose.lovedinwords.yml \
  -p typortrait-lovedinwords \
  "$@"
rc=$?
# After a build, reclaim space from the now-dangling OLD image layers so repeated
# rebuilds don't fill the disk (a full '/' can even lock you out of SSH). Only ever
# removes UNTAGGED/dangling images -- never a tagged image in use. Best-effort.
case " $* " in
  *" --build"*|*" build "*) docker image prune -f >/dev/null 2>&1 || true ;;
esac
exit "$rc"
