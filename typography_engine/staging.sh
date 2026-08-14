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
set -uo pipefail
docker compose \
  -f docker-compose.staging.yml \
  -p typortrait-staging \
  "$@"
rc=$?
# After a build, reclaim space from the now-dangling OLD image layers so repeated
# rebuilds don't fill the disk (a full '/' can even lock you out of SSH). Only ever
# removes UNTAGGED/dangling images -- never a tagged image in use. Best-effort.
case " $* " in
  *" --build"*|*" build "*) docker image prune -f >/dev/null 2>&1 || true ;;
esac
exit "$rc"
