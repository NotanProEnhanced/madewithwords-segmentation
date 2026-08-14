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
set -uo pipefail
docker compose \
  -f docker-compose.pawsinwords.yml \
  -p typortrait-pawsinwords \
  "$@"
rc=$?
# After a build, reclaim space from the now-dangling OLD image layers so repeated
# rebuilds don't fill the disk (a full '/' can even lock you out of SSH). Only ever
# removes UNTAGGED/dangling images -- never a tagged image in use. Best-effort.
case " $* " in
  *" --build"*|*" build "*) docker image prune -f >/dev/null 2>&1 || true ;;
esac
exit "$rc"
