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
set -uo pipefail
docker compose \
  -f docker-compose.faithinwords.yml \
  -p typortrait-faithinwords \
  "$@"
rc=$?
# After a build, reclaim space from the now-dangling OLD image layers so repeated
# rebuilds don't fill the disk (a full '/' can even lock you out of SSH). Only ever
# removes UNTAGGED/dangling images -- never a tagged image in use. Best-effort.
case " $* " in
  *" --build"*|*" build "*) docker image prune -f >/dev/null 2>&1 || true ;;
esac
exit "$rc"
