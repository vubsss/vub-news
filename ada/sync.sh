#!/bin/bash
# Move code and data between this laptop, Ada's /home, and Ada's /share1.
#
#     ada/sync.sh code       laptop  ->  ada:/home     the repo, minus the bulk
#     ada/sync.sh data       laptop  ->  ada:/home     raw archives, ~3.4 GB
#     ada/sync.sh park                   ada:/home -> ada:/share1   (on Ada)
#     ada/sync.sh pull       ada:/home  ->  laptop     artifacts and results
#
# Three filesystems, and only two of them can see each other at a time:
#
#     /home     30 GB   the only shared space a compute node can read
#     /share1  100 GB   roomy and persistent, login node only
#     /scratch   2 TB   fast and node-local, purged after 7 days
#
# So `park` exists because /home is too small to be the archive and /share1 is
# unreachable from the jobs. Everything a job needs passes through /home.
#
# Set ADA_HOST to something other than `ada` if your ssh config names it
# differently; the repo assumes a Host entry rather than a bare hostname so no
# username is hard-coded here.

set -euo pipefail

HOST="${ADA_HOST:-ada}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="vub-news"

case "${1:-}" in
  code)
    # The code only. data/, feature_store/ and artifacts/ are 9 GB between them
    # and each has its own route.
    rsync -az --info=stats1 \
      --exclude 'data/' --exclude 'feature_store/' --exclude 'artifacts/' \
      --exclude 'predictions/' --exclude '.checkpoints/' --exclude '__pycache__/' \
      --exclude '.pytest_cache/' --exclude 'screenshots/' --exclude 'AI_logs/' \
      "$REPO/" "$HOST:$REMOTE/"
    ;;

  data)
    # The downloaded archives, not the extracted tree: acquire re-extracts them
    # on the far side, and sending 8.2 GB where 3.4 GB will do wastes the link
    # and the /home quota alike.
    rsync -az --info=stats1 --relative \
      "$REPO/./data/raw/mind/_archives" "$REPO/./data/raw/ebnerd/_archives" \
      "$HOST:$REMOTE-data/"
    ;;

  park)
    # Run this *on Ada*: /share1 is invisible from the compute nodes, so this
    # cannot be part of a job.
    ssh "$HOST" '
      set -euo pipefail
      mkdir -p /share1/$USER/vub-news
      for d in vub-news-run vub-news-data; do
        [ -d "$HOME/$d" ] && rsync -a "$HOME/$d" "/share1/$USER/vub-news/" && echo "parked $d"
      done
      du -sh /share1/$USER/vub-news/* 2>/dev/null
    '
    ;;

  pull)
    rsync -az --info=stats1 "$HOST:$REMOTE-run/artifacts/" "$REPO/artifacts/"
    ;;

  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
