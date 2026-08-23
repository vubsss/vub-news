#!/bin/bash
# Move code and data between this laptop, Ada's /home, and Ada's /share1.
#
#     ada/sync.sh code       laptop  ->  ada:/home     the repo, minus the bulk
#     ada/sync.sh data       laptop  ->  ada:/home     raw archives, ~3.4 GB
#     ada/sync.sh artifacts  laptop  ->  ada:/home     computed artifacts
#     ada/sync.sh park                   ada:/home -> ada:/share1   (on Ada)
#     ada/sync.sh pull       ada:/home  ->  laptop     artifacts and results
#     ada/sync.sh submissions ada:/home ->  laptop     the CodaBench zips
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
    # Every dataset's archives, rather than a list to keep in step with the
    # registry -- phase 8 added two entries and this is where that would have
    # been noticed late. rsync skips what is already there, so the large
    # bundles cost their 3.19 GB once.
    rsync -az --info=stats1 --relative \
      "$REPO"/./data/raw/*/_archives "$HOST:$REMOTE-data/"
    ;;

  artifacts)
    # MIND's embedding artifact, and anything else already computed. The embed
    # stage would otherwise fetch the vectors from Drive with gdown, which is
    # 100 MB down a link that gives PyPI 0.22 MB/s -- while this ssh hop runs
    # at 10, and cannot half-succeed forty minutes into a queued job.
    rsync -az --info=stats1 --rsync-path="mkdir -p $REMOTE-run/artifacts && rsync" \
      "$REPO/artifacts/" "$HOST:$REMOTE-run/artifacts/"
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

  submissions)
    # The zips, which `pull` deliberately does not carry: artifacts are pulled
    # often and these are ~1 GB of files that change once. Uploaded by hand to
    # CodaBench afterwards, so they have to reach this machine.
    rsync -az --info=stats1 --include '*/' --include '*.zip' --exclude '*' \
      "$HOST:$REMOTE-run/predictions/" "$REPO/predictions/"
    ;;

  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
