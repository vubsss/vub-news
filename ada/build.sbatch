#!/bin/bash
# Run the pipeline on an Ada compute node, staging data in and results out.
#
#     sbatch ada/build.sbatch                  # every pending stage
#     sbatch ada/build.sbatch --force bm25     # arguments go to build.py
#
# The point of this file is that it is a *wrapper*. It sets two environment
# variables and calls `python build.py` with whatever arguments it was given.
# The assignment requires a one-command rebuild on a single machine (Q1.5), so
# SLURM has to stay a deployment detail — if this script ever needs build.py to
# know it exists, the change belongs in this file instead.
#
# The filesystem shapes everything. From a compute node:
#
#     /home         30 GB, visible, persistent      <- code, env, results
#     /scratch       2 TB, visible, node-local,     <- data and feature store
#                    purged after 7 days
#     /share1      100 GB, NOT VISIBLE from here    <- long-term storage
#
# So the heavy directories go to /scratch, where they are fast and roomy and
# temporary, and the outputs go to /home, where they survive the job. /share1
# holds the durable copy but can only be reached from the login node, which is
# what `ada/sync.sh pull` is for.
#
# Phase 1's relocatable root is the whole mechanism: VUB_NEWS_DATA_ROOT moves
# data/ and feature_store/, VUB_NEWS_ROOT moves artifacts/, predictions/ and
# .checkpoints/, and they move independently.

#SBATCH --job-name=vub-build
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log

# 20 CPUs and 60 GB, with no GPU requested. Measured on this cluster: a
# zero-GPU job of this size allocates without complaint, so the 1:10 GPU:CPU
# ratio in the user guide does not bind jobs that ask for none. 32 CPUs and
# 100 GB also allocate; 40 does not, because a node has exactly 40 and one is
# rarely wholly free. Only the `embed` stage would use a GPU, and it reads a
# cached artifact in the ordinary case.

set -euo pipefail

REPO="$HOME/vub-news"
STAGE="/scratch/$USER/vub-news"
RESULTS="$HOME/vub-news-run"
ENV_PY="$HOME/envs/vub-news/bin/python"

echo "node        $(hostname)"
echo "job         ${SLURM_JOB_ID:-none}"
echo "cpus        $(nproc)"
echo "started     $(date -Is)"

[ -x "$ENV_PY" ] || { echo "no environment at $ENV_PY — run ada/env.sbatch first"; exit 1; }

# --- stage in ---------------------------------------------------------------
# Copied rather than symlinked: /scratch is node-local disk and the point of
# using it is that reads do not cross NFS. A symlink would leave every read on
# /home and cost the quota nothing while buying nothing either.
mkdir -p "$STAGE" "$RESULTS"
if [ -d "$HOME/vub-news-data/data" ]; then
    echo "staging data in ..."
    /usr/bin/time -f "  staged in %e s" \
        rsync -a --delete "$HOME/vub-news-data/data" "$STAGE/"
fi
if [ -d "$HOME/vub-news-data/feature_store" ]; then
    rsync -a --delete "$HOME/vub-news-data/feature_store" "$STAGE/"
fi
df -h "$STAGE" | tail -1

# --- stage out --------------------------------------------------------------
# The feature store comes back because rebuilding it costs more than storing
# it, and /scratch is purged weekly. artifacts/ and .checkpoints/ were written
# straight to $RESULTS and are already here.
#
# In a trap, because staging out is placement rather than computation and this
# script is `set -e`: job 2674571 failed in the ninth of nine stages, the shell
# abandoned the rest of the file, and eight stages' worth of feature store sat
# on node-local /scratch until it was rescued by hand four days into a
# seven-day purge window. A run that fails late has still earned what the
# stages before it produced.
#
# `--delete` only on the way out clean. A crash mid-`ingest` leaves a partial
# store on /scratch while /home still holds the whole one from last time, and
# mirroring the first onto the second would faithfully destroy it. Adding and
# overwriting without deleting can leave a stale file behind; the next
# successful run deletes it, and stale is recoverable where deleted is not.
#
# EXIT covers the failure path; TERM turns SLURM's warning shot at the walltime
# into an exit, which fires EXIT once. Nothing survives SIGKILL, so a job that
# overruns its walltime still loses /scratch.
stage_out() {
    local status=$?
    [ -d "$STAGE/feature_store" ] || return 0

    echo "staging out ..."
    mkdir -p "$HOME/vub-news-data"
    if [ "$status" -eq 0 ]; then
        rsync -a --delete "$STAGE/feature_store" "$HOME/vub-news-data/"
    else
        echo "  build exited $status — staging out without --delete"
        rsync -a "$STAGE/feature_store" "$HOME/vub-news-data/"
    fi

    # Printed here so they are printed on both paths. The wall clock the scale
    # analysis cites was reconstructed from `sacct` last time, because the run
    # that produced it never reached these lines.
    echo "finished    $(date -Is)"
    echo "results in  $RESULTS  (pull to /share1 with: ada/sync.sh pull)"
    du -sh "$RESULTS" "$HOME/vub-news-data" 2>/dev/null || true
}
trap stage_out EXIT
trap 'exit 143' TERM

# --- run --------------------------------------------------------------------
# The unmodified command. Everything above and below is placement.
export VUB_NEWS_DATA_ROOT="$STAGE"
export VUB_NEWS_ROOT="$RESULTS"
cd "$REPO"

echo "--- build.py $* ---"
/usr/bin/time -v "$ENV_PY" build.py "$@" 2> >(tee "$RESULTS/last-build.time" >&2)
