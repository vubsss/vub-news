#!/bin/bash
# Phase 8: build a large distribution and record what every stage cost.
#
#     sbatch ada/scale.sbatch mind_large
#     sbatch ada/scale.sbatch ebnerd_large
#
# Separate from build.sbatch because the placement is different, and the
# placement is the whole difference. That one keeps results on /home, which is
# right for the working scale: artifacts are small and worth keeping. This one
# cannot -- /home is a 30 GB quota with about 10 GB free, and ebnerd_large is
# 3.19 GB of archive that extracts to considerably more before a feature store
# is written on top of it.
#
# So everything goes to node-local /scratch, which is 2 TB and purged weekly,
# and the one thing that comes back is `build-timings.jsonl`: a few KB, and the
# only output this run exists to produce. The large feature stores are
# rebuildable and nothing downstream is reported from them yet -- the numbers
# every result in this project cites are the small ones. If that changes, this
# is the file to change.
#
# The archives are staged in from /home rather than downloaded by the job, and
# that is a per-host decision rather than a policy. Measured from Ada in one
# minute on 2026-08-23: EB-NeRD's S3 bucket **43 KB/s**, HuggingFace 11 MB/s,
# download.pytorch.org 18.7 MB/s. At 43 KB/s `ebnerd_large.zip` is a 26-hour
# download and the first attempt at this job had fetched 80 MB of 3.19 GB when
# it was killed at 40 minutes. The same laptop pulls it at 5.8 MB/s, so it is
# fetched there and rsynced up -- which is phase 6's design, right for this
# bucket and unnecessary for the other two.

#SBATCH --job-name=vub-scale
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log

# 32 CPUs and 100 GB: the practical ceiling for this account, and the point of
# asking for it is to find out whether the large build needs it. Phase 6
# measured the small build peaking at 23.6 GB in `evaluate`.

set -euo pipefail

DATASET="${1:?usage: sbatch ada/scale.sbatch <dataset>}"
REPO="$HOME/vub-news"
STAGE="/scratch/$USER/vub-news-scale"
RESULTS="$STAGE/run"
KEEP="$HOME/vub-news-run/scale"
ENV_PY="$HOME/envs/vub-news/bin/python"

echo "node        $(hostname)"
echo "job         ${SLURM_JOB_ID:-none}"
echo "dataset     $DATASET"
echo "cpus        $(nproc)"
echo "started     $(date -Is)"

[ -x "$ENV_PY" ] || { echo "no environment at $ENV_PY — run ada/env.sbatch first"; exit 1; }

mkdir -p "$STAGE" "$RESULTS" "$KEEP"

# Archives from /home, where `ada/sync.sh data` put them. `acquire` finds them
# already cached and extracts rather than downloading; anything missing it
# still fetches itself, which is fine for the fast hosts.
FROM_HOME="$HOME/vub-news-data/data/raw/$DATASET/_archives"
if [ -d "$FROM_HOME" ]; then
    echo "staging archives in ..."
    mkdir -p "$STAGE/data/raw/$DATASET"
    /usr/bin/time -f "  staged in %e s" \
        rsync -aL "$FROM_HOME" "$STAGE/data/raw/$DATASET/"
    du -sh "$STAGE/data/raw/$DATASET/_archives"
fi

# The timings file accumulates across runs and is the deliverable, so a
# previous dataset's rows are carried in rather than started over.
mkdir -p "$RESULTS/artifacts"
[ -f "$KEEP/build-timings.jsonl" ] && cp "$KEEP/build-timings.jsonl" "$RESULTS/artifacts/"

keep_timings() {
    local status=$?
    if [ -f "$RESULTS/artifacts/build-timings.jsonl" ]; then
        cp "$RESULTS/artifacts/build-timings.jsonl" "$KEEP/"
        echo "timings kept in $KEEP/build-timings.jsonl ($(wc -l < "$KEEP/build-timings.jsonl") rows)"
    fi
    echo "exit        $status"
    echo "finished    $(date -Is)"
    df -h "$STAGE" | tail -1
    du -sh "$STAGE"/* 2>/dev/null || true
}
trap keep_timings EXIT
trap 'exit 143' TERM

export VUB_NEWS_DATA_ROOT="$STAGE"
export VUB_NEWS_ROOT="$RESULTS"
cd "$REPO"

echo "--- build.py --dataset $DATASET ---"
/usr/bin/time -v "$ENV_PY" build.py --dataset "$DATASET" \
    2> >(tee "$KEEP/last-scale-$DATASET.time" >&2)
