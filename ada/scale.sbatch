#!/bin/bash
# Phase 8: build a large distribution and record what every stage cost.
#
#     sbatch ada/scale.sbatch ebnerd_large
#     sbatch ada/scale.sbatch ebnerd_large --only bm25 ann
#
# Arguments after the dataset go to build.py, which is how the large runs are
# staged: `evaluate` at this scale is projected past the walltime (see below),
# so the growth table is collected from the stages that finish and `evaluate`
# is measured separately, bounded.
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
#
# `evaluate` is also the stage this walltime cannot hold at large scale. It
# scored EB-NeRD's 100,981 validation impressions in 1h04m -- 26 a second --
# and ebnerd_large carries roughly fifty times the users, which projects past
# 40 hours. That is the ticket's own question answered rather than a setback:
# the stage that breaks first at the next 10x is `evaluate`, and running it to
# completion would spend a day of cluster time confirming a projection. So the
# large runs are staged with --only, and evaluate is measured bounded.

set -euo pipefail

DATASET="${1:?usage: sbatch ada/scale.sbatch <dataset> [build.py args...]}"
shift
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

# A dataset that generates its own vectors has no artifact the first time, and
# `embed` refuses rather than quietly spending eighty CPU-minutes producing one
# -- which is the right default and the wrong one here, where producing it is
# the job. So it is produced first, explicitly, and `embed` then finds it.
#
# MIND only. EB-NeRD's four vector sources ship with the release and cover it
# whole, so there is nothing to encode.
if [ "${DATASET#mind}" != "$DATASET" ]; then
    GPU_PY="$HOME/envs/vub-news-gpu/bin/python"
    ENCODER="$([ -x "$GPU_PY" ] && echo "$GPU_PY" || echo "$ENV_PY")"
    DEVICE="$([ -n "${SLURM_JOB_GPUS:-}" ] && echo cuda || echo cpu)"

    echo "--- the catalogue, through stage 4, so there is something to encode ---"
    "$ENV_PY" build.py --dataset "$DATASET" --only acquire ingest split preprocess

    echo "--- encoding the active embedding source on $DEVICE ---"
    /usr/bin/time -f "  encoded in %e s" \
        "$ENCODER" -m pipeline.encode_variants --dataset "$DATASET" \
            --variant "e5-base-v2 (query:)" --device "$DEVICE"
fi

echo "--- build.py --dataset $DATASET $* ---"
/usr/bin/time -v "$ENV_PY" build.py --dataset "$DATASET" "$@" \
    2> >(tee "$KEEP/last-scale-$DATASET.time" >&2)
