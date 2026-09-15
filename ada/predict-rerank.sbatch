#!/bin/bash
# Rank a competition's test set with the two-stage re-ranker and write the zip.
#
#     sbatch ada/predict-rerank.sbatch --dataset mind
#     sbatch ada/predict-rerank.sbatch --dataset mind --bench-chunks   # size first
#     sbatch ada/predict-rerank.sbatch --dataset ebnerd --chunk 200000
#
# Arguments go to `python -m pipeline.predict`; this wrapper adds placement and
# the `--retriever rerank` that makes it the re-ranker's submission rather than
# the semantic one's.
#
# Why this is its own job and not `build.sbatch predict`. The rebuild's predict
# stage ranks with the registry's default retriever over the pipeline's own
# artifacts. This ranks 2.37M MIND impressions or 13.5M EB-NeRD ones through
# the full two-stage path -- a BM25 index and a FAISS index over the
# *competition's* catalogue, a counter store spanning two periods, an NRMS
# forward pass and a LightGBM predict per chunk. It is the longest-running
# thing in A2 and the only one whose input is an archive the rebuild never
# downloads.
#
# Run the benchmark first. `--bench-chunks` ranks only the first chunk at each
# of three sizes and records what each cost, writing no submission; the full
# run then gets `--chunk <the largest that fit with headroom>`. A1's lesson is
# the reason: what takes the node out is not the impressions in a chunk but the
# histories they carry, and EB-NeRD's test histories average 144 clicks over
# 808k users.

#SBATCH --job-name=vub-predict-rerank
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log

# More memory than build.sbatch and no GPU. The competition's catalogue is
# indexed in memory for the whole job and the counter store spans the training
# period *and* the test log, which is the pair that sets the floor. NRMS scores
# on the CPU here: `--device cuda` is measured against it by --bench-chunks on
# one chunk, and a GPU job that waits an hour in the queue to save forty
# minutes of scoring has not saved anything. Add `--gres=gpu:1` and pass
# `--device cuda` if that measurement says otherwise.

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
# The whole data root, because this job reads three things that live in it: the
# competition's test archive (which `acquire.ensure` downloads on demand if it
# is not there), the feature store's counters and catalogue, and the embedding
# artifact. Node-local for the same reason the build stages in -- these are
# streaming reads over millions of rows and NFS is the wrong disk for them.
mkdir -p "$STAGE" "$RESULTS"
for directory in data feature_store; do
    if [ -d "$HOME/vub-news-data/$directory" ]; then
        echo "staging $directory in ..."
        /usr/bin/time -f "  staged in %e s" \
            rsync -a "$HOME/vub-news-data/$directory" "$STAGE/"
    fi
done
df -h "$STAGE" | tail -1

# --- stage out --------------------------------------------------------------
# The archive the test set was extracted from comes back, because acquiring it
# is a 1.6 GB download and /scratch is purged weekly. The prediction file and
# its zip were written straight to $RESULTS/predictions and are already on
# /home.
#
# In a trap for the same reason build.sbatch's is: this script is `set -e`, and
# a run that fails after downloading two gigabytes has still earned the
# download. No `--delete` on either path -- nothing here is a mirror of
# something authoritative, so deleting can only lose.
stage_out() {
    local status=$?
    if [ -d "$STAGE/data" ]; then
        echo "staging data out ..."
        mkdir -p "$HOME/vub-news-data"
        rsync -a "$STAGE/data" "$HOME/vub-news-data/"
    fi
    echo "finished    $(date -Is)"
    echo "exit        $status"
    echo "results in  $RESULTS"
    # The md5 of every zip, printed here so the other end of the transfer has
    # something to compare against. `rsync --no-compress --whole-file` brings
    # them back; a zip that does not match on both ends is not uploaded.
    md5sum "$RESULTS"/predictions/*/*.zip 2>/dev/null || true
    # Per-stage rows/s -- read, features, nrms, gbdt, write -- which is what
    # ticket 09's 10x argument extrapolates from.
    grep -h '"stage": "submit"' "$RESULTS/artifacts/tradeoffs.jsonl" | tail -20 || true
}
trap stage_out EXIT
trap 'exit 143' TERM

# --- run --------------------------------------------------------------------
export VUB_NEWS_DATA_ROOT="$STAGE"
export VUB_NEWS_ROOT="$RESULTS"
cd "$REPO"

echo "--- predict --retriever rerank $* ---"
/usr/bin/time -v "$ENV_PY" -m pipeline.predict --retriever rerank "$@" \
    2> >(tee "$RESULTS/last-predict-rerank.time" >&2)
