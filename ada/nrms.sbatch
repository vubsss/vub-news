#!/bin/bash
# Fit NRMS-DocVec, or sweep the grid that chooses its shape, on a GPU node.
#
#     sbatch ada/nrms.sbatch                           # both datasets, the registry's cell
#     sbatch ada/nrms.sbatch --dataset mind --grid     # the tune grid, one axis at a time
#
# Arguments go to `python -m pipeline.nrms`; this wrapper adds placement, a
# card, and the environment that can use it.
#
# Why a job at all. Everything else A2 trains is a gradient-boosted tree on a
# laptop; this is the one stage with a backward pass. The model itself is small
# -- two linears and a 16-head attention over 80 vectors, a few hundred
# thousand parameters -- but the grid is six axes swept one at a time, each a
# full fit, and the checkpoint has to exist before the re-ranker of ticket 06
# can take its score as a feature. A card turns that from an evening into
# minutes, and `--device cuda` is the only difference in the code path: the
# gather stays on the host, the batch moves, and the checkpoint comes back to
# the CPU before it is saved so a laptop can load it.
#
# Placement follows encode.sbatch rather than scale.sbatch. This reads the
# feature store and one .npy and writes a checkpoint of a few megabytes, so
# nothing is staged onto node-local /scratch: copying a 564 MB feature store to
# read two parquet files out of it would cost more than the read. The model
# cache is not needed here at all -- the vectors are already on disk, produced
# by encode.sbatch, and nothing in this stage calls a transformer.

#SBATCH --job-name=vub-nrms
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --exclude=gnode066
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.log

# gnode066 has no /dev/nvidia-uvm, so cuInit returns 999 there and CUDA never
# starts -- while nvidia-smi answers normally, which is what makes it look like
# an environment problem rather than a broken node.

set -euo pipefail

REPO="$HOME/vub-news"
RESULTS="$HOME/vub-news-run"
ENV_PY="$HOME/envs/vub-news-gpu/bin/python"

echo "node        $(hostname)"
echo "job         ${SLURM_JOB_ID:-none}"
echo "started     $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

[ -x "$ENV_PY" ] || { echo "no GPU environment at $ENV_PY — run ada/env-gpu.sbatch first"; exit 1; }

export VUB_NEWS_DATA_ROOT="$HOME/vub-news-data"
export VUB_NEWS_ROOT="$RESULTS"
cd "$REPO"

echo "--- nrms $* ---"
/usr/bin/time -v "$ENV_PY" -m pipeline.nrms --device cuda "$@" \
    2> >(tee "$RESULTS/last-nrms.time" >&2)

echo "finished    $(date -Is)"
# Every cell that ran is a ledger row; the checkpoints are what ticket 06 reads.
grep -h '"stage": "nrms"' "$RESULTS/artifacts/tradeoffs.jsonl" | tail -20 || true
ls -la "$RESULTS"/artifacts/*/nrms/ 2>/dev/null || true
