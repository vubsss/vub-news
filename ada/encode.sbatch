#!/bin/bash
# Encode MIND's alternate embedding variants on a GPU node.
#
#     sbatch ada/encode.sbatch                        # every variant not yet on disk
#     sbatch ada/encode.sbatch --variant bge-base-en-v1.5 --force
#
# Arguments go to `python -m pipeline.encode_variants`, which is the whole of
# what this wrapper adds to: placement, a card, and the environment that can
# use it. MIND ships no vectors, so every variant phase 7 compares is produced
# here -- where EB-NeRD's four arrived in its release.
#
# Unlike `build.sbatch` this stages nothing onto /scratch. It reads one parquet
# (the catalogue) and one id index, and writes one .npy per variant straight to
# $HOME/vub-news-run/artifacts, which is where the pipeline already looks for
# them. Copying a 564 MB feature store to node-local disk to read a single file
# out of it would cost more than the read.
#
# The model cache does go to /scratch. GoogleNews word2vec alone is 3.6 GB and
# the three transformers another 1.3 GB, against a 30 GB /home quota -- and the
# Hub serves them at 29 MB/s from here, measured, so re-fetching them on a
# later run costs about three minutes and no quota at all.

#SBATCH --job-name=vub-encode
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --exclude=gnode066
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

export HF_HOME="/scratch/$USER/hf"
mkdir -p "$HF_HOME"

# The catalogue is read straight from /home; only the model cache is node-local.
export VUB_NEWS_DATA_ROOT="$HOME/vub-news-data"
export VUB_NEWS_ROOT="$RESULTS"
cd "$REPO"

echo "--- encode_variants $* ---"
/usr/bin/time -v "$ENV_PY" -m pipeline.encode_variants \
    --dataset mind --device cuda "$@" 2> >(tee "$RESULTS/last-encode.time" >&2)

echo "finished    $(date -Is)"
ls -la "$RESULTS/artifacts/mind/" | grep -E "embeddings.*\.npy" || true
