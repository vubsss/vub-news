#!/bin/bash
# Build the CUDA environment phase 7 encodes with, on a GPU compute node.
#
#     sbatch ada/env-gpu.sbatch
#
# Separate from `ada/env.sbatch` on purpose. That one is deliberately CPU-only:
# eight of the pipeline's nine stages never touch a GPU, and keeping the CUDA
# runtime out of it keeps it small against a 30 GB quota. Only `embed` wants a
# card, and only when a dataset generates its own vectors rather than loading a
# cached artifact -- which on this project is MIND, in this phase.
#
# On a GPU node rather than any compute node, because the thing most worth
# checking is not that torch imports: it is that the wheel carries kernels for
# the cards this cluster actually has.
#
# **The fleet is mixed and SLURM will not let you choose.** Its gres is plain
# `gpu:N` with no model in it, so a job takes whatever the node has: measured
# on 2026-08-23, gnode004/005/025 are GTX 1080 Ti and gnode048/066 are RTX
# 2080 Ti. That is Pascal `sm_61` and Turing `sm_75`, and the wheel has to
# carry both or the job dies at its first kernel launch on half the cluster.
#
# Which rules out the default PyPI build. `torch==2.13.0` from PyPI is
# `+cu130`, and CUDA 13 dropped Pascal: its arch list is sm_75 upward, so it
# cannot run on three of the five nodes above. `+cu126` is the same torch
# against a CUDA that still ships sm_61.
#
# The version stays `2.13.0` on purpose. `requirements.txt` pins it, and a
# specifier with no local version admits any local version -- so installing
# the cu126 build *first* leaves the requirements install below satisfied and
# it does not overwrite. Pinning a different version instead is what broke the
# first attempt at this: 2.5.1 does not satisfy ==2.13.0, so pip quietly
# upgraded it back to the cu130 build.
#
# Pascal has no tensor cores and runs fp16 at 1/64 rate on GP102, so autocast
# would make encoding *slower* on the nodes we are most likely to land on.
# Whatever runs here runs fp32.
#
# gnode066 is excluded: its `/dev/nvidia-uvm` is missing, so `cuInit` returns
# 999 and CUDA cannot start at all there, however good the wheel is. nvidia-smi
# works on it, which is what makes the failure confusing rather than obvious.

#SBATCH --job-name=vub-env-gpu
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --exclude=gnode066
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

SYSTEM_PY="/usr/local/apps/python-3.12.4/bin/python3"
ENV_DIR="$HOME/envs/vub-news-gpu"
REPO="$HOME/vub-news"

echo "node       $(hostname)"
echo "os         $(grep -oP '(?<=PRETTY_NAME=").*(?=")' /etc/os-release)"
echo "started    $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

[ -x "$SYSTEM_PY" ] || { echo "no python 3.12 at $SYSTEM_PY"; exit 1; }

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "creating venv ..."
    "$SYSTEM_PY" -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --no-input -q --upgrade pip

# The download is worth timing every run. `ada/env.sbatch` records PyPI at
# 0.22 MB/s, measured the hard way in phase 6 -- 28 minutes for 200 MB. The
# same host on 2026-08-23 pulled the CUDA stack in 234 s and the rest of the
# requirements in 441 s. The link is not a constant, so the number below is
# the one that describes *this* run rather than a figure to plan against.
echo "--- torch, from the cu126 index ---"
/usr/bin/time -f "  torch installed in %e s" \
    "$ENV_DIR/bin/python" -m pip install --no-input -q \
        --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0

echo "--- the rest, from PyPI ---"
/usr/bin/time -f "  requirements installed in %e s" \
    "$ENV_DIR/bin/python" -m pip install --no-input -q -r "$REPO/requirements.txt"

# gensim is an encode-time dependency and lives here rather than in
# requirements.txt: only the word2vec variant needs it, and only while it is
# being produced. Reading the .npy afterwards needs nothing.
echo "--- gensim, for the word2vec variant ---"
"$ENV_DIR/bin/python" -m pip install --no-input -q gensim

echo "--- does this wheel have kernels for this card? ---"
"$ENV_DIR/bin/python" - <<'PY'
import torch
print("torch        ", torch.__version__)
print("cuda runtime ", torch.version.cuda)
print("available    ", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA device visible from the job"

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
print("device       ", name, f"sm_{major}{minor}")
print("arch list    ", " ".join(arches))

# Both, not just this node's. The encode job cannot choose its card, so a
# wheel that works here and not on the Pascal half of the fleet is a job that
# fails whenever the scheduler feels like it.
for needed in ("sm_61", "sm_75"):
    assert needed in arches, (
        f"this wheel ships {arches} and the cluster has {needed} cards: "
        f"the job would fail on whichever nodes it lands on. Use a CUDA 12 "
        f"build -- CUDA 13 dropped Pascal."
    )

# Not a formality: an arch list can match and the kernels still be absent.
a = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
print("matmul       ", float((a @ a).sum()) == float((a @ a).sum()), "- kernels launch")
print("vram total   ", f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
PY

echo "--- size against the 30 GB /home quota ---"
du -sh "$ENV_DIR"
quota -s 2>/dev/null | tail -2 || true
echo "finished   $(date -Is)"
