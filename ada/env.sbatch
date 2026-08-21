#!/bin/bash
# Build the Python environment for this project on Ada, on a compute node.
#
#     sbatch ada/env.sbatch
#
# On a *compute* node deliberately, not on the login node. Ada's master runs
# CentOS 7 with glibc 2.17 and its u22 compute nodes run Ubuntu 22.04 with
# glibc 2.35 — an environment resolved against the master's libc is not the one
# the jobs would run under, and the failure would appear as an import error
# inside a queued job rather than here.
#
# Ada's *system* Python is 3.10.12 and this project pins numpy 2.3.5, which
# needs 3.11 or newer. The cluster provides 3.12.4 under /usr/local/apps, which
# is what this uses -- referenced by path rather than through `module load`,
# because the module command is not initialised in a non-interactive batch
# shell and silently leaves you on 3.10.
#
# Downloading an interpreter was the first plan and does not work here:
# **github.com is unreachable from Ada** even though pypi.org is, so the
# miniforge installer never arrives. pip works; release tarballs from GitHub do
# not.
#
# Everything lands in $HOME because that is the only shared filesystem a compute
# node can see. /share1 is larger and invisible from here; /scratch is visible
# and node-local, so an environment built there would be gone next week and on
# another node today.

#SBATCH --job-name=vub-env
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

# The cluster's 3.12, by path. `module load` does not survive into a batch
# shell, and the failure mode is silent: you get 3.10 and a numpy that will not
# install.
SYSTEM_PY="/usr/local/apps/python-3.12.4/bin/python3"
ENV_DIR="$HOME/envs/vub-news"
REPO="$HOME/vub-news"

echo "node       $(hostname)"
echo "os         $(grep -oP '(?<=PRETTY_NAME=").*(?=")' /etc/os-release)"
echo "glibc      $(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')"

[ -x "$SYSTEM_PY" ] || { echo "no python 3.12 at $SYSTEM_PY"; exit 1; }
echo "python     $("$SYSTEM_PY" -V 2>&1)"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "creating venv ..."
    "$SYSTEM_PY" -m venv "$ENV_DIR"
fi

echo "installing pinned requirements ..."
"$ENV_DIR/bin/python" -m pip install --no-input -q --upgrade pip
"$ENV_DIR/bin/python" -m pip install --no-input -q -r "$REPO/requirements.txt"

echo "--- versions ---"
"$ENV_DIR/bin/python" - <<'PY'
import importlib.metadata as md, sys
print("python", sys.version.split()[0])
for name in ("numpy", "pandas", "pyarrow", "bm25s", "faiss-cpu", "scikit-learn", "torch"):
    try:
        print(f"{name:14} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name:14} MISSING")
PY

echo "--- size against the 30 GB /home quota ---"
du -sh "$ENV_DIR"

echo "--- the suite, on the node that will run the pipeline ---"
cd "$REPO" && "$ENV_DIR/bin/python" -m pytest tests/ -q 2>&1 | tail -3
