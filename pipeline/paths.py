"""Filesystem layout. Every path in the project is derived from here.

Three roots, because a cluster does not have one filesystem. REPO_ROOT is where
the code and its credentials live and is never overridden. ROOT is where small
outputs go — artifacts, predictions, checkpoints — and DATA_ROOT is where the
large ones go. Both default to REPO_ROOT, so a single machine sees exactly the
layout it always did.

The split matters because the two have opposite requirements: on Ada the only
shared filesystem a compute node can see is /home at 25 GB, while the space with
room for the data is node-local /scratch, which is purged weekly. Checkpoints
are kilobytes recording work worth hours and must not live on the purging disk;
the feature store is gigabytes and cannot live anywhere else.
"""

import os
from pathlib import Path

# Where the code is. Fixed, because .env sits beside it.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _root(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


# Small outputs: artifacts, predictions, checkpoints.
ROOT = _root("VUB_NEWS_ROOT", REPO_ROOT)
# Large outputs: downloaded archives and the feature store.
DATA_ROOT = _root("VUB_NEWS_DATA_ROOT", ROOT)

RAW_DIR = DATA_ROOT / "data" / "raw"
FEATURE_STORE_DIR = DATA_ROOT / "feature_store"
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS_DIR = ROOT / "predictions"
CHECKPOINT_DIR = ROOT / ".checkpoints"


def load_env_file() -> None:
    """Read credentials from .env, so they survive across shells.

    An already-exported variable wins, so `HF_TOKEN=... python build.py` still
    does what it looks like it does. Every entry point that can reach a gated
    source calls this — the rebuild and the submission both do, and a token
    that only one of them can see is a download that fails depending on which
    command was typed.

    Read from REPO_ROOT rather than ROOT: the credential belongs with the
    checkout, not with wherever this run was told to write its outputs.
    """
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
