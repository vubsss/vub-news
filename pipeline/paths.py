"""Filesystem layout. Every path in the project is derived from here."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
FEATURE_STORE_DIR = ROOT / "feature_store"
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
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
