"""Filesystem layout. Every path in the project is derived from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
FEATURE_STORE_DIR = ROOT / "feature_store"
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS_DIR = ROOT / "predictions"
CHECKPOINT_DIR = ROOT / ".checkpoints"
