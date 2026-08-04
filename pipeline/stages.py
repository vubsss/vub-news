"""Pipeline stages, in dependency order, and their completion checkpoints.

A stage's `run` is None until the ticket that implements it lands. The build
reports those stages as "not built" and skips them, so the entry point stays
runnable while the pipeline is assembled ticket by ticket.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipeline import paths
from pipeline.datasets import DatasetConfig


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    run: Callable[[DatasetConfig], None] | None = None


STAGES = (
    Stage("acquire", "download and extract raw archives (ticket 2)"),
    Stage("ingest", "raw files -> unified schema feature store (ticket 3)"),
    Stage("split", "temporal train/val/test split + leakage guards (ticket 4)"),
    Stage("preprocess", "build lexical_text for the dataset's language (ticket 5)"),
    Stage("bm25", "build BM25 index, report recall@K (ticket 6)"),
    Stage("embed", "obtain article embeddings (ticket 7)"),
    Stage("ann", "build FAISS index, report recall@K (ticket 8)"),
    Stage("evaluate", "ranking and beyond-accuracy metrics (tickets 9, 10)"),
    Stage("predict", "generate CodaBench submission file (tickets 13, 14)"),
)

STAGES_BY_NAME = {stage.name: stage for stage in STAGES}


def checkpoint(stage: Stage, dataset: DatasetConfig) -> Path:
    return paths.CHECKPOINT_DIR / dataset.name / f"{stage.name}.done"


def is_done(stage: Stage, dataset: DatasetConfig) -> bool:
    return checkpoint(stage, dataset).exists()


def mark_done(stage: Stage, dataset: DatasetConfig) -> None:
    path = checkpoint(stage, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def clear(stage: Stage, dataset: DatasetConfig) -> None:
    checkpoint(stage, dataset).unlink(missing_ok=True)
