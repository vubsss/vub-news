"""Pipeline stages, in dependency order, and their completion checkpoints.

A stage's `run` is None until the ticket that implements it lands. The build
reports those stages as "not built" and skips them, so the entry point stays
runnable while the pipeline is assembled ticket by ticket.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipeline import (
    acquire,
    ann_index,
    bm25_index,
    embed,
    evaluate,
    ingest,
    paths,
    preprocess,
    split,
)
from pipeline.datasets import DatasetConfig


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    # Called as run(config, force). A stage that skips itself when its outputs
    # already exist must do that work anyway when force is set — otherwise
    # `build.py --force <stage>` would silently do nothing.
    run: Callable[[DatasetConfig, bool], None] | None = None


STAGES = (
    Stage("acquire", "download and extract raw archives", acquire.run),
    Stage("ingest", "raw files -> unified schema feature store", ingest.run),
    Stage(
        "split", "temporal train/val/test split + leakage guards", split.run
    ),
    Stage(
        "preprocess",
        "build lexical_text for the dataset's language",
        preprocess.run,
    ),
    Stage("bm25", "build BM25 index, report recall@K", bm25_index.run),
    Stage("embed", "obtain article embeddings", embed.run),
    Stage("ann", "build FAISS index, report recall@K", ann_index.run),
    Stage("evaluate", "ranking metrics per retriever", evaluate.run),
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
