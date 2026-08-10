"""Ranking metrics over the shared ranked-candidates shape.

The harness knows nothing about BM25 or embeddings. It asks a retriever module
for `rank_candidates` and scores whatever comes back, so a third retriever
needs no change here — only an entry in RETRIEVERS.

    python -m pipeline.evaluate                                  every combination
    python -m pipeline.evaluate --dataset mind --retriever bm25  one of them
    python -m pipeline.evaluate --split test                     the held-back split

Every run prints the same columns in the same order and writes one json per
scored run, so tickets 11 and 12 aggregate from disk rather than re-ranking.

Metrics are computed over an impression's own candidate list, not over the
corpus: recall@K in tickets 6 and 8 asks whether a global retriever surfaces a
clicked article at all, while AUC, MRR and nDCG ask whether it orders the
candidates the competition supplies. The two answer different questions and
neither replaces the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline import ann_index, bm25_index, paths
from pipeline.datasets import DATASETS, DatasetConfig

# The retrievers the harness can score. The values are modules, each exposing
# rank_candidates(config, behaviors, history) -> ranked candidates.
RETRIEVERS = {"bm25": bm25_index, "ann": ann_index}

# nDCG cut-offs, per SPEC.
NDCG_DEPTHS = (5, 10)

# Metrics are reported on these; train is refused outright.
VALIDATION, TEST = "validation", "test"
SCORABLE = (VALIDATION, TEST)
TRAIN = "train"

# The four metrics the assignment asks for, named once so the table, the json
# and the nDCG depths cannot drift apart.
METRICS = ("auc", "mrr", *(f"ndcg@{depth}" for depth in NDCG_DEPTHS))

# The printed columns, in a fixed order. Tickets 11 and 12 aggregate from the
# json rather than by parsing this, but a stable table is what makes two runs
# diffable line by line — and it carries the degenerate-impression counts, so
# the metrics are never read without the population they were averaged over.
TEXT_COLUMNS = ("dataset", "retriever", "split")
COLUMNS = (
    *TEXT_COLUMNS,
    "impressions",
    "scored",
    "no_positive",
    "all_positive",
    "all_scores_tied",
    *METRICS,
)

# Where a scored run is written for tickets 11 and 12 to aggregate.
EVALUATE_DIR = "evaluate"


class EvaluationError(RuntimeError):
    """The harness was asked for something it must not answer."""


def reciprocal_rank(relevance: np.ndarray) -> float:
    """1 / the rank of the first relevant candidate, ranks counted from 1.

    The textbook definition: the first hit, not an average over every hit. It
    is worth being explicit because 29.6% of MIND's validation impressions have
    more than one positive, so a mean-over-all-positives MRR — which some
    leaderboard evaluators use — would give a visibly different number on the
    same predictions.
    """
    hits = np.flatnonzero(relevance)
    return 1.0 / (hits[0] + 1)


def dcg(relevance: np.ndarray, depth: int) -> float:
    """Binary-gain DCG: each relevant candidate contributes 1/log2(rank+1)."""
    top = relevance[:depth]
    return float((top / np.log2(np.arange(2, len(top) + 2))).sum())


def ndcg(relevance: np.ndarray, depth: int) -> float:
    """DCG against the best ordering the same labels could possibly achieve."""
    ideal = dcg(np.sort(relevance)[::-1], depth)
    return dcg(relevance, depth) / ideal if ideal else 0.0


def relevance_of(ranked_ids: list[str], labels: dict[str, int]) -> np.ndarray:
    """The label of each candidate, in the order the retriever ranked them.

    The ranking has to be a permutation of the candidate list: the leaderboard
    scores every candidate, and a retriever that quietly dropped some would
    score better here than it deserves — the dropped ones can only be misses.
    """
    if len(ranked_ids) != len(labels) or set(ranked_ids) != set(labels):
        raise EvaluationError(
            f"ranking is not a permutation of the candidates: "
            f"{len(ranked_ids)} ranked against {len(labels)} candidates"
        )
    return np.array([labels[article_id] for article_id in ranked_ids])


def metrics(ranked: pd.DataFrame, behaviors: pd.DataFrame) -> dict[str, float]:
    """AUC, MRR and nDCG over every impression that admits them.

    AUC is computed from the scores and MRR and nDCG from the order, which is
    the standard split: AUC is defined on score values and handles ties by
    construction, while a rank metric has to break them somehow. Ties are left
    in the order the retriever emitted, which for a stable sort is the order
    the candidates arrived in — never reordered to put a positive first.
    """
    truth = behaviors[["impression_id", "candidate_ids", "labels"]].merge(
        ranked[["impression_id", "ranked_ids", "scores"]], on="impression_id"
    )

    collected: dict[str, list[float]] = {"auc": [], "mrr": []}
    for depth in NDCG_DEPTHS:
        collected[f"ndcg@{depth}"] = []
    degenerate = {"no_positive": 0, "all_positive": 0, "all_scores_tied": 0}

    for _, candidates, labels, ranked_ids, scores in truth.itertuples(index=False):
        label_of = dict(zip(candidates, labels, strict=True))
        relevance = relevance_of(ranked_ids, label_of)

        positives = int(relevance.sum())
        if positives == 0:
            # AUC, MRR and nDCG are all undefined; averaging a zero in would
            # report the share of such impressions rather than any ranking.
            degenerate["no_positive"] += 1
            continue
        if positives == len(relevance):
            degenerate["all_positive"] += 1
            continue
        if len(set(scores)) == 1:
            # Every candidate scored the same, so the order is the one the
            # candidates arrived in. The rank metrics still get a number, but
            # it is a property of the competition's file, not the retriever.
            degenerate["all_scores_tied"] += 1

        collected["auc"].append(roc_auc_score(relevance, scores))
        collected["mrr"].append(reciprocal_rank(relevance))
        for depth in NDCG_DEPTHS:
            collected[f"ndcg@{depth}"].append(ndcg(relevance, depth))

    report = {name: float(np.mean(values)) if values else 0.0
              for name, values in collected.items()}
    report["scored"] = len(collected["auc"])
    report.update(degenerate)
    return report


def table(reports: list[dict]) -> str:
    """Reports as one fixed-column table, metrics to four decimals.

    Columns are padded to the widest cell rather than to a fixed width, so a
    table stays readable whatever the impression counts are, while the columns
    themselves and their order never change.
    """
    rows = [
        [f"{r[c]:.4f}" if c in METRICS else str(r[c]) for c in COLUMNS]
        for r in reports
    ]
    widths = [
        max(len(name), *(len(row[i]) for row in rows))
        for i, name in enumerate(COLUMNS)
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) if name in TEXT_COLUMNS else cell.rjust(width)
            for name, cell, width in zip(COLUMNS, cells, widths)
        ).rstrip()

    return "\n".join([line(list(COLUMNS)), *(line(row) for row in rows)])


def evaluate(
    config: DatasetConfig, retriever: str, split: str = VALIDATION
) -> dict[str, float]:
    """Score one retriever on one dataset and split."""
    if split == TRAIN:
        raise EvaluationError(
            f"refusing to score the {TRAIN} split. Metrics on data the "
            f"retriever was tuned against measure memorisation, and reporting "
            f"them is the self-deception the assignment's anti-gaming section "
            f"is about. Use one of: {', '.join(SCORABLE)}."
        )
    if split not in SCORABLE:
        raise EvaluationError(
            f"unknown split {split!r}; use one of: {', '.join(SCORABLE)}"
        )
    if retriever not in RETRIEVERS:
        raise EvaluationError(
            f"unknown retriever {retriever!r}; use one of: "
            f"{', '.join(RETRIEVERS)}"
        )

    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    history = pd.read_parquet(config.feature_store_dir / "history.parquet")
    impressions = behaviors[behaviors["split"] == split]

    ranked = RETRIEVERS[retriever].rank_candidates(config, impressions, history)
    report = metrics(ranked, impressions)
    return {
        "dataset": config.name,
        "retriever": retriever,
        "split": split,
        "impressions": len(impressions),
        **report,
    }


def save(report: dict, config: DatasetConfig) -> Path:
    """One json per scored run, for tickets 11 and 12 to aggregate."""
    directory = config.artifacts_dir / EVALUATE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report['retriever']}-{report['split']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def report_on(paths_written: list[Path], reports: list[dict]) -> None:
    """The table, then where each scored run was written."""
    print("\n" + table(reports) + "\n")
    for path in paths_written:
        print(f"  -> {path.relative_to(paths.ARTIFACTS_DIR.parent)}")
    print()


def run(config: DatasetConfig, force: bool = False) -> None:
    """The build stage: every retriever on the validation split.

    Test is not scored here. It is scored once, deliberately, by the command
    below — a split that is rebuilt into every `python build.py` is one that
    gets looked at repeatedly, which is how a held-back split stops being one.
    """
    reports = [evaluate(config, retriever, VALIDATION) for retriever in RETRIEVERS]
    report_on([save(report, config) for report in reports], reports)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.evaluate",
        description="Score a retriever on a dataset and split.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="restrict to one dataset (repeatable); default is all of them",
    )
    parser.add_argument(
        "--retriever",
        action="append",
        choices=sorted(RETRIEVERS),
        help="restrict to one retriever (repeatable); default is all of them",
    )
    # Deliberately not an argparse choice: `--split train` has to reach
    # evaluate() and be turned down with its reason, not with "invalid choice".
    parser.add_argument(
        "--split",
        default=VALIDATION,
        help=f"which split to score: {' or '.join(SCORABLE)} "
        f"(default: {VALIDATION})",
    )
    args = parser.parse_args(argv)

    configs = [DATASETS[name] for name in (args.dataset or sorted(DATASETS))]
    retrievers = args.retriever or sorted(RETRIEVERS)

    reports: list[dict] = []
    written: list[Path] = []
    try:
        for config in configs:
            for retriever in retrievers:
                report = evaluate(config, retriever, args.split)
                reports.append(report)
                written.append(save(report, config))
    except EvaluationError as error:
        print(f"\nerror: {error}\n", file=sys.stderr)
        return 2

    report_on(written, reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
