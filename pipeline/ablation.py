"""The arms: the same re-ranker with one thing taken away, each paired against
the full model.

The grade is on ablation rigour, so this module's whole job is to make "the
improvement is understood" a table rather than a claim. Every arm is
`pipeline.rerank` with a different `RerankSpec` -- a tier dropped, a family of
columns dropped by name, the NRMS score withheld, the counters read over the
whole log, a literal top-K cut applied -- retrained, rescored, and subtracted
from the full arm impression by impression with a bootstrap interval on the
difference.

**Paired, because unpaired intervals cannot settle anything here.** Two arms
score the same impressions in the same order, so subtracting per impression
cancels the variance that comes from the impressions themselves -- which on
this data is most of the width of an unpaired interval. Two overlapping
per-arm intervals would be the same non-answer phase 3 of A1 ran into.

**One frame, read once.** The arms differ in which columns they read, not in
what is in them, so the split's frame is read a single time and each arm is a
projection of it. No arm rebuilds features; the builder is not called at all.
That is the opposite of `rerank.rank_candidates`, which computes the frame
because it is the serving path -- and `test_ablation` checks the two agree, so
the cheap path here is a read of the same numbers rather than a second
definition of them.

**The bridge.** The literal-cut arms say what a hard two-stage cut costs, and
`retrieval.recall_at_k` on the same split says why: a cut can only lose what
the candidate generator failed to surface. Both come out of one batched corpus
search, and they are printed next to each other so the cut row is read against
its own ceiling.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import (
    bench,
    embed_compare,
    evaluate,
    features,
    ingest,
    ledger,
    rerank,
    retrieval,
    three_way,
    timings,
)
from pipeline.datasets import DATASETS, DEFAULT_DATASETS, DatasetConfig, RerankSpec

STAGE = "ablation"
RESULTS = "ablation-{split}.jsonl"
DOCUMENT = "ablation-{split}.md"

FULL = "full"

# The retrievers the side-by-side table reports, in the order the project built
# them: the two A1 parents, their fusion, the reproduced baseline, the
# re-ranker. Not sorted by score -- a table that reorders itself per dataset is
# harder to read across datasets.
ORDER = ("bm25", "ann", "hybrid", "nrms", "rerank")


class AblationError(RuntimeError):
    """An arm was asked for that the frame or the registry cannot express."""


# Where a cut arm's corpus ranks come from. The flat index is what every
# reported number in the project was measured on; the IVF is what a deployment
# would serve at ten times the catalogue, and the arm exists so that ticket
# 09's "IVF is 7-15x faster" has a functional column beside it rather than a
# latency with no cost attached.
FLAT, IVF = "flat", "ivf"


@dataclass(frozen=True)
class Arm:
    """One row of the table: a name, why it is there, and the spec that is it.

    `ranks` is the one thing an arm can vary that is not a `RerankSpec` field,
    because it is not a property of the model: it is which index supplied the
    top-K the cut is taken against.
    """

    name: str
    question: str
    changes: dict
    ranks: str = FLAT


# The functional arms. Leave-one-out first, because that is what "which tier
# earns its place" means; then the cumulative build-up, which is what a team
# deciding how much of this to build would actually ask; then the two arms that
# are about serving and about leakage rather than about features.
def arms_for(spec: RerankSpec) -> tuple[Arm, ...]:
    tiers = tuple(features.FEATURE_GROUPS)
    return (
        Arm(FULL, "everything the registry chose", {}),
        *(
            Arm(
                f"-{tier}",
                f"what the {tier} tier is worth",
                {"groups": tuple(name for name in tiers if name != tier)},
            )
            # `content` is not dropped whole: without the retriever scores and
            # the category match there is nothing left for a re-ranker to
            # re-rank, and the row would measure the absence of a candidate
            # generator rather than the value of a tier.
            for tier in tiers
            if tier != "content"
        ),
        Arm(
            "-session/dwell",
            "the columns only one of the two datasets has",
            {
                "drop": (
                    "session_impressions",
                    "session_clicks",
                    "past_dwell",
                    "past_depth",
                )
            },
        ),
        Arm("-nrms", "whether the baseline is worth serving inside", {"nrms": False}),
        Arm(
            "-retriever-scores",
            "whether stage one's scores carry the ranking",
            {"drop": ("ann_score", "ann_rank", "bm25_score", "bm25_rank")},
        ),
        *(
            Arm(
                "+".join(tiers[: depth + 1]),
                "the cumulative build-up, in serving order",
                {"groups": tiers[: depth + 1]},
            )
            for depth in range(len(tiers) - 1)
        ),
        *(
            Arm(
                f"cut@{depth}",
                "what a hard stage-one cut costs at this K",
                {"top_k": depth},
            )
            for depth in retrieval.DEPTHS
        ),
        *(
            Arm(
                f"cut@{depth} ({IVF})",
                "what an approximate index loses at the same K",
                {"top_k": depth},
                ranks=IVF,
            )
            for depth in retrieval.DEPTHS
        ),
        Arm("leaky", "Q9, at the registry's window", {"causal": False}),
        Arm(
            "clean-cumulative",
            "the pair the cumulative leak is measured against",
            {"window": "cumulative"},
        ),
        Arm(
            "leaky-cumulative",
            "Q9, where the leak is largest: counters run to the end of the log",
            {"causal": False, "window": "cumulative"},
        ),
    )


def spec_for(base: RerankSpec, arm: Arm) -> RerankSpec:
    return dataclasses.replace(base, **arm.changes)


# ---------------------------------------------------------------------------
# Scoring an arm off one frame.


@dataclass
class Split:
    """The split's frames, its labels and its corpus ranks, read once.

    Two frames per run at most -- the causal one and, if a leaky arm is in the
    list, the leaky one. Everything else is a projection. `ranks` holds one
    lookup per index the cut arms ask for.
    """

    impressions: pd.DataFrame
    frames: dict[bool, pd.DataFrame]
    ranks: dict[str, dict[str, dict[str, float]]]

    def frame(self, causal: bool) -> pd.DataFrame:
        if causal not in self.frames:
            raise AblationError(
                f"the {'causal' if causal else 'leaky'} frame for this split "
                f"was not read: build it with `python -m pipeline.features "
                f"{'' if causal else '--leaky'}`"
            )
        return self.frames[causal]


def approximate_ranks(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    depth: int,
    history_k: int = retrieval.HISTORY_K,
) -> dict[str, dict[str, float]]:
    """The same corpus search through the IVF index `bench` benchmarks.

    Same vectors, same queries, same depth as `rerank.global_ranks` -- the one
    difference is the index, so the arm's AUC against the flat arm's is what
    the approximation costs, and `bench`'s latency table is what it buys.
    """
    from pipeline import ann_index, embed

    embeddings = embed.load(config)
    wanted = set(behaviors["impression_id"])
    clicks = history[history["impression_id"].isin(wanted)]
    queries, _ = ann_index.build_user_vectors(clicks, embeddings, history_k)

    index = bench.ivf_index(np.ascontiguousarray(embeddings.vectors, dtype="float32"))
    asked = [row for row, vector in enumerate(queries.vectors) if vector.any()]
    found: dict[str, dict[str, float]] = {
        impression: {} for impression in queries.impression_ids
    }
    if asked:
        _, positions = index.search(
            np.ascontiguousarray(queries.vectors[asked]), depth
        )
        for row, position in zip(asked, positions):
            found[queries.impression_ids[row]] = {
                embeddings.article_ids[article]: place + 1.0
                for place, article in enumerate(position)
                if article >= 0
            }
    return found


def read_split(
    config: DatasetConfig,
    split: str,
    specs: list[RerankSpec],
    depth: int = max(retrieval.DEPTHS),
    ranks: bool = True,
    indexes: tuple[str, ...] = (FLAT,),
) -> Split:
    """The split's frames and corpus ranks, once for the whole run.

    One batched corpus search at the deepest K; every cut arm slices the same
    ranking, and `retrieval.recall_at_k` reads it too -- so the cut rows and
    the recall table describe one retrieval rather than two.
    """
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == split]
    if impressions.empty:
        raise AblationError(f"{config.name} has no {split} impressions")
    history = ingest.history_for(config, impressions)

    wanted = {spec.causal for spec in specs}
    precision = {spec.precision for spec in specs}
    if len(precision) > 1:
        raise AblationError(
            f"one run reads one frame: these arms ask for {sorted(precision)}. "
            f"Run the precisions as two ablations, or they are not paired."
        )
    scores = (
        rerank.nrms_scores(config, split, impressions)
        if any(spec.nrms for spec in specs)
        else None
    )

    frames = {}
    for causal in sorted(wanted, reverse=True):
        frame = features.read(
            config, split, causal=causal, precision=next(iter(precision))
        )
        frames[causal] = frame if scores is None else rerank.with_nrms(frame, scores)
    return Split(
        impressions=impressions,
        frames=frames,
        # Only the split the arms are *scored* on needs them: a corpus search
        # over the training half would be minutes spent on a column no arm
        # reads there. One search per index, at the deepest K; every cut arm
        # slices the same ranking.
        ranks=searched(config, split, impressions, history, depth, indexes)
        if ranks
        else {},
    )


def searched(
    config: DatasetConfig,
    split: str,
    impressions: pd.DataFrame,
    history: pd.DataFrame,
    depth: int,
    indexes: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    """One corpus search per index, timed, at the deepest K.

    Recorded as its own ledger row rather than folded into the cut arms'
    `p50_ms`: the search happens once for all three depths, and charging each
    cut row a third of it -- or all of it -- would be an allocation rather than
    a measurement. The row says what a stage-one search costs per impression,
    and ticket 09 reads it beside the re-ranker's own milliseconds.
    """
    found = {}
    for name in indexes:
        with timings.sample() as measured:
            found[name] = (
                rerank.global_ranks(config, impressions, history, depth)
                if name == FLAT
                else approximate_ranks(config, impressions, history, depth)
            )
        ledger.record(
            {
                "dataset": config.name,
                "stage": STAGE,
                "variant": f"corpus-search-{name}",
                "split": split,
                "train_seconds": round(measured["seconds"], 2),
                "peak_rss_mb": round(measured["peak_rss_mb"], 1),
                "p50_ms": 1000 * measured["seconds"] / max(len(impressions), 1),
                "rows_per_s": len(impressions) / max(measured["seconds"], 1e-9),
                "note": (
                    f"one batched search at depth {depth} over "
                    f"{len(impressions):,} impressions, shared by every cut arm; "
                    f"p50 is the per-impression mean, not a per-request sample"
                ),
            }
        )
    return found


def score(booster, split: Split, spec: RerankSpec, index: str = FLAT) -> pd.DataFrame:
    """One arm's ranking of the split, from the frame already in memory."""
    frame = split.frame(spec.causal)
    if spec.top_k is not None and index not in split.ranks:
        raise AblationError(
            f"the {index} ranks were not searched for this split, so a cut "
            f"against them cannot be scored"
        )
    return rerank.ranked_from(
        frame,
        rerank.score_frame(booster, frame, spec),
        spec,
        rerank.ranks_for(frame, split.ranks[index]) if spec.top_k is not None else None,
    )


def metrics_of(split: Split, ranked: pd.DataFrame) -> dict[str, np.ndarray]:
    return evaluate.per_impression_metrics(
        ranked, rerank.labels_for(split.impressions, ranked)
    )


def paired(
    arm: dict[str, np.ndarray], full: dict[str, np.ndarray], resamples: int
) -> dict:
    """One arm's difference from the full model, per metric, with an interval.

    The arm minus the full model, so a negative number is what the arm lost by
    dropping what it dropped -- which is the direction the note reads in.
    """
    found = {}
    for metric in evaluate.ACCURACY_METRICS:
        difference = arm[metric] - full[metric]
        low, high = embed_compare.interval(difference, resamples)
        found[f"{metric}_gap"] = (
            float(difference.mean()) if len(difference) else 0.0
        )
        found[f"{metric}_gap_lo"], found[f"{metric}_gap_hi"] = low, high
    return found


# ---------------------------------------------------------------------------
# The run.


def run_arms(
    config: DatasetConfig,
    split: str,
    arms: tuple[Arm, ...] | None = None,
    resamples: int = 1000,
    fit_split: str = rerank.FIT_SPLIT,
) -> list[dict]:
    """Every arm trained, scored on `split`, and paired against the full model.

    Trained here rather than loaded: an arm is a different model, and reusing
    the full model's trees with a projected input would be scoring a model on
    features it was not fitted on -- which LightGBM would answer without
    complaining.
    """
    base = config.rerank
    arms = arms or arms_for(base)
    specs = [spec_for(base, arm) for arm in arms]
    if not any(arm.name == FULL for arm in arms):
        raise AblationError(
            f"every arm is a difference from `{FULL}`, so the run has to "
            f"include it"
        )

    indexes = tuple(dict.fromkeys(arm.ranks for arm in arms if arm.changes.get("top_k")))
    scored = read_split(config, split, specs, indexes=indexes or (FLAT,))
    # Early stopping always reads `tune`, whatever split the arms are scored
    # on. Stopping on the split a number is reported from is selection on the
    # reporting split, which is the rule this project has kept since A1 --
    # chosen on tune, reported on validation, test scored once.
    stopping = (
        scored
        if split == rerank.TUNE_SPLIT
        else read_split(config, rerank.TUNE_SPLIT, specs, ranks=False)
    )
    fitted = read_split(config, fit_split, specs, ranks=False)
    later = rerank.later_half(config)

    rows: list[dict] = []
    measured: dict[str, dict[str, np.ndarray]] = {}
    # A cut is applied when the ranking is produced, not when the trees are
    # grown, so every arm that differs only in `top_k` or in which index
    # supplied the ranks is the *same model*. Training it once and scoring it
    # several times is not a shortcut: retraining would produce the same trees
    # and invite a reader to think the difference between two cut rows
    # included a difference in fitting.
    trained: dict[str, tuple] = {}
    with timings.sample() as whole:
        for arm, spec in zip(arms, specs):
            started = time.perf_counter()
            key = rerank.variant_of(dataclasses.replace(spec, top_k=None))
            if key not in trained:
                fit = rows_of(fitted, spec, keep=later)
                stop = rows_of(stopping, spec)
                trained[key] = rerank.train(config, spec, fit, stop)
            booster, report = trained[key]
            here = rows_of(scored, spec)
            ranked = score(booster, scored, spec, arm.ranks)
            measured[arm.name] = metrics_of(scored, ranked)
            path = rerank.save(booster, config, spec)
            rows.append(
                {
                    "dataset": config.name,
                    "split": split,
                    "arm": arm.name,
                    "question": arm.question,
                    "variant": rerank.variant_of(spec),
                    "rounds": report["rounds"],
                    "columns": len(report["columns"]),
                    "model_bytes": path.stat().st_size,
                    "train_seconds": round(report["train_seconds"], 2),
                    "n": int(len(measured[arm.name]["auc"])),
                    **{
                        metric: float(measured[arm.name][metric].mean())
                        for metric in evaluate.ACCURACY_METRICS
                    },
                    **rerank.prediction_cost(booster, here, spec),
                    "seconds": round(time.perf_counter() - started, 1),
                }
            )
            print(
                f"    {arm.name:<24} auc {rows[-1]['auc']:.4f}  "
                f"{rows[-1]['rounds']:>4} rounds  {rows[-1]['seconds']:>5.1f}s",
                flush=True,
            )

    for row in rows:
        if row["arm"] == FULL:
            continue
        row["against"] = FULL
        row |= paired(measured[row["arm"]], measured[FULL], resamples)
    record(rows, config, split, whole)
    return rows


def rows_of(split: Split, spec: RerankSpec, keep: set[str] | None = None) -> rerank.Rows:
    """One arm's matrix, projected out of the frame already in memory.

    The projection is the same `rerank.feature_columns` the training read uses,
    so an arm scores on exactly the columns it was fitted on and in the same
    order -- which LightGBM would not check.
    """
    frame = split.frame(spec.causal)
    if keep is not None:
        frame = frame[frame["impression_id"].isin(keep)]
    columns = rerank.feature_columns(spec)
    return rerank.Rows(
        matrix=frame[list(columns)].to_numpy(dtype="float32"),
        labels=frame["label"].to_numpy(dtype="int32"),
        impression_ids=frame["impression_id"].to_numpy(),
        columns=columns,
    )


def record(rows: list[dict], config: DatasetConfig, split: str, whole: dict) -> None:
    """Every arm as a ledger row, and the whole ablation as one more.

    The last row is what the note needs to say what a full re-run costs: the
    arms are not free, and "we ran the ablation" is a claim with a wall time.
    """
    for row in rows:
        ledger.record(
            {
                "dataset": row["dataset"],
                "stage": STAGE,
                "variant": row["arm"],
                "split": row["split"],
                **{
                    metric: row[metric]
                    for metric in evaluate.ACCURACY_METRICS
                    if metric in row
                },
                "model_bytes": row["model_bytes"],
                "train_seconds": row["train_seconds"],
                "p50_ms": row["p50_ms"],
                "p99_ms": row["p99_ms"],
                "rows_per_s": row["rows_per_s"],
                "delta_vs": row.get("against"),
                "delta": row.get("auc_gap"),
                "delta_lo": row.get("auc_gap_lo"),
                "delta_hi": row.get("auc_gap_hi"),
                "note": f"{row['question']}; {row['columns']} columns, "
                f"{row['rounds']} rounds, `{row['variant']}`",
            }
        )
    ledger.record(
        {
            "dataset": config.name,
            "stage": STAGE,
            "variant": "whole-run",
            "split": split,
            "train_seconds": round(whole["seconds"], 2),
            "peak_rss_mb": round(whole["peak_rss_mb"], 1),
            "note": f"{len(rows)} arms, one frame read once per causality",
        }
    )


def recall(config: DatasetConfig, split: str) -> dict[str, float]:
    """Stage-one recall at each depth: the ceiling the cut arms sit under."""
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == split]
    history = ingest.history_for(config, impressions)
    from pipeline import ann_index

    ranked, _ = ann_index.retrieve_corpus(
        config, impressions, history, retrieval.HISTORY_K, max(retrieval.DEPTHS)
    )
    return retrieval.recall_at_k(ranked, impressions, retrieval.DEPTHS)


# ---------------------------------------------------------------------------
# The document.


def gap(row: dict, metric: str = "auc") -> str:
    if f"{metric}_gap" not in row:
        return "—"
    return (
        f"{row[f'{metric}_gap']:+.4f} "
        f"[{row[f'{metric}_gap_lo']:+.4f}, {row[f'{metric}_gap_hi']:+.4f}]"
    )


def arm_table(rows: list[dict]) -> list[str]:
    lines = [
        "| arm | auc | ndcg@10 | Δ auc vs full (paired) | cols | rounds | model | p50 ms |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['arm']}` | {row['auc']:.4f} | {row['ndcg@10']:.4f} | "
            f"{gap(row)} | {row['columns']} | {row['rounds']} | "
            f"{row['model_bytes'] / 1024:,.0f} KB | {row['p50_ms']:.2f} |"
        )
    return lines


def five_way(config: DatasetConfig, split: str) -> list[str]:
    """Every retriever the harness scores, from the stored evaluation reports.

    Read rather than re-ranked, through `three_way`'s loader, so this table
    cannot disagree with the harness -- it can only fail to find a report that
    was never written.
    """
    lines = [
        "| retriever | " + " | ".join(three_way.METRICS) + " |",
        "| --- |" + " ---: |" * len(three_way.METRICS),
    ]
    for retriever in ORDER:
        try:
            found = three_way.load(config, retriever, split)
        except three_way.MissingReport:
            lines.append(
                f"| `{retriever}` |"
                + " not scored |" * len(three_way.METRICS)
            )
            continue
        lines.append(
            f"| `{retriever}` | "
            + " | ".join(
                f"{found[metric]['value']:.4f} "
                f"[{found[metric]['lo']:.4f}, {found[metric]['hi']:.4f}]"
                for metric in three_way.METRICS
            )
            + " |"
        )
    return lines


def document(config: DatasetConfig, split: str, rows: list[dict], found: dict) -> str:
    lines = [
        f"# Ablation — {config.name}, {split} split",
        "",
        f"Every arm is the re-ranker with one thing changed, retrained and "
        f"rescored on `{split}`, and subtracted from `{FULL}` impression by "
        f"impression. The interval is a 95% bootstrap over "
        f"{rows[0]['n']:,} paired impressions; an interval that contains zero "
        f"is an arm this data cannot separate from the full model.",
        "",
        *arm_table(rows),
        "",
        "## The stage-one bridge",
        "",
        "The `cut@K` arms rank everything outside the semantic retriever's "
        "corpus top-K last, which is what a two-stage system actually serves. "
        "What such a cut can lose is bounded by what stage one surfaces at all:",
        "",
        "| depth | recall |",
        "| ---: | ---: |",
        *(
            f"| {depth} | {found[f'recall@{depth}']:.4f} |"
            for depth in retrieval.DEPTHS
        ),
        "",
        f"over {found['scored']:,} impressions with a click; "
        f"{found['no_positive']:,} had none and are not averaged in.",
        "",
        "## Every retriever, side by side",
        "",
        *five_way(config, split),
        "",
        "Read out of `artifacts/<dataset>/evaluate/<retriever>-"
        f"{split}.json`, so this table cannot disagree with the harness.",
        "",
        "## Q9: what the leak is worth",
        "",
        *q9(rows),
        "",
    ]
    return "\n".join(lines)


def q9(rows: list[dict]) -> list[str]:
    """The leaky arms against their clean pairs, in the note's own words."""
    by_arm = {row["arm"]: row for row in rows}
    lines = []
    for leaky, clean in (("leaky", FULL), ("leaky-cumulative", "clean-cumulative")):
        if leaky not in by_arm or clean not in by_arm:
            continue
        difference = by_arm[leaky]["auc"] - by_arm[clean]["auc"]
        lines.append(
            f"- `{leaky}` reads the same counters over the whole log and scores "
            f"{by_arm[leaky]['auc']:.4f} against `{clean}`'s "
            f"{by_arm[clean]['auc']:.4f}: {difference:+.4f}. "
            f"That gap is the part of the leaky model's quality that no server "
            f"could reproduce, because it comes from clicks that had not "
            f"happened when the impression was served."
        )
    return lines or ["- No leaky arm was run."]


def write(config: DatasetConfig, split: str, rows: list[dict], found: dict) -> Path:
    directory = config.artifacts_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RESULTS.format(split=split)).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    path = directory / DOCUMENT.format(split=split)
    path.write_text(document(config, split, rows, found), encoding="utf-8")
    return path


def run(config: DatasetConfig, split: str = "tune", resamples: int = 1000) -> Path:
    print(f"  {config.name} / {split}")
    rows = run_arms(config, split, resamples=resamples)
    path = write(config, split, rows, recall(config, split))
    ledger.render()
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.ablation",
        description="Run the ablation arms on a split and write the paired table.",
    )
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument(
        "--split",
        default="tune",
        help="the split to score the arms on; arms are inspected on tune and "
        "reported on validation, and test is scored once by ticket 10",
    )
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    if args.split == "train":
        print("\nerror: the arms are not scored on the split they fit on\n")
        return 2
    for name in args.dataset or DEFAULT_DATASETS:
        path = run(DATASETS[name], args.split, args.resamples)
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
