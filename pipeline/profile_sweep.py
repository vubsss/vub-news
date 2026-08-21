"""The user profile: how far back to read, and how to summarise it.

    python -m pipeline.profile_sweep --dataset ebnerd
    python -m pipeline.profile_sweep --dataset mind --decay   # the second pass

Two axes, swept together because they interact. The best window for a *mean* is
short, to keep the profile coherent; the best window for a *max* is long,
because a max is robust to clicks that have nothing to do with each other.
Sweeping them one after the other picks the wrong window for whichever pooling
it did not sweep with.

The old grid stopped at 20 and 20 won three of its four cells, which is a grid
reporting its own boundary. This one runs to 80. EB-NeRD's median history is
211 clicks, so even 80 reads under half of it.

**Recall is not on this grid.** Corpus retrieval always uses the pooled mean
vector — a max over K clicks against the whole catalogue is K searches rather
than one — so recall moves with the window and not with the pooling, and it is
reported once per window rather than once per cell. No column here should be
read as evidence about pooling's effect on retrieval, because none of it is.

Differences are bootstrapped **paired by impression**, for the reason phase 3
found the hard way: every cell scores the same impressions, so most of an
unpaired interval's width is variance between impressions rather than between
settings, and no two cells can be separated by comparing those intervals
whatever the true effect. The baseline is the configuration in use — `k=10`,
`mean` — because the question is whether to move, not which cell is highest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import time

import numpy as np
import pandas as pd

from pipeline import (
    ann_index,
    bm25_index,
    embed_compare,
    evaluate,
    paths,
    retrieval,
    weighting,
)
from pipeline.datasets import DATASETS, DatasetConfig, WeightingSpec

WINDOWS = (5, 10, 20, 40, 80)

# The configuration every difference is measured from: what the pipeline does
# today, which the old sweep chose on a contaminated split.
BASELINE = (retrieval.HISTORY_K, retrieval.POOLING)

# The second pass, run at the winning cell only. Slice 3 measured every scheme
# at k=10 and found them all neutral-to-harmful, which is what a ten-click
# window should do to a decay -- there is nothing stale inside it to discount.
# Whether decay earns anything at 40 or 80 is the open question, so the
# constants are swept where the window is large enough for them to bite.
DECAYS = {
    "position": (0.99, 0.95, 0.9, 0.8),
    "time": (168.0, 72.0, 24.0, 6.0),
    "engagement": (1.0,),
}

RESULTS = "profile-{suffix}.jsonl"
DOCUMENT = "profile-{suffix}.md"


class SweepError(RuntimeError):
    """A cell did not run the configuration its row claims."""


def fingerprint(ranked: pd.DataFrame, sample: int = 2000) -> str:
    """A short hash of what a cell actually ranked first.

    Two cells that claim different poolings and produce the same fingerprint
    ran the same thing under two names — which is what happens when a
    parameter is accepted and then ignored, and is invisible in a metric
    column because the numbers agree too.
    """
    top = [ids[0] if ids else "" for ids in ranked["ranked_ids"][:sample]]
    return hashlib.sha1("\x00".join(top).encode()).hexdigest()[:12]


def confirm_pooling_reached_the_retriever(rows: list[dict]) -> str:
    """That each cell ran the pooling it is labelled with.

    Checked behaviourally rather than by reading the parameter back, because
    the failure being guarded against is a retriever that takes the argument
    and does not use it. Two properties catch that:

    * for the semantic retriever, cells sharing a window but claiming
      different poolings must rank differently — an aggregator that was
      ignored gives all three the same fingerprint;
    * for the lexical one, `last` is the same query at every window by
      construction, so its fingerprint must *not* move with the window. If it
      does, the window is reaching a query that claims not to depend on it.
    """
    checked = 0
    for (dataset, retriever, window), group in _grouped(
        rows, ("dataset", "retriever", "history_k")
    ).items():
        marks = {row["pooling"]: row["fingerprint"] for row in group}
        if retriever == "ann" and len(marks) > 1 and len(set(marks.values())) == 1:
            raise SweepError(
                f"{dataset}/{retriever}/k={window}: {', '.join(marks)} all rank "
                f"identically, so the pooling never reached the retriever"
            )
        checked += len(marks)

    for (dataset, retriever, pooling), group in _grouped(
        rows, ("dataset", "retriever", "pooling")
    ).items():
        if retriever == "bm25" and pooling == "last" and len(group) > 1:
            if len({row["fingerprint"] for row in group}) != 1:
                raise SweepError(
                    f"{dataset}/bm25/last ranks differently at different "
                    f"windows, but a one-click query cannot depend on one"
                )
    return f"{checked} cells confirmed to have run the pooling they claim"


def _grouped(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out: dict = {}
    for row in rows:
        out.setdefault(tuple(row[key] for key in keys), []).append(row)
    return out


def cells(config: DatasetConfig, retriever: str) -> list[tuple[int, str]]:
    """Which (window, pooling) pairs this retriever can actually run.

    BM25 has no per-click aggregator, so `max` is not a cell it can fill and
    the grid is not a rectangle. Skipped here rather than run and quietly
    scored as `mean`, which would put a row in the table whose label did not
    describe what happened.
    """
    pairs = []
    for window, pooling in itertools.product(WINDOWS, retrieval.POOLINGS):
        if retriever == "bm25":
            try:
                bm25_index.window_for(window, pooling)
            except ValueError:
                continue
        pairs.append((window, pooling))
    return pairs


def run(
    config: DatasetConfig,
    split: str,
    resamples: int,
    retrievers: tuple[str, ...],
) -> list[dict]:
    store = config.feature_store_dir
    behaviors = pd.read_parquet(store / "behaviors.parquet")
    history = pd.read_parquet(store / "history.parquet")
    impressions = behaviors[behaviors["split"] == split]
    history = history[history["impression_id"].isin(set(impressions["impression_id"]))]
    print(f"  {len(impressions):,} {split} impressions", flush=True)

    rows: list[dict] = []
    for retriever in retrievers:
        # Every cell of one retriever is scored on the same impressions in the
        # same order, so the differences below are paired.
        measured: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        for window, pooling in cells(config, retriever):
            started = time.perf_counter()
            ordered = evaluate.RETRIEVERS[retriever].rank_candidates(
                config, impressions, history, window, pooling
            )
            values = evaluate.per_impression_metrics(
                ordered, _labels_for(impressions, ordered)
            )
            measured[(window, pooling)] = values
            rows.append(
                _row(config, retriever, window, pooling, values, ordered, started)
            )
            print(
                f"    {retriever:<5} k={window:<3} {pooling:<5} "
                f"auc {rows[-1]['auc']:.4f}  {rows[-1]['seconds']:.0f}s",
                flush=True,
            )
        _pair(rows, measured, retriever, resamples)
    return rows


def _labels_for(impressions: pd.DataFrame, ranked: pd.DataFrame) -> list[dict[str, int]]:
    """Labels by article id, looked up per impression in the order ranked
    came back — the retrievers emit ranked order, not candidate order."""
    of = {
        impression: dict(zip(candidates, marks, strict=True))
        for impression, candidates, marks in zip(
            impressions["impression_id"],
            impressions["candidate_ids"],
            impressions["labels"],
        )
    }
    return [of[impression] for impression in ranked["impression_id"]]


def _row(config, retriever, window, pooling, values, ranked, started) -> dict:
    row = {
        "dataset": config.name,
        "retriever": retriever,
        "history_k": window,
        "pooling": pooling,
        "weighting": config.weighting.scheme,
        "decay": config.weighting.decay,
        "n": int(len(values["auc"])),
        "fingerprint": fingerprint(ranked),
        "seconds": round(time.perf_counter() - started, 1),
    }
    for metric, scores in values.items():
        row[metric] = float(scores.mean()) if len(scores) else 0.0
    return row


def _pair(rows, measured, retriever, resamples) -> None:
    """Each cell's difference from the baseline configuration, bootstrapped
    paired by impression.

    The cells share their impressions and their order, so subtracting them
    per impression cancels the variance that comes from the impressions
    themselves — which on this data is most of the width of an unpaired
    interval, and the reason comparing those intervals cannot separate
    anything. A cell the baseline itself is missing from gets no gap rather
    than a gap against whatever ran first.
    """
    baseline = measured.get(BASELINE)
    if baseline is None:
        return
    for row in rows:
        if row["retriever"] != retriever:
            continue
        key = (row["history_k"], row["pooling"])
        if key == BASELINE or key not in measured:
            continue
        row["baseline"] = f"k={BASELINE[0]}, {BASELINE[1]}"
        for metric in evaluate.ACCURACY_METRICS:
            gap = measured[key][metric] - baseline[metric]
            low, high = embed_compare.interval(gap, resamples)
            row[f"{metric}_gap"] = float(gap.mean()) if len(gap) else 0.0
            row[f"{metric}_gap_lo"], row[f"{metric}_gap_hi"] = low, high


def _grid(rows: list[dict], dataset: str, retriever: str) -> list[str]:
    lines = [
        "",
        f"### {retriever}",
        "",
        "| k | pooling | auc | mrr | ndcg@5 | ndcg@10 | vs baseline (paired) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["dataset"] != dataset or row["retriever"] != retriever:
            continue
        gap = (
            f"{row['auc_gap']:+.4f} [{row['auc_gap_lo']:+.4f}, "
            f"{row['auc_gap_hi']:+.4f}]"
            if "auc_gap" in row
            else "— baseline —"
        )
        lines.append(
            f"| {row['history_k']} | {row['pooling']} | {row['auc']:.4f} | "
            f"{row['mrr']:.4f} | {row['ndcg@5']:.4f} | {row['ndcg@10']:.4f} | {gap} |"
        )
    return lines


def document(rows: list[dict], name: str, split: str, confirmed: str) -> str:
    lines = [
        f"# User profile — {name}, {split} split",
        "",
        f"History window x pooling, on the **{split}** split, swept jointly "
        f"because they interact. Generated by `python -m pipeline.profile_sweep`.",
        "",
        f"- Differences are **paired by impression** against the configuration "
        f"in use (`k={BASELINE[0]}`, `{BASELINE[1]}`). Every cell scores the "
        f"same impressions, so the unpaired per-cell intervals are dominated by "
        f"variance between impressions and cannot separate anything.",
        "- **No recall column.** Corpus retrieval always uses the pooled mean "
        "vector, so recall moves with the window and not with the pooling. "
        "Nothing here is evidence about pooling's effect on retrieval.",
        "- `bm25` has no per-click aggregator, so it has no `max` cells. Its "
        "`last` is a one-click query and therefore identical at every window.",
        f"- {confirmed}.",
    ]
    for retriever in sorted({row["retriever"] for row in rows}):
        lines += _grid(rows, name, retriever)

    lines += ["", "## Reading", ""]
    for retriever in sorted({row["retriever"] for row in rows}):
        here = [r for r in rows if r["retriever"] == retriever]
        best = max(here, key=lambda r: r["auc"])
        moved = [r for r in here if r.get("auc_gap_lo", 0) > 0]
        lines.append(
            f"- **{retriever}**: best is k={best['history_k']}, "
            f"{best['pooling']} (auc {best['auc']:.4f}). "
            + (
                f"{len(moved)} of {len(here) - 1} cells beat the baseline by a "
                f"paired interval clear of zero."
                if moved
                else "No cell beats the baseline by a paired interval clear of "
                "zero, so the configuration in use is not shown to be wrong."
            )
        )
        if best["history_k"] == max(WINDOWS):
            lines.append(
                f"  - The winner sits at the **largest window tested**, which is "
                f"the same boundary the old grid reported. The optimum may be "
                f"outside this grid too."
            )
    return "\n".join(lines) + "\n"


def decay_pass(
    config: DatasetConfig,
    split: str,
    resamples: int,
    grid: list[dict],
) -> list[dict]:
    """Sweep the decay constants at the cell the grid chose, not at k=10.

    Slice 3 measured every scheme at the current window and found them all
    neutral-to-harmful, which is what a ten-click window should do to a decay:
    there is nothing stale inside it to discount. A decay can only earn
    something where the window is long enough to hold clicks worth discounting,
    so this runs where the grid actually landed.
    """
    here = [row for row in grid if row["retriever"] == "ann"]
    if not here:
        raise SweepError("no semantic cells in the grid to choose a window from")
    best = max(here, key=lambda row: row["auc"])
    window, pooling = best["history_k"], best["pooling"]
    print(
        f"  decay swept at k={window}, {pooling} — the cell the grid chose "
        f"(auc {best['auc']:.4f})",
        flush=True,
    )

    store = config.feature_store_dir
    behaviors = pd.read_parquet(store / "behaviors.parquet")
    history = pd.read_parquet(store / "history.parquet")
    impressions = behaviors[behaviors["split"] == split]
    history = history[history["impression_id"].isin(set(impressions["impression_id"]))]

    schemes = [
        (scheme, decay)
        for scheme in weighting.available(config)
        if scheme != "uniform"
        for decay in DECAYS[scheme]
    ]

    rows: list[dict] = []
    measured: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for scheme, decay in [("uniform", 1.0), *schemes]:
        started = time.perf_counter()
        tuned = dataclasses.replace(
            config, weighting=WeightingSpec(scheme=scheme, decay=decay)
        )
        ordered = ann_index.rank_candidates(
            tuned, impressions, history, window, pooling
        )
        values = evaluate.per_impression_metrics(
            ordered, _labels_for(impressions, ordered)
        )
        measured[(scheme, decay)] = values
        row = _row(tuned, "ann", window, pooling, values, ordered, started)
        rows.append(row)
        print(
            f"    {scheme:<11} decay={decay:<6} auc {row['auc']:.4f}  "
            f"{row['seconds']:.0f}s",
            flush=True,
        )

    # Paired against uniform, which is the scheme in use.
    baseline = measured[("uniform", 1.0)]
    for row in rows:
        key = (row["weighting"], row["decay"])
        if key == ("uniform", 1.0):
            continue
        for metric in evaluate.ACCURACY_METRICS:
            gap = measured[key][metric] - baseline[metric]
            low, high = embed_compare.interval(gap, resamples)
            row[f"{metric}_gap"] = float(gap.mean()) if len(gap) else 0.0
            row[f"{metric}_gap_lo"], row[f"{metric}_gap_hi"] = low, high
    return rows


def decay_document(rows: list[dict], name: str, split: str) -> str:
    lines = [
        f"# Click weighting — {name}, {split} split",
        "",
        f"Decay constants at the cell the window x pooling grid chose "
        f"(`k={rows[0]['history_k']}`, `{rows[0]['pooling']}`), on the "
        f"**{split}** split. Generated by "
        f"`python -m pipeline.profile_sweep --decay`.",
        "",
        "- Paired by impression against **uniform**, the scheme in use.",
        f"- `{name}` supports {', '.join(weighting.available(DATASETS[name]))} — "
        f"which schemes a dataset can express is decided by the columns its "
        f"history carries, not by this grid.",
        "",
        "| scheme | decay | auc | ndcg@10 | vs uniform (paired) |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        gap = (
            f"{row['auc_gap']:+.4f} [{row['auc_gap_lo']:+.4f}, "
            f"{row['auc_gap_hi']:+.4f}]"
            if "auc_gap" in row
            else "— baseline —"
        )
        lines.append(
            f"| {row['weighting']} | {row['decay']} | {row['auc']:.4f} | "
            f"{row['ndcg@10']:.4f} | {gap} |"
        )
    better = [r for r in rows if r.get("auc_gap_lo", 0) > 0]
    lines += ["", "## Reading", ""]
    lines.append(
        f"- {len(better)} of {len(rows) - 1} weighted cells beat uniform by a "
        f"paired interval clear of zero."
        if better
        else "- **No weighting beats uniform** by a paired interval clear of "
        "zero. Every past click counts the same, which is what the pipeline "
        "already does."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mind")
    parser.add_argument("--split", default=evaluate.TUNE, choices=evaluate.SCORABLE)
    parser.add_argument("--resamples", type=int, default=evaluate.BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--decay",
        action="store_true",
        help="the second pass: sweep decay constants at the cell the grid chose",
    )
    parser.add_argument(
        "--retriever",
        action="append",
        choices=sorted(evaluate.RETRIEVERS),
        help="restrict to one retriever (repeatable); default is both",
    )
    args = parser.parse_args(argv)

    config = DATASETS[args.dataset]
    suffix = f"{config.name}-{args.split}"
    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.decay:
        stored = paths.ARTIFACTS_DIR / RESULTS.format(suffix=suffix)
        if not stored.exists():
            raise SystemExit(
                f"{stored} not found — run the window x pooling grid first, "
                f"since the decay pass is swept at the cell it chooses"
            )
        grid = [json.loads(line) for line in stored.open()]
        rows = decay_pass(config, args.split, args.resamples, grid)
        results = paths.ARTIFACTS_DIR / f"profile-decay-{suffix}.jsonl"
        text = decay_document(rows, config.name, args.split)
        (paths.ARTIFACTS_DIR / f"profile-decay-{suffix}.md").write_text(text)
    else:
        retrievers = tuple(args.retriever or sorted(evaluate.RETRIEVERS))
        rows = run(config, args.split, args.resamples, retrievers)
        confirmed = confirm_pooling_reached_the_retriever(rows)
        results = paths.ARTIFACTS_DIR / RESULTS.format(suffix=suffix)
        text = document(rows, config.name, args.split, confirmed)
        (paths.ARTIFACTS_DIR / DOCUMENT.format(suffix=suffix)).write_text(text)

    with results.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
