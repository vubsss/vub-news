"""The ablation grid: every dataset x retriever x history window, in one file.

    python -m pipeline.sweep                  the full grid on validation
    python -m pipeline.sweep --quick          a reduced grid, for development
    python -m pipeline.sweep --dataset mind   restrict any axis (repeatable)
    python -m pipeline.sweep --restart        ignore what is already done

The grid the spec asks for is dataset x retriever x history window x retrieval
depth. It runs here as twelve cells rather than thirty-six, because depth is
not an axis the two measurements share:

  * Retrieval depth belongs to corpus retrieval only. `retrieve_corpus` ranks
    the whole catalogue and recall@K asks how far down you had to look; the
    three depths are prefixes of one depth-200 retrieval, so a cell measures
    all three from one search rather than searching three times.
  * The ranking metrics come from `rank_candidates`, which reorders the
    candidate list an impression already carries and truncates nothing. No
    depth can move them. Running the depth axis over them would write the same
    AUC into the file three times and invite a reader to average it.

So depth is recorded next to every recall number and is absent from the ranking
metrics, which is where it actually is. The history window, by contrast, moves
both: it is what the query is built from.

Each cell is one line of `artifacts/sweep-<split>.jsonl`, holding its own
configuration, its recall at each depth, and the evaluation report the harness
produced — the same shape `pipeline.evaluate` writes, so `pipeline.compare
--window K` reads a cell straight out of this file with nothing transcribed by
hand. One line per cell is also what makes the sweep resumable: a line is
written whole or not at all, so an interrupted run has no half-finished cell to
detect, and re-running picks up the cells that have no line yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline import compare, evaluate, ingest, paths, retrieval
from pipeline.datasets import DATASETS, DatasetConfig

# The history windows the spec sweeps. Both retrievers get the same one in a
# given cell — that is the point of the axis, and `confirm_same_window` below
# refuses to report a grid where it did not hold.
WINDOWS = (5, 10, 20)

# The reduced grid. The extremes only: if 5 and 20 cannot be told apart, 10
# will not separate them either, so this is the cheapest grid that can still
# say whether the window matters at all.
QUICK_WINDOWS = (5, 20)
QUICK_RESAMPLES = 100

# The metric the best window is chosen on. It is the one the assignment leads
# with, it carries a bootstrap interval, and it reads the whole ranking rather
# than a prefix of it.
PRIMARY = "auc"
OVERALL = "overall"

RESULTS = "sweep-{split}.jsonl"
DOCUMENT = "sweep-{split}.md"


class SweepError(RuntimeError):
    """The grid cannot be run, or cannot be reported, as asked."""


def results_path(split: str) -> Path:
    return paths.ARTIFACTS_DIR / RESULTS.format(split=split)


def cell_of(row: dict) -> tuple[str, str, int]:
    """What identifies a cell, and what resuming matches on."""
    return row["dataset"], row["retriever"], row["history_k"]


def load(split: str) -> list[dict]:
    """The cells already on disk, oldest first."""
    path = results_path(split)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append(row: dict, split: str) -> None:
    """One cell, written whole. The unit the sweep resumes at."""
    path = results_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_cell(
    config: DatasetConfig,
    retriever: str,
    history_k: int,
    split: str,
    resamples: int,
) -> dict:
    """One grid cell: recall at every depth, and the full evaluation report.

    Corpus retrieval runs once at the deepest depth and the shallower figures
    are read off the same ranking, which is what `retrieval.DEPTHS` documents.
    """
    started = time.perf_counter()

    store = config.feature_store_dir
    behaviors = pd.read_parquet(store / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == split]
    history = ingest.history_for(config, impressions)

    ranked, _ = evaluate.RETRIEVERS[retriever].retrieve_corpus(
        config, impressions, history, history_k, max(retrieval.DEPTHS)
    )
    recall = retrieval.recall_at_k(ranked, impressions, retrieval.DEPTHS)
    report = evaluate.evaluate(config, retriever, split, resamples, history_k)

    return {
        "dataset": config.name,
        "retriever": retriever,
        "history_k": history_k,
        "split": split,
        "seconds": round(time.perf_counter() - started, 1),
        "recall": [
            {
                "depth": depth,
                "value": recall[f"recall@{depth}"],
                "scored": recall["scored"],
                "no_positive": recall["no_positive"],
            }
            for depth in retrieval.DEPTHS
        ],
        "report": report,
    }


def grid(
    configs: list[DatasetConfig],
    retrievers: list[str],
    windows: tuple[int, ...],
    split: str,
    resamples: int,
    done: set[tuple[str, str, int]],
) -> list[tuple[DatasetConfig, str, int]]:
    """The cells still to run, in a fixed order."""
    return [
        (config, retriever, window)
        for config in configs
        for retriever in retrievers
        for window in windows
        if (config.name, retriever, window) not in done
    ]


def sweep(
    configs: list[DatasetConfig],
    retrievers: list[str],
    windows: tuple[int, ...],
    split: str,
    resamples: int,
    restart: bool = False,
) -> list[dict]:
    """Run the outstanding cells, appending each as it finishes.

    Every cell is written before the next one starts, so an interrupted sweep
    keeps everything it had already measured and the next run costs only what
    is left. The reported runtime distinguishes the two: a resumed run that
    reports four hours of grid time and two minutes of its own is telling the
    truth about both.
    """
    if restart:
        results_path(split).unlink(missing_ok=True)

    stored = load(split)
    done = {cell_of(row) for row in stored}
    outstanding = grid(configs, retrievers, windows, split, resamples, done)

    wanted = {
        (config.name, retriever, window)
        for config in configs
        for retriever in retrievers
        for window in windows
    }
    if not outstanding:
        print(f"    all {len(wanted)} cells already in {results_path(split).name}")
    else:
        skipped = len(wanted) - len(outstanding)
        if skipped:
            print(f"    resuming: {skipped} of {len(wanted)} cells already done")

    started = time.perf_counter()
    for i, (config, retriever, window) in enumerate(outstanding, start=1):
        print(
            f"    [{i}/{len(outstanding)}] {config.name}/{retriever}/k={window} ...",
            end=" ",
            flush=True,
        )
        row = run_cell(config, retriever, window, split, resamples)
        append(row, split)
        stored.append(row)
        print(f"{row['seconds']:.1f} s")

    this_run = time.perf_counter() - started
    rows = [row for row in stored if cell_of(row) in wanted]
    total = sum(row["seconds"] for row in rows)
    print(
        f"    {len(rows)} cells, {total / 60:.1f} min of grid time"
        + (f", {this_run / 60:.1f} min of it in this run" if outstanding else "")
    )
    return rows


# --- reading the grid -------------------------------------------------------


def confirm_same_window(rows: list[dict]) -> str:
    """That both retrievers really ran on the window their cell claims.

    The window is the axis, so a cell whose two retrievers disagree about it
    would compare a five-click query against a twenty-click one and call the
    difference lexical-against-semantic. The retrievers take it through one
    parameter of one call, so this cannot drift silently — but a comparison
    that depends on it should say it checked rather than assume.
    """
    for row in rows:
        recorded = row["report"]["history_k"]
        if recorded != row["history_k"]:
            raise SweepError(
                f"{row['dataset']}/{row['retriever']}: the cell asked for a "
                f"window of {row['history_k']} and the retriever reports "
                f"{recorded}"
            )

    windows: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        windows.setdefault((row["dataset"], row["history_k"]), set()).add(
            row["retriever"]
        )
    both = sorted({window for (_, window), seen in windows.items() if len(seen) > 1})
    return (
        f"Both retrievers ran on the same history window in every cell, "
        f"confirmed against the window each one reports back: "
        f"{', '.join(str(window) for window in both)}."
        if both
        else "Only one retriever was swept, so no window is shared to confirm."
    )


def overall(row: dict, metric: str) -> dict | None:
    """One metric on the overall slice of a cell's report."""
    for result in row["report"]["results"]:
        if result["slice"] == OVERALL and result["metric"] == metric:
            return result
    return None


def best_window(rows: list[dict], dataset: str, retriever: str) -> str | None:
    """Which window won for one retriever, and what that is worth.

    A window is only better than another where the two intervals are disjoint,
    the same test `compare` applies between retrievers and for the same reason:
    a grid this size will always have a highest number, and the interesting
    question is whether it is a finding or the noise floor.
    """
    cells = sorted(
        (row for row in rows
         if row["dataset"] == dataset and row["retriever"] == retriever),
        key=lambda row: row["history_k"],
    )
    scored = [(row, overall(row, PRIMARY)) for row in cells]
    scored = [(row, cell) for row, cell in scored if cell and cell["value"] is not None]
    if not scored:
        return None

    top, best = max(scored, key=lambda pair: pair[1]["value"])
    separated = [
        row["history_k"]
        for row, cell in scored
        if row["history_k"] != top["history_k"] and compare.disjoint(cell, best)
    ]
    tied = [
        row["history_k"]
        for row, cell in scored
        if row["history_k"] != top["history_k"] and not compare.disjoint(cell, best)
    ]

    interval = (
        f"[{best['lo']:.4f}, {best['hi']:.4f}]"
        if best["lo"] is not None
        else "no interval"
    )
    verdict = f"**k={top['history_k']}** scores highest ({PRIMARY} {best['value']:.4f} {interval})"
    if len(scored) == 1:
        return (
            f"- {dataset}/{retriever}: k={top['history_k']} is the only window "
            f"swept ({PRIMARY} {best['value']:.4f} {interval}), so nothing here "
            f"says whether the window matters."
        )
    if not tied:
        return f"- {dataset}/{retriever}: {verdict}, above every other window by a disjoint interval."
    if not separated:
        return (
            f"- {dataset}/{retriever}: {verdict}, but its interval overlaps "
            f"{', '.join(f'k={k}' for k in tied)} — the window is **not established** "
            f"as mattering here, and k={min(row['history_k'] for row, _ in scored)} "
            f"is the cheaper choice at equal evidence."
        )
    return (
        f"- {dataset}/{retriever}: {verdict}, separated from "
        f"{', '.join(f'k={k}' for k in separated)} but overlapping "
        f"{', '.join(f'k={k}' for k in tied)}, which it is not established above."
    )


def recall_table(rows: list[dict]) -> str:
    """Recall at each depth for each cell — the only place depth appears."""
    header = (
        "| dataset | retriever | window | "
        + " | ".join(f"recall@{depth}" for depth in retrieval.DEPTHS)
        + " | scored |"
    )
    rule = "| --- " * (4 + len(retrieval.DEPTHS)) + "|"
    lines = [header, rule]
    for row in sorted(rows, key=cell_of):
        by_depth = {entry["depth"]: entry for entry in row["recall"]}
        cells = " | ".join(
            f"{by_depth[depth]['value']:.4f}" if depth in by_depth else "-"
            for depth in retrieval.DEPTHS
        )
        scored = row["recall"][0]["scored"] if row["recall"] else 0
        lines.append(
            f"| {row['dataset']} | {row['retriever']} | {row['history_k']} | "
            f"{cells} | {scored} |"
        )
    return "\n".join(lines)


def window_table(rows: list[dict]) -> str:
    """Every ranking metric on the overall slice, by window, with intervals."""
    lines = [
        "| dataset | retriever | window | metric | value | 95% interval |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=cell_of):
        for metric in evaluate.ACCURACY_METRICS:
            cell = overall(row, metric)
            if cell is None or cell["value"] is None:
                continue
            interval = (
                f"[{cell['lo']:.4f}, {cell['hi']:.4f}]"
                if cell["lo"] is not None
                else "-"
            )
            lines.append(
                f"| {row['dataset']} | {row['retriever']} | {row['history_k']} | "
                f"{metric} | {cell['value']:.4f} | {interval} |"
            )
    return "\n".join(lines)


def document(rows: list[dict], split: str) -> str:
    """The grid as it goes into the design note."""
    datasets = sorted({row["dataset"] for row in rows})
    retrievers = sorted({row["retriever"] for row in rows})
    total = sum(row["seconds"] for row in rows)
    resamples = sorted({row["report"]["resamples"] for row in rows})

    readings = [
        line
        for dataset in datasets
        for retriever in retrievers
        if (line := best_window(rows, dataset, retriever))
    ]

    return "\n\n".join(
        [
            f"# Ablation sweep — {split} split",
            f"{len(rows)} cells over {len(datasets)} dataset(s), "
            f"{len(retrievers)} retriever(s) and history windows "
            f"{', '.join(str(w) for w in sorted({r['history_k'] for r in rows}))}, "
            f"in {total / 60:.1f} minutes of compute. Bootstrap resamples behind "
            f"every interval: {', '.join(str(r) for r in resamples)}.",
            "## How to read this",
            "- **The history window** is what the query is built from, and it "
            "moves both measurements below.\n"
            "- **Retrieval depth** appears only in the recall table. It is a "
            "property of searching the catalogue; the ranking metrics reorder "
            "an impression's own candidate list and truncate nothing, so no "
            "depth can move them.\n"
            "- A window is called better than another **only where their "
            "intervals are disjoint**. Overlapping intervals establish nothing "
            "in either direction, which is not the same as establishing that "
            "the two are equal.\n"
            "- Recall figures are point estimates and carry no interval: they "
            "come from the corpus-retrieval path, which the harness's bootstrap "
            "does not run over. The best window is therefore chosen on "
            f"{PRIMARY}, which does carry one.",
            "## Best history window",
            "\n".join(readings) if readings else "No cell carried a usable "
            f"{PRIMARY}.",
            confirm_same_window(rows),
            "## Ranking metrics by window (overall slice)",
            window_table(rows),
            "## Recall by retrieval depth",
            recall_table(rows),
        ]
    ) + "\n"


def write(text: str, split: str) -> Path:
    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = paths.ARTIFACTS_DIR / DOCUMENT.format(split=split)
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.sweep",
        description="Run the ablation grid and write one results file.",
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
        choices=sorted(evaluate.STAGE_ONE),
        help="restrict to one retriever (repeatable); default is all of them",
    )
    parser.add_argument(
        "--window",
        action="append",
        type=int,
        choices=WINDOWS,
        help=f"restrict to one history window (repeatable); default is "
        f"{', '.join(str(window) for window in WINDOWS)}",
    )
    parser.add_argument(
        "--split",
        default=evaluate.TUNE,
        help=f"which split to sweep: {' or '.join(evaluate.SCORABLE)} "
        f"(default: {evaluate.TUNE})",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=evaluate.BOOTSTRAP_RESAMPLES,
        help=f"bootstrap resamples behind every interval "
        f"(default: {evaluate.BOOTSTRAP_RESAMPLES})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"a reduced grid for development: windows "
        f"{' and '.join(str(window) for window in QUICK_WINDOWS)} only, and "
        f"{QUICK_RESAMPLES} resamples. The extremes are the cheapest grid that "
        f"can still say whether the window matters; the numbers it produces "
        f"are for looking at, not for quoting",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard the stored results and run every cell again",
    )
    args = parser.parse_args(argv)

    configs = [DATASETS[name] for name in (args.dataset or sorted(DATASETS))]
    retrievers = args.retriever or sorted(evaluate.STAGE_ONE)
    windows = tuple(args.window) if args.window else WINDOWS
    resamples = args.resamples
    if args.quick:
        windows = tuple(w for w in windows if w in QUICK_WINDOWS)
        if args.resamples == evaluate.BOOTSTRAP_RESAMPLES:
            resamples = QUICK_RESAMPLES

    try:
        rows = sweep(
            configs, retrievers, windows, args.split, resamples, args.restart
        )
        if not rows:
            raise SweepError("the grid is empty; nothing to report")
        text = document(rows, args.split)
    except (SweepError, evaluate.EvaluationError, compare.ComparisonError) as error:
        print(f"\nerror: {error}\n", file=sys.stderr)
        return 2

    path = write(text, args.split)
    print("\n" + text)
    print(f"  -> {results_path(args.split).relative_to(paths.ARTIFACTS_DIR.parent)}")
    print(f"  -> {path.relative_to(paths.ARTIFACTS_DIR.parent)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
