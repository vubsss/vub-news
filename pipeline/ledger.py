"""The trade-off ledger: every variant tried, its quality and its cost, on one row.

Every other artifact in this repository answers one question about one axis --
an evaluate JSON says how well a retriever ranks, a bench JSONL says how fast
it answers, build-timings says what a stage costs to run. The design note has
to put those side by side for every option that was tried, and assembling that
table at write-up time from a dozen files is where numbers get transcribed
wrong. So each module that tries an option records one row here, with both
metric families, when it computes the number.

The schema is fixed rather than free-form on purpose. A row that is missing
its engineering side renders as a blank cell in the markdown, and a blank cell
is a ticket that is not done. A dict with whatever keys the caller had would
hide that.

Keyed on `(dataset, stage, variant, split)`. Recording a key again replaces
the earlier row: a re-run describes the pipeline as it stands, and the renderer
must never have two rows for one cell to choose between. `delta` fields carry a
paired difference against `delta_vs` -- the variant this one is compared with
-- and its bootstrap interval, so "beats the baseline, CI excludes zero" is a
row's own claim rather than a subtraction the reader does.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import paths

RESULTS = "tradeoffs.jsonl"
DOCUMENT = "tradeoffs.md"

KEY = ("dataset", "stage", "variant", "split")

# What the harness scores. Each carries a bootstrap interval as `<name>_lo`
# and `<name>_hi`, in the same order pipeline.evaluate reports them.
FUNCTIONAL = ("auc", "mrr", "ndcg@5", "ndcg@10", "diversity", "novelty", "coverage")

# What it costs. Bytes are of the stores a request touches; seconds are wall
# time to build the variant; the rest describe the serving path.
ENGINEERING = (
    "index_bytes",
    "feature_bytes",
    "model_bytes",
    "train_seconds",
    "peak_rss_mb",
    "p50_ms",
    "p99_ms",
    "rows_per_s",
)

DELTA = ("delta_vs", "delta", "delta_lo", "delta_hi")

COLUMNS = (
    *KEY,
    *(name for metric in FUNCTIONAL for name in (metric, f"{metric}_lo", f"{metric}_hi")),
    *ENGINEERING,
    *DELTA,
    "note",
)


class LedgerError(ValueError):
    pass


def path() -> Path:
    return paths.ARTIFACTS_DIR / RESULTS


def normalise(row: dict) -> dict:
    """The row with every column present, in schema order, unknown keys refused."""
    unknown = sorted(set(row) - set(COLUMNS))
    if unknown:
        raise LedgerError(f"not ledger columns: {', '.join(unknown)}")
    missing = [name for name in KEY if row.get(name) in (None, "")]
    if missing:
        raise LedgerError(f"a ledger row needs {', '.join(missing)}")
    return {name: row.get(name) for name in COLUMNS}


def load() -> list[dict]:
    """Every row, newest-per-key wins, in file order of first appearance."""
    file = path()
    if not file.exists():
        return []
    rows: dict[tuple, dict] = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = normalise(json.loads(line))
        rows[tuple(row[name] for name in KEY)] = row
    return list(rows.values())


def record(row: dict) -> dict:
    """Append `row`, replacing any earlier row with the same key.

    Rewrites the file rather than appending, so the file on disk holds one
    row per key and a reader that does not go through `load` sees the same
    thing one that does would.
    """
    new = normalise(row)
    key = tuple(new[name] for name in KEY)
    kept = [old for old in load() if tuple(old[name] for name in KEY) != key]
    kept.append(new)
    file = path()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        "".join(json.dumps(old) + "\n" for old in kept), encoding="utf-8"
    )
    return new


def cell(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "bytes":
        if value >= 1 << 30:
            return f"{value / (1 << 30):.2f} GB"
        if value >= 1 << 20:
            return f"{value / (1 << 20):.1f} MB"
        return f"{value / 1024:.0f} KB"
    if kind == "seconds":
        return f"{value:.0f} s" if value >= 10 else f"{value:.1f} s"
    if kind == "ms":
        return f"{value:.3f}" if value < 1 else f"{value:.2f}"
    if kind == "rate":
        return f"{value:,.0f}"
    if kind == "mb":
        return f"{value:,.0f}"
    return f"{value:.4f}"


def interval(row: dict, metric: str) -> str:
    value = row.get(metric)
    if value is None:
        return "—"
    lo, hi = row.get(f"{metric}_lo"), row.get(f"{metric}_hi")
    if lo is None or hi is None:
        return f"{value:.4f}"
    return f"{value:.4f} [{lo:.4f}, {hi:.4f}]"


def delta(row: dict) -> str:
    if row.get("delta") is None:
        return "—"
    text = f"{row['delta']:+.4f}"
    if row.get("delta_lo") is not None and row.get("delta_hi") is not None:
        text += f" [{row['delta_lo']:+.4f}, {row['delta_hi']:+.4f}]"
    if row.get("delta_vs"):
        text += f" vs `{row['delta_vs']}`"
    return text


ENGINEERING_KIND = {
    "index_bytes": "bytes",
    "feature_bytes": "bytes",
    "model_bytes": "bytes",
    "train_seconds": "seconds",
    "peak_rss_mb": "mb",
    "p50_ms": "ms",
    "p99_ms": "ms",
    "rows_per_s": "rate",
}


def table(rows: list[dict]) -> list[str]:
    head = (
        "| variant | split | AUC | MRR | nDCG@5 | nDCG@10 | div | nov | cov | Δ AUC "
        "| index | features | model | train | peak RSS MB | p50 ms | p99 ms | rows/s | note |"
    )
    lines = [head, "|" + "---|" * 19]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['variant']}`",
                    row["split"],
                    interval(row, "auc"),
                    cell(row.get("mrr"), "metric"),
                    cell(row.get("ndcg@5"), "metric"),
                    cell(row.get("ndcg@10"), "metric"),
                    cell(row.get("diversity"), "metric"),
                    cell(row.get("novelty"), "metric"),
                    cell(row.get("coverage"), "metric"),
                    delta(row),
                    *(cell(row.get(name), ENGINEERING_KIND[name]) for name in ENGINEERING),
                    row.get("note") or "",
                ]
            )
            + " |"
        )
    return lines


def document(rows: list[dict]) -> str:
    lines = [
        "# Trade-offs: every variant tried, what it scores and what it costs",
        "",
        "One row per `(dataset, stage, variant, split)`, recorded by the module "
        "that computed the number and re-recorded on every re-run. A `—` is a "
        "measurement that has not been taken, not a zero. Regenerate with "
        "`python -m pipeline.ledger`.",
        "",
        "- **AUC** carries its 95% bootstrap interval; **Δ AUC** is the paired "
        "difference against the variant named, with its interval.",
        "- **index / features / model** are the bytes of the stores a request "
        "touches; **train** is wall time to build the variant; **p50 / p99** "
        "are per single-user request through the serving path.",
        "",
    ]
    if not rows:
        lines.append("_No rows recorded yet._")
        return "\n".join(lines) + "\n"
    for dataset in sorted({row["dataset"] for row in rows}):
        lines += [f"## {dataset}", ""]
        of_dataset = [row for row in rows if row["dataset"] == dataset]
        for stage in sorted({row["stage"] for row in of_dataset}):
            lines += [f"### {stage}", ""]
            lines += table([row for row in of_dataset if row["stage"] == stage])
            lines.append("")
    return "\n".join(lines) + "\n"


def render() -> Path:
    file = paths.ARTIFACTS_DIR / DOCUMENT
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(document(load()), encoding="utf-8")
    return file


# ---------------------------------------------------------------------------
# Seeding the ledger with the rows Assignment 1 measured.
#
# A one-off. From here on a row is recorded by the code that computes the
# number, in the same run; this reads the artifacts the A1 modules wrote so
# that stage one has rows for A2's variants to be deltas against. It fills what
# A1 measured and leaves blank what it did not -- the serving p50/p99 for the
# retrievers is ticket 09's, and nothing here pretends otherwise.

A1_STAGE = "retrieve"
A1_RETRIEVERS = ("bm25", "ann", "hybrid")


def _bytes_under(directory: Path) -> int | None:
    if not directory.exists():
        return None
    return sum(file.stat().st_size for file in directory.rglob("*") if file.is_file())


def _index_bytes(config, retriever: str) -> int | None:
    """The on-disk size of what the retriever loads: the bm25s index, the
    vector matrix the flat index wraps (the two are the same bytes), or both."""
    from pipeline import embed

    lexical = _bytes_under(config.artifacts_dir / "bm25")
    vectors = embed.output_dir(config) / embed.VECTORS
    semantic = vectors.stat().st_size if vectors.exists() else None
    if retriever == "bm25":
        return lexical
    if retriever == "ann":
        return semantic
    if lexical is None or semantic is None:
        return None
    return lexical + semantic


def _serve_rows(dataset: str) -> dict[str, dict]:
    file = paths.ARTIFACTS_DIR / f"bench-serve-{dataset}.jsonl"
    if not file.exists():
        return {}
    rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    from pipeline import retrieval

    return {
        row["retriever"]: row for row in rows if row.get("history_k") == retrieval.HISTORY_K
    }


def seed_a1(split: str = "validation") -> list[dict]:
    from pipeline import evaluate, timings
    from pipeline.datasets import DEFAULT_DATASETS, DATASETS

    stage_cost = {(row["dataset"], row["stage"]): row for row in timings.latest()}
    recorded = []
    for name in DEFAULT_DATASETS:
        config = DATASETS[name]
        served = _serve_rows(name)
        for retriever in A1_RETRIEVERS:
            report_path = config.artifacts_dir / evaluate.EVALUATE_DIR / f"{retriever}-{split}.json"
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            overall = {
                row["metric"]: row
                for row in report["results"]
                if row["slice"] == "overall"
            }
            row: dict = {
                "dataset": name,
                "stage": A1_STAGE,
                "variant": retriever,
                "split": split,
                "index_bytes": _index_bytes(config, retriever),
            }
            for metric in FUNCTIONAL:
                if metric in overall:
                    row[metric] = overall[metric]["value"]
                    row[f"{metric}_lo"] = overall[metric].get("lo")
                    row[f"{metric}_hi"] = overall[metric].get("hi")
            # The build stage that produces this retriever's index. The hybrid
            # fuses the other two and builds nothing of its own.
            cost = stage_cost.get((name, retriever))
            if cost is not None:
                row["train_seconds"] = cost["seconds"]
                row["peak_rss_mb"] = cost["peak_rss_mb"]
            bench = served.get(retriever)
            if bench is not None:
                row["rows_per_s"] = bench["impressions_per_second"]
                row["note"] = f"mean {bench['marginal_ms']} ms/impression (bench-serve, tune)"
            recorded.append(record(row))
    return recorded


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pipeline.ledger",
        description="Regenerate the trade-off table, optionally seeding A1's rows first.",
    )
    parser.add_argument(
        "--seed-a1",
        action="store_true",
        help="record the A1 retrievers' rows from the evaluate, bench and timings artifacts",
    )
    args = parser.parse_args(argv)
    if args.seed_a1:
        for row in seed_a1():
            print(f"  {row['dataset']}/{row['variant']}: auc {row['auc']:.4f}")
    file = render()
    print(f"-> {file.relative_to(paths.ARTIFACTS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
