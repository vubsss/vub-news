"""`test`, scored once — and the record that says when, and that it was once.

Every other module in this project is meant to be re-run. This one is not, and
the difference is the whole of its design.

`tune` chose every option. `validation` reported every comparison. `test` has
been held back since the split stage and exists to confirm the choices, not to
inform them — which is only true if nothing is ever selected on it. A number
looked at twice is a number that can be selected on, even without intending to:
the second run happens because the first was disappointing, and that is
selection whatever it is called.

So this module keeps a durable record of when `test` was scored, with what
invocation and at what commit, and refuses to score it again. `--force` does
not delete that record; it *appends* to it, so a repository where `test` was
scored twice says so in a file rather than in somebody's memory.

It also copies the engineering columns onto the `test` rows rather than
re-measuring them. A second latency measurement is a second chance to select —
"we re-ran it and it came out faster" is exactly the pattern this is here to
prevent — and the rows say `measured on validation` so a reader knows which run
the milliseconds came from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from pipeline import ablation, evaluate, features, ledger, paths, three_way, timings
from pipeline.datasets import DATASETS, DEFAULT_DATASETS, DatasetConfig

# Where the fact lives. Under artifacts/ rather than .checkpoints/ because a
# checkpoint is something a rebuild is allowed to clear and this is not: the
# rebuild's `--force` should not be able to quietly make `test` fresh again.
STATE = "test-scored.json"

# What CodaBench returned, filled in by whoever uploaded. Not derivable from
# anything in this repository -- the leaderboard holds the labels -- so it is
# read from a file that a person writes and reported as outstanding when the
# file has no entry. The alternative is a three-way table with one column
# quietly missing, which reads as agreement.
LEADERBOARD = "leaderboard.json"

# The split that is scored once, and the one every reported comparison came
# from. Named rather than passed: this module exists for one split.
SPLIT = "test"
REPORTED_ON = "validation"

# The frames `ablation.run_arms` reads: it fits on the later half of `train`,
# stops on `tune` and scores on `test`, and Q9's leaky arms read their own
# file of each. Checked before anything is scored, because the alternative
# discovered on this project is a run that scores `test`, gets most of the way
# through nineteen arms, fails on the leaky one -- and is then blocked from
# retrying by the refusal above, having already spent the split.
FRAMES = ("train", "tune", SPLIT)

# Columns that are copied from the `validation` row rather than measured again.
# Bytes and milliseconds are properties of the model and the machine, not of
# the split it was scored on, so re-measuring them on `test` would produce a
# second number for one fact and invite a choice between them.
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

# What the final table reports, in the order a reader wants them.
METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


class AlreadyScored(RuntimeError):
    """`test` has been scored for this dataset, and that is meant to be once."""


# ---------------------------------------------------------------------------
# The record.


def state_path() -> Path:
    return paths.ARTIFACTS_DIR / STATE


def load_state() -> list[dict]:
    """Every scoring of `test` there has ever been, oldest first."""
    path = state_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def scored_already(dataset: str) -> list[dict]:
    return [entry for entry in load_state() if entry["dataset"] == dataset]


def commit() -> str | None:
    """The commit `test` was scored at, so the numbers name their code.

    None outside a git checkout rather than a raised error: the record is worth
    writing even where the commit cannot be read, and a scoring that failed
    because `git` was missing would be the worst possible reason to run this
    module a second time.
    """
    try:
        found = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return found.stdout.strip() or None if found.returncode == 0 else None


def record_state(dataset: str, invocation: str, arms: int, seconds: float) -> dict:
    """Append this scoring to the record. Never replaces an earlier one."""
    entry = {
        "dataset": dataset,
        "split": SPLIT,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "invocation": invocation,
        "commit": commit(),
        "arms": arms,
        "seconds": round(seconds, 1),
        # Which scoring this is. `1` is the one the note is allowed to quote;
        # anything higher is a fact the note has to state.
        "run": len(scored_already(dataset)) + 1,
    }
    history = load_state()
    history.append(entry)
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return entry


# ---------------------------------------------------------------------------
# The engineering columns, copied rather than re-measured.


def copy_engineering(config: DatasetConfig) -> list[dict]:
    """Put `validation`'s bytes and milliseconds on the matching `test` rows.

    Matched on `(stage, variant)`, which is what identifies a *model*; the
    split is what identifies a scoring of it. A test row with no validation
    twin is left alone and reported: it means an arm was scored on `test` that
    was never reported on `validation`, which is a hole in the note rather than
    a number to invent.
    """
    rows = ledger.load()
    by_model = {
        (row["stage"], row["variant"]): row
        for row in rows
        if row["dataset"] == config.name and row["split"] == REPORTED_ON
    }

    copied = []
    for row in rows:
        if row["dataset"] != config.name or row["split"] != SPLIT:
            continue
        twin = by_model.get((row["stage"], row["variant"]))
        if twin is None:
            continue
        updated = dict(row)
        for column in ENGINEERING:
            updated[column] = twin[column]
        note = (updated.get("note") or "").rstrip("; ")
        updated["note"] = (
            f"{note}; " if note else ""
        ) + f"engineering columns measured on {REPORTED_ON}"
        copied.append(ledger.record(updated))
    return copied


def unmatched(config: DatasetConfig) -> list[str]:
    """Arms scored on `test` that were never reported on `validation`.

    A hole worth naming: the note's story is "chosen on tune, reported on
    validation, confirmed on test", and a row that skipped the middle step did
    not follow it.
    """
    rows = ledger.load()
    reported = {
        (row["stage"], row["variant"])
        for row in rows
        if row["dataset"] == config.name and row["split"] == REPORTED_ON
    }
    return sorted(
        row["variant"]
        for row in rows
        if row["dataset"] == config.name
        and row["split"] == SPLIT
        and (row["stage"], row["variant"]) not in reported
    )


# ---------------------------------------------------------------------------
# The three-way check.


def leaderboard(config: DatasetConfig, retriever: str) -> float | None:
    """What CodaBench returned for this dataset's submission, if anyone said.

    Read from a file a person writes after uploading, because nothing in this
    repository can compute it: the leaderboard is holding the labels, which is
    the point of a leaderboard. Absent, this returns None and the table says
    "outstanding" -- a missing column that renders as a blank cell reads as
    agreement, which is the one thing it must not say.
    """
    path = paths.ARTIFACTS_DIR / LEADERBOARD
    if not path.exists():
        return None
    scores = json.loads(path.read_text(encoding="utf-8"))
    entry = scores.get(config.name, {}).get(retriever)
    if entry is None:
        return None
    return float(entry["auc"] if isinstance(entry, dict) else entry)


def shifts(config: DatasetConfig, retriever: str = "rerank") -> dict:
    """`validation` -> `test` -> leaderboard, with the sign of each step.

    Three numbers for one model on three populations, and what the note has to
    do with them is say which way each moved and why that is or is not
    expected. A1's version of this found a real thing -- the leaderboard week
    is a different week -- so the shift is reported rather than explained away.
    """
    found = {}
    for split in (REPORTED_ON, SPLIT):
        try:
            found[split] = three_way.load(config, retriever, split)["auc"]["value"]
        except (three_way.MissingReport, KeyError):
            found[split] = None
    found["leaderboard"] = leaderboard(config, retriever)

    steps = []
    order = (REPORTED_ON, SPLIT, "leaderboard")
    for earlier, later in zip(order, order[1:]):
        before, after = found[earlier], found[later]
        steps.append(
            {
                "from": earlier,
                "to": later,
                "delta": None if before is None or after is None else after - before,
                "sign": (
                    None
                    if before is None or after is None
                    else ("+" if after > before else "-" if after < before else "0")
                ),
            }
        )
    return {"retriever": retriever, "auc": found, "steps": steps}


# ---------------------------------------------------------------------------
# The run.


def build_frame(config: DatasetConfig) -> None:
    """Materialise the `test` feature frame, which the rebuild does not.

    `features.SPLITS` is the three splits `python build.py` builds, and `test`
    is deliberately not among them: a frame the rebuild materialises is a frame
    that exists on disk from the first day, and the ablation reads its arms out
    of exactly that file. Building it here keeps every read of the test period
    inside the one module that is allowed to make them.

    Both causalities, because the ablation's arms include Q9's leaky pair and
    they read their own file. A run that built only the causal one would get
    most of the way through the arms and then fail on the leaky arm, having
    already scored `test` -- which is the worst place for this to stop, since
    the refusal below would then block the retry.

    Building a frame is not scoring one -- no metric comes out of it -- so this
    is idempotent and a re-run costs the build and reveals nothing. Each is
    skipped when its file is already there.
    """
    for split in FRAMES:
        for causal in (True, False):
            path = features.path_for(
                config, split, causal, config.features.precision
            )
            if path.exists():
                continue
            report = features.build(config, split, config.features, causal=causal)
            features.record(report, config.features)
            print(
                f"    built {split}{'' if causal else ' (leaky)'}: "
                f"{report['rows']:,} rows, {report['seconds']:.1f} s"
            )


def evaluate_all(config: DatasetConfig, resamples: int) -> list[dict]:
    """Every retriever on `test`, through the harness that scored the others.

    `evaluate.evaluate` and `evaluate.save`, which is what the validation run
    calls -- not `evaluate.run`, which is the build stage and is hard-wired to
    validation on purpose: a split rebuilt into every `python build.py` is one
    that gets looked at repeatedly, which is how a held-back split stops being
    one.
    """
    reports = [
        evaluate.evaluate(config, retriever, SPLIT, resamples)
        for retriever in evaluate.RETRIEVERS
    ]
    evaluate.report_on([evaluate.save(report, config) for report in reports], reports)
    return reports


def run(
    config: DatasetConfig,
    resamples: int = 1000,
    force: bool = False,
    invocation: str = "python -m pipeline.final",
) -> dict:
    """Score `test` for this dataset, once.

    The order matters: the retrievers first, then the arms, then the copy of
    the engineering columns -- which has to come after both, because it reads
    the rows they wrote.
    """
    earlier = scored_already(config.name)
    if earlier and not force:
        raise AlreadyScored(
            f"{config.name}: `{SPLIT}` was scored on {earlier[-1]['scored_at']} "
            f"at commit {earlier[-1]['commit']} by `{earlier[-1]['invocation']}`.\n"
            f"  It is held out, and a number looked at twice is a number that "
            f"can be selected on. If it genuinely has to be re-scored, "
            f"`--force` appends to {state_path()} rather than replacing it, so "
            f"the design note has to say it happened."
        )

    print(f"  {config.name} / {SPLIT}" + ("  (RE-SCORED)" if earlier else ""))
    with timings.sample() as cost:
        build_frame(config)
        evaluate_all(config, resamples)
        arms = ablation.run_arms(config, SPLIT, resamples=resamples)
        ablation.write(config, SPLIT, arms, ablation.recall(config, SPLIT))

    copied = copy_engineering(config)
    entry = record_state(config.name, invocation, len(arms), cost["seconds"])
    ledger.record(
        {
            "dataset": config.name,
            "stage": "final",
            "variant": "whole-run",
            "split": SPLIT,
            "train_seconds": round(cost["seconds"], 2),
            "peak_rss_mb": round(cost["peak_rss_mb"], 1),
            "note": (
                f"{len(arms)} arms scored once on {SPLIT}, run {entry['run']}, "
                f"commit {entry['commit']}; {len(copied)} rows took their "
                f"engineering columns from {REPORTED_ON}"
            ),
        }
    )
    (paths.ARTIFACTS_DIR / f"test-once-{config.name}.md").write_text(
        document(config, entry), encoding="utf-8"
    )
    ledger.render()
    return entry


def document(config: DatasetConfig, entry: dict) -> str:
    """The page the design note's headline numbers are read off."""
    rows = [
        row
        for row in ledger.load()
        if row["dataset"] == config.name and row["split"] == SPLIT
    ]
    missing = unmatched(config)
    lines = [
        f"# {config.name}: `{SPLIT}`, scored once",
        "",
        f"Scored on **{entry['scored_at']}** at commit `{entry['commit']}` by "
        f"`{entry['invocation']}`. This is run **{entry['run']}**"
        + (
            "."
            if entry["run"] == 1
            else " — `test` has been scored more than once for this dataset, "
            "and every reading of these numbers has to say so."
        ),
        "",
        "Nothing was chosen here. Every option was tried on `tune` and every "
        f"comparison read on `{REPORTED_ON}`; this confirms the chosen ones. "
        "The bytes and milliseconds below were **not re-measured** — they are "
        f"the `{REPORTED_ON}` run's, because a second measurement is a second "
        "chance to select.",
        "",
        "## Every arm on `test`",
        "",
        "| stage | variant | auc | 95% interval | vs | delta | model bytes | p50 ms |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["stage"], r["variant"])):
        lines.append(
            f"| {row['stage']} | {row['variant']} | "
            f"{ledger.cell(row['auc'], 'metric')} | {ledger.interval(row, 'auc')} | "
            f"{row['delta_vs'] or '—'} | {ledger.delta(row)} | "
            f"{ledger.cell(row['model_bytes'], 'bytes')} | "
            f"{ledger.cell(row['p50_ms'], 'ms')} |"
        )

    lines += ["", "## The three-way check", ""]
    for retriever in ("nrms", "rerank"):
        moved = shifts(config, retriever)
        lines += [
            f"### {retriever}",
            "",
            "| population | auc |",
            "|---|---:|",
        ]
        for name in (REPORTED_ON, SPLIT, "leaderboard"):
            value = moved["auc"][name]
            lines.append(
                f"| {name} | "
                + ("**outstanding**" if value is None else f"{value:.4f}")
                + " |"
            )
        lines.append("")
        for step in moved["steps"]:
            if step["delta"] is None:
                lines.append(
                    f"- `{step['from']}` → `{step['to']}`: **outstanding** — "
                    + (
                        f"no leaderboard score recorded in "
                        f"`artifacts/{LEADERBOARD}`."
                        if step["to"] == "leaderboard"
                        else f"`{step['to']}` has not been scored."
                    )
                )
            else:
                lines.append(
                    f"- `{step['from']}` → `{step['to']}`: **{step['sign']}"
                    f"{abs(step['delta']):.4f}**."
                )
        lines.append("")

    if missing:
        lines += [
            "## Scored here but never reported on `validation`",
            "",
            "These arms skipped the middle step of *chosen on tune, reported "
            f"on {REPORTED_ON}, confirmed on test*, so their engineering "
            "columns are their own and the note should say which rows they are.",
            "",
            *(f"- `{variant}`" for variant in missing),
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.final",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="score `test` again although it has been scored. Appends to the "
        "record rather than replacing it: a repository where test was scored "
        "twice says so in a file",
    )
    args = parser.parse_args(argv)

    invocation = "python -m pipeline.final " + " ".join(argv or [])
    status = 0
    for name in args.dataset or DEFAULT_DATASETS:
        try:
            entry = run(DATASETS[name], args.resamples, args.force, invocation.strip())
        except AlreadyScored as refused:
            print(f"\nrefused: {refused}\n")
            status = 2
            continue
        print(f"  recorded run {entry['run']} at {entry['scored_at']}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
