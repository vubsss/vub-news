"""CodaBench submission: the competition's own impressions, ranked and packaged.

A different task from the evaluation harness, and the difference is the point.
The harness measures a retriever against a held-out slice of the feature store;
this scores what the competition hands over. The competition supplies the
candidate list per impression and expects exactly those back in an order, so
nothing here searches a corpus — `rank` is called and `retrieve` is not, and no
retrieval depth appears anywhere below.

It also supplies its own catalogue and its own click histories, covering a
later week than the feature store does. That is why the retriever is built over
the competition's articles rather than the pipeline's: MIND's test period holds
almost none of the articles MINDsmall does, so an index over the feature store
would score nearly every candidate 0 and submit the candidate file's own order.
For the same reason a user this project has never seen is not a special case —
the history the ranking is built from arrives with the competition's own files,
on the impression row or in a table beside it.

Everything a leaderboard file's shape depends on — where the test archive comes
from, how its files parse, what a line looks like, what the zip is called — is a
SubmissionSpec field. Nothing here reads a dataset name.
"""

from __future__ import annotations

import argparse
import dataclasses
import csv
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

import inspect

from pipeline import (
    acquire,
    evaluate,
    features,
    ledger,
    paths,
    preprocess,
    retrieval,
    timings,
    weighting,
)
from pipeline.datasets import (
    DATASETS,
    DEFAULT_DATASETS,
    DatasetConfig,
    TableSource,
    WeightingSpec,
)

# The retriever a submission is generated with unless one is named. Ticket 11
# separated the two on MIND's validation split by disjoint bootstrap intervals
# on every ranking metric the leaderboard reports — auc 0.6252 against 0.5597,
# mrr 0.3297 against 0.2999, ndcg@5 0.3058 against 0.2712, ndcg@10 0.3642
# against 0.3276, all favouring the semantic side. Submitting the lexical one
# instead would be submitting the retriever we measured to be worse.
DEFAULT_RETRIEVER = "ann"

# Impressions read, ranked and written at a time. MINDlarge_test holds 2.37M of
# them; one user vector per impression at 384 float32 is 3.6 GB before a single
# candidate is scored, so the file is streamed rather than loaded. The chunk is
# also the unit the output is written in, and the file is written in input
# order because the competition requires the rows to keep it.
CHUNK = 100_000

# Rows of the history table read at a time, where a competition ships one.
# Smaller than the impression chunk because a row here is a whole reading
# history rather than one impression: EB-NeRD's test histories average 144
# clicks over 808k users, so a chunk of this table costs what a chunk of ten
# times as many impressions does, and all but the last few clicks of it are
# dropped again immediately.
HISTORY_CHUNK = 10_000

# Where the submission path's own index and vectors live. Apart from the
# pipeline's, under artifacts/<dataset>/, because they are built over a
# different catalogue: one directory for both would have a forced rebuild of
# either quietly overwrite the other with a matrix of the wrong articles.
WORK_DIR = "predict"

# The stages a submission's wall time divides into, in the order a chunk
# passes through them. `read` and `write` are this module's; `features`,
# `nrms` and `gbdt` are filled by the ranker itself through
# `features.measured`, and a retriever that keeps no breakdown reports its
# ranking whole under `rank`.
#
# `rank` is the total of the three above it, not a sixth stage beside them --
# which is why it is reported last and labelled. The note needs to be able to
# say *which* stage breaks first at 10x, and a breakdown that did not sum to
# the thing it breaks down would not support that claim.
STAGES = ("read", "features", "nrms", "gbdt", "rank", "write")

# Chunk sizes the cost of a run is measured at before the run. Three points is
# enough for the shape: too small and the per-chunk fixed cost dominates, too
# large and the histories a chunk carries take the node out. A1's lesson was
# that the second is what actually happens, so the largest size that fits with
# headroom is the one submitted -- and the row that chose it is in the ledger.
BENCH_CHUNKS = (50_000, 200_000, 500_000)


class SubmissionError(RuntimeError):
    """The prediction file would not be what the competition asked for."""


class NotBuilt(RuntimeError):
    """This competition's submission is not described in the registry yet.

    Raised rather than returned quietly, because the caller is `build.py` and a
    stage that returns normally gets a checkpoint. One written here would have
    the rebuild skip this dataset's submission forever after the ticket that
    fills its registry entry in lands.
    """


def _unbuilt(config: DatasetConfig) -> str | None:
    """Which registry fields this dataset's submission is still missing."""
    spec = config.submission
    absent = [
        name
        for name in ("archives", "articles", "impressions", "filename", "bundle", "line")
        if getattr(spec, name) is None
    ]
    return ", ".join(absent) if absent else None


def catalogue(config: DatasetConfig) -> pd.DataFrame:
    """The competition's own article catalogue, preprocessed as the pipeline's.

    Through `preprocess.build_lexical_text` rather than a local copy of it: the
    lexical index built below only shares a vocabulary with its queries because
    both sides went through this dataset's language rules, and a submission
    corpus cleaned any other way would be a different retriever from the one
    every local number was measured on.
    """
    source = config.submission.articles
    articles = pd.concat(
        [source.adapt(_read(config, source, name)) for name in source.files],
        ignore_index=True,
    )
    articles["article_id"] = articles["article_id"].astype("string")
    articles["lexical_text"], _ = preprocess.build_lexical_text(articles, config)
    return articles


def impressions(
    config: DatasetConfig,
    chunk_size: int = CHUNK,
    history_k: int = retrieval.HISTORY_K,
) -> Iterator[pd.DataFrame]:
    """The competition's test impressions, in file order, in chunks.

    File order is not a convenience: the leaderboard matches predictions to its
    gold labels by row, so a submission whose rows are reordered scores as one
    that ranked at random.

    Each chunk carries the click history its rankings are built from, whether
    the competition put it on the impression row or in a table of its own —
    which of the two is a registry field, and the only thing below that knows
    the difference is whether `spec.history` is there.
    """
    source = config.submission.impressions
    streamed = _histories(config, history_k)
    for name in source.files:
        for raw in _read(config, source, name, chunk_size):
            chunk = source.adapt(raw)
            if streamed is not None:
                users = list(chunk["user_id"])
                chunk["click_history"] = [
                    streamed.clicks.get(user_id, []) for user_id in users
                ]
                for column, per_user in streamed.columns.items():
                    chunk[column] = [
                        per_user.get(user_id, _NOTHING) for user_id in users
                    ]
            yield chunk


# What a user the history table never mentions contributes to a weighted
# profile: no clicks, so no weights. Shared rather than allocated per row.
_NOTHING = np.empty(0)


@dataclass(frozen=True)
class Histories:
    """Every user's last history_k clicks, and what the weighting reads of them.

    `clicks` is the ids; `columns` holds one dict per parallel array the
    configured scheme needs — `click_read_times` and `click_scroll` for
    engagement, `click_times` for time, nothing at all for uniform. Keyed by
    column and then by user rather than a record per user, because a dict per
    user is 150 MB of dict headers at EB-NeRD's 808k of them.

    The arrays are truncated to the same `[-history_k:]` suffix as the ids, so
    they stay aligned with the window every consumer slices.
    """

    clicks: dict[str, list[str]]
    columns: dict[str, dict[str, np.ndarray]]


def _histories(config: DatasetConfig, history_k: int) -> Histories | None:
    """The history table a competition ships apart from its impressions.
    None when the impression row carries its own.

    Two things keep this inside a machine's memory rather than several times
    over. EB-NeRD's test table is 808k users averaging 144 clicks each, and
    both retrievers read only `[-k:]` of one — so the tail is taken as each
    chunk arrives and the rest is dropped before the next chunk is read, which
    is 116M ids read and 8M kept. And the ids that survive are interned against
    the 126k articles that exist, so a click on a popular article costs a
    pointer rather than another copy of its id.

    The same argument decides which of the parallel arrays come along: only the
    ones the configured scheme reads. Carrying `click_times` for an engagement
    profile that never looks at it would be 500 MB spent on nothing.
    """
    source = config.submission.history
    if source is None:
        return None

    wanted = weighting.columns_for(config.weighting.scheme)
    kept: dict[str, list[str]] = {}
    columns: dict[str, dict[str, np.ndarray]] = {name: {} for name in wanted}
    unique: dict[str, str] = {}
    for name in source.files:
        for raw in _read(config, source, name, HISTORY_CHUNK):
            rows = source.adapt(raw)
            for user_id, clicks in zip(rows["user_id"], rows["click_history"]):
                kept[user_id] = [
                    unique.setdefault(click, click) for click in clicks[-history_k:]
                ]
            for column in wanted:
                per_user = columns[column]
                for user_id, values in zip(rows["user_id"], rows[column]):
                    per_user[user_id] = np.asarray(values)[-history_k:]
    return Histories(clicks=kept, columns=columns)


def _read(
    config: DatasetConfig,
    source: TableSource,
    name: str,
    chunk_size: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """One raw file, whole or in chunks, read as ingest reads the same shapes."""
    path = config.raw_dir / name
    if source.format == "tsv":
        # Headerless, and MIND's titles carry quotes that are not delimiters —
        # the same reading ingest does, for the same files in a later week.
        frame = pd.read_csv(
            path,
            sep="\t",
            names=source.header,
            dtype=str,
            quoting=csv.QUOTE_NONE,
            chunksize=chunk_size,
        )
    elif chunk_size is None:
        frame = pd.read_parquet(path)
    else:
        # Batched off the file rather than sliced out of a frame that was read
        # whole: EB-NeRD's test behaviours are millions of rows of list-typed
        # columns, and reading them all to hand back a hundred thousand at a
        # time would put the thing chunking exists to avoid in memory first.
        return (
            batch.to_pandas()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_size)
        )
    return frame


def source_rows(config: DatasetConfig) -> int:
    """How many impressions the competition's files hold, counted from the file.

    Counted again from the bytes rather than taken from the reader that
    produced the predictions, so "every impression appears exactly once" is
    checked against the input and not against the thing being checked.
    """
    source = config.submission.impressions
    total = 0
    for name in source.files:
        path = config.raw_dir / name
        if source.format == "tsv":
            total += _lines(path)
        else:
            # Out of the footer rather than out of a read: a parquet file
            # records its own row count, and asking pandas for no columns hands
            # back a frame with no rows either, which would make this guard
            # certain that every competition sent nothing.
            total += pq.ParquetFile(path).metadata.num_rows
    return total


def _lines(path: Path) -> int:
    """Data lines in a headerless text file, whether or not it ends in one."""
    last = b""
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            count += block.count(b"\n")
            last = block[-1:]
    return count + (1 if last and last != b"\n" else 0)


def ranks(candidates: list[str], ranked: list[str]) -> list[int]:
    """The 1-based rank of each candidate, in the order the competition gave.

    The leaderboard reads position k of the line as "where did the k-th
    candidate place", so this is not the reordered candidate list — writing
    that instead is a submission that parses, scores, and means nothing.

    Every check the ticket asks for lands here, on every impression rather than
    on a sample: `ranked` must hold each candidate exactly once and add none of
    its own. The three conditions below leave only a bijection — `ranked` is
    the right length with no repeats, every candidate is in it, and no two
    candidates claim the same place.
    """
    place = {article_id: rank for rank, article_id in enumerate(ranked, start=1)}
    if len(ranked) != len(candidates) or len(place) != len(ranked):
        raise SubmissionError(
            f"ranking of {len(ranked)} article(s) ({len(place)} distinct) does "
            f"not match the {len(candidates)} candidate(s) it was built from"
        )

    try:
        order = [place[candidate] for candidate in candidates]
    except KeyError as error:
        raise SubmissionError(
            f"candidate {error.args[0]} is missing from its own ranking"
        ) from error

    if len(set(order)) != len(order):
        raise SubmissionError(
            f"a candidate appears more than once in an impression of "
            f"{len(candidates)}, so no ranking of it is well defined"
        )
    return order


def build_ranker(
    config: DatasetConfig,
    retriever: str,
    articles: pd.DataFrame,
    workdir: Path,
    history_k: int,
    device: str | None = None,
):
    """The named retriever over the competition's catalogue.

    `device` reaches only the two retrievers that have one. Giving all five a
    parameter that means nothing to three of them would be a wider interface
    for a narrower reason, and a `--device` silently ignored by BM25 is worse
    than one that is simply not passed to it.
    """
    module = evaluate.RETRIEVERS[retriever]
    extra = {}
    if device is not None and "device" in inspect.signature(module.ranker).parameters:
        extra["device"] = device
    return module.ranker(articles, config, workdir, history_k, **extra)


def write(
    config: DatasetConfig,
    retriever: str = DEFAULT_RETRIEVER,
    history_k: int = retrieval.HISTORY_K,
    chunk_size: int = CHUNK,
    device: str | None = None,
) -> tuple[Path, dict[str, int]]:
    """Rank every test impression and write the prediction file.

    The file is written to a `.part` and renamed only once the impression count
    has been checked against the input, so a run that stops halfway — or one
    whose guards fire on the last chunk — never leaves something that looks
    like a finished submission.
    """
    spec = config.submission
    articles = catalogue(config)
    print(f"    {len(articles):,} articles in the competition catalogue")

    workdir = config.artifacts_dir / WORK_DIR
    rank_with = build_ranker(
        config, retriever, articles, workdir, history_k, device
    )

    destination = (
        paths.PREDICTIONS_DIR
        / config.name
        / f"{label_for(config, retriever, history_k)}-{spec.filename}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    expected = source_rows(config)
    seen: set[str] = set()
    report = {"impressions": 0, "cold": 0, "flat": 0, "candidates": 0, "repeated": 0}
    # The ranker's own breakdown, if it keeps one, so the stages it measures
    # inside a chunk and the stages measured around it land in one dict.
    cost: dict[str, float] = getattr(rank_with, "cost", {})
    # Resident megabytes after each chunk. First and last are what say whether
    # the run is streaming: a path that accumulates anything shows it here and
    # nowhere else, because a leaderboard file from a run that grew is exactly
    # as correct as one from a run that did not.
    resident: list[float] = []
    started = time.perf_counter()

    with partial.open("w", encoding="utf-8") as handle:
        progress = tqdm(total=expected, unit="impression", desc="    ranking")
        stream = impressions(config, chunk_size, history_k)
        while True:
            with features.measured(cost, "read"):
                chunk = next(stream, None)
            if chunk is None:
                break
            candidates = list(chunk["candidate_ids"])
            with features.measured(cost, "rank"):
                ranked = rank_with.rank(chunk, candidates)
            _check_alignment(chunk, ranked)

            with features.measured(cost, "write"):
                for impression_id, given, order, scores in zip(
                    chunk["impression_id"], candidates, ranked["ranked_ids"],
                    ranked["scores"], strict=True,
                ):
                    if impression_id == spec.repeated_impression_id:
                        report["repeated"] += 1
                    elif impression_id in seen:
                        raise SubmissionError(
                            f"impression {impression_id} appears twice in "
                            f"{spec.impressions.files}"
                        )
                    else:
                        seen.add(impression_id)
                    handle.write(spec.line(impression_id, ranks(given, order)) + "\n")
                    report["candidates"] += len(given)
                    # A ranking whose scores are all equal is the candidate
                    # file's own order coming back out. Counted rather than
                    # hidden: it is the share of the leaderboard score this
                    # system did not earn.
                    report["flat"] += len(set(scores)) <= 1
            report["cold"] += int(chunk["click_history"].map(len).eq(0).sum())
            report["impressions"] += len(chunk)
            resident.append(timings.resident_mb())
            progress.update(len(chunk))
        progress.close()

    if report["impressions"] != expected:
        partial.unlink()
        raise SubmissionError(
            f"{config.name}: the competition's files hold {expected:,} "
            f"impressions but {report['impressions']:,} were written — the "
            f"submission would be scored against rows it does not line up with"
        )

    partial.replace(destination)
    report["seconds"] = round(time.perf_counter() - started)
    report["cost"] = dict(cost)
    report["chunk_size"] = chunk_size
    report["first_rss_mb"] = resident[0] if resident else None
    report["last_rss_mb"] = resident[-1] if resident else None
    report["peak_rss_mb"] = max(resident) if resident else None
    return destination, report


def _check_alignment(chunk: pd.DataFrame, ranked: pd.DataFrame) -> None:
    """The retriever's rows must still be the chunk's, in the chunk's order.

    The rankings are zipped back onto the chunk positionally, so a retriever
    that returned them in any other order would attach each one to a different
    impression and produce a file that is well-formed and entirely wrong.
    """
    if list(ranked["impression_id"]) != [str(i) for i in chunk["impression_id"]]:
        raise SubmissionError(
            "the retriever returned rankings for a different set of "
            "impressions, or in a different order, than it was given"
        )


# ---------------------------------------------------------------------------
# What the run cost, per stage.


def stage_rows(report: dict) -> list[tuple[str, float, float]]:
    """`(stage, seconds, impressions per second)` for the stages measured.

    In `STAGES` order rather than the dict's, so two runs' tables can be read
    side by side, and only for stages this retriever actually kept: a
    breakdown with an empty `nrms` row would suggest the stage ran and cost
    nothing.
    """
    seen = report["impressions"]
    rows = []
    for stage in STAGES:
        seconds = report.get("cost", {}).get(stage)
        if seconds is None:
            continue
        rows.append((stage, seconds, seen / seconds if seconds > 0 else float("nan")))
    return rows


def cost_lines(report: dict) -> list[str]:
    """The per-stage table, for the run's own output."""
    lines = [f"    {'stage':<10}{'seconds':>12}{'rows/s':>12}{'share':>8}"]
    total = report["seconds"] or 1
    for stage, seconds, per_second in stage_rows(report):
        share = "" if stage == "rank" else f"{100 * seconds / total:>7.1f}%"
        label = "rank*" if stage == "rank" else stage
        lines.append(f"    {label:<10}{seconds:>12,.1f}{per_second:>12,.0f}{share:>8}")
    if any(stage == "rank" for stage, _, _ in stage_rows(report)):
        lines.append("    * rank is the total of the stages above it, not a stage beside them")
    return lines


def record_cost(
    config: DatasetConfig, retriever: str, report: dict, split: str = "test"
) -> list[dict]:
    """One ledger row for the run and one per stage it measured.

    The functional side of these rows stays blank on purpose: the leaderboard
    holds the labels, so nothing here can score itself, and the number that
    fills that half arrives from CodaBench rather than from this process. The
    engineering side is the whole point of the row -- the per-stage rows/s are
    what ticket 09's 10x argument multiplies out, and it cannot say which
    stage breaks first from a single total.
    """
    chunk_size = report.get("chunk_size", CHUNK)
    # Thousands only where the size is exactly that many, so two sizes never
    # round to one variant name and quietly replace each other's ledger row.
    size = f"{chunk_size // 1000}k" if chunk_size % 1000 == 0 else str(chunk_size)
    base = f"{retriever}@{size}"
    first, last = report.get("first_rss_mb"), report.get("last_rss_mb")
    flat = (
        f"rss {first:,.0f} MB at the first chunk, {last:,.0f} MB at the last"
        if first is not None and last is not None
        else "rss not sampled"
    )
    written = [
        ledger.record(
            {
                "dataset": config.name,
                "stage": "submit",
                "variant": base,
                "split": split,
                "rows_per_s": round(report["impressions"] / report["seconds"], 1)
                if report["seconds"]
                else None,
                "peak_rss_mb": report.get("peak_rss_mb"),
                "train_seconds": report["seconds"],
                "note": (
                    f"{report['impressions']:,} impressions, "
                    f"{report['candidates']:,} candidates; {flat}"
                ),
            }
        )
    ]
    for stage, seconds, per_second in stage_rows(report):
        written.append(
            ledger.record(
                {
                    "dataset": config.name,
                    "stage": "submit",
                    "variant": f"{base}:{stage}",
                    "split": split,
                    "rows_per_s": round(per_second, 1),
                    "train_seconds": round(seconds, 1),
                    "note": "total of the three stages above it"
                    if stage == "rank"
                    else f"{stage} stage of the {base} submission run",
                }
            )
        )
    return written


def bench_chunks(
    config: DatasetConfig,
    retriever: str = DEFAULT_RETRIEVER,
    sizes: tuple[int, ...] = BENCH_CHUNKS,
    history_k: int = retrieval.HISTORY_K,
    device: str | None = None,
) -> list[dict]:
    """Rank the *first* chunk at each size and record what each one cost.

    Before the full run, not after it: the point is to choose the size the run
    uses, and a size chosen after the run is a size that was not chosen. The
    catalogue and the ranker are built once and reused across sizes, so what
    differs between the rows is the chunk and nothing else.

    Only the first chunk of each size is ranked, and nothing is written -- this
    produces no submission file, and says so by returning ledger rows instead
    of a path.
    """
    articles = catalogue(config)
    workdir = config.artifacts_dir / WORK_DIR
    rank_with = build_ranker(config, retriever, articles, workdir, history_k, device)
    rows = []
    for size in sizes:
        cost = getattr(rank_with, "cost", {})
        cost.clear()
        with timings.sample() as sampled:
            chunk = next(impressions(config, size, history_k), None)
            if chunk is None:
                break
            ranked = rank_with.rank(chunk, list(chunk["candidate_ids"]))
            _check_alignment(chunk, ranked)
        report = {
            "impressions": len(chunk),
            "candidates": int(chunk["candidate_ids"].map(len).sum()),
            "seconds": sampled["seconds"],
            "peak_rss_mb": round(sampled["peak_rss_mb"], 1),
            "first_rss_mb": sampled["entry_rss_mb"],
            "last_rss_mb": sampled["peak_rss_mb"],
            "chunk_size": size,
            "cost": dict(cost),
        }
        print(f"    chunk {size:,}: " + "; ".join(
            f"{stage} {per_second:,.0f} rows/s"
            for stage, _, per_second in stage_rows(report)
        ) + f"; peak {report['peak_rss_mb']:,.0f} MB")
        rows.extend(record_cost(config, f"{retriever}-bench", report))
        if len(chunk) < size:
            # The file ran out before the size did, so every larger size would
            # measure the same chunk again under a different name.
            break
    return rows


def label_for(
    config: DatasetConfig, retriever: str, history_k: int = retrieval.HISTORY_K
) -> str:
    """What to call this submission, given how it was actually produced.

    Anything that differs from the registry joins the name: the weighting
    scheme, and the history window. A file that ranked under a different model
    from the configured one must not be able to pass for the configured one on
    disk -- that is the failure class the guards in this pipeline exist
    against, and a filename is the cheapest place to keep the distinction.

    `ebnerd_submission-bm25-uniform-k10.zip` is a baseline produced knowingly:
    the configured model is engagement-weighted at k=80, which repeats each
    clicked title up to MAX_REPEAT times and projects to about thirty hours
    over 13.5M impressions.
    """
    parts = [retriever]
    if config.weighting.scheme != DATASETS[config.name].weighting.scheme:
        parts.append(config.weighting.scheme)
    if history_k != retrieval.HISTORY_K:
        parts.append(f"k{history_k}")
    return "-".join(parts)


def bundle_for(
    config: DatasetConfig, retriever: str, history_k: int = retrieval.HISTORY_K
) -> Path:
    """Where this retriever's submission zip goes.

    The retriever is in the *archive* name and never in the prediction file
    inside it: CodaBench looks for an exact filename, so renaming that would
    fail the upload, while the archive name is ours. Three retrievers per
    dataset would otherwise write over each other and the last one to run
    would silently be the submission.
    """
    stem = Path(config.submission.bundle).stem
    return paths.PREDICTIONS_DIR / f"{stem}-{label_for(config, retriever, history_k)}.zip"


def bundle(
    prediction: Path,
    config: DatasetConfig,
    retriever: str,
    history_k: int = retrieval.HISTORY_K,
) -> Path:
    """Zip the prediction file under the name the leaderboard looks for.

    Written with `arcname` so the archive holds a bare file: CodaBench rejects
    a submission whose prediction file sits inside a directory, and a zip built
    from a path keeps the path.
    """
    archive = bundle_for(config, retriever, history_k)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(prediction, arcname=config.submission.filename)
    return archive


def run(config: DatasetConfig, force: bool = False) -> None:
    """Fetch the competition's test set if needed, rank it, write the zip."""
    unbuilt = _unbuilt(config)
    if unbuilt:
        raise NotBuilt(
            f"{config.name}: no submission built yet — its registry entry has "
            f"no {unbuilt}"
        )

    submit(config, force=force)


def submit(
    config: DatasetConfig,
    retriever: str = DEFAULT_RETRIEVER,
    history_k: int = retrieval.HISTORY_K,
    chunk_size: int = CHUNK,
    force: bool = False,
    device: str | None = None,
) -> Path:
    spec = config.submission
    archive = bundle_for(config, retriever, history_k)
    if archive.exists() and not force:
        print(f"    {archive} is already built")
        return archive

    acquire.ensure(config, spec.archives, spec.expected_files, spec.token_env)
    prediction, report = write(config, retriever, history_k, chunk_size, device)
    archive = bundle(prediction, config, retriever, history_k)

    total = report["impressions"]
    print(
        f"    {total:,} impressions, {report['candidates']:,} candidates "
        f"ranked by {retriever} in {report['seconds']:,} s"
    )
    print(
        f"    {report['cold']:,} "
        f"({100 * report['cold'] / total if total else 0:.2f}%) carry no click "
        f"history; {report['flat']:,} "
        f"({100 * report['flat'] / total if total else 0:.2f}%) scored flat and "
        f"keep the order the competition listed them in"
    )
    if report["repeated"]:
        print(
            f"    {report['repeated']:,} carry the competition's repeated id "
            f"{spec.repeated_impression_id!r} and are matched by row, not by id"
        )
    for line in cost_lines(report):
        print(line)
    record_cost(config, retriever, report)
    print(f"    {archive} -> upload at {spec.competition_url}")
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="restrict to one competition (repeatable); default is all of them",
    )
    parser.add_argument(
        "--retriever",
        choices=sorted(evaluate.RETRIEVERS),
        default=DEFAULT_RETRIEVER,
        help=f"which retriever ranks the candidates (default {DEFAULT_RETRIEVER})",
    )
    parser.add_argument(
        "--weighting",
        choices=weighting.SCHEMES,
        help="rank under this click weighting instead of the registry's. The "
        "configured scheme is what every reported number was measured under, "
        "so a submission built this way is a different model, and it says so "
        "in its filename",
    )
    parser.add_argument(
        "--history-k",
        type=int,
        default=retrieval.HISTORY_K,
        help=f"clicks the query is built from (default {retrieval.HISTORY_K})",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=CHUNK,
        help=f"impressions held in memory at a time (default {CHUNK:,})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if the submission archive is already there",
    )
    parser.add_argument(
        "--device",
        help="torch device the NRMS half scores on, where the retriever has "
        "one; the other retrievers are not given the argument at all",
    )
    parser.add_argument(
        "--bench-chunks",
        nargs="?",
        const=",".join(str(size) for size in BENCH_CHUNKS),
        help="rank only the first chunk at each of these sizes and record what "
        "each cost, instead of writing a submission — the measurement that "
        "chooses --chunk for the full run",
    )
    args = parser.parse_args(argv)
    paths.load_env_file()

    for name in args.dataset or DEFAULT_DATASETS:
        config = DATASETS[name]
        unbuilt = _unbuilt(config)
        if unbuilt:
            print(
                f"  {name}: no submission built yet — its registry entry has "
                f"no {unbuilt}"
            )
            continue
        print(f"  {name}")
        if args.weighting:
            config = dataclasses.replace(
                config, weighting=WeightingSpec(scheme=args.weighting)
            )
            print(
                f"  ranking under {args.weighting!r} weighting, not the "
                f"registry's {DATASETS[name].weighting.scheme!r} — a different "
                f"model from the one every reported number was measured under"
            )
        if args.bench_chunks:
            bench_chunks(
                config,
                args.retriever,
                tuple(int(size) for size in args.bench_chunks.split(",")),
                args.history_k,
                args.device,
            )
            continue
        submit(
            config,
            args.retriever,
            args.history_k,
            args.chunk,
            args.force,
            args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
