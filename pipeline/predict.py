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
the history the ranking is built from arrives on the impression row itself.

Everything a leaderboard file's shape depends on — where the test archive comes
from, how its files parse, what a line looks like, what the zip is called — is a
SubmissionSpec field. Nothing here reads a dataset name.
"""

from __future__ import annotations

import argparse
import csv
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline import acquire, evaluate, paths, preprocess, retrieval
from pipeline.datasets import DATASETS, DatasetConfig, TableSource

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

# Where the submission path's own index and vectors live. Apart from the
# pipeline's, under artifacts/<dataset>/, because they are built over a
# different catalogue: one directory for both would have a forced rebuild of
# either quietly overwrite the other with a matrix of the wrong articles.
WORK_DIR = "predict"


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


def impressions(config: DatasetConfig, chunk_size: int = CHUNK) -> Iterator[pd.DataFrame]:
    """The competition's test impressions, in file order, in chunks.

    File order is not a convenience: the leaderboard matches predictions to its
    gold labels by row, so a submission whose rows are reordered scores as one
    that ranked at random.
    """
    source = config.submission.impressions
    for name in source.files:
        for raw in _read(config, source, name, chunk_size):
            yield source.adapt(raw)


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
    else:
        frame = pd.read_parquet(path)
        if chunk_size is not None:
            return (
                frame.iloc[start : start + chunk_size]
                for start in range(0, len(frame), chunk_size)
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
            total += len(pd.read_parquet(path, columns=[]))
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


def write(
    config: DatasetConfig,
    retriever: str = DEFAULT_RETRIEVER,
    history_k: int = retrieval.HISTORY_K,
    chunk_size: int = CHUNK,
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

    module = evaluate.RETRIEVERS[retriever]
    workdir = config.artifacts_dir / WORK_DIR
    rank_with = module.ranker(articles, config, workdir, history_k)

    destination = paths.PREDICTIONS_DIR / config.name / spec.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    expected = source_rows(config)
    seen: set[str] = set()
    report = {"impressions": 0, "cold": 0, "flat": 0, "candidates": 0}
    started = time.perf_counter()

    with partial.open("w", encoding="utf-8") as handle:
        progress = tqdm(total=expected, unit="impression", desc="    ranking")
        for chunk in impressions(config, chunk_size):
            candidates = list(chunk["candidate_ids"])
            ranked = rank_with.rank(chunk, candidates)
            _check_alignment(chunk, ranked)

            for impression_id, given, order, scores in zip(
                chunk["impression_id"], candidates, ranked["ranked_ids"],
                ranked["scores"], strict=True,
            ):
                if impression_id in seen:
                    raise SubmissionError(
                        f"impression {impression_id} appears twice in "
                        f"{spec.impressions.files}"
                    )
                seen.add(impression_id)
                handle.write(spec.line(impression_id, ranks(given, order)) + "\n")
                report["candidates"] += len(given)
                # A ranking whose scores are all equal is the candidate file's
                # own order coming back out. Counted rather than hidden: it is
                # the share of the leaderboard score this system did not earn.
                report["flat"] += len(set(scores)) <= 1
            report["cold"] += int(chunk["click_history"].map(len).eq(0).sum())
            report["impressions"] += len(chunk)
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


def bundle(prediction: Path, config: DatasetConfig) -> Path:
    """Zip the prediction file under the name the leaderboard looks for.

    Written with `arcname` so the archive holds a bare file: CodaBench rejects
    a submission whose prediction file sits inside a directory, and a zip built
    from a path keeps the path.
    """
    archive = paths.PREDICTIONS_DIR / config.submission.bundle
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(prediction, arcname=config.submission.filename)
    return archive


def run(config: DatasetConfig, force: bool = False) -> None:
    """Fetch the competition's test set if needed, rank it, write the zip."""
    unbuilt = _unbuilt(config)
    if unbuilt:
        raise NotBuilt(
            f"{config.name}: no submission built yet — its registry entry has "
            f"no {unbuilt} (ticket 14)"
        )

    submit(config, force=force)


def submit(
    config: DatasetConfig,
    retriever: str = DEFAULT_RETRIEVER,
    history_k: int = retrieval.HISTORY_K,
    chunk_size: int = CHUNK,
    force: bool = False,
) -> Path:
    spec = config.submission
    archive = paths.PREDICTIONS_DIR / spec.bundle
    if archive.exists() and not force:
        print(f"    {archive} is already built")
        return archive

    acquire.ensure(config, spec.archives, spec.expected_files, spec.token_env)
    prediction, report = write(config, retriever, history_k, chunk_size)
    archive = bundle(prediction, config)

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
    args = parser.parse_args(argv)
    paths.load_env_file()

    for name in args.dataset or sorted(DATASETS):
        config = DATASETS[name]
        unbuilt = _unbuilt(config)
        if unbuilt:
            print(
                f"  {name}: no submission built yet — its registry entry has "
                f"no {unbuilt} (ticket 14)"
            )
            continue
        print(f"  {name}")
        submit(config, args.retriever, args.history_k, args.chunk, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
