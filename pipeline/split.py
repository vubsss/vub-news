"""Temporal train/tune/validation/test split, with the leakage guards it needs.

The split is never random: an impression's label is a function of its timestamp
and nothing else. Window sizes are the only thing that differs between datasets
and they come from the registry, so nothing here knows which dataset it holds.

Four partitions rather than the three SPEC.md describes, because selecting a
parameter and reporting a result are different jobs and one partition cannot do
both. `train` is what a model fits on, `tune` is where every parameter is
chosen, `validation` is what gets reported, and `test` is held back for the end.
"""

from __future__ import annotations

import pandas as pd

from pipeline import ingest
from pipeline.datasets import DatasetConfig, SplitSpec

TRAIN, TUNE, VALIDATION, TEST = "train", "tune", "validation", "test"
# In time order, which is what check_ordering and every report iterate in.
SPLITS = (TRAIN, TUNE, VALIDATION, TEST)


class SplitError(RuntimeError):
    """The split is not strictly temporal. The build must not continue."""


# Above this share of histories, a future click stops being bad publication
# metadata and starts being a history joined from the wrong period or a split
# boundary off by one -- which is what the guard exists to catch, and which
# must stop the build rather than be dropped and mentioned. ebnerd_large sits
# at 0.001%, three orders of magnitude below.
FUTURE_CLICK_CEILING = 0.001


class LeakageError(RuntimeError):
    """An impression can see the future. The build must not continue."""


def check_leakage(
    behaviors: pd.DataFrame, history: pd.DataFrame, articles: pd.DataFrame
) -> dict[str, int]:
    """Reject any impression whose click history contains a future article.

    A click can only have happened after the impression if the article was not
    published until afterwards, so publication time is the evidence. Where a
    dataset ships none the check has nothing to work with, so the returned
    report says how many clicks it could actually see: a guard that is blind
    must not read as a clean result.

    An article appearing both in the history and in that impression's own
    candidate list is a different matter. Both datasets re-show articles a
    user has already read, so it is counted and reported, not rejected.
    """
    published = dict(
        zip(articles["article_id"], articles["published_time"], strict=True)
    )
    keyed = behaviors[["impression_id", "impression_time", "candidate_ids", *ingest.HISTORY_KEYS]]
    joined = keyed.merge(
        history[["impression_id", "click_history"]], on="impression_id"
    )

    report = {
        "impressions": len(joined),
        # Two bases, and they are different questions. `clicks` is what the
        # ranking stages will read, counted once per impression. `distinct` and
        # `checked` are the guard's own coverage, counted once per history --
        # reporting the guard's work against the readers' total would say the
        # check saw 6% of the clicks when it saw all of them.
        "clicks": 0,
        "distinct_clicks": 0,
        "clicks_checked": 0,
        "repeat_candidates": 0,
        # Clicks on an article published after the user's first impression.
        "future_clicks": 0,
        "future_histories": 0,
    }
    dropped: dict[tuple, set] = {}

    # One history per (user, source), not one per impression. The two are the
    # same object -- that is what the schema change made explicit -- and here
    # the difference is 1.6M checks instead of 24.6M on ebnerd_large, which is
    # what stopped this stage finishing at that scale.
    #
    # Checking a history against its key's **earliest** impression is exactly
    # equivalent to checking it against each of them. The history does not vary
    # between them, so a click published after any impression of that key is
    # published after the earliest one, and a click that clears the earliest
    # clears them all.
    earliest = joined.groupby(ingest.HISTORY_KEYS, sort=False)["impression_time"].min()
    seen: dict[tuple, set] = {}
    for key, clicks in zip(
        zip(*(joined[k] for k in ingest.HISTORY_KEYS)), joined["click_history"]
    ):
        if key in seen:
            continue
        seen[key] = set(clicks)
        report["distinct_clicks"] += len(clicks)
        at = earliest[key]
        for article_id in clicks:
            published_at = published.get(article_id)
            if published_at is None or pd.isna(published_at):
                continue
            report["clicks_checked"] += 1
            if published_at > at:
                # Dropped rather than fatal, and counted rather than dropped
                # quietly. On ebnerd_large this is 16 histories in 1,579,672
                # with overshoots up to 31 days, which is publication metadata
                # being wrong rather than a user seeing the future -- and it
                # never fired on ebnerd_small only because that samples 18,827
                # of the same 974,791 users. A guard that has never fired on a
                # sample has not been shown to be clean on the whole.
                #
                # What the guard is *for* is systematic leakage: a history
                # joined from the wrong period, an off-by-one in the split.
                # That shows up as a rate, not as sixteen rows, and the rate is
                # printed every run so it cannot become invisible.
                report["future_clicks"] += 1
                dropped.setdefault(key, set()).add(article_id)

    for key, gone in dropped.items():
        seen[key] -= gone
    report["future_histories"] = len(dropped)

    share = len(dropped) / max(len(seen), 1)
    if share > FUTURE_CLICK_CEILING:
        raise LeakageError(
            f"{len(dropped):,} of {len(seen):,} histories ({share:.2%}) hold a "
            f"click on an article published after that user's first impression. "
            f"Above {FUTURE_CLICK_CEILING:.1%} that is not publication metadata, "
            f"it is a history joined from the wrong period or a split boundary "
            f"off by one, and the build must not continue."
        )

    # Clicks are counted per impression, because that is what the number means:
    # how much history the ranking stages will read.
    for key, candidates in zip(
        zip(*(joined[k] for k in ingest.HISTORY_KEYS)), joined["candidate_ids"]
    ):
        clicked = seen[key]
        report["clicks"] += len(clicked)
        if clicked & set(candidates):
            report["repeat_candidates"] += 1
    return report


def check_ordering(labelled: pd.DataFrame) -> None:
    """Every train impression precedes every tune one, and so on.

    This is the anti-gaming assertion: a random split, or an off-by-one in the
    boundaries, shows up here as partitions that overlap in time.
    """
    windows = {
        name: group["impression_time"]
        for name, group in labelled.groupby("split", observed=True)
        if len(group)
    }
    ordered = [name for name in SPLITS if name in windows]
    for earlier, later in zip(ordered, ordered[1:]):
        if windows[earlier].max() >= windows[later].min():
            raise SplitError(
                f"{earlier} and {later} overlap in time: last {earlier} "
                f"impression is {windows[earlier].max()}, first {later} "
                f"impression is {windows[later].min()}"
            )


def assign(behaviors: pd.DataFrame, spec: SplitSpec) -> pd.DataFrame:
    """Label every impression train, tune, validation or test by its timestamp.

    Boundaries land on calendar-day edges, so a partition is a whole number of
    days and the printed report reads the way the split is described.

    The three held-out windows are measured back from the end of the log, so
    adding the tune window moves only the train/tune boundary: validation and
    test cover exactly the days they covered before it existed, and numbers
    reported on them stay comparable across the change.

    Bins are right-closed, so an impression landing exactly on a boundary
    belongs to the *earlier* partition. That is arbitrary but it has to be
    fixed: MIND has a single impression at exactly 2019-11-14 00:00:00, and a
    row that changed sides between runs would silently move a population.
    """
    times = behaviors["impression_time"]
    end = times.max().normalize() + pd.Timedelta(days=1)
    test_start = end - pd.Timedelta(days=spec.test_days)
    val_start = test_start - pd.Timedelta(days=spec.val_days)
    tune_start = val_start - pd.Timedelta(days=spec.tune_days)

    # Each partition paired with the edge it ends at. A window of zero days is
    # dropped rather than passed to pd.cut as a repeated edge, which it
    # rejects -- so `tune_days=0` means "no tune split" instead of a crash, and
    # a distribution long enough not to need one can say so in the registry.
    edges = [times.min() - pd.Timedelta(days=1)]
    labels: list[str] = []
    for name, boundary in (
        (TRAIN, tune_start),
        (TUNE, val_start),
        (VALIDATION, test_start),
        (TEST, end),
    ):
        if boundary > edges[-1]:
            edges.append(boundary)
            labels.append(name)

    labelled = behaviors.copy()
    labelled["split"] = pd.Series(
        pd.cut(times, bins=edges, labels=labels).astype(str),
        index=behaviors.index,
        dtype="string",
    )
    check_ordering(labelled)
    return labelled


def report_partitions(labelled: pd.DataFrame) -> None:
    """Row count and date range per partition, in split order."""
    for name in SPLITS:
        rows = labelled[labelled["split"] == name]
        if rows.empty:
            print(f"    {0:>9,} rows  {name:<11} (empty)")
            continue
        times = rows["impression_time"]
        share = 100 * len(rows) / len(labelled)
        print(
            f"    {len(rows):>9,} rows  {name:<11} {share:5.1f}%  "
            f"{times.min()} -> {times.max()}"
        )


def run(config: DatasetConfig, force: bool = False) -> None:
    """Label the feature store's impressions, guarding against leakage.

    Always recomputed rather than skipped when the column is already there:
    the labels are a pure function of the timestamps, so a second run writes
    exactly what the first one did, and there is nothing to preserve.
    """
    behaviors_path = config.feature_store_dir / "behaviors.parquet"
    behaviors = pd.read_parquet(behaviors_path)
    articles = pd.read_parquet(config.feature_store_dir / "articles.parquet")

    labelled = assign(behaviors, config.split)
    # The per-impression view, because the guard is about one impression's own
    # clicks against its own timestamp. The stored table is keyed by the user.
    history = ingest.history_for(config, labelled, columns=("click_history",))
    leakage = check_leakage(labelled, history, articles)
    labelled.to_parquet(behaviors_path, index=False)

    report_partitions(labelled)

    checked, distinct = leakage["clicks_checked"], leakage["distinct_clicks"]
    coverage = 100 * checked / distinct if distinct else 0.0
    print(
        f"    future-click guard checked {checked:,} of {distinct:,} distinct "
        f"clicks ({coverage:.1f}% have a publication time to check against), "
        f"over {leakage['clicks']:,} the ranking stages will read"
    )
    if leakage["future_clicks"]:
        share = 100 * leakage["future_histories"] / max(leakage["distinct_clicks"], 1)
        print(
            f"    dropped {leakage['future_clicks']:,} clicks from "
            f"{leakage['future_histories']:,} histories: published after that "
            f"user's first impression, which is publication metadata rather "
            f"than a user seeing the future at this rate ({share:.4f}%). A rate "
            f"that is not tiny is systematic leakage and must stop the build."
        )
    repeats, impressions = leakage["repeat_candidates"], leakage["impressions"]
    share = 100 * repeats / impressions if impressions else 0.0
    print(
        f"    {repeats:,} of {impressions:,} impressions ({share:.2f}%) offer "
        f"an article the user already clicked"
    )
