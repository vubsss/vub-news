"""Temporal train/validation/test split, with the leakage guards it needs.

The split is never random: an impression's label is a function of its timestamp
and nothing else. Window sizes are the only thing that differs between datasets
and they come from the registry, so nothing here knows which dataset it holds.
"""

from __future__ import annotations

import pandas as pd

from pipeline.datasets import DatasetConfig, SplitSpec

TRAIN, VALIDATION, TEST = "train", "validation", "test"
SPLITS = (TRAIN, VALIDATION, TEST)


class SplitError(RuntimeError):
    """The split is not strictly temporal. The build must not continue."""


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
    joined = behaviors[["impression_id", "impression_time", "candidate_ids"]].merge(
        history[["impression_id", "click_history"]], on="impression_id"
    )

    report = {
        "impressions": len(joined),
        "clicks": 0,
        "clicks_checked": 0,
        "repeat_candidates": 0,
    }
    for impression_id, at, candidates, clicks in joined.itertuples(index=False):
        report["clicks"] += len(clicks)
        for article_id in clicks:
            published_at = published.get(article_id)
            if published_at is None or pd.isna(published_at):
                continue
            report["clicks_checked"] += 1
            if published_at > at:
                raise LeakageError(
                    f"impression {impression_id} at {at} has {article_id} in "
                    f"its click history, but {article_id} was not published "
                    f"until {published_at}"
                )
        if set(clicks) & set(candidates):
            report["repeat_candidates"] += 1
    return report


def check_ordering(labelled: pd.DataFrame) -> None:
    """Every train impression precedes every validation one, and so on.

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
    """Label every impression train, validation or test by its timestamp.

    Boundaries land on calendar-day edges, so a partition is a whole number of
    days and the printed report reads the way the split is described.
    """
    times = behaviors["impression_time"]
    end = times.max().normalize() + pd.Timedelta(days=1)
    test_start = end - pd.Timedelta(days=spec.test_days)
    val_start = test_start - pd.Timedelta(days=spec.val_days)

    labelled = behaviors.copy()
    labelled["split"] = pd.Series(
        pd.cut(
            times,
            bins=[times.min() - pd.Timedelta(days=1), val_start, test_start, end],
            labels=[TRAIN, VALIDATION, TEST],
        ).astype(str),
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
    history = pd.read_parquet(config.feature_store_dir / "history.parquet")
    articles = pd.read_parquet(config.feature_store_dir / "articles.parquet")

    labelled = assign(behaviors, config.split)
    leakage = check_leakage(labelled, history, articles)
    labelled.to_parquet(behaviors_path, index=False)

    report_partitions(labelled)

    checked, clicks = leakage["clicks_checked"], leakage["clicks"]
    coverage = 100 * checked / clicks if clicks else 0.0
    print(
        f"    future-click guard checked {checked:,} of {clicks:,} clicks "
        f"({coverage:.1f}% have a publication time to check against)"
    )
    repeats, impressions = leakage["repeat_candidates"], leakage["impressions"]
    share = 100 * repeats / impressions if impressions else 0.0
    print(
        f"    {repeats:,} of {impressions:,} impressions ({share:.2f}%) offer "
        f"an article the user already clicked"
    )
