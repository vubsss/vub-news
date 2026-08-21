"""Raw files -> the unified schema feature store.

Reads whatever the registry declares, hands it to that dataset's adapter, and
conforms the result to the canonical columns. Nothing here knows which dataset
it is looking at.
"""

from __future__ import annotations

import csv
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from pipeline.datasets import (
    ARTICLE_COLUMNS,
    BEHAVIOR_COLUMNS,
    COLUMN_DTYPES,
    HISTORY_COLUMNS,
    DatasetConfig,
    TableSource,
)


class SchemaError(RuntimeError):
    """Ingested data violates the unified schema's guarantees."""


def _as_clicks(clicks) -> list[str]:
    if clicks is None or (isinstance(clicks, float) and pd.isna(clicks)):
        return []
    return list(clicks)


def _label(name: str) -> str:
    """The source file's own name for itself: train, dev, validation."""
    path = PurePosixPath(name)
    return path.parent.name or path.stem


def _read_one(config: DatasetConfig, source: TableSource, name: str) -> pd.DataFrame:
    path = config.raw_dir / name
    if source.format == "tsv":
        # MIND's tsv files are headerless, and their titles contain quotes that
        # must not be read as field delimiters.
        return pd.read_csv(
            path,
            sep="\t",
            names=source.header,
            dtype=str,
            quoting=csv.QUOTE_NONE,
        )
    return pd.read_parquet(path)


def _read(config: DatasetConfig, source: TableSource) -> pd.DataFrame:
    frames = [_read_one(config, source, name) for name in source.files]
    return pd.concat(frames, ignore_index=True)


def _adapt_each_file(config: DatasetConfig, source: TableSource) -> pd.DataFrame:
    """Adapt each source file separately, so rows keep their file of origin.

    Impression ids are only unique within a file — MIND numbers both train and
    dev from 1 — so they are qualified with that origin before the files are
    concatenated.
    """
    parts = []
    for name in source.files:
        frame = source.adapt(_read_one(config, source, name))
        frame["impression_id"] = f"{_label(name)}-" + frame["impression_id"].astype(
            str
        )
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _conform(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Add the columns this dataset has no source for, fix order and dtypes."""
    for column in columns:
        if column not in frame:
            frame[column] = None
        frame[column] = frame[column].astype(COLUMN_DTYPES[column])
    return frame[list(columns)]


def build_articles(config: DatasetConfig) -> pd.DataFrame:
    source = config.sources.articles
    articles = source.adapt(_read(config, source))
    articles["dataset"] = config.name
    # The same article appears in more than one source file — MIND ships its
    # whole catalogue alongside both the train and the dev impressions.
    articles = articles.drop_duplicates(subset="article_id", ignore_index=True)
    return _conform(articles, ARTICLE_COLUMNS)


def build_behaviors(config: DatasetConfig) -> pd.DataFrame:
    source = config.sources.behaviors
    behaviors = _adapt_each_file(config, source)
    behaviors["dataset"] = config.name

    ragged = behaviors["candidate_ids"].map(len) != behaviors["labels"].map(len)
    if ragged.any():
        raise SchemaError(
            f"{config.name}: {int(ragged.sum())} impressions have a different "
            f"number of labels than candidates"
        )

    return _conform(behaviors, BEHAVIOR_COLUMNS)


def build_history(config: DatasetConfig, behaviors: pd.DataFrame) -> pd.DataFrame:
    """One row per impression, carrying that user's click history.

    Where the history lives differs: MIND repeats it on every impression row,
    EB-NeRD keeps one row per user. An adapter that already knows the
    impression is qualified per file like behaviors; one that only knows the
    user is joined onto the impressions instead.
    """
    source = config.sources.history

    if "impression_id" in source.adapt(_read_one(config, source, source.files[0])):
        frame = _adapt_each_file(config, source)
    else:
        # A user appears in more than one history file, with a different
        # history in each, so the join has to stay inside one source file or
        # every impression fans out across all of them.
        parts = []
        for name in source.files:
            label = _label(name)
            same_file = behaviors["impression_id"].str.startswith(f"{label}-")
            parts.append(
                source.adapt(_read_one(config, source, name)).merge(
                    behaviors.loc[same_file, ["impression_id", "user_id"]],
                    on="user_id",
                    how="right",
                )
            )
        frame = pd.concat(parts, ignore_index=True)

    # A user can have impressions but no history row at all — a cold user, not
    # a missing row, so the history is empty rather than null.
    frame["click_history"] = frame["click_history"].map(_as_clicks)
    # The same for the arrays that run parallel to it, and only where the
    # dataset supplies them. Two different emptinesses meet here: a dataset
    # with no engagement column at all keeps it null, which `_conform` fills,
    # while a dataset that has one gives this particular user an empty array —
    # and only the second may be zipped against the clicks. A null left here
    # would raise on the first weighting scheme to pair them.
    for parallel, dtype in (
        ("click_times", "datetime64[us]"),
        ("click_read_times", "float32"),
        ("click_scroll", "float32"),
    ):
        if parallel in frame:
            frame[parallel] = frame[parallel].map(
                lambda values, dtype=dtype: np.empty(0, dtype=dtype)
                if values is None or np.isscalar(values)
                else values
            )
    frame["n_clicks"] = frame["click_history"].map(len)
    frame["dataset"] = config.name
    return _conform(frame, HISTORY_COLUMNS)


def count_dangling(
    articles: pd.DataFrame, behaviors: pd.DataFrame, history: pd.DataFrame
) -> dict[str, int]:
    """How many referenced articles the catalogue does not actually contain.

    Reported rather than repaired: dropping the rows would quietly shrink the
    evaluation set, and dropping the ids would misalign candidates and labels.
    """
    catalogue = set(articles["article_id"])
    candidates = [aid for row in behaviors["candidate_ids"] for aid in row]
    clicks = [aid for row in history["click_history"] for aid in row]
    return {
        "candidates_total": len(candidates),
        "candidates_missing": sum(aid not in catalogue for aid in candidates),
        "clicks_total": len(clicks),
        "clicks_missing": sum(aid not in catalogue for aid in clicks),
    }


TABLES = ("articles", "behaviors", "history")


def run(config: DatasetConfig, force: bool = False) -> None:
    written = {name: config.feature_store_dir / f"{name}.parquet" for name in TABLES}
    if not force and all(path.exists() for path in written.values()):
        print(f"  {config.name} feature store is already built")
        return

    articles = build_articles(config)
    behaviors = build_behaviors(config)
    history = build_history(config, behaviors)
    tables = {"articles": articles, "behaviors": behaviors, "history": history}

    config.feature_store_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(written[name], index=False)
        print(f"    {len(table):>9,} rows  {name}")

    dangling = count_dangling(articles, behaviors, history)
    for kind in ("candidates", "clicks"):
        total, missing = dangling[f"{kind}_total"], dangling[f"{kind}_missing"]
        share = 100 * missing / total if total else 0.0
        print(
            f"    {missing:,} of {total:,} referenced {kind} "
            f"({share:.2f}%) are not in the article catalogue"
        )
