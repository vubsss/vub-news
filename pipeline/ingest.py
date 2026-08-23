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
        frame["source"] = _label(name)
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


# What a history row is keyed by, and what an impression is joined to it on.
HISTORY_KEYS = ["user_id", "source"]


def _one_per_user(frame: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Collapse a per-impression history table to one row per user per file.

    Lossless only if a user's history really is constant within one source
    file, which is a property of the data rather than of the schema. It holds
    on MINDsmall -- 94,057 users, not one of them with two different histories
    -- and this phase exists to move onto data it has not been checked on. So
    it is checked, on the cheap statistic that catches the realistic failure: a
    history that *grows* between two impressions of the same user, which is
    what a per-impression history would be for.

    A same-length substitution would slip through. That is a trade against
    fingerprinting every list on every ingest, and it is recorded here rather
    than left for someone to discover.
    """
    lengths = frame["click_history"].map(len)
    varies = lengths.groupby([frame[k] for k in HISTORY_KEYS]).nunique()
    if (varies > 1).any():
        worst = varies.idxmax()
        raise SchemaError(
            f"{config.name}: user {worst[0]!r} has {varies.max()} differently "
            f"sized histories inside {worst[1]!r}, so its history is not a "
            f"property of the user and cannot be keyed by one. "
            f"{int((varies > 1).sum()):,} users are affected."
        )
    return frame.drop_duplicates(HISTORY_KEYS, keep="first")


def history_for(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """The per-impression view of the per-user history table.

    The shape every ranking stage expects -- one row per impression, carrying
    that user's clicks -- rebuilt from a table that stores it once. The join is
    cheap in memory as well as on disk: a pandas object column holds pointers,
    so 477,534 impressions sharing 18,827 lists cost 477,534 pointers and
    18,827 lists, not 477,534 lists.

    A user with impressions and no history row is a cold start, not a missing
    row, so the clicks come back empty rather than null. Two emptinesses meet
    here and only one of them may be zipped against the clicks: a dataset with
    no engagement column at all keeps it null, while a dataset that has one
    gives this particular user an empty array.
    """
    stored = config.feature_store_dir / "history.parquet"
    frame = pd.read_parquet(
        stored,
        columns=None
        if columns is None
        else sorted({*HISTORY_KEYS, *columns} - {"impression_id"}),
    )
    return per_impression(frame, behaviors)


def per_impression(history: pd.DataFrame, behaviors: pd.DataFrame) -> pd.DataFrame:
    """The join itself, over a history frame already in hand.

    Separate from `history_for` because the stages have the store on disk and
    the tests have a frame -- and because the join is the part worth reading
    on its own.
    """
    joined = behaviors[["impression_id", *HISTORY_KEYS]].merge(
        history, on=HISTORY_KEYS, how="left"
    )

    if "click_history" in joined:
        joined["click_history"] = joined["click_history"].map(_as_clicks)
    for parallel, dtype in (
        ("click_times", "datetime64[us]"),
        ("click_read_times", "float32"),
        ("click_scroll", "float32"),
    ):
        if parallel in joined:
            joined[parallel] = joined[parallel].map(
                lambda values, dtype=dtype: np.empty(0, dtype=dtype)
                if values is None or np.isscalar(values)
                else values
            )
    if "n_clicks" in joined:
        joined["n_clicks"] = joined["n_clicks"].fillna(0).astype("int64")
    return joined


def build_history(config: DatasetConfig, behaviors: pd.DataFrame) -> pd.DataFrame:
    """One row per user per source file, carrying that user's click history.

    Where the history lives differs: MIND repeats it on every impression row,
    EB-NeRD keeps one row per user. Both are reduced to the same thing here --
    the second shape, which is the one with no redundancy in it. `history_for`
    turns it back into the per-impression view every ranking stage reads.
    """
    source = config.sources.history

    if "impression_id" in source.adapt(_read_one(config, source, source.files[0])):
        # The history arrives on the impression rows themselves, so there is one
        # copy of it per impression and they have to be collapsed. Which is
        # safe only if a user's history really is constant within one file --
        # asserted below rather than assumed, because it is a property of the
        # data and this project is about to change which data.
        frame = _one_per_user(_adapt_each_file(config, source), config)
    else:
        # Its own table already, one row per user per file. Nothing to join:
        # the fan-out onto impressions used to happen here and is what made
        # this table 25x larger than the information in it.
        parts = []
        for name in source.files:
            part = source.adapt(_read_one(config, source, name))
            part["source"] = _label(name)
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)

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
