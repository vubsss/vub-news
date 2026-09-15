"""The long frame: one row per (impression, candidate), and nothing in it that
the server would not have had.

Everything the re-ranker sees is assembled here, which makes this the one place
a future click could get into A2 -- so the module is organised around
availability rather than around where a number comes from. The four groups in
`FEATURE_GROUPS` are tiers, not folders: `content` is what a server holds
before anyone clicks anything, `history` what it holds about this user from
their past clicks, `exposure` what the log has *shown* by the moment of the
impression, and `clicked` what the log has recorded as clicked by that moment.
Ticket 07's ablation drops them one at a time, and the tier a column sits in is
the claim about when it would be available -- which is why `check_columns`
refuses a frame whose columns do not partition exactly.

Three rules the frame is built to keep:

**Candidates stay in arrival order.** The frame is the candidate list exploded
in the order the dataset gave it, so a re-ranker that ties two candidates
leaves them where the harness would have left them. Reading a retriever's
scores back therefore goes *through* `ranked_ids` -- both retrievers return
their candidates sorted by score, and zipping `scores` onto the arrival order
would pair every candidate with another's number while looking entirely
well-formed.

**Every read of the log is strictly before `impression_time`.** The counters
enforce that themselves (see `pipeline.counters`); the session features do the
same arithmetic with `searchsorted(..., side="left")`. `build(..., causal=False)`
is the one exception and exists so Q9 can measure what the leakage is worth: it
reads the same counter store over the whole log and writes to its own file, so
the leaky arm is a flag rather than a second feature set somebody has to keep
in step.

**The outcome columns are named and refused.** `FORBIDDEN` lists the four that
describe the impression being scored rather than a past one -- EB-NeRD's
`next_read_time` and `next_scroll_percentage`, and the impression row's own
`read_time` and `scroll_percentage`. `check_columns` fails the build if any of
them reaches the frame, under its own name or as the tail of a feature's. The
dwell and scroll features here read `click_read_times` and `click_scroll`,
which are the *history* table's arrays over clicks that already happened.

Two things that are deliberately **not** variations:

- *Freshness encoding.* A gradient-boosted tree splits on thresholds and is
  invariant to any monotone transform of a feature, so `log_hours` would be the
  same column as `hours` and would only split one column's gain in two. Raw
  hours is the only encoding, and this sentence is here so nobody adds the
  other one later thinking it was overlooked.
- *A weighted `last` pooling.* `last` is the similarity to the single most
  recent click; a profile's weight on it is one positive per-impression scalar
  that says nothing about the candidate. `mean` and `max` get a column per
  profile because the weights genuinely change them -- a different average and
  a different argmax -- and `last` gets one column.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline import (
    ann_index,
    bm25_index,
    counters,
    embed,
    ingest,
    ledger,
    retrieval,
    timings,
    weighting,
)
from pipeline.datasets import DATASETS, DEFAULT_DATASETS, DatasetConfig, FeatureSpec

STAGE = "features"
DIRECTORY = "features"

# The splits the stage materialises. `train` because the re-ranker fits on its
# later half, `tune` because every choice is made there, `validation` because
# that is where results are reported. `test` is buildable by name from the
# command below and is not built here: a frame rebuilt into every
# `python build.py` is a held-back split somebody ends up looking at.
SPLITS = ("train", "tune", "validation")

# The columns that describe the impression being scored rather than one that
# already happened. Named here so the guard is a list somebody can read, and
# checked by name *and* by suffix: a feature called `mean_read_time` would be
# this data under another name.
FORBIDDEN = (
    "next_read_time",
    "next_scroll_percentage",
    "read_time",
    "scroll_percentage",
)

# What identifies a row. `impression_id` and `article_id` are what the frame is
# joined back on; `label` is the target and is never a feature.
KEY_COLUMNS = ("impression_id", "article_id", "label")

# The retrievers whose scores are content features, under the names their
# ledger rows already carry. Each is reached through `rank_candidates`, the
# same call the harness makes -- a score computed here by any other path would
# be a different number under an existing name.
SCORERS = {"ann": ann_index, "bm25": bm25_index}


class FeatureError(RuntimeError):
    """The frame holds something it must not, or is missing something it must."""


@dataclass(frozen=True)
class Profile:
    """One way of weighting the clicks a user profile is built from.

    `scheme` and `decay` are `pipeline.weighting`'s, so a profile a dataset
    cannot express -- MIND has no click timestamps and no engagement -- is
    detected from the registry's ColumnMap rather than from the dataset's name,
    and its columns come out null the way `session_id` does. LightGBM takes the
    NaN; nothing downstream asks which dataset it is holding.
    """

    name: str
    scheme: str
    decay: float


# The recency axis, as columns side by side rather than as a switch. A1 chose
# one scheme per dataset for the *retriever*, by ranking with it; the question
# here is a different one -- a tree can use several profiles at once, and which
# of them earn gain is ticket 06's table to read rather than this module's
# choice to make. The half-lives at 6/24/72 hours bracket the day a news cycle
# lasts; the two position decays are the same idea for a dataset whose clicks
# carry no clock.
PROFILES = (
    Profile("uniform", "uniform", 1.0),
    Profile("pos90", "position", 0.90),
    Profile("pos99", "position", 0.99),
    Profile("t6", "time", 6.0),
    Profile("t24", "time", 24.0),
    Profile("t72", "time", 72.0),
    Profile("engagement", "engagement", 1.0),
)

# The availability tiers, and the whole of the frame's schema. Ticket 07's
# ablation iterates this mapping, so a feature in no group is a feature no arm
# can drop and one in two groups is a feature two arms would both claim to have
# dropped. `check_columns` refuses both.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "content": (
        "ann_score",
        "ann_rank",
        "bm25_score",
        "bm25_rank",
        "n_candidates",
        "category_share",
        "subcategory_share",
    ),
    "history": (
        "n_clicks",
        *(f"hist_cos_mean_{profile.name}" for profile in PROFILES),
        *(f"hist_cos_max_{profile.name}" for profile in PROFILES),
        "hist_cos_last",
        "past_dwell",
        "past_depth",
    ),
    "exposure": (
        *(f"exposures_{window}" for window in counters.WINDOWS),
        "freshness_hours",
        "session_impressions",
        "session_clicks",
    ),
    "clicked": (
        *(f"clicks_{window}" for window in counters.WINDOWS),
        *(f"ctr_{window}" for window in counters.WINDOWS),
    ),
}

FEATURES = tuple(name for group in FEATURE_GROUPS.values() for name in group)
COLUMNS = (*KEY_COLUMNS, *FEATURES)

# The two engagement features, and the history column each reads. Named
# `past_*` rather than `dwell`/`scroll` so that the forbidden-suffix check
# stays a check rather than a thing to remember: the impression's own
# `read_time` and `scroll_percentage` cannot be spelled this way by accident.
ENGAGEMENT = {"past_dwell": "click_read_times", "past_depth": "click_scroll"}


def check_columns(columns) -> None:
    """That the frame is exactly the schema, and that the future stayed out.

    Three failures, every one of which would otherwise be silent: an outcome
    column riding along under its own name or as the tail of a feature's, a
    feature in two tiers or in none, and a frame that has drifted from
    `COLUMNS` in either direction.
    """
    columns = list(columns)
    leaked = [
        name
        for name in columns
        if any(name == bad or name.endswith(f"_{bad}") for bad in FORBIDDEN)
    ]
    if leaked:
        raise FeatureError(
            f"the frame carries the impression's own outcome: "
            f"{', '.join(sorted(leaked))}. Those columns describe the click "
            f"being predicted rather than one that already happened; the "
            f"engagement features read the history table's parallel arrays."
        )

    placed: dict[str, list[str]] = {}
    for tier, names in FEATURE_GROUPS.items():
        for name in names:
            placed.setdefault(name, []).append(tier)
    twice = {name: tiers for name, tiers in placed.items() if len(tiers) > 1}
    if twice:
        raise FeatureError(
            "a feature belongs to exactly one availability tier: "
            + "; ".join(
                f"{name} is in {', '.join(tiers)}" for name, tiers in twice.items()
            )
        )

    missing = [name for name in COLUMNS if name not in columns]
    extra = [name for name in columns if name not in COLUMNS]
    if missing or extra:
        raise FeatureError(
            f"the frame is not the schema: missing {missing or 'nothing'}, "
            f"unexpected {extra or 'nothing'}"
        )


def tier_of(column: str) -> str:
    for tier, names in FEATURE_GROUPS.items():
        if column in names:
            return tier
    raise FeatureError(f"{column!r} is in no availability tier")


def columns_for(groups) -> tuple[str, ...]:
    """The key columns plus every feature in the named tiers.

    What ticket 07 hands `read`: an arm is a set of tiers, and dropping one is
    a projection rather than a column of zeros -- a zeroed column still costs
    the read, and a tree can still split on it.
    """
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise FeatureError(
            f"not availability tiers: {', '.join(unknown)}. "
            f"Known: {', '.join(FEATURE_GROUPS)}"
        )
    return (*KEY_COLUMNS, *(name for group in groups for name in FEATURE_GROUPS[group]))


@contextmanager
def measured(cost: dict[str, float], family: str):
    """Add a block's wall seconds to `cost[family]`, so the note can say which
    feature family is the expensive one rather than only what the frame took.

    Public because the submission path times its own stages -- read, features,
    nrms, gbdt, write -- into a dict of the same shape, and two accumulators
    that round differently would make the per-stage table and the total
    disagree about the same run.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        cost[family] = cost.get(family, 0.0) + time.perf_counter() - started


# ---------------------------------------------------------------------------
# What a chunk reads and does not rebuild.


@dataclass(frozen=True)
class Loaded:
    """The stores every chunk shares, opened once per build.

    The vectors are memory-mapped: the cosine features gather a few dozen rows
    per impression out of a matrix that is 190 MB on MIND, and a resident copy
    per process is what ticket 09 would otherwise have to quote as this stage's
    RAM. The retriever scores are the exception and deliberately so -- they come
    from `rank_candidates`, the harness's own call, which loads its own vectors
    per chunk. Recomputing them here from the mmap would be a second
    implementation of a published number.

    `published` is None for a dataset whose registry maps no publish time, and
    freshness falls back to the log's first sighting of the article. The choice
    is read off the ColumnMap rather than from the dataset's name, because it
    is a property of what the source ships.
    """

    embeddings: embed.Embeddings
    category: pd.Series
    subcategory: pd.Series
    published: pd.Series | None
    counts: counters.CounterStore
    sessions: pd.DataFrame


def _keyed(articles: pd.DataFrame, column: str) -> pd.Series:
    """One article column, indexed by article id, for a bulk `reindex`."""
    return pd.Series(
        articles[column].to_numpy(), index=articles["article_id"].to_numpy()
    )


def session_counts(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Impressions and clicks earlier in the same session, per impression.

    Strictly earlier, by the rule the counters keep: an impression does not see
    itself, and two impressions stamped at the same instant do not see each
    other. `searchsorted(..., side="left")` is what says so -- the count before
    `t` is where `t` would be inserted ahead of its equals.

    A dataset with no sessions gets NaN counts, which is the shape MIND's
    `session_id` already has in the feature store.
    """
    ids = behaviors["impression_id"].to_numpy()
    # A log with no labels is the competition's test file: how many of a
    # session's earlier impressions were clicked is exactly what it is holding
    # back, so that column is unknown rather than zero. The count of earlier
    # impressions is not -- the server saw those.
    known = "labels" in behaviors
    impressions_before = np.full(len(behaviors), np.nan)
    clicks_before = np.full(len(behaviors), np.nan)
    counted = pd.DataFrame(
        {
            "session_impressions": impressions_before,
            "session_clicks": clicks_before,
        },
        index=pd.Index(ids, name="impression_id"),
    )
    if "session_id" not in behaviors or behaviors["session_id"].isna().all():
        return counted

    moments = behaviors["impression_time"].to_numpy("datetime64[us]")
    clicked = (
        behaviors["labels"].map(lambda labels: int(np.sum(labels))).to_numpy()
        if known
        else np.zeros(len(behaviors), dtype="int64")
    )
    for positions in behaviors.groupby("session_id", dropna=True).indices.values():
        order = positions[np.argsort(moments[positions], kind="stable")]
        times = moments[order]
        before = np.searchsorted(times, times, side="left")
        running = np.concatenate([[0], np.cumsum(clicked[order])])
        impressions_before[order] = before
        if known:
            clicks_before[order] = running[before]
    counted["session_impressions"] = impressions_before
    counted["session_clicks"] = clicks_before
    return counted


def load(config: DatasetConfig, behaviors: pd.DataFrame) -> Loaded:
    """Every store a chunk reads, opened once for the whole build."""
    articles = pd.read_parquet(
        config.feature_store_dir / "articles.parquet",
        columns=["article_id", "category", "subcategory", "published_time"],
    )
    return Loaded(
        embeddings=embed.load(config, mmap=True),
        category=_keyed(articles, "category"),
        subcategory=_keyed(articles, "subcategory"),
        published=(
            _keyed(articles, "published_time")
            if config.columns.articles["published_time"]
            else None
        ),
        counts=counters.load(config.artifacts_dir / counters.DIRECTORY),
        sessions=session_counts(behaviors),
    )


def for_submission(
    config: DatasetConfig,
    articles: pd.DataFrame,
    embeddings: embed.Embeddings,
    test_impressions: pd.DataFrame,
) -> Loaded:
    """The same context, over the competition's catalogue and the test log.

    Every field means what it means offline; only where it comes from differs,
    which is what keeps the submission on one frame builder:

    - the vectors are the competition's, corrected as the retriever corrected
      them, because the feature store's artifact barely covers its catalogue;
    - the article maps are the competition's own `articles.parquet`;
    - the counters are `counters.ServingCounters` -- the training log frozen,
      the test log's exposures read strictly before `t`, and the clicks a test
      user's history reveals -- behind the same `at` and `first_seen`;
    - the session counts come from the test log, with the click side unknown
      because the leaderboard holds the labels back.
    """
    from pipeline import counters as counter_store

    trained = counter_store.load(config.artifacts_dir / counter_store.DIRECTORY)
    histories = (
        test_impressions["click_history"]
        if "click_history" in test_impressions
        else []
    )
    clicked_ids, clicked_counts = counter_store.clicks_in_histories(histories)
    return Loaded(
        embeddings=embeddings,
        category=_keyed(articles, "category"),
        subcategory=_keyed(articles, "subcategory"),
        published=(
            _keyed(articles, "published_time")
            if config.columns.articles["published_time"]
            else None
        ),
        counts=counter_store.ServingCounters(
            trained=trained,
            shown=counter_store.exposures_only(test_impressions),
            clicked_ids=clicked_ids,
            clicked_counts=clicked_counts,
        ),
        sessions=session_counts(test_impressions),
    )


# ---------------------------------------------------------------------------
# The feature families that need more than a lookup.


def read_back(
    ranked: pd.DataFrame, chunk: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """A retriever's score and 1-based rank per candidate, in *arrival* order.

    Through `ranked_ids`, never by position: both retrievers emit their
    candidates sorted by score, so zipping `scores` onto the candidate list
    would give every candidate another one's number in a frame that still had
    the right shape and the right row count.
    """
    by_impression = {
        impression: (ids, scores)
        for impression, ids, scores in zip(
            ranked["impression_id"], ranked["ranked_ids"], ranked["scores"]
        )
    }
    missing = set(chunk["impression_id"]) - set(by_impression)
    if missing:
        raise FeatureError(
            f"{len(missing)} impressions came back without a ranking, "
            f"e.g. {sorted(missing)[:3]}"
        )

    scores: list[float] = []
    ranks: list[float] = []
    for impression, candidates in zip(chunk["impression_id"], chunk["candidate_ids"]):
        ranked_ids, ranked_scores = by_impression[impression]
        place = {
            article: (score, position + 1.0)
            for position, (article, score) in enumerate(zip(ranked_ids, ranked_scores))
        }
        for candidate in candidates:
            score, rank = place.get(candidate, (np.nan, np.nan))
            scores.append(score)
            ranks.append(rank)
    return np.asarray(scores, dtype="float64"), np.asarray(ranks, dtype="float64")


def profiles_for(config: DatasetConfig, history_k: int) -> tuple[Profile, ...]:
    """The profiles this dataset's history has the columns to express.

    From `weighting.available`, which reads the registry's ColumnMap -- so the
    answer is a property of what the source ships rather than a second list
    here that would have to be kept true.
    """
    expressible = weighting.available(config)
    chosen = tuple(profile for profile in PROFILES if profile.scheme in expressible)
    for profile in chosen:
        weighting.check(config, profile.scheme, history_k)
    return chosen


def _window(column: pd.Series | None, position: int, history_k: int):
    """One impression's engagement array, cut to the same last-K window every
    other history feature reads."""
    if column is None:
        return None
    values = column.iloc[position]
    if values is None:
        return None
    array = np.asarray(values)
    return array[-history_k:] if array.size else array


def cosine_columns(
    chunk: pd.DataFrame,
    history: pd.DataFrame,
    embeddings: embed.Embeddings,
    profiles: tuple[Profile, ...],
    history_k: int,
) -> dict[str, np.ndarray]:
    """One column per (pooling, profile) over the user's last-K clicks.

    The (candidates x clicks) similarity matrix is computed once per impression
    and every column is an aggregate of it. Calling `rank_candidates` once per
    profile instead would re-read the vectors and rebuild the index for each of
    fifteen columns and produce the same numbers.

    The mean's weights are normalised to sum to one, so the column is a
    weighted average of similarities whatever the history's length. The max's
    are normalised to their own maximum instead, so an unweighted max is the
    plain largest similarity and recency enters as a discount on older clicks
    rather than as a rescale of all of them.
    """
    clicks = ann_index.build_clicks(history, embeddings, history_k)
    if list(clicks.impression_ids) != list(chunk["impression_id"].astype("string")):
        raise FeatureError(
            "the history rows are not in the impressions' order, so every "
            "profile would be built from another impression's clicks"
        )

    rows_total = int(chunk["candidate_ids"].map(len).sum())
    columns = {
        name: np.full(rows_total, np.nan)
        for name in FEATURE_GROUPS["history"]
        if name.startswith("hist_cos_")
    }
    row_of = embeddings.index
    engagement = {
        name: (history[name] if name in history else None)
        for name in ("click_times", "click_read_times", "click_scroll")
    }

    at = 0
    for position, candidates in enumerate(chunk["candidate_ids"]):
        start, at = at, at + len(candidates)
        rows, kept = clicks.rows[position], clicks.kept[position]
        candidate_rows = np.array([row_of.get(article, -1) for article in candidates])
        known = candidate_rows >= 0
        if not len(rows) or not known.any():
            continue

        similarity = np.asarray(
            embeddings.vectors[candidate_rows[known]] @ embeddings.vectors[rows].T,
            dtype="float32",
        )
        placed = start + np.flatnonzero(known)
        columns["hist_cos_last"][placed] = similarity[:, -1]

        window = len(history["click_history"].iloc[position][-history_k:])
        for profile in profiles:
            found = weighting.weights(
                profile.scheme,
                profile.decay,
                window,
                times=_window(engagement["click_times"], position, history_k),
                read_times=_window(
                    engagement["click_read_times"], position, history_k
                ),
                scroll=_window(engagement["click_scroll"], position, history_k),
                at=chunk["impression_time"].iloc[position],
            )
            weights = found[kept] if len(found) else found
            if not len(weights):
                continue
            columns[f"hist_cos_mean_{profile.name}"][placed] = similarity @ (
                weighting.normalise(weights).astype("float32")
            )
            largest = weights.max()
            discount = weights / largest if largest > 0 else np.ones_like(weights)
            columns[f"hist_cos_max_{profile.name}"][placed] = (
                similarity * discount.astype("float32")
            ).max(axis=1)
    return columns


def category_columns(
    chunk: pd.DataFrame,
    history: pd.DataFrame,
    loaded: Loaded,
    article_ids: np.ndarray,
    history_k: int,
) -> dict[str, np.ndarray]:
    """The share of the user's recent clicks that sat in the candidate's
    category, and the same for its subcategory.

    A content feature by the tier's definition -- the category is on the
    article before anyone clicks it -- read over the same last-K window every
    other history feature uses, so that "recent" means one thing in the frame.
    """
    columns = {
        name: np.full(len(article_ids), np.nan)
        for name in ("category_share", "subcategory_share")
    }
    at = 0
    for position, candidates in enumerate(chunk["candidate_ids"]):
        start, at = at, at + len(candidates)
        clicked = list(history["click_history"].iloc[position])[-history_k:]
        if not clicked:
            continue
        for name, keyed in (
            ("category_share", loaded.category),
            ("subcategory_share", loaded.subcategory),
        ):
            seen = pd.Series(keyed.reindex(clicked).to_numpy()).value_counts()
            share = (
                keyed.reindex(article_ids[start:at])
                .map(seen)
                .to_numpy(dtype="float64", na_value=0.0)
            )
            columns[name][start:at] = share / len(clicked)
    return columns


def counter_columns(
    loaded: Loaded,
    article_ids: np.ndarray,
    moments: np.ndarray,
    causal: bool,
) -> dict[str, np.ndarray]:
    """Exposures, clicks, CTR per window, and freshness.

    One `at()` call per window over the chunk's whole flattened
    `(article, moment)` list -- never one call per impression, which would pay
    the store's setup cost a million times over a run.

    `causal=False` is Q9's leaky arm, and it is the same store read at a later
    moment rather than a second implementation: the cumulative window reads the
    whole log, and a sliding window is shifted forward by its own length, so
    `[t - 1h, t)` becomes `[t, t + 1h)` -- the hour the server could not have
    had, including the impression's own outcome. Freshness leaks the same way,
    through an article's first sighting anywhere in the log rather than its
    first sighting before `t`.
    """
    columns: dict[str, np.ndarray] = {}
    for name, window in counters.WINDOWS.items():
        if causal:
            counted = loaded.counts.at(article_ids, moments, window)
        elif window is None:
            counted = loaded.counts.at(article_ids, counters.WHOLE_LOG)
        else:
            ahead = moments + np.timedelta64(window.value // 1000, "us")
            counted = loaded.counts.at(article_ids, ahead, window)
        columns[f"exposures_{name}"] = counted.exposures.astype("float64")
        columns[f"clicks_{name}"] = counted.clicks.astype("float64")
        columns[f"ctr_{name}"] = counted.ctr

    if loaded.published is not None:
        published = loaded.published.reindex(article_ids).to_numpy("datetime64[us]")
    else:
        published = loaded.counts.first_seen(
            article_ids, moments if causal else counters.WHOLE_LOG
        )
    columns["freshness_hours"] = (moments - published) / np.timedelta64(1, "h")
    return columns


def engagement_columns(history: pd.DataFrame, history_k: int) -> dict[str, np.ndarray]:
    """The mean dwell and scroll depth of the window's *past* clicks.

    NaN where the dataset ships no such column -- MIND's history is a bare id
    list -- and NaN for a user with no clicks, which is not a zero: a cold user
    does not have a dwell time of nothing, they have no dwell time.
    """
    found = {}
    for column, source in ENGAGEMENT.items():
        values = np.full(len(history), np.nan)
        if source in history:
            for position, row in enumerate(history[source]):
                if row is None:
                    continue
                window = np.asarray(row, dtype="float64")[-history_k:]
                window = window[~np.isnan(window)]
                if window.size:
                    values[position] = window.mean()
        found[column] = values
    return found


# ---------------------------------------------------------------------------
# The frame.


# What a submission's label column holds. The leaderboard is holding the
# outcomes back, so a submission frame's rows are neither clicked nor not
# clicked -- and a zero would say "not clicked", which is a claim about data
# nobody has. Nothing in scoring reads the column; it is in the frame because
# the frame has one schema.
UNKNOWN_LABEL = -1


def module_scorers(config: DatasetConfig, history_k: int) -> dict:
    """The offline scorers: each retriever over the feature store's own index.

    `frame_for` takes these as an argument so the submission can pass its own
    -- rankers built over the competition's catalogue -- and still go through
    one frame builder. A second builder for the submission would be the same
    forty columns written twice, and the failure it would produce is a model
    served features that are subtly not the ones it was trained on.
    """
    return {
        name: (
            lambda chunk, history, module=module: module.rank_candidates(
                config, chunk, history, history_k
            )
        )
        for name, module in SCORERS.items()
    }


def frame_for(
    config: DatasetConfig,
    chunk: pd.DataFrame,
    history: pd.DataFrame,
    loaded: Loaded,
    *,
    causal: bool = True,
    history_k: int = retrieval.HISTORY_K,
    cost: dict[str, float] | None = None,
    scorers: dict | None = None,
) -> pd.DataFrame:
    """One chunk of impressions as rows of (impression, candidate).

    `chunk` and `history` are positionally paired -- row i of one is row i of
    the other -- which is what `ingest.history_for` returns and what
    `cosine_columns` asserts before it uses either.

    A chunk with no `labels` column is a submission chunk: its rows carry
    `UNKNOWN_LABEL` rather than a zero.
    """
    cost = {} if cost is None else cost
    scorers = module_scorers(config, history_k) if scorers is None else scorers
    lengths = chunk["candidate_ids"].map(len).to_numpy()
    article_ids = np.concatenate(
        [np.asarray(ids, dtype=object) for ids in chunk["candidate_ids"]]
    )
    labels = (
        np.concatenate([np.asarray(values) for values in chunk["labels"]])
        if "labels" in chunk
        else np.full(len(article_ids), UNKNOWN_LABEL)
    )
    moments = np.repeat(chunk["impression_time"].to_numpy("datetime64[us]"), lengths)

    frame = pd.DataFrame(
        {
            "impression_id": np.repeat(
                chunk["impression_id"].to_numpy().astype(object), lengths
            ),
            "article_id": article_ids,
            "label": labels.astype("int8"),
        }
    )

    with measured(cost, "retrievers"):
        for name, scorer in scorers.items():
            frame[f"{name}_score"], frame[f"{name}_rank"] = read_back(
                scorer(chunk, history), chunk
            )
    frame["n_candidates"] = np.repeat(lengths.astype("float64"), lengths)

    with measured(cost, "categories"):
        for name, values in category_columns(
            chunk, history, loaded, article_ids, history_k
        ).items():
            frame[name] = values

    with measured(cost, "profiles"):
        frame["n_clicks"] = np.repeat(
            history["n_clicks"].to_numpy(dtype="float64"), lengths
        )
        for name, values in cosine_columns(
            chunk,
            history,
            loaded.embeddings,
            profiles_for(config, history_k),
            history_k,
        ).items():
            frame[name] = values
        for name, values in engagement_columns(history, history_k).items():
            frame[name] = np.repeat(values, lengths)

    with measured(cost, "counters"):
        for name, values in counter_columns(
            loaded, article_ids, moments, causal
        ).items():
            frame[name] = values

    with measured(cost, "sessions"):
        sessions = loaded.sessions.reindex(chunk["impression_id"].to_numpy())
        for name in ("session_impressions", "session_clicks"):
            frame[name] = np.repeat(sessions[name].to_numpy("float64"), lengths)

    check_columns(frame.columns)
    return frame[list(COLUMNS)]


# ---------------------------------------------------------------------------
# Storage: a row group per chunk, a projected read per arm.


def path_for(
    config: DatasetConfig,
    split: str,
    causal: bool = True,
    precision: str | None = None,
) -> Path:
    """Where one split's frame lives, named for what distinguishes its rows.

    The precision is in the name because ticket 06 trains on both files and
    compares them; a shared name would mean the second build silently replaced
    the first and the comparison would be a model against itself. The chunk and
    row-group sizes are *not* in the name, for the opposite reason: they change
    what the build costs and not one value in the file.
    """
    precision = precision or config.features.precision
    name = split if causal else f"{split}-leaky"
    return config.feature_store_dir / DIRECTORY / f"{name}-{precision}.parquet"


def schema_for(spec: FeatureSpec) -> pa.Schema:
    value = pa.float16() if spec.precision == "float16" else pa.float32()
    return pa.schema(
        [
            pa.field("impression_id", pa.string()),
            pa.field("article_id", pa.string()),
            pa.field("label", pa.int8()),
            *(pa.field(name, value) for name in FEATURES),
        ]
    )


def as_table(frame: pd.DataFrame, spec: FeatureSpec) -> pa.Table:
    """The chunk at the spec's precision, in the schema's column order.

    float16 halves the largest thing A2 writes, at three decimal digits of
    features whose own measurement error is larger than that. Whether it costs
    any AUC is ticket 06's question, answered by training at both precisions
    rather than by this docstring.
    """
    narrowed = frame[list(COLUMNS)].copy()
    for name in FEATURES:
        narrowed[name] = narrowed[name].astype(spec.precision)
    return pa.Table.from_pandas(narrowed, schema=schema_for(spec), preserve_index=False)


def chunks(behaviors: pd.DataFrame, spec: FeatureSpec):
    """The split in `impression_time` order, in chunks of the spec's size.

    In time order because that is the order the counters are read in and the
    order a submission streams in, so the frame's build path and the serving
    path touch the store the same way. `chunk_impressions=None` is the whole
    split at once, the row the sweep compares the others against.
    """
    ordered = behaviors.sort_values("impression_time", kind="stable").reset_index(
        drop=True
    )
    size = spec.chunk_impressions or max(len(ordered), 1)
    for start in range(0, len(ordered), size):
        yield ordered.iloc[start : start + size].reset_index(drop=True)


def read(
    config: DatasetConfig,
    split: str,
    columns: tuple[str, ...] | None = None,
    causal: bool = True,
    precision: str | None = None,
) -> pd.DataFrame:
    """The frame back, optionally projected onto one arm's columns.

    An ablation arm reads its tiers and the serving path reads what it can
    compute, and neither pays for what it drops: parquet keeps a column
    together, so a projection is fewer bytes off the disk rather than a filter
    after the fact. A forbidden column cannot be projected because it is not in
    the file; asked for by name it is refused here, rather than coming back as
    a frame that quietly lacks it.
    """
    path = path_for(config, split, causal, precision)
    if not path.exists():
        raise FeatureError(
            f"no feature frame at {path}. Build it with "
            f"`python -m pipeline.features --dataset {config.name} --split {split}`"
        )
    if columns is not None:
        unknown = [name for name in columns if name not in COLUMNS]
        if unknown:
            raise FeatureError(
                f"not feature-frame columns: {', '.join(unknown)}. "
                f"A forbidden column is not among them by construction: the "
                f"frame is written from {len(COLUMNS)} columns and no other."
            )
    projection = None if columns is None else list(dict.fromkeys(columns))
    return pq.read_table(path, columns=projection).to_pandas()


# ---------------------------------------------------------------------------
# The stage.


def variant_of(spec: FeatureSpec, causal: bool = True) -> str:
    """What the ledger calls this build: the two storage choices and the arm.

    The chunk size is not in the name. It changes what the build costs in RSS
    and in seconds, which are columns on the row; it changes no number in the
    frame, and a variant name implying otherwise would invite somebody to
    compare two chunk sizes' AUC.
    """
    return f"{spec.precision}-rg{spec.row_group // 1000}k" + ("" if causal else "-leaky")


def build(
    config: DatasetConfig,
    split: str,
    spec: FeatureSpec | None = None,
    *,
    causal: bool = True,
    history_k: int = retrieval.HISTORY_K,
) -> dict:
    """Materialise one split's frame, a row group per chunk, and report it.

    The whole frame is never in memory: a chunk is built, written and dropped,
    which is what makes peak RSS a function of the chunk size rather than of
    the split's length -- the measurement the sweep over `chunk_impressions`
    exists to take.
    """
    spec = spec or config.features
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == split]
    if impressions.empty:
        raise FeatureError(f"{config.name} has no {split} impressions")

    loaded = load(config, behaviors)
    path = path_for(config, split, causal, spec.precision)
    path.parent.mkdir(parents=True, exist_ok=True)

    cost: dict[str, float] = {}
    rows = 0
    with timings.sample() as measured:
        writer = pq.ParquetWriter(path, schema_for(spec))
        try:
            for chunk in chunks(impressions, spec):
                history = ingest.history_for(config, chunk)
                frame = frame_for(
                    config,
                    chunk,
                    history,
                    loaded,
                    causal=causal,
                    history_k=history_k,
                    cost=cost,
                )
                writer.write_table(as_table(frame, spec), row_group_size=spec.row_group)
                rows += len(frame)
        finally:
            writer.close()

    candidates = int(impressions["candidate_ids"].map(len).sum())
    if rows != candidates:
        raise FeatureError(
            f"{rows:,} rows written for {candidates:,} candidates: the frame "
            f"is not one row per (impression, candidate)"
        )
    return {
        "dataset": config.name,
        "split": split,
        "causal": causal,
        "variant": variant_of(spec, causal),
        "rows": rows,
        "impressions": len(impressions),
        "bytes": path.stat().st_size,
        "seconds": measured["seconds"],
        "peak_rss_mb": measured["peak_rss_mb"],
        "families": cost,
        "path": path,
    }


def record(report: dict, spec: FeatureSpec) -> dict:
    """The build as one ledger row. Functional columns stay blank on purpose:
    a feature frame has no AUC of its own, and ticket 06's rows are where the
    precision and the windows get their quality side."""
    families = ", ".join(
        f"{family} {seconds:.1f} s"
        for family, seconds in sorted(
            report["families"].items(), key=lambda item: -item[1]
        )
    )
    chunk = spec.chunk_impressions or report["impressions"]
    return ledger.record(
        {
            "dataset": report["dataset"],
            "stage": STAGE,
            "variant": report["variant"],
            "split": report["split"],
            "feature_bytes": report["bytes"],
            "train_seconds": round(report["seconds"], 2),
            "peak_rss_mb": round(report["peak_rss_mb"], 1),
            "rows_per_s": report["rows"] / max(report["seconds"], 1e-9),
            "note": (
                f"{report['rows']:,} rows x {len(FEATURES)} features over "
                f"{report['impressions']:,} impressions in chunks of "
                f"{chunk:,}; {families}"
            ),
        }
    )


def run(config: DatasetConfig, force: bool = False) -> None:
    spec = config.features
    for split in SPLITS:
        path = path_for(config, split, precision=spec.precision)
        if not force and path.exists():
            print(f"    {split} frame is already built at {path}")
            continue
        report = build(config, split, spec)
        record(report, spec)
        print(
            f"    {split:<11} {report['rows']:>10,} rows  "
            f"{report['bytes'] / (1 << 20):>7.1f} MB  "
            f"{report['seconds']:>6.1f} s  peak {report['peak_rss_mb']:,.0f} MB"
        )
    ledger.render()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.features",
        description="Build the (impression, candidate) feature frame for a split.",
    )
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--split", action="append", help=f"default: {', '.join(SPLITS)}")
    parser.add_argument(
        "--leaky",
        action="store_true",
        help="Q9's arm: read the counters over the whole log rather than "
        "strictly before the impression, into its own file",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        help="impressions per chunk, overriding the registry; 0 is the whole split",
    )
    parser.add_argument("--precision", choices=("float32", "float16"))
    parser.add_argument("--row-group", type=int)
    args = parser.parse_args(argv)

    for name in args.dataset or DEFAULT_DATASETS:
        config = DATASETS[name]
        spec = config.features
        if args.chunk is not None:
            spec = dataclasses.replace(spec, chunk_impressions=args.chunk or None)
        if args.precision:
            spec = dataclasses.replace(spec, precision=args.precision)
        if args.row_group:
            spec = dataclasses.replace(spec, row_group=args.row_group)
        for split in args.split or SPLITS:
            report = build(config, split, spec, causal=not args.leaky)
            row = record(report, spec)
            print(
                f"  {name}/{split} [{row['variant']}]: {report['rows']:,} rows, "
                f"{report['bytes'] / (1 << 20):.1f} MB, {report['seconds']:.1f} s, "
                f"peak {report['peak_rss_mb']:,.0f} MB"
            )
    print(f"-> {ledger.render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
