"""Causal per-article exposure and click counters: fitted over the whole log,
read strictly before a moment.

A popularity feature is the cheapest signal a click log offers and the easiest
to get wrong. Counted over the whole log it tells the model how often an
article *will* be clicked, including in the impression being scored; counted
only over what happened before the impression it tells the model what a
server could have known. The two differ by exactly the leakage Q9 asks about,
so both must come from one object and differ only in the moment asked for --
a second implementation of the leaky arm would be a second thing to get wrong.

The store follows the pattern the BM25 index set: global structure built once,
causal read at query time. Every exposure is one `int64` key, article position
times the log's span plus the microsecond offset of its impression, kept
sorted; the number of exposures strictly before `t` is then one binary search,
and a sliding window is the difference of two. Clicks are the same keys over
the labelled subset. `t = +inf` reads to the end of the article's slot and is
the whole-log answer.

"Strictly before" is the whole point. An impression at exactly `t` is not
counted at `t`: the impression being scored does not see its own outcome, and
two impressions served in the same second do not see each other's.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from pipeline import ledger, timings
from pipeline.datasets import DatasetConfig

# The moment past the end of the log: `at(ids, WHOLE_LOG)` is the leaky arm.
WHOLE_LOG = math.inf

# The windows tried. `None` counts everything since the log began; a duration
# counts the interval [t - window, t). Which one the re-ranker uses is chosen
# on tune when it exists; this module measures what each costs to read.
WINDOWS: dict[str, pd.Timedelta | None] = {
    "cumulative": None,
    "1h": pd.Timedelta(hours=1),
    "3h": pd.Timedelta(hours=3),
    "24h": pd.Timedelta(hours=24),
}

STAGE = "counters"
DIRECTORY = "counters"
STORE = "store.npz"
LATENCY_SAMPLE = 500
BENCH_SPLIT = "tune"


class CounterError(RuntimeError):
    pass


class Counts(NamedTuple):
    exposures: np.ndarray
    clicks: np.ndarray
    # NaN where the article has not been exposed: a rate over nothing is not
    # zero, and the re-ranker can tell the two apart.
    ctr: np.ndarray


@dataclass(frozen=True)
class CounterStore:
    article_ids: np.ndarray  # sorted, unique
    origin: np.datetime64  # the log's first impression, at microseconds
    span: int  # one more than any offset a key can carry; the slot per article
    exposures: np.ndarray  # sorted int64 keys, one per (article, impression)
    clicks: np.ndarray  # the same, over exposures that were clicked

    @property
    def nbytes(self) -> int:
        return int(
            self.article_ids.nbytes + self.exposures.nbytes + self.clicks.nbytes
        )

    def _positions(self, article_ids) -> tuple[np.ndarray, np.ndarray]:
        """Corpus position per id, and whether the id is in the log at all."""
        ids = np.asarray(article_ids, dtype=self.article_ids.dtype)
        pos = np.searchsorted(self.article_ids, ids)
        pos = np.minimum(pos, len(self.article_ids) - 1)
        known = self.article_ids[pos] == ids
        return pos, known

    def _offsets(self, t, size: int) -> np.ndarray:
        """`t` as microseconds since the origin, one per id, unclamped."""
        moments = pd.to_datetime(pd.Series(t) if np.ndim(t) else pd.Series([t]))
        offsets = (
            moments.to_numpy("datetime64[us]").astype(np.int64)
            - self.origin.astype("datetime64[us]").astype(np.int64)
        )
        if offsets.size == 1 and size != 1:
            offsets = np.full(size, offsets[0], dtype=np.int64)
        if offsets.size != size:
            raise CounterError(f"{size} article ids but {offsets.size} moments")
        return offsets

    def _bounds(self, t, window, size: int) -> tuple[np.ndarray, np.ndarray]:
        """The half-open [lower, upper) each id is counted over, as offsets
        clamped into the slot: before the log began reads as 0 -- nothing
        happened before it -- and after it as `span - 1`, which every
        recorded offset is strictly less than. The window is taken from the
        real `t` before clamping, so a moment past the log's end still reads
        the last hour, not the whole log.
        """
        last = self.span - 1
        if np.isscalar(t) and isinstance(t, float) and math.isinf(t):
            if window is not None:
                raise CounterError("the whole-log read is cumulative by definition")
            return np.zeros(size, dtype=np.int64), np.full(size, last, dtype=np.int64)
        upper = self._offsets(t, size)
        if window is None:
            lower = np.zeros(size, dtype=np.int64)
        else:
            lower = upper - int(window / pd.Timedelta(microseconds=1))
        return np.clip(lower, 0, last), np.clip(upper, 0, last)

    def _before(self, keys: np.ndarray, pos: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """How many of `keys` fall in each article's slot strictly before `offsets`."""
        return np.searchsorted(keys, pos * self.span + offsets, side="left")

    def at(self, article_ids, t, window: pd.Timedelta | None = None) -> Counts:
        """Exposures, clicks and CTR of each article strictly before `t`.

        `t` is one moment for every id or one per id; `WHOLE_LOG` reads the
        entire log. `window` restricts the count to [t - window, t). An id
        the log never showed has 0 exposures, 0 clicks and a NaN rate.
        """
        pos, known = self._positions(article_ids)
        lower, upper = self._bounds(t, window, len(pos))
        counts = []
        for keys in (self.exposures, self.clicks):
            count = self._before(keys, pos, upper) - self._before(keys, pos, lower)
            counts.append(np.where(known, count, 0).astype(np.int64))
        exposures, clicks = counts
        with np.errstate(divide="ignore", invalid="ignore"):
            ctr = np.where(exposures > 0, clicks / np.maximum(exposures, 1), np.nan)
        return Counts(exposures, clicks, ctr)

    def first_seen(self, article_ids, t) -> np.ndarray:
        """The earliest appearance of each article strictly before `t`, or
        NaT: for an article the log had not shown by `t`, there is no
        freshness to compute, and a time at or after `t` would be one the
        server could not have had."""
        pos, known = self._positions(article_ids)
        _, upper = self._bounds(t, None, len(pos))
        first = self._before(self.exposures, pos, np.zeros_like(upper))
        first = np.minimum(first, len(self.exposures) - 1)
        offset = self.exposures[first] - pos * self.span
        seen = known & (self.exposures[first] // self.span == pos) & (offset < upper)
        moments = self.origin.astype("datetime64[us]") + offset.astype("timedelta64[us]")
        return np.where(seen, moments, np.datetime64("NaT", "us"))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / STORE
        np.savez(
            path,
            article_ids=self.article_ids,
            origin=np.asarray(self.origin.astype("datetime64[us]")),
            span=np.asarray(self.span, dtype=np.int64),
            exposures=self.exposures,
            clicks=self.clicks,
        )
        return path


def load(directory: Path) -> CounterStore:
    with np.load(directory / STORE, allow_pickle=False) as saved:
        return CounterStore(
            article_ids=saved["article_ids"],
            origin=saved["origin"][()],
            span=int(saved["span"]),
            exposures=saved["exposures"],
            clicks=saved["clicks"],
        )


def build(behaviors: pd.DataFrame) -> CounterStore:
    """One store from a behaviours frame: `impression_time`, `candidate_ids`,
    `labels`. Every candidate shown is an exposure; every candidate labelled
    1 is a click."""
    if behaviors.empty:
        raise CounterError("a counter store needs at least one impression")
    lengths = behaviors["candidate_ids"].map(len).to_numpy()
    shown = np.concatenate(behaviors["candidate_ids"].to_numpy()).astype(str)
    clicked = np.concatenate(behaviors["labels"].to_numpy()).astype(bool)
    if len(clicked) != len(shown):
        raise CounterError("candidate_ids and labels differ in length")

    moments = behaviors["impression_time"].to_numpy("datetime64[us]")
    origin = moments.min()
    offsets = (moments - origin).astype(np.int64)
    span = int(offsets.max()) + 2
    article_ids, position = np.unique(shown, return_inverse=True)
    if len(article_ids) * span >= np.iinfo(np.int64).max:
        raise CounterError("the log spans too long for one int64 key per exposure")

    keys = position.astype(np.int64) * span + np.repeat(offsets, lengths)
    return CounterStore(
        article_ids=article_ids,
        origin=origin,
        span=span,
        exposures=np.sort(keys),
        clicks=np.sort(keys[clicked]),
    )


@dataclass(frozen=True)
class ServingCounters:
    """What a server knows at test time, behind the same two methods.

    The submission ranks a *later* period than the feature store covers, so its
    counters come from three places, and which three is the whole "what a live
    server knows" line of Q4:

    `trained` -- the whole training-period log, exposures and clicks. It is
    entirely before the test period, so it is read at the same `t` as
    everything else and simply answers with all of itself.

    `shown` -- the test log's own impressions, exposures only, read **strictly
    before `t`** exactly as the harness reads them. Fitted once over the whole
    test file and read causally rather than accumulated chunk by chunk: those
    two are the same answer -- `test_removing_every_row_after_t_leaves_the_
    causal_read_unchanged` is the proof -- and one of them is a single pass
    instead of a re-sort per chunk.

    `clicked` -- how often each article appears in a test user's click history.
    A click a server knows happened, but with no timestamp on it (MIND's
    histories carry none), so it counts toward the cumulative counter and
    toward no window. Said here because the alternative is a window that
    quietly includes clicks from an unknown time.

    The test log has no labels -- that is what the leaderboard is holding back
    -- so no click of the test period itself is ever counted. A server would
    have them; we do not, and the design note says so rather than the code
    pretending otherwise.
    """

    trained: CounterStore
    shown: CounterStore | None
    clicked_ids: np.ndarray  # sorted, unique
    clicked_counts: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(
            self.trained.nbytes
            + (self.shown.nbytes if self.shown is not None else 0)
            + self.clicked_ids.nbytes
            + self.clicked_counts.nbytes
        )

    def _from_histories(self, article_ids) -> np.ndarray:
        ids = np.asarray(article_ids, dtype=self.clicked_ids.dtype)
        if not len(self.clicked_ids):
            return np.zeros(len(ids), dtype=np.int64)
        position = np.searchsorted(self.clicked_ids, ids)
        position = np.minimum(position, len(self.clicked_ids) - 1)
        known = self.clicked_ids[position] == ids
        return np.where(known, self.clicked_counts[position], 0).astype(np.int64)

    def at(self, article_ids, t, window: pd.Timedelta | None = None) -> Counts:
        """The same signature the harness's store answers, so the feature code
        that reads it does not know which period it is serving."""
        counted = self.trained.at(article_ids, t, window)
        exposures, clicks = counted.exposures, counted.clicks
        if self.shown is not None:
            exposures = exposures + self.shown.at(article_ids, t, window).exposures
        if window is None:
            clicks = clicks + self._from_histories(article_ids)
        with np.errstate(divide="ignore", invalid="ignore"):
            ctr = np.where(exposures > 0, clicks / np.maximum(exposures, 1), np.nan)
        return Counts(exposures, clicks, ctr)

    def first_seen(self, article_ids, t) -> np.ndarray:
        """The earlier of the two logs' first sightings, strictly before `t`.

        An article the training period showed is older than the test period can
        make it look, and an article only the test period has seen is as old as
        its first test impression -- so the answer is the earlier of the two,
        which is what freshness means whichever log it came from.
        """
        found = self.trained.first_seen(article_ids, t)
        if self.shown is None:
            return found
        later = self.shown.first_seen(article_ids, t)
        return np.where(pd.isna(found), later, np.where(pd.isna(later), found, np.minimum(found, later)))


def clicks_in_histories(histories) -> tuple[np.ndarray, np.ndarray]:
    """How often each article appears in a test user's click history.

    One count per article over every history the competition ships, which is
    the click side of what a server has at test time. Returned sorted so the
    lookup above is a binary search rather than a dict of a million strings.
    """
    flat = [click for clicks in histories for click in clicks]
    if not flat:
        return np.empty(0, dtype=object), np.empty(0, dtype=np.int64)
    ids, counts = np.unique(np.asarray(flat, dtype=str), return_counts=True)
    return ids.astype(object), counts.astype(np.int64)


def exposures_only(impressions: pd.DataFrame) -> CounterStore:
    """A store over impressions whose outcomes are not known.

    The test file carries candidates and no labels, so every exposure is real
    and no click is: `build` is handed all-zero labels rather than a second
    constructor, because a second one is a second thing that could disagree
    about what an exposure is.
    """
    frame = impressions[["impression_time", "candidate_ids"]].copy()
    frame["labels"] = [
        np.zeros(len(candidates), dtype=bool) for candidates in frame["candidate_ids"]
    ]
    return build(frame)


def bench_lookups(store: CounterStore, impressions: pd.DataFrame, window) -> dict:
    """What a read costs: per-impression latency over a sample, and bulk
    throughput over every candidate row of `impressions` in one call."""
    from pipeline import bench

    sample = impressions.head(LATENCY_SAMPLE)
    # The first window measured would otherwise pay for faulting the store
    # into memory, and look slower than the ones that follow it.
    for candidates, moment in zip(sample["candidate_ids"], sample["impression_time"]):
        store.at(candidates, moment, window)
    seconds = []
    for candidates, moment in zip(sample["candidate_ids"], sample["impression_time"]):
        started = time.perf_counter()
        store.at(candidates, moment, window)
        seconds.append(time.perf_counter() - started)

    lengths = impressions["candidate_ids"].map(len).to_numpy()
    shown = np.concatenate(impressions["candidate_ids"].to_numpy())
    moments = np.repeat(impressions["impression_time"].to_numpy("datetime64[us]"), lengths)
    started = time.perf_counter()
    store.at(shown, moments, window)
    elapsed = time.perf_counter() - started
    return {**bench.percentiles(seconds), "rows_per_s": len(shown) / elapsed}


def run(config: DatasetConfig, force: bool = False) -> None:
    """Build the store over the whole log, save it, and record what each
    window costs to read. The functional side of these rows is the
    re-ranker's to fill: a counter has no AUC of its own."""
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    directory = config.artifacts_dir / DIRECTORY

    with timings.sample() as cost:
        store = build(behaviors)
    path = store.save(directory)
    store_bytes = path.stat().st_size
    print(
        f"    {len(store.exposures):,} exposures and {len(store.clicks):,} clicks "
        f"over {len(store.article_ids):,} articles built in {cost['seconds']:.1f} s, "
        f"{store_bytes / (1 << 20):.1f} MB on disk"
    )

    impressions = behaviors[behaviors["split"] == BENCH_SPLIT]
    if impressions.empty:
        raise CounterError(f"no {BENCH_SPLIT} impressions to bench lookups on")
    for name, window in WINDOWS.items():
        read = bench_lookups(store, impressions, window)
        ledger.record(
            {
                "dataset": config.name,
                "stage": STAGE,
                "variant": name,
                "split": BENCH_SPLIT,
                "index_bytes": store_bytes,
                "train_seconds": round(cost["seconds"], 2),
                "peak_rss_mb": round(cost["peak_rss_mb"], 1),
                "p50_ms": read["p50_ms"],
                "p99_ms": read["p99_ms"],
                "rows_per_s": read["rows_per_s"],
                "note": (
                    f"store over the whole log ({len(behaviors):,} impressions); "
                    "one store serves every window, so bytes are shared"
                ),
            }
        )
        print(
            f"    {name:<10} lookup p50 {read['p50_ms']:.3f} ms  p99 {read['p99_ms']:.3f} ms  "
            f"bulk {read['rows_per_s']:,.0f} rows/s"
        )
    ledger.render()
