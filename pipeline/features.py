"""Per-candidate features, and the line between what a server has and what it does not.

The two retrievers this joins score a candidate on content alone — how much it
reads like the articles a user has clicked. That is a real signal and on MIND it
is most of one, but a click on a news site is mostly a fact about *what is on the
front page right now*: on EB-NeRD, content similarity ranks at chance while an
article's click-through rate over the previous three hours ranks at 0.72. A
retriever with no behavioural feature is not a weak recommender, it is a
recommender that has been told nothing about the thing being recommended.

So this module builds one row per (impression, candidate) with three kinds of
column on it, and keeps the three kinds apart on purpose, because the assignment
asks for metrics with and without features unavailable at serving time:

    CONTENT     what the two retrievers already compute, plus the article and
                user metadata around it. Available anywhere.
    EXPOSURE    how often the article has been *shown* recently. Read off the
                candidate lists themselves, so it needs no labels and is
                available on a competition test file, which has none.
    CLICKED     how often it was *clicked* when shown. Needs labels for the
                impressions before this one.

CLICKED is available to a live server — yesterday's clicks are not the future —
but it is not available on a CodaBench test file, which ships candidates and no
outcomes. That is the split the two feature sets exist to make measurable rather
than arguable: `fusion` trains one model on all three and one on the first two,
and the evaluation reports both.

Every count is strictly past-only. A counter is queried at the impression's own
timestamp with `allow_exact_matches=False`, so an impression can never see its
own outcome, nor another impression's from the same instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from pipeline import bm25_index, embed, retrieval
from pipeline.datasets import DatasetConfig

# Time windows the popularity counters are kept over, in hours. News decays
# fast enough that an all-time count and a three-hour count disagree, and the
# disagreement is itself the signal — an article on the way up and one on the
# way down have the same total.
WINDOWS = (3, 24)

# Laplace weight on the click-through rate. An article shown twice and clicked
# once is not a 50% article; this pulls it back towards the corpus rate by the
# weight of ten pseudo-impressions.
SMOOTHING = 10.0

# How finely the popularity counters keep time. Events are bucketed to this,
# and an impression sees every bucket that closed strictly before its own — so
# it can be as much as one bucket stale and can never be one event early.
#
# Bucketing rather than counting per event is what lets the same counter serve
# a 100k-impression validation split and a 13.5M-impression competition file:
# the table it holds is (article x bucket) rather than (candidate row), which
# on EB-NeRD's test set is 24M entries at worst instead of 206M. Five
# minutes because the strongest feature here reads three hours back and an
# hourly bucket throws away a third of that window's freshness. It shows up in
# the number: on EB-NeRD's validation split the same model scores 0.7602 at
# five minutes, 0.7558 at fifteen and 0.7411 at an hour.
RESOLUTION = np.timedelta64(5, "m")

# How far back the category profile of a user is read. Their whole history
# would let a year-old reading habit outvote this week's.
PROFILE_CLICKS = 50

# Candidate rows whose history is gathered at once in the similarity pass.
# See `_semantic_scores`: the gather is the largest allocation anywhere here.
SIMILARITY_BLOCK = 20_000

# Hours assumed for an article with no publication timestamp. MIND ships none
# at all, so on that dataset `fresh` is a constant and the model drops it.
UNKNOWN_AGE_HOURS = 48.0

CONTENT = (
    "bm25",
    "sem",
    "sem_max",
    "sem_last",
    "cat_aff",
    "sub_aff",
    "repeat",
    "fresh",
)
EXPOSURE = ("expo_all", *(f"expo_{h}h" for h in WINDOWS))
CLICKED = ("ctr_all", *(f"ctr_{h}h" for h in WINDOWS))

# What a model may look at. `SERVING` is the subset a competition test file can
# supply; `ALL` adds the counters that need labels.
SERVING = (*CONTENT, *EXPOSURE)
ALL = (*SERVING, *CLICKED)


def explode(behaviors: pd.DataFrame, labelled: bool = True) -> pd.DataFrame:
    """One row per candidate: the shape everything below reads and writes.

    Built by repeating each impression's scalars across its candidate list
    rather than by `DataFrame.explode`, which is an order of magnitude slower
    on the three million candidate rows EB-NeRD's train split holds.
    """
    widths = behaviors["candidate_ids"].str.len().to_numpy()
    frame = pd.DataFrame(
        {
            # The impression's *row position*, not its id. EB-NeRD's test file
            # stamps all 200,000 beyond-accuracy impressions with id "0" — the
            # registry records it as `repeated_impression_id` — so an id does
            # not identify an impression there, and every `_runs` and groupby
            # below would fold those 200,000 into one. A position is a row key
            # on every file by construction. Everything keyed off this reads
            # `behaviors` and `history` positionally too, which they are: both
            # arrive one row per impression, in the same order.
            "imp": np.repeat(np.arange(len(behaviors), dtype="int64"), widths),
            "aid": np.concatenate(behaviors["candidate_ids"].to_numpy()),
            "t": np.repeat(behaviors["impression_time"].to_numpy(), widths),
        }
    )
    if labelled:
        frame["y"] = np.concatenate(behaviors["labels"].to_numpy()).astype("int8")
    return frame


@dataclass(frozen=True)
class Popularity:
    """Cumulative shows and clicks per article, queryable at any timestamp.

    Held as one step function per article rather than as a total, because the
    question a feature asks is "how had this article done *by the time this
    impression happened*" and the answer has to differ between two impressions
    of the same article a day apart. A total would answer with the same number
    for both, and on the training split that number would include the outcome
    of the row being scored — which is how a model learns to predict a click
    from the click.

    `clicks` is None when the counter was fitted on candidate lists with no
    labels attached, which is what a competition test file is. Then only the
    exposure columns come back, and `columns` says so.
    """

    times: np.ndarray  # bucket start, sorted; one row per (article, bucket)
    ids: np.ndarray  # the article each row belongs to
    shows: np.ndarray  # cumulative shows of that article through that bucket
    clicks: np.ndarray | None  # cumulative clicks; None when fitted unlabelled
    prior: float  # corpus click-through rate, what smoothing pulls towards
    resolution: np.timedelta64 = RESOLUTION

    @property
    def columns(self) -> tuple[str, ...]:
        return EXPOSURE if self.clicks is None else (*EXPOSURE, *CLICKED)

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, resolution: np.timedelta64 = RESOLUTION
    ) -> "Popularity":
        """Bucket an exploded frame by (article, time) and accumulate.

        A frame with no `y` column fits an exposure-only counter — which is not
        a degraded one but the only kind a competition test file can produce,
        and the reason the serving variant exists.
        """
        labelled = "y" in frame.columns
        counts = pd.DataFrame(
            {
                "aid": frame["aid"].to_numpy(),
                "t": _floor(frame["t"].to_numpy(), resolution),
                "shows": 1,
                **({"clicks": frame["y"].to_numpy()} if labelled else {}),
            }
        ).groupby(["aid", "t"], sort=True, as_index=False).sum()
        return cls.from_counts(counts, labelled, resolution)

    @classmethod
    def from_counts(
        cls,
        counts: pd.DataFrame,
        labelled: bool,
        resolution: np.timedelta64 = RESOLUTION,
    ) -> "Popularity":
        """Accumulate an already-bucketed (aid, t, shows[, clicks]) table.

        The entry point for the submission path, which aggregates its buckets a
        chunk at a time while streaming a file too large to explode whole.
        """
        counts = counts.sort_values(["aid", "t"], kind="stable")
        grouped = counts.groupby("aid", sort=False)
        shows = grouped["shows"].cumsum().to_numpy()
        clicks = grouped["clicks"].cumsum().to_numpy() if labelled else None
        return cls(
            times=counts["t"].to_numpy(),
            ids=counts["aid"].to_numpy(),
            shows=shows,
            clicks=clicks,
            prior=float(clicks[-1] / shows[-1]) if labelled and len(shows) else 0.0,
            resolution=resolution,
        )

    def _at(self, ids: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Counts for each (article, time), over everything strictly earlier.

        Both sides are floored to the counter's resolution and matched with
        `allow_exact_matches=False`, so an impression sees every bucket that
        closed before its own began and never its own. That is what makes
        "strictly earlier" true rather than approximately true: the impression's
        own outcome, and every other impression served alongside it, sit inside
        the bucket this excludes.
        """
        query = pd.DataFrame(
            {"aid": ids, "t": _floor(times, self.resolution)}
        ).reset_index()
        table = pd.DataFrame(
            {
                "aid": self.ids,
                "t": self.times,
                "_shows": self.shows,
                **({"_clicks": self.clicks} if self.clicks is not None else {}),
            }
        )
        found = pd.merge_asof(
            query.sort_values("t", kind="stable"),
            table.sort_values("t", kind="stable"),
            on="t",
            by="aid",
            direction="backward",
            allow_exact_matches=False,
        ).sort_values("index")
        shows = found["_shows"].fillna(0).to_numpy()
        clicks = (
            found["_clicks"].fillna(0).to_numpy()
            if self.clicks is not None
            else np.zeros(len(shows))
        )
        return shows, clicks

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add this counter's columns to an exploded frame, in place."""
        ids, times = frame["aid"].to_numpy(), frame["t"].to_numpy()
        shows, clicks = self._at(ids, times)
        self._write(frame, "all", shows, clicks)
        for hours in WINDOWS:
            before = times - np.timedelta64(hours, "h")
            older_shows, older_clicks = self._at(ids, before)
            self._write(
                frame,
                f"{hours}h",
                np.maximum(shows - older_shows, 0),
                np.maximum(clicks - older_clicks, 0),
            )
        return frame

    def _write(
        self, frame: pd.DataFrame, suffix: str, shows: np.ndarray, clicks: np.ndarray
    ) -> None:
        # log1p rather than the raw count: the difference between an article
        # shown 10 times and one shown 100 is the difference that matters, and
        # between 10,000 and 10,100 there is none.
        frame[f"expo_{suffix}"] = np.log1p(shows)
        if self.clicks is not None:
            frame[f"ctr_{suffix}"] = (clicks + self.prior * SMOOTHING) / (
                shows + SMOOTHING
            )


@dataclass(frozen=True)
class Content:
    """The content side: both retrievers' scores, and the metadata around them.

    Holds the indices rather than rebuilding them per call, because the sweep
    scores the same catalogue at four history windows and a BM25 index costs
    more to build than to query.
    """

    config: DatasetConfig
    articles: pd.DataFrame
    lexical: bm25_index.Index
    embeddings: embed.Embeddings

    @cached_property
    def article_row(self) -> dict[str, int]:
        """article id -> catalogue position. One integer key space for the
        `repeat` lookup, built once because the submission path asks for it
        once per chunk over a 121k-article catalogue."""
        return {a: i for i, a in enumerate(self.articles["article_id"])}

    @cached_property
    def taxonomy(self) -> dict[str, tuple[dict[str, int], int]]:
        """Per taxonomy column, its value codes and each article's code."""
        out = {}
        for column in ("category", "subcategory"):
            values = self.articles[column].astype("string").fillna("")
            code_of = {value: code for code, value in enumerate(pd.Index(values).unique())}
            out[column] = (
                dict(zip(self.articles["article_id"], values.map(code_of))),
                len(code_of),
            )
        return out

    @cached_property
    def published(self) -> dict[str, pd.Timestamp]:
        if "published_time" not in self.articles:
            return {}
        return dict(zip(self.articles["article_id"], self.articles["published_time"]))

    @classmethod
    def load(cls, config: DatasetConfig, articles: pd.DataFrame) -> "Content":
        return cls(
            config=config,
            articles=articles,
            lexical=bm25_index.load(config.artifacts_dir / "bm25"),
            embeddings=embed.load(config),
        )

    def attach(
        self,
        frame: pd.DataFrame,
        behaviors: pd.DataFrame,
        history: pd.DataFrame,
        history_k: int,
    ) -> pd.DataFrame:
        """Add the content columns for every candidate row in `frame`."""
        clicks_of = history["click_history"].to_numpy()
        frame["bm25"] = self._lexical_scores(behaviors, history, history_k, frame)
        for name, values in self._semantic_scores(frame, clicks_of, history_k).items():
            frame[name] = values
        for name, values in self._profile(frame, clicks_of, history_k).items():
            frame[name] = values
        frame["fresh"] = self._freshness(frame)
        return frame

    def _lexical_scores(
        self,
        behaviors: pd.DataFrame,
        history: pd.DataFrame,
        history_k: int,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """BM25 of each candidate against the user's history query.

        Through `score_pairs` rather than `score_candidates`: the two compute
        the same number, but the second one scores the whole catalogue to keep
        fifteen of it, which at competition scale is the difference between a
        submission that builds and one that does not.
        """
        queries, _ = bm25_index.build_queries(
            history, self.articles, self.config, history_k
        )
        # Indexed by row position, like the frame's `imp`. Keyed by impression
        # id these would collapse every impression sharing one — which on
        # EB-NeRD's test file is 200,000 of them, and produced a column 419,481
        # long for a frame of 16,241,731 rows rather than a wrong number.
        query_of = queries["query"].to_numpy()
        candidates_of = behaviors["candidate_ids"].to_numpy()
        # Driven off the frame's own impression order rather than the query
        # frame's. `score_pairs` returns one flat array in the order it was
        # asked, and the history frame need not arrive in the behaviours'
        # order — asking in the wrong one would score each impression against
        # another's candidates and produce a perfectly well-formed column.
        impressions, _ = _runs(frame["imp"].to_numpy())
        return self.lexical.score_pairs(
            [
                query_of[i].split()
                if pd.notna(query_of[i]) and query_of[i]
                else []
                for i in impressions
            ],
            [candidates_of[i] for i in impressions],
        )

    def _semantic_scores(
        self, frame: pd.DataFrame, clicks_of: np.ndarray, history_k: int
    ) -> dict[str, np.ndarray]:
        """Cosine of each candidate against the user's history, three ways.

        The mean-pooled vector both `ann` and the SPEC use is one summary of a
        reading history and a lossy one: a user who reads football and recipes
        mean-pools to someone who reads neither. `sem_max` asks whether the
        candidate looks like *any* recent click and `sem_last` whether it
        follows on from the most recent one, which is the session signal.

        The history is padded to a fixed `history_k` slots and the whole thing
        computed one slot at a time, so the cost is `history_k` dot products
        over the frame rather than one small matrix multiply per impression.
        On EB-NeRD's competition file the difference is hours. Absent slots get
        a zero vector, which scores zero against every candidate — right for
        the mean (a shorter history is averaged over its real length, below)
        and right for the max as long as no real cosine is negative, which for
        these two encoders it is not.
        """
        row_of = self.embeddings.index
        vectors = self.embeddings.vectors

        rows = np.fromiter(
            (row_of.get(a, -1) for a in frame["aid"]), dtype="int64", count=len(frame)
        )
        known = rows >= 0

        # Both sides are held as *row numbers* into the catalogue rather than
        # as the vectors themselves, and gathered a block at a time below.
        # Held whole they are what makes this the peak of the whole pipeline:
        # EB-NeRD's competition chunk is 1.5M candidate rows over 100k
        # impressions, which at 768 float32 is 4.7 GB of candidate and 3.1 GB
        # of history before a dot product is taken. As indices the same two are
        # 12 MB and 4 MB.
        impressions, widths = _runs(frame["imp"].to_numpy())
        history = np.zeros((len(impressions), history_k), dtype="int64")
        # Which slots hold a real click. Absent ones gather catalogue row 0 and
        # are zeroed after the dot product, which costs (rows x history_k)
        # rather than the (rows x history_k x width) that zeroing the gathered
        # vectors would.
        filled = np.zeros((len(impressions), history_k), dtype="float32")
        length = np.zeros(len(impressions), dtype="float32")
        for i, impression in enumerate(impressions):
            clicked = [
                row_of[c] for c in clicks_of[impression][-history_k:] if c in row_of
            ]
            if not clicked:
                continue
            # Right-aligned, so slot -1 is always the most recent click
            # whatever the history's length.
            history[i, history_k - len(clicked):] = clicked
            filled[i, history_k - len(clicked):] = 1.0
            length[i] = len(clicked)

        of_impression = np.repeat(np.arange(len(impressions)), widths)
        total = np.zeros(len(frame), dtype="float32")
        best = np.zeros(len(frame), dtype="float32")
        last = np.zeros(len(frame), dtype="float32")
        # In row blocks, because the gathered history of a block is
        # (rows x history_k x width) and a competition chunk is four million
        # rows — which at 384 float32 and ten slots is 60 GB gathered whole and
        # 300 MB gathered like this.
        for start in range(0, len(frame), SIMILARITY_BLOCK):
            block = slice(start, min(start + SIMILARITY_BLOCK, len(frame)))
            of_block = of_impression[block]
            candidate = np.zeros((block.stop - block.start, vectors.shape[1]), dtype="float32")
            here = known[block]
            candidate[here] = vectors[rows[block][here]]
            similarity = np.einsum(
                "rd,rkd->rk", candidate, vectors[history[of_block]]
            )
            similarity *= filled[of_block]
            total[block] = similarity.sum(axis=1)
            best[block] = similarity.max(axis=1)
            last[block] = similarity[:, -1]

        empty = length[of_impression] == 0
        mean = np.divide(
            total, np.maximum(length[of_impression], 1.0), dtype="float32"
        )
        best = np.where(empty | ~known, 0.0, best).astype("float32")
        return {
            "sem": np.where(empty | ~known, 0.0, mean).astype("float32"),
            "sem_max": best,
            "sem_last": np.where(empty | ~known, 0.0, last).astype("float32"),
        }

    def _profile(
        self, frame: pd.DataFrame, clicks_of: np.ndarray, history_k: int
    ) -> dict[str, np.ndarray]:
        """Category affinity and whether the candidate is a re-run.

        Read over more history than the retrievers' window: a category
        preference is a slower-moving fact than a topic, and estimating it from
        five clicks is estimating it from noise.

        Categories are coded to integers and counted into one (impression x
        category) table by `bincount`, so a competition chunk costs a handful of
        array passes instead of a `Counter` per impression. The table is dense
        because both datasets have tens of categories and a few hundred
        subcategories — dense is a megabyte per hundred thousand impressions,
        and a sparse one would cost more to index than to hold.
        """
        impressions, widths = _runs(frame["imp"].to_numpy())
        of_impression = np.repeat(np.arange(len(impressions)), widths)
        clicked_ids = [clicks_of[i][-PROFILE_CLICKS:] for i in impressions]
        length = np.fromiter((len(c) for c in clicked_ids), dtype="int64", count=len(impressions))

        out = {}
        for name, column in (("cat_aff", "category"), ("sub_aff", "subcategory")):
            article_code, distinct = self.taxonomy[column]
            counts = np.zeros((len(impressions), distinct + 1), dtype="float32")
            flat_impression = np.repeat(np.arange(len(impressions)), length)
            flat_code = np.fromiter(
                (article_code.get(c, distinct) for clicks in clicked_ids for c in clicks),
                dtype="int64",
                count=int(length.sum()),
            )
            np.add.at(counts, (flat_impression, flat_code), 1.0)
            candidate_code = np.fromiter(
                (article_code.get(a, distinct) for a in frame["aid"]),
                dtype="int64",
                count=len(frame),
            )
            out[name] = (
                counts[of_impression, candidate_code]
                / np.maximum(length[of_impression], 1)
            ).astype("float32")

        # A candidate the user has already clicked, found by one sorted search
        # over (impression, article) keys rather than a set per impression.
        article_row = self.article_row
        width = len(article_row) + 1
        seen = np.sort(
            np.repeat(np.arange(len(impressions), dtype="int64"), length) * width
            + np.fromiter(
                (article_row.get(c, len(article_row)) for clicks in clicked_ids for c in clicks),
                dtype="int64",
                count=int(length.sum()),
            )
        )
        asked = of_impression * width + np.fromiter(
            (article_row.get(a, len(article_row)) for a in frame["aid"]),
            dtype="int64",
            count=len(frame),
        )
        found = np.searchsorted(seen, asked)
        out["repeat"] = (
            (found < len(seen)) & (seen[np.minimum(found, max(len(seen) - 1, 0))] == asked)
        ).astype("float32")
        return out

    def _freshness(self, frame: pd.DataFrame) -> np.ndarray:
        """Minus the log age of the article in hours at impression time.

        Negated and logged so that larger is fresher and so that the first
        hours of an article's life, where the whole of its click curve happens,
        are not compressed against a tail measured in weeks. MIND publishes no
        timestamps, so there this is one constant and carries no information —
        which the model discovers rather than being told.
        """
        published = self.published
        # MIND carries no publication time — as NaT through ingest, and as no
        # column at all on the submission path, where the raw adapter runs.
        if not any(pd.notna(v) for v in published.values()):
            return np.full(len(frame), -np.log1p(UNKNOWN_AGE_HOURS), dtype="float32")
        age = (
            frame["t"] - frame["aid"].map(published)
        ).dt.total_seconds().to_numpy() / 3600.0
        age = np.nan_to_num(age, nan=UNKNOWN_AGE_HOURS)
        return -np.log1p(np.clip(age, 0.0, None)).astype("float32")


def normalise_within_impression(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Add a `_z` column per feature: its z-score among this impression's own candidates.

    A ranking model only ever compares candidates *inside* one impression, and
    the raw features do not say what is high for one. A three-hour CTR of 0.10
    is the pick of a quiet list and an also-ran on the front page. The z-score
    says which, and adding it rather than replacing the raw value keeps the
    absolute level, which the popularity features carry real information in.

    A constant column inside an impression — one candidate, or a tie — gets
    zero rather than a division by zero, which is the truthful answer: nothing
    here distinguishes them.
    """
    grouped = frame.groupby("imp", sort=False)
    for column in columns:
        mean = grouped[column].transform("mean")
        deviation = grouped[column].transform("std")
        frame[f"{column}_z"] = (
            ((frame[column] - mean) / deviation.replace(0, np.nan)).fillna(0.0)
        )
    return frame


def model_columns(columns) -> list[str]:
    """The feature names a model sees: each raw column and its z-score."""
    return [*columns, *(f"{c}_z" for c in columns)]


def _runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive runs of equal values, as (value, length).

    The exploded frame keeps an impression's candidates adjacent, so the
    per-impression loops below walk it in blocks instead of grouping it.
    """
    if len(values) == 0:
        return np.array([]), np.array([])
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], edges))
    lengths = np.diff(np.concatenate((starts, [len(values)])))
    return values[starts], lengths


def _scatter(mask: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size, dtype="float32")
    out[mask] = values
    return out


def _lookup(frame: pd.DataFrame, ranked: pd.DataFrame) -> np.ndarray:
    """Read a retriever's per-candidate scores back onto the exploded frame."""
    scores: dict[tuple[str, str], float] = {}
    for imp, ids, values in zip(
        ranked["impression_id"].astype(str), ranked["ranked_ids"], ranked["scores"]
    ):
        for article_id, score in zip(ids, values):
            scores[(imp, article_id)] = score
    return np.array(
        [scores.get(key, 0.0) for key in zip(frame["imp"], frame["aid"])],
        dtype="float32",
    )


def build(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    articles: pd.DataFrame,
    popularity: Popularity,
    content: Content,
    history_k: int = retrieval.HISTORY_K,
    labelled: bool = True,
) -> pd.DataFrame:
    """One row per candidate, with every feature the counter can supply.

    The popularity counter is passed in rather than fitted here because it is
    fitted on a *different* population from the one being scored — every
    labelled impression up to now, which for the validation split includes the
    training split and for a competition file includes nothing at all.
    """
    frame = explode(behaviors, labelled=labelled)
    popularity.attach(frame)
    content.attach(frame, behaviors, history, history_k)
    return normalise_within_impression(frame, [*popularity.columns, *CONTENT])


def _floor(times: np.ndarray, resolution: np.timedelta64) -> np.ndarray:
    """Round timestamps down to the counter's bucket edge."""
    scale = resolution.astype("timedelta64[ns]")
    return (times.astype("datetime64[ns]").view("int64") // scale.view("int64")
            * scale.view("int64")).view("datetime64[ns]")
