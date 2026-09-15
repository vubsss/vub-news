"""BM25 lexical retrieval: click history -> query -> ranked article ids.

Both the documents and the queries go through preprocess.cleaner for this
dataset's language, so the two sides of the index share a vocabulary. Nothing
here knows which dataset it holds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

from pipeline import ingest, preprocess, retrieval, weighting
from pipeline.datasets import DatasetConfig

# The article ids, saved next to the bm25s index, which does not store them.
ARTICLE_IDS = "article_ids.npy"


def build_queries(
    history: pd.DataFrame,
    articles: pd.DataFrame,
    config: DatasetConfig,
    history_k: int = retrieval.HISTORY_K,
    with_abstract: bool | None = None,
    weights: list[np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One query per impression: the last history_k clicked articles, cleaned.

    Titles or titles and abstracts, per the registry's `query_abstract` — the
    module used to assert titles, on the argument that abstracts would drown
    the terms identifying what a user actually reads. `with_abstract`
    overrides the registry, which is how `pipeline.lexical_ablation` measures
    the two against each other rather than restating the argument.

    A user with no history has no query, which is reported rather than filled
    in — see `retrieve`.
    """
    if with_abstract is None:
        with_abstract = config.lexical.query_abstract
    clean = preprocess.cleaner(config)
    text = articles["title"].fillna("")
    if with_abstract:
        text = (text + " " + articles["abstract"].fillna("")).str.strip()
    source = dict(zip(articles["article_id"], text))

    if weights is None:
        query = [
            clean(
                " ".join(
                    source.get(article_id, "") for article_id in clicks[-history_k:]
                )
            )
            for clicks in history["click_history"]
        ]
    else:
        # Recency as term frequency: a click worth more contributes its title
        # more times, which raises those terms before BM25 saturates them --
        # the same mechanism `title_weight` uses on the document side, and the
        # only one a bag of terms has. The tuned k1, b and title weight are
        # untouched; this changes what the query says, not how it is scored.
        query = [
            clean(
                " ".join(
                    " ".join([source.get(article_id, "")] * int(count))
                    for article_id, count in zip(
                        clicks[-history_k:], repeats_for(found), strict=True
                    )
                    if count > 0
                )
            )
            for clicks, found in zip(history["click_history"], weights, strict=True)
        ]
    # Indexed positionally, not by whatever the caller sliced: run passes the
    # validation slice of the history, and carrying its gappy index onward
    # would misalign the queries against anything built alongside them.
    queries = pd.DataFrame(
        {
            "impression_id": list(history["impression_id"].astype("string")),
            "query": pd.Series(query, dtype="string"),
        }
    )
    report = {
        "impressions": len(queries),
        # No clicks at all, versus clicks whose ids the catalogue does not
        # contain: both leave nothing to search with, for different reasons.
        "cold": int(history["click_history"].map(len).eq(0).sum()),
        "empty_query": int((queries["query"] == "").sum()),
    }
    return queries, report


@dataclass(frozen=True)
class Index:
    """A BM25 index over one dataset's article corpus.

    bm25s works in corpus positions; article_ids is what turns those back into
    ids, and is the only thing a retrieved id can come from — so retrieval
    cannot return an article the corpus does not contain.
    """

    bm25: bm25s.BM25
    article_ids: np.ndarray

    @cached_property
    def position(self) -> dict[str, int]:
        """article id -> corpus position. Built once; scoring a candidate list
        does one lookup per candidate."""
        return {article_id: row for row, article_id in enumerate(self.article_ids)}

    def save(self, directory: Path) -> None:
        """bm25s stores the term statistics; the article ids are ours to keep,
        and without them its corpus positions mean nothing."""
        directory.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(directory), show_progress=False)
        np.save(directory / ARTICLE_IDS, self.article_ids, allow_pickle=True)

    def retrieve(self, queries: pd.DataFrame, depth: int) -> pd.DataFrame:
        """Top-`depth` article ids and scores per impression, best first.

        An empty query is never handed to bm25s: it would come back with
        `depth` arbitrary articles at score 0, which reads downstream as a
        retriever that missed rather than one that was never asked. It gets an
        empty ranking instead.
        """
        depth = min(depth, len(self.article_ids))
        text = list(queries["query"])
        asked = [i for i, query in enumerate(text) if query]

        ranked_ids: list[list[str]] = [[] for _ in text]
        scores: list[list[float]] = [[] for _ in text]
        if asked:
            positions, found = self.bm25.retrieve(
                [text[i].split() for i in asked], k=depth, show_progress=False
            )
            for row, i in enumerate(asked):
                ranked_ids[i] = list(self.article_ids[positions[row]])
                scores[i] = [float(score) for score in found[row]]

        return pd.DataFrame(
            {
                "impression_id": list(queries["impression_id"]),
                "ranked_ids": ranked_ids,
                "scores": scores,
            }
        )

    def score_candidates(
        self, queries: pd.DataFrame, candidates: list[list[str]]
    ) -> pd.DataFrame:
        """Rank an impression's own candidate list instead of the whole corpus.

        What the evaluation harness scores and what tickets 13 and 14 submit:
        the candidates are given, and every one of them must come back in an
        order, so unlike `retrieve` this drops nothing. A candidate outside the
        corpus scores 0, and an empty query scores every candidate 0 — in both
        cases the sort is stable, so what comes back is the order it arrived
        in, which is the honest answer when there is nothing to rank on.
        """
        ranked_ids: list[list[str]] = []
        scores: list[list[float]] = []

        for text, impression in zip(queries["query"], candidates, strict=True):
            found = np.zeros(len(impression), dtype="float32")
            if text:
                rows = np.array(
                    [self.position.get(candidate, -1) for candidate in impression]
                )
                known = rows >= 0
                if known.any():
                    corpus = self.bm25.get_scores(text.split())
                    found[known] = corpus[rows[known]]
            order = np.argsort(-found, kind="stable")
            ranked_ids.append([impression[position] for position in order])
            scores.append([float(found[position]) for position in order])

        return pd.DataFrame(
            {
                "impression_id": list(queries["impression_id"]),
                "ranked_ids": ranked_ids,
                "scores": scores,
            }
        )


    @cached_property
    def by_document(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The index transposed: per document, its terms and their BM25 weights.

        bm25s stores the index the way retrieval wants it — a posting list per
        term — and `get_scores` walks those lists to score the entire corpus.
        That is the right shape for "which documents match this query" and the
        wrong one for "what does this query score against these fifteen
        documents", which is the only question `score_pairs` asks. Answering it
        off the term lists costs the length of every posting list the query
        touches; answering it off this one costs the number of distinct terms in
        the fifteen documents.

        The difference is not an optimisation, it is what makes the feature
        exist at all: scoring EB-NeRD's 13.5M competition impressions against a
        125k-article corpus through `get_scores` is on the order of a day.

        Built once per index and cached, because the transpose costs a sort of
        300k entries and the caller does this millions of times.
        """
        weights = self.bm25.scores["data"]
        # CSC over terms: indptr is one entry per term, indices the document
        # each weight belongs to. Regrouping it by document is a sort of the
        # documents and reading the term off the column each entry came from.
        documents = self.bm25.scores["indices"]
        column = self.bm25.scores["indptr"]
        term = np.repeat(
            np.arange(len(column) - 1, dtype="int32"), np.diff(column)
        )
        order = np.argsort(documents, kind="stable")
        boundaries = np.searchsorted(
            documents[order], np.arange(self.bm25.scores["num_docs"] + 1)
        )
        return term[order].astype("int32"), weights[order], boundaries

    def score_pairs(
        self, queries: list[list[str]], candidates: list[list[str]], batch: int = 200_000
    ) -> np.ndarray:
        """BM25 of each query against its own candidates, concatenated.

        Identical arithmetic to `score_candidates` — same weights, summed over
        the same terms — read out of the transpose above rather than out of a
        corpus-wide score vector. A term repeated in the query counts as many
        times as it appears, which is what `get_scores` does and what makes the
        two agree to the last bit rather than approximately. A candidate outside
        the corpus, and a query with no term the index knows, both score 0.

        Written without a Python loop over pairs because the submission path
        calls it on 206 million of them. Each candidate's term block is
        expanded, every (impression, term) pair is looked up in the query table
        by one sorted search, and the products are summed back per candidate by
        `bincount` — three vectorised passes over the expansion rather than one
        interpreted step per candidate.

        Returned flat, in the order the candidate lists were given, because the
        caller holds a per-candidate frame in exactly that order.
        """
        terms, weights, boundaries = self.by_document
        vocabulary = self.bm25.vocab_dict
        position = self.position
        vocabulary_size = len(vocabulary)

        widths = np.fromiter((len(c) for c in candidates), dtype="int64", count=len(candidates))
        rows = np.fromiter(
            (position.get(candidate, -1) for impression in candidates for candidate in impression),
            dtype="int64",
            count=int(widths.sum()),
        )
        impression_of = np.repeat(np.arange(len(candidates), dtype="int64"), widths)

        # The query side as one sorted table of (impression * vocabulary + term)
        # -> how many times the term occurs in that impression's query. One key
        # space for both halves is what turns the lookup into a searchsorted.
        keys: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for i, query in enumerate(queries):
            asked = [vocabulary[token] for token in query if token in vocabulary]
            if not asked:
                continue
            token, count = np.unique(np.asarray(asked, dtype="int64"), return_counts=True)
            keys.append(i * vocabulary_size + token)
            counts.append(count.astype("float32"))
        if keys:
            query_keys = np.concatenate(keys)
            query_counts = np.concatenate(counts)
            order = np.argsort(query_keys, kind="stable")
            query_keys, query_counts = query_keys[order], query_counts[order]
        else:
            query_keys = np.empty(0, dtype="int64")
            query_counts = np.empty(0, dtype="float32")

        scores = np.zeros(len(rows), dtype="float32")
        for start in range(0, len(rows), batch):
            block = slice(start, min(start + batch, len(rows)))
            scores[block] = self._score_block(
                rows[block],
                impression_of[block],
                query_keys,
                query_counts,
                vocabulary_size,
                terms,
                weights,
                boundaries,
            )
        return scores

    @staticmethod
    def _score_block(
        rows: np.ndarray,
        impression_of: np.ndarray,
        query_keys: np.ndarray,
        query_counts: np.ndarray,
        vocabulary_size: int,
        terms: np.ndarray,
        weights: np.ndarray,
        boundaries: np.ndarray,
    ) -> np.ndarray:
        """One batch of candidates, expanded into term entries and summed back."""
        known = rows >= 0
        starts = np.where(known, boundaries[np.maximum(rows, 0)], 0)
        lengths = np.where(known, boundaries[np.maximum(rows, 0) + 1] - starts, 0)
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(len(rows), dtype="float32")

        # The ragged concatenation of every candidate's entries, built by
        # offsetting a flat arange rather than by concatenating per candidate.
        candidate_of = np.repeat(np.arange(len(rows), dtype="int64"), lengths)
        offsets = np.cumsum(lengths) - lengths
        entry = (
            np.arange(total, dtype="int64")
            - np.repeat(offsets, lengths)
            + np.repeat(starts, lengths)
        )

        wanted = impression_of[candidate_of] * vocabulary_size + terms[entry]
        found = np.searchsorted(query_keys, wanted)
        matched = (found < len(query_keys)) & (
            query_keys[np.minimum(found, max(len(query_keys) - 1, 0))] == wanted
        )
        multiplicity = np.where(matched, query_counts[np.minimum(found, max(len(query_keys) - 1, 0))], 0.0)

        return np.bincount(
            candidate_of, weights=multiplicity * weights[entry], minlength=len(rows)
        ).astype("float32")


def build(articles: pd.DataFrame, config: DatasetConfig) -> Index:
    """Index the catalogue's lexical_text, which preprocess already cleaned.

    Tokenising is a plain split rather than bm25s' own tokenizer: that one
    would apply English stopwords and its own token pattern on top of the
    language-correct cleaning preprocess did, which for Danish means removing
    English words and nothing else. Queries are split the same way.
    """
    text = articles["lexical_text"].fillna("")
    bm25 = bm25s.BM25(k1=config.lexical.k1, b=config.lexical.b)
    bm25.index([document.split() for document in text], show_progress=False)
    return Index(bm25=bm25, article_ids=articles["article_id"].to_numpy(dtype=object))


def load(directory: Path) -> Index:
    return Index(
        bm25=bm25s.BM25.load(str(directory), show_progress=False),
        article_ids=np.load(directory / ARTICLE_IDS, allow_pickle=True),
    )


def run(config: DatasetConfig, force: bool = False) -> None:
    """Build the index, retrieve for the validation split, report recall@K."""
    articles = pd.read_parquet(config.feature_store_dir / "articles.parquet")
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    # Only the split this stage retrieves for: the per-impression view is
    # rebuilt on demand, so building it for impressions nobody scores is work
    # and memory spent on nothing.
    history = ingest.history_for(
        config, behaviors[behaviors["split"] == retrieval.VALIDATION]
    )

    directory = config.artifacts_dir / "bm25"
    if force or not (directory / ARTICLE_IDS).exists():
        started = time.perf_counter()
        index = build(articles, config)
        elapsed = time.perf_counter() - started
        index.save(directory)
        print(f"    index over {len(articles):,} articles built in {elapsed:.1f} s")
    else:
        index = load(directory)
        print(f"    index over {len(articles):,} articles loaded from {directory}")

    validation = behaviors[behaviors["split"] == retrieval.VALIDATION]
    clicks = history[history["impression_id"].isin(set(validation["impression_id"]))]
    queries, asked = build_queries(clicks, articles, config)

    started = time.perf_counter()
    ranked = index.retrieve(queries, depth=max(retrieval.DEPTHS))
    latency = 1000 * (time.perf_counter() - started) / max(len(queries), 1)
    retrieval.check_within_corpus(ranked, index.article_ids)

    total = asked["impressions"]
    cold, empty = asked["cold"], asked["empty_query"]
    print(
        f"    {total:,} validation impressions, {cold:,} "
        f"({100 * cold / total if total else 0:.2f}%) cold — no history to "
        f"query with, so they retrieve nothing and recall 0"
    )
    if empty > cold:
        print(
            f"    a further {empty - cold:,} have a history of article ids the "
            f"catalogue does not contain, so they are equally unsearchable"
        )
    print(f"    mean query latency {latency:.2f} ms")

    recall = retrieval.recall_at_k(ranked, validation, retrieval.DEPTHS)
    for depth in retrieval.DEPTHS:
        print(f"    recall@{depth:<4} {recall[f'recall@{depth}']:.4f}")
    print(
        f"    over {recall['scored']:,} impressions with a click; "
        f"{recall['no_positive']:,} had none and are not averaged in"
    )


def retrieve_corpus(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
    depth: int = max(retrieval.DEPTHS),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rank the whole catalogue per impression: what recall@K is measured on.

    The other half of the pair below. `rank_candidates` reorders the candidates
    an impression already carries; this one searches the corpus, which is the
    only one of the two a retrieval depth means anything to — a candidate list
    is scored whole. ann_index exposes this under the same name and signature,
    so the sweep varies the history window through one call for both retrievers
    and has no way to hand them different ones.
    """
    articles = pd.read_parquet(config.feature_store_dir / "articles.parquet")
    index = load(config.artifacts_dir / "bm25")

    wanted = set(behaviors["impression_id"])
    clicks = history[history["impression_id"].isin(wanted)]
    queries, asked = build_queries(clicks, articles, config, history_k)

    ranked = index.retrieve(queries, depth=depth)
    retrieval.check_within_corpus(ranked, index.article_ids)
    return ranked, asked


# The most times any one clicked title is repeated in a query. Recency
# weighting on the lexical side has to become an integer count of tokens, so
# the continuous weights are quantised onto 0..MAX_REPEAT: the most-weighted
# click is repeated MAX_REPEAT times and one weighted below half a step drops
# out of the query entirely, which is what a decay is for. Small because a
# query is already the concatenation of K titles and this multiplies its
# length -- at K=80 and MAX_REPEAT=3 the worst case is 240 titles.
MAX_REPEAT = 3


def repeats_for(found: np.ndarray) -> np.ndarray:
    """Per-click token repetitions, from the weights the profile would use.

    Scaled by the largest weight rather than by their sum, so the newest click
    lands on MAX_REPEAT whatever the decay constant is and the shape of the
    query depends on the decay's *ratios*. A weight of zero repeats zero times.
    """
    if not len(found):
        return np.empty(0, dtype="int64")
    peak = found.max()
    if peak <= 0:
        return np.ones(len(found), dtype="int64")
    return np.rint(MAX_REPEAT * found / peak).astype("int64")


def window_for(history_k: int, pooling: str) -> int:
    """The history window this pooling leaves BM25 with.

    A bag of terms has no aggregator to vary — the query is one document
    however it was assembled — so what pooling means on the lexical side is how
    much of the history goes into that bag. `mean` reads the last K clicks;
    `last` reads only the most recent, which is the same query at K=1.

    `max` has no lexical form at all. Scoring a candidate against each clicked
    title separately and keeping the best is a different retriever, not a
    different query, and quietly returning the `mean` query instead would put a
    row in a sweep whose label did not describe what ran.
    """
    if pooling not in retrieval.POOLINGS:
        raise ValueError(
            f"unknown pooling {pooling!r}, expected one of "
            f"{', '.join(retrieval.POOLINGS)}"
        )
    if pooling == "max":
        raise ValueError(
            "bm25 cannot express 'max' pooling: a bag of terms has no "
            "per-click aggregator. Sweep it on the semantic retriever only."
        )
    return 1 if pooling == "last" else history_k


def query_weights(
    config: DatasetConfig,
    history: pd.DataFrame,
    history_k: int,
    served: dict,
) -> list[np.ndarray] | None:
    """One weight per click in the window, or None when the scheme is uniform.

    Unlike the semantic side nothing is dropped here — a click the catalogue
    does not hold contributes an empty string rather than disappearing — so the
    weights pair with the window directly and need no `kept` subsetting.
    """
    spec = config.weighting
    if spec.scheme == "uniform":
        return None
    weighting.check(config, spec.scheme, history_k)

    columns = {
        name: (history[name] if name in history else None)
        for name in ("click_times", "click_read_times", "click_scroll")
    }
    built: list[np.ndarray] = []
    for position, (impression_id, clicks) in enumerate(
        zip(history["impression_id"], history["click_history"])
    ):
        window = {
            name: None
            if column is None
            else np.asarray(column.iloc[position])[-history_k:]
            for name, column in columns.items()
        }
        built.append(
            weighting.weights(
                spec.scheme,
                spec.decay,
                len(clicks[-history_k:]),
                times=window["click_times"],
                read_times=window["click_read_times"],
                scroll=window["click_scroll"],
                at=served.get(impression_id),
            )
        )
    return built


def rank_candidates(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
    pooling: str = retrieval.POOLING,
    stores: dict | None = None,
) -> pd.DataFrame:
    """Score each impression's own candidates. The harness's only entry here.

    ann_index exposes the same function with the same signature, which is what
    lets the harness score both retrievers without knowing which it holds — and
    what stops a sweep handing the two of them different windows or different
    poolings while reporting one cell.

    `stores` is for the serving benchmark, which issues one request at a time:
    reading the catalogue and the index inside the call is right for a batch
    and is the entire cost of a single request. The three retrievers take the
    same argument and each reads the keys it needs, so the harness's one
    signature stays one signature. Passed in rather than given a second scoring
    function, so a served request and a measured batch are the same arithmetic
    -- `test_the_served_scorers_agree_with_the_harness`.
    """
    opened = stores or {}
    articles = opened.get("articles")
    if articles is None:
        articles = pd.read_parquet(config.feature_store_dir / "articles.parquet")
    index = opened.get("bm25_index")
    if index is None:
        index = load(config.artifacts_dir / "bm25")

    wanted = set(behaviors["impression_id"])
    clicks = history[history["impression_id"].isin(wanted)]
    window = window_for(history_k, pooling)
    served = dict(zip(behaviors["impression_id"], behaviors["impression_time"]))
    queries, _ = build_queries(
        clicks,
        articles,
        config,
        window,
        weights=query_weights(config, clicks, window, served),
    )

    # Looked up by id rather than zipped: the history frame is filtered from a
    # larger one and need not arrive in the behaviours frame's order, and a
    # positional pairing would score each impression against another's
    # candidates while looking entirely well-formed.
    candidates_of = dict(zip(behaviors["impression_id"], behaviors["candidate_ids"]))
    candidates = [candidates_of[i] for i in queries["impression_id"]]
    return index.score_candidates(queries, candidates)


@dataclass(frozen=True)
class Ranker:
    """Scores a supplied candidate list against a supplied catalogue.

    `rank_candidates` above is the same operation over the feature store; this
    one is handed the corpus, so the submission path can rank against the
    competition's own catalogue, which the feature store does not contain.
    ann_index exposes the same pair of names, and that is all `predict` knows
    about either retriever.
    """

    index: Index
    articles: pd.DataFrame
    config: DatasetConfig
    history_k: int

    def rank(self, history: pd.DataFrame, candidates: list[list[str]]) -> pd.DataFrame:
        # The weights the feature-store path builds, built here too. Omitting
        # them used to be silent: the query was assembled unweighted while the
        # registry said otherwise, so the file ranked by a different profile
        # from the one every reported number was measured on and nothing in the
        # output could show it.
        served = dict(zip(history["impression_id"], history["impression_time"]))
        queries, _ = build_queries(
            history,
            self.articles,
            self.config,
            self.history_k,
            weights=query_weights(self.config, history, self.history_k, served),
        )
        return self.index.score_candidates(queries, candidates)


def ranker(
    articles: pd.DataFrame,
    config: DatasetConfig,
    workdir: Path,
    history_k: int = retrieval.HISTORY_K,
) -> Ranker:
    """An index over `articles`, built once and kept under `workdir`.

    Saved rather than rebuilt per run because the caller streams millions of
    impressions past it and may well be resuming a run that stopped partway.
    """
    directory = workdir / "bm25"
    index = (
        load(directory) if (directory / ARTICLE_IDS).exists() else build(articles, config)
    )
    if not (directory / ARTICLE_IDS).exists():
        index.save(directory)
    return Ranker(
        index=index, articles=articles, config=config, history_k=history_k
    )
