"""Semantic retrieval: click history -> user vector -> ranked article ids.

The counterpart to bm25_index, emitting the same ranked-candidates shape and
scored by the same recall, so the two are read side by side. Nothing here
knows which dataset it holds: it asks embed for a matrix and the registry for
nothing at all.

The index is exact rather than approximate. At 20,738 articles for EB-NeRD and
65,238 for MIND a brute-force inner product costs milliseconds, so approximate
search would buy nothing and would put its own recall loss inside the number
this ticket exists to measure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from pipeline import embed, retrieval, weighting
from pipeline.datasets import DatasetConfig


@dataclass(frozen=True)
class Queries:
    """One user vector per impression, with the impression ids attached.

    The ids travel with the matrix because the matrix is positional and the
    history frame it was built from is a filtered slice of a larger one —
    carrying that frame's gappy index onward is how a ranking gets attached to
    the wrong impression.
    """

    impression_ids: list[str]
    vectors: np.ndarray


def build_user_vectors(
    history: pd.DataFrame,
    embeddings: embed.Embeddings,
    history_k: int = retrieval.HISTORY_K,
) -> tuple[Queries, dict[str, int]]:
    """One vector per impression: the mean of the last history_k clicked rows.

    The mean is taken over the clicks the catalogue actually holds. A click on
    an article with no embedding row contributes nothing rather than dragging
    the user toward the origin, which is what averaging in its zero row would
    do — that would make a user with unknown clicks look like a user with
    different interests rather than one we know less about.

    The result is scaled back to unit length, so a score against a unit article
    vector is a cosine and scores mean the same thing for every user whatever
    their history length.
    """
    row_of = embeddings.index
    width = embeddings.vectors.shape[1]
    vectors = np.zeros((len(history), width), dtype="float32")

    cold = unresolved = 0
    for i, clicks in enumerate(history["click_history"]):
        if len(clicks) == 0:
            cold += 1
            continue
        rows = [row_of[click] for click in clicks[-history_k:] if click in row_of]
        if not rows:
            unresolved += 1
            continue
        vectors[i] = embeddings.vectors[rows].mean(axis=0)

    report = {
        "impressions": len(history),
        # No clicks at all, versus clicks the catalogue does not contain: both
        # leave a zero vector, for different reasons.
        "cold": cold,
        "empty_query": cold + unresolved,
    }
    # normalise leaves the zero rows at zero rather than turning them into NaN.
    return Queries(
        impression_ids=list(history["impression_id"].astype("string")),
        vectors=embed.normalise(vectors),
    ), report


def weighted_user_vectors(
    history: pd.DataFrame,
    embeddings: embed.Embeddings,
    config: DatasetConfig,
    history_k: int = retrieval.HISTORY_K,
) -> Queries:
    """One vector per impression: the *weighted* mean of its clicked rows.

    `build_user_vectors` above is this with equal weights, and the reason both
    exist is cost. Scoring a weighted profile the way the feature-store path
    does — every candidate against every click, then weight, then pool — is a
    (candidates x clicks) matmul per impression, which the submission cannot
    afford 13.5 million times. The weighted mean is the one aggregator that
    collapses into a single vector first:

        sum_i w_i (c . x_i)  ==  c . (sum_i w_i x_i)

    so the weights are applied to the clicks once, and the ranking that comes
    out is the same ranking `score_candidates_pooled` produces the slow way.
    `require_expressible` is what keeps that identity's condition true.

    The weights arrive already normalised and already subset to the clicks the
    catalogue holds, which is what `Clicks.kept` is for: pairing them with the
    full window instead would land every weight on the wrong click.
    """
    clicks = build_clicks(history, embeddings, history_k)
    at = (
        history["impression_time"].reset_index(drop=True)
        if "impression_time" in history
        else None
    )
    weights = click_weights(config, history, clicks, history_k, at)

    width = embeddings.vectors.shape[1]
    vectors = np.zeros((len(clicks.rows), width), dtype="float32")
    for i, (rows, weight) in enumerate(zip(clicks.rows, weights, strict=True)):
        if len(rows):
            vectors[i] = embeddings.vectors[rows].T @ weight.astype("float32")

    # Rescaled like the unweighted path, so a score is a cosine. A positive
    # rescale per impression cannot reorder that impression's candidates.
    return Queries(
        impression_ids=clicks.impression_ids,
        vectors=embed.normalise(vectors),
    )


def require_expressible(pooling: str = retrieval.POOLING) -> None:
    """That the submission path can rank the way the registry says to.

    Every weighting scheme is expressible here now that `predict` streams the
    columns each one reads. What is not is an aggregator other than `mean`.
    The submission ranks 13.5M impressions against a catalogue, which it can
    only afford as one vector per impression, and the identity that makes a
    weighted profile *fit* in one vector

        sum_i w_i (c . x_i)  ==  c . (sum_i w_i x_i)

    holds for a weighted mean and for nothing else. `max` asks which single
    click matches best, which is not a dot product with any vector.

    So it raises rather than quietly ranking by the mean: a submission file
    produced by a different aggregator from the one every reported number came
    from would be well-formed, correctly ordered, and wrong in the one way
    nothing downstream could see.
    """
    if pooling != "mean":
        raise weighting.WeightingError(
            f"the submission path cannot express {pooling!r} pooling: it ranks "
            f"by one vector per impression, which only a weighted mean folds "
            f"into. Submit under 'mean', or give predict a per-click scorer."
        )


@dataclass(frozen=True)
class Clicks:
    """Per impression, the embedding rows of its last K resolvable clicks.

    What `build_user_vectors` pools into one vector, kept unpooled — because
    `max` and `last` are not expressible as a single vector and a mean is. The
    rows are already restricted to clicks the catalogue has a vector for, and
    already truncated to the window, so the aggregator sees exactly the clicks
    the profile would have averaged.
    """

    impression_ids: list[str]
    rows: list[np.ndarray]
    # Which positions of the last-K window each row came from. Clicks the
    # catalogue has no vector for are dropped, so a weight computed over the
    # window has to be subset by this before it can pair with `rows` -- zipping
    # the two directly would land every weight on the wrong click and look
    # entirely well-formed.
    kept: list[np.ndarray]


def build_clicks(
    history: pd.DataFrame,
    embeddings: embed.Embeddings,
    history_k: int = retrieval.HISTORY_K,
) -> Clicks:
    """The same clicks `build_user_vectors` averages, before it averages them.

    A click the catalogue has no vector for is dropped rather than carried as
    a zero row, which is what the mean does too: a user with unknown clicks is
    one we know less about, not one with different interests.
    """
    row_of = embeddings.index
    rows: list[np.ndarray] = []
    kept: list[np.ndarray] = []
    for clicks in history["click_history"]:
        window = clicks[-history_k:]
        found = [
            (position, row_of[click])
            for position, click in enumerate(window)
            if click in row_of
        ]
        kept.append(np.array([p for p, _ in found], dtype="int64"))
        rows.append(np.array([r for _, r in found], dtype="int64"))
    return Clicks(
        impression_ids=list(history["impression_id"].astype("string")),
        rows=rows,
        kept=kept,
    )


@dataclass(frozen=True)
class Index:
    """An exact inner-product index over one dataset's article embeddings.

    faiss works in row positions; embeddings.article_ids is what turns those
    back into ids, and is the only thing a retrieved id can come from — so
    retrieval cannot return an article the corpus does not contain.
    """

    index: faiss.Index
    embeddings: embed.Embeddings

    @property
    def article_ids(self) -> np.ndarray:
        return self.embeddings.article_ids

    def retrieve(self, queries: Queries, depth: int) -> pd.DataFrame:
        """Top-`depth` article ids and scores per impression, best first.

        A zero vector is never handed to faiss. Every article would score 0
        against it and the index would return `depth` arbitrary articles, some
        of which would be clicked ones by luck — semantic recall would come out
        above lexical recall for the very users neither retriever knows
        anything about. bm25_index returns an empty ranking for an empty query
        for the same reason, and the two figures are only comparable if the
        users neither can serve are dropped from both the same way.
        """
        depth = min(depth, len(self.article_ids))
        asked = [i for i, vector in enumerate(queries.vectors) if vector.any()]

        ranked_ids: list[list[str]] = [[] for _ in queries.impression_ids]
        scores: list[list[float]] = [[] for _ in queries.impression_ids]
        if asked:
            found, positions = self.index.search(
                np.ascontiguousarray(queries.vectors[asked]), depth
            )
            for row, i in enumerate(asked):
                ranked_ids[i] = list(self.article_ids[positions[row]])
                scores[i] = [float(score) for score in found[row]]

        return pd.DataFrame(
            {
                "impression_id": queries.impression_ids,
                "ranked_ids": ranked_ids,
                "scores": scores,
            }
        )

    def score_candidates(
        self, queries: Queries, candidates: list[list[str]]
    ) -> pd.DataFrame:
        """Rank an impression's own candidate list instead of the whole corpus.

        What tickets 13 and 14 submit: the competition supplies the candidates
        and every one of them must come back ranked, so unlike `retrieve` this
        drops nothing. A candidate the corpus has no vector for scores 0, and a
        cold user scores every candidate 0 — in both cases the sort is stable,
        so what comes back is the order the competition gave, which is the
        honest answer when there is nothing to rank on.
        """
        row_of = self.embeddings.index
        ranked_ids: list[list[str]] = []
        scores: list[list[float]] = []

        for vector, impression in zip(queries.vectors, candidates, strict=True):
            rows = [row_of.get(candidate, -1) for candidate in impression]
            known = np.array(rows) >= 0
            found = np.zeros(len(impression), dtype="float32")
            if known.any():
                matrix = self.embeddings.vectors[np.array(rows)[known]]
                found[known] = matrix @ vector
            order = np.argsort(-found, kind="stable")
            ranked_ids.append([impression[position] for position in order])
            scores.append([float(found[position]) for position in order])

        return pd.DataFrame(
            {
                "impression_id": queries.impression_ids,
                "ranked_ids": ranked_ids,
                "scores": scores,
            }
        )


    def score_candidates_pooled(
        self,
        clicks: Clicks,
        candidates: list[list[str]],
        pooling: str,
        weights: list[np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """Rank an impression's candidates by an aggregator over its clicks.

        `score = AGG_i (candidate . click_i)`, which is the one form all three
        poolings share. `mean` is here for completeness and to be checked
        against the pooled-vector path rather than assumed equal to it: the
        mean of the dot products is the dot product with the mean, so the two
        rank identically even though the pooled path rescales to unit length.

        Only the re-ranking path offers this. Corpus retrieval keeps the
        pooled vector, because a max over K clicks against the whole catalogue
        is K searches rather than one, and recall@K must stay a measurement of
        the same thing across every cell of a sweep.

        `weights` says how much each click counts, one array per impression
        over the window, already subset to the clicks that survived. They scale
        the similarities before the aggregator sees them, so `mean` becomes a
        weighted mean and `max` asks which click matches best *after* recency
        is applied. `last` is unaffected by construction: scaling one click's
        similarity by a positive number cannot reorder the candidates.
        """
        if pooling not in retrieval.POOLINGS:
            raise ValueError(
                f"unknown pooling {pooling!r}, expected one of "
                f"{', '.join(retrieval.POOLINGS)}"
            )
        row_of = self.embeddings.index
        ranked_ids: list[list[str]] = []
        scores: list[list[float]] = []

        per_click = weights if weights is not None else [None] * len(clicks.rows)
        for rows, impression, weight in zip(
            clicks.rows, candidates, per_click, strict=True
        ):
            found = np.zeros(len(impression), dtype="float32")
            candidate_rows = np.array(
                [row_of.get(candidate, -1) for candidate in impression]
            )
            known = candidate_rows >= 0
            if len(rows) and known.any():
                # (candidates x clicks): every candidate against every click.
                similarity = (
                    self.embeddings.vectors[candidate_rows[known]]
                    @ self.embeddings.vectors[rows].T
                )
                if weight is not None and len(weight):
                    if len(weight) != similarity.shape[1]:
                        raise ValueError(
                            f"{len(weight)} weights for "
                            f"{similarity.shape[1]} clicks"
                        )
                    similarity = similarity * weight.astype("float32")
                if pooling == "mean":
                    # A weighted mean: the weights are normalised, so this is
                    # the sum. Unweighted they are all 1/K and it is the mean.
                    found[known] = (
                        similarity.sum(axis=1)
                        if weight is not None and len(weight)
                        else similarity.mean(axis=1)
                    )
                elif pooling == "max":
                    found[known] = similarity.max(axis=1)
                else:
                    # The window is a suffix of the history in click order, so
                    # the most recent click is its last column.
                    found[known] = similarity[:, -1]
            order = np.argsort(-found, kind="stable")
            ranked_ids.append([impression[position] for position in order])
            scores.append([float(found[position]) for position in order])

        return pd.DataFrame(
            {
                "impression_id": clicks.impression_ids,
                "ranked_ids": ranked_ids,
                "scores": scores,
            }
        )


def build(embeddings: embed.Embeddings) -> Index:
    """Index every article vector for exact inner-product search.

    The vectors are unit length by the time embed hands them over, so an inner
    product is a cosine similarity — which is the whole reason ticket 7 asserts
    that property rather than assuming it.

    Nothing is written to disk: a flat inner-product index holds the same
    matrix `vectors.npy` already stores, so saving it would be storing the
    embeddings twice to skip a copy that takes milliseconds.
    """
    index = faiss.IndexFlatIP(embeddings.vectors.shape[1])
    index.add(np.ascontiguousarray(embeddings.vectors))
    return Index(index=index, embeddings=embeddings)


def run(config: DatasetConfig, force: bool = False) -> None:
    """Build the index, retrieve for the validation split, report recall@K."""
    embeddings = embed.load(config)
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    history = pd.read_parquet(config.feature_store_dir / "history.parquet")

    started = time.perf_counter()
    index = build(embeddings)
    elapsed = time.perf_counter() - started
    print(
        f"    index over {len(embeddings.article_ids):,} articles x "
        f"{embeddings.vectors.shape[1]} dimensions built in {elapsed:.1f} s"
    )

    validation = behaviors[behaviors["split"] == retrieval.VALIDATION]
    clicks = history[history["impression_id"].isin(set(validation["impression_id"]))]
    queries, asked = build_user_vectors(clicks, embeddings)

    started = time.perf_counter()
    ranked = index.retrieve(queries, depth=max(retrieval.DEPTHS))
    latency = 1000 * (time.perf_counter() - started) / max(len(queries.vectors), 1)
    retrieval.check_within_corpus(ranked, index.article_ids)

    total = asked["impressions"]
    cold, empty = asked["cold"], asked["empty_query"]
    print(
        f"    {total:,} validation impressions, {cold:,} "
        f"({100 * cold / total if total else 0:.2f}%) cold — no history to "
        f"build a vector from, so they retrieve nothing and recall 0"
    )
    if empty > cold:
        print(
            f"    a further {empty - cold:,} clicked only articles the "
            f"catalogue holds no vector for, so they are equally unsearchable"
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

    Takes no pooling and never will. A max over K clicks against the whole
    catalogue is K searches rather than one, and recall@K has to keep measuring
    the same thing in every cell of a sweep — so this is always the pooled mean
    vector, and the sweep's document says so where the recall column appears.

    The other half of the pair below. `rank_candidates` reorders the candidates
    an impression already carries; this one searches the corpus, which is the
    only one of the two a retrieval depth means anything to — a candidate list
    is scored whole. bm25_index exposes this under the same name and signature,
    so the sweep varies the history window through one call for both retrievers
    and has no way to hand them different ones.
    """
    embeddings = embed.load(config)
    index = build(embeddings)

    wanted = set(behaviors["impression_id"])
    clicks = history[history["impression_id"].isin(wanted)]
    queries, asked = build_user_vectors(clicks, embeddings, history_k)

    ranked = index.retrieve(queries, depth=depth)
    retrieval.check_within_corpus(ranked, index.article_ids)
    return ranked, asked


def click_weights(
    config: DatasetConfig,
    history: pd.DataFrame,
    clicks: Clicks,
    history_k: int,
    at: pd.Series | None = None,
) -> list[np.ndarray] | None:
    """One weight per surviving click, per impression, normalised to sum to 1.

    None when the scheme is uniform, so the unweighted path stays exactly the
    path every recorded number came from rather than a weighted one that
    happens to use equal weights.

    The window is sliced the same way everywhere — `[-history_k:]`, a suffix —
    and then subset by `clicks.kept`, which is what makes a weight land on the
    click it was computed for even though the catalogue dropped some.
    """
    spec = config.weighting
    if spec.scheme == "uniform":
        return None
    weighting.check(config, spec.scheme, history_k)

    columns = {
        name: (history[name] if name in history else None)
        for name in ("click_times", "click_read_times", "click_scroll")
    }
    stamps = list(at) if at is not None else [None] * len(history)

    # The window's own length, not the length of whichever column happens to
    # be present. `kept` is a subset -- the clicks the catalogue has vectors
    # for -- so sizing the weights by it would build an array shorter than the
    # positions `found[kept]` then indexes with.
    windows = [len(clicks[-history_k:]) for clicks in history["click_history"]]

    built: list[np.ndarray] = []
    for position, kept in enumerate(clicks.kept):
        window = {}
        for name, column in columns.items():
            values = None if column is None else column.iloc[position]
            window[name] = (
                None if values is None else np.asarray(values)[-history_k:]
            )
        found = weighting.weights(
            spec.scheme,
            spec.decay,
            windows[position],
            times=window["click_times"],
            read_times=window["click_read_times"],
            scroll=window["click_scroll"],
            at=stamps[position],
        )
        built.append(weighting.normalise(found[kept] if len(found) else found))
    return built


def rank_candidates(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
    pooling: str = retrieval.POOLING,
) -> pd.DataFrame:
    """Score each impression's own candidates. The harness's only entry here.

    bm25_index exposes the same function with the same signature, which is what
    lets the harness score both retrievers without knowing which it holds — and
    what stops a sweep handing the two of them different windows or different
    poolings while reporting one cell.
    """
    embeddings = embed.load(config)
    index = build(embeddings)

    wanted = set(behaviors["impression_id"])
    history = history[history["impression_id"].isin(wanted)]
    clicks = build_clicks(history, embeddings, history_k)

    # Looked up by id rather than zipped: the history frame is filtered from a
    # larger one and need not arrive in the behaviours frame's order, and a
    # positional pairing would score each impression against another's
    # candidates while looking entirely well-formed.
    candidates_of = dict(zip(behaviors["impression_id"], behaviors["candidate_ids"]))
    candidates = [candidates_of[i] for i in clicks.impression_ids]
    # The impression's own timestamp, which time decay measures back from. It
    # is the moment the ranking is served, so reading it is not the future.
    served = dict(zip(behaviors["impression_id"], behaviors["impression_time"]))
    at = pd.Series([served.get(i) for i in clicks.impression_ids])
    weights = click_weights(config, history, clicks, history_k, at)
    return index.score_candidates_pooled(clicks, candidates, pooling, weights)


@dataclass(frozen=True)
class Ranker:
    """Scores a supplied candidate list against a supplied catalogue.

    `rank_candidates` above is the same operation over the feature store; this
    one is handed the corpus, so the submission path can rank against the
    competition's own catalogue, which the feature store does not contain and
    which the stored embedding artifact does not cover either — embed.for_corpus
    is what closes that gap. bm25_index exposes the same pair of names, and that
    is all `predict` knows about either retriever.
    """

    index: Index
    embeddings: embed.Embeddings
    history_k: int
    config: DatasetConfig

    def rank(self, history: pd.DataFrame, candidates: list[list[str]]) -> pd.DataFrame:
        require_expressible()
        # Uniform keeps the original path rather than a weighted one with equal
        # weights. The two agree, but only one of them is the path every
        # recorded number came from, and MIND's submission is already ranked by
        # it -- see click_weights, which returns None for the same reason.
        if self.config.weighting.scheme == "uniform":
            queries, _ = build_user_vectors(history, self.embeddings, self.history_k)
        else:
            queries = weighted_user_vectors(
                history, self.embeddings, self.config, self.history_k
            )
        return self.index.score_candidates(queries, candidates)


def ranker(
    articles: pd.DataFrame,
    config: DatasetConfig,
    workdir: Path,
    history_k: int = retrieval.HISTORY_K,
) -> Ranker:
    """Vectors for `articles`, cached under `workdir`, in a flat index."""
    embeddings, report = embed.for_corpus(articles, config, workdir / embed.EMBED_DIR)
    if report["cached"]:
        print(f"    {report['articles']:,} article vectors loaded from {workdir}")
    else:
        print(
            f"    {report['from_artifact']:,} article vectors from the stored "
            f"artifact, {report['encoded']:,} encoded here, "
            f"{report['missing']:,} left at zero"
        )
    return Ranker(
        index=build(embeddings),
        embeddings=embeddings,
        history_k=history_k,
        config=config,
    )
