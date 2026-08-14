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

import faiss
import numpy as np
import pandas as pd

from pipeline import embed, retrieval
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


def rank_candidates(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
) -> pd.DataFrame:
    """Score each impression's own candidates. The harness's only entry here.

    bm25_index exposes the same function with the same signature, which is what
    lets the harness score both retrievers without knowing which it holds.
    """
    embeddings = embed.load(config)
    index = build(embeddings)

    wanted = set(behaviors["impression_id"])
    clicks = history[history["impression_id"].isin(wanted)]
    queries, _ = build_user_vectors(clicks, embeddings, history_k)

    # Looked up by id rather than zipped: the history frame is filtered from a
    # larger one and need not arrive in the behaviours frame's order, and a
    # positional pairing would score each impression against another's
    # candidates while looking entirely well-formed.
    candidates_of = dict(zip(behaviors["impression_id"], behaviors["candidate_ids"]))
    candidates = [candidates_of[i] for i in queries.impression_ids]
    return index.score_candidates(queries, candidates)
