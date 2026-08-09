"""BM25 lexical retrieval: click history -> query -> ranked article ids.

Both the documents and the queries go through preprocess.cleaner for this
dataset's language, so the two sides of the index share a vocabulary. Nothing
here knows which dataset it holds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

from pipeline import preprocess, retrieval
from pipeline.datasets import DatasetConfig

# SPEC's BM25 parameters. Language-agnostic, so they are not registry entries.
K1, B = 1.5, 0.75

# The article ids, saved next to the bm25s index, which does not store them.
ARTICLE_IDS = "article_ids.npy"


def build_queries(
    history: pd.DataFrame,
    articles: pd.DataFrame,
    config: DatasetConfig,
    history_k: int = retrieval.HISTORY_K,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One query per impression: the last history_k clicked titles, cleaned.

    Titles rather than the full lexical_text, because a query built from
    abstracts too drowns the terms that identify what the user actually reads.
    A user with no history has no query, which is reported rather than filled
    in — see `retrieve`.
    """
    clean = preprocess.cleaner(config)
    titles = dict(zip(articles["article_id"], articles["title"].fillna("")))

    query = [
        clean(" ".join(titles.get(article_id, "") for article_id in clicks[-history_k:]))
        for clicks in history["click_history"]
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


def build(articles: pd.DataFrame, config: DatasetConfig) -> Index:
    """Index the catalogue's lexical_text, which preprocess already cleaned.

    Tokenising is a plain split rather than bm25s' own tokenizer: that one
    would apply English stopwords and its own token pattern on top of the
    language-correct cleaning preprocess did, which for Danish means removing
    English words and nothing else. Queries are split the same way.
    """
    text = articles["lexical_text"].fillna("")
    bm25 = bm25s.BM25(k1=K1, b=B)
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
    history = pd.read_parquet(config.feature_store_dir / "history.parquet")

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
