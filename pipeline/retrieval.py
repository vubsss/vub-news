"""What the two retrievers share: the shape they emit, and how it is scored.

BM25 and the semantic index are interchangeable by design — ticket 8's numbers
are only meaningful next to ticket 6's if both are measured the same way — so
the ranked-candidates shape, the window the query is built from, the depths
reported and the recall definition live here rather than inside either one.
Nothing in this module knows which retriever, or which dataset, it is serving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The click-history window a query is built from. The same for both
# retrievers, or their recall figures are answering different questions.
# Swept in ticket 12.
HISTORY_K = 10

# Retrieval depths reported. Retrieval runs once at the deepest; the shallower
# figures are prefixes of the same ranking.
DEPTHS = (50, 100, 200)

# The split retrieval is reported on. Test is held back for the final run.
VALIDATION = "validation"


class CorpusError(RuntimeError):
    """Retrieval returned an article the corpus does not contain."""


def recall_at_k(
    ranked: pd.DataFrame, behaviors: pd.DataFrame, depths: tuple[int, ...]
) -> dict[str, float]:
    """Mean per-impression fraction of clicked articles found in the top-K.

    Averaged per impression rather than pooled over all clicks, so every
    impression counts once whatever its candidate list looks like — which is
    what makes the cold and warm slices of ticket 10 comparable to each other
    and the lexical and semantic figures comparable to one another.

    An impression with no clicked article has a recall of 0/0. It is counted
    and left out of the mean; scoring it as zero would drag every figure down
    by the share of such impressions rather than by anything retrieval did.
    """
    truth = behaviors[["impression_id", "candidate_ids", "labels"]].merge(
        ranked[["impression_id", "ranked_ids"]], on="impression_id"
    )

    found: dict[int, list[float]] = {depth: [] for depth in depths}
    no_positive = 0
    for _, candidates, labels, ids in truth.itertuples(index=False):
        clicked = {
            candidate
            for candidate, label in zip(candidates, labels, strict=True)
            if label
        }
        if not clicked:
            no_positive += 1
            continue
        for depth in depths:
            hits = clicked & set(ids[:depth])
            found[depth].append(len(hits) / len(clicked))

    report: dict[str, float] = {
        "scored": len(truth) - no_positive,
        "no_positive": no_positive,
    }
    for depth in depths:
        scores = found[depth]
        report[f"recall@{depth}"] = sum(scores) / len(scores) if scores else 0.0
    return report


def check_within_corpus(ranked: pd.DataFrame, article_ids: np.ndarray) -> None:
    """No retrieved article may come from outside the corpus.

    Structurally it cannot — retrieved ids are positions into article_ids —
    but that is exactly the kind of guarantee that quietly stops holding when
    the corpus and the index are built from different frames, and the symptom
    would be a recall figure that is wrong rather than an error.
    """
    corpus = set(article_ids)
    seen: set[str] = set()
    for ids in ranked["ranked_ids"]:
        seen.update(ids)
    stray = seen - corpus
    if stray:
        raise CorpusError(
            f"{len(stray)} retrieved article(s) are not in the corpus, "
            f"e.g. {sorted(stray)[:5]}"
        )
