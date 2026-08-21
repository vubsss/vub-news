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
#
# Chosen on tune in phase 4, jointly with the pooling, and the largest single
# effect the project has measured: k=10 to k=80 is worth +0.0182 AUC for MIND's
# semantic retriever and +0.0263 for EB-NeRD's, paired by impression, against
# +0.001 for the whole of phase 3's lexical parameter grid. One constant rather
# than a registry field because both datasets and both retrievers chose the
# same value, monotonically, with no cell of either grid dissenting.
#
# It is the *largest window tested*, and every marginal was still rising there,
# so this is a boundary rather than an optimum -- the same defect the sweep it
# replaces had. Raising it past sources.ENGAGEMENT_WINDOW needs a re-ingest.
HISTORY_K = 80

# How an impression's per-click similarities become one score per candidate.
#
#   mean  the profile every retriever shipped with: the user is the average of
#         what they read. Lossy in a known way -- someone who reads football
#         and recipes averages to someone who reads neither.
#   max   does this candidate look like *any* recent click, which survives a
#         history that mixes unrelated interests.
#   last  does it follow from the most recent click, which is the session
#         signal rather than the standing interest.
#
# The three are one operation with three aggregators, over the same last-K
# clicks: score = AGG_i (candidate . click_i). That matters for `mean`, where
# the mean of the dot products is the dot product with the mean vector, so the
# pooled-vector path and this one rank identically and corpus retrieval can
# keep using the cheap one.
POOLINGS = ("mean", "max", "last")
POOLING = "mean"

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
