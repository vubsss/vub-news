"""The shape both retrievers emit, and how it is scored.

These sit apart from either retriever on purpose: ticket 8's semantic numbers
are only meaningful beside ticket 6's lexical ones if a single definition of
recall produced both.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import retrieval


def behaviors(rows):
    """rows: (impression_id, candidate_ids, labels)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
            "candidate_ids": [list(row[1]) for row in rows],
            "labels": [list(row[2]) for row in rows],
        }
    )


def ranking(rows):
    """rows: (impression_id, ranked_ids)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
            "ranked_ids": [list(row[1]) for row in rows],
        }
    )


def test_recall_at_k_matches_a_hand_computed_example():
    """Worked by hand, not recomputed the way the code does it:

    dev-1 clicked a1 and a3 and its ranking is [a3, a1, a9] — 1 of 2 at
    depth 1, 2 of 2 at depth 2. dev-2 clicked a5 and its ranking is
    [a9, a5, a7] — 0 of 1 at depth 1, 1 of 1 at depth 2. Averaging the
    per-impression fractions gives 0.25 and 1.0. dev-3 has no click at all,
    so its recall is 0/0 and it is counted rather than averaged in as a zero.
    """
    truth = behaviors(
        [
            ("dev-1", ["a1", "a2", "a3"], [1, 0, 1]),
            ("dev-2", ["a5", "a7"], [1, 0]),
            ("dev-3", ["a1"], [0]),
        ]
    )
    ranked = ranking(
        [
            ("dev-1", ["a3", "a1", "a9"]),
            ("dev-2", ["a9", "a5", "a7"]),
            ("dev-3", ["a1", "a2", "a3"]),
        ]
    )

    got = retrieval.recall_at_k(ranked, truth, depths=(1, 2))

    assert got["recall@1"] == pytest.approx(0.25)
    assert got["recall@2"] == pytest.approx(1.0)
    assert got["scored"] == 2
    assert got["no_positive"] == 1


def test_a_retrieved_id_from_outside_the_corpus_is_an_error():
    """The guard ticket 6 asks to be asserted in code. Both retrievers map index
    positions back through article_ids, so structurally this cannot happen —
    but if the corpus and the index are ever built from different frames the
    symptom is a recall figure that is quietly wrong rather than a crash, and
    a wrong number that looks fine is the one failure this project cannot
    afford. The stray id has to be named, so the message points at the frame
    that disagreed."""
    ranked = ranking([("dev-1", ["a1", "ghost"]), ("dev-2", ["a2"])])

    with pytest.raises(retrieval.CorpusError, match="ghost"):
        retrieval.check_within_corpus(ranked, np.array(["a1", "a2"], dtype=object))
