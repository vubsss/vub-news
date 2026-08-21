"""Two rankings into one, and what happens when one of them is empty.

The failure a hybrid has that its parents do not: a parent that returned
something meaningless — an empty BM25 query, a user with no usable clicks —
looks exactly like a parent that ranked the impression, and folding it in adds
noise with the authority of a retriever behind it.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import evaluate, hybrid
from pipeline.datasets import HybridSpec

RRF = HybridSpec(rule="rrf", k=60.0)
LINEAR = HybridSpec(rule="linear", alpha=0.5)


def ranked(rows):
    """rows: (impression_id, ranked_ids, scores) — best first, as the parents
    emit them."""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([r[0] for r in rows], dtype="string"),
            "ranked_ids": [list(r[1]) for r in rows],
            "scores": [list(r[2]) for r in rows],
        }
    )


def both(lex, sem):
    return {"bm25": lex, "ann": sem}


# --- the harness must not learn it exists -----------------------------------


def test_the_hybrid_is_a_registered_retriever_like_the_other_two():
    """An entry in RETRIEVERS and nothing else. The harness reaches a retriever
    only through `rank_candidates`, so a third one is a registry line."""
    import inspect

    assert "hybrid" in evaluate.RETRIEVERS
    for name in ("rank_candidates", "retrieve_corpus"):
        assert hasattr(hybrid, name)
    signature = inspect.signature(hybrid.rank_candidates).parameters
    parent = inspect.signature(evaluate.RETRIEVERS["bm25"].rank_candidates).parameters
    assert list(signature) == list(parent)


# --- fusion itself ----------------------------------------------------------


def test_rrf_prefers_the_article_both_parents_rank_well():
    """The point of fusing. `b` is 2nd for the lexical parent and 1st for the
    semantic one, and beats `a`, which the lexical parent puts 1st but the
    semantic one only 3rd.

    Note what this does *not* claim. `1/(k+r)` is convex, so two candidates
    with the same rank *sum* are separated in favour of the extreme pair — a
    candidate ranked first-and-last beats one ranked middle-and-middle, at
    every k. RRF rewards a better rank sum, not agreement as such.
    """
    lex = ranked([("i1", ["a", "b", "c"], [9.0, 5.0, 1.0])])
    sem = ranked([("i1", ["b", "c", "a"], [9.0, 5.0, 1.0])])

    out = hybrid.fuse(both(lex, sem), RRF)

    assert out["ranked_ids"][0] == ["b", "a", "c"]


def test_k_decides_between_a_parents_favourite_and_a_broad_agreement():
    """What the sweep is choosing. Small k makes `1/(k+rank)` steep, so a rank
    of 1 outweighs almost anything the other parent says; large k flattens it
    toward ordering by the sum of the ranks.

    `x` is ranked 1st by the lexical parent and 10th by the semantic one; `y`
    is 4th and 5th, the better rank sum. Same two lists, opposite winners at
    k=0.5 and k=60 — which is the whole reason k is swept rather than left at
    its conventional default.
    """
    filler = [f"f{i}" for i in range(10)]
    lex_order = ["x", filler[0], filler[1], "y", *filler[2:]]
    # 12 candidates, so the favourite is not ranked last by the other parent —
    # a candidate ranked first-and-last is tied by its mirror at every k, and
    # the comparison would be undecidable rather than merely close.
    sem_order = [
        filler[9], filler[8], filler[7], filler[6], "y", filler[5],
        filler[4], filler[3], filler[2], "x", filler[1], filler[0],
    ]
    scores = [12.0 - i for i in range(12)]
    parents = both(
        ranked([("i1", lex_order, scores)]),
        ranked([("i1", sem_order, scores)]),
    )

    steep = hybrid.fuse(parents, HybridSpec(rule="rrf", k=0.5))
    flat = hybrid.fuse(parents, HybridSpec(rule="rrf", k=60.0))

    assert steep["ranked_ids"][0][0] == "x", "a steep curve keeps the favourite"
    assert flat["ranked_ids"][0][0] == "y", "a flat curve prefers the rank sum"


def test_rrf_ignores_how_far_apart_the_scores_were():
    """Why it is the default: BM25's unbounded sums and a cosine in [-1, 1]
    never have to be reconciled, because only the order survives."""
    lex = ranked([("i1", ["a", "b"], [900.0, 1.0])])
    same_order_tiny_gap = ranked([("i1", ["a", "b"], [0.02, 0.01])])
    sem = ranked([("i1", ["b", "a"], [0.9, 0.1])])

    wide = hybrid.fuse(both(lex, sem), RRF)
    narrow = hybrid.fuse(both(same_order_tiny_gap, sem), RRF)

    assert wide["ranked_ids"][0] == narrow["ranked_ids"][0]


def test_alpha_slides_the_linear_rule_between_its_parents():
    """alpha weights the lexical side, so alpha=1 is BM25's order and alpha=0
    is the semantic one — which is what makes a per-dataset alpha reportable."""
    lex = ranked([("i1", ["a", "b"], [9.0, 1.0])])
    sem = ranked([("i1", ["b", "a"], [9.0, 1.0])])

    lexical = hybrid.fuse(both(lex, sem), HybridSpec(rule="linear", alpha=1.0))
    semantic = hybrid.fuse(both(lex, sem), HybridSpec(rule="linear", alpha=0.0))

    assert lexical["ranked_ids"][0] == ["a", "b"]
    assert semantic["ranked_ids"][0] == ["b", "a"]


# --- a parent with nothing to say -------------------------------------------


def test_a_flat_parent_is_skipped_rather_than_folded_in():
    """An empty BM25 query scores every candidate the same, so its "ranking" is
    the order the candidates arrived in. RRF cannot tell that from a real
    ranking and would read the permutation as evidence, so the hybrid has to
    drop it and let the other parent decide alone."""
    silent = ranked([("i1", ["a", "b", "c"], [0.0, 0.0, 0.0])])
    sem = ranked([("i1", ["c", "b", "a"], [9.0, 5.0, 1.0])])

    out = hybrid.fuse(both(silent, sem), RRF)

    assert out["ranked_ids"][0] == ["c", "b", "a"], "the semantic order stands"


def test_the_linear_rule_degrades_to_the_working_parent_too():
    """Same requirement, different arithmetic: a flat parent normalises to a
    constant, and the weights are re-spread over the parents that spoke so the
    survivor is not shrunk toward nothing."""
    silent = ranked([("i1", ["a", "b"], [3.0, 3.0])])
    sem = ranked([("i1", ["b", "a"], [9.0, 1.0])])

    out = hybrid.fuse(both(silent, sem), LINEAR)

    assert out["ranked_ids"][0] == ["b", "a"]
    assert out["scores"][0][0] == pytest.approx(1.0), (
        "the surviving parent's best candidate keeps its full weight"
    )


def test_a_cold_user_neither_parent_could_rank_keeps_the_given_order():
    """No history at all: both parents score everything zero. The order the
    competition supplied is the honest answer, and the scores say so rather
    than inventing a permutation."""
    silent = ranked([("i1", ["a", "b", "c"], [0.0, 0.0, 0.0])])

    for spec in (RRF, LINEAR):
        out = hybrid.fuse(both(silent, silent), spec)

        assert out["ranked_ids"][0] == ["a", "b", "c"]
        assert out["scores"][0] == [0.0, 0.0, 0.0]


# --- alignment --------------------------------------------------------------


def test_impressions_are_matched_by_id_not_by_position():
    """The parents filter the same history frame and need not emit their rows
    in the same order. A positional pairing would fuse one impression's lexical
    ranking with another's semantic one and look entirely well-formed."""
    lex = ranked([("i1", ["a", "b"], [9.0, 1.0]), ("i2", ["c", "d"], [9.0, 1.0])])
    sem = ranked([("i2", ["d", "c"], [9.0, 1.0]), ("i1", ["b", "a"], [9.0, 1.0])])

    out = hybrid.fuse(both(lex, sem), RRF).set_index("impression_id")

    assert set(out.at["i1", "ranked_ids"]) == {"a", "b"}
    assert set(out.at["i2", "ranked_ids"]) == {"c", "d"}


def test_parents_that_ranked_different_impressions_are_refused():
    """Fusing what the two parents happen to share would quietly report a
    smaller population than the split has."""
    lex = ranked([("i1", ["a"], [1.0]), ("i2", ["b"], [1.0])])
    sem = ranked([("i1", ["a"], [1.0])])

    with pytest.raises(hybrid.HybridError, match="same set"):
        hybrid.fuse(both(lex, sem), RRF)


def test_an_unknown_rule_is_refused_rather_than_defaulted():
    lex = ranked([("i1", ["a"], [1.0])])

    with pytest.raises(hybrid.HybridError, match="unknown fusion rule"):
        hybrid.fuse(both(lex, lex), HybridSpec(rule="borda"))
