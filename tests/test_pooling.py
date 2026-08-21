"""How an impression's clicks become one score per candidate.

A mean over the last K clicks is lossy in a known way: a user who reads
football and recipes averages to someone who reads neither. `max` asks whether
a candidate looks like *any* recent click and `last` asks whether it follows
from the most recent one, which is the short-term signal the long-term profile
smooths away.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import ann_index, bm25_index, embed, retrieval


def embeddings(rows: dict[str, list[float]]) -> embed.Embeddings:
    """Unit-length article vectors, so an inner product is a cosine."""
    ids = list(rows)
    matrix = np.array([rows[article] for article in ids], dtype="float32")
    return embed.Embeddings(
        article_ids=np.array(ids, dtype=object),
        vectors=embed.normalise(matrix),
    )


def history(clicks: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "click_history": [clicks],
        }
    )


# Two interests that share nothing, and one article halfway between them.
FOOTBALL, RECIPES = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
CATALOGUE = {
    "football": FOOTBALL,
    "recipes": RECIPES,
    "both": [1.0, 1.0, 0.0],
    "neither": [0.0, 0.0, 1.0],
}


def ranked(clicks, candidates, pooling, catalogue=CATALOGUE, history_k=10):
    store = embeddings(catalogue)
    index = ann_index.build(store)
    seen = ann_index.build_clicks(history(clicks), store, history_k)
    return index.score_candidates_pooled(seen, [candidates], pooling)


# --- mean is the profile that already shipped -------------------------------


def test_mean_pooling_ranks_exactly_as_the_pooled_vector_does():
    """The claim that lets corpus retrieval keep its single pooled vector: the
    mean of the dot products is the dot product with the mean, so the two paths
    order an impression identically. The pooled path rescales to unit length,
    which changes the scores and not the ranking — so this compares the order.
    """
    clicks, candidates = ["football", "recipes"], ["football", "both", "neither"]
    store = embeddings(CATALOGUE)
    index = ann_index.build(store)

    queries, _ = ann_index.build_user_vectors(history(clicks), store)
    pooled = index.score_candidates(queries, [candidates])
    unpooled = ranked(clicks, candidates, "mean")

    assert pooled["ranked_ids"][0] == unpooled["ranked_ids"][0]


# --- what the other two are for ---------------------------------------------


def test_max_prefers_a_candidate_matching_one_interest_over_their_average():
    """The reason `max` is in the grid. A user who reads football and recipes
    means to their midpoint, so the mean scores `both` — an article about
    neither in particular — above a real football story. `max` asks whether the
    candidate looks like *any* click and puts football first.
    """
    clicks, candidates = ["football", "recipes"], ["football", "both"]

    assert ranked(clicks, candidates, "mean")["ranked_ids"][0][0] == "both"
    assert ranked(clicks, candidates, "max")["ranked_ids"][0][0] == "football"


def test_last_reads_the_most_recent_click_and_nothing_before_it():
    """The session signal. Three clicks on football and a turn to recipes: the
    mean still calls the user a football reader, `last` does not."""
    clicks = ["football", "football", "football", "recipes"]
    candidates = ["football", "recipes"]

    assert ranked(clicks, candidates, "mean")["ranked_ids"][0][0] == "football"
    assert ranked(clicks, candidates, "last")["ranked_ids"][0][0] == "recipes"


def test_the_window_is_a_suffix_so_last_is_the_newest_click_in_it():
    """`last` takes the final column of the click matrix, which is only the
    most recent click if the window kept the *end* of the history. A window
    that kept the head would make `last` the oldest click of the window."""
    clicks = ["recipes", "recipes", "football"]
    candidates = ["football", "recipes"]

    assert ranked(clicks, candidates, "last", history_k=2)["ranked_ids"][0][0] == (
        "football"
    )


# --- the cases where there is nothing to pool -------------------------------


def test_a_cold_user_scores_every_candidate_zero_under_every_pooling():
    """No clicks is no profile, whatever the aggregator. The stable sort then
    returns the candidates in the order they arrived, which is the honest
    answer when there is nothing to rank on."""
    for pooling in retrieval.POOLINGS:
        frame = ranked([], ["football", "recipes"], pooling)

        assert frame["scores"][0] == [0.0, 0.0]
        assert frame["ranked_ids"][0] == ["football", "recipes"]


def test_a_click_the_catalogue_cannot_place_is_dropped_not_zeroed():
    """The mean does this already — a click with no vector makes the user one
    we know less about, not one with different interests — and an aggregator
    over the clicks has to agree, or `max` would take the max against a zero
    row and `last` could read one."""
    clicks = ["football", "not-in-the-catalogue"]
    candidates = ["football", "neither"]

    assert ranked(clicks, candidates, "last")["ranked_ids"][0][0] == "football"


def test_an_unknown_pooling_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="unknown pooling"):
        ranked(["football"], ["football"], "median")


# --- the parameter reaches both retrievers ----------------------------------


def test_both_retrievers_take_pooling_through_the_same_call():
    """The property that stops a sweep handing the two of them different
    aggregators while reporting one cell, exactly as `history_k` already
    cannot drift between them."""
    import inspect

    for module in (ann_index, bm25_index):
        parameters = inspect.signature(module.rank_candidates).parameters
        assert "pooling" in parameters, module.__name__
        assert parameters["pooling"].default == retrieval.POOLING


def test_bm25_reads_the_whole_window_for_mean_and_one_click_for_last():
    """A bag of terms has no aggregator: what pooling means lexically is how
    much of the history goes into the bag."""
    assert bm25_index.window_for(20, "mean") == 20
    assert bm25_index.window_for(20, "last") == 1


def test_bm25_refuses_max_rather_than_silently_scoring_the_mean():
    """`max` has no lexical form. Returning the mean query under the label
    `max` would put a row in a sweep whose name did not describe what ran."""
    with pytest.raises(ValueError, match="cannot express 'max'"):
        bm25_index.window_for(20, "max")
