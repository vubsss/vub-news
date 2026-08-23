"""How much each past click counts, and which datasets can ask.

Every scheme reads a column describing a click that already happened, so every
one is available at serving time. What separates them is what MIND has: bare
ids, which support position but not time and not engagement.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pipeline import ann_index, bm25_index, embed, weighting
from pipeline.datasets import DATASETS, WeightingSpec

MIND, EBNERD = DATASETS["mind"], DATASETS["ebnerd"]


def weighted(config, **changes):
    return dataclasses.replace(config, weighting=WeightingSpec(**changes))


# --- which dataset can express what -----------------------------------------


def test_availability_is_read_off_the_columns_the_dataset_has():
    """Not a second list to keep true: a dataset supports a scheme exactly when
    it carries the column that scheme reads."""
    assert weighting.available(MIND) == ("uniform", "position")
    assert weighting.available(EBNERD) == (
        "uniform", "position", "time", "engagement",
    )


def test_asking_mind_for_time_decay_fails_loudly():
    """The alternative is a run that falls back to uniform and reports its
    numbers under the name of a scheme that never ran."""
    with pytest.raises(weighting.WeightingError, match="carries no such column"):
        weighting.check(MIND, "time")


def test_a_window_past_the_stored_arrays_is_refused_not_clipped():
    """The arrays are truncated at ingest, so past that point `times[-k:]` and
    `clicks[-k:]` are different lengths and every weight lands on the wrong
    click. Clipping would silently weight a 160-click window as a 100-click
    one and report it under the larger number."""
    from pipeline import sources

    beyond = sources.ENGAGEMENT_WINDOW + 1
    with pytest.raises(weighting.WeightingError, match="cannot reach back"):
        weighting.check(EBNERD, "time", beyond)

    # position needs no stored array, so it reaches as far as the clicks do.
    weighting.check(EBNERD, "position", beyond)


def test_an_unknown_scheme_is_refused():
    with pytest.raises(weighting.WeightingError, match="unknown weighting"):
        weighting.check(EBNERD, "exponential-ish")


# --- the schemes themselves -------------------------------------------------


def test_position_decay_counts_the_newest_click_fully():
    """The window is a suffix, so the most recent click is the *last* entry. A
    scheme that read it the other way round would weight the stalest click most
    and still look entirely plausible."""
    found = weighting.weights("position", 0.5, 3)

    assert list(found) == [0.25, 0.5, 1.0]


def test_uniform_is_flat_whatever_the_decay():
    assert list(weighting.weights("uniform", 0.1, 3)) == [1.0, 1.0, 1.0]


def test_time_decay_measures_back_from_the_impression_not_the_last_click():
    """News decays in real time. Two clicks a day apart weight by when they
    happened relative to the moment the ranking is served."""
    at = np.datetime64("2023-05-10T00:00:00")
    times = np.array(
        ["2023-05-08T00:00:00", "2023-05-09T00:00:00"], dtype="datetime64[us]"
    )

    found = weighting.weights("time", 24.0, 2, times=times, at=at)

    assert found[1] > found[0], "the more recent click counts for more"
    assert found[0] == pytest.approx(np.exp(-2.0), rel=1e-6)
    assert found[1] == pytest.approx(np.exp(-1.0), rel=1e-6)


def test_a_click_stamped_after_the_impression_is_treated_as_just_now():
    """A clock artifact rather than the future leaking in — but it must not
    come back as a weight above 1, which an unclamped exp would give it."""
    at = np.datetime64("2023-05-10T00:00:00")
    times = np.array(["2023-05-11T00:00:00"], dtype="datetime64[us]")

    assert weighting.weights("time", 24.0, 1, times=times, at=at)[0] == 1.0


def test_engagement_treats_a_missing_scroll_as_neutral_not_as_zero():
    """scroll_percentage is 10.8% null *inside* its arrays on the real data.
    The click still happened; only the measurement is absent, so it must not
    zero the weight."""
    read = np.array([100.0, 100.0])
    scroll = np.array([np.nan, 50.0])

    found = weighting.weights(
        "engagement", 1.0, 2, read_times=read, scroll=scroll
    )

    assert found[0] > found[1], "an unmeasured scroll is not a shallow one"
    assert found[0] == pytest.approx(np.log1p(100.0))


def test_engagement_damps_the_read_time_tail():
    """read_time is clipped at 1800 seconds with a median of 14, so raw it
    would let one long read outvote a hundred ordinary ones."""
    found = weighting.weights(
        "engagement", 1.0, 2,
        read_times=np.array([14.0, 1800.0]), scroll=np.array([100.0, 100.0]),
    )

    assert found[1] / found[0] < 3, "128x in seconds is under 3x in weight"


def test_clicks_nobody_engaged_with_fall_back_to_uniform():
    """A weighted mean over all-zero weights divides by zero. An impression
    whose every click scores nothing is one this scheme has nothing to say
    about, not one whose profile is empty."""
    found = weighting.weights(
        "engagement", 1.0, 2,
        read_times=np.array([0.0, 0.0]), scroll=np.array([0.0, 0.0]),
    )

    assert list(found) == [1.0, 1.0]


def test_normalise_leaves_a_dead_impression_uniform_rather_than_nan():
    assert list(weighting.normalise(np.zeros(2))) == [0.5, 0.5]


# --- the alignment trap -----------------------------------------------------


def catalogue():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    return embed.Embeddings(
        article_ids=np.array(["a", "b"], dtype=object),
        vectors=embed.normalise(matrix),
    )


def test_weights_are_subset_to_the_clicks_that_survived_the_catalogue():
    """`build_clicks` drops a click the catalogue has no vector for, so a
    weight computed over the whole window would land on the wrong click from
    that point on — and look entirely well-formed doing it. `kept` records the
    window positions that survived, which is what makes the pairing right.
    """
    history = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "click_history": [["a", "missing", "b"]],
        }
    )

    clicks = ann_index.build_clicks(history, catalogue(), history_k=10)

    assert list(clicks.kept[0]) == [0, 2], "the gap is recorded, not closed up"
    found = weighting.weights("position", 0.5, 3)
    assert list(found[clicks.kept[0]]) == [0.25, 1.0], (
        "'b' keeps the weight of the newest click, not of the one it replaced"
    )


# --- uniform stays exactly the path every recorded number came from ---------


def test_a_uniform_registry_builds_no_weights_at_all():
    """Not "weights that happen to be equal" — None, so the unweighted branch
    runs and the numbers on record cannot move under a refactor."""
    history = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "click_history": [["a", "b"]],
        }
    )
    clicks = ann_index.build_clicks(history, catalogue(), history_k=10)

    assert ann_index.click_weights(MIND, history, clicks, 10) is None
    assert bm25_index.query_weights(MIND, history, 10, {}) is None


# --- the lexical side -------------------------------------------------------


def test_repetition_puts_the_newest_click_at_the_cap():
    """Scaled by the largest weight, so the newest click lands on MAX_REPEAT
    whatever the decay is and the query's shape depends on the ratios."""
    found = weighting.weights("position", 0.5, 3)

    repeats = bm25_index.repeats_for(found)

    assert repeats[-1] == bm25_index.MAX_REPEAT
    assert list(repeats) == [1, 2, 3]


def test_a_click_decayed_below_half_a_step_drops_out_of_the_query():
    """Which is what a decay is for. Keeping every click at a floor of one
    would make a steep decay indistinguishable from a shallow one."""
    found = weighting.weights("position", 0.1, 3)

    assert list(bm25_index.repeats_for(found)) == [0, 0, 3]


def test_the_weighted_query_repeats_a_recent_title_more_than_an_old_one():
    """The only mechanism a bag of terms has: recency as term frequency, which
    raises those terms before BM25 saturates them."""
    articles = pd.DataFrame(
        {
            "article_id": pd.Series(["old", "new"], dtype="string"),
            "title": pd.Series(["sharks", "bears"], dtype="string"),
            "abstract": pd.Series(["", ""], dtype="string"),
        }
    )
    history = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "click_history": [["old", "new"]],
        }
    )

    queries, _ = bm25_index.build_queries(
        history,
        articles,
        dataclasses.replace(MIND, lexical=dataclasses.replace(MIND.lexical, query_abstract=False)),
        history_k=10,
        weights=[weighting.weights("position", 0.5, 2)],
    )
    terms = queries["query"][0].split()

    assert terms.count("bears") > terms.count("sharks")


# --- the submission path cannot silently disagree with the measurement ------


def test_the_guard_no_longer_turns_on_the_weighting_scheme():
    """`predict` streams the columns each scheme reads, so none of them is
    measurable on the feature store and unreproducible on the submission any
    more. That gap used to be what this guard was about, and it was a real one:
    the file would have been well-formed, correctly ordered, and produced by a
    different model from the one every reported number came from. What decides
    now is the aggregator, which no scheme changes."""
    assert weighting.columns_for("engagement") == (
        "click_read_times", "click_scroll",
    )
    ann_index.require_expressible()


def test_the_submission_path_refuses_a_pooling_it_cannot_express():
    """What is still inexpressible is an aggregator that does not collapse into
    one vector. `max` asks which single click matches best, which is not a dot
    product with any vector, and the submission can only afford one vector per
    impression across 13.5M of them."""
    for pooling in ("max", "last"):
        with pytest.raises(weighting.WeightingError, match="cannot express"):
            ann_index.require_expressible(pooling)


def test_the_weighted_mean_is_what_the_guard_lets_through():
    """The identity the submission path rests on holds for `mean` alone."""
    ann_index.require_expressible("mean")


# --- the two paths must produce the same ranking ----------------------------


def engaged_history():
    """One impression whose three clicks were read to very different depths."""
    return pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "user_id": ["u1"],
            "impression_time": pd.to_datetime(["2023-06-02 07:00"]),
            "click_history": [["a", "b", "c"]],
            "click_read_times": [np.array([600.0, 1.0, 1.0])],
            "click_scroll": [np.array([100.0, 5.0, 5.0])],
        }
    )


def spread_catalogue():
    """Four articles, three of them clicked, pointing in different directions."""
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],   # a, the one that was read properly
            [0.0, 1.0, 0.0],   # b
            [0.0, 1.0, 0.0],   # c, b again
            [0.9, 0.4, 0.0],   # d, a candidate leaning toward a
            [0.1, 1.0, 0.0],   # e, a candidate leaning toward b and c
        ],
        dtype="float32",
    )
    return embed.Embeddings(
        article_ids=np.array(["a", "b", "c", "d", "e"], dtype=object),
        vectors=embed.normalise(matrix),
    )


def test_the_submission_path_ranks_as_the_feature_store_path_does():
    """The whole point of the identity. `score_candidates_pooled` weights every
    candidate against every click and then pools; the submission folds the
    weights into one vector first because it cannot afford the matmul 13.5M
    times. Same ranking, or the file does not describe the measured model."""
    config = weighted(EBNERD, scheme="engagement")
    history, embeddings = engaged_history(), spread_catalogue()
    index = ann_index.build(embeddings)
    candidates = [["d", "e"]]

    clicks = ann_index.build_clicks(history, embeddings, history_k=10)
    weights = ann_index.click_weights(config, history, clicks, 10)
    slow = index.score_candidates_pooled(clicks, candidates, "mean", weights)

    queries = ann_index.weighted_user_vectors(history, embeddings, config, 10)
    fast = index.score_candidates(queries, candidates)

    assert list(fast["ranked_ids"][0]) == list(slow["ranked_ids"][0])


def test_engagement_actually_moves_the_submissions_ranking():
    """Guards the test above against being vacuously true, and guards the
    wiring: this goes through `Ranker.rank`, which is what `predict` calls, so
    a rank() that stopped consulting the weighting fails here. Two of the three
    clicks were abandoned; unweighted they outvote the one that was read."""
    history, embeddings = engaged_history(), spread_catalogue()
    index = ann_index.build(embeddings)
    candidates = [["d", "e"]]

    def ranked(scheme):
        found = ann_index.Ranker(
            index=index,
            embeddings=embeddings,
            history_k=10,
            config=weighted(EBNERD, scheme=scheme),
        ).rank(history, candidates)
        return list(found["ranked_ids"][0])

    assert ranked("engagement") == ["d", "e"], "the read click carries it"
    assert ranked("uniform") == ["e", "d"], "two abandoned ones outvote it"


def test_the_lexical_submission_path_weights_its_query_too(tmp_path):
    """The sibling defect, and the quieter one. `Ranker.rank` used to call
    `build_queries` with no weights at all while the registry said engagement,
    so the submission ranked by a different profile from the one every reported
    number came from — and unlike the semantic side it did not refuse, it just
    did it. Nothing in a well-formed, correctly ordered file could show it."""
    corpus = pd.DataFrame(
        {
            "article_id": pd.Series(["a", "b", "c"], dtype="string"),
            "title": pd.Series(["kage", "fodbold", "fodbold"], dtype="string"),
            "abstract": pd.Series(["", "", ""], dtype="string"),
            "lexical_text": pd.Series(["kage", "fodbold", "fodbold"], dtype="string"),
        }
    )
    chunk = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "impression_time": pd.to_datetime(["2023-06-02 07:00"]),
            "click_history": [["a", "b", "c"]],
            "click_read_times": [np.array([600.0, 1.0, 1.0])],
            "click_scroll": [np.array([100.0, 5.0, 5.0])],
        }
    )
    candidates = [["a", "b"]]

    engaged = bm25_index.ranker(
        corpus, weighted(EBNERD, scheme="engagement"), tmp_path / "engaged"
    ).rank(chunk, candidates)
    flat = bm25_index.ranker(
        corpus, weighted(EBNERD, scheme="uniform"), tmp_path / "flat"
    ).rank(chunk, candidates)

    assert list(engaged["ranked_ids"][0]) == ["a", "b"], (
        "the click that was read to the end carries the query"
    )
    assert list(flat["ranked_ids"][0]) == ["b", "a"], (
        "unweighted, the two abandoned clicks outvote it two to one"
    )
