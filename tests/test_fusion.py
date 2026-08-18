"""The fusion retriever: what it refuses, and that its ranking is a ranking.

The parts worth a test here are not the boosting — sklearn's own tests cover
that — but the seams: that the harness sees the same shape it sees from the
other two, that the serving variant is actually restricted to serving
features, and that a submission cannot be built from a model that reads a
column a competition test file does not have.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import evaluate, features, fusion


def test_both_variants_are_registered_and_differ_only_in_their_features():
    assert evaluate.RETRIEVERS["fusion"].columns == features.ALL
    assert evaluate.RETRIEVERS["fusion-serving"].columns == features.SERVING
    for name in ("fusion", "fusion-serving"):
        entry = evaluate.RETRIEVERS[name]
        # The three functions the harness, the sweep and the submission call.
        assert callable(entry.rank_candidates)
        assert callable(entry.retrieve_corpus)
        assert callable(entry.ranker)


def test_a_submission_refuses_the_click_reading_variant(tmp_path):
    """`fusion` reads how often a candidate was clicked before the impression,
    and a competition test file ships candidates and no clicks. Refused with
    that explanation rather than scoring a column of zeros under the name of
    the model the offline numbers were measured on."""
    from pipeline.datasets import DATASETS

    with pytest.raises(fusion.NotTrainedError, match="no clicks to count"):
        fusion.ranker(
            pd.DataFrame({"article_id": []}),
            DATASETS["mind"],
            tmp_path,
            columns=features.ALL,
            variant="fusion",
        )


def test_the_serving_counter_is_fitted_without_labels():
    """Not "the click columns are dropped afterwards" — the counter is fitted
    from a frame with no labels in it, so it has nothing to leak."""
    from pipeline.datasets import DATASETS

    behaviors = pd.DataFrame(
        {
            "impression_id": ["i1", "i2"],
            "impression_time": pd.to_datetime(["2023-05-24 07:00", "2023-05-24 09:00"]),
            "candidate_ids": [["a1", "a2"], ["a1"]],
            "labels": [[1, 0], [0]],
        }
    )
    serving = fusion.counters(DATASETS["mind"], behaviors, features.SERVING)
    full = fusion.counters(DATASETS["mind"], behaviors, features.ALL)

    assert serving.clicks is None
    assert full.clicks is not None


def test_order_sorts_each_impression_by_score_and_leaves_ties_alone():
    """A flat-scored impression comes back in the order it arrived, which is
    the convention both other retrievers use for a user they know nothing
    about — and what makes the harness's all_scores_tied count comparable."""
    frame = pd.DataFrame(
        {"imp": [0, 0, 0, 1, 1], "aid": ["a", "b", "c", "x", "y"]}
    )
    ranked = fusion._order(
        frame, np.array([0.1, 0.9, 0.5, 0.4, 0.4]), np.array(["i1", "i2"])
    )

    assert ranked["impression_id"].tolist() == ["i1", "i2"]
    assert ranked["ranked_ids"].tolist() == [["b", "c", "a"], ["x", "y"]]
    assert ranked["scores"].tolist() == [[0.9, 0.5, 0.1], [0.4, 0.4]]


def test_a_ranking_holds_each_candidate_exactly_once():
    """What `predict.ranks` asserts on every impression of a submission, and
    the reason it can: nothing here adds or drops a candidate."""
    frame = pd.DataFrame(
        {"imp": [0] * 4, "aid": ["a", "b", "c", "d"]}
    )
    ranked = fusion._order(
        frame, np.array([0.2, 0.2, 0.9, 0.1]), np.array(["i1"])
    )
    assert sorted(ranked["ranked_ids"][0]) == ["a", "b", "c", "d"]


def test_impressions_sharing_an_id_each_get_their_own_ranking():
    """The submission zips rankings back onto its chunk positionally and
    checks the two line up, so a retriever that returned one ranking for the
    200,000 EB-NeRD impressions stamped id "0" would fail the whole file.
    Each row in, each row out, id repeated as given."""
    frame = pd.DataFrame(
        {"imp": [0, 0, 1, 2, 2], "aid": ["a", "b", "c", "x", "y"]}
    )
    ranked = fusion._order(
        frame,
        np.array([0.1, 0.9, 0.5, 0.4, 0.8]),
        np.array(["0", "0", "0"]),
    )

    assert ranked["impression_id"].tolist() == ["0", "0", "0"]
    assert ranked["ranked_ids"].tolist() == [["b", "a"], ["c"], ["y", "x"]]


def test_the_exposure_pass_folds_chunks_into_one_bucket_table():
    """The submission's counter is accumulated a chunk at a time; two chunks
    holding the same (article, bucket) must add rather than appear twice."""
    chunk = pd.DataFrame(
        {
            "impression_id": ["i1", "i2"],
            "impression_time": pd.to_datetime(
                ["2023-05-24 07:00:00", "2023-05-24 07:01:00"]
            ),
            "candidate_ids": [["a1", "a2"], ["a1"]],
        }
    )
    table = fusion._fold(None, [fusion._bucket(chunk), fusion._bucket(chunk)])

    assert len(table) == 2  # (a1, 07:00) and (a2, 07:00)
    assert table.set_index("aid")["shows"]["a1"] == 4
    assert table.set_index("aid")["shows"]["a2"] == 2
