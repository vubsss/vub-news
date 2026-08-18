"""The feature layer, and the two lines it exists to hold.

The first is causality: every popularity number is read strictly before the
impression it is read for. The second is what a competition test file can
supply, which is what separates the two fusion variants. Both are properties a
model would happily train through if they broke, and the symptom would be a
validation AUC that flatters and a leaderboard score that does not.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import features


def behaviors(rows):
    """rows: (impression_id, iso timestamp, candidate_ids, labels)"""
    return pd.DataFrame(
        {
            "impression_id": [row[0] for row in rows],
            "impression_time": pd.to_datetime([row[1] for row in rows]),
            "candidate_ids": [list(row[2]) for row in rows],
            "labels": [list(row[3]) for row in rows],
        }
    )


def test_a_counter_never_shows_an_impression_its_own_outcome():
    """The one property the whole popularity idea rests on.

    Both impressions below show a1 and one of them clicks it, in the same
    five-minute bucket. Either row seeing the other's click would be a model
    trained to predict a click from the click.
    """
    frame = features.explode(
        behaviors(
            [
                ("i1", "2023-05-24 07:00:00", ["a1"], [1]),
                ("i2", "2023-05-24 07:04:59", ["a1"], [0]),
            ]
        )
    )
    counter = features.Popularity.fit(frame)
    counter.attach(frame)

    assert (frame["expo_all"] == 0).all()
    assert frame["ctr_all"].tolist() == pytest.approx([counter.prior] * 2)


def test_a_counter_shows_an_impression_an_earlier_bucket():
    """And it has to show the past, or the feature is a constant."""
    frame = features.explode(
        behaviors(
            [
                ("i1", "2023-05-24 07:00:00", ["a1"], [1]),
                ("i2", "2023-05-24 09:00:00", ["a1"], [0]),
            ]
        )
    )
    counter = features.Popularity.fit(frame)
    counter.attach(frame)

    early, late = frame.iloc[0], frame.iloc[1]
    assert early["expo_all"] == 0
    assert late["expo_all"] == pytest.approx(np.log1p(1))
    assert late["ctr_all"] > early["ctr_all"]


def test_the_windowed_counter_forgets():
    """A click four hours ago is in ctr_all and out of the three-hour window."""
    frame = features.explode(
        behaviors(
            [
                ("i1", "2023-05-24 04:00:00", ["a1"], [1]),
                ("i2", "2023-05-24 08:00:00", ["a1"], [0]),
            ]
        )
    )
    counter = features.Popularity.fit(frame)
    counter.attach(frame)

    late = frame.iloc[1]
    assert late["expo_all"] == pytest.approx(np.log1p(1))
    assert late["expo_3h"] == 0
    assert late["expo_24h"] == pytest.approx(np.log1p(1))


def test_an_unlabelled_counter_has_no_click_column():
    """What a competition test file produces, and the reason for the second
    variant: the columns are simply absent rather than present and zero."""
    frame = features.explode(
        behaviors([("i1", "2023-05-24 07:00:00", ["a1"], [1])]), labelled=False
    )
    counter = features.Popularity.fit(frame)
    counter.attach(frame)

    assert counter.clicks is None
    assert set(counter.columns) == set(features.EXPOSURE)
    assert not any(column in frame for column in features.CLICKED)


def test_the_serving_feature_set_holds_no_click_feature():
    """The split the anti-gaming section is about, asserted rather than
    described: nothing in SERVING is a column that needs a label to exist."""
    assert not set(features.SERVING) & set(features.CLICKED)
    assert set(features.ALL) == set(features.SERVING) | set(features.CLICKED)


def test_within_impression_z_scores_are_relative():
    """Two impressions with the same shape and different levels z-score the
    same, which is the point: a linear or split-finding model comparing
    candidates inside an impression should see the shape."""
    frame = pd.DataFrame(
        {"imp": ["i1", "i1", "i2", "i2"], "x": [1.0, 3.0, 101.0, 103.0]}
    )
    features.normalise_within_impression(frame, ["x"])
    assert frame["x_z"].tolist() == pytest.approx([-0.7071, 0.7071, -0.7071, 0.7071], abs=1e-4)


def test_a_tied_impression_z_scores_to_zero_rather_than_to_nan():
    """One candidate, or every candidate equal, is a division by zero. Zero is
    the truthful answer — nothing here distinguishes them — and a NaN would
    reach the model as a missing value that means something else."""
    frame = pd.DataFrame({"imp": ["i1", "i1", "i2"], "x": [2.0, 2.0, 5.0]})
    features.normalise_within_impression(frame, ["x"])
    assert frame["x_z"].tolist() == [0.0, 0.0, 0.0]


def test_explode_keeps_an_impressions_candidates_adjacent_and_in_order():
    """Everything below reads the frame in blocks rather than grouping it."""
    frame = features.explode(
        behaviors(
            [
                ("i1", "2023-05-24 07:00:00", ["a1", "a2", "a3"], [0, 1, 0]),
                ("i2", "2023-05-24 07:30:00", ["a9"], [1]),
            ]
        )
    )
    # Row positions, not ids — see `explode`.
    assert frame["imp"].tolist() == [0, 0, 0, 1]
    assert frame["aid"].tolist() == ["a1", "a2", "a3", "a9"]
    assert frame["y"].tolist() == [0, 1, 0, 1]


def test_explode_separates_impressions_that_share_an_id():
    """EB-NeRD's test file stamps every one of its 200,000 beyond-accuracy
    impressions with the id "0". Keyed by id they became a single block, and
    every feature that walks the frame in blocks then built one impression's
    worth of column for all of them — which is how a submission died 98% of
    the way through a two-hour ranking pass with a length mismatch."""
    frame = features.explode(
        behaviors(
            [
                ("0", "2023-05-24 07:00:00", ["a1", "a2"], [0, 1]),
                ("0", "2023-05-24 07:30:00", ["a3"], [1]),
                ("0", "2023-05-24 08:00:00", ["a4", "a5"], [1, 0]),
            ]
        )
    )

    assert frame["imp"].tolist() == [0, 0, 1, 2, 2]
    values, lengths = features._runs(frame["imp"].to_numpy())
    assert list(lengths) == [2, 1, 2], "one block per impression, not per id"


def test_runs_reports_the_blocks_the_loops_walk():
    values, lengths = features._runs(np.array(["i1", "i1", "i2", "i3", "i3"]))
    assert list(values) == ["i1", "i2", "i3"]
    assert list(lengths) == [2, 1, 2]


def _similarity_frame(n_impressions, width, history_k, seed=0):
    """An exploded frame and a matching click history, both dense enough that
    the block loop has boundaries to get wrong."""
    rng = np.random.default_rng(seed)
    catalogue = [f"a{i}" for i in range(40)]
    vectors = rng.normal(size=(len(catalogue), 8)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    imps = list(range(n_impressions))
    frame = pd.DataFrame(
        {
            "imp": np.repeat(imps, width),
            # Some candidates are not in the catalogue at all, which is the
            # `known` mask's job.
            "aid": [
                rng.choice(catalogue) if rng.random() < 0.9 else "missing"
                for _ in range(n_impressions * width)
            ],
        }
    )
    # Indexed by row position, like the frame's `imp`. Histories of every
    # length from empty to longer than the window, so padding, right-alignment
    # and the empty guard are all exercised.
    clicks_of = np.empty(len(imps), dtype=object)
    for i in imps:
        clicks_of[i] = list(rng.choice(catalogue, size=i % (history_k + 2)))
    return frame, clicks_of, np.array(catalogue, dtype=object), vectors


def _reference(frame, clicks_of, article_ids, vectors, history_k):
    """The similarity features computed the obvious way — one impression at a
    time, no blocking, no index arithmetic. Slow and plainly correct."""
    row_of = {a: i for i, a in enumerate(article_ids)}
    sem, sem_max, sem_last = [], [], []
    for imp, aid in zip(frame["imp"], frame["aid"]):
        clicked = [row_of[c] for c in clicks_of[imp][-history_k:] if c in row_of]
        if aid not in row_of or not clicked:
            sem.append(0.0), sem_max.append(0.0), sem_last.append(0.0)
            continue
        sims = [float(vectors[row_of[aid]] @ vectors[c]) for c in clicked]
        sem.append(sum(sims) / len(sims))
        # The max runs over the padded slots too, and a padded slot scores 0 —
        # so it is clamped at zero exactly when the history is short of the
        # window, and not when it fills it. Only reachable with a negative
        # cosine, which the pipeline's own encoders do not produce and this
        # test's random vectors do.
        sem_max.append(max(sims + [0.0] * (history_k - len(clicked))))
        sem_last.append(sims[-1])
    return {
        "sem": np.array(sem, dtype="float32"),
        "sem_max": np.array(sem_max, dtype="float32"),
        "sem_last": np.array(sem_last, dtype="float32"),
    }


@pytest.mark.parametrize("block", [7, 100, 10_000])
def test_the_similarity_features_do_not_depend_on_the_block_size(block, monkeypatch):
    """The block loop is a memory measure, not a modelling choice: a chunk of
    EB-NeRD's competition file holds the candidate and history matrices whole
    at 7.8 GB, so both are carried as catalogue row numbers and gathered a
    block at a time. That is only allowed to change what the machine holds.
    Blocks that divide the frame evenly and blocks that do not, against a
    reference that walks one impression at a time.
    """
    from pipeline import embed

    history_k = 4
    frame, clicks_of, article_ids, vectors = _similarity_frame(30, 5, history_k)
    monkeypatch.setattr(features, "SIMILARITY_BLOCK", block)
    content = features.Content(
        config=None,
        articles=None,
        lexical=None,
        embeddings=embed.Embeddings(vectors=vectors, article_ids=article_ids),
    )

    got = content._semantic_scores(frame, clicks_of, history_k)
    want = _reference(frame, clicks_of, article_ids, vectors, history_k)

    for name in ("sem", "sem_max", "sem_last"):
        np.testing.assert_allclose(got[name], want[name], atol=1e-5, err_msg=name)
