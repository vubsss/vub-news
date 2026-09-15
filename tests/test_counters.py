"""The counters are causal -- Q9's leakage test -- and the whole-log read is
the same object asked for a later moment. Then the store round-trips and
the stage records what each window costs."""

import numpy as np
import pandas as pd
import pytest

from pipeline import counters, ledger, paths
from pipeline.counters import WHOLE_LOG
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]

T0 = pd.Timestamp("2019-11-09 10:00:00")


def log(rows, split="train"):
    """Impressions as (minutes after T0, candidate ids, labels)."""
    return pd.DataFrame(
        {
            "impression_id": [f"i{i}" for i in range(len(rows))],
            "impression_time": [T0 + pd.Timedelta(minutes=m) for m, _, _ in rows],
            "candidate_ids": [np.array(c) for _, c, _ in rows],
            "labels": [np.array(l) for _, _, l in rows],
            "split": pd.Series([split] * len(rows), dtype="string"),
        }
    )


FOUR = log(
    [
        (0, ["A", "B"], [1, 0]),
        (10, ["A", "C"], [0, 1]),
        (10, ["A", "B"], [1, 1]),  # the same instant as the one above
        (200, ["B"], [0]),
    ]
)


# --- Q9: no future click leaks into a causal read --------------------------


def test_an_impression_never_sees_its_own_outcome():
    store = counters.build(FOUR)
    at_first = store.at(["A", "B"], T0)
    assert list(at_first.exposures) == [0, 0]
    assert list(at_first.clicks) == [0, 0]
    assert np.isnan(at_first.ctr).all()


def test_two_impressions_at_the_same_instant_do_not_see_each_other():
    store = counters.build(FOUR)
    ten = T0 + pd.Timedelta(minutes=10)
    # Only the impression at minute 0 is before minute 10, whichever of the
    # two impressions at minute 10 is asking.
    at_ten = store.at(["A", "B", "C"], ten)
    assert list(at_ten.exposures) == [1, 1, 0]
    assert list(at_ten.clicks) == [1, 0, 0]


def test_removing_every_row_after_t_leaves_the_causal_read_unchanged():
    full = counters.build(FOUR)
    for minutes in (0, 5, 10, 11, 200, 201):
        t = T0 + pd.Timedelta(minutes=minutes)
        truncated = FOUR[FOUR["impression_time"] < t]
        if truncated.empty:
            continue
        cut = counters.build(truncated)
        ids = ["A", "B", "C"]
        for window in counters.WINDOWS.values():
            for name in ("exposures", "clicks"):
                assert list(getattr(full.at(ids, t, window), name)) == list(
                    getattr(cut.at(ids, t, window), name)
                ), (minutes, window, name)
            np.testing.assert_array_equal(full.first_seen(ids, t), cut.first_seen(ids, t))


def test_the_whole_log_read_dominates_the_causal_read_everywhere():
    store = counters.build(FOUR)
    ids = list(store.article_ids)
    leaky = store.at(ids, WHOLE_LOG)
    assert list(leaky.exposures) == [3, 3, 1]
    assert list(leaky.clicks) == [2, 1, 1]
    for t in FOUR["impression_time"]:
        causal = store.at(ids, t)
        assert (causal.exposures <= leaky.exposures).all()
        assert (causal.clicks <= leaky.clicks).all()
    # And +inf is a moment, not a second code path: the last impression
    # itself reads everything before it and one less than the whole log.
    last = store.at(["B"], FOUR["impression_time"].max())
    assert last.exposures[0] == leaky.exposures[1] - 1


def test_first_seen_is_strictly_before_t_or_null():
    store = counters.build(FOUR)
    assert pd.isna(store.first_seen(["A"], T0)[0])
    ten = T0 + pd.Timedelta(minutes=10)
    assert store.first_seen(["A", "C"], ten)[0] == np.datetime64(T0, "us")
    assert pd.isna(store.first_seen(["A", "C"], ten)[1])
    seen = store.first_seen(["A", "B", "C"], WHOLE_LOG)
    assert list(seen) == [np.datetime64(T0, "us")] * 2 + [np.datetime64(ten, "us")]
    for t in FOUR["impression_time"]:
        before = store.first_seen(list(store.article_ids), t)
        assert all(pd.isna(m) or m < np.datetime64(t, "us") for m in before)


# --- the interface ---------------------------------------------------------


def test_a_sliding_window_forgets_exposures_older_than_the_window():
    store = counters.build(FOUR)
    t = T0 + pd.Timedelta(minutes=200)
    assert store.at(["A"], t).exposures[0] == 3
    assert store.at(["A"], t, pd.Timedelta(hours=1)).exposures[0] == 0
    assert store.at(["A"], t, pd.Timedelta(hours=4)).exposures[0] == 3
    # A window reaching before the log began is the cumulative count.
    early = T0 + pd.Timedelta(minutes=11)
    assert store.at(["B"], early, pd.Timedelta(days=1)).clicks[0] == 1
    # A moment past the log's end still reads its own last hour, not the log.
    late = T0 + pd.Timedelta(days=3)
    assert store.at(["A"], late, pd.Timedelta(hours=1)).exposures[0] == 0
    assert store.at(["A"], late).exposures[0] == 3
    # And the whole-log read is cumulative by definition.
    with pytest.raises(counters.CounterError):
        store.at(["A"], WHOLE_LOG, pd.Timedelta(hours=1))


def test_one_moment_per_id_is_how_a_feature_frame_reads():
    store = counters.build(FOUR)
    moments = [T0, T0 + pd.Timedelta(minutes=11), T0 + pd.Timedelta(minutes=300)]
    read = store.at(["A", "A", "A"], moments)
    assert list(read.exposures) == [0, 3, 3]
    assert list(read.clicks) == [0, 2, 2]
    assert read.ctr[1] == pytest.approx(2 / 3)
    with pytest.raises(counters.CounterError):
        store.at(["A", "B"], moments)


def test_an_article_the_log_never_showed_reads_as_nothing():
    store = counters.build(FOUR)
    read = store.at(["Z"], WHOLE_LOG)
    assert (read.exposures[0], read.clicks[0]) == (0, 0)
    assert np.isnan(read.ctr[0])
    assert pd.isna(store.first_seen(["Z"], WHOLE_LOG)[0])


def test_the_store_round_trips_through_disk(tmp_path):
    store = counters.build(FOUR)
    store.save(tmp_path)
    loaded = counters.load(tmp_path)
    ids = ["A", "B", "C", "Z"]
    for t in (T0 + pd.Timedelta(minutes=10), WHOLE_LOG):
        for read, expected in zip(loaded.at(ids, t), store.at(ids, t)):
            np.testing.assert_array_equal(read, expected)
        np.testing.assert_array_equal(loaded.first_seen(ids, t), store.first_seen(ids, t))


def test_an_empty_log_is_refused():
    with pytest.raises(counters.CounterError):
        counters.build(FOUR.iloc[:0])


# --- the stage -------------------------------------------------------------


def test_the_stage_records_one_ledger_row_per_window(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(counters, "LATENCY_SAMPLE", 2)
    store_dir = MIND.feature_store_dir
    store_dir.mkdir(parents=True)
    behaviors = pd.concat([FOUR, log([(300, ["A", "C"], [0, 1])], split="tune")])
    behaviors.to_parquet(store_dir / "behaviors.parquet", index=False)

    counters.run(MIND)

    assert (MIND.artifacts_dir / counters.DIRECTORY / counters.STORE).exists()
    rows = {row["variant"]: row for row in ledger.load()}
    assert set(rows) == set(counters.WINDOWS)
    for row in rows.values():
        assert row["dataset"] == "mind" and row["stage"] == counters.STAGE
        assert row["index_bytes"] > 0
        assert row["train_seconds"] >= 0 and row["peak_rss_mb"] > 0
        assert row["p50_ms"] >= 0 and row["p99_ms"] >= row["p50_ms"]
        assert row["rows_per_s"] > 0
    assert (paths.ARTIFACTS_DIR / ledger.DOCUMENT).exists()


def test_the_stage_refuses_a_log_without_the_bench_split(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    MIND.feature_store_dir.mkdir(parents=True)
    FOUR.to_parquet(MIND.feature_store_dir / "behaviors.parquet", index=False)
    with pytest.raises(counters.CounterError):
        counters.run(MIND)


# --- what a server knows at submission time ---------------------------------
#
# The competition is a later period, so the counters a submission reads come
# from three places at once and one of them is a log whose outcomes nobody has.
# These are the tests that say the three are combined the way a live server's
# would be, rather than the way that scores best.


TEST_LOG = log(
    [
        (300, ["A", "B"], [0, 0]),
        (400, ["A", "C"], [0, 0]),
        (500, ["B", "C", "E"], [0, 0, 0]),  # E: an article only the test week carries
    ],
    split="test",
)

# Two test users whose reading histories mention A twice and D once. These are
# clicks a server genuinely knows happened -- the competition ships them -- and
# they carry no timestamp, which is why they count cumulatively and nowhere
# else.
TEST_HISTORIES = [["A", "D"], ["A"]]


def serving() -> counters.ServingCounters:
    ids, counts = counters.clicks_in_histories(TEST_HISTORIES)
    return counters.ServingCounters(
        trained=counters.build(FOUR),
        shown=counters.exposures_only(TEST_LOG),
        clicked_ids=ids,
        clicked_counts=counts,
    )


def test_the_training_period_is_read_whole_because_it_is_all_before():
    """Every training impression precedes every test one, so a test-time read
    sees the training log entire -- not because the window was widened but
    because `t` is later than all of it."""
    whole = counters.build(FOUR).at(["A", "B"], T0 + pd.Timedelta(minutes=300))
    served = serving().at(["A", "B"], T0 + pd.Timedelta(minutes=300))

    assert list(served.exposures) == list(whole.exposures)


def test_a_test_impression_sees_earlier_test_exposures_and_no_later_ones():
    """The strictness that makes this the 'what a live server knows' line: the
    impression at minute 400 counts the one at 300 and not the one at 500."""
    at_400 = serving().at(["A", "B", "C"], T0 + pd.Timedelta(minutes=400))
    at_500 = serving().at(["A", "B", "C"], T0 + pd.Timedelta(minutes=500))

    # A: 3 training exposures + 1 test exposure at minute 300.
    assert at_400.exposures[0] == 4
    # C: only the training one at minute 10 until minute 500 adds nothing yet.
    assert at_400.exposures[2] == 1
    # By minute 500 the impression at 400 has been counted, and its own has not.
    assert at_500.exposures[2] == 2


def test_no_click_of_the_test_period_is_ever_counted():
    """The leaderboard is holding the outcomes back, so the only clicks a
    submission can count are the ones the histories reveal. An implementation
    that inferred a click from an exposure would show up here."""
    served = serving()
    late = served.at(["A", "B", "C"], T0 + pd.Timedelta(minutes=600))
    trained = counters.build(FOUR).at(["A", "B", "C"], T0 + pd.Timedelta(minutes=600))

    # B and C gain exposures from the test log and no clicks at all.
    assert served.at(["B"], T0 + pd.Timedelta(minutes=600)).exposures[0] > trained.exposures[1]
    assert late.clicks[1] == trained.clicks[1]
    assert late.clicks[2] == trained.clicks[2]


def test_a_history_click_counts_cumulatively_and_in_no_window():
    """A click with no timestamp cannot be placed in an hour. Counting it in
    one would be inventing the moment it happened; leaving it out of the
    cumulative counter would be discarding a click the server has."""
    served = serving()
    t = T0 + pd.Timedelta(minutes=600)
    trained = counters.build(FOUR)

    cumulative = served.at(["A"], t).clicks[0]
    windowed = served.at(["A"], t, pd.Timedelta(hours=1)).clicks[0]

    assert cumulative == trained.at(["A"], t).clicks[0] + 2  # both histories
    assert windowed == trained.at(["A"], t, pd.Timedelta(hours=1)).clicks[0]


def test_an_article_only_the_test_period_has_shown_is_as_old_as_its_first_test_view():
    """Freshness is the earlier of the two logs' first sightings. An article
    the training week never carried is not ageless; it is as old as the first
    test impression that showed it."""
    served = serving()
    t = T0 + pd.Timedelta(minutes=600)
    first = served.first_seen(["A", "E"], t)

    # A was first shown at minute 0, in the training log, which is earlier than
    # any test view of it -- so the training log wins.
    assert first[0] == np.datetime64(T0.to_datetime64())
    # E appears in neither training impression, so its age is the test log's.
    assert first[1] == np.datetime64((T0 + pd.Timedelta(minutes=500)).to_datetime64())


def test_an_article_neither_log_has_shown_has_no_first_sighting():
    served = serving()
    assert pd.isna(served.first_seen(["Z"], T0 + pd.Timedelta(minutes=600))[0])


def test_the_histories_are_counted_once_per_appearance():
    ids, counts = counters.clicks_in_histories(TEST_HISTORIES)
    assert dict(zip(ids, counts)) == {"A": 2, "D": 1}


def test_a_log_with_no_outcomes_contributes_exposures_and_nothing_else():
    """`exposures_only` is handed all-zero labels rather than being a second
    constructor, so there is one definition of what an exposure is."""
    store = counters.exposures_only(TEST_LOG)
    late = store.at(["A", "B", "C"], T0 + pd.Timedelta(minutes=600))

    assert list(late.exposures) == [2, 2, 2]  # A, B, C once each in two rows
    assert list(late.clicks) == [0, 0, 0]
