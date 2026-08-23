"""The temporal split is tested at three seams: assign, check_leakage, run."""

import pandas as pd
import pytest

from pipeline import ingest, paths, split
from pipeline.datasets import DATASETS, SplitSpec

MIND = DATASETS["mind"]


def behaviors(times, candidates=None):
    """Impressions carrying what the split looks at: id, timestamp, candidates."""
    return pd.DataFrame(
        {
            "impression_id": [f"i{i}" for i in range(len(times))],
            # One user per impression, and the other half of the history
            # table's key. `check_leakage` reads the per-impression view that
            # `ingest.history_for` rebuilds from it.
            "user_id": pd.Series([f"u{i}" for i in range(len(times))], dtype="string"),
            "source": pd.Series(["train"] * len(times), dtype="string"),
            "impression_time": pd.to_datetime(list(times)),
            "candidate_ids": candidates or [[] for _ in times],
            "split": pd.Series([None] * len(times), dtype="string"),
        }
    )


def test_every_impression_is_labelled_by_its_timestamp():
    """The last test_days are test, the val_days before them validation."""
    frame = behaviors(
        [
            "2019-11-09 00:00:19",
            "2019-11-13 23:59:59",
            "2019-11-14 08:00:00",
            "2019-11-15 23:58:03",
        ]
    )

    labelled = split.assign(frame, SplitSpec(tune_days=1, val_days=1, test_days=1))

    assert not labelled["split"].isna().any()
    assert list(labelled["split"]) == ["train", "tune", "validation", "test"]


def test_partitions_that_overlap_in_time_abort_the_run():
    """max(train) < min(validation) < min(test), or the build stops."""
    labelled = split.assign(
        behaviors(
            [
                "2019-11-09 00:00:19",
                "2019-11-14 08:00:00",
                "2019-11-15 23:58:03",
            ]
        ),
        SplitSpec(tune_days=1, val_days=1, test_days=1),
    )
    split.check_ordering(labelled)

    # A test impression back-dated into the train window: the partitions now
    # overlap in time, which is exactly what a random split would look like.
    leaked = labelled.copy()
    leaked.loc[2, "impression_time"] = pd.Timestamp("2019-11-10 12:00:00")

    with pytest.raises(split.SplitError, match="test"):
        split.check_ordering(leaked)


def articles(rows):
    """rows: (article_id, published_time or None)"""
    return pd.DataFrame(
        {
            "article_id": pd.Series([row[0] for row in rows], dtype="string"),
            "published_time": pd.to_datetime([row[1] for row in rows]),
        }
    )


def joined(history, impressions):
    """The per-impression view `check_leakage` is handed in production."""
    return ingest.per_impression(history, impressions)


def history(rows):
    """rows: (impression_id, click_history) — stored keyed by its user.

    The ids are `iN` and the users `uN`, one apiece, so a row named for an
    impression here is the history of the user that impression belongs to.
    """
    return pd.DataFrame(
        {
            "user_id": pd.Series(
                [row[0].replace("i", "u") for row in rows], dtype="string"
            ),
            "source": pd.Series(["train"] * len(rows), dtype="string"),
            "click_history": [list(row[1]) for row in rows],
        }
    )


def test_a_click_on_an_article_published_later_aborts_the_run():
    """The behaviour-window guard: nobody clicks an article that does not
    exist yet, so a history containing one is leakage, not a data quirk."""
    impressions = behaviors(["2019-11-10 12:00:00", "2019-11-11 12:00:00"])
    catalogue = articles(
        [("a1", "2019-11-08 06:00:00"), ("a2", "2019-11-20 06:00:00")]
    )

    clean = history([("i0", ["a1"]), ("i1", ["a1"])])
    report = split.check_leakage(impressions, joined(clean, impressions), catalogue)
    assert report["future_clicks"] == 0

    leaked = history([("i0", ["a1", "a2"]), ("i1", ["a1"])])
    # Half the histories, which is systematic rather than metadata, so it stops
    # the build — that is what the guard is for.
    with pytest.raises(split.LeakageError, match="histories"):
        split.check_leakage(impressions, joined(leaked, impressions), catalogue)


def test_a_rare_future_click_is_dropped_and_counted_rather_than_fatal():
    """Publication metadata is wrong occasionally and a user is not seeing the
    future because of it: ebnerd_large has 16 such histories in 1,579,672, with
    overshoots up to 31 days, and ebnerd_small has none only because it samples
    18,827 of the same 974,791 users. The click is removed from the profile and
    the rate printed; the guard still stops a build where the rate says the
    history came from the wrong period."""
    stamp = "2019-11-10 12:00:00"
    impressions = behaviors([stamp] * 2000)
    catalogue = articles(
        [("a1", "2019-11-08 06:00:00"), ("a2", "2019-11-12 06:00:00")]
    )
    # One history in two thousand holds the article published later.
    rows = [(f"i{i}", ["a1"]) for i in range(2000)]
    rows[7] = ("i7", ["a1", "a2"])

    report = split.check_leakage(impressions, joined(history(rows), impressions), catalogue)

    assert report["future_clicks"] == 1
    assert report["future_histories"] == 1
    assert report["repeat_candidates"] == 0


def test_an_already_clicked_candidate_is_counted_not_assumed_impossible():
    """Both datasets re-show articles a user has already read, so this is a
    rate to report, not an invariant to assert. The guard that does abort is
    the future click above; this one only has to be visible."""
    impressions = behaviors(
        ["2019-11-10 12:00:00", "2019-11-11 12:00:00", "2019-11-12 12:00:00"],
        candidates=[["a1", "a3"], ["a3"], ["a1"]],
    )
    catalogue = articles(
        [("a1", "2019-11-08 06:00:00"), ("a3", "2019-11-09 06:00:00"), ("a4", None)]
    )
    clicks = history([("i0", ["a1"]), ("i1", ["a1", "a4"]), ("i2", [])])

    report = split.check_leakage(impressions, joined(clicks, impressions), catalogue)

    # Only i0 re-offers an article its own user already clicked.
    assert report["impressions"] == 3
    assert report["repeat_candidates"] == 1
    # a4 has no publication time, so the future-click guard cannot see it.
    assert report["clicks"] == 3
    assert report["clicks_checked"] == 2


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    MIND.feature_store_dir.mkdir(parents=True)
    return MIND.feature_store_dir


def write_store(store, impressions, clicks, catalogue):
    impressions.to_parquet(store / "behaviors.parquet", index=False)
    clicks.to_parquet(store / "history.parquet", index=False)
    catalogue.to_parquet(store / "articles.parquet", index=False)


def week():
    """Seven days of impressions, one per day, the shape MIND ships."""
    impressions = behaviors(
        [f"2019-11-{day:02d} 12:00:00" for day in range(9, 16)],
        candidates=[["a1"] for _ in range(7)],
    )
    clicks = history([(f"i{i}", ["a1"]) for i in range(7)])
    return impressions, clicks, articles([("a1", "2019-11-01 06:00:00")])


def test_run_labels_every_impression_and_reports_each_partition(store, capsys):
    write_store(store, *week())

    split.run(MIND)

    written = pd.read_parquet(store / "behaviors.parquet")
    assert not written["split"].isna().any()
    counts = written["split"].value_counts()
    # MIND's registry windows: a day each of test, validation and tune, and the
    # four remaining days are train.
    assert counts["train"] == 4
    assert counts["tune"] == 1
    assert counts["validation"] == 1
    assert counts["test"] == 1

    # Row counts and date ranges per partition, so an empty test week or a
    # validation partition larger than train is visible at a glance.
    printed = capsys.readouterr().out
    for label in ("train", "tune", "validation", "test"):
        assert label in printed
    assert "2019-11-09" in printed
    assert "2019-11-15" in printed


def test_running_the_split_twice_changes_nothing(store):
    write_store(store, *week())

    split.run(MIND)
    first = pd.read_parquet(store / "behaviors.parquet")
    split.run(MIND)
    second = pd.read_parquet(store / "behaviors.parquet")

    pd.testing.assert_frame_equal(first, second)


def test_run_aborts_when_the_future_click_rate_says_systematic(store):
    """One history in seven is well past the ceiling, so this is a history
    joined from the wrong period rather than bad publication metadata, and the
    stage must stop rather than drop it."""
    impressions, clicks, catalogue = week()
    clicks.at[0, "click_history"] = ["a1", "a2"]
    catalogue = pd.concat(
        [catalogue, articles([("a2", "2019-12-25 06:00:00")])], ignore_index=True
    )
    write_store(store, impressions, clicks, catalogue)

    with pytest.raises(split.LeakageError, match="histories"):
        split.run(MIND)


def test_the_tune_window_is_carved_from_the_end_of_train():
    """Tune sits between train and validation, not at the start of the log.

    Which end it comes from is the whole point: a tuning window taken from the
    beginning of train would be the furthest thing in time from the population
    it stands in for.
    """
    frame = behaviors([f"2019-11-{day:02d} 12:00:00" for day in range(9, 16)])

    labelled = split.assign(frame, SplitSpec(tune_days=2, val_days=1, test_days=1))

    windows = labelled.groupby("split")["impression_time"]
    assert windows.max()["train"] < windows.min()["tune"]
    assert windows.max()["tune"] < windows.min()["validation"]


def test_adding_a_tune_window_moves_only_the_train_boundary():
    """Validation and test cover the same days they did without a tune split.

    This is what makes numbers reported before and after this change
    comparable: the held-out windows are measured back from the end of the log,
    so carving a tuning window out of train cannot reach them.
    """
    frame = behaviors([f"2019-11-{day:02d} 12:00:00" for day in range(9, 16)])

    without = split.assign(frame, SplitSpec(tune_days=0, val_days=1, test_days=1))
    with_tune = split.assign(frame, SplitSpec(tune_days=2, val_days=1, test_days=1))

    for partition in ("validation", "test"):
        assert list(without[without["split"] == partition]["impression_id"]) == list(
            with_tune[with_tune["split"] == partition]["impression_id"]
        )


def test_an_impression_exactly_on_a_boundary_joins_the_earlier_partition():
    """MIND ships one impression at exactly 2019-11-14 00:00:00, which is a
    partition boundary. Which side it lands on is arbitrary; that it lands on
    the same side every run is not, because a row changing sides silently moves
    a population between two reported numbers.

    Bins are right-closed, so the boundary row belongs to the earlier
    partition: at MIND's windows, 2019-11-14 00:00:00 is the last moment of
    tune rather than the first of validation.
    """
    frame = behaviors(
        [
            "2019-11-09 00:00:19",
            "2019-11-14 00:00:00",  # exactly the tune/validation boundary
            "2019-11-14 00:00:11",
            "2019-11-15 12:00:00",
        ]
    )

    labelled = split.assign(frame, MIND.split)

    assert list(labelled["split"]) == ["train", "tune", "validation", "test"]


@pytest.mark.parametrize("config", list(DATASETS.values()), ids=lambda c: c.name)
def test_the_real_feature_store_is_split_in_time(config):
    """The artifact on disk, not a fixture: every impression labelled, and the
    partitions strictly ordered."""
    path = config.feature_store_dir / "behaviors.parquet"
    if not path.exists():
        pytest.skip(f"{config.name} feature store not built")

    behaviors = pd.read_parquet(path, columns=["impression_time", "split"])
    if behaviors["split"].isna().all():
        pytest.skip(f"{config.name} split stage has not run")

    assert not behaviors["split"].isna().any()
    assert set(behaviors["split"]) == set(split.SPLITS)
    split.check_ordering(behaviors)
