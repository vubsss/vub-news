"""The temporal split is tested at three seams: assign, check_leakage, run."""

import pandas as pd
import pytest

from pipeline import paths, split
from pipeline.datasets import DATASETS, SplitSpec

MIND = DATASETS["mind"]


def behaviors(times, candidates=None):
    """Impressions carrying what the split looks at: id, timestamp, candidates."""
    return pd.DataFrame(
        {
            "impression_id": [f"i{i}" for i in range(len(times))],
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

    labelled = split.assign(frame, SplitSpec(val_days=1, test_days=1))

    assert not labelled["split"].isna().any()
    assert list(labelled["split"]) == ["train", "train", "validation", "test"]


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
        SplitSpec(val_days=1, test_days=1),
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


def history(rows):
    """rows: (impression_id, click_history)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
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
    split.check_leakage(impressions, clean, catalogue)

    leaked = history([("i0", ["a1", "a2"]), ("i1", ["a1"])])
    with pytest.raises(split.LeakageError, match="a2"):
        split.check_leakage(impressions, leaked, catalogue)


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

    report = split.check_leakage(impressions, clicks, catalogue)

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
    # MIND's registry window: one day test, one day validation, five train.
    assert counts["train"] == 5
    assert counts["validation"] == 1
    assert counts["test"] == 1

    # Row counts and date ranges per partition, so an empty test week or a
    # validation partition larger than train is visible at a glance.
    printed = capsys.readouterr().out
    for label in ("train", "validation", "test"):
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


def test_run_aborts_on_a_future_click(store):
    impressions, clicks, catalogue = week()
    clicks.at[0, "click_history"] = ["a1", "a2"]
    catalogue = pd.concat(
        [catalogue, articles([("a2", "2019-12-25 06:00:00")])], ignore_index=True
    )
    write_store(store, impressions, clicks, catalogue)

    with pytest.raises(split.LeakageError, match="a2"):
        split.run(MIND)


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
