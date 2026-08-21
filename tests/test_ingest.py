"""Ingest is tested at three seams: the table builders and run()."""

import dataclasses

import pandas as pd
import pytest

from pipeline import ingest, paths, sources
from pipeline.datasets import (
    ARTICLE_COLUMNS,
    BEHAVIOR_COLUMNS,
    DATASETS,
    HISTORY_COLUMNS,
)

MIND = DATASETS["mind"]
EBNERD = DATASETS["ebnerd"]


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path)
    return tmp_path


def write_mind_news(split, rows):
    """rows: (news_id, category, subcategory, title, abstract)"""
    path = MIND.raw_dir / split / "news.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\t".join([*row, "http://url", "[]", "[]"]) + "\n" for row in rows
    ]
    path.write_text("".join(lines), encoding="utf-8")


def write_ebnerd_articles(rows):
    """rows: (article_id, title, subtitle, body, category_str, subcategory_ids)"""
    path = EBNERD.raw_dir / "articles.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        rows,
        columns=[
            "article_id",
            "title",
            "subtitle",
            "body",
            "category_str",
            "subcategory",
        ],
    )
    frame["article_id"] = frame["article_id"].astype("int32")
    frame["published_time"] = pd.Timestamp("2023-05-01 08:00:00")
    frame.to_parquet(path)


def test_mind_articles_map_onto_the_unified_schema(raw):
    write_mind_news("train", [("N1", "sports", "football", "Title one", "Abstract one")])
    write_mind_news("dev", [("N2", "news", "world", "Title two", "Abstract two")])

    articles = ingest.build_articles(MIND)

    assert list(articles.columns) == list(ARTICLE_COLUMNS)
    assert set(articles["article_id"]) == {"N1", "N2"}

    first = articles.set_index("article_id").loc["N1"]
    assert first["title"] == "Title one"
    assert first["abstract"] == "Abstract one"
    assert first["category"] == "sports"
    assert first["subcategory"] == "football"
    assert first["dataset"] == "mind"
    # MIND ships no article body and no publication timestamp.
    assert pd.isna(first["body"])
    assert pd.isna(first["published_time"])


def test_ebnerd_articles_map_onto_the_unified_schema(raw):
    write_ebnerd_articles(
        [
            (101, "Titel en", "Undertitel en", "Brødtekst", "sport", [414, 5]),
            (102, "Titel to", "Undertitel to", "Mere tekst", "nyheder", []),
        ]
    )

    articles = ingest.build_articles(EBNERD)

    assert list(articles.columns) == list(ARTICLE_COLUMNS)

    first = articles.set_index("article_id").loc["101"]
    assert first["title"] == "Titel en"
    # EB-NeRD has no abstract; the subtitle plays that role.
    assert first["abstract"] == "Undertitel en"
    assert first["body"] == "Brødtekst"
    assert first["category"] == "sport"
    # Only the first subcategory is kept, for parity with MIND's single value.
    assert first["subcategory"] == "414"
    assert first["published_time"] == pd.Timestamp("2023-05-01 08:00:00")
    assert first["dataset"] == "ebnerd"

    # An article with no subcategory at all must not break the mapping.
    assert pd.isna(articles.set_index("article_id").loc["102", "subcategory"])


def test_an_article_in_several_source_files_is_kept_once(raw):
    """MIND ships the same articles in both the train and dev news files."""
    write_mind_news("train", [("N1", "sports", "football", "Title", "Abstract")])
    write_mind_news(
        "dev",
        [
            ("N1", "sports", "football", "Title", "Abstract"),
            ("N2", "news", "world", "Other", "Other abstract"),
        ],
    )

    articles = ingest.build_articles(MIND)

    assert articles["article_id"].is_unique
    assert len(articles) == 2


def test_both_datasets_produce_the_same_article_schema(raw):
    write_mind_news("train", [("N1", "sports", "football", "Title", "Abstract")])
    write_mind_news("dev", [("N2", "news", "world", "Other", "Other abstract")])
    write_ebnerd_articles([(101, "Titel", "Undertitel", "Tekst", "sport", [414])])

    mind = ingest.build_articles(MIND)
    ebnerd = ingest.build_articles(EBNERD)

    assert list(mind.columns) == list(ebnerd.columns)
    assert mind.dtypes.to_dict() == ebnerd.dtypes.to_dict()


def write_mind_behaviors(split, rows):
    """rows: (impression_id, user_id, time, history, impressions)"""
    path = MIND.raw_dir / split / "behaviors.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_mind_behaviors_split_candidates_from_labels(raw):
    write_mind_behaviors(
        "train",
        [("1", "U1", "11/11/2019 9:05:58 AM", "N1 N2", "N3-1 N4-0 N5-0")],
    )
    write_mind_behaviors(
        "dev", [("1", "U2", "11/15/2019 7:00:00 PM", "N9", "N3-0 N7-1")]
    )

    behaviors = ingest.build_behaviors(MIND)

    assert list(behaviors.columns) == list(BEHAVIOR_COLUMNS)

    first = behaviors[behaviors["user_id"] == "U1"].iloc[0]
    assert first["candidate_ids"] == ["N3", "N4", "N5"]
    assert first["labels"] == [1, 0, 0]
    assert first["impression_time"] == pd.Timestamp("2019-11-11 09:05:58")
    assert first["dataset"] == "mind"
    # The split column is assigned later, by the temporal split stage.
    assert pd.isna(first["split"])


def test_impression_ids_from_different_source_files_do_not_collide(raw):
    """MIND numbers train and dev impressions from 1 independently."""
    write_mind_behaviors(
        "train", [("1", "U1", "11/11/2019 9:05:58 AM", "N1", "N3-1")]
    )
    write_mind_behaviors("dev", [("1", "U2", "11/15/2019 7:00:00 PM", "N9", "N3-0")])

    behaviors = ingest.build_behaviors(MIND)

    assert len(behaviors) == 2
    assert behaviors["impression_id"].is_unique


def write_ebnerd_behaviors(split, rows):
    """rows: (impression_id, user_id, impression_time, inview, clicked)"""
    path = EBNERD.raw_dir / split / "behaviors.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        rows,
        columns=[
            "impression_id",
            "user_id",
            "impression_time",
            "article_ids_inview",
            "article_ids_clicked",
        ],
    )
    frame["impression_id"] = frame["impression_id"].astype("uint32")
    frame["user_id"] = frame["user_id"].astype("uint32")
    frame.to_parquet(path)


def test_ebnerd_labels_come_from_the_clicked_article_ids(raw):
    stamp = pd.Timestamp("2023-05-23 07:31:00")
    write_ebnerd_behaviors("train", [(7, 55, stamp, [11, 12, 13], [12])])
    write_ebnerd_behaviors("validation", [(8, 56, stamp, [21, 22], [])])

    behaviors = ingest.build_behaviors(EBNERD)

    assert list(behaviors.columns) == list(BEHAVIOR_COLUMNS)

    first = behaviors[behaviors["user_id"] == "55"].iloc[0]
    assert first["candidate_ids"] == ["11", "12", "13"]
    assert first["labels"] == [0, 1, 0]
    assert first["impression_time"] == stamp
    assert first["dataset"] == "ebnerd"

    # An impression nobody clicked is still a valid impression.
    second = behaviors[behaviors["user_id"] == "56"].iloc[0]
    assert second["labels"] == [0, 0]


def test_both_datasets_produce_the_same_behavior_schema(raw):
    write_mind_behaviors(
        "train", [("1", "U1", "11/11/2019 9:05:58 AM", "N1", "N3-1 N4-0")]
    )
    write_mind_behaviors("dev", [("2", "U2", "11/15/2019 7:00:00 PM", "N9", "N3-0")])
    stamp = pd.Timestamp("2023-05-23 07:31:00")
    write_ebnerd_behaviors("train", [(7, 55, stamp, [11, 12], [12])])
    write_ebnerd_behaviors("validation", [(8, 56, stamp, [21], [])])

    mind = ingest.build_behaviors(MIND)
    ebnerd = ingest.build_behaviors(EBNERD)

    assert list(mind.columns) == list(ebnerd.columns)
    assert mind.dtypes.to_dict() == ebnerd.dtypes.to_dict()


def test_candidates_and_labels_must_be_the_same_length(raw):
    """A silent length mismatch would corrupt every metric downstream."""
    write_mind_behaviors(
        "train", [("1", "U1", "11/11/2019 9:05:58 AM", "N1", "N3-1 N4-0")]
    )
    write_mind_behaviors("dev", [("2", "U2", "11/15/2019 7:00:00 PM", "N9", "N3-0")])

    def drops_a_label(frame):
        adapted = sources.mind_behaviors(frame)
        adapted["labels"] = adapted["labels"].map(lambda labels: labels[:-1])
        return adapted

    broken = dataclasses.replace(
        MIND,
        sources=dataclasses.replace(
            MIND.sources,
            behaviors=dataclasses.replace(
                MIND.sources.behaviors, adapt=drops_a_label
            ),
        ),
    )

    with pytest.raises(ingest.SchemaError, match="labels"):
        ingest.build_behaviors(broken)


def test_mind_history_splits_the_click_string(raw):
    write_mind_behaviors(
        "train", [("1", "U1", "11/11/2019 9:05:58 AM", "N1 N2 N3", "N3-1")]
    )
    write_mind_behaviors("dev", [("1", "U2", "11/15/2019 7:00:00 PM", "", "N3-0")])

    behaviors = ingest.build_behaviors(MIND)
    history = ingest.build_history(MIND, behaviors)

    assert list(history.columns) == list(HISTORY_COLUMNS)

    warm = history[history["user_id"] == "U1"].iloc[0]
    assert warm["click_history"] == ["N1", "N2", "N3"]
    assert warm["n_clicks"] == 3
    # Every history row belongs to a real impression.
    assert warm["impression_id"] in set(behaviors["impression_id"])

    # A user with no history at all is a cold user, not a missing row.
    cold = history[history["user_id"] == "U2"].iloc[0]
    assert cold["click_history"] == []
    assert cold["n_clicks"] == 0


def write_ebnerd_history(split, rows):
    """rows: (user_id, article_id_fixed)

    The three engagement columns are written alongside, one entry per click,
    because the real file carries them and a fixture that did not would let a
    reader of them pass here and fail on the dataset."""
    path = EBNERD.raw_dir / split / "history.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["user_id", "article_id_fixed"])
    frame["user_id"] = frame["user_id"].astype("uint32")
    stamp = pd.Timestamp("2023-05-20 09:00:00")
    frame["impression_time_fixed"] = [
        [stamp + pd.Timedelta(hours=i) for i in range(len(clicks))]
        for clicks in frame["article_id_fixed"]
    ]
    frame["read_time_fixed"] = [
        [10.0 * (i + 1) for i in range(len(clicks))]
        for clicks in frame["article_id_fixed"]
    ]
    frame["scroll_percentage_fixed"] = [
        [50.0] * len(clicks) for clicks in frame["article_id_fixed"]
    ]
    frame.to_parquet(path)


def test_ebnerd_history_is_joined_onto_every_impression(raw):
    stamp = pd.Timestamp("2023-05-23 07:31:00")
    write_ebnerd_behaviors(
        "train", [(7, 55, stamp, [11, 12], [12]), (9, 55, stamp, [13], [13])]
    )
    write_ebnerd_behaviors("validation", [(8, 56, stamp, [21], [])])
    write_ebnerd_history("train", [(55, [1, 2, 3])])
    # User 56 has an impression but no history row at all.
    write_ebnerd_history("validation", [])

    behaviors = ingest.build_behaviors(EBNERD)
    history = ingest.build_history(EBNERD, behaviors)

    assert list(history.columns) == list(HISTORY_COLUMNS)
    # One history row per impression, not per user.
    assert len(history) == len(behaviors)
    assert set(history["impression_id"]) == set(behaviors["impression_id"])

    warm = history[history["user_id"] == "55"]
    assert len(warm) == 2
    assert warm.iloc[0]["click_history"] == ["1", "2", "3"]
    assert warm.iloc[0]["n_clicks"] == 3

    cold = history[history["user_id"] == "56"].iloc[0]
    assert cold["click_history"] == []
    assert cold["n_clicks"] == 0


def test_dangling_article_references_are_counted_not_dropped(raw):
    """The catalogue does not cover every id the logs mention."""
    write_mind_news("train", [("N1", "sports", "football", "Title", "Abstract")])
    write_mind_news("dev", [("N2", "news", "world", "Other", "Other abstract")])
    write_mind_behaviors(
        "train", [("1", "U1", "11/11/2019 9:05:58 AM", "N1 N9", "N1-1 N3-0")]
    )
    write_mind_behaviors("dev", [("1", "U2", "11/15/2019 7:00:00 PM", "N2", "N2-1")])

    articles = ingest.build_articles(MIND)
    behaviors = ingest.build_behaviors(MIND)
    history = ingest.build_history(MIND, behaviors)
    report = ingest.count_dangling(articles, behaviors, history)

    # N3 is offered as a candidate but is not in the catalogue.
    assert report["candidates_total"] == 3
    assert report["candidates_missing"] == 1
    # N9 was clicked historically but is not in the catalogue either.
    assert report["clicks_total"] == 3
    assert report["clicks_missing"] == 1

    # Counted, not dropped: every impression survives.
    assert len(behaviors) == 2
    assert behaviors[behaviors["user_id"] == "U1"].iloc[0]["candidate_ids"] == [
        "N1",
        "N3",
    ]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    return tmp_path / "feature_store"


def build_mind_fixtures():
    write_mind_news("train", [("N1", "sports", "football", "Title", "Abstract")])
    write_mind_news("dev", [("N2", "news", "world", "Other", "Other abstract")])
    write_mind_behaviors(
        "train", [("1", "U1", "11/11/2019 9:05:58 AM", "N1", "N1-1 N2-0")]
    )
    write_mind_behaviors("dev", [("1", "U2", "11/15/2019 7:00:00 PM", "N2", "N2-1")])


def test_run_writes_the_three_tables_and_reports_row_counts(raw, store, capsys):
    build_mind_fixtures()

    ingest.run(MIND)

    written = {
        name: pd.read_parquet(MIND.feature_store_dir / f"{name}.parquet")
        for name in ("articles", "behaviors", "history")
    }
    assert list(written["articles"].columns) == list(ARTICLE_COLUMNS)
    assert list(written["behaviors"].columns) == list(BEHAVIOR_COLUMNS)
    assert list(written["history"].columns) == list(HISTORY_COLUMNS)
    assert len(written["articles"]) == 2
    assert len(written["behaviors"]) == 2
    assert len(written["history"]) == 2

    # Row counts are reported, so a change that quietly loses rows is visible.
    printed = capsys.readouterr().out
    assert "articles" in printed and "behaviors" in printed and "history" in printed
    assert "2" in printed


def test_run_skips_an_existing_feature_store_unless_forced(raw, store, monkeypatch):
    build_mind_fixtures()
    ingest.run(MIND)

    rebuilt = []
    real_build = ingest.build_articles

    def counted(config):
        rebuilt.append(config.name)
        return real_build(config)

    monkeypatch.setattr(ingest, "build_articles", counted)

    ingest.run(MIND)
    assert rebuilt == []

    ingest.run(MIND, force=True)
    assert rebuilt == ["mind"]


def downloaded(config):
    return all((config.raw_dir / name).exists() for name in config.raw.expected_files)


@pytest.mark.parametrize("config", [MIND, EBNERD], ids=lambda c: c.name)
def test_real_downloaded_data_ingests(config):
    """Guards against surprises no hand-written fixture would contain."""
    if not downloaded(config):
        pytest.skip(f"{config.name} raw data not downloaded")

    articles = ingest.build_articles(config)
    behaviors = ingest.build_behaviors(config)
    history = ingest.build_history(config, behaviors)

    assert articles["article_id"].is_unique
    assert behaviors["impression_id"].is_unique
    assert not articles["article_id"].isna().any()
    # Real datetimes, and naive throughout so no comparison downstream can
    # silently mix an aware and a naive timestamp.
    assert behaviors["impression_time"].dtype == "datetime64[us]"
    assert behaviors["impression_time"].dt.tz is None
    assert not behaviors["impression_time"].isna().any()
    assert len(history) == len(behaviors)
    assert (
        behaviors["candidate_ids"].map(len) == behaviors["labels"].map(len)
    ).all()


def test_history_is_matched_within_its_own_split(raw):
    """EB-NeRD repeats a user across split files, with a different history each
    time — matching on user alone would fan every impression out across both."""
    stamp = pd.Timestamp("2023-05-23 07:31:00")
    write_ebnerd_behaviors("train", [(7, 55, stamp, [11], [11])])
    write_ebnerd_behaviors("validation", [(8, 55, stamp, [21], [21])])
    write_ebnerd_history("train", [(55, [1, 2])])
    write_ebnerd_history("validation", [(55, [1, 2, 3, 4])])

    behaviors = ingest.build_behaviors(EBNERD)
    history = ingest.build_history(EBNERD, behaviors)

    assert len(history) == len(behaviors) == 2
    by_impression = history.set_index("impression_id")
    assert by_impression.loc["train-7", "n_clicks"] == 2
    assert by_impression.loc["validation-8", "n_clicks"] == 4


def test_both_datasets_produce_the_same_history_schema(raw):
    build_mind_fixtures()
    stamp = pd.Timestamp("2023-05-23 07:31:00")
    write_ebnerd_behaviors("train", [(7, 55, stamp, [11], [11])])
    write_ebnerd_behaviors("validation", [(8, 56, stamp, [21], [])])
    write_ebnerd_history("train", [(55, [1, 2])])
    write_ebnerd_history("validation", [(56, [3])])

    mind = ingest.build_history(MIND, ingest.build_behaviors(MIND))
    ebnerd = ingest.build_history(EBNERD, ingest.build_behaviors(EBNERD))

    assert list(mind.columns) == list(ebnerd.columns)
    assert mind.dtypes.to_dict() == ebnerd.dtypes.to_dict()

    # The asymmetry the registry describes: EB-NeRD has the engagement
    # columns, MIND has them as null. Null and not missing, so a stage reading
    # them needs no branch on the dataset's name -- it asks whether the column
    # holds anything, which is a question about data rather than provenance.
    for column in ("click_times", "click_read_times", "click_scroll"):
        assert mind[column].isna().all(), f"MIND cannot have {column}"
        assert ebnerd[column].map(len).sum() > 0, f"EB-NeRD should have {column}"


def test_ebnerds_engagement_columns_line_up_with_the_clicks_they_describe(raw):
    """A weighting scheme takes the last K clicks and the last K of a parallel
    array and pairs them by position. If the arrays came back a different
    length, or in a different order, every weight would land on the wrong
    click and the profile would look plausible while being wrong."""
    stamp = pd.Timestamp("2023-05-23 07:31:00")
    write_ebnerd_behaviors("train", [(7, 55, stamp, [11], [11])])
    write_ebnerd_behaviors("validation", [(8, 56, stamp, [21], [])])
    write_ebnerd_history("train", [(55, [1, 2, 3])])
    write_ebnerd_history("validation", [(56, [3])])

    history = ingest.build_history(EBNERD, ingest.build_behaviors(EBNERD))

    row = history[history["user_id"] == "55"].iloc[0]
    assert len(row["click_history"]) == 3
    for column in ("click_times", "click_read_times", "click_scroll"):
        assert len(row[column]) == len(row["click_history"])
    # The fixture writes one hour later and ten seconds longer per click, in
    # click order, so the arrays are ordered rather than merely the right size.
    assert list(row["click_read_times"]) == [10.0, 20.0, 30.0]
    assert row["click_times"][0] < row["click_times"][-1]


def test_the_engagement_arrays_are_capped_and_keep_the_most_recent_clicks(raw):
    """Untruncated, these arrays are copied once per impression -- a 31x
    duplication on EB-NeRD -- and the first version of this was OOM-killed at
    10 GB. The cap is what makes it affordable, and it has to keep the *recent*
    end: every consumer reads the last k clicks, so dropping the tail rather
    than the head would leave the arrays describing clicks nobody looks at.

    `click_history` itself stays whole, because `n_clicks` counts all of it.
    """
    stamp = pd.Timestamp("2023-05-23 07:31:00")
    long_history = list(range(sources.ENGAGEMENT_WINDOW + 40))
    write_ebnerd_behaviors("train", [(7, 55, stamp, [11], [11])])
    write_ebnerd_behaviors("validation", [(8, 56, stamp, [21], [])])
    write_ebnerd_history("train", [(55, long_history)])
    write_ebnerd_history("validation", [(56, [3])])

    history = ingest.build_history(EBNERD, ingest.build_behaviors(EBNERD))
    row = history[history["user_id"] == "55"].iloc[0]

    assert len(row["click_history"]) == len(long_history), "ids are not truncated"
    assert row["n_clicks"] == len(long_history)
    for column in ("click_times", "click_read_times", "click_scroll"):
        assert len(row[column]) == sources.ENGAGEMENT_WINDOW

    # The fixture's read times count up in click order, so the kept window has
    # to end on the last click rather than start on the first.
    last = 10.0 * len(long_history)
    assert row["click_read_times"][-1] == last
    # And the last k of each array line up, which is the whole alignment rule.
    k = 5
    assert list(row["click_read_times"][-k:]) == [
        10.0 * (len(long_history) - i) for i in range(k - 1, -1, -1)
    ]


def test_a_user_with_no_history_row_gets_empty_parallel_arrays(raw):
    """The cold user arrives through a right join with nothing on the left. Its
    click list is empty rather than null, and the arrays beside it have to
    agree -- a null there would break a `zip` that an empty list satisfies."""
    stamp = pd.Timestamp("2023-05-23 07:31:00")
    write_ebnerd_behaviors("train", [(7, 55, stamp, [11], [11])])
    write_ebnerd_behaviors("validation", [(8, 56, stamp, [21], [])])
    write_ebnerd_history("train", [])
    write_ebnerd_history("validation", [])

    history = ingest.build_history(EBNERD, ingest.build_behaviors(EBNERD))

    row = history.iloc[0]
    assert row["click_history"] == []
    for column in ("click_times", "click_read_times", "click_scroll"):
        assert len(row[column]) == 0
