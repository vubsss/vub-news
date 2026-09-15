"""The feature frame is tested at the seams where the future could get in --
the causality of every log read (Q9), the forbidden outcome columns, the tier
partition an ablation arm depends on -- and then at the three places a
well-formed frame could still be wrong: scores read back by position instead of
by id, rows that are not one per candidate, and a projection that quietly
returns something other than what was asked for."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pipeline import bm25_index, counters, embed, features, ingest, ledger, paths
from pipeline.datasets import DATASETS, FeatureSpec

MIND = DATASETS["mind"]
EBNERD = DATASETS["ebnerd"]

T0 = pd.Timestamp("2019-11-14 10:00:00")

# Five articles in three categories, as five unit vectors in the plane, so
# every cosine in these tests is one a reader can check by hand.
ARTICLES = (
    ("a1", "sports", "s1", (1.0, 0.0)),
    ("a2", "sports", "s1", (0.8, 0.6)),
    ("a3", "finance", "f1", (0.0, 1.0)),
    ("a4", "finance", "f1", (0.6, 0.8)),
    ("a5", "tech", "t1", (2.0**-0.5, 2.0**-0.5)),
)

# (impression, user, hours after T0, candidates, labels, split, session)
IMPRESSIONS = (
    ("i1", "u1", 0, ["a1", "a3"], [1, 0], "train", "sess-1"),
    ("i2", "u2", 1, ["a2", "a4"], [0, 1], "train", "sess-2"),
    ("i3", "u1", 2, ["a3", "a1", "a5"], [0, 1, 0], "tune", "sess-1"),
    ("i4", "u2", 3, ["a1", "a5"], [1, 0], "validation", "sess-2"),
    ("i5", "u3", 3, ["a4", "a2"], [0, 1], "validation", "sess-3"),
)

# Each user's past clicks, oldest first, with the engagement arrays EB-NeRD
# ships alongside them and MIND does not.
HISTORIES = {
    "u1": ["a2", "a4"],
    "u2": ["a1"],
    "u3": [],
}


def _articles(config):
    published = [
        T0 - pd.Timedelta(hours=position + 1) for position in range(len(ARTICLES))
    ]
    return pd.DataFrame(
        {
            "article_id": pd.Series([row[0] for row in ARTICLES], dtype="string"),
            "title": pd.Series(
                [f"headline about {row[1]}" for row in ARTICLES], dtype="string"
            ),
            "abstract": pd.Series([""] * len(ARTICLES), dtype="string"),
            "lexical_text": pd.Series(
                [f"headline {row[1]} {row[0]}" for row in ARTICLES], dtype="string"
            ),
            "category": pd.Series([row[1] for row in ARTICLES], dtype="string"),
            "subcategory": pd.Series([row[2] for row in ARTICLES], dtype="string"),
            "published_time": pd.Series(
                published if config.columns.articles["published_time"] else [pd.NaT] * 5,
                dtype="datetime64[us]",
            ),
        }
    )


def _behaviors(config, rows=IMPRESSIONS):
    sessions = (
        [row[6] for row in rows]
        if config.columns.behaviors["session_id"]
        else [None] * len(rows)
    )
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
            "user_id": pd.Series([row[1] for row in rows], dtype="string"),
            "source": pd.Series(["train"] * len(rows), dtype="string"),
            "impression_time": pd.Series(
                [T0 + pd.Timedelta(hours=row[2]) for row in rows],
                dtype="datetime64[us]",
            ),
            "session_id": pd.Series(sessions, dtype="string"),
            "candidate_ids": [list(row[3]) for row in rows],
            "labels": [list(row[4]) for row in rows],
            "split": pd.Series([row[5] for row in rows], dtype="string"),
        }
    )


def _history(config):
    engaged = bool(config.columns.history["click_read_times"])
    timed = bool(config.columns.history["click_times"])
    rows = []
    for user, clicks in HISTORIES.items():
        rows.append(
            {
                "user_id": user,
                "source": "train",
                "click_history": list(clicks),
                "click_times": (
                    np.array(
                        [T0 - pd.Timedelta(hours=len(clicks) - i) for i in range(len(clicks))],
                        dtype="datetime64[us]",
                    )
                    if timed
                    else None
                ),
                "click_read_times": (
                    np.array([20.0 + 10 * i for i in range(len(clicks))], dtype="float32")
                    if engaged
                    else None
                ),
                "click_scroll": (
                    np.array([60.0 + 10 * i for i in range(len(clicks))], dtype="float32")
                    if engaged
                    else None
                ),
                "n_clicks": len(clicks),
            }
        )
    frame = pd.DataFrame(rows)
    frame["user_id"] = frame["user_id"].astype("string")
    frame["source"] = frame["source"].astype("string")
    return frame


def write_store(config, rows=IMPRESSIONS):
    """A whole small dataset: the three feature-store tables, the embedding
    artifact, the BM25 index and the counter store — everything a frame reads."""
    store = config.feature_store_dir
    store.mkdir(parents=True, exist_ok=True)
    _articles(config).to_parquet(store / "articles.parquet", index=False)
    behaviors = _behaviors(config, rows)
    behaviors.to_parquet(store / "behaviors.parquet", index=False)
    _history(config).to_parquet(store / "history.parquet", index=False)

    embed.Embeddings(
        vectors=np.array([row[3] for row in ARTICLES], dtype="float32"),
        article_ids=np.array([row[0] for row in ARTICLES], dtype=object),
    ).save(embed.output_dir(config))
    bm25_index.run(config)
    counters.build(behaviors).save(config.artifacts_dir / counters.DIRECTORY)
    return behaviors


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path


def chunk_of(config, behaviors, split):
    """One split's impressions and the history rows paired with them."""
    chunk = behaviors[behaviors["split"] == split].reset_index(drop=True)
    return chunk, ingest.history_for(config, chunk)


def rows_of(frame, impression):
    return frame[frame["impression_id"] == impression].reset_index(drop=True)


# --- the schema and its guards ---------------------------------------------


def test_every_feature_belongs_to_exactly_one_availability_tier():
    """What ticket 07's ablation depends on: an arm drops a tier, so a feature
    in no tier is one no arm can drop and a feature in two is one that two arms
    would each report having dropped."""
    placed = [name for group in features.FEATURE_GROUPS.values() for name in group]
    assert sorted(placed) == sorted(set(placed))
    assert set(placed) == set(features.FEATURES)
    for name in features.FEATURES:
        assert features.tier_of(name) in features.FEATURE_GROUPS
    features.check_columns(features.COLUMNS)


def test_an_outcome_column_fails_the_build():
    """Q9's named columns, and anything ending in one of them: a feature called
    `mean_read_time` is the impression's own dwell under another name."""
    for forbidden in features.FORBIDDEN:
        with pytest.raises(features.FeatureError, match="own outcome"):
            features.check_columns([*features.COLUMNS, forbidden])
        with pytest.raises(features.FeatureError, match="own outcome"):
            features.check_columns([*features.COLUMNS, f"mean_{forbidden}"])


def test_a_frame_that_drifted_from_the_schema_is_refused():
    with pytest.raises(features.FeatureError, match="not the schema"):
        features.check_columns(features.COLUMNS[:-1])
    with pytest.raises(features.FeatureError, match="not the schema"):
        features.check_columns([*features.COLUMNS, "invented"])


def test_an_arm_selects_whole_tiers_and_nothing_else():
    chosen = features.columns_for(["content", "clicked"])
    assert set(chosen) == {
        *features.KEY_COLUMNS,
        *features.FEATURE_GROUPS["content"],
        *features.FEATURE_GROUPS["clicked"],
    }
    assert not set(chosen) & set(features.FEATURE_GROUPS["history"])
    with pytest.raises(features.FeatureError, match="not availability tiers"):
        features.columns_for(["content", "invented"])


# --- reading a retriever's scores back -------------------------------------


def test_scores_are_read_back_through_ranked_ids_not_by_position():
    """Both retrievers emit their candidates sorted by score. Zipping those
    scores onto the arrival order gives every candidate another's number in a
    frame with the right shape and the right row count, which is exactly the
    bug this reads through `ranked_ids` to avoid."""
    chunk = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "candidate_ids": [["a1", "a2", "a3"]],
        }
    )
    ranked = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "ranked_ids": [["a3", "a1", "a2"]],
            "scores": [[0.9, 0.5, 0.1]],
        }
    )
    scores, ranks = features.read_back(ranked, chunk)
    assert list(scores) == [0.5, 0.1, 0.9]
    assert list(ranks) == [2.0, 3.0, 1.0]


def test_a_candidate_the_retriever_did_not_rank_is_null_not_zero():
    chunk = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "candidate_ids": [["a1", "a9"]],
        }
    )
    ranked = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "ranked_ids": [["a1"]],
            "scores": [[0.5]],
        }
    )
    scores, ranks = features.read_back(ranked, chunk)
    assert scores[0] == 0.5 and np.isnan(scores[1])
    assert np.isnan(ranks[1])


def test_an_impression_that_came_back_without_a_ranking_is_an_error():
    chunk = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1", "i2"], dtype="string"),
            "candidate_ids": [["a1"], ["a2"]],
        }
    )
    ranked = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "ranked_ids": [["a1"]],
            "scores": [[0.5]],
        }
    )
    with pytest.raises(features.FeatureError, match="without a ranking"):
        features.read_back(ranked, chunk)


# --- sessions: strictly earlier, and null where there are none -------------


def test_session_counts_see_only_what_came_earlier_in_the_session():
    behaviors = _behaviors(EBNERD)
    counted = features.session_counts(behaviors)
    # i1 opens sess-1 and i3 follows it; i1 clicked one article.
    assert counted.loc["i1", "session_impressions"] == 0
    assert counted.loc["i3", "session_impressions"] == 1
    assert counted.loc["i3", "session_clicks"] == 1
    # i5 is alone in its session and sees nothing, though it is stamped at the
    # same instant as i4 in another.
    assert counted.loc["i5", "session_impressions"] == 0


def test_two_impressions_at_the_same_instant_do_not_see_each_other():
    rows = (
        ("j1", "u1", 0, ["a1"], [1], "train", "sess-1"),
        ("j2", "u1", 0, ["a2"], [1], "train", "sess-1"),
        ("j3", "u1", 1, ["a3"], [0], "train", "sess-1"),
    )
    counted = features.session_counts(_behaviors(EBNERD, rows))
    assert counted.loc["j1", "session_impressions"] == 0
    assert counted.loc["j2", "session_impressions"] == 0
    assert counted.loc["j3", "session_impressions"] == 2
    assert counted.loc["j3", "session_clicks"] == 2


def test_a_dataset_with_no_sessions_counts_nothing_rather_than_zero():
    counted = features.session_counts(_behaviors(MIND))
    assert counted["session_impressions"].isna().all()
    assert counted["session_clicks"].isna().all()


# --- Q9: the frame does not move when the future is removed ----------------


@pytest.mark.parametrize("config", [MIND, EBNERD], ids=["mind", "ebnerd"])
def test_removing_every_row_after_t_leaves_the_frame_identical(store, config):
    """The leakage test the assignment asks for, on both datasets — EB-NeRD
    because it is the one with sessions, which are the other feature family
    that reads the log rather than the user. Rebuilding the frame from a log
    with every impression at or after `t` removed must leave the rows built at
    earlier moments bit-identical: if any feature moved, something in it was
    reading a click that had not happened yet."""
    behaviors = write_store(config)
    chunk, history = chunk_of(config, behaviors, "tune")
    whole = features.frame_for(config, chunk, history, features.load(config, behaviors))

    cut = T0 + pd.Timedelta(hours=2, seconds=1)
    truncated = behaviors[behaviors["impression_time"] < cut]
    counters.build(truncated).save(config.artifacts_dir / counters.DIRECTORY)
    without_future = features.frame_for(
        config, chunk, history, features.load(config, truncated)
    )

    pd.testing.assert_frame_equal(whole, without_future)


def test_a_feature_family_that_emitted_an_outcome_column_fails_the_build(
    store, monkeypatch
):
    """The guard is wired into the build rather than merely available to it.
    A family that started returning the impression's own dwell — the shape the
    mistake would actually take — stops the frame instead of shipping it."""
    write_store(MIND)
    honest = features.category_columns

    def leaking(*arguments, **keywords):
        columns = honest(*arguments, **keywords)
        return {**columns, "read_time": next(iter(columns.values()))}

    monkeypatch.setattr(features, "category_columns", leaking)
    with pytest.raises(features.FeatureError, match="own outcome"):
        features.build(MIND, "validation")


def test_the_leaky_arm_sees_what_the_causal_one_cannot(store):
    """Q9's pair, from one store and one flag. The leaky counters include the
    impression's own outcome, so they dominate the causal read everywhere and
    exceed it somewhere — if they did not, the arm would not be measuring
    anything."""
    behaviors = write_store(MIND)
    chunk, history = chunk_of(MIND, behaviors, "tune")
    loaded = features.load(MIND, behaviors)
    causal = features.frame_for(MIND, chunk, history, loaded, causal=True)
    leaky = features.frame_for(MIND, chunk, history, loaded, causal=False)

    for column in ("exposures_cumulative", "clicks_cumulative"):
        assert (leaky[column] >= causal[column]).all()
        assert (leaky[column] > causal[column]).any()
    # a5 is first shown by this very impression, so only the leaky arm has a
    # freshness for it at all.
    a5 = leaky["article_id"] == "a5"
    assert causal.loc[a5, "freshness_hours"].isna().all()
    assert leaky.loc[a5, "freshness_hours"].notna().all()


def test_a_counter_feature_counts_what_happened_before_the_impression(store):
    behaviors = write_store(MIND)
    chunk, history = chunk_of(MIND, behaviors, "validation")
    frame = features.frame_for(MIND, chunk, history, features.load(MIND, behaviors))

    # a1 was shown at 10:00 (clicked) and at 12:00 (clicked); i4 is at 13:00.
    i4 = rows_of(frame, "i4")
    assert i4.loc[0, "article_id"] == "a1"
    assert i4.loc[0, "exposures_cumulative"] == 2
    assert i4.loc[0, "clicks_cumulative"] == 2
    assert i4.loc[0, "ctr_cumulative"] == pytest.approx(1.0)
    # The hour back from 13:00 is [12:00, 13:00), which holds the second of
    # them and not the first.
    assert i4.loc[0, "exposures_1h"] == 1
    assert i4.loc[0, "clicks_1h"] == 1
    # a5 was shown at 12:00 too but never clicked, so its rate in the window
    # is a measured zero.
    assert i4.loc[1, "exposures_1h"] == 1
    assert i4.loc[1, "ctr_1h"] == 0.0
    # a4's only exposure was at 11:00, outside the window, so the window has
    # nothing to take a rate over and the rate is NaN rather than zero — a
    # distinction the re-ranker can act on.
    i5 = rows_of(frame, "i5")
    assert i5.loc[0, "article_id"] == "a4"
    assert i5.loc[0, "exposures_1h"] == 0
    assert np.isnan(i5.loc[0, "ctr_1h"])
    # MIND has no publish time, so freshness is measured from the log's first
    # sighting of the article: 10:00, three hours earlier.
    assert i4.loc[0, "freshness_hours"] == pytest.approx(3.0)


def test_freshness_comes_from_publish_time_where_the_registry_maps_one(store):
    behaviors = write_store(EBNERD)
    chunk, history = chunk_of(EBNERD, behaviors, "validation")
    frame = features.frame_for(EBNERD, chunk, history, features.load(EBNERD, behaviors))
    i4 = rows_of(frame, "i4")
    # a1 was published an hour before T0 and i4 is served three hours after it.
    assert i4.loc[0, "freshness_hours"] == pytest.approx(4.0)


# --- the history features ---------------------------------------------------


def test_the_cosine_columns_are_the_similarities_to_the_users_clicks(store):
    """u2's only click is a1 = (1, 0), so every profile over that one click is
    the plain cosine and the three poolings agree."""
    behaviors = write_store(MIND)
    chunk, history = chunk_of(MIND, behaviors, "validation")
    frame = features.frame_for(MIND, chunk, history, features.load(MIND, behaviors))
    i4 = rows_of(frame, "i4")

    assert list(i4["article_id"]) == ["a1", "a5"]
    assert i4["hist_cos_mean_uniform"].tolist() == pytest.approx([1.0, 2**-0.5])
    assert i4["hist_cos_max_uniform"].tolist() == pytest.approx([1.0, 2**-0.5])
    assert i4["hist_cos_last"].tolist() == pytest.approx([1.0, 2**-0.5])
    assert i4["n_clicks"].tolist() == [1.0, 1.0]
    # The candidate's category against the user's recent clicks: a1 is sports
    # and so was the only click; a5 is tech and was not.
    assert i4["category_share"].tolist() == pytest.approx([1.0, 0.0])
    assert i4["subcategory_share"].tolist() == pytest.approx([1.0, 0.0])


def test_a_cold_user_has_no_history_features_rather_than_zeroed_ones(store):
    behaviors = write_store(MIND)
    chunk, history = chunk_of(MIND, behaviors, "validation")
    frame = features.frame_for(MIND, chunk, history, features.load(MIND, behaviors))
    i5 = rows_of(frame, "i5")
    assert i5["n_clicks"].tolist() == [0.0, 0.0]
    for column in ("hist_cos_mean_uniform", "hist_cos_last", "category_share"):
        assert i5[column].isna().all()


def test_mind_carries_the_columns_it_has_no_source_for_as_null(store):
    """The registry's asymmetry, straight through to the frame: no sessions, no
    click timestamps, no engagement — so those columns are NaN and the build
    does not notice which dataset it is holding."""
    behaviors = write_store(MIND)
    chunk, history = chunk_of(MIND, behaviors, "validation")
    frame = features.frame_for(MIND, chunk, history, features.load(MIND, behaviors))

    for column in (
        "session_impressions",
        "session_clicks",
        "past_dwell",
        "past_depth",
        "hist_cos_mean_t24",
        "hist_cos_max_engagement",
    ):
        assert frame[column].isna().all(), column
    # And the ones it can express are filled.
    assert frame["hist_cos_mean_pos90"].notna().any()


def test_ebnerd_fills_the_session_and_engagement_columns(store):
    behaviors = write_store(EBNERD)
    chunk, history = chunk_of(EBNERD, behaviors, "validation")
    frame = features.frame_for(EBNERD, chunk, history, features.load(EBNERD, behaviors))

    i4 = rows_of(frame, "i4")
    assert i4["session_impressions"].tolist() == [1.0, 1.0]
    assert i4["session_clicks"].tolist() == [1.0, 1.0]
    # u2's one past click was read for 20 seconds and scrolled to 60%.
    assert i4["past_dwell"].tolist() == pytest.approx([20.0, 20.0])
    assert i4["past_depth"].tolist() == pytest.approx([60.0, 60.0])
    for column in ("hist_cos_mean_t24", "hist_cos_max_engagement"):
        assert i4[column].notna().all()


def test_a_profile_the_dataset_cannot_express_is_left_out_rather_than_faked():
    assert [profile.name for profile in features.profiles_for(MIND, 80)] == [
        "uniform",
        "pos90",
        "pos99",
    ]
    assert len(features.profiles_for(EBNERD, 80)) == len(features.PROFILES)


# --- the frame's shape ------------------------------------------------------


def test_the_frame_is_one_row_per_candidate_in_arrival_order(store):
    behaviors = write_store(MIND)
    report = features.build(MIND, "validation")

    frame = features.read(MIND, "validation")
    assert report["rows"] == len(frame) == 4
    for impression, candidates, labels in zip(
        behaviors["impression_id"], behaviors["candidate_ids"], behaviors["labels"]
    ):
        rows = rows_of(frame, impression)
        if rows.empty:
            continue
        assert list(rows["article_id"]) == list(candidates)
        assert list(rows["label"]) == list(labels)
    assert list(frame.columns) == list(features.COLUMNS)


def test_a_split_with_no_impressions_is_refused(store):
    write_store(MIND)
    with pytest.raises(features.FeatureError, match="no test impressions"):
        features.build(MIND, "test")


# --- storage: row groups, projection, precision -----------------------------


def test_a_chunk_is_written_as_its_own_row_group(store):
    """The frame is the biggest thing A2 materialises and is built again per
    chunk at submission time, so it is never assembled whole in memory: each
    chunk is appended as a row group and dropped."""
    write_store(MIND)
    spec = FeatureSpec(chunk_impressions=1)
    report = features.build(MIND, "train", spec)
    assert pq.ParquetFile(report["path"]).num_row_groups == 2
    assert features.read(MIND, "train").shape[0] == report["rows"] == 4


def test_a_projected_read_takes_only_the_arms_columns(store):
    write_store(MIND)
    features.build(MIND, "validation")

    projected = features.read(
        MIND, "validation", columns=features.columns_for(["content"])
    )
    assert list(projected.columns) == list(features.columns_for(["content"]))
    for forbidden in features.FORBIDDEN:
        assert forbidden not in projected.columns
        with pytest.raises(features.FeatureError, match="not feature-frame columns"):
            features.read(MIND, "validation", columns=(forbidden,))


def test_float16_halves_the_frame_and_keeps_the_values(store):
    """One of ticket 04's two storage options. The bytes are this ticket's
    column; whether the precision costs any AUC is ticket 06's, measured by
    training on both files rather than argued here."""
    write_store(MIND)
    wide = features.build(MIND, "validation", FeatureSpec(precision="float32"))
    narrow = features.build(MIND, "validation", FeatureSpec(precision="float16"))
    assert narrow["bytes"] < wide["bytes"]

    frame = features.read(MIND, "validation")
    assert frame["ann_score"].dtype == np.float16
    assert frame["hist_cos_mean_uniform"].astype("float32").tolist() == pytest.approx(
        [1.0, 2**-0.5, np.nan, np.nan], nan_ok=True, abs=1e-3
    )


def test_the_leaky_arm_is_written_beside_the_causal_one_not_over_it(store):
    write_store(MIND)
    features.build(MIND, "tune")
    features.build(MIND, "tune", causal=False)
    assert features.path_for(MIND, "tune").exists()
    assert features.path_for(MIND, "tune", causal=False).exists()
    assert (
        features.read(MIND, "tune", causal=False)["exposures_cumulative"].sum()
        > features.read(MIND, "tune")["exposures_cumulative"].sum()
    )


# --- the stage --------------------------------------------------------------


def test_the_stage_records_a_ledger_row_per_split(store, capsys):
    write_store(MIND)
    features.run(MIND)

    rows = {row["split"]: row for row in ledger.load() if row["stage"] == features.STAGE}
    assert set(rows) == set(features.SPLITS)
    for split, row in rows.items():
        assert row["dataset"] == "mind"
        assert row["variant"] == "float32-rg512k"
        assert row["feature_bytes"] > 0
        assert row["train_seconds"] >= 0 and row["peak_rss_mb"] > 0
        assert row["rows_per_s"] > 0
        # Which family cost what, so the note can say where the time went.
        assert "retrievers" in row["note"] and "counters" in row["note"]
        # The functional side is deliberately blank: a frame has no AUC.
        assert row["auc"] is None
    assert (paths.ARTIFACTS_DIR / ledger.DOCUMENT).exists()


def test_a_second_run_leaves_the_frames_alone_unless_forced(store, capsys):
    write_store(MIND)
    features.run(MIND)
    built = features.path_for(MIND, "validation").stat().st_mtime_ns

    features.run(MIND)
    assert "already built" in capsys.readouterr().out
    assert features.path_for(MIND, "validation").stat().st_mtime_ns == built

    features.run(MIND, force=True)
    assert features.path_for(MIND, "validation").stat().st_mtime_ns != built
