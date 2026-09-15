"""The re-ranker is tested where a fifth retriever can be wrong without any
metric saying so: the half it is fitted on, the columns an arm actually reads,
the NRMS column joined by pair rather than by position, the group boundaries
`lambdarank` draws, the literal top-K cut, and the headline delta being a
paired one."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pipeline import (
    bm25_index,
    counters,
    embed,
    evaluate,
    features,
    ingest,
    ledger,
    nrms,
    paths,
    rerank,
    retrieval,
)
from pipeline.datasets import DATASETS, NrmsSpec, RerankSpec

MIND = DATASETS["mind"]

T0 = pd.Timestamp("2019-11-11 06:00:00")

ARTICLES = (
    ("a1", "sports", "s1", (1.0, 0.0)),
    ("a2", "sports", "s1", (0.87, 0.5)),
    ("a3", "finance", "f1", (0.0, 1.0)),
    ("a4", "finance", "f1", (0.5, 0.87)),
    ("a5", "tech", "t1", (2.0**-0.5, 2.0**-0.5)),
    ("a6", "tech", "t1", (-1.0, 0.0)),
)

# Small enough to train in a moment, and the same model either way.
SMALL_NRMS = NrmsSpec(
    history_length=4,
    heads=2,
    head_dim=4,
    attention_dim=8,
    dropout=0.0,
    negatives=2,
    epochs=1,
    batch_size=8,
    score_batch=8,
)
SMALL_RERANK = RerankSpec(
    leaves=4, rounds=12, min_data_in_leaf=1, early_stopping=5, learning_rate=0.2
)


def impressions_of(count=16):
    """A train split over distinct hours, then tune and validation after it."""
    rows = []
    for i in range(count):
        sporty = i % 2 == 0
        rows.append(
            (
                f"t{i}",
                f"u{i % 3}",
                i,
                ["a1", "a3", "a5"] if sporty else ["a2", "a4", "a6"],
                [1, 0, 0] if sporty else [0, 1, 0],
                "train",
            )
        )
    for i in range(4):
        rows.append(
            (
                f"n{i}",
                f"u{i % 3}",
                count + i,
                ["a1", "a4", "a6"],
                [1, 0, 0] if i % 2 == 0 else [0, 1, 0],
                "tune",
            )
        )
    for i in range(4):
        rows.append(
            (
                f"v{i}",
                f"u{i % 3}",
                count + 8 + i,
                ["a2", "a3", "a5"],
                [1, 0, 0] if i % 2 == 0 else [0, 0, 1],
                "validation",
            )
        )
    return pd.DataFrame(
        {
            "impression_id": pd.Series([r[0] for r in rows], dtype="string"),
            "user_id": pd.Series([r[1] for r in rows], dtype="string"),
            "source": pd.Series(["train"] * len(rows), dtype="string"),
            "impression_time": pd.Series(
                [T0 + pd.Timedelta(hours=r[2]) for r in rows], dtype="datetime64[us]"
            ),
            "session_id": pd.Series([None] * len(rows), dtype="string"),
            "candidate_ids": [list(r[3]) for r in rows],
            "labels": [list(r[4]) for r in rows],
            "split": pd.Series([r[5] for r in rows], dtype="string"),
        }
    )


def write_store(config=MIND):
    """The whole pipeline's outputs for a small log: the three tables, the
    vectors, the BM25 index, the counters, and the three feature frames."""
    store = config.feature_store_dir
    store.mkdir(parents=True, exist_ok=True)
    behaviors = impressions_of()
    behaviors.to_parquet(store / "behaviors.parquet", index=False)
    pd.DataFrame(
        {
            "article_id": pd.Series([a[0] for a in ARTICLES], dtype="string"),
            "title": pd.Series([f"headline {a[1]}" for a in ARTICLES], dtype="string"),
            "abstract": pd.Series([""] * len(ARTICLES), dtype="string"),
            "lexical_text": pd.Series(
                [f"headline {a[1]} {a[0]}" for a in ARTICLES], dtype="string"
            ),
            "category": pd.Series([a[1] for a in ARTICLES], dtype="string"),
            "subcategory": pd.Series([a[2] for a in ARTICLES], dtype="string"),
            "published_time": pd.Series([pd.NaT] * len(ARTICLES), dtype="datetime64[us]"),
        }
    ).to_parquet(store / "articles.parquet", index=False)
    users = {"u0": ["a1", "a3"], "u1": ["a2", "a4"], "u2": ["a5"]}
    pd.DataFrame(
        {
            "user_id": pd.Series(list(users), dtype="string"),
            "source": pd.Series(["train"] * len(users), dtype="string"),
            "click_history": [list(clicks) for clicks in users.values()],
            "n_clicks": [len(clicks) for clicks in users.values()],
        }
    ).to_parquet(store / "history.parquet", index=False)

    embed.Embeddings(
        vectors=np.array([a[3] for a in ARTICLES], dtype="float32"),
        article_ids=np.array([a[0] for a in ARTICLES], dtype=object),
    ).save(embed.output_dir(config))
    bm25_index.run(config)
    counters.build(behaviors).save(config.artifacts_dir / counters.DIRECTORY)
    for split in features.SPLITS:
        features.build(config, split)
    model, _ = nrms.train(config, SMALL_NRMS)
    nrms.save(model, SMALL_NRMS, 2, nrms.checkpoint_path(config, SMALL_NRMS))
    return behaviors


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path


@pytest.fixture
def small(monkeypatch):
    """The registry's two specs, shrunk, since `run` and `rank_candidates` read
    them off the config rather than taking them as arguments."""
    monkeypatch.setattr(
        MIND.__class__, "nrms", property(lambda self: SMALL_NRMS), raising=False
    )
    monkeypatch.setattr(
        MIND.__class__, "rerank", property(lambda self: SMALL_RERANK), raising=False
    )


# --- what an arm reads ------------------------------------------------------


def test_an_arm_reads_its_tiers_and_one_counter_window():
    spec = dataclasses.replace(SMALL_RERANK, window="3h")
    columns = rerank.feature_columns(spec)

    assert "exposures_3h" in columns and "ctr_3h" in columns
    for window in ("cumulative", "1h", "24h"):
        assert f"exposures_{window}" not in columns
    # Non-counter columns of the same tier stay: the window selects the counter
    # columns, not the tier.
    assert "freshness_hours" in columns and "session_impressions" in columns
    assert rerank.NRMS_COLUMN in columns


def test_every_window_is_kept_when_the_arm_asks_for_all():
    columns = rerank.feature_columns(
        dataclasses.replace(SMALL_RERANK, window=rerank.ALL_WINDOWS)
    )
    for window in counters.WINDOWS:
        assert f"exposures_{window}" in columns and f"ctr_{window}" in columns


def test_dropping_a_tier_drops_its_columns_and_nothing_else():
    spec = dataclasses.replace(SMALL_RERANK, groups=("content", "history"))
    columns = set(rerank.feature_columns(spec))
    assert not columns & set(features.FEATURE_GROUPS["exposure"])
    assert not columns & set(features.FEATURE_GROUPS["clicked"])
    assert set(features.FEATURE_GROUPS["content"]) <= columns


def test_the_nrms_column_is_dropped_by_name_not_by_tier():
    with_it = rerank.feature_columns(SMALL_RERANK)
    without = rerank.feature_columns(dataclasses.replace(SMALL_RERANK, nrms=False))
    assert set(with_it) - set(without) == {rerank.NRMS_COLUMN}


def test_an_unknown_window_is_refused():
    with pytest.raises(rerank.RerankError, match="unknown counter window"):
        rerank.feature_columns(dataclasses.replace(SMALL_RERANK, window="7d"))


def test_the_variant_names_the_fields_that_make_a_different_model():
    assert rerank.variant_of(SMALL_RERANK) == "binary-l4-24h-drop:none-float32"
    leaky = dataclasses.replace(SMALL_RERANK, causal=False, nrms=False, top_k=50)
    assert rerank.variant_of(leaky) == "binary-l4-24h-drop:none-nonrms-leaky-cut50-float32"
    dropped = dataclasses.replace(SMALL_RERANK, groups=("content",))
    assert "drop:history+exposure+clicked" in rerank.variant_of(dropped)


# --- grouping ---------------------------------------------------------------


def test_group_sizes_follow_the_frames_own_row_order():
    ids = np.array(["i1", "i1", "i1", "i2", "i3", "i3"])
    assert list(rerank.group_sizes(ids)) == [3, 1, 2]
    assert [(i, s.start, s.stop) for i, s in rerank.group_spans(ids)] == [
        ("i1", 0, 3),
        ("i2", 3, 4),
        ("i3", 4, 6),
    ]


def test_an_impression_split_across_the_frame_is_refused():
    """A group boundary drawn inside an impression would train the ranker to
    order one impression's candidates against another's, and nothing
    downstream could see it."""
    with pytest.raises(rerank.RerankError, match="not contiguous"):
        rerank.group_sizes(np.array(["i1", "i2", "i1"]))


# --- the NRMS column --------------------------------------------------------


def test_the_nrms_score_is_joined_by_pair_not_by_position():
    """The NRMS frame comes back in *ranked* order. A positional join would
    give every row another candidate's score — the same mistake
    `features.read_back` exists to avoid, one stage later."""
    frame = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1", "i1"], dtype="string"),
            "article_id": pd.Series(["a1", "a2"], dtype="string"),
        }
    )
    scores = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1", "i1"], dtype="string"),
            "article_id": pd.Series(["a2", "a1"], dtype="string"),
            rerank.NRMS_COLUMN: [0.9, 0.1],
        }
    )
    joined = rerank.with_nrms(frame, scores)
    assert joined[rerank.NRMS_COLUMN].tolist() == [0.1, 0.9]


def test_scores_that_are_not_one_per_candidate_are_refused():
    frame = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "article_id": pd.Series(["a1"], dtype="string"),
        }
    )
    doubled = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1", "i1"], dtype="string"),
            "article_id": pd.Series(["a1", "a1"], dtype="string"),
            rerank.NRMS_COLUMN: [0.1, 0.2],
        }
    )
    with pytest.raises(rerank.RerankError, match="one per"):
        rerank.with_nrms(frame, doubled)


def test_the_nrms_column_is_computed_once_and_cached(store, small):
    write_store()
    first = rerank.nrms_scores(MIND, "tune")
    assert rerank.scores_path(MIND, "tune").exists()
    assert set(first.columns) == {"impression_id", "article_id", rerank.NRMS_COLUMN}

    # Cached: every arm of the ablation must read the *same* column, or a
    # difference between two arms would be partly a difference in features.
    rerank.scores_path(MIND, "tune").touch()
    stamped = rerank.scores_path(MIND, "tune").stat().st_mtime_ns
    again = rerank.nrms_scores(MIND, "tune")
    assert rerank.scores_path(MIND, "tune").stat().st_mtime_ns == stamped
    pd.testing.assert_frame_equal(first, again)


# --- fitting ----------------------------------------------------------------


def test_it_fits_only_on_the_half_nrms_did_not_see(store, small):
    write_store()
    later = rerank.later_half(MIND)
    earlier, _ = nrms.halves(
        pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet").pipe(
            lambda frame: frame[frame["split"] == "train"]
        ),
        SMALL_NRMS.train_fraction,
    )

    assert later and not later & set(earlier["impression_id"])
    rows = rerank.read_rows(MIND, "train", SMALL_RERANK, later)
    assert set(rows.impression_ids) <= later
    assert set(rows.impression_ids).isdisjoint(earlier["impression_id"])


def test_the_training_read_takes_only_the_arms_columns(store, small, monkeypatch):
    """The projection is the point of the tier structure: an arm that dropped a
    tier and still read it would report a memory saving it never made."""
    write_store()
    asked: list[list[str]] = []
    real = rerank.pq.ParquetFile.read_row_group

    def watched(self, group, columns=None, **keywords):
        asked.append(list(columns or []))
        return real(self, group, columns=columns, **keywords)

    monkeypatch.setattr(rerank.pq.ParquetFile, "read_row_group", watched)
    spec = dataclasses.replace(SMALL_RERANK, groups=("content",), window="1h")
    rerank.read_rows(MIND, "tune", spec)

    assert asked, "the frame was not read a row group at a time"
    for columns in asked:
        assert not set(columns) & set(features.FEATURE_GROUPS["history"])
        assert "exposures_24h" not in columns
        for forbidden in features.FORBIDDEN:
            assert forbidden not in columns


def test_training_stops_on_tune_and_the_model_round_trips(store, small):
    write_store()
    booster, report = rerank.train(MIND, SMALL_RERANK)

    assert report["rounds"] >= 1
    assert report["fit_rows"] == len(
        rerank.read_rows(MIND, "train", SMALL_RERANK, rerank.later_half(MIND)).labels
    )
    assert list(report["columns"]) == list(rerank.feature_columns(SMALL_RERANK))

    path = rerank.save(booster, MIND, SMALL_RERANK)
    loaded = rerank.load_model(MIND, SMALL_RERANK)
    tune = rerank.read_rows(MIND, "tune", SMALL_RERANK)
    np.testing.assert_allclose(
        booster.predict(tune.matrix), loaded.predict(tune.matrix), rtol=1e-9
    )
    assert path.exists()


def test_lambdarank_groups_by_impression(store, small):
    write_store()
    spec = dataclasses.replace(SMALL_RERANK, objective="lambdarank")
    rows = rerank.read_rows(MIND, "tune", spec)
    assert rows.groups.sum() == len(rows.labels)
    assert set(rows.groups) == {3}

    booster, report = rerank.train(MIND, spec)
    assert report["rounds"] >= 1


def test_an_unknown_objective_is_refused():
    with pytest.raises(rerank.RerankError, match="unknown objective"):
        rerank.parameters(dataclasses.replace(SMALL_RERANK, objective="poisson"))


def test_the_importance_table_sums_gain_per_availability_tier(store, small):
    write_store()
    booster, _ = rerank.train(MIND, SMALL_RERANK)
    found = rerank.importance(booster, SMALL_RERANK)

    assert set(found["columns"]) == set(rerank.feature_columns(SMALL_RERANK))
    assert set(found["tiers"]) <= {*features.FEATURE_GROUPS, rerank.NRMS_COLUMN}
    assert sum(found["tiers"].values()) == pytest.approx(
        sum(found["columns"].values())
    )
    # Saved beside the model, which is where ticket 07 reads it from.
    rerank.save(booster, MIND, SMALL_RERANK)
    beside = (
        rerank.model_path(MIND, SMALL_RERANK).parent
        / f"{rerank.variant_of(SMALL_RERANK)}-{rerank.IMPORTANCE}"
    )
    assert beside.exists()


# --- scoring ----------------------------------------------------------------


def test_the_cut_is_about_the_corpus_top_k_not_the_candidate_list(store, small):
    """The claim the cut row makes: a production system retrieves K out of the
    whole catalogue and re-ranks those, so a logged candidate the retriever
    would never have surfaced is one the user would never have seen. Asking
    for the cut without those corpus ranks is refused rather than answered
    with the in-impression order, which is a weaker, different claim."""
    write_store()
    behaviors = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == "validation"]
    history = ingest.history_for(MIND, impressions)

    lookup = rerank.global_ranks(MIND, impressions, history, depth=2)
    assert set(lookup) == set(impressions["impression_id"])
    for ranks in lookup.values():
        assert sorted(ranks.values()) == [1.0, 2.0]

    frame = pd.DataFrame(
        {
            "impression_id": pd.Series(["v0", "v0"], dtype="string"),
            "article_id": pd.Series(["a2", "a3"], dtype="string"),
        }
    )
    aligned = rerank.ranks_for(frame, lookup)
    assert len(aligned) == 2
    with pytest.raises(rerank.RerankError, match="corpus ranks"):
        rerank.ranked_from(
            frame, np.array([0.5, 0.4]), dataclasses.replace(SMALL_RERANK, top_k=1)
        )


def test_the_literal_cut_ranks_everything_outside_k_last():
    """What a real two-stage system does, and the row that says what it costs:
    the candidates the retriever did not surface are never scored, so they go
    below every scored one — in the retriever's own order."""
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    ranks = np.array([1.0, 3.0, 2.0, np.nan])
    cut = rerank.cut_outside_k(scores, ranks, top_k=2)

    assert cut[0] == 0.1 and cut[2] == 0.8
    assert cut[1] < min(scores) and cut[3] < min(scores)
    # Among the cut, the stage-one order survives: rank 3 above the unranked.
    assert cut[1] > cut[3]


def test_a_k_that_keeps_everything_changes_nothing():
    scores = np.array([0.1, 0.9])
    np.testing.assert_array_equal(
        rerank.cut_outside_k(scores, np.array([1.0, 2.0]), top_k=2), scores
    )


def test_rank_candidates_emits_the_shape_the_harness_scores(store, small):
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)

    behaviors = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == "validation"]
    history = ingest.history_for(MIND, impressions)
    ranked = rerank.rank_candidates(MIND, impressions, history)

    assert set(ranked["impression_id"]) == set(impressions["impression_id"])
    arrived = dict(zip(impressions["impression_id"], impressions["candidate_ids"]))
    for impression, ids, scores in zip(
        ranked["impression_id"], ranked["ranked_ids"], ranked["scores"]
    ):
        assert sorted(ids) == sorted(arrived[impression])
        assert list(scores) == sorted(scores, reverse=True)


def test_scoring_computes_the_frame_rather_than_reading_the_stored_one(
    store, small, monkeypatch
):
    """The serving path and the submission path are the same code: the frame is
    built per chunk. A `rank_candidates` that read the materialised split would
    report a latency no served request pays."""
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    behaviors = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == "validation"]
    history = ingest.history_for(MIND, impressions)

    built = []
    real = features.frame_for
    monkeypatch.setattr(
        features,
        "frame_for",
        lambda *arguments, **keywords: built.append(1) or real(*arguments, **keywords),
    )
    rerank.rank_candidates(MIND, impressions, history)
    assert built, "the frame was read, not computed"


def test_a_pooling_it_does_not_have_is_refused(store, small):
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    behaviors = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == "validation"]
    history = ingest.history_for(MIND, impressions)
    with pytest.raises(rerank.RerankError, match="no 'max' pooling"):
        rerank.rank_candidates(MIND, impressions, history, pooling="max")


# --- the submission path ----------------------------------------------------
#
# The competition is a later period over its own catalogue, so the submission
# assembles a frame the feature store cannot answer for. What these check is
# that it is the *same* frame -- same columns, same builder, same code path --
# because a model served forty columns that are subtly not the ones it was
# trained on produces a well-formed leaderboard file and a meaningless score.


COMPETITION_NEWS = "".join(
    f"{article}\t{category}\t{sub}\theadline {category}\t\t"
    f"https://example.com\t[]\t[]\n"
    for article, category, sub, _ in ARTICLES
)

# Two impressions a week after the feature store's log ends: one from a user
# the training period knows, one from a stranger with no history at all.
COMPETITION_BEHAVIORS = (
    "s1\tu0\t11/20/2019 9:00:00 AM\ta1 a3\ta2 a4 a6\n"
    "s2\tu9\t11/20/2019 9:05:00 AM\t\ta1 a5\n"
)


@pytest.fixture
def competition(store, monkeypatch):
    """The competition's own test files on disk, over the same six articles.

    The same six on purpose: what these tests are about is the frame, not the
    vector alignment `test_predict` already covers at length, so the
    catalogue's vectors are the store's own rather than a re-encoding. That
    substitution is `for_corpus`'s whole job and is stubbed here rather than
    exercised, which keeps a failure in one of these tests pointing at the
    submission frame.
    """
    monkeypatch.setattr(paths, "RAW_DIR", store / "raw")
    monkeypatch.setattr(paths, "PREDICTIONS_DIR", store / "predictions")
    test_dir = MIND.raw_dir / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "news.tsv").write_text(COMPETITION_NEWS, encoding="utf-8")
    (test_dir / "behaviors.tsv").write_text(COMPETITION_BEHAVIORS, encoding="utf-8")

    def stored_vectors(articles, config, directory):
        stored = embed.load(config)
        where = {article: row for row, article in enumerate(stored.article_ids)}
        ids = articles["article_id"].to_numpy(dtype=object)
        return (
            embed.Embeddings(
                vectors=stored.vectors[[where[article] for article in ids]],
                article_ids=ids,
            ),
            {
                "articles": len(ids),
                "cached": 1,
                "encoded": 0,
                "from_artifact": len(ids),
                "missing": 0,
            },
        )

    monkeypatch.setattr(embed, "for_corpus", stored_vectors)
    return store


def submission_ranker(store, **keywords):
    """The trained arm over the competition's catalogue."""
    from pipeline import predict

    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    return rerank.ranker(
        predict.catalogue(MIND), MIND, store / "work", SMALL_NRMS.history_length,
        **keywords,
    )


def test_the_submission_frame_is_the_frame_the_model_was_trained_on(
    competition, small
):
    """One builder, so the served columns are the trained columns in the
    trained order. A submission assembled by a second implementation would
    differ first in column order, which LightGBM reads positionally."""
    from pipeline import predict

    ranker = submission_ranker(competition)
    chunk = next(predict.impressions(MIND, history_k=SMALL_NRMS.history_length))
    frame = ranker.build_frame(chunk)

    assert list(frame.columns) == [*features.COLUMNS, rerank.NRMS_COLUMN]
    assert set(rerank.feature_columns(SMALL_RERANK)) <= set(frame.columns)
    assert set(frame["article_id"]) == {"a1", "a2", "a4", "a5", "a6"}


def test_the_submission_frame_says_the_labels_are_unknown(competition, small):
    """Not zero. A zero in the label column is the claim that nobody clicked,
    which is precisely the thing the leaderboard is holding back."""
    from pipeline import predict

    ranker = submission_ranker(competition)
    chunk = next(predict.impressions(MIND, history_k=SMALL_NRMS.history_length))
    frame = ranker.build_frame(chunk)

    assert (frame["label"] == features.UNKNOWN_LABEL).all()


def test_the_submission_and_the_harness_are_one_code_path(competition, small):
    """The A1 self-consistency check, applied to the fifth retriever.

    Handed the offline frame, the submission's `Ranker` must produce exactly
    what `rank_candidates` produces -- same order, same scores. The two differ
    in where the frame comes from and in nothing else, and this is what says
    so; a divergence here is the id-collision class of bug, which is invisible
    in every metric because both sides still rank something.
    """
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    behaviors = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == "validation"].reset_index(drop=True)
    history = ingest.history_for(MIND, impressions)

    harness = rerank.rank_candidates(MIND, impressions, history)

    whole = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    loaded = features.load(MIND, whole)
    nrms_scores = rerank.nrms_scores(MIND, "validation")
    served = rerank.Ranker(
        booster=rerank.load_model(MIND, SMALL_RERANK),
        spec=SMALL_RERANK,
        config=MIND,
        history_k=retrieval.HISTORY_K,
        build_frame=lambda chunk, candidates=None: rerank.with_nrms(
            features.frame_for(
                MIND, chunk, history, loaded, causal=SMALL_RERANK.causal
            ),
            nrms_scores,
        ),
    ).rank(impressions, list(impressions["candidate_ids"]))

    assert list(served["impression_id"]) == list(harness["impression_id"])
    for mine, theirs in zip(served["scores"], harness["scores"]):
        assert np.allclose(list(mine), list(theirs))


def test_a_top_k_cut_is_an_ablation_row_and_not_a_submission(competition, small, monkeypatch):
    """The one thing the submission path still refuses. A cut is a measurement
    of what stage one costs; a leaderboard file produced under it would not be
    the configuration any reported number came from."""
    monkeypatch.setattr(
        MIND.__class__,
        "rerank",
        property(lambda self: dataclasses.replace(SMALL_RERANK, top_k=2)),
        raising=False,
    )
    with pytest.raises(rerank.RerankError, match="ablation row"):
        submission_ranker(competition)


def test_the_submission_ranker_times_its_own_stages(competition, small):
    """Ticket 08's per-stage rows/s has to come from inside the ranker: `predict`
    can time the ranking, but only the ranker can say how much of it was the
    frame and how much was the two models."""
    from pipeline import predict

    ranker = submission_ranker(competition)
    chunk = next(predict.impressions(MIND, history_k=SMALL_NRMS.history_length))
    ranker.rank(chunk, list(chunk["candidate_ids"]))

    assert set(ranker.cost) == {"features", "nrms", "gbdt"}
    assert all(seconds > 0 for seconds in ranker.cost.values())


# --- the headline and the ledger --------------------------------------------


def test_the_headline_is_a_paired_delta_against_nrms(store, small):
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    delta = rerank.headline(MIND, SMALL_RERANK, resamples=50)

    assert delta["against"] == "nrms" and delta["split"] == "validation"
    assert delta["n"] > 0
    for metric in evaluate.ACCURACY_METRICS:
        assert delta[f"{metric}_lo"] <= delta[metric] <= delta[f"{metric}_hi"]


def test_the_stage_records_a_row_with_its_gain_table_and_snapshots(store, small, capsys):
    write_store()
    rerank.run(MIND)

    rows = {row["variant"]: row for row in ledger.load() if row["stage"] == rerank.STAGE}
    assert rerank.variant_of(SMALL_RERANK) in rows
    row = rows[rerank.variant_of(SMALL_RERANK)]
    assert row["dataset"] == "mind" and row["split"] == "tune"
    assert 0.0 <= row["auc"] <= 1.0
    assert row["model_bytes"] > 0 and row["train_seconds"] >= 0
    assert row["p99_ms"] >= row["p50_ms"] >= 0
    assert "rounds over" in row["note"]

    printed = capsys.readouterr().out
    assert "gain" in printed


def test_a_second_run_keeps_the_model_unless_forced(store, small, capsys):
    write_store()
    rerank.run(MIND)
    trained = rerank.model_path(MIND).stat().st_mtime_ns

    rerank.run(MIND)
    assert "already trained" in capsys.readouterr().out
    assert rerank.model_path(MIND).stat().st_mtime_ns == trained

    rerank.run(MIND, force=True)
    assert rerank.model_path(MIND).stat().st_mtime_ns != trained


def test_the_grid_trains_one_arm_per_alternative(store, small):
    write_store()
    tried = rerank.grid(MIND, {"leaves": (4, 8), "nrms": (True, False)})

    variants = [report["variant"] for report in tried]
    assert len(variants) == len(set(variants)) == 3
    assert variants.count(rerank.variant_of(SMALL_RERANK)) == 1
    assert {row["variant"] for row in ledger.load() if row["stage"] == rerank.STAGE} >= set(
        variants
    )


def test_the_leaky_arm_reads_the_leaky_frame(store, small):
    """Q9's pair: the same model, one flag. It reads the file ticket 04 wrote
    with the whole-log counters, and says so rather than falling back."""
    write_store()
    leaky = dataclasses.replace(SMALL_RERANK, causal=False)
    with pytest.raises(rerank.RerankError, match="--leaky"):
        rerank.read_rows(MIND, "tune", leaky)

    for split in ("train", "tune"):
        features.build(MIND, split, causal=False)

    every = dataclasses.replace(SMALL_RERANK, window=rerank.ALL_WINDOWS)
    rows = rerank.read_rows(MIND, "tune", dataclasses.replace(every, causal=False))
    clean = rerank.read_rows(MIND, "tune", every)
    # The cumulative counter is the one the two arms can be ordered on: the
    # leaky read is the whole log and the causal one a prefix of it. A sliding
    # window is not a superset — the leaky hour is the hour *after* `t`, which
    # is the leak — so it is compared for difference rather than for size.
    cumulative = list(rerank.feature_columns(every)).index("exposures_cumulative")
    assert rows.matrix[:, cumulative].sum() > clean.matrix[:, cumulative].sum()
    assert not np.array_equal(rows.matrix, clean.matrix)


def test_the_same_model_is_timed_on_one_core_and_on_every_core(store, small):
    """Ticket 09's cost per 1000 queries is built from a single-core QPS.
    Dividing a multi-core measurement by the core count is arithmetic, not
    measurement, so both are rows — on one model, with one AUC."""
    write_store()
    report = rerank.fit_one(MIND, SMALL_RERANK)

    rows = {row["variant"]: row for row in ledger.load() if row["stage"] == rerank.STAGE}
    base = rerank.variant_of(SMALL_RERANK)
    # The spec scores on every core, so the base row *is* the multi-core one
    # and the second row is the single-core measurement of the same model —
    # one extra row, not a duplicate of one that already exists.
    assert f"{base}-t1" in rows
    assert f"{base}-tall" not in rows
    assert rows[f"{base}-t1"]["auc"] == pytest.approx(rows[base]["auc"])
    assert rows[f"{base}-t1"]["p50_ms"] >= 0
    assert [row["variant"] for row in report["threads"]] == [f"{base}-t1"]

    single = dataclasses.replace(SMALL_RERANK, threads=1)
    assert [
        row["variant"]
        for row in rerank.thread_rows(
            rerank.load_model(MIND, SMALL_RERANK),
            rerank.read_rows(MIND, "tune", single),
            single,
            {**report, "variant": rerank.variant_of(single)},
        )
    ] == [f"{base}-tall"]


def test_the_projection_is_measured_rather_than_asserted(store, small):
    write_store()
    rows = rerank.projection_rows(
        MIND, dataclasses.replace(SMALL_RERANK, groups=("content",), window="1h")
    )

    projected, everything = rows
    assert "columns over" in projected["note"]
    assert projected["peak_rss_mb"] > 0 and everything["peak_rss_mb"] > 0
    recorded = {row["variant"] for row in ledger.load() if row["stage"] == rerank.STAGE}
    assert {projected["variant"], everything["variant"]} <= recorded
