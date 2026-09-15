"""The ablation is tested for the properties that make its table evidence
rather than decoration: an arm really lacks the columns it claims to drop, the
leaky arm differs from the clean one in the counter's `t` and in nothing else,
every arm is paired against the full model, early stopping never reads the
split the arms are reported on, and the frame is read once rather than rebuilt
per arm."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from pipeline import ablation, evaluate, features, ingest, ledger, paths, rerank, retrieval
from pipeline.datasets import DATASETS, RerankSpec

from tests.test_rerank import (  # the same small store, one definition of it
    SMALL_NRMS,
    SMALL_RERANK,
    write_store,
)

MIND = DATASETS["mind"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(
        MIND.__class__, "nrms", property(lambda self: SMALL_NRMS), raising=False
    )
    monkeypatch.setattr(
        MIND.__class__, "rerank", property(lambda self: SMALL_RERANK), raising=False
    )


def built(store):
    """The store plus the leaky frames, which the Q9 arms read."""
    behaviors = write_store()
    for split in features.SPLITS:
        features.build(MIND, split, causal=False)
    return behaviors


# --- the arms themselves ----------------------------------------------------


def test_the_arms_cover_what_the_ticket_asks_for():
    names = {arm.name for arm in ablation.arms_for(SMALL_RERANK)}
    assert ablation.FULL in names
    for tier in ("history", "exposure", "clicked"):
        assert f"-{tier}" in names
    assert {"-session/dwell", "-nrms", "-retriever-scores"} <= names
    assert {f"cut@{depth}" for depth in retrieval.DEPTHS} <= names
    assert {"leaky", "leaky-cumulative", "clean-cumulative"} <= names
    # The cumulative build-up, in serving order.
    assert {"content", "content+history", "content+history+exposure"} <= names


def test_content_is_never_dropped_whole():
    """Without the retriever scores and the category match there is nothing
    left for a re-ranker to re-rank, so that row would measure the absence of a
    candidate generator rather than the value of a tier."""
    assert "-content" not in {arm.name for arm in ablation.arms_for(SMALL_RERANK)}


def test_an_arm_that_drops_a_tier_has_none_of_its_columns():
    for arm in ablation.arms_for(SMALL_RERANK):
        spec = ablation.spec_for(SMALL_RERANK, arm)
        columns = set(rerank.feature_columns(spec))
        for tier in features.FEATURE_GROUPS:
            if tier in spec.groups:
                continue
            assert not columns & set(features.FEATURE_GROUPS[tier]), arm.name
        assert not columns & set(spec.drop), arm.name


def test_the_leaky_arm_differs_from_the_clean_one_in_one_field():
    """Q9's pair is one flag. Anything else differing between them would make
    the gap a difference between two models rather than a measurement of what
    reading the future is worth."""
    arms = {arm.name: arm for arm in ablation.arms_for(SMALL_RERANK)}
    clean = ablation.spec_for(SMALL_RERANK, arms[ablation.FULL])
    leaky = ablation.spec_for(SMALL_RERANK, arms["leaky"])
    assert leaky == dataclasses.replace(clean, causal=False)

    cumulative = ablation.spec_for(SMALL_RERANK, arms["clean-cumulative"])
    leaky_cumulative = ablation.spec_for(SMALL_RERANK, arms["leaky-cumulative"])
    assert leaky_cumulative == dataclasses.replace(cumulative, causal=False)
    # And the columns the two read are the same ones, in the same order.
    assert rerank.feature_columns(clean) == rerank.feature_columns(leaky)


def test_a_run_without_the_full_arm_is_refused(store, small):
    built(store)
    with pytest.raises(ablation.AblationError, match="has to"):
        ablation.run_arms(
            MIND, "tune", (ablation.Arm("-nrms", "why", {"nrms": False}),)
        )


def test_two_precisions_are_not_one_ablation(store, small):
    built(store)
    with pytest.raises(ablation.AblationError, match="one frame"):
        ablation.read_split(
            MIND,
            "tune",
            [SMALL_RERANK, dataclasses.replace(SMALL_RERANK, precision="float16")],
        )


# --- the run ----------------------------------------------------------------


def test_every_arm_is_paired_against_the_full_model(store, small):
    built(store)
    rows = ablation.run_arms(MIND, "tune", resamples=40)

    by_arm = {row["arm"]: row for row in rows}
    assert set(by_arm) == {arm.name for arm in ablation.arms_for(SMALL_RERANK)}
    for name, row in by_arm.items():
        assert row["n"] > 0 and row["columns"] > 0
        assert row["model_bytes"] > 0 and row["p99_ms"] >= row["p50_ms"] >= 0
        if name == ablation.FULL:
            assert "auc_gap" not in row
            continue
        assert row["against"] == ablation.FULL
        for metric in evaluate.ACCURACY_METRICS:
            assert row[f"{metric}_gap_lo"] <= row[f"{metric}_gap"] <= row[f"{metric}_gap_hi"]


def test_the_frame_is_read_rather_than_rebuilt_per_arm(store, small, monkeypatch):
    """Sixteen arms differ in which columns they read, not in what is in them.
    Rebuilding features per arm would be sixteen builds of one frame — and the
    arms would no longer be paired on identical inputs."""
    built(store)
    rebuilt = []
    monkeypatch.setattr(
        features, "frame_for", lambda *a, **k: rebuilt.append(1)
    )
    reads = []
    real = features.read
    monkeypatch.setattr(
        features,
        "read",
        lambda *a, **k: (reads.append((a, k)) or real(*a, **k)),
    )

    ablation.run_arms(MIND, "tune", resamples=20)

    assert not rebuilt, "an arm rebuilt the feature frame"
    # tune (causal + leaky) and train (causal + leaky): one read per split per
    # causality, however many arms there are.
    assert len(reads) == 4


def test_early_stopping_never_reads_the_split_the_arms_are_reported_on(
    store, small, monkeypatch
):
    """Chosen on tune, reported on validation — the rule this project has kept
    since A1. An arm stopped on validation would have selected on the split its
    number is quoted from."""
    built(store)
    stopped_on = []
    real = rerank.train

    def watched(config, spec=None, fit=None, tune=None):
        stopped_on.append(set(tune.impression_ids))
        return real(config, spec, fit, tune)

    monkeypatch.setattr(rerank, "train", watched)
    ablation.run_arms(
        MIND,
        "validation",
        (ablation.Arm(ablation.FULL, "everything", {}),),
        resamples=20,
    )

    behaviors = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    validation = set(behaviors[behaviors["split"] == "validation"]["impression_id"])
    tune = set(behaviors[behaviors["split"] == "tune"]["impression_id"])
    assert stopped_on and stopped_on[0] <= tune
    assert not stopped_on[0] & validation


def test_the_full_arm_scores_what_the_harness_would_score(store, small):
    """The ablation reads the stored frame and `rank_candidates` computes it.
    They must agree, or the cheap path here is a second definition of the
    features rather than a read of the same numbers."""
    built(store)
    specs = [SMALL_RERANK]
    split = ablation.read_split(MIND, "validation", specs)
    fit = ablation.read_split(MIND, "train", specs, ranks=False)
    booster, _ = rerank.train(
        MIND,
        SMALL_RERANK,
        ablation.rows_of(fit, SMALL_RERANK, keep=rerank.later_half(MIND)),
        ablation.rows_of(
            ablation.read_split(MIND, "tune", specs, ranks=False), SMALL_RERANK
        ),
    )
    rerank.save(booster, MIND, SMALL_RERANK)

    read = ablation.score(booster, split, SMALL_RERANK)
    computed = rerank.rank_candidates(
        MIND, split.impressions, ingest.history_for(MIND, split.impressions)
    )

    read = read.set_index("impression_id").sort_index()
    computed = computed.set_index("impression_id").sort_index()
    assert list(read.index) == list(computed.index)
    for impression in read.index:
        assert list(read.loc[impression, "ranked_ids"]) == list(
            computed.loc[impression, "ranked_ids"]
        )
        np.testing.assert_allclose(
            read.loc[impression, "scores"],
            computed.loc[impression, "scores"],
            rtol=1e-6,
        )


# --- the artifacts ----------------------------------------------------------


def test_the_run_writes_the_table_the_bridge_and_the_q9_pair(store, small):
    built(store)
    path = ablation.run(MIND, "tune", resamples=20)
    text = path.read_text()

    assert "Δ auc vs full (paired)" in text
    for depth in retrieval.DEPTHS:
        assert f"| {depth} |" in text
    assert "Q9: what the leak is worth" in text
    assert "no server" in text
    # Every retriever the harness scores has a row, even the ones this store
    # has no report for — a missing report is said, not omitted.
    for retriever in ablation.ORDER:
        assert f"`{retriever}`" in text

    rows = [
        json.loads(line)
        for line in (MIND.artifacts_dir / "ablation-tune.jsonl").read_text().splitlines()
    ]
    assert {row["arm"] for row in rows} == {
        arm.name for arm in ablation.arms_for(SMALL_RERANK)
    }


def test_every_arm_lands_in_the_ledger_with_its_delta(store, small):
    built(store)
    ablation.run(MIND, "tune", resamples=20)

    rows = {
        row["variant"]: row for row in ledger.load() if row["stage"] == ablation.STAGE
    }
    names = {arm.name for arm in ablation.arms_for(SMALL_RERANK)}
    assert names <= set(rows)
    for name, row in rows.items():
        # The rows that are not arms — the whole run's cost, and what each
        # corpus search cost — carry no delta, because there is nothing for
        # them to be a difference from.
        if name in (ablation.FULL, "whole-run") or name not in names:
            continue
        assert row["delta_vs"] == ablation.FULL
        assert row["delta_lo"] <= row["delta"] <= row["delta_hi"]
    # What a full re-run costs, which is a claim with a wall time on it.
    whole = rows["whole-run"]
    assert whole["train_seconds"] >= 0 and whole["peak_rss_mb"] > 0
    assert "arms" in whole["note"]


def test_the_command_refuses_the_split_the_arms_fit_on(capsys):
    assert ablation.main(["--split", "train"]) == 2
    assert "fit on" in capsys.readouterr().out


def test_the_cut_arms_are_scored_against_both_indexes(store, small):
    """Ticket 09 measures what the approximate index saves in latency; this is
    what it costs in quality, at the same K, over the same vectors and the same
    queries — so the two numbers describe one index."""
    names = {arm.name for arm in ablation.arms_for(SMALL_RERANK)}
    for depth in retrieval.DEPTHS:
        assert f"cut@{depth}" in names and f"cut@{depth} ({ablation.IVF})" in names

    built(store)
    arms = (
        ablation.Arm(ablation.FULL, "everything", {}),
        ablation.Arm("cut@50", "flat", {"top_k": 50}),
        ablation.Arm("cut@50 (ivf)", "approximate", {"top_k": 50}, ranks=ablation.IVF),
    )
    rows = {row["arm"]: row for row in ablation.run_arms(MIND, "tune", arms, resamples=20)}
    assert set(rows) == {arm.name for arm in arms}
    # Same model, two rankings: the cut arms differ from the full arm in when
    # the cut is applied, not in how the trees were grown.
    assert rows["cut@50"]["model_bytes"] == rows[ablation.FULL]["model_bytes"]


def test_a_cut_is_not_scored_against_an_index_that_was_never_searched(store, small):
    built(store)
    split = ablation.read_split(MIND, "tune", [SMALL_RERANK], indexes=(ablation.FLAT,))
    booster = object()
    with pytest.raises(ablation.AblationError, match="never|not searched"):
        ablation.score(
            booster, split, dataclasses.replace(SMALL_RERANK, top_k=50), ablation.IVF
        )


def test_arms_that_differ_only_in_the_cut_share_one_training(store, small, monkeypatch):
    built(store)
    fits = []
    real = rerank.train
    monkeypatch.setattr(
        rerank,
        "train",
        lambda *a, **k: (fits.append(1) or real(*a, **k)),
    )
    arms = (
        ablation.Arm(ablation.FULL, "everything", {}),
        *(
            ablation.Arm(f"cut@{depth}", "cut", {"top_k": depth})
            for depth in retrieval.DEPTHS
        ),
    )
    ablation.run_arms(MIND, "tune", arms, resamples=20)
    assert len(fits) == 1, "the same trees were grown once per cut"


def test_the_corpus_search_is_one_row_rather_than_a_third_of_three(store, small):
    """The search happens once for every depth, so charging each cut arm a
    share of it would be an allocation rather than a measurement."""
    built(store)
    ablation.run_arms(
        MIND,
        "tune",
        (
            ablation.Arm(ablation.FULL, "everything", {}),
            ablation.Arm("cut@50", "flat", {"top_k": 50}),
            ablation.Arm("cut@50 (ivf)", "approximate", {"top_k": 50}, ranks=ablation.IVF),
        ),
        resamples=20,
    )

    rows = {row["variant"]: row for row in ledger.load() if row["stage"] == ablation.STAGE}
    for index in (ablation.FLAT, ablation.IVF):
        row = rows[f"corpus-search-{index}"]
        assert row["p50_ms"] >= 0 and row["rows_per_s"] > 0
        assert "shared by every cut arm" in row["note"]
