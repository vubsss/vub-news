"""NRMS is tested where a reproduction can go quietly wrong: the stacking
boundary it fits inside, the negatives it draws, the window baked into its
weights, the cold user whose attention has nothing to attend to, and the
harness seam — one `rank_candidates`, scored like any other retriever."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from pipeline import embed, ingest, ledger, nrms, paths
from pipeline.datasets import DATASETS, NrmsSpec

MIND = DATASETS["mind"]

T0 = pd.Timestamp("2019-11-11 08:00:00")

# A small spec: the same model, sized so a test trains in well under a second.
SMALL = NrmsSpec(
    history_length=4,
    heads=2,
    head_dim=4,
    attention_dim=8,
    dropout=0.0,
    negatives=2,
    epochs=2,
    batch_size=8,
    score_batch=4,
)

ARTICLES = ("a1", "a2", "a3", "a4", "a5", "a6")
VECTORS = np.array(
    [
        [1.0, 0.0],
        [0.9, 0.436],
        [0.0, 1.0],
        [0.436, 0.9],
        [-1.0, 0.0],
        [0.0, -1.0],
    ],
    dtype="float32",
)


def behaviours(rows):
    """rows: (impression, user, hours after T0, candidates, labels, split)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
            "user_id": pd.Series([row[1] for row in rows], dtype="string"),
            "source": pd.Series(["train"] * len(rows), dtype="string"),
            "impression_time": pd.Series(
                [T0 + pd.Timedelta(hours=row[2]) for row in rows], dtype="datetime64[us]"
            ),
            "candidate_ids": [list(row[3]) for row in rows],
            "labels": [list(row[4]) for row in rows],
            "split": pd.Series([row[5] for row in rows], dtype="string"),
        }
    )


def log(count=12):
    """A train split spanning enough distinct moments to stack on, then tune
    and validation after it."""
    rows = []
    for i in range(count):
        clicked = i % 2
        rows.append(
            (
                f"t{i}",
                f"u{i % 3}",
                i,
                ["a1", "a3", "a5"] if clicked else ["a2", "a4", "a6"],
                [1, 0, 0] if clicked else [0, 1, 0],
                "train",
            )
        )
    rows += [
        ("n1", "u0", count + 1, ["a1", "a5"], [1, 0], "tune"),
        ("n2", "u1", count + 2, ["a3", "a6"], [1, 0], "tune"),
        ("v1", "u2", count + 3, ["a2", "a4"], [0, 1], "validation"),
        ("v2", "cold", count + 4, ["a1", "a3"], [1, 0], "validation"),
    ]
    return behaviours(rows)


def stored_history():
    users = {"u0": ["a1", "a3"], "u1": ["a2", "a4"], "u2": ["a5"], "cold": []}
    return pd.DataFrame(
        {
            "user_id": pd.Series(list(users), dtype="string"),
            "source": pd.Series(["train"] * len(users), dtype="string"),
            "click_history": [list(clicks) for clicks in users.values()],
            "n_clicks": [len(clicks) for clicks in users.values()],
        }
    )


def write_store(config=MIND):
    store = config.feature_store_dir
    store.mkdir(parents=True, exist_ok=True)
    behaviors = log()
    behaviors.to_parquet(store / "behaviors.parquet", index=False)
    stored_history().to_parquet(store / "history.parquet", index=False)
    pd.DataFrame(
        {
            "article_id": pd.Series(ARTICLES, dtype="string"),
            "title": pd.Series([f"headline {a}" for a in ARTICLES], dtype="string"),
        }
    ).to_parquet(store / "articles.parquet", index=False)
    embed.Embeddings(
        vectors=VECTORS, article_ids=np.array(ARTICLES, dtype=object)
    ).save(embed.output_dir(config))
    return behaviors


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path


@pytest.fixture
def small(monkeypatch):
    """The registry's spec, shrunk — so `nrms.run` and `rank_candidates`, which
    read `config.nrms`, reach a model a test can afford to train."""
    monkeypatch.setattr(
        DATASETS["mind"].__class__, "nrms", property(lambda self: SMALL), raising=False
    )
    return SMALL


# --- the stacking boundary --------------------------------------------------


def test_the_halves_are_disjoint_in_time(store):
    behaviors = log()
    train = behaviors[behaviors["split"] == "train"]
    earlier, later = nrms.halves(train, 0.5)

    assert len(earlier) + len(later) == len(train)
    assert earlier["impression_time"].max() < later["impression_time"].min()
    # The re-ranker of ticket 06 fits on the later half, so a model fitted here
    # has seen none of the impressions its own score will be a feature for.
    assert not set(earlier["impression_id"]) & set(later["impression_id"])


def test_impressions_at_the_boundarys_instant_all_fall_on_one_side():
    """A cut by row count would put two impressions of the same second in
    different halves, which is the same leak in miniature as a random split."""
    rows = [(f"t{i}", "u0", i // 4, ["a1", "a2"], [1, 0], "train") for i in range(8)]
    earlier, later = nrms.halves(behaviours(rows), 0.5)
    assert set(earlier["impression_time"]) & set(later["impression_time"]) == set()


def test_a_log_too_short_to_stack_on_is_refused():
    rows = [(f"t{i}", "u0", 0, ["a1", "a2"], [1, 0], "train") for i in range(4)]
    with pytest.raises(nrms.NrmsError, match="one half empty"):
        nrms.halves(behaviours(rows), 0.5)
    with pytest.raises(nrms.NrmsError, match="strictly inside"):
        nrms.halves(log(), 1.0)


def test_training_fits_only_on_the_earlier_half(store, small):
    write_store()
    data = nrms.prepare(MIND, SMALL)
    behaviors = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    train = behaviors[behaviors["split"] == "train"]
    _, later = nrms.halves(train, SMALL.train_fraction)

    assert set(data.fit["impression_id"]).isdisjoint(later["impression_id"])
    assert data.fit["impression_time"].max() < later["impression_time"].min()
    # And nothing from tune or validation is fitted on at all.
    assert set(data.fit["split"]) == {"train"}


# --- the negatives ----------------------------------------------------------


def test_every_negative_comes_from_the_impression_its_positive_did():
    """The paper's setting, and not interchangeable with catalogue negatives:
    an in-impression negative is an article the same user was shown at the same
    moment and did not click."""
    behaviors = log().reset_index(drop=True)
    rng = np.random.default_rng(0)
    drawn = nrms.examples(behaviors, negatives=2, rng=rng)

    assert drawn
    for position, articles in drawn:
        candidates = behaviors["candidate_ids"].iloc[position]
        labels = np.asarray(behaviors["labels"].iloc[position])
        positive, *negatives = articles
        assert labels[list(candidates).index(positive)] == 1
        for negative in negatives:
            assert negative in candidates
            assert labels[list(candidates).index(negative)] == 0


def test_the_catalogue_arm_draws_from_everywhere_and_never_the_positive():
    """The arm that measures what the paper's setting was worth. It is the
    easier problem — a uniform article is rarely a near miss — so the row it
    produces is expected to lose, which is why it has to exist."""
    rows = [("one", "u0", 0, ["a1", "a2"], [1, 0], "train")]
    catalogue = np.array(ARTICLES, dtype=object)
    drawn = nrms.examples(
        behaviours(rows), 8, np.random.default_rng(0), "catalogue", catalogue
    )
    (_, articles), = drawn
    positive, *negatives = articles
    assert positive == "a1"
    assert set(negatives) - set(["a1", "a2"]), "never left the impression"
    assert "a1" not in negatives
    assert set(negatives) <= set(ARTICLES)

    with pytest.raises(nrms.NrmsError, match="need a catalogue"):
        nrms.examples(behaviours(rows), 2, np.random.default_rng(0), "catalogue")
    with pytest.raises(nrms.NrmsError, match="unknown negative source"):
        nrms.examples(behaviours(rows), 2, np.random.default_rng(0), "invented")


def test_an_impression_with_no_negative_still_trains_under_the_catalogue_arm():
    """An impression where everything was clicked has no in-impression
    negative and is skipped; the catalogue arm can still learn from it, which
    is a difference between the two arms and not an accident."""
    rows = [("all", "u0", 0, ["a1", "a2"], [1, 1], "train")]
    catalogue = np.array(ARTICLES, dtype=object)
    assert nrms.examples(behaviours(rows), 2, np.random.default_rng(0)) == []
    assert len(
        nrms.examples(behaviours(rows), 2, np.random.default_rng(0), "catalogue", catalogue)
    ) == 2


def test_an_impression_with_nothing_to_contrast_is_not_an_example():
    rows = [
        ("all", "u0", 0, ["a1", "a2"], [1, 1], "train"),
        ("none", "u0", 1, ["a1", "a2"], [0, 0], "train"),
        ("real", "u0", 2, ["a1", "a2"], [1, 0], "train"),
    ]
    drawn = nrms.examples(behaviours(rows), 2, np.random.default_rng(0))
    assert [position for position, _ in drawn] == [2]


def test_a_slate_is_filled_even_when_the_impression_is_short():
    """One negative and four slots: drawing with replacement keeps the example
    rather than dropping the impressions with the shortest candidate lists."""
    rows = [("short", "u0", 0, ["a1", "a2"], [1, 0], "train")]
    (_, articles), = nrms.examples(behaviours(rows), 4, np.random.default_rng(0))
    assert articles == ["a1", "a2", "a2", "a2", "a2"]


# --- the model --------------------------------------------------------------


def test_a_cold_users_attention_has_something_to_attend_to():
    """A fully masked row would make the softmax NaN. The padding position
    stays unmasked and gathers a zero vector, so a cold user scores every
    candidate alike and the ranking falls back to arrival order — which is what
    the other retrievers do with a user they know nothing about."""
    history = pd.DataFrame({"click_history": [[], ["a1"]]})
    index = {article: row for row, article in enumerate(ARTICLES)}
    rows, mask = nrms.history_rows(history, index, 4)
    assert mask[0].tolist() == [True, False, False, False]
    assert mask[1].tolist() == [True, False, False, False]

    model = nrms.Nrms(2, SMALL)
    clicked = nrms.gather(VECTORS, rows, mask)
    assert clicked[0].abs().sum() == 0
    candidates, candidate_mask = nrms.candidate_rows(
        [["a1", "a3"], ["a1", "a3"]], index, 2
    )
    scores = model(clicked, torch.from_numpy(mask), nrms.gather(VECTORS, candidates, candidate_mask))
    assert torch.isfinite(scores).all()


def test_the_user_vector_is_computed_once_per_impression_not_per_candidate():
    """The scoring path's whole shape: one user encoding dotted against every
    candidate. Scoring the same impression with two candidate lists must reuse
    one user vector, which is checkable as the scores agreeing on the candidate
    the two lists share."""
    model = nrms.Nrms(2, SMALL).eval()
    index = {article: row for row, article in enumerate(ARTICLES)}
    rows, mask = nrms.history_rows(pd.DataFrame({"click_history": [["a1", "a2"]]}), index, 4)
    clicked = nrms.gather(VECTORS, rows, mask)

    with torch.no_grad():
        one, _ = nrms.candidate_rows([["a3"]], index, 1)
        two, _ = nrms.candidate_rows([["a5", "a3"]], index, 2)
        alone = model(clicked, torch.from_numpy(mask), nrms.gather(VECTORS, one, np.ones_like(one, dtype=bool)))
        together = model(clicked, torch.from_numpy(mask), nrms.gather(VECTORS, two, np.ones_like(two, dtype=bool)))
    assert float(alone[0, 0]) == pytest.approx(float(together[0, 1]), abs=1e-6)


def test_a_candidate_with_no_vector_still_comes_back_ranked():
    index = {article: row for row, article in enumerate(ARTICLES)}
    rows, mask = nrms.candidate_rows([["a1", "unknown"]], index, 2)
    assert mask[0].tolist() == [True, False]
    gathered = nrms.gather(VECTORS, rows, mask)
    assert gathered[0, 1].abs().sum() == 0


# --- training, scoring, and the harness seam --------------------------------


def test_training_reads_the_store_once_however_many_epochs_it_runs(store, monkeypatch):
    """The history table is opened per *run*, not per epoch and certainly not
    per batch. Counted rather than asserted by inspection, because the failure
    is invisible in the result and shows up only on the wall clock."""
    write_store()
    reads: list[str] = []
    real = pd.read_parquet

    def counted(path, *arguments, **keywords):
        reads.append(str(path))
        return real(path, *arguments, **keywords)

    monkeypatch.setattr(pd, "read_parquet", counted)
    nrms.train(MIND, dataclasses.replace(SMALL, epochs=1))
    one_epoch = len(reads)
    reads.clear()
    nrms.train(MIND, dataclasses.replace(SMALL, epochs=3))
    assert len(reads) == one_epoch
    assert sum("history" in path for path in reads) == 1


def test_training_stops_on_tune_and_keeps_the_curve_that_chose_the_epoch(store):
    write_store()
    _, report = nrms.train(MIND, dataclasses.replace(SMALL, epochs=4, patience=1))

    assert report["curve"], "no epochs ran"
    assert [row["epoch"] for row in report["curve"]] == list(
        range(1, len(report["curve"]) + 1)
    )
    best = max(report["curve"], key=lambda row: row["tune_auc"])
    assert report["epoch"] == best["epoch"]
    assert report["tune_auc"] == best["tune_auc"]
    # Early stopping means at most one epoch past the best one was run.
    assert len(report["curve"]) <= best["epoch"] + 1


def test_a_checkpoint_round_trips_and_carries_the_spec_it_was_fitted_under(store):
    write_store()
    model, _ = nrms.train(MIND, SMALL)
    path = nrms.save(model, SMALL, 2, nrms.checkpoint_path(MIND, SMALL))

    loaded, spec, dim = nrms.load_model(MIND, SMALL)
    assert spec == SMALL and dim == 2
    index = {article: row for row, article in enumerate(ARTICLES)}
    rows, mask = nrms.history_rows(
        pd.DataFrame({"click_history": [["a1", "a2"]]}), index, SMALL.history_length
    )
    before = nrms.score_all(model, VECTORS, rows, mask, [["a1", "a3"]], index, 4)
    after = nrms.score_all(loaded, VECTORS, rows, mask, [["a1", "a3"]], index, 4)
    np.testing.assert_allclose(before[0], after[0], rtol=1e-6)
    assert path.exists()


def test_the_half_precision_checkpoint_is_smaller_and_still_scores(store):
    """`precision` is the file's dtype, not the arithmetic's: the model comes
    back as float32 either way, so the row records the bytes against whatever
    the rounding costs in tune AUC."""
    write_store()
    model, _ = nrms.train(MIND, SMALL)
    wide = nrms.save(model, SMALL, 2, nrms.checkpoint_path(MIND, SMALL))
    half_spec = dataclasses.replace(SMALL, precision="fp16")
    narrow = nrms.save(model, half_spec, 2, nrms.checkpoint_path(MIND, half_spec))

    assert narrow.stat().st_size < wide.stat().st_size
    loaded, _, _ = nrms.load_model(MIND, half_spec)
    assert next(iter(loaded.state_dict().values())).dtype == torch.float32


def test_rank_candidates_emits_the_shape_the_harness_scores(store, small):
    write_store()
    nrms.fit_one(MIND, SMALL)

    behaviors = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == "validation"]
    history = ingest.history_for(MIND, impressions)
    ranked = nrms.rank_candidates(MIND, impressions, history, SMALL.history_length)

    assert set(ranked["impression_id"]) == set(impressions["impression_id"])
    for _, row in ranked.iterrows():
        assert list(row["scores"]) == sorted(row["scores"], reverse=True)
    arrived = dict(zip(impressions["impression_id"], impressions["candidate_ids"]))
    for impression, ids in zip(ranked["impression_id"], ranked["ranked_ids"]):
        assert sorted(ids) == sorted(arrived[impression])
    # v2's user is cold: every candidate scores the same, so the ranking is the
    # order the dataset listed them in rather than one the model invented.
    cold = ranked[ranked["impression_id"] == "v2"].iloc[0]
    assert list(cold["ranked_ids"]) == list(arrived["v2"])


def test_a_window_the_checkpoint_was_not_fitted_at_is_refused(store, small):
    write_store()
    nrms.fit_one(MIND, SMALL)
    behaviors = pd.read_parquet(MIND.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == "validation"]
    history = ingest.history_for(MIND, impressions)

    with pytest.raises(nrms.NrmsError, match="window is in the weights"):
        nrms.rank_candidates(MIND, impressions, history, SMALL.history_length + 10)
    with pytest.raises(nrms.NrmsError, match="no 'max' pooling"):
        nrms.rank_candidates(
            MIND, impressions, history, SMALL.history_length, pooling="max"
        )


def test_the_stage_records_a_ledger_row_with_both_halves_of_the_cost(store, small):
    write_store()
    nrms.run(MIND)

    rows = [row for row in ledger.load() if row["stage"] == nrms.STAGE]
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "mind" and row["split"] == "tune"
    assert row["variant"] == nrms.variant_of(SMALL)
    assert 0.0 <= row["auc"] <= 1.0
    assert row["model_bytes"] > 0 and row["train_seconds"] >= 0
    assert row["p99_ms"] >= row["p50_ms"] >= 0
    # The two halves ticket 09 needs to say what a cache in front of the user
    # encoder would remove.
    assert "user encoder" in row["note"] and "candidate dot" in row["note"]


def test_a_second_run_keeps_the_checkpoint_unless_forced(store, small, capsys):
    write_store()
    nrms.run(MIND)
    trained = nrms.checkpoint_path(MIND).stat().st_mtime_ns

    nrms.run(MIND)
    assert "already trained" in capsys.readouterr().out
    assert nrms.checkpoint_path(MIND).stat().st_mtime_ns == trained

    nrms.run(MIND, force=True)
    assert nrms.checkpoint_path(MIND).stat().st_mtime_ns != trained


def test_the_grid_sweeps_one_axis_at_a_time_from_the_registrys_spec(store, small):
    write_store()
    tried = nrms.grid(MIND, {"heads": (2, 1), "negatives": (2, 1)})

    # Three cells, not four: the spec's own value appears on both axes and is
    # trained once. A grid that re-trained it would put two rows in the table
    # for one configuration and invite a reader to wonder which was real.
    variants = [report["variant"] for report in tried]
    assert len(variants) == len(set(variants)) == 3
    assert variants.count(nrms.variant_of(SMALL)) == 1
    for report in tried:
        assert report["model_bytes"] > 0
        assert "cost" in report
    # Every cell is a ledger row, which is what makes the grid a table the
    # design note can print with its losers in it.
    assert len({row["variant"] for row in ledger.load() if row["stage"] == nrms.STAGE}) == 3


def test_the_grid_document_says_the_papers_number_is_outstanding(store, small):
    write_store()
    tried = nrms.grid(MIND, {"heads": (2,)})
    text = nrms.document(tried, MIND)
    assert "Outstanding" in text and nrms.PAPER["source"] in text
    assert nrms.variant_of(SMALL) in text
