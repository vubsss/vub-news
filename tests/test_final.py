"""`test` is scored once, and this is the module where "once" is a property
somebody can check rather than a promise. So the tests are about the record and
the refusal, about engineering columns being copied instead of re-measured, and
about the three-way table saying "outstanding" where a number is genuinely not
known -- a blank cell in that column reads as agreement, which is the one thing
it must not say."""

import json

import pytest

from pipeline import final, ledger, paths

from small_store import MIND, small, store  # noqa: F401 -- fixtures


def a_row(split, variant, stage="rerank", **extra):
    return ledger.record(
        {
            "dataset": "mind",
            "stage": stage,
            "variant": variant,
            "split": split,
            **extra,
        }
    )


# --- the record -------------------------------------------------------------


def test_the_first_scoring_is_recorded_with_its_invocation(store):
    entry = final.record_state("mind", "python -m pipeline.final", arms=19, seconds=61.0)

    assert entry["run"] == 1
    assert entry["split"] == "test"
    assert entry["invocation"] == "python -m pipeline.final"
    assert entry["arms"] == 19
    assert json.loads(final.state_path().read_text()) == [entry]


def test_a_second_scoring_appends_rather_than_replaces(store):
    """The point of the file. A repository where `test` was scored twice has to
    say so in something durable -- if the record could be overwritten, the
    second run would erase the evidence that there was a first."""
    final.record_state("mind", "first", 19, 60.0)
    second = final.record_state("mind", "second", 19, 58.0)

    history = json.loads(final.state_path().read_text())
    assert [entry["run"] for entry in history] == [1, 2]
    assert history[0]["invocation"] == "first"
    assert second["run"] == 2


def test_each_dataset_counts_its_own_runs(store):
    final.record_state("mind", "first", 19, 60.0)
    entry = final.record_state("ebnerd", "first", 19, 60.0)

    assert entry["run"] == 1
    assert len(final.load_state()) == 2


def test_the_record_names_the_commit_the_numbers_came_from(store):
    entry = final.record_state("mind", "x", 1, 1.0)
    # A checkout has one; the field exists either way, because a scoring that
    # failed for want of `git` would be the worst possible reason to run this
    # module twice.
    assert "commit" in entry


# --- the refusal ------------------------------------------------------------


def test_a_second_run_is_refused_and_says_when_the_first_was(store, monkeypatch):
    final.record_state("mind", "python -m pipeline.final", 19, 60.0)
    monkeypatch.setattr(
        final, "evaluate_all", lambda *a, **k: pytest.fail("test was re-scored")
    )

    with pytest.raises(final.AlreadyScored, match="was scored on"):
        final.run(MIND)


def test_the_refusal_names_the_file_that_force_would_append_to(store):
    final.record_state("mind", "python -m pipeline.final", 19, 60.0)
    with pytest.raises(final.AlreadyScored) as refused:
        final.run(MIND)

    assert final.STATE in str(refused.value)
    assert "--force" in str(refused.value)


def test_the_cli_refuses_without_scoring_anything(store, monkeypatch, capsys):
    final.record_state("mind", "python -m pipeline.final", 19, 60.0)
    monkeypatch.setattr(
        final, "evaluate_all", lambda *a, **k: pytest.fail("test was re-scored")
    )

    assert final.main(["--dataset", "mind"]) == 2
    assert "refused" in capsys.readouterr().out


# --- the engineering columns ------------------------------------------------


def test_the_test_rows_take_their_bytes_and_milliseconds_from_validation(store):
    """A second latency measurement is a second chance to select. The model is
    the same model, so the numbers are the same numbers, and the row says which
    run they came from."""
    a_row("validation", "full", model_bytes=4096, p50_ms=1.25, rows_per_s=800.0)
    a_row("test", "full", auc=0.61)

    final.copy_engineering(MIND)

    row = next(
        entry
        for entry in ledger.load()
        if entry["split"] == "test" and entry["variant"] == "full"
    )
    assert row["model_bytes"] == 4096
    assert row["p50_ms"] == 1.25
    assert row["auc"] == 0.61  # the functional half is the test run's own
    assert "measured on validation" in row["note"]


def test_a_note_that_was_already_there_is_kept(store):
    a_row("validation", "full", model_bytes=4096)
    a_row("test", "full", note="19 arms, one frame read once")

    final.copy_engineering(MIND)

    row = next(
        entry
        for entry in ledger.load()
        if entry["split"] == "test" and entry["variant"] == "full"
    )
    assert row["note"].startswith("19 arms")
    assert row["note"].endswith("measured on validation")


def test_an_arm_that_skipped_validation_is_named_rather_than_filled_in(store):
    """The note's story is chosen on tune, reported on validation, confirmed on
    test. A row that skipped the middle step did not follow it, and inventing
    its engineering columns would hide that."""
    a_row("validation", "full", model_bytes=4096)
    a_row("test", "full")
    a_row("test", "-history")

    final.copy_engineering(MIND)

    assert final.unmatched(MIND) == ["-history"]
    orphan = next(
        entry
        for entry in ledger.load()
        if entry["split"] == "test" and entry["variant"] == "-history"
    )
    assert orphan["model_bytes"] is None


def test_columns_are_matched_on_the_model_not_on_the_dataset(store):
    """`(stage, variant)` identifies a model; the split identifies a scoring of
    it. Matching any more loosely would copy one dataset's milliseconds onto
    another's rows."""
    ledger.record(
        {
            "dataset": "ebnerd",
            "stage": "rerank",
            "variant": "full",
            "split": "validation",
            "model_bytes": 999,
        }
    )
    a_row("test", "full")

    final.copy_engineering(MIND)

    row = next(
        entry
        for entry in ledger.load()
        if entry["dataset"] == "mind" and entry["split"] == "test"
    )
    assert row["model_bytes"] is None


# --- the three-way check ----------------------------------------------------


def test_a_leaderboard_nobody_recorded_is_outstanding_not_zero(store):
    assert final.leaderboard(MIND, "rerank") is None

    moved = final.shifts(MIND, "rerank")
    last = moved["steps"][-1]
    assert last["to"] == "leaderboard"
    assert last["delta"] is None and last["sign"] is None


def test_a_recorded_leaderboard_score_is_read_back(store):
    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.ARTIFACTS_DIR / final.LEADERBOARD).write_text(
        json.dumps({"mind": {"rerank": {"auc": 0.6431}}}), encoding="utf-8"
    )
    assert final.leaderboard(MIND, "rerank") == 0.6431


def test_a_bare_number_is_accepted_as_well_as_a_record(store):
    """Whoever types this in has just read it off a screenshot. Refusing the
    shorter spelling would make the file harder to write correctly, and the
    file being written correctly is the whole of its value."""
    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.ARTIFACTS_DIR / final.LEADERBOARD).write_text(
        json.dumps({"mind": {"nrms": 0.58}}), encoding="utf-8"
    )
    assert final.leaderboard(MIND, "nrms") == 0.58


def test_each_step_of_the_three_way_reports_its_sign(store, monkeypatch):
    monkeypatch.setattr(
        final.three_way,
        "load",
        lambda config, retriever, split: {
            "auc": {"value": 0.60 if split == "validation" else 0.59}
        },
    )
    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.ARTIFACTS_DIR / final.LEADERBOARD).write_text(
        json.dumps({"mind": {"rerank": 0.62}}), encoding="utf-8"
    )

    steps = final.shifts(MIND, "rerank")["steps"]
    assert [step["sign"] for step in steps] == ["-", "+"]
    assert steps[0]["delta"] == pytest.approx(-0.01)
    assert steps[1]["delta"] == pytest.approx(0.03)


# --- the page ---------------------------------------------------------------


def test_the_page_says_which_run_it_is_and_where_the_milliseconds_came_from(store):
    a_row("validation", "full", model_bytes=4096, p50_ms=1.25)
    a_row("test", "full", auc=0.61, auc_lo=0.60, auc_hi=0.62)
    final.copy_engineering(MIND)
    entry = final.record_state("mind", "python -m pipeline.final", 1, 12.0)

    written = final.document(MIND, entry)

    assert "scored once" in written
    assert "run **1**" in written
    assert "not re-measured" in written
    assert "0.6100" in written
    assert "outstanding" in written.lower()  # no leaderboard score recorded


def test_a_re_scored_dataset_says_so_on_its_own_page(store):
    """Whoever reads the page next has to be told, not left to find the file."""
    final.record_state("mind", "first", 1, 12.0)
    entry = final.record_state("mind", "second", 1, 12.0)

    written = final.document(MIND, entry)
    assert "run **2**" in written
    assert "more than once" in written


# --- end to end -------------------------------------------------------------


def test_the_whole_thing_runs_once_and_then_refuses(store, small, capsys):
    """The one path that matters, over a real store: build the frame the
    rebuild does not, score every retriever, run the arms, copy the columns,
    write the record -- and then refuse."""
    from small_store import write_store
    from pipeline import rerank

    write_store()
    rerank.fit_one(MIND, MIND.rerank)
    entry = final.run(MIND, resamples=20)

    assert entry["run"] == 1 and entry["arms"] > 0
    assert (paths.ARTIFACTS_DIR / "test-once-mind.md").exists()
    # The frame the rebuild does not build.
    assert final.features.path_for(MIND, "test").exists()
    # Every retriever has a test report beside its validation one.
    for retriever in final.evaluate.RETRIEVERS:
        assert (
            MIND.artifacts_dir / final.evaluate.EVALUATE_DIR / f"{retriever}-test.json"
        ).exists()

    with pytest.raises(final.AlreadyScored):
        final.run(MIND, resamples=20)


def test_a_forced_second_run_is_allowed_and_recorded_as_the_second(store, small):
    from small_store import write_store
    from pipeline import rerank

    write_store()
    rerank.fit_one(MIND, MIND.rerank)
    final.run(MIND, resamples=20)
    again = final.run(MIND, resamples=20, force=True)

    assert again["run"] == 2
    assert "more than once" in (paths.ARTIFACTS_DIR / "test-once-mind.md").read_text()
