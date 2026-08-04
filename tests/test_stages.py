import dataclasses
import os

import build
from pipeline import paths, stages
from pipeline.datasets import DATASETS
from pipeline.stages import STAGES


def test_stage_names_are_unique():
    names = [stage.name for stage in STAGES]
    assert len(names) == len(set(names))


def test_stages_are_in_dependency_order():
    order = {stage.name: i for i, stage in enumerate(STAGES)}
    for earlier, later in [
        ("acquire", "ingest"),
        ("ingest", "split"),
        ("ingest", "preprocess"),
        ("split", "bm25"),
        ("preprocess", "bm25"),
        ("split", "ann"),
        ("embed", "ann"),
        ("bm25", "evaluate"),
        ("ann", "evaluate"),
    ]:
        assert order[earlier] < order[later], f"{earlier} must precede {later}"


def test_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path)
    stage, dataset = STAGES[0], DATASETS["mind"]

    assert not stages.is_done(stage, dataset)
    stages.mark_done(stage, dataset)
    assert stages.is_done(stage, dataset)
    stages.clear(stage, dataset)
    assert not stages.is_done(stage, dataset)


def test_checkpoints_are_per_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path)
    stage = STAGES[0]

    stages.mark_done(stage, DATASETS["mind"])
    assert not stages.is_done(stage, DATASETS["ebnerd"])


def test_build_runs_on_a_clone_with_no_data():
    # --plan, because the bare command now really downloads things.
    assert build.main(["--plan"]) == 0


def test_acquisition_failure_exits_non_zero_without_a_traceback(
    tmp_path, monkeypatch, capsys
):
    # Isolated from whatever this machine has already downloaded, so the run
    # really does need a token. Also stops the test hitting the network.
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    # ROOT too, or the repo's real .env would hand the run a working token.
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert build.main(["--dataset", "mind"]) == 1

    stderr = capsys.readouterr().err
    assert "HF_TOKEN" in stderr
    assert "Traceback" not in stderr


def test_env_file_supplies_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("# comment\n\nHF_TOKEN='from-file'\n")

    build.load_env_file()

    assert os.environ["HF_TOKEN"] == "from-file"


def test_exported_variable_beats_the_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "from-shell")
    (tmp_path / ".env").write_text("HF_TOKEN=from-file\n")

    build.load_env_file()

    assert os.environ["HF_TOKEN"] == "from-shell"


def test_missing_env_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)

    build.load_env_file()


def test_unknown_force_target_is_rejected():
    assert build.main(["--force", "nonsense"]) == 2


def test_unbuilt_stages_are_reported_as_such():
    stage, dataset = STAGES[0], DATASETS["mind"]
    if stage.run is None:
        assert build.status(stage, dataset, forced=set()) == build.NOT_BUILT


def test_force_marks_a_done_stage_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path)
    dataset = DATASETS["mind"]
    stage = dataclasses.replace(STAGES[0], run=lambda config, force=False: None)
    stages.mark_done(stage, dataset)

    assert build.status(stage, dataset, forced=set()) == build.DONE
    assert build.status(stage, dataset, forced={stage.name}) == build.PENDING
