import dataclasses
import os

import build
from pipeline import paths, stages
from pipeline.datasets import DATASETS, SubmissionSpec
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
        ("bm25", "predict"),
        ("ann", "predict"),
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
    # REPO_ROOT too, or the repo's real .env would hand the run a working
    # token: load_env_file reads the credential from beside the code, not from
    # wherever this run was told to write its outputs.
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert build.main(["--dataset", "mind"]) == 1

    stderr = capsys.readouterr().err
    assert "HF_TOKEN" in stderr
    assert "Traceback" not in stderr


def test_a_missing_embedding_artifact_exits_without_a_traceback(
    tmp_path, monkeypatch, capsys
):
    """The state a fresh clone is in: MIND's vectors are generated on a hosted
    GPU, so until the notebook has been run and its output uploaded the stage
    cannot proceed. The message names the two things the user has to do, and a
    traceback would bury exactly the part they need to read."""
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    # The absent drive id is constructed, not read from the registry, which
    # now carries a real one — a test whose premise expires the moment the
    # project moves on is worse than no test.
    mind = DATASETS["mind"]
    monkeypatch.setitem(
        DATASETS,
        "mind",
        dataclasses.replace(
            mind,
            embeddings=dataclasses.replace(mind.embeddings, gdrive_file_id=None),
        ),
    )
    # Every stage before embed is already done, so the run reaches it.
    for stage in STAGES:
        if stage.name == "embed":
            break
        stages.mark_done(stage, DATASETS["mind"])

    assert build.main(["--dataset", "mind"]) == 1

    stderr = capsys.readouterr().err
    assert "gdrive_file_id" in stderr
    assert "Traceback" not in stderr


def test_env_file_supplies_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("# comment\n\nHF_TOKEN='from-file'\n")

    paths.load_env_file()

    assert os.environ["HF_TOKEN"] == "from-file"


def test_exported_variable_beats_the_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "from-shell")
    (tmp_path / ".env").write_text("HF_TOKEN=from-file\n")

    paths.load_env_file()

    assert os.environ["HF_TOKEN"] == "from-shell"


def test_missing_env_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)

    paths.load_env_file()


def test_a_stage_not_built_for_a_dataset_earns_no_checkpoint(
    tmp_path, monkeypatch, capsys
):
    """A competition whose submission is a url and nothing else — which both
    of them were before their submission ticket landed, and which a third
    dataset would be added as. A checkpoint written for it would have every
    rebuild after that ticket skip the stage the ticket added."""
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    ebnerd = dataclasses.replace(
        DATASETS["ebnerd"],
        submission=SubmissionSpec(
            competition_url=DATASETS["ebnerd"].submission.competition_url
        ),
    )
    monkeypatch.setitem(DATASETS, "ebnerd", ebnerd)
    for stage in STAGES:
        if stage.name == "predict":
            break
        stages.mark_done(stage, ebnerd)

    assert build.main(["--dataset", "ebnerd"]) == 0

    assert "no submission built yet" in capsys.readouterr().out
    assert not stages.is_done(STAGES[-1], ebnerd)


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
