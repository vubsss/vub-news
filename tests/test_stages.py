import dataclasses

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
    assert build.main([]) == 0


def test_unknown_force_target_is_rejected():
    assert build.main(["--force", "nonsense"]) == 2


def test_unbuilt_stages_are_reported_as_such():
    stage, dataset = STAGES[0], DATASETS["mind"]
    if stage.run is None:
        assert build.status(stage, dataset, forced=set()) == build.NOT_BUILT


def test_force_marks_a_done_stage_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path)
    dataset = DATASETS["mind"]
    stage = dataclasses.replace(STAGES[0], run=lambda config: None)
    stages.mark_done(stage, dataset)

    assert build.status(stage, dataset, forced=set()) == build.DONE
    assert build.status(stage, dataset, forced={stage.name}) == build.PENDING
