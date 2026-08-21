"""Per-stage cost: that it is recorded, that it is the stage's own, and that a
stage which fails records nothing."""

import dataclasses
import json
import time

import pytest

import build
from pipeline import paths, stages, timings
from pipeline.datasets import DATASETS
from pipeline.stages import STAGES


@pytest.fixture(autouse=True)
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path / "artifacts"


def rows(artifacts):
    path = artifacts / timings.RESULTS
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_stage_records_its_wall_time_and_peak(artifacts):
    with timings.measure("bm25", "mind", "run-1"):
        pass

    (row,) = rows(artifacts)
    assert row["stage"] == "bm25"
    assert row["dataset"] == "mind"
    assert row["run"] == "run-1"
    assert row["seconds"] >= 0
    assert row["peak_rss_mb"] > 0


def test_the_peak_is_what_was_resident_during_the_stage(artifacts, monkeypatch):
    """Why the sampler exists. `ru_maxrss` is the whole process's high-water
    mark and never comes down, so a stage that allocates 8 GB and frees it is
    indistinguishable afterwards from one that allocated nothing — and it is
    exactly that stage which decides how many CPUs the SLURM job has to buy.

    Driven through a fake reading rather than a real allocation: under the
    full suite the interpreter already holds gigabytes of freed arenas, so
    allocating 200 MB need not move RSS at all and the test would pass or fail
    on what ran before it."""
    monkeypatch.setattr(timings, "INTERVAL", 0.001)
    resident = [100.0]
    monkeypatch.setattr(timings, "resident_mb", lambda: resident[0])

    with timings.measure("ingest", "mind", "run-1"):
        resident[0] = 9000.0
        time.sleep(0.05)
        resident[0] = 100.0

    (row,) = rows(artifacts)
    assert row["peak_rss_mb"] == 9000.0


def test_resident_mb_reads_this_process(artifacts):
    assert timings.resident_mb() > 1


def test_a_stage_that_raises_records_nothing(artifacts):
    with pytest.raises(ValueError):
        with timings.measure("embed", "mind", "run-1"):
            raise ValueError("no vectors")

    assert not (artifacts / timings.RESULTS).exists()


def test_the_newest_row_per_cell_wins(artifacts):
    for run, seconds in (("run-1", 10.0), ("run-2", 4.0)):
        timings.record(
            {
                "run": run,
                "host": "laptop",
                "cpus": 8,
                "dataset": "mind",
                "stage": "bm25",
                "seconds": seconds,
                "peak_rss_mb": 100.0,
            }
        )

    assert [row["seconds"] for row in timings.latest()] == [4.0]


def test_rows_are_reported_in_the_order_the_stages_run(artifacts):
    for stage in ("evaluate", "acquire", "bm25"):
        timings.record(
            {
                "run": "run-1",
                "host": "laptop",
                "cpus": 8,
                "dataset": "mind",
                "stage": stage,
                "seconds": 1.0,
                "peak_rss_mb": 100.0,
            }
        )

    assert [row["stage"] for row in timings.latest()] == [
        "acquire",
        "bm25",
        "evaluate",
    ]


def test_the_document_names_the_stage_that_sets_the_ceiling(artifacts):
    for stage, mb in (("bm25", 1000.0), ("ingest", 9000.0)):
        timings.record(
            {
                "run": "2026-08-21T00:00:00+00:00",
                "host": "gnode025",
                "cpus": 20,
                "dataset": "mind",
                "stage": stage,
                "seconds": 30.0,
                "peak_rss_mb": mb,
            }
        )

    text = timings.write_document().read_text()

    assert "8.8 GB" in text
    # 9000 MB at Ada's 3000 MB per CPU.
    assert "-c 3" in text
    assert "`ingest`" in text


def test_no_measurements_is_not_an_error(artifacts):
    assert "Nothing measured yet." in timings.document(timings.latest())


def test_a_build_that_runs_a_stage_leaves_a_row(tmp_path, monkeypatch, artifacts):
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    # Stage is frozen, so the fake is a replacement rather than a patched
    # attribute, and build.py reads STAGES by module attribute.
    fake = dataclasses.replace(STAGES[0], run=lambda config, force: None)
    monkeypatch.setattr(build, "STAGES", (fake,))

    assert build.main(["--dataset", "mind"]) == 0

    assert [row["stage"] for row in rows(artifacts)] == ["acquire"]
    assert (artifacts / timings.DOCUMENT).exists()
