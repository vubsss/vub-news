"""The trade-off ledger: a row round-trips, a re-recorded key replaces, a
blank measurement renders as a blank rather than failing or vanishing, and
the A1 seed reads what the A1 modules wrote."""

import json

import pytest

from pipeline import ledger, paths


@pytest.fixture(autouse=True)
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path / "artifacts"


def row(**overrides):
    base = {
        "dataset": "mind",
        "stage": "retrieve",
        "variant": "ann",
        "split": "validation",
        "auc": 0.65,
        "auc_lo": 0.64,
        "auc_hi": 0.66,
        "index_bytes": 200_000_000,
        "p99_ms": 1.5,
    }
    base.update(overrides)
    return base


def test_a_recorded_row_comes_back_with_every_column_present():
    ledger.record(row())
    (loaded,) = ledger.load()
    assert set(loaded) == set(ledger.COLUMNS)
    assert loaded["auc"] == 0.65
    assert loaded["index_bytes"] == 200_000_000
    assert loaded["model_bytes"] is None


def test_recording_the_same_key_again_replaces_rather_than_duplicates(artifacts):
    ledger.record(row(auc=0.60))
    ledger.record(row(auc=0.65))
    assert [r["auc"] for r in ledger.load()] == [0.65]
    # And the file on disk agrees, for a reader that does not go through load.
    lines = (artifacts / ledger.RESULTS).read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["auc"] == 0.65


def test_a_different_key_is_a_second_row():
    ledger.record(row(variant="ann"))
    ledger.record(row(variant="bm25"))
    assert sorted(r["variant"] for r in ledger.load()) == ["ann", "bm25"]


def test_a_row_without_its_key_is_refused():
    with pytest.raises(ledger.LedgerError, match="variant"):
        ledger.record(row(variant=None))


def test_a_column_the_schema_does_not_know_is_refused():
    with pytest.raises(ledger.LedgerError, match="latency"):
        ledger.record(row(latency=3.0))


def test_a_missing_measurement_renders_as_a_blank_cell_not_an_error():
    ledger.record(row(p99_ms=None, model_bytes=None))
    text = ledger.document(ledger.load())
    line = next(l for l in text.splitlines() if "`ann`" in l)
    assert "—" in line
    assert "0.6500 [0.6400, 0.6600]" in line
    assert "190.7 MB" in line


def test_a_delta_is_rendered_against_the_variant_it_was_paired_with():
    ledger.record(row(variant="rerank", delta_vs="nrms", delta=0.012, delta_lo=0.008, delta_hi=0.016))
    text = ledger.document(ledger.load())
    assert "+0.0120 [+0.0080, +0.0160] vs `nrms`" in text


def test_render_writes_the_document_grouped_by_dataset_then_stage(artifacts):
    ledger.record(row(dataset="ebnerd", stage="rerank", variant="full"))
    ledger.record(row(dataset="mind", stage="retrieve", variant="ann"))
    file = ledger.render()
    text = file.read_text()
    assert text.index("## ebnerd") < text.index("## mind")
    assert "### rerank" in text and "### retrieve" in text


def test_render_with_nothing_recorded_says_so(artifacts):
    text = ledger.render().read_text()
    assert "No rows recorded yet" in text


def test_the_a1_seed_reads_the_evaluate_report_and_leaves_unmeasured_cells_blank(
    artifacts, monkeypatch
):
    from pipeline import evaluate
    from pipeline.datasets import DATASETS

    config = DATASETS["mind"]
    monkeypatch.setattr(
        type(config), "artifacts_dir", property(lambda self: artifacts / self.name)
    )
    directory = artifacts / "mind" / evaluate.EVALUATE_DIR
    directory.mkdir(parents=True)
    report = {
        "retriever": "ann",
        "split": "validation",
        "results": [
            {"slice": "overall", "metric": "auc", "value": 0.65, "lo": 0.64, "hi": 0.66},
            {"slice": "overall", "metric": "mrr", "value": 0.35, "lo": 0.34, "hi": 0.36},
            {"slice": "cold", "metric": "auc", "value": 0.55, "lo": 0.54, "hi": 0.56},
        ],
    }
    (directory / "ann-validation.json").write_text(json.dumps(report))

    rows = ledger.seed_a1()

    (seeded,) = [r for r in rows if r["dataset"] == "mind"]
    assert seeded["variant"] == "ann" and seeded["stage"] == ledger.A1_STAGE
    assert seeded["auc"] == 0.65 and seeded["auc_lo"] == 0.64
    assert seeded["mrr"] == 0.35
    # No bench, no timings, no index on disk in this fixture: blank, not invented.
    assert seeded["p99_ms"] is None
    assert seeded["index_bytes"] is None
    assert seeded["train_seconds"] is None
