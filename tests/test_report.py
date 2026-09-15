"""The note's tables are generated so that the note and the repository cannot
disagree. So the tests are about the two ways a generator can break that
promise: by emitting a number that has no ledger row behind it, and by
producing TeX that will not compile because a variant name contained TeX."""

import pytest

from pipeline import ledger, paths, report
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    return tmp_path


def a_row(stage, variant, split="validation", **extra):
    return ledger.record(
        {
            "dataset": "mind",
            "stage": stage,
            "variant": variant,
            "split": split,
            **extra,
        }
    )


# --- escaping ---------------------------------------------------------------


def test_a_variant_name_full_of_tex_is_escaped():
    """Variant names are `cut@100`, `content+history`, `drop:none`,
    `binary-l31-24h`. An unescaped underscore is a compile error three hundred
    lines from the thing that caused it."""
    assert report.escape("a_b") == "a\\_b"
    assert report.escape("100%") == "100\\%"
    assert report.escape("a&b") == "a\\&b"
    assert report.escape("#{x}") == "\\#\\{x\\}"


def test_a_missing_value_is_an_em_dash_not_a_zero():
    """A table of an unfinished project should look unfinished."""
    assert report.escape(None) == "---"


def test_every_generated_table_escapes_what_it_prints(artifacts):
    a_row("rerank", "drop:none_100%", auc=0.6)
    written = report.decision_table(MIND, "rerank")

    assert "drop:none\\_100\\%" in written
    assert "none_100%" not in written


# --- what is marked chosen --------------------------------------------------


def test_the_tick_follows_the_registry_not_the_best_row(artifacts):
    """The chosen option is the one the code will run. If that is not the one
    that won its sweep, the table has to be able to show it losing -- a
    generator that ticked the highest AUC could not express that, and the
    disagreement is exactly what a reader needs to see."""
    from pipeline import rerank

    picked = rerank.variant_of(MIND.rerank)
    a_row("rerank", picked, auc=0.60)
    a_row("rerank", "some-other-arm", auc=0.99)

    written = report.decision_table(MIND, "rerank")
    chosen_line = next(
        line for line in written.splitlines() if line.startswith(report.escape(picked))
    )
    other = next(line for line in written.splitlines() if "some-other-arm" in line)

    assert "checkmark" in chosen_line
    assert "checkmark" not in other


def test_a_registry_entry_with_no_row_is_flagged(artifacts):
    """A configuration nobody measured is more serious than a missing cell, so
    it is said in the file rather than left to the reader."""
    a_row("rerank", "an-arm-that-is-not-the-registrys", auc=0.6)
    written = report.decision_table(MIND, "rerank")

    assert "WARNING" in written
    assert "has no row here" in written


def test_a_stage_the_registry_names_nothing_for_ticks_nothing(artifacts):
    a_row("serve", "flat-fp32-k100", p50_ms=12.0)
    assert report.chosen(MIND, "serve") is None
    assert "checkmark" not in report.decision_table(MIND, "serve")


# --- losers are in the table ------------------------------------------------


def test_every_row_the_sweep_produced_is_in_the_table(artifacts):
    """The note's claim is that each choice was made against alternatives. A
    table showing only the winner is that claim without its evidence."""
    for leaves in (4, 31, 127):
        a_row("rerank", f"binary-l{leaves}-24h", auc=0.5 + leaves / 1000)

    written = report.decision_table(MIND, "rerank")
    for leaves in (4, 31, 127):
        assert f"binary-l{leaves}-24h" in written


def test_a_stage_with_no_rows_says_so_rather_than_emitting_an_empty_table(artifacts):
    written = report.decision_table(MIND, "nrms")

    assert "no rows recorded" in written
    # Still a well-formed tabular, so the note compiles either way.
    assert written.count("\\begin{tabular}") == 1
    assert written.count("\\end{tabular}") == 1


def test_rows_of_another_dataset_never_reach_this_ones_table(artifacts):
    ledger.record(
        {
            "dataset": "ebnerd",
            "stage": "rerank",
            "variant": "danish-only-arm",
            "split": "validation",
            "auc": 0.7,
        }
    )
    assert "danish-only-arm" not in report.decision_table(MIND, "rerank")


# --- the curves -------------------------------------------------------------


@pytest.mark.parametrize(
    "variant, pattern, expected",
    [
        ("cut@100", r"cut@(\d+)\b", 100.0),
        ("cut@50 (ivf)", r"cut@(\d+)\b", 50.0),
        ("bm25@200k", r"@(\d+k?)(?::|$)", 200_000.0),
        ("bm25@200k:read", r"@(\d+k?)(?::|$)", 200_000.0),
        ("flat-fp32-k100", r"-k(\d+)", 100.0),
        ("full", r"cut@(\d+)\b", None),
    ],
)
def test_the_axis_is_read_out_of_the_variants_own_name(variant, pattern, expected):
    """The sweep put the axis in the name, so that is where it is read from.
    The alternative is a ledger column per swept axis, which every future axis
    would have to add."""
    assert report.axis_from(variant, pattern) == expected


def test_a_curve_is_sorted_by_its_axis_not_by_variant_name(artifacts):
    """`cut@200` sorts before `cut@50` as a string. A plot-ready table that
    came out in that order would be plotted in it."""
    for k in (200, 50, 100):
        a_row("ablation", f"cut@{k}", auc=0.6 + k / 10_000)

    written = report.curve_table(MIND, "k")
    numbers = [
        line.split("&")[0].strip()
        for line in written.splitlines()
        if line and line[0].isdigit()
    ]
    assert numbers == ["50", "100", "200"]


def test_a_row_with_no_axis_in_its_name_is_not_a_point(artifacts):
    a_row("ablation", "full", auc=0.64)
    a_row("ablation", "cut@100", auc=0.61)

    written = report.curve_table(MIND, "k")
    assert "0.6100" in written
    assert "0.6400" not in written


def test_a_curve_with_no_rows_still_compiles(artifacts):
    written = report.curve_table(MIND, "rounds")
    assert "no rounds rows recorded" in written
    assert "\\end{tabular}" in written


# --- the serving table ------------------------------------------------------


def test_the_serving_table_names_the_command_that_would_fill_it(artifacts):
    written = report.serving_table(MIND)
    assert "has not been run" in written
    assert "pipeline.serve" in written


def test_the_serving_table_shows_the_four_stages_behind_the_total(artifacts):
    """The ledger keeps one p50 per variant; this table wants the four stages
    behind it, so it reads the benchmark's own file. Both come out of one run
    and the ledger's number is this row's total."""
    import json

    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.ARTIFACTS_DIR / "bench-serve-mind.jsonl").write_text(
        json.dumps(
            {
                "variant": "flat-fp32-k100",
                "retrieve_p50_ms": 0.4,
                "features_p50_ms": 6.1,
                "nrms_p50_ms": 2.2,
                "gbdt_p50_ms": 0.9,
                "total_p50_ms": 9.6,
                "total_p99_ms": 21.3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    written = report.serving_table(MIND)

    assert "flat-fp32-k100" in written
    for cell in ("0.40", "6.10", "2.20", "0.90", "9.60", "21.30"):
        assert cell in written


# --- writing them out -------------------------------------------------------


def test_every_table_the_note_inputs_is_written(artifacts):
    written = report.write_tables(MIND)
    names = {path.name for path in written}

    for stage in report.STAGES:
        assert f"mind-{stage}.tex" in names
    for curve in report.CURVES:
        assert f"mind-curve-{curve}.tex" in names
    assert "mind-serving.tex" in names


def test_the_tables_live_in_the_repository_not_in_artifacts(artifacts):
    """`artifacts/` is gitignored. A note whose tables vanish on a clean
    checkout is not a note a grader can build."""
    written = report.write_tables(MIND)

    for path in written:
        assert report.REPORT_DIR in path.parts
        assert "artifacts" not in path.parts


def test_the_note_inputs_a_table_for_every_file_the_generator_writes():
    """The two halves of the contract, checked against each other: a table the
    note asks for and the generator does not write renders as a warning in the
    PDF, and one the generator writes that the note never asks for is dead."""
    note = (
        report.paths.REPO_ROOT / report.REPORT_DIR / report.NOTE
    ).read_text(encoding="utf-8")
    asked = set(
        line.split("{")[1].split("}")[0]
        for line in note.splitlines()
        if line.strip().startswith("\\inputtable{")
    )

    assert asked, "the note inputs no tables at all"
    for name in asked:
        dataset, rest = name.split("-", 1)
        assert dataset in ("mind", "ebnerd")
        assert (
            rest in report.STAGES
            or rest == "serving"
            or rest.removeprefix("curve-") in report.CURVES
        ), f"the note asks for {name}, which nothing generates"
