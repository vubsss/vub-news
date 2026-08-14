"""The comparison is tested where it could quietly lie: the interval-overlap
verdict, the metric whose interval is not a bracket, the guards against
comparing two reports that are not about the same measurement, and the reading
that has to say "not established" rather than name a winner."""

import json

import pytest

from pipeline import compare, evaluate, paths, retrieval
from pipeline.datasets import DATASETS

MIND, EBNERD = DATASETS["mind"], DATASETS["ebnerd"]


POPULATION = 100


def report(
    retriever,
    values,
    dataset="mind",
    split="validation",
    populations=None,
    **overrides,
):
    """A stored report shaped like evaluate's, with the given (slice, metric)
    values. Anything not named gets a placeholder, so a test states only the
    numbers it is about.

    values: {(slice, metric): (value, lo, hi)}
    populations: {slice: how many impressions it holds}
    """
    results = []
    for slice_name in evaluate.SLICES:
        held = (populations or {}).get(slice_name, POPULATION)
        for metric in evaluate.METRICS:
            value, lo, hi = values.get((slice_name, metric), (0.5, 0.4, 0.6))
            results.append(
                {
                    "slice": slice_name,
                    "metric": metric,
                    "population": held,
                    "n": held,
                    "value": value,
                    "lo": lo,
                    "hi": hi,
                }
            )
    return {
        "dataset": dataset,
        "retriever": retriever,
        "split": split,
        "history_k": retrieval.HISTORY_K,
        "impressions": POPULATION,
        "no_positive": 0,
        "all_positive": 0,
        "all_scores_tied": 0,
        "catalogue": 1000,
        "candidate_pool": 200,
        "coverage_of": "catalogue",
        "head_cutoff": 3,
        "list_depth": evaluate.LIST_DEPTH,
        "confidence": evaluate.CONFIDENCE,
        "resamples": 1000,
        "results": results,
        **overrides,
    }


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return paths.ARTIFACTS_DIR


def store(config, *reports):
    directory = config.artifacts_dir / evaluate.EVALUATE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for report in reports:
        path = directory / f"{report['retriever']}-{report['split']}.json"
        path.write_text(json.dumps(report), encoding="utf-8")


def compared(
    artifacts, values_lex, values_sem, config=MIND, populations=None, **overrides
):
    """Both retrievers stored and read back, as the command would read them.
    `overrides` land on the semantic report only, so a test can make the two
    disagree about what they measured."""
    store(
        config,
        report(
            compare.LEXICAL,
            values_lex,
            dataset=config.name,
            populations=populations,
        ),
        report(
            compare.SEMANTIC,
            values_sem,
            dataset=config.name,
            populations=populations,
            **overrides,
        ),
    )
    return compare.comparison(config, "validation")


# --- the verdict ------------------------------------------------------------


def test_disjoint_intervals_name_the_higher_retriever(artifacts):
    """The one case a difference may be called a difference."""
    c = compared(
        artifacts,
        {("overall", "auc"): (0.55, 0.54, 0.56)},
        {("overall", "auc"): (0.62, 0.61, 0.63)},
    )
    row = c.at("overall", "auc")

    assert row.established
    assert row.leader == compare.SEMANTIC
    assert row.higher == compare.SEMANTIC
    assert row.delta == pytest.approx(0.07)


def test_overlapping_intervals_establish_nothing_in_either_direction(artifacts):
    """The point of the ticket: a gap smaller than its own uncertainty is not
    a win for the retriever it happens to favour. If this ever names a winner,
    every 'X beats Y' in the design note becomes unsafe."""
    c = compared(
        artifacts,
        {("overall", "auc"): (0.550, 0.540, 0.561)},
        {("overall", "auc"): (0.560, 0.549, 0.570)},
    )
    row = c.at("overall", "auc")

    assert not row.established
    assert row.leader is None
    assert row.higher == compare.NOT_ESTABLISHED


def test_intervals_that_touch_at_a_bound_are_not_disjoint(artifacts):
    """Equality at the boundary is an overlap. The strict comparison is what
    keeps a difference of exactly nothing from being reported as one."""
    c = compared(
        artifacts,
        {("overall", "auc"): (0.55, 0.54, 0.56)},
        {("overall", "auc"): (0.57, 0.56, 0.58)},
    )

    assert c.at("overall", "auc").higher == compare.NOT_ESTABLISHED


def test_a_metric_with_no_interval_is_never_established(artifacts):
    """`--resamples 0` leaves the bounds null. No interval is a reason to say
    nothing, not a reason to say everything."""
    c = compared(
        artifacts,
        {("overall", "auc"): (0.55, None, None)},
        {("overall", "auc"): (0.62, None, None)},
    )

    assert c.at("overall", "auc").higher == compare.NOT_ESTABLISHED


def test_an_undefined_metric_compares_to_a_dash(artifacts):
    """An empty slice has no value to compare, and a dash is what says so."""
    c = compared(
        artifacts,
        {("cold", "auc"): (None, None, None)},
        {("cold", "auc"): (None, None, None)},
    )
    row = c.at("cold", "auc")

    assert row.higher == compare.UNDEFINED
    assert row.delta is None


def test_coverage_is_compared_by_value_and_marked_as_such(artifacts):
    """Coverage's interval is a width, not a bracket around its own value
    (evaluate.interval), so overlapping coverage intervals mean nothing. Run
    the overlap test on it and every coverage row would read 'not established'
    while the two retrievers genuinely reach different amounts of catalogue."""
    c = compared(
        artifacts,
        {("overall", "coverage"): (0.0566, 0.0483, 0.0497)},
        {("overall", "coverage"): (0.0543, 0.0463, 0.0477)},
    )
    row = c.at("overall", "coverage")

    assert not row.established
    assert row.leader == compare.LEXICAL
    assert row.higher == compare.LEXICAL + compare.BY_VALUE


# --- the guards -------------------------------------------------------------


def test_a_missing_report_names_the_command_that_produces_it(artifacts):
    store(MIND, report(compare.LEXICAL, {}))

    with pytest.raises(compare.ComparisonError) as error:
        compare.comparison(MIND, "validation")

    assert "python -m pipeline.evaluate" in str(error.value)
    assert compare.SEMANTIC in str(error.value)


def test_reports_from_different_populations_are_refused(artifacts):
    """A bm25 report from before a re-ingest against a fresh ann one would
    produce a difference between two populations wearing the clothes of a
    difference between two retrievers."""
    with pytest.raises(compare.ComparisonError) as error:
        compared(artifacts, {}, {}, impressions=99)

    assert "impressions" in str(error.value)


def test_reports_with_different_resample_counts_are_refused(artifacts):
    """Intervals from 100 resamples and from 1000 are not the same width for
    the same data, so an overlap between them is not evidence about either."""
    with pytest.raises(compare.ComparisonError) as error:
        compared(artifacts, {}, {}, resamples=100)

    assert "resamples" in str(error.value)


def test_a_slice_scored_over_different_populations_is_refused(artifacts):
    store(
        MIND,
        report(compare.LEXICAL, {}),
        report(compare.SEMANTIC, {}),
    )
    path = (
        MIND.artifacts_dir
        / evaluate.EVALUATE_DIR
        / f"{compare.SEMANTIC}-validation.json"
    )
    doctored = json.loads(path.read_text())
    doctored["results"][0]["population"] = 7
    path.write_text(json.dumps(doctored), encoding="utf-8")

    with pytest.raises(compare.ComparisonError) as error:
        compare.comparison(MIND, "validation")

    assert "different populations" in str(error.value)


# --- the table --------------------------------------------------------------


def test_the_table_covers_every_slice_and_every_metric(artifacts):
    """The ticket asks for both retrievers x all metrics x overall plus all
    four slices, in one table."""
    c = compared(artifacts, {}, {})
    header, rule, *rows = compare.table(c.rows).splitlines()

    assert len(rows) == len(evaluate.SLICES) * len(evaluate.METRICS)
    assert all(name in header for name in (compare.LEXICAL, compare.SEMANTIC))
    for slice_name in evaluate.SLICES:
        assert sum(row.split("|")[1].strip() == slice_name for row in rows) == len(
            evaluate.METRICS
        )


def test_every_cell_carries_its_interval(artifacts):
    c = compared(
        artifacts,
        {("overall", "auc"): (0.55, 0.54, 0.56)},
        {("overall", "auc"): (0.62, 0.61, 0.63)},
    )
    line = [
        row
        for row in compare.table(c.rows).splitlines()
        if row.startswith("| overall | auc")
    ]

    assert line == [
        "| overall | auc       |        100 | 100 | 0.5500 [0.5400, 0.5600] |"
        " 0.6200 [0.6100, 0.6300] |            +0.0700 | ann             |"
    ]


# --- the reading ------------------------------------------------------------


def test_the_reading_calls_an_overlapping_difference_not_established(artifacts):
    """The ticket's fourth line. The reading is what a reader quotes, so a
    verdict the table refuses must not reappear as prose."""
    c = compared(
        artifacts,
        {(s, "auc"): (0.550, 0.540, 0.561) for s in evaluate.SLICES},
        {(s, "auc"): (0.560, 0.549, 0.570) for s in evaluate.SLICES},
    )
    overall = compare.overall_bullet(c)

    assert "the intervals overlap" in overall
    assert "leads" not in overall


def test_the_reading_names_the_winner_where_the_intervals_separate(artifacts):
    everywhere = [(s, m) for s in evaluate.SLICES for m in compare.ACCURACY]
    c = compared(
        artifacts,
        {key: (0.55, 0.54, 0.56) for key in everywhere},
        {key: (0.62, 0.61, 0.63) for key in everywhere},
    )

    assert "ann leads on all 4 ranking metrics" in compare.overall_bullet(c)


def test_the_reading_calls_out_a_slice_where_the_ordering_reverses(artifacts):
    """A retriever that wins overall and loses on a slice is the finding the
    ticket is after, and it has to survive being read off a table of 35 rows."""
    everywhere = [(s, m) for s in evaluate.SLICES for m in compare.ACCURACY]
    lex = {key: (0.55, 0.54, 0.56) for key in everywhere}
    sem = {key: (0.62, 0.61, 0.63) for key in everywhere}
    for metric in compare.ACCURACY:
        sem[("cold", metric)] = (0.50, 0.49, 0.51)

    c = compared(artifacts, lex, sem)

    assert "The ordering reverses here" in compare.slice_bullet(c, "cold")
    assert "reverses" not in compare.slice_bullet(c, "warm")


def test_an_empty_slice_is_reported_as_empty_rather_than_scored(artifacts):
    """EB-NeRD's cold slice holds nobody. A reading that skipped it would let
    a reader assume it was checked."""
    empty = {("cold", m): (None, None, None) for m in evaluate.METRICS}
    c = compared(artifacts, empty, empty, populations={"cold": 0})

    assert "is empty" in compare.slice_bullet(c, "cold")


def test_a_retriever_at_chance_is_read_as_at_chance(artifacts):
    """An AUC interval containing 0.5 makes the retriever's ordering
    indistinguishable from a shuffle, and a gap against it has to be read
    against that rather than as a ranking result."""
    c = compared(
        artifacts,
        {("overall", "auc"): (0.5051, 0.5034, 0.5067)},
        {("overall", "auc"): (0.4984, 0.4963, 0.5004)},
    )

    assert compare.at_chance(c) == [compare.SEMANTIC]
    assert "contains 0.5" in compare.chance_bullet(c)


def test_the_beyond_accuracy_reading_refuses_to_call_higher_better(artifacts):
    """Diversity, novelty and coverage describe a list rather than score it."""
    c = compared(artifacts, {}, {})

    assert "Higher is not better" in compare.beyond_bullet(c)


# --- across the datasets ----------------------------------------------------


def flat(auc_lex, auc_sem):
    return (
        {(s, "auc"): auc_lex for s in evaluate.SLICES},
        {(s, "auc"): auc_sem for s in evaluate.SLICES},
    )


def both_datasets(artifacts, ebnerd_cold=POPULATION):
    """MIND where the semantic side leads, EB-NeRD where the lexical one does
    and no user is cold — the shape the real results have."""
    mind_lex, mind_sem = flat((0.55, 0.54, 0.56), (0.62, 0.61, 0.63))
    eb_lex, eb_sem = flat((0.5051, 0.5034, 0.5067), (0.4984, 0.4963, 0.5004))
    return [
        compared(artifacts, mind_lex, mind_sem, config=MIND),
        compared(
            artifacts,
            eb_lex,
            eb_sem,
            config=EBNERD,
            populations={"cold": ebnerd_cold},
        ),
    ]


def test_a_reversal_between_the_datasets_is_stated_as_one(artifacts):
    text = compare.cross_dataset(both_datasets(artifacts))

    assert "The ordering reverses between the datasets" in text
    assert "ann leads by 0.0700" in text
    assert "bm25 leads by 0.0067" in text


def test_the_confounded_explanations_are_named_and_refused(artifacts):
    """The ticket asks which candidate explanations the data supports and
    which it cannot distinguish. Language, embedding provenance and catalogue
    size all vary with the dataset and with nothing else, so the document has
    to name them and say that none of them can be told from the others."""
    text = compare.cross_dataset(both_datasets(artifacts))

    rows = {
        line.split("|")[1].strip(): line for line in text.splitlines() if "|" in line
    }
    for factor, _ in compare.CONFOUNDED:
        assert "cannot be separated" in rows[factor]
    assert MIND.language in text and EBNERD.language in text
    assert EBNERD.embeddings.model in text


def test_an_explanation_the_data_rules_out_is_ruled_out(artifacts):
    """EB-NeRD has no cold users at all, so cold-start starvation of the
    semantic user vector cannot be why the semantic side trails there."""
    explanation, evidence, verdict = compare.cold_start_row(
        both_datasets(artifacts, ebnerd_cold=0)
    )

    assert "Cold users" in explanation
    assert "0 cold impressions" in evidence
    assert verdict == "ruled out"


def test_a_dataset_where_both_arms_are_at_chance_is_not_a_semantic_failure(
    artifacts,
):
    """Both retrievers sit at chance on EB-NeRD, so whatever is happening
    there is not specific to the semantic arm — the one cross-dataset claim
    these results can actually support."""
    _, evidence, verdict = compare.semantic_failure_row(both_datasets(artifacts))

    assert "not supported" in verdict
    assert compare.LEXICAL in evidence and compare.SEMANTIC in evidence


# --- the command ------------------------------------------------------------


def test_the_command_writes_one_document_and_prints_it(artifacts, capsys):
    """The ticket's first line: one command, the whole comparison, no
    retrieval re-run."""
    both_datasets(artifacts)

    assert compare.main([]) == 0

    path = artifacts / compare.DOCUMENT.format(split="validation")
    text = path.read_text(encoding="utf-8")
    printed = capsys.readouterr().out

    assert text in printed
    assert text.startswith("# Lexical versus semantic retrieval")
    for config in (MIND, EBNERD):
        assert f"## {config.name}" in text
    assert "## Where the datasets disagree" in text


def test_one_dataset_leaves_out_the_cross_dataset_section(artifacts, capsys):
    """Nothing to compare across, so nothing is claimed across."""
    both_datasets(artifacts)

    assert compare.main(["--dataset", "mind"]) == 0

    text = (artifacts / compare.DOCUMENT.format(split="validation")).read_text()
    assert "## mind" in text
    assert "## ebnerd" not in text
    assert "Where the datasets disagree" not in text


def test_a_missing_split_fails_with_its_reason(artifacts, capsys):
    both_datasets(artifacts)

    assert compare.main(["--split", "test"]) == 2
    assert "python -m pipeline.evaluate" in capsys.readouterr().err
