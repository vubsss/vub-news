"""The harness is tested at seven seams: the ranking metrics against rankings
worked by hand, the beyond-accuracy metrics against lists worked by hand, the
degenerate-impression bookkeeping, the population slices, the bootstrap, the
train refusal, and the retriever-agnostic path through evaluate and its
command."""

import json

import numpy as np
import pandas as pd
import pytest

import dataclasses

from pipeline import (
    ann_index,
    bm25_index,
    counters,
    evaluate,
    features,
    ingest,
    nrms,
    paths,
    rerank,
)
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]


def ranked(rows):
    """rows: (impression_id, ranked_ids, scores)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([r[0] for r in rows], dtype="string"),
            "ranked_ids": [list(r[1]) for r in rows],
            "scores": [list(r[2]) for r in rows],
        }
    )


def behaviours(rows, split="validation", day="2019-11-14"):
    """rows: (impression_id, candidate_ids, labels)

    One per split, ordered so that train precedes validation precedes test,
    as the real temporal split guarantees.
    """
    return pd.DataFrame(
        {
            "impression_id": pd.Series([r[0] for r in rows], dtype="string"),
            # One user per impression, which keeps these fixtures 1:1 with the
            # history table now that it is keyed by the user rather than the
            # impression. `source` is the other half of that key.
            "user_id": pd.Series([f"u-{r[0]}" for r in rows], dtype="string"),
            "source": pd.Series(["train"] * len(rows), dtype="string"),
            "impression_time": pd.to_datetime([day] * len(rows)),
            # MIND ships no sessions, and the feature store carries the column
            # as null rather than leaving it out — so the fixtures do too.
            "session_id": pd.Series([None] * len(rows), dtype="string"),
            "candidate_ids": [list(r[1]) for r in rows],
            "labels": [list(r[2]) for r in rows],
            "split": pd.Series([split] * len(rows), dtype="string"),
        }
    )


def histories(rows):
    """rows: (impression_id, n_clicks) — keyed by the user it belongs to."""
    return pd.DataFrame(
        {
            "user_id": pd.Series([f"u-{r[0]}" for r in rows], dtype="string"),
            "source": pd.Series(["train"] * len(rows), dtype="string"),
            "click_history": [["a1"] * r[1] for r in rows],
            "n_clicks": [r[1] for r in rows],
        }
    )


def paired(rows, n_clicks=10):
    """The frame every measurement is positional in.

    rows: (impression_id, candidate_ids, labels, ranked_ids, scores). n_clicks
    is the user's history size, one per row or one for all of them.
    """
    clicks = n_clicks if isinstance(n_clicks, list) else [n_clicks] * len(rows)
    return pd.DataFrame(
        {
            "impression_id": pd.Series([r[0] for r in rows], dtype="string"),
            "candidate_ids": [list(r[1]) for r in rows],
            "labels": [list(r[2]) for r in rows],
            "ranked_ids": [list(r[3]) for r in rows],
            "scores": [list(r[4]) for r in rows],
            "n_clicks": clicks,
        }
    )


def catalogue(frame, categories):
    """A catalogue over the articles in `frame`, one category each.

    categories: article_id -> category. Popularity and head/tail come off the
    impressions in the frame, which is what the real one does too.
    """
    return evaluate.catalogue_of(
        pd.DataFrame(
            {
                "article_id": pd.Series(list(categories), dtype="string"),
                "category": pd.Series(list(categories.values()), dtype="string"),
            }
        ),
        frame,
    )


def measured(frame, categories):
    """(values, shown, population, catalogue) for one small frame."""
    cat = catalogue(frame, categories)
    values, shown, population, _ = evaluate.measure(frame, cat)
    return values, shown, population, cat


def only(rows, metric, slice_name="overall"):
    """The one result row for a metric on a slice."""
    return next(
        r for r in rows if r["slice"] == slice_name and r["metric"] == metric
    )


def value_of(report, metric, slice_name="overall"):
    return only(report["results"], metric, slice_name)["value"]


# --- ranking metrics -------------------------------------------------------


def test_reciprocal_rank_is_the_first_hit_not_an_average_over_hits():
    """The textbook definition, and the one this project reports. It matters:
    29.6% of MIND's validation impressions carry more than one positive, so a
    mean-over-all-positives MRR — which some leaderboard evaluators use —
    scores the same predictions differently. Here relevance is at ranks 2 and
    4: first-hit gives 1/2, averaging over both would give (1/2 + 1/4)/2 =
    0.375."""
    assert evaluate.reciprocal_rank(np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_ndcg_matches_a_hand_computed_example():
    """Worked by hand rather than recomputed the way the code does it.

    Relevance [0, 1, 1] at ranks 1, 2, 3 with binary gains:
      DCG@3  = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0.63093 + 0.5 = 1.13093
      ideal  = [1, 1, 0] -> 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
      nDCG@3 = 1.13093 / 1.63093 = 0.69343
    """
    assert evaluate.ndcg(np.array([0, 1, 1]), depth=3) == pytest.approx(0.69343, abs=1e-5)


def test_a_perfect_ranking_scores_one_and_a_reversed_one_scores_less():
    perfect = np.array([1, 1, 0, 0])
    assert evaluate.ndcg(perfect, depth=4) == pytest.approx(1.0)
    assert evaluate.ndcg(perfect[::-1], depth=4) < 0.7


def test_ndcg_only_counts_down_to_its_depth():
    """The cut-off has to bite, or nDCG@5 and nDCG@10 would be the same number
    on any candidate list shorter than ten."""
    late = np.array([0, 0, 0, 0, 0, 1])
    assert evaluate.ndcg(late, depth=5) == 0.0
    assert evaluate.ndcg(late, depth=10) > 0.0


def test_auc_mrr_and_ndcg_over_one_hand_checked_impression():
    """One impression, ranked [a2, a1, a3] with a2 and a3 relevant, so the
    relevance in ranked order is [1, 0, 1].

    AUC: the two positives score 3 and 1, the negative 2, so of the 2x1
      pairs one is ordered right and one wrong -> 0.5.
    MRR: first hit at rank 1 -> 1.0.
    nDCG: DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5; the ideal order
      [1, 1, 0] gives 1 + 1/log2(3) = 1.63093; 1.5 / 1.63093 = 0.91972.
    """
    frame = paired(
        [("d1", ["a1", "a2", "a3"], [0, 1, 1], ["a2", "a1", "a3"], [3.0, 2.0, 1.0])]
    )
    values, _, _, _ = measured(frame, {"a1": "x", "a2": "y", "a3": "z"})

    assert values[0, evaluate.COLUMN["auc"]] == pytest.approx(0.5)
    assert values[0, evaluate.COLUMN["mrr"]] == pytest.approx(1.0)
    assert values[0, evaluate.COLUMN["ndcg@10"]] == pytest.approx(0.91972, abs=1e-5)


def test_impressions_no_ranking_metric_is_defined_on_are_left_out_not_zeroed():
    """Ticket 9 asks for these to be handled explicitly and their counts
    reported. AUC needs both classes present; MRR and nDCG need at least one
    positive. Averaging a zero in for them would report the share of degenerate
    impressions rather than anything the retriever did — so they carry NaN,
    which the aggregation skips and counts."""
    frame = paired(
        [
            ("d1", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0]),
            ("d2", ["a1", "a2"], [0, 0], ["a1", "a2"], [2.0, 1.0]),
            ("d3", ["a1", "a2"], [1, 1], ["a1", "a2"], [2.0, 1.0]),
        ]
    )
    cat = catalogue(frame, {"a1": "x", "a2": "y"})
    values, shown, _, degenerate = evaluate.measure(frame, cat)

    assert degenerate == {"no_positive": 1, "all_positive": 1, "all_scores_tied": 0}
    point, counts = evaluate.summarise(np.arange(3), values, shown, cat.size)
    assert counts[evaluate.COLUMN["auc"]] == 1
    assert point[evaluate.COLUMN["mrr"]] == pytest.approx(1.0)


def test_an_impression_scored_flat_is_counted_as_such():
    """A cold user scores every candidate the same, so the rank metrics read
    off the order the competition supplied. The number is still reported, but
    it is a property of their file rather than of this retriever, and the count
    is what says how much of the metric that describes."""
    frame = paired([("d1", ["a1", "a2"], [1, 0], ["a1", "a2"], [0.0, 0.0])])
    cat = catalogue(frame, {"a1": "x", "a2": "y"})
    values, _, _, degenerate = evaluate.measure(frame, cat)

    assert degenerate["all_scores_tied"] == 1
    assert not np.isnan(values[0, evaluate.COLUMN["auc"]])


def test_a_ranking_that_drops_candidates_is_an_error():
    """The leaderboard scores every candidate. A retriever that returned only
    the ones it liked would score better here than it deserves, because the
    candidates it dropped can only have been misses."""
    frame = paired([("d1", ["a1", "a2"], [1, 0], ["a1"], [1.0])])
    with pytest.raises(evaluate.EvaluationError, match="permutation"):
        measured(frame, {"a1": "x", "a2": "y"})


def test_impressions_that_lost_their_ranking_are_an_error_not_a_shorter_table():
    """The ones a retriever drops are the hard ones — users with no history to
    build a query from — so letting them fall out of the join would quietly
    raise every metric in the report."""
    shown = behaviours([("d1", ["a1"], [1]), ("d2", ["a1"], [1])])
    with pytest.raises(evaluate.EvaluationError, match="came back paired"):
        evaluate.paired(
            ranked([("d1", ["a1"], [1.0])]),
            shown,
            # `paired` is handed the per-impression view, which is what
            # `evaluate.run` builds from the user-keyed table.
            ingest.per_impression(histories([("d1", 3), ("d2", 3)]), shown),
        )


# --- beyond-accuracy metrics ----------------------------------------------


def test_diversity_is_the_share_of_pairs_from_different_categories():
    """Worked by hand: four items in categories [x, x, y, z] have 6 pairs, of
    which the one x-x pair is a same-category pair -> 5/6 = 0.83333."""
    got = evaluate.intra_list_diversity(np.array([0, 0, 1, 2]))
    assert got == pytest.approx(5 / 6)

    assert evaluate.intra_list_diversity(np.array([0, 0, 0])) == 0.0
    assert evaluate.intra_list_diversity(np.array([0, 1])) == 1.0
    assert np.isnan(evaluate.intra_list_diversity(np.array([0])))


def test_novelty_rewards_the_article_fewer_people_clicked():
    """Novelty is inverse popularity, so a list of obscure articles has to
    score above a list of the ones everyone clicks. Here a1 is clicked in both
    impressions and a3 in neither."""
    frame = paired(
        [
            ("d1", ["a1", "a2", "a3"], [1, 0, 0], ["a1", "a2", "a3"], [3.0, 2.0, 1.0]),
            ("d2", ["a1", "a2", "a3"], [1, 0, 0], ["a3", "a2", "a1"], [3.0, 2.0, 1.0]),
        ]
    )
    cat = catalogue(frame, {"a1": "x", "a2": "x", "a3": "x"})

    assert cat.novelty[cat.index_of["a3"]] > cat.novelty[cat.index_of["a1"]]


def test_diversity_and_novelty_are_recorded_even_with_no_positive_label():
    """They describe the list that was shown, which exists whether or not the
    user clicked anything in it. Dropping those impressions the way the ranking
    metrics must would make the two families cover different populations."""
    frame = paired([("d1", ["a1", "a2"], [0, 0], ["a1", "a2"], [2.0, 1.0])])
    values, _, _, _ = measured(frame, {"a1": "x", "a2": "y"})

    assert np.isnan(values[0, evaluate.COLUMN["auc"]])
    assert values[0, evaluate.COLUMN["diversity"]] == pytest.approx(1.0)
    assert not np.isnan(values[0, evaluate.COLUMN["novelty"]])


def test_coverage_is_a_fraction_of_the_catalogue_not_of_the_candidates():
    """Ticket 10's distinction. Two articles are shown, out of a catalogue of
    four — coverage is 0.5, not the 1.0 it would be against the articles that
    happened to be offered as candidates."""
    frame = paired([("d1", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0])])
    cat = catalogue(frame, {"a1": "x", "a2": "y", "a3": "x", "a4": "y"})
    values, shown, _, _ = evaluate.measure(frame, cat)

    point, _ = evaluate.summarise(np.arange(1), values, shown, cat.size)
    assert cat.size == 4
    assert cat.pool == 2
    assert point[len(evaluate.MEAN_METRICS)] == pytest.approx(0.5)


def test_coverage_only_counts_the_top_of_the_list():
    """Coverage is a fraction of the catalogue the system would actually show,
    so it stops where the shown list stops. A candidate ranked past the depth
    was retrieved but not surfaced, and counting it would report the candidate
    generator's reach as the recommender's."""
    ids = [f"a{i}" for i in range(evaluate.LIST_DEPTH + 5)]
    frame = paired(
        [("d1", ids, [1] + [0] * (len(ids) - 1), ids, list(range(len(ids), 0, -1)))]
    )
    cat = catalogue(frame, dict.fromkeys(ids, "x"))
    values, shown, _, _ = evaluate.measure(frame, cat)

    point, _ = evaluate.summarise(np.arange(1), values, shown, cat.size)
    assert point[len(evaluate.MEAN_METRICS)] == pytest.approx(
        evaluate.LIST_DEPTH / len(ids)
    )


# --- slices ----------------------------------------------------------------


def test_cold_and_warm_split_the_population_at_the_click_threshold():
    frame = paired(
        [
            ("d1", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0]),
            ("d2", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0]),
            ("d3", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0]),
        ],
        n_clicks=[0, evaluate.COLD_CLICKS - 1, evaluate.COLD_CLICKS],
    )
    values, shown, population, cat = measured(frame, {"a1": "x", "a2": "y"})
    rows = evaluate.results(values, shown, population, cat, resamples=0)

    assert only(rows, "auc", "cold")["population"] == 2
    assert only(rows, "auc", "warm")["population"] == 1
    assert only(rows, "auc", "overall")["population"] == 3


def test_head_and_tail_place_an_impression_by_the_articles_its_user_clicked():
    """a1 is shown in every impression and a4 in one, so a1 is the most-shown
    fifth. An impression whose click landed on a1 is a head impression; one
    whose click landed on a4 is a tail impression."""
    frame = paired(
        [
            ("d1", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0]),
            ("d2", ["a1", "a3"], [1, 0], ["a1", "a3"], [2.0, 1.0]),
            ("d3", ["a1", "a4"], [0, 1], ["a1", "a4"], [2.0, 1.0]),
            ("d4", ["a1", "a4"], [0, 1], ["a1", "a4"], [2.0, 1.0]),
        ]
    )
    values, shown, population, cat = measured(
        frame, {"a1": "x", "a2": "y", "a3": "y", "a4": "y"}
    )
    assert cat.is_head[cat.index_of["a1"]]
    assert not cat.is_head[cat.index_of["a4"]]

    rows = evaluate.results(values, shown, population, cat, resamples=0)
    assert only(rows, "auc", "head")["population"] == 2
    assert only(rows, "auc", "tail")["population"] == 2


def test_a_slice_nobody_falls_into_reports_a_population_of_zero_not_a_number():
    """EB-NeRD's smallest history is five clicks, so its cold slice is empty.
    An empty slice has to print as empty — a metric invented for a population
    of nobody is the failure this column exists to make impossible."""
    frame = paired(
        [("d1", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0])], n_clicks=50
    )
    values, shown, population, cat = measured(frame, {"a1": "x", "a2": "y"})
    rows = evaluate.results(values, shown, population, cat, resamples=10)

    cold = only(rows, "auc", "cold")
    assert cold["population"] == 0
    assert cold["value"] is None and cold["lo"] is None and cold["hi"] is None


def test_every_metric_is_reported_on_every_slice():
    frame = paired([("d1", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0])])
    values, shown, population, cat = measured(frame, {"a1": "x", "a2": "y"})
    rows = evaluate.results(values, shown, population, cat, resamples=0)

    assert {(r["slice"], r["metric"]) for r in rows} == {
        (s, m) for s in evaluate.SLICES for m in evaluate.METRICS
    }


def test_a_new_slice_is_one_entry_in_the_registry_and_nothing_else(monkeypatch):
    """Ticket 10 asks that adding a slice touch one place. The registry is
    that place: the metrics, the aggregation and the bootstrap all iterate over
    it and none of them names a slice."""
    monkeypatch.setitem(
        evaluate.SLICES, "silent", lambda pop: pop["n_clicks"] == 0
    )
    frame = paired(
        [
            ("d1", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0]),
            ("d2", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0]),
        ],
        n_clicks=[0, 9],
    )
    values, shown, population, cat = measured(frame, {"a1": "x", "a2": "y"})
    rows = evaluate.results(values, shown, population, cat, resamples=10)

    assert {r["metric"] for r in rows if r["slice"] == "silent"} == set(
        evaluate.METRICS
    )
    assert only(rows, "auc", "silent")["population"] == 1


# --- bootstrap -------------------------------------------------------------


def test_a_population_with_no_spread_gets_an_interval_with_no_width():
    """The wrong-axis catcher ticket 10 asks for. Five identical impressions
    have identical per-impression metrics, so every resample of impressions
    gives back exactly the same mean and the interval collapses onto the point
    estimate. A bootstrap that resampled candidates within an impression
    instead — an easy thing to write and impossible to spot in a plausible
    number — would reorder the candidates and produce a visibly wide one."""
    frame = paired(
        [
            (f"d{i}", ["a1", "a2", "a3"], [0, 1, 1], ["a2", "a1", "a3"], [3.0, 2.0, 1.0])
            for i in range(5)
        ]
    )
    values, shown, population, cat = measured(frame, {"a1": "x", "a2": "y", "a3": "z"})
    rows = evaluate.results(values, shown, population, cat, resamples=200)

    assert only(rows, "auc")["value"] == pytest.approx(0.5)
    for metric in evaluate.METRICS:
        row = only(rows, metric)
        assert row["lo"] == pytest.approx(row["value"])
        assert row["hi"] == pytest.approx(row["value"])


def test_a_population_that_disagrees_gets_an_interval_that_brackets_the_mean():
    """The other half of the same check: with impressions that genuinely
    differ, the interval has to open up and contain the estimate."""
    frame = paired(
        [
            (f"d{i}", ["a1", "a2"], [1, 0], ["a1", "a2"] if i % 2 else ["a2", "a1"],
             [2.0, 1.0])
            for i in range(40)
        ]
    )
    values, shown, population, cat = measured(frame, {"a1": "x", "a2": "y"})
    rows = evaluate.results(values, shown, population, cat, resamples=200)

    auc = only(rows, "auc")
    assert auc["lo"] < auc["value"] < auc["hi"]


def test_the_resample_count_is_configurable_for_a_fast_run():
    frame = paired(
        [
            (f"d{i}", ["a1", "a2"], [1, 0], ["a1", "a2"] if i % 2 else ["a2", "a1"],
             [2.0, 1.0])
            for i in range(20)
        ]
    )
    values, shown, population, cat = measured(frame, {"a1": "x", "a2": "y"})

    assert only(evaluate.results(values, shown, population, cat, 0), "auc")["lo"] is None
    assert (
        only(evaluate.results(values, shown, population, cat, 25), "auc")["lo"]
        is not None
    )


def test_the_same_rankings_bootstrap_to_the_same_interval_twice():
    """The seed is fixed, so a change in the fourth decimal between two runs is
    a change in the data rather than in the draw."""
    frame = paired(
        [
            (f"d{i}", ["a1", "a2"], [1, 0], ["a1", "a2"] if i % 3 else ["a2", "a1"],
             [2.0, 1.0])
            for i in range(30)
        ]
    )
    values, shown, population, cat = measured(frame, {"a1": "x", "a2": "y"})
    first = evaluate.results(values, shown, population, cat, resamples=50)
    second = evaluate.results(values, shown, population, cat, resamples=50)

    assert first == second


# --- refusals --------------------------------------------------------------


def test_scoring_the_train_split_is_refused():
    """Ticket 9 asks for this to fail rather than return numbers: metrics on
    the split the retriever was built against measure memorisation."""
    with pytest.raises(evaluate.EvaluationError, match="anti-gaming"):
        evaluate.evaluate(MIND, "bm25", split="train")


def test_the_refusal_names_the_split_to_use_instead():
    """A refusal that only says no leaves the caller guessing which of the
    four partitions they were supposed to ask for."""
    with pytest.raises(evaluate.EvaluationError) as raised:
        evaluate.evaluate(MIND, "bm25", split="train")

    assert "tune" in str(raised.value)
    assert "validation" in str(raised.value)


def test_fit_is_refused_as_an_alias_for_train():
    """`fit` is the name the phase-1 ticket used for this partition. It is not
    the name the registry settled on, so asking for it must land on the same
    refusal rather than on "unknown split", which would read as a typo."""
    with pytest.raises(evaluate.EvaluationError, match="memorisation"):
        evaluate.evaluate(MIND, "bm25", split="fit")


def test_the_tune_split_is_scorable():
    """Selecting a parameter needs a number to select on. Tune is the only
    partition that may be scored without being reported."""
    assert "tune" in evaluate.SCORABLE
    assert "train" not in evaluate.SCORABLE


def test_an_unknown_split_or_retriever_is_refused():
    with pytest.raises(evaluate.EvaluationError, match="unknown split"):
        evaluate.evaluate(MIND, "bm25", split="dev")
    with pytest.raises(evaluate.EvaluationError, match="unknown retriever"):
        evaluate.evaluate(MIND, "word2vec", split="validation")


# --- output ----------------------------------------------------------------


def report(**overrides):
    frame = paired([("d1", ["a1", "a2"], [1, 0], ["a1", "a2"], [2.0, 1.0])])
    values, shown, population, cat = measured(frame, {"a1": "x", "a2": "y"})
    return {
        "dataset": "mind",
        "retriever": "bm25",
        "split": "validation",
        "impressions": 1,
        "no_positive": 0,
        "all_positive": 0,
        "all_scores_tied": 0,
        "catalogue": cat.size,
        "candidate_pool": cat.pool,
        "coverage_of": "catalogue",
        "head_cutoff": cat.head_cutoff,
        "list_depth": evaluate.LIST_DEPTH,
        "confidence": evaluate.CONFIDENCE,
        "resamples": 10,
        "results": evaluate.results(values, shown, population, cat, resamples=10),
        **overrides,
    }


def test_the_table_prints_fixed_columns_in_a_fixed_order():
    """Ticket 9 asks for a stable tabular form and ticket 10 grows what goes in
    it. Stable means the same columns in the same order whatever was scored, so
    two runs diff line by line and no metric is ever read without the slice
    population it was averaged over."""
    header, *rows = evaluate.table([report()]).splitlines()

    assert header.split() == list(evaluate.COLUMNS)
    assert len(rows) == len(evaluate.SLICES) * len(evaluate.METRICS)
    assert rows[0].split()[:7] == [
        "mind", "bm25", "validation", "overall", "1", "1", "auc",
    ]


def test_the_table_holds_its_columns_when_a_second_run_is_added():
    """Two runs of very different magnitudes still line up under the same
    header — the padding widens, the columns do not move."""
    rows = evaluate.table(
        [report(), report(retriever="ann", impressions=1234567)]
    ).splitlines()

    assert rows[0].split() == list(evaluate.COLUMNS)
    assert all(len(row.split()) == len(evaluate.COLUMNS) for row in rows)


def test_an_undefined_metric_prints_as_a_dash_rather_than_as_a_number():
    """A slice nobody falls into has no AUC. Printing 0.0000 there would be a
    number a reader could compare against another retriever's."""
    lines = evaluate.table([report()]).splitlines()
    cold = [line for line in lines if " cold " in line]

    assert cold and all(line.split()[-3:] == ["-", "-", "-"] for line in cold)


def test_the_notes_state_what_coverage_is_a_fraction_of():
    """Ticket 10 asks for the distinction to be stated in the output, not only
    honoured in the arithmetic."""
    note = evaluate.notes(report())

    assert "full catalogue" in note
    assert "appear as a candidate" in note
    assert "bootstrap 95%" in note
    # The one interval that is not a bracket around its own value.
    assert "union over the slice" in note


# --- the command over both retrievers --------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    MIND.feature_store_dir.mkdir(parents=True)
    return MIND.feature_store_dir


def _store_of(store):
    """A two-article corpus with one validation and one test impression, plus
    the artifacts every retriever loads. Small enough that every metric the
    stage-one retrievers produce is one, which is what makes it a seam test
    rather than a measurement: what is under test is that the harness reaches
    every retriever the same way, not what any of them scores.

    The train split spans four distinct moments and there is a `tune`
    impression, because the fourth retriever is a *model*: it fits on the
    earlier half of train by time and stops on tune, so a store without those
    is one it cannot be trained against at all.
    """
    from pipeline import embed

    pd.DataFrame(
        {
            "article_id": pd.Series(["a1", "a2"], dtype="string"),
            "title": pd.Series(["sharks win", "markets fall"], dtype="string"),
            "abstract": pd.Series(["", ""], dtype="string"),
            "category": pd.Series(["sports", "finance"], dtype="string"),
            "subcategory": pd.Series(["hockey", "markets"], dtype="string"),
            "lexical_text": pd.Series(["sharks win", "markets fall"], dtype="string"),
            "published_time": pd.Series([pd.NaT] * 2, dtype="datetime64[us]"),
        }
    ).to_parquet(store / "articles.parquet", index=False)
    train = behaviours(
        [(f"t{i}", ["a1", "a2"], [1, 0] if i % 2 else [0, 1]) for i in range(1, 9)],
        split="train",
        day="2019-11-13",
    )
    # Eight distinct moments, so the stacking boundary has somewhere to fall
    # and both halves hold rows for the two models to be fitted on.
    train["impression_time"] = pd.to_datetime(
        [f"2019-11-13 0{hour}:00:00" for hour in range(1, 9)]
    )
    pd.concat(
        [
            train,
            behaviours([("n1", ["a1", "a2"], [1, 0])], split="tune", day="2019-11-13"),
            behaviours([("d1", ["a1", "a2"], [1, 0])], split="validation"),
            behaviours([("d2", ["a1", "a2"], [1, 0])], split="test", day="2019-11-15"),
        ]
    ).to_parquet(store / "behaviors.parquet", index=False)
    histories(
        [(f"t{i}", 1) for i in range(1, 9)] + [("n1", 1), ("d1", 1), ("d2", 1)]
    ).to_parquet(store / "history.parquet", index=False)

    articles = pd.read_parquet(store / "articles.parquet")
    bm25_index.build(articles, MIND).save(MIND.artifacts_dir / "bm25")
    embed.Embeddings(
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        article_ids=np.array(["a1", "a2"], dtype=object),
    ).save(embed.output_dir(MIND))
    # The last two retrievers load a checkpoint rather than an index, so the
    # store is not complete until both have been fitted against it — which
    # needs the counters and the feature frames underneath them. Fitted at a
    # shrunken size: the fields that shrink are not in either variant name, so
    # the files land exactly where the registry's spec says to look.
    behaviors = pd.read_parquet(store / "behaviors.parquet")
    counters.build(behaviors).save(MIND.artifacts_dir / counters.DIRECTORY)
    for split in features.SPLITS:
        features.build(MIND, split)

    model, _ = nrms.train(MIND, dataclasses.replace(MIND.nrms, epochs=1))
    nrms.save(model, MIND.nrms, 2, nrms.checkpoint_path(MIND))
    booster, _ = rerank.train(
        MIND,
        dataclasses.replace(
            MIND.rerank, rounds=4, min_data_in_leaf=1, early_stopping=2
        ),
    )
    rerank.save(booster, MIND, MIND.rerank)


def test_the_command_scores_one_dataset_retriever_and_split(store, capsys):
    """The ticket's first line: one command, one combination, every metric on
    every slice printed and the result set on disk for tickets 11 and 12."""
    _store_of(store)

    assert evaluate.main(
        ["--dataset", "mind", "--retriever", "bm25", "--split", "test",
         "--resamples", "20"]
    ) == 0

    header, row, *_ = capsys.readouterr().out.strip().splitlines()
    assert header.split() == list(evaluate.COLUMNS)
    assert row.split()[:4] == ["mind", "bm25", "test", "overall"]

    # This store has no cold users, so its cold slice has no numbers to write.
    # Strict json has no NaN literal, and parse_constant is the only way to
    # notice one being written rather than the null a consumer can act on.
    written = json.loads(
        (MIND.artifacts_dir / evaluate.EVALUATE_DIR / "bm25-test.json").read_text(),
        parse_constant=lambda name: pytest.fail(f"wrote {name}, which is not json"),
    )
    assert written["split"] == "test"
    assert written["resamples"] == 20
    assert written["coverage_of"] == "catalogue"
    assert {(r["slice"], r["metric"]) for r in written["results"]} == {
        (s, m) for s in evaluate.SLICES for m in evaluate.METRICS
    }


def test_the_command_refuses_the_train_split_with_its_reason(capsys):
    """Argparse would have said "invalid choice", which teaches nothing. The
    refusal has to carry why, and the command has to exit non-zero so a script
    that asks for train metrics cannot mistake silence for success."""
    assert evaluate.main(["--split", "train"]) != 0

    assert "anti-gaming" in capsys.readouterr().err


def test_the_command_scores_every_combination_by_default(store, capsys):
    """Ticket 9's four result sets, from one invocation, with the datasets and
    retrievers named nowhere in the harness's own code path."""
    _store_of(store)

    assert evaluate.main(["--dataset", "mind", "--resamples", "20"]) == 0

    rows = capsys.readouterr().out.strip().splitlines()[1:]
    scored = {tuple(row.split()[:2]) for row in rows if row.startswith("mind")}
    assert scored == {("mind", retriever) for retriever in evaluate.RETRIEVERS}


def test_every_retriever_is_scored_by_the_same_code_path(store, capsys):
    """Ticket 9 asks for a result set per retriever with no retriever-specific
    path. The harness only ever calls rank_candidates, which every entry in
    RETRIEVERS exposes with the same signature — so this exercises the seam,
    not the retrievers, and adding a third entry should need no change here.
    """
    _store_of(store)

    evaluate.run(MIND)

    rows = capsys.readouterr().out.strip().splitlines()[1:]
    assert {tuple(row.split()[:3]) for row in rows if row.startswith("mind")} == {
        ("mind", retriever, "validation") for retriever in evaluate.RETRIEVERS
    }
    for retriever in evaluate.RETRIEVERS:
        path = MIND.artifacts_dir / evaluate.EVALUATE_DIR / f"{retriever}-validation.json"
        written = json.loads(path.read_text())
        assert written["dataset"] == "mind"
        assert written["retriever"] == retriever
        assert {result["metric"] for result in written["results"]} == set(
            evaluate.METRICS
        )
        assert value_of(written, "coverage") == pytest.approx(1.0)
        if retriever in evaluate.STAGE_ONE:
            # The stage-one retrievers rank this store by construction. The
            # trained one does not: what it scores is a property of a fitted
            # model, not of the harness, and asserting a number for it here
            # would be asserting something this test is not about.
            assert value_of(written, "auc") == pytest.approx(1.0)
        else:
            assert not np.isnan(value_of(written, "auc"))


def test_the_build_stage_leaves_the_test_split_alone(store):
    """The held-back split stays held back. `python build.py` runs on every
    rebuild; a test figure produced on every rebuild is a test figure someone
    ends up tuning against, which is the failure mode ticket 9 exists to
    prevent. Scoring it takes a deliberate command."""
    _store_of(store)

    evaluate.run(MIND)

    written = MIND.artifacts_dir / evaluate.EVALUATE_DIR
    assert not list(written.glob("*-test.json"))
    assert len(list(written.glob("*-validation.json"))) == len(evaluate.RETRIEVERS)
