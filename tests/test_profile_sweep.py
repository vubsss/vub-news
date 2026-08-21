"""What the window x pooling grid concludes, and what it refuses to run."""

import pandas as pd
import pytest

from pipeline import profile_sweep
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]


def cell(retriever, k, pooling, auc=0.60, mark="aaa", gap=None, dataset="mind"):
    row = {
        "dataset": dataset, "retriever": retriever, "history_k": k,
        "pooling": pooling, "weighting": "uniform", "decay": 1.0,
        "auc": auc, "mrr": 0.3, "ndcg@5": 0.3, "ndcg@10": 0.35,
        "fingerprint": mark, "n": 100, "seconds": 1.0,
    }
    if gap is not None:
        row["auc_gap"], row["auc_gap_lo"], row["auc_gap_hi"] = gap
    return row


# --- which cells exist at all -----------------------------------------------


def test_bm25_has_no_max_cells_because_it_has_no_aggregator():
    """The grid is not a rectangle. Running `max` on bm25 and scoring it as
    `mean` would put a row in the table whose label did not describe what
    happened, so those cells are skipped rather than filled."""
    pairs = profile_sweep.cells(MIND, "bm25")

    assert not any(pooling == "max" for _, pooling in pairs)
    assert {pooling for _, pooling in pairs} == {"mean", "last"}


def test_the_semantic_retriever_runs_the_whole_grid():
    pairs = profile_sweep.cells(MIND, "ann")

    assert len(pairs) == len(profile_sweep.WINDOWS) * 3


def test_the_grid_runs_past_the_window_the_old_sweep_stopped_at():
    """The old grid's winner was its own largest value, which is a grid
    reporting its boundary rather than an optimum."""
    assert max(profile_sweep.WINDOWS) == 80
    assert 20 in profile_sweep.WINDOWS, "the old winner stays in, to compare"


# --- the guard that a parameter actually arrived ----------------------------


def test_a_pooling_that_never_reached_the_retriever_is_caught():
    """The failure this guards against is a retriever that accepts the
    argument and ignores it — invisible in a metric column, because the
    numbers agree too. Three poolings at one window that rank identically is
    that failure and nothing else."""
    rows = [
        cell("ann", 10, pooling, mark="same") for pooling in ("mean", "max", "last")
    ]

    with pytest.raises(profile_sweep.SweepError, match="never reached"):
        profile_sweep.confirm_pooling_reached_the_retriever(rows)


def test_poolings_that_rank_differently_pass_the_guard():
    rows = [
        cell("ann", 10, "mean", mark="a"),
        cell("ann", 10, "max", mark="b"),
        cell("ann", 10, "last", mark="c"),
    ]

    assert "confirmed" in profile_sweep.confirm_pooling_reached_the_retriever(rows)


def test_bm25_last_must_not_move_with_a_window_it_cannot_read():
    """`last` is a one-click query, so the window is degenerate for it. A
    fingerprint that moves with the window means the window reached a query
    that claims not to depend on it."""
    rows = [
        cell("bm25", 10, "last", mark="a"),
        cell("bm25", 80, "last", mark="b"),
    ]

    with pytest.raises(profile_sweep.SweepError, match="cannot depend on one"):
        profile_sweep.confirm_pooling_reached_the_retriever(rows)


def test_bm25_last_identical_across_windows_is_what_should_happen():
    rows = [
        cell("bm25", 10, "last", mark="same"),
        cell("bm25", 80, "last", mark="same"),
    ]

    assert profile_sweep.confirm_pooling_reached_the_retriever(rows)


# --- what the document is willing to claim ----------------------------------


def test_a_grid_that_beats_nothing_says_the_current_setting_stands():
    """The baseline is the configuration in use, so the question is whether to
    move. A grid where nothing separates has to say so rather than promote its
    argmax."""
    rows = [
        cell("ann", 10, "mean", auc=0.600),
        cell("ann", 20, "mean", auc=0.601, mark="b", gap=(0.001, -0.002, 0.004)),
    ]

    text = profile_sweep.document(rows, "mind", "tune", "2 cells confirmed")

    assert "not shown to be wrong" in text


def test_a_cell_clear_of_zero_is_counted_as_beating_the_baseline():
    rows = [
        cell("ann", 10, "mean", auc=0.600),
        cell("ann", 80, "max", auc=0.620, mark="b", gap=(0.020, 0.015, 0.025)),
    ]

    text = profile_sweep.document(rows, "mind", "tune", "2 cells confirmed")

    assert "1 of 1 cells beat the baseline" in text


def test_a_winner_at_the_largest_window_says_it_is_at_the_boundary():
    """The defect the old sweep had. A grid whose winner is its own edge has
    not found an optimum, and the document must not present one."""
    rows = [
        cell("ann", 10, "mean", auc=0.600),
        cell("ann", 80, "mean", auc=0.620, mark="b", gap=(0.020, 0.015, 0.025)),
    ]

    text = profile_sweep.document(rows, "mind", "tune", "2 cells confirmed")

    assert "largest window tested" in text


def test_the_document_disclaims_the_recall_column_it_does_not_have():
    """Corpus retrieval always uses the pooled mean, so nothing here is
    evidence about pooling's effect on retrieval — and a reader who assumed
    otherwise would be drawing the phase's conclusion from the wrong path."""
    text = profile_sweep.document(
        [cell("ann", 10, "mean")], "mind", "tune", "1 cell confirmed"
    )

    assert "No recall column" in text
    assert "pooled mean" in text
