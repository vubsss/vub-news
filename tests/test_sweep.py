"""The sweep is tested where it could quietly lie: the window that has to
reach both retrievers or the whole axis is a confound, the resume that must not
re-run or lose a cell, the best-window verdict that has to say "not
established" rather than name the highest number, and the path that carries a
swept cell into the comparison without anyone retyping it."""

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import bm25_index, compare, evaluate, paths, retrieval, sweep
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]


def cell(dataset="mind", retriever="bm25", history_k=10, auc=(0.6, 0.55, 0.65),
         recall=None, seconds=1.0, resamples=1000):
    """A stored cell with the overall AUC a test is about; everything else is
    a placeholder, so a test states only the numbers it cares about."""
    value, lo, hi = auc
    results = [
        {
            "slice": slice_name,
            "metric": metric,
            "population": 10,
            "n": 10,
            "value": value if (slice_name, metric) == ("overall", sweep.PRIMARY)
            else 0.5,
            "lo": lo if (slice_name, metric) == ("overall", sweep.PRIMARY) else 0.4,
            "hi": hi if (slice_name, metric) == ("overall", sweep.PRIMARY) else 0.6,
        }
        for slice_name in evaluate.SLICES
        for metric in evaluate.METRICS
    ]
    return {
        "dataset": dataset,
        "retriever": retriever,
        "history_k": history_k,
        "split": "validation",
        "seconds": seconds,
        "recall": [
            {"depth": depth, "value": (recall or {}).get(depth, 0.3),
             "scored": 10, "no_positive": 0}
            for depth in retrieval.DEPTHS
        ],
        "report": {
            "dataset": dataset,
            "retriever": retriever,
            "history_k": history_k,
            "split": "validation",
            "resamples": resamples,
            "results": results,
        },
    }


# --- the best-window verdict ------------------------------------------------


def test_the_best_window_is_named_where_its_interval_clears_the_others():
    """A window is better than another only where nothing overlaps."""
    rows = [
        cell(history_k=5, auc=(0.50, 0.49, 0.51)),
        cell(history_k=10, auc=(0.55, 0.54, 0.56)),
        cell(history_k=20, auc=(0.60, 0.59, 0.61)),
    ]

    reading = sweep.best_window(rows, "mind", "bm25")

    assert "k=20" in reading
    assert "disjoint" in reading
    assert "not established" not in reading


def test_a_highest_number_inside_overlapping_intervals_is_not_a_finding():
    """The trap the whole ticket exists to avoid: three windows, one of them
    highest by a hair, and an interval that says nothing separates them. The
    reading must refuse to call it the best window."""
    rows = [
        cell(history_k=5, auc=(0.550, 0.52, 0.58)),
        cell(history_k=10, auc=(0.552, 0.52, 0.58)),
        cell(history_k=20, auc=(0.554, 0.52, 0.58)),
    ]

    reading = sweep.best_window(rows, "mind", "bm25")

    assert "not established" in reading
    # And it should point at the cheaper of two equally-evidenced windows.
    assert "k=5" in reading


def test_a_window_above_one_and_tied_with_another_says_both():
    rows = [
        cell(history_k=5, auc=(0.40, 0.39, 0.41)),
        cell(history_k=10, auc=(0.59, 0.58, 0.61)),
        cell(history_k=20, auc=(0.60, 0.59, 0.62)),
    ]

    reading = sweep.best_window(rows, "mind", "bm25")

    assert "separated from k=5" in reading
    assert "overlapping k=10" in reading


def test_one_window_alone_is_not_a_finding_about_windows():
    """A grid restricted to a single window has a highest number by having
    only one. Reporting it as the best window would read as a swept result."""
    reading = sweep.best_window([cell(history_k=5)], "mind", "bm25")

    assert "only window swept" in reading
    assert "disjoint" not in reading


def test_a_window_with_no_interval_is_never_separated():
    """A cell run with no resamples behind it has no interval, and the honest
    reading of no interval is that nothing was established."""
    rows = [
        cell(history_k=5, auc=(0.50, None, None)),
        cell(history_k=20, auc=(0.60, None, None)),
    ]

    assert "not established" in sweep.best_window(rows, "mind", "bm25")


# --- the window reaching both retrievers ------------------------------------


def test_a_cell_whose_retriever_used_another_window_is_refused():
    """The cell asked for 5 and the report says 10: the axis did not reach the
    retriever, so every number in the grid is about a window nobody chose."""
    row = cell(history_k=5)
    row["report"]["history_k"] = 10

    with pytest.raises(sweep.SweepError, match="asked for a window of 5"):
        sweep.confirm_same_window([row])


def test_the_confirmation_names_the_windows_both_retrievers_shared():
    rows = [
        cell(retriever="bm25", history_k=5),
        cell(retriever="ann", history_k=5),
        cell(retriever="bm25", history_k=20),
        cell(retriever="ann", history_k=20),
    ]

    confirmed = sweep.confirm_same_window(rows)

    assert "same history window in every cell" in confirmed
    assert "5, 20" in confirmed


# --- what the document says -------------------------------------------------


def test_the_document_puts_depth_only_where_depth_can_matter():
    """Depth moves recall and cannot move a metric computed over a candidate
    list that was never truncated. The document has to place it accordingly,
    or a reader averages three identical AUCs and reports a tighter number
    than the data supports."""
    rows = [cell(history_k=5), cell(history_k=20)]

    text = sweep.document(rows, "validation")

    assert "truncate" in text
    for depth in retrieval.DEPTHS:
        assert f"recall@{depth}" in text
    # The ranking table is by window, and carries no depth column.
    ranking = text.split("## Ranking metrics by window")[1].split("## Recall")[0]
    assert "depth" not in ranking


def test_the_document_says_recall_carries_no_interval():
    """Reported without one, so it has to be said rather than left for a
    reader to notice a missing column."""
    text = sweep.document([cell()], "validation")

    assert "no interval" in text
    assert sweep.PRIMARY in text


# --- the command ------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    MIND.feature_store_dir.mkdir(parents=True)
    _store_of(MIND.feature_store_dir)
    return MIND.feature_store_dir


def _store_of(store):
    """A three-article corpus whose histories are built so that the last five
    clicks and the last twenty are about different articles.

    That is the point of this fixture rather than an incidental detail: a
    history of one article repeated says nothing about whether the window
    reached the retriever, because every window slices it to the same thing.
    Here d1's last five clicks are all a1 and its full history is mostly a2,
    so a retriever that ignored the window would score the two candidates the
    other way round.
    """
    from pipeline import embed

    pd.DataFrame(
        {
            "article_id": pd.Series(["a1", "a2", "a3"], dtype="string"),
            "title": pd.Series(
                ["sharks win", "markets fall", "election result"], dtype="string"
            ),
            "category": pd.Series(["sports", "finance", "politics"], dtype="string"),
            "subcategory": pd.Series(["nfl", "markets", "vote"], dtype="string"),
            "published_time": pd.to_datetime(
                ["2019-11-08", "2019-11-08", "2019-11-09"]
            ),
            "lexical_text": pd.Series(
                ["sharks win", "markets fall", "election result"], dtype="string"
            ),
        }
    ).to_parquet(store / "articles.parquet", index=False)

    # The two validation impressions the window tests are about, preceded by
    # train impressions that exist so the fusion retriever has a split to fit
    # on: it is registered like the other two, so the grid runs it, and a
    # booster needs both classes present before it will fit at all.
    train = [f"t{i}" for i in range(8)]
    pd.DataFrame(
        {
            "impression_id": pd.Series([*train, "d1", "d2"], dtype="string"),
            "user_id": pd.Series([*train, "d1", "d2"], dtype="string"),
            "impression_time": pd.to_datetime(
                [f"2019-11-10 0{i}:00:00" for i in range(len(train))]
                + ["2019-11-11 09:00:00", "2019-11-11 10:00:00"]
            ),
            "candidate_ids": [["a1", "a2"]] * len(train) + [["a1", "a2"], ["a1", "a2"]],
            "labels": [[i % 2, 1 - i % 2] for i in range(len(train))]
            + [[1, 0], [0, 1]],
            "split": pd.Series(
                ["train"] * len(train) + ["validation", "validation"], dtype="string"
            ),
        }
    ).to_parquet(store / "behaviors.parquet", index=False)

    clicks = {"d1": ["a2"] * 6 + ["a1"] * 5, "d2": ["a1"] * 6 + ["a2"] * 5}
    clicks.update({name: ["a1", "a2", "a3"] for name in train})
    pd.DataFrame(
        {
            "impression_id": pd.Series(list(clicks), dtype="string"),
            "click_history": [clicks[i] for i in clicks],
            "n_clicks": [len(clicks[i]) for i in clicks],
        }
    ).to_parquet(store / "history.parquet", index=False)

    articles = pd.read_parquet(store / "articles.parquet")
    bm25_index.build(articles, MIND).save(MIND.artifacts_dir / "bm25")
    embed.Embeddings(
        vectors=np.eye(3, dtype="float32"),
        article_ids=np.array(["a1", "a2", "a3"], dtype=object),
    ).save(embed.output_dir(MIND))


def stored(split="validation"):
    return sweep.load(split)


def test_the_history_window_reaches_both_retrievers(store):
    """The guard the whole axis rests on. A window that never arrived would
    leave both retrievers on their default, every cell would hold the same
    numbers, and the sweep would report a window effect of exactly zero while
    looking entirely well-formed.

    d1's last five clicks are a1 and its longer history is mostly a2, so the
    two windows must not score its candidates the same way — for each
    retriever independently, since they read the window through different
    code."""
    behaviors = pd.read_parquet(store / "behaviors.parquet")
    history = pd.read_parquet(store / "history.parquet")

    for retriever in (evaluate.RETRIEVERS["bm25"], evaluate.RETRIEVERS["ann"]):
        near = retriever.rank_candidates(MIND, behaviors, history, 5)
        far = retriever.rank_candidates(MIND, behaviors, history, 11)
        assert near["scores"].tolist() != far["scores"].tolist(), retriever.__name__


def test_the_command_runs_the_grid_and_writes_one_line_per_cell(store, capsys):
    """The ticket's first line: one command, the whole grid, one file."""
    assert sweep.main(["--dataset", "mind", "--resamples", "20"]) == 0

    rows = stored()
    assert {(r["retriever"], r["history_k"]) for r in rows} == {
        (retriever, window)
        for retriever in evaluate.RETRIEVERS
        for window in sweep.WINDOWS
    }
    assert "min of grid time" in capsys.readouterr().out


def test_every_line_carries_the_configuration_its_numbers_came_from(store):
    """A number in the file that needs a filename or a line position read to
    know what produced it is a number that gets quoted against the wrong
    configuration."""
    assert sweep.main(
        ["--dataset", "mind", "--retriever", "bm25", "--window", "5",
         "--resamples", "20"]
    ) == 0

    line = json.loads(
        sweep.results_path("validation").read_text().splitlines()[0],
        parse_constant=lambda name: pytest.fail(f"wrote {name}, which is not json"),
    )
    assert (line["dataset"], line["retriever"], line["history_k"], line["split"]) == (
        "mind", "bm25", 5, "validation",
    )
    assert {entry["depth"] for entry in line["recall"]} == set(retrieval.DEPTHS)
    assert line["report"]["history_k"] == 5


def test_an_interrupted_sweep_resumes_instead_of_starting_over(store, capsys):
    """The cells already measured must survive, and must not be measured
    again: a grid that re-ran everything on resume would cost the same as
    --restart and quietly discard hours."""
    assert sweep.main(
        ["--dataset", "mind", "--retriever", "bm25", "--window", "5",
         "--resamples", "20"]
    ) == 0
    first = sweep.results_path("validation").read_text()
    capsys.readouterr()

    assert sweep.main(
        ["--dataset", "mind", "--retriever", "bm25", "--window", "5",
         "--window", "20", "--resamples", "20"]
    ) == 0

    after = sweep.results_path("validation").read_text()
    assert after.startswith(first), "the completed cell was rewritten"
    assert len(stored()) == 2
    assert "resuming: 1 of 2 cells already done" in capsys.readouterr().out


def test_restart_discards_the_stored_cells(store):
    assert sweep.main(
        ["--dataset", "mind", "--retriever", "bm25", "--window", "5",
         "--resamples", "20"]
    ) == 0

    assert sweep.main(
        ["--dataset", "mind", "--retriever", "bm25", "--window", "20",
         "--resamples", "20", "--restart"]
    ) == 0

    assert [row["history_k"] for row in stored()] == [20]


def test_the_quick_flag_runs_a_reduced_grid(store):
    """Documented and cheap, so a development run is a real option rather than
    a reason to lower the resample count on the run that gets quoted."""
    assert sweep.main(["--dataset", "mind", "--retriever", "bm25", "--quick"]) == 0

    rows = stored()
    assert {row["history_k"] for row in rows} == set(sweep.QUICK_WINDOWS)
    assert {row["report"]["resamples"] for row in rows} == {sweep.QUICK_RESAMPLES}


def test_the_runtime_is_reported_per_cell_and_in_total(store, capsys):
    assert sweep.main(
        ["--dataset", "mind", "--retriever", "bm25", "--window", "5",
         "--resamples", "20"]
    ) == 0

    out = capsys.readouterr().out
    assert "1/1" in out and "min of grid time" in out
    assert stored()[0]["seconds"] >= 0


# --- feeding the comparison -------------------------------------------------


def test_the_comparison_reads_a_swept_cell_without_transcription(store, capsys):
    """Ticket 11's document, rebuilt from a cell of the grid by naming the
    window — no number copied out of the sweep file by hand."""
    assert sweep.main(
        ["--dataset", "mind", "--window", "5", "--resamples", "20"]
    ) == 0
    capsys.readouterr()

    assert compare.main(["--dataset", "mind", "--window", "5"]) == 0

    text = (paths.ARTIFACTS_DIR / "comparison-validation-k5.md").read_text()
    assert "history window 5" in text
    swept = next(r for r in stored() if r["retriever"] == "bm25")
    auc = next(
        r["value"] for r in swept["report"]["results"]
        if r["slice"] == "overall" and r["metric"] == "auc"
    )
    assert f"{auc:.4f}" in text


def test_the_default_comparison_is_not_overwritten_by_a_swept_one(store, capsys):
    """Two documents holding different numbers must not share a filename."""
    assert sweep.main(
        ["--dataset", "mind", "--window", "5", "--resamples", "20"]
    ) == 0
    capsys.readouterr()

    assert compare.main(["--dataset", "mind", "--window", "5"]) == 0

    assert not (paths.ARTIFACTS_DIR / "comparison-validation.md").exists()


def test_a_window_the_grid_never_ran_names_the_command_that_runs_it(store, capsys):
    assert sweep.main(
        ["--dataset", "mind", "--window", "5", "--resamples", "20"]
    ) == 0
    capsys.readouterr()

    assert compare.main(["--dataset", "mind", "--window", "20"]) != 0

    assert "python -m pipeline.sweep" in capsys.readouterr().err


def test_two_retrievers_swept_at_different_windows_are_never_compared(store, capsys):
    """The confound the axis exists to prevent, at the one place it could still
    get through: comparing a cell of one window against a cell of another."""
    assert sweep.main(
        ["--dataset", "mind", "--retriever", "bm25", "--window", "5",
         "--resamples", "20"]
    ) == 0
    assert sweep.main(
        ["--dataset", "mind", "--retriever", "ann", "--window", "20",
         "--resamples", "20"]
    ) == 0
    capsys.readouterr()

    # Neither window has both retrievers, so neither can be compared.
    assert compare.main(["--dataset", "mind", "--window", "5"]) != 0
    assert compare.main(["--dataset", "mind", "--window", "20"]) != 0
