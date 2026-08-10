"""The harness is tested at five seams: the three metrics against rankings
worked by hand, the degenerate-impression bookkeeping, the train refusal, the
retriever-agnostic path through evaluate, and the command that drives it."""

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import ann_index, bm25_index, evaluate, paths
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


def behaviours(rows, split="validation"):
    """rows: (impression_id, candidate_ids, labels)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([r[0] for r in rows], dtype="string"),
            "candidate_ids": [list(r[1]) for r in rows],
            "labels": [list(r[2]) for r in rows],
            "split": pd.Series([split] * len(rows), dtype="string"),
        }
    )


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
    got = evaluate.metrics(
        ranked([("d1", ["a2", "a1", "a3"], [3.0, 2.0, 1.0])]),
        behaviours([("d1", ["a1", "a2", "a3"], [0, 1, 1])]),
    )

    assert got["auc"] == pytest.approx(0.5)
    assert got["mrr"] == pytest.approx(1.0)
    assert got["ndcg@10"] == pytest.approx(0.91972, abs=1e-5)
    assert got["scored"] == 1


def test_impressions_no_ranking_metric_is_defined_on_are_counted_not_averaged():
    """Ticket 9 asks for these to be handled explicitly and their counts
    reported. AUC needs both classes present; MRR and nDCG need at least one
    positive. Averaging a zero in for them would report the share of degenerate
    impressions rather than anything the retriever did."""
    got = evaluate.metrics(
        ranked(
            [
                ("d1", ["a1", "a2"], [2.0, 1.0]),
                ("d2", ["a1", "a2"], [2.0, 1.0]),
                ("d3", ["a1", "a2"], [2.0, 1.0]),
            ]
        ),
        behaviours(
            [
                ("d1", ["a1", "a2"], [1, 0]),
                ("d2", ["a1", "a2"], [0, 0]),
                ("d3", ["a1", "a2"], [1, 1]),
            ]
        ),
    )

    assert got["scored"] == 1
    assert got["no_positive"] == 1
    assert got["all_positive"] == 1
    assert got["mrr"] == pytest.approx(1.0)


def test_an_impression_scored_flat_is_counted_as_such():
    """A cold user scores every candidate the same, so the rank metrics read
    off the order the competition supplied. The number is still reported, but
    it is a property of their file rather than of this retriever, and the count
    is what says how much of the metric that describes."""
    got = evaluate.metrics(
        ranked([("d1", ["a1", "a2"], [0.0, 0.0])]),
        behaviours([("d1", ["a1", "a2"], [1, 0])]),
    )

    assert got["all_scores_tied"] == 1
    assert got["scored"] == 1


def test_a_ranking_that_drops_candidates_is_an_error():
    """The leaderboard scores every candidate. A retriever that returned only
    the ones it liked would score better here than it deserves, because the
    candidates it dropped can only have been misses."""
    with pytest.raises(evaluate.EvaluationError, match="permutation"):
        evaluate.metrics(
            ranked([("d1", ["a1"], [1.0])]),
            behaviours([("d1", ["a1", "a2"], [1, 0])]),
        )


def test_scoring_the_train_split_is_refused():
    """Ticket 9 asks for this to fail rather than return numbers: metrics on
    the split the retriever was built against measure memorisation."""
    with pytest.raises(evaluate.EvaluationError, match="anti-gaming"):
        evaluate.evaluate(MIND, "bm25", split="train")


def test_an_unknown_split_or_retriever_is_refused():
    with pytest.raises(evaluate.EvaluationError, match="unknown split"):
        evaluate.evaluate(MIND, "bm25", split="dev")
    with pytest.raises(evaluate.EvaluationError, match="unknown retriever"):
        evaluate.evaluate(MIND, "word2vec", split="validation")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    MIND.feature_store_dir.mkdir(parents=True)
    return MIND.feature_store_dir


def _store_of(store):
    """A two-article corpus with one validation and one test impression, plus
    the artifacts both retrievers load. Small enough that every metric it
    produces is one, which is what makes it a seam test rather than a
    measurement: what is under test is that the harness reaches both
    retrievers the same way, not what either of them scores."""
    from pipeline import embed

    pd.DataFrame(
        {
            "article_id": pd.Series(["a1", "a2"], dtype="string"),
            "title": pd.Series(["sharks win", "markets fall"], dtype="string"),
            "lexical_text": pd.Series(["sharks win", "markets fall"], dtype="string"),
        }
    ).to_parquet(store / "articles.parquet", index=False)
    pd.concat(
        [
            behaviours([("d1", ["a1", "a2"], [1, 0])], split="validation"),
            behaviours([("d2", ["a1", "a2"], [1, 0])], split="test"),
        ]
    ).to_parquet(store / "behaviors.parquet", index=False)
    pd.DataFrame(
        {
            "impression_id": pd.Series(["d1", "d2"], dtype="string"),
            "click_history": [["a1"], ["a1"]],
        }
    ).to_parquet(store / "history.parquet", index=False)

    articles = pd.read_parquet(store / "articles.parquet")
    bm25_index.build(articles, MIND).save(MIND.artifacts_dir / "bm25")
    embed.Embeddings(
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        article_ids=np.array(["a1", "a2"], dtype=object),
    ).save(embed.output_dir(MIND))


def test_the_table_prints_fixed_columns_in_a_fixed_order():
    """Ticket 9 asks for a stable tabular form. Stable means the same columns
    in the same order whatever was scored, so two runs diff line by line and
    the metrics are never read without the counts they were averaged over."""
    header, *rows = evaluate.table(
        [
            {
                "dataset": "mind",
                "retriever": "bm25",
                "split": "validation",
                "impressions": 3,
                "scored": 2,
                "no_positive": 1,
                "all_positive": 0,
                "all_scores_tied": 0,
                "auc": 0.5,
                "mrr": 1.0,
                "ndcg@5": 0.25,
                "ndcg@10": 0.125,
            }
        ]
    ).splitlines()

    assert header.split() == list(evaluate.COLUMNS)
    assert rows[0].split() == [
        "mind", "bm25", "validation", "3", "2", "1", "0", "0",
        "0.5000", "1.0000", "0.2500", "0.1250",
    ]


def test_the_table_holds_its_columns_when_a_second_run_is_added():
    """Two rows of very different magnitudes still line up under the same
    header — the padding widens, the columns do not move."""
    reports = [
        {
            "dataset": "ebnerd", "retriever": "ann", "split": "test",
            "impressions": 1, "scored": 1, "no_positive": 0,
            "all_positive": 0, "all_scores_tied": 0,
            "auc": 0.5, "mrr": 0.5, "ndcg@5": 0.5, "ndcg@10": 0.5,
        },
        {
            "dataset": "mind", "retriever": "bm25", "split": "test",
            "impressions": 1234567, "scored": 1234567, "no_positive": 0,
            "all_positive": 0, "all_scores_tied": 0,
            "auc": 0.5, "mrr": 0.5, "ndcg@5": 0.5, "ndcg@10": 0.5,
        },
    ]
    header, *rows = evaluate.table(reports).splitlines()

    assert header.split() == list(evaluate.COLUMNS)
    assert all(len(row.split()) == len(evaluate.COLUMNS) for row in rows)


def test_the_command_scores_one_dataset_retriever_and_split(store, capsys):
    """The ticket's first line: one command, one combination, all four metrics
    printed and the result set on disk for tickets 11 and 12."""
    _store_of(store)

    assert evaluate.main(
        ["--dataset", "mind", "--retriever", "bm25", "--split", "test"]
    ) == 0

    header, row, *_ = capsys.readouterr().out.strip().splitlines()
    assert header.split() == list(evaluate.COLUMNS)
    assert row.split()[:3] == ["mind", "bm25", "test"]

    written = json.loads(
        (MIND.artifacts_dir / evaluate.EVALUATE_DIR / "bm25-test.json").read_text()
    )
    assert written["split"] == "test"
    assert all(metric in written for metric in evaluate.METRICS)


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

    assert evaluate.main(["--dataset", "mind"]) == 0

    rows = capsys.readouterr().out.strip().splitlines()[1:]
    scored = {tuple(row.split()[:2]) for row in rows if row.startswith("mind")}
    assert scored == {("mind", retriever) for retriever in evaluate.RETRIEVERS}


def test_both_retrievers_are_scored_by_the_same_code_path(store, capsys):
    """Ticket 9 asks for four result sets with no retriever-specific path. The
    harness only ever calls rank_candidates, which both modules expose with the
    same signature — so this exercises the seam, not the retrievers."""
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
        assert written["auc"] == pytest.approx(1.0)


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
