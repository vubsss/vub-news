"""Semantic retrieval is tested at four seams: build_user_vectors, the index's
build/retrieve pair, score_candidates, and run."""

import numpy as np
import pandas as pd
import pytest

from pipeline import ann_index, embed, paths, retrieval
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]
DIM = 4


def vectors(*rows):
    """Unit-length rows of the width the index is built at."""
    matrix = np.zeros((len(rows), DIM), dtype="float32")
    for i, row in enumerate(rows):
        matrix[i, : len(row)] = row
    return embed.normalise(matrix)


def corpus(ids, *rows):
    return embed.Embeddings(
        vectors=vectors(*rows), article_ids=np.array(ids, dtype=object)
    )


def history(rows):
    """rows: (impression_id, click_history)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
            "click_history": [list(row[1]) for row in rows],
        }
    )


def test_a_user_vector_is_the_mean_of_the_rows_its_clicks_point_at():
    """The exact expected vector, from geometry rather than from rerunning the
    code: clicking (1,0) and (0,1) puts the user at their mean (0.5,0.5), which
    scaled to unit length is (0.7071, 0.7071). A user vector built from the
    wrong rows — off by one, or keyed by position instead of id — would still
    be unit length and would still rank things, just not this."""
    articles = corpus(["a1", "a2", "a3"], (1.0, 0.0), (0.0, 1.0), (0.0, 0.0, 1.0))

    queries, _ = ann_index.build_user_vectors(history([("d1", ["a1", "a2"])]), articles)

    assert queries.vectors[0].tolist() == pytest.approx(
        [0.70710678, 0.70710678, 0.0, 0.0]
    )


def test_only_the_last_k_clicks_count():
    """The window is the *last* K clicks, the same K the lexical query uses.
    Reading the wrong end of the history would still produce a plausible
    vector, and the two retrievers would then be answering different
    questions."""
    articles = corpus(["old", "new"], (1.0, 0.0), (0.0, 1.0))

    queries, _ = ann_index.build_user_vectors(
        history([("d1", ["old", "new"])]), articles, history_k=1
    )

    assert queries.vectors[0].tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_a_click_the_catalogue_has_no_vector_for_is_skipped_not_averaged_in():
    """Averaging in a missing article's zero row would pull the user toward
    the origin in proportion to how much of their history we lack — they would
    read as someone with different interests rather than someone we know less
    about. Here the surviving click alone must decide the direction."""
    articles = corpus(["a1"], (1.0, 0.0))

    queries, report = ann_index.build_user_vectors(
        history([("d1", ["a1", "not-in-catalogue"])]), articles
    )

    assert queries.vectors[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert report["cold"] == 0
    assert report["empty_query"] == 0


def test_cold_and_unresolvable_histories_are_counted_apart():
    """Ticket 8 asks for the cold-start frequency to be reported, because it
    sets how much the cold-user slice in ticket 10 can mean. A history of ids
    the catalogue does not hold is equally unsearchable but is not the same
    thing, so it is counted separately."""
    articles = corpus(["a1"], (1.0, 0.0))

    _, report = ann_index.build_user_vectors(
        history([("d1", []), ("d2", ["ghost"]), ("d3", ["a1"])]), articles
    )

    assert report == {"impressions": 3, "cold": 1, "empty_query": 2}


def test_retrieval_ranks_the_nearest_article_first():
    articles = corpus(
        ["far", "near", "middle"], (0.0, 1.0), (1.0, 0.0), (0.7, 0.7)
    )
    queries, _ = ann_index.build_user_vectors(history([("d1", ["near"])]), articles)

    ranked = ann_index.build(articles).retrieve(queries, depth=2)

    assert ranked["ranked_ids"][0] == ["near", "middle"]
    assert ranked["scores"][0][0] == pytest.approx(1.0)


def test_a_cold_user_retrieves_nothing_rather_than_arbitrary_articles():
    """A zero vector scores 0 against every article, so faiss would hand back
    `depth` articles in whatever order the index happens to hold them — and
    some of those would be clicked ones by luck, lifting semantic recall above
    lexical recall for exactly the users neither retriever knows anything
    about. bm25_index drops empty queries for the same reason; the two recall
    figures are only comparable if both drop the same impressions."""
    articles = corpus(["a1", "a2"], (1.0, 0.0), (0.0, 1.0))
    queries, _ = ann_index.build_user_vectors(history([("d1", [])]), articles)

    ranked = ann_index.build(articles).retrieve(queries, depth=2)

    assert ranked["ranked_ids"][0] == []
    assert ranked["scores"][0] == []


def test_scoring_a_candidate_list_returns_every_candidate_ranked():
    """What tickets 13 and 14 submit. The competition supplies the candidates
    and expects all of them back in an order, so unlike full-corpus retrieval
    this drops nothing — including a candidate the corpus has no vector for,
    which scores 0 rather than vanishing from the submission."""
    articles = corpus(["a1", "a2"], (1.0, 0.0), (0.0, 1.0))
    queries, _ = ann_index.build_user_vectors(history([("d1", ["a2"])]), articles)

    ranked = ann_index.build(articles).score_candidates(
        queries, [["a1", "ghost", "a2"]]
    )

    assert ranked["ranked_ids"][0] == ["a2", "a1", "ghost"]
    assert ranked["scores"][0] == pytest.approx([1.0, 0.0, 0.0])


def test_a_cold_user_scores_candidates_in_the_order_they_arrived():
    """Every candidate ties at 0, and a stable sort then hands back the
    competition's own order. Any other order would be inventing a preference
    out of the index's internal layout."""
    articles = corpus(["a1", "a2"], (1.0, 0.0), (0.0, 1.0))
    queries, _ = ann_index.build_user_vectors(history([("d1", [])]), articles)

    ranked = ann_index.build(articles).score_candidates(queries, [["a2", "a1"]])

    assert ranked["ranked_ids"][0] == ["a2", "a1"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    MIND.feature_store_dir.mkdir(parents=True)
    return MIND.feature_store_dir


def test_run_reports_recall_on_the_validation_split(store, capsys):
    """The end-to-end shape ticket 8 asks to be printed: recall at each depth,
    the cold-start frequency, and the build and latency figures ticket 16 has
    to cite.

    d1's user vector is a1, whose nearest neighbour is a3, which d1 clicked —
    recall 1.0. d2 is cold, retrieves nothing and recalls 0, which is counted
    in the mean rather than skipped: a cold user is a real miss, and dropping
    them would report the recall of the users the retriever happens to serve.
    d3 is a train impression and must not be scored at all. So the mean over
    the two validation impressions is 0.5, and bm25_index counts cold users
    the same way."""
    articles = corpus(["a1", "a2", "a3"], (1.0, 0.0), (0.0, 1.0), (0.9, 0.1))
    embed.Embeddings(vectors=articles.vectors, article_ids=articles.article_ids).save(
        embed.output_dir(MIND)
    )
    pd.DataFrame(
        {
            "impression_id": pd.Series(["d1", "d2", "d3"], dtype="string"),
            "candidate_ids": [["a3", "a2"], ["a1"], ["a1"]],
            "labels": [[1, 0], [1], [1]],
            "split": pd.Series(
                ["validation", "validation", "train"], dtype="string"
            ),
        }
    ).to_parquet(store / "behaviors.parquet", index=False)
    history(
        [("d1", ["a1"]), ("d2", []), ("d3", ["a1"])]
    ).to_parquet(store / "history.parquet", index=False)

    ann_index.run(MIND)

    printed = capsys.readouterr().out
    assert "recall@50   0.5000" in printed
    assert "over 2 impressions with a click" in printed
    assert "1 (50.00%) cold" in printed
    assert "built in" in printed and "mean query latency" in printed
