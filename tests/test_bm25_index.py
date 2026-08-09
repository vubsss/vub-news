"""BM25 retrieval is tested at five seams: build_queries, the index's
build/retrieve pair, recall_at_k, check_within_corpus, and run."""

import numpy as np
import pandas as pd
import pytest

from pipeline import bm25_index, paths
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]


def articles(rows):
    """rows: (article_id, title)"""
    return pd.DataFrame(
        {
            "article_id": pd.Series([row[0] for row in rows], dtype="string"),
            "title": pd.Series([row[1] for row in rows], dtype="string"),
        }
    )


def history(rows):
    """rows: (impression_id, click_history)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
            "click_history": [list(row[1]) for row in rows],
        }
    )


def test_a_query_is_the_last_k_clicked_titles_cleaned():
    """The spec's query construction. The window is the *last* K clicks, so
    the oldest one here must not appear — a query built from the first K would
    still look plausible but would be reading the wrong end of the history."""
    catalogue = articles(
        [
            ("a1", "Sharks beat the Bears"),
            ("a2", "A late goal decided the game"),
            ("a3", "Markets fall on trade news"),
        ]
    )
    clicks = history([("dev-1", ["a3", "a1", "a2"])])

    queries, report = bm25_index.build_queries(
        clicks, catalogue, MIND, history_k=2
    )

    assert list(queries["query"]) == ["sharks beat bears late goal decided game"]
    assert report["cold"] == 0


def test_no_history_and_unusable_history_are_counted_apart():
    """The documented cold-start fallback: no clicks means no lexical evidence,
    so there is no query rather than one that quietly matches everything. An
    impression whose clicks are all dangling ids ends up equally unsearchable
    but for a different reason, and MIND has both — folding them into one
    number would hide which problem the recall figure is actually reporting."""
    catalogue = articles([("a1", "Sharks beat the Bears")])
    clicks = history([("dev-1", ["a1"]), ("dev-2", []), ("dev-3", ["gone"])])

    queries, report = bm25_index.build_queries(
        clicks, catalogue, MIND, history_k=2
    )

    assert list(queries["query"]) == ["sharks beat bears", "", ""]
    assert report["impressions"] == 3
    assert report["cold"] == 1
    assert report["empty_query"] == 2


def test_queries_from_a_filtered_history_are_positionally_aligned():
    """run builds queries from the validation slice of the history, which
    carries a gappy index. Passing that index on would make any positional or
    concat-based use of the frame downstream — ticket 8 emits the same shape —
    silently misalign queries with the impressions they belong to."""
    catalogue = articles(
        [("a1", "Sharks beat the Bears"), ("a2", "Markets fall on trade news")]
    )
    clicks = history([("dev-1", ["a1"]), ("dev-2", ["a2"]), ("dev-3", ["a1"])])
    validation = clicks[clicks["impression_id"] != "dev-1"]

    queries, _ = bm25_index.build_queries(validation, catalogue, MIND)

    assert list(queries.index) == [0, 1]
    assert list(queries["impression_id"]) == ["dev-2", "dev-3"]
    assert queries["query"][0] == "markets fall trade news"


def catalogue():
    """Three articles whose cleaned text shares no terms across them."""
    return pd.DataFrame(
        {
            "article_id": pd.Series(["a1", "a2", "a3"], dtype="string"),
            "lexical_text": pd.Series(
                [
                    "sharks beat bears late goal",
                    "markets fall trade news",
                    "bears win hockey game",
                ],
                dtype="string",
            ),
        }
    )


def queries(rows):
    """rows: (impression_id, query)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
            "query": pd.Series([row[1] for row in rows], dtype="string"),
        }
    )


def test_retrieval_ranks_the_article_that_shares_the_query_terms_first():
    """Only a2 contains 'markets' or 'trade', so it has to come back first —
    the one assertion that would catch the index being built over the wrong
    column, or the ids being mapped back from bm25s positions wrongly."""
    index = bm25_index.build(catalogue(), MIND)

    ranked = index.retrieve(queries([("dev-1", "markets trade")]), depth=2)

    assert list(ranked["impression_id"]) == ["dev-1"]
    assert ranked["ranked_ids"][0][0] == "a2"
    assert len(ranked["ranked_ids"][0]) == 2
    assert ranked["scores"][0][0] > ranked["scores"][0][1]


def test_an_empty_query_retrieves_nothing_rather_than_arbitrary_articles():
    """Handed an empty query, bm25s returns `depth` articles at score 0 —
    downstream that is indistinguishable from a retriever that searched and
    missed. A cold user gets an empty ranking instead, so the zero recall is
    attributable to having had nothing to search with."""
    index = bm25_index.build(catalogue(), MIND)

    ranked = index.retrieve(queries([("dev-1", ""), ("dev-2", "markets")]), depth=2)

    assert ranked["ranked_ids"][0] == []
    assert ranked["scores"][0] == []
    # The warm impression alongside it is unaffected.
    assert ranked["ranked_ids"][1][0] == "a2"


def test_a_reloaded_index_retrieves_exactly_what_the_built_one_did(tmp_path):
    """The index is saved so a rerun need not rebuild it, which is only safe
    if the saved one answers identically — including the article ids, which
    bm25s does not store and which are what make its positions meaningful."""
    built = bm25_index.build(catalogue(), MIND)
    built.save(tmp_path / "bm25")

    reloaded = bm25_index.load(tmp_path / "bm25")

    asked = queries([("dev-1", "markets trade"), ("dev-2", "bears goal")])
    pd.testing.assert_frame_equal(
        built.retrieve(asked, depth=3), reloaded.retrieve(asked, depth=3)
    )


def behaviors(rows):
    """rows: (impression_id, candidate_ids, labels)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
            "candidate_ids": [list(row[1]) for row in rows],
            "labels": [list(row[2]) for row in rows],
        }
    )


def ranking(rows):
    """rows: (impression_id, ranked_ids)"""
    return pd.DataFrame(
        {
            "impression_id": pd.Series([row[0] for row in rows], dtype="string"),
            "ranked_ids": [list(row[1]) for row in rows],
        }
    )


def test_recall_at_k_matches_a_hand_computed_example():
    """Worked by hand, not recomputed the way the code does it:

    dev-1 clicked a1 and a3 and its ranking is [a3, a1, a9] — 1 of 2 at
    depth 1, 2 of 2 at depth 2. dev-2 clicked a5 and its ranking is
    [a9, a5, a7] — 0 of 1 at depth 1, 1 of 1 at depth 2. Averaging the
    per-impression fractions gives 0.25 and 1.0. dev-3 has no click at all,
    so its recall is 0/0 and it is counted rather than averaged in as a zero.
    """
    truth = behaviors(
        [
            ("dev-1", ["a1", "a2", "a3"], [1, 0, 1]),
            ("dev-2", ["a5", "a7"], [1, 0]),
            ("dev-3", ["a1"], [0]),
        ]
    )
    ranked = ranking(
        [
            ("dev-1", ["a3", "a1", "a9"]),
            ("dev-2", ["a9", "a5", "a7"]),
            ("dev-3", ["a1", "a2", "a3"]),
        ]
    )

    got = bm25_index.recall_at_k(ranked, truth, depths=(1, 2))

    assert got["recall@1"] == pytest.approx(0.25)
    assert got["recall@2"] == pytest.approx(1.0)
    assert got["scored"] == 2
    assert got["no_positive"] == 1


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    MIND.feature_store_dir.mkdir(parents=True)
    return MIND.feature_store_dir


def write_store(store):
    """Four articles; two validation impressions, one of them cold; and one
    train impression, which must not be scored."""
    pd.DataFrame(
        {
            "article_id": pd.Series(["a1", "a2", "a3", "a4"], dtype="string"),
            "title": pd.Series(
                [
                    "Sharks beat the Bears",
                    "Markets fall on trade news",
                    "Bears win hockey game",
                    "A late goal decided it",
                ],
                dtype="string",
            ),
            "lexical_text": pd.Series(
                [
                    "sharks beat bears",
                    "markets fall trade news",
                    "bears win hockey game",
                    "late goal decided",
                ],
                dtype="string",
            ),
        }
    ).to_parquet(store / "articles.parquet", index=False)

    frame = behaviors(
        [
            ("dev-1", ["a2", "a3"], [1, 0]),
            ("dev-2", ["a1"], [1]),
            ("dev-3", ["a4"], [1]),
        ]
    )
    frame["split"] = pd.Series(
        ["validation", "validation", "train"], dtype="string"
    )
    frame.to_parquet(store / "behaviors.parquet", index=False)

    history(
        [("dev-1", ["a2"]), ("dev-2", []), ("dev-3", ["a1"])]
    ).to_parquet(store / "history.parquet", index=False)


def test_run_builds_the_index_and_reports_recall_on_the_validation_split(
    store, capsys
):
    write_store(store)

    bm25_index.run(MIND)

    # Saved, so a later run need not rebuild it.
    assert (MIND.artifacts_dir / "bm25" / bm25_index.ARTICLE_IDS).exists()

    printed = capsys.readouterr().out
    for depth in bm25_index.DEPTHS:
        assert f"recall@{depth}" in printed
    # The train impression is not scored: two validation impressions, of which
    # dev-2 has no history at all.
    assert "2 validation impressions" in printed
    assert "1 (50.00%) cold" in printed
    # The scale-analysis figures ticket 16 has to cite.
    assert "built in" in printed
    assert "mean query latency" in printed


def test_a_retrieved_id_from_outside_the_corpus_is_an_error():
    """The guard ticket 6 asks to be asserted in code. Retrieval maps bm25s
    positions back through article_ids, so structurally this cannot happen —
    but if the corpus and the index are ever built from different frames the
    symptom is a recall figure that is quietly wrong rather than a crash, and
    a wrong number that looks fine is the one failure this project cannot
    afford. The stray id has to be named, so the message points at the frame
    that disagreed."""
    ranked = ranking([("dev-1", ["a1", "ghost"]), ("dev-2", ["a2"])])

    with pytest.raises(bm25_index.CorpusError, match="ghost"):
        bm25_index.check_within_corpus(ranked, np.array(["a1", "a2"], dtype=object))


def test_a_second_run_reuses_the_saved_index_instead_of_rebuilding(store, capsys):
    """Ticket 6 asks for the index to be saved so it need not be rebuilt every
    run. Over MIND's 65k articles a rebuild is seconds, not minutes, so the
    only thing that would notice the saving quietly not working is the wall
    clock — and the build would still be correct, just slower every time. The
    printed line is what distinguishes the two paths."""
    write_store(store)

    bm25_index.run(MIND)
    assert "built in" in capsys.readouterr().out

    bm25_index.run(MIND)

    printed = capsys.readouterr().out
    assert "loaded from" in printed
    assert "built in" not in printed


def test_forcing_the_stage_rebuilds_the_index_it_would_otherwise_reuse(store, capsys):
    """stages.py's contract: a stage that skips itself when its outputs already
    exist must still do the work when force is set, or `build.py --force bm25`
    silently does nothing. That is the failure mode where you change the
    tokenising, re-run with --force, and get the old index's numbers back
    while believing you measured the change."""
    write_store(store)
    bm25_index.run(MIND)
    capsys.readouterr()

    bm25_index.run(MIND, force=True)

    printed = capsys.readouterr().out
    assert "built in" in printed
    assert "loaded from" not in printed
