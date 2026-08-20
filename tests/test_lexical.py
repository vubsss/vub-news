"""The BM25 parameters, and the fact that they are parameters at all.

k1 and b were module constants copied from SPEC.md and never measured. b
controls document-length normalisation and these documents are unusually short
-- a title plus an abstract, with 5% of MIND and 8% of EB-NeRD having no
abstract at all -- which is exactly the regime where a default chosen against
corpora of full documents is least likely to hold.
"""

import dataclasses

import pandas as pd
import pytest

from pipeline import bm25_index, preprocess
from pipeline.datasets import DATASETS, LexicalSpec

MIND = DATASETS["mind"]


def articles(rows, config=MIND):
    """rows: (article_id, title, abstract), with the indexed text built the way
    the preprocess stage builds it rather than by hand."""
    frame = pd.DataFrame(
        {
            "article_id": pd.Series([r[0] for r in rows], dtype="string"),
            "title": pd.Series([r[1] for r in rows], dtype="string"),
            "abstract": pd.Series([r[2] for r in rows], dtype="string"),
        }
    )
    frame["lexical_text"], _ = preprocess.build_lexical_text(frame, config)
    return frame


def tuned(**changes):
    return dataclasses.replace(
        MIND, lexical=dataclasses.replace(MIND.lexical, **changes)
    )


# --- the parameters reach the index -----------------------------------------


def test_every_dataset_declares_its_bm25_parameters():
    """A registry field, so the sweep can move it. As module constants they
    could only be changed by editing the module."""
    for config in DATASETS.values():
        assert config.lexical.k1 > 0
        assert 0.0 <= config.lexical.b <= 1.0


def test_the_index_is_built_with_the_registry_parameters():
    """The whole point of the field: if build ignored it, the sweep would
    report a grid of identical numbers and read as a flat surface."""
    corpus = articles([("a1", "sharks win", "a hockey report")])

    index = bm25_index.build(corpus, tuned(k1=2.0, b=0.3))

    assert index.bm25.k1 == 2.0
    assert index.bm25.b == 0.3


def test_changing_b_changes_the_ranking_of_documents_of_different_length():
    """b is the length-normalisation weight: at b=0 length is ignored entirely,
    at b=1 it is fully divided out. A long document padded with other words and
    a short one on the same term must therefore swap order between the two."""
    corpus = articles(
        [
            ("short", "sharks", ""),
            ("long", "sharks", " ".join(f"filler{i}" for i in range(60))),
        ]
    )
    query = pd.DataFrame(
        {
            "impression_id": pd.Series(["i1"], dtype="string"),
            "query": pd.Series(["sharks"], dtype="string"),
        }
    )

    ignores_length = bm25_index.build(corpus, tuned(b=0.0)).retrieve(query, depth=2)
    divides_it_out = bm25_index.build(corpus, tuned(b=1.0)).retrieve(query, depth=2)


    assert ignores_length["scores"][0][0] == pytest.approx(
        ignores_length["scores"][0][1]
    ), "at b=0 length is ignored, so both documents score the same on one term"
    assert divides_it_out["ranked_ids"][0][0] == "short", (
        "at b=1 the short document must win"
    )
