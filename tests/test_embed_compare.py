"""The embedding-source comparison: what its table says, and what it refuses to say."""

import pytest

from pipeline import embed_compare
from pipeline.datasets import EBNERD


def row(variant, method, auc, lo, hi, anisotropy=0.0, dim=768):
    return {
        "variant": variant, "method": method, "auc": auc, "lo": lo, "hi": hi,
        "anisotropy": anisotropy, "dim": dim,
    }


def test_every_shipped_variant_is_scored_including_the_active_one():
    """The active source is one cell of the grid, not the baseline the grid is
    measured against -- otherwise the comparison could not conclude that the
    pipeline is already using the right file."""
    names = [spec.model for spec in embed_compare.variants(EBNERD)]

    assert names[0] == EBNERD.embeddings.model
    assert set(names) == {
        "google_bert_base_multilingual_cased",
        "contrastive_vector",
        "document_vector",
        "xlm_roberta_base",
    }


def test_the_document_names_the_best_cell():
    text = embed_compare.document(
        [
            row("mbert", "none", 0.4877, 0.4850, 0.4904),
            row("mbert", "abtt:3", 0.5477, 0.5450, 0.5504),
        ],
        "ebnerd",
        "tune",
    )

    assert "mbert under abtt:3" in text
    assert "0.5477" in text


def test_an_overlapping_cell_is_not_called_worse():
    """The project's standard everywhere else: a difference is a difference
    only where the intervals are disjoint. A grid this size always has a
    highest number and the question is whether it is a finding."""
    text = embed_compare.document(
        [
            row("a", "none", 0.5500, 0.5470, 0.5530),
            row("b", "none", 0.5490, 0.5460, 0.5520),
        ],
        "ebnerd",
        "tune",
    )

    assert "Not separated from it" in text
    assert "b/none" in text


def test_a_clean_sweep_says_the_choice_is_established():
    text = embed_compare.document(
        [
            row("a", "none", 0.5500, 0.5470, 0.5530),
            row("b", "none", 0.5000, 0.4970, 0.5030),
        ],
        "ebnerd",
        "tune",
    )

    assert "disjoint interval" in text
    assert "Not separated" not in text


def test_the_interval_comes_from_the_harness_seed():
    """Two runs over the same per-impression values must print the same
    interval, or a change in the fourth decimal reads as a change in the data
    rather than in the draw."""
    import numpy as np

    values = np.random.default_rng(1).random(500)

    assert embed_compare.interval(values, 200) == embed_compare.interval(values, 200)


def test_a_promoted_variant_is_scored_once_not_twice():
    """Phase 7 promoted e5 into MIND's active slot and left it in the variant
    list, which is right -- the grid it won is part of its record. But
    `variants()` reads the active spec *and* the list, so without deduplication
    the table would carry two identical rows under one name and a reader would
    reasonably wonder which was the real one."""
    from pipeline.datasets import DATASETS

    for config in DATASETS.values():
        names = [spec.name for spec in embed_compare.variants(config)]
        assert len(names) == len(set(names)), f"{config.name} scores a source twice"

    mind = DATASETS["mind"]
    assert mind.embeddings.name in [s.name for s in mind.embedding_variants], (
        "the premise: the active source is still listed as a variant"
    )
    assert embed_compare.variants(mind)[0].name == mind.embeddings.name, (
        "and the active one still comes first"
    )
