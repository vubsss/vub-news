"""What the BM25 grid concludes, and what it refuses to conclude."""

import numpy as np
import pandas as pd

from pipeline import lexical_sweep


def cell(k1, b, weight, auc, lo, hi):
    return {"k1": k1, "b": b, "title_weight": weight, "auc": auc, "lo": lo, "hi": hi}


def ranked(rows):
    """rows: (ranked_ids, scores) -- best first, as score_candidates returns."""
    return pd.DataFrame(
        {"ranked_ids": [r[0] for r in rows], "scores": [r[1] for r in rows]}
    )


def test_labels_are_read_through_the_ranking_not_the_input_order():
    """score_candidates returns its scores sorted best-first, aligned to
    ranked_ids rather than to the candidate list it was given. Pairing the
    sorted scores with input-order labels is a random pairing, and it scored a
    whole grid at 0.4985 -- a coin flip that read as a finding about BM25.

    Here the clicked article `c` is ranked first and scored highest, so the
    AUC is a perfect 1.0. Zipped positionally against the input order it would
    be 0.0, because `c` is last in the candidate list.
    """
    frame = ranked([(["c", "a", "b"], [9.0, 2.0, 1.0])])
    label_of = [{"a": 0, "b": 0, "c": 1}]

    assert list(lexical_sweep.per_impression_auc(frame, label_of)) == [1.0]


def test_a_flat_ranking_scores_half_rather_than_being_dropped():
    """A lexical retriever whose query shares no term with any candidate scores
    every one of them zero. That is a real outcome and the impression stays in
    the population -- dropping it would let each cell be measured on a
    different set of impressions, so the cells would not be comparable."""
    values = lexical_sweep.per_impression_auc(
        ranked([(["a", "b", "c"], [0.0, 0.0, 0.0])]), [{"a": 1, "b": 0, "c": 0}]
    )

    assert list(values) == [0.5]


def test_a_degenerate_impression_is_left_out():
    """AUC is undefined with no positive candidate or with every candidate
    positive; the harness counts those out rather than scoring them zero."""
    values = lexical_sweep.per_impression_auc(
        ranked([(["a", "b"], [1.0, 0.5]), (["a", "b"], [1.0, 0.5])]),
        [{"a": 0, "b": 0}, {"a": 1, "b": 1}],
    )

    assert len(values) == 0


def test_the_document_measures_the_winner_against_the_inherited_setting():
    """The question the sweep exists to answer is not "which cell is highest"
    but "was the setting we inherited wrong", so the baseline has to appear."""
    text = lexical_sweep.document(
        [
            cell(1.5, 0.75, 1, 0.5600, 0.5570, 0.5630),
            cell(0.9, 0.30, 3, 0.5900, 0.5870, 0.5930),
        ],
        "mind",
        "tune",
    )

    assert "k1=0.9, b=0.3, title weight 3" in text
    assert "0.5600" in text
    assert "+0.0300" in text
    assert "**is**" in text


def test_an_overlapping_winner_is_not_called_established():
    text = lexical_sweep.document(
        [
            cell(1.5, 0.75, 1, 0.5600, 0.5570, 0.5630),
            cell(0.9, 0.30, 1, 0.5610, 0.5580, 0.5640),
        ],
        "mind",
        "tune",
    )

    assert "**is not**" in text


def test_a_flat_surface_says_so():
    """A grid this size always has a highest number; whether it is a finding
    is the question."""
    rows = [cell(1.0 + i / 10, 0.5, 1, 0.56, 0.55, 0.57) for i in range(6)]
    rows[0]["auc"] = 0.5605

    assert "largely noise" in lexical_sweep.document(rows, "mind", "tune")
