"""What the BM25 grid concludes, and what it refuses to conclude."""

import numpy as np

from pipeline import lexical_sweep


def cell(k1, b, weight, auc, lo, hi):
    return {"k1": k1, "b": b, "title_weight": weight, "auc": auc, "lo": lo, "hi": hi}


def flat(rows):
    """rows: (labels, scores) in candidate order -- the shape `score_pairs`
    returns, every impression's candidates concatenated."""
    scores = np.array([s for row in rows for s in row[1]], dtype="float32")
    widths = np.array([len(row[1]) for row in rows], dtype="int64")
    return scores, widths, [row[0] for row in rows]


def test_the_scores_are_paired_with_the_labels_in_candidate_order():
    """`score_pairs` returns one score per candidate in the order the
    candidate lists were given, so the labels pair positionally. The ranked
    path does not: it sorts its scores best-first, and zipping those against
    input-order labels scored a whole grid at 0.4985 -- a coin flip that read
    as a finding about BM25.

    Here the third candidate is the clicked one and scores highest, so the AUC
    is a perfect 1.0. Sorted-against-unsorted would give 0.0.
    """
    scores, widths, labels = flat([([0, 0, 1], [1.0, 2.0, 9.0])])

    assert list(lexical_sweep.per_impression_auc(scores, widths, labels)) == [1.0]


def test_each_impression_reads_its_own_slice_of_the_scores():
    """The scores arrive as one flat array over every impression's candidates,
    so an off-by-one in the widths would score each impression against the
    next one's candidates while looking entirely well-formed."""
    scores, widths, labels = flat(
        [([0, 1], [1.0, 9.0]), ([1, 0, 0], [9.0, 1.0, 2.0])]
    )

    values = lexical_sweep.per_impression_auc(scores, widths, labels)

    assert list(values) == [1.0, 1.0]


def test_a_flat_ranking_scores_half_rather_than_being_dropped():
    """A lexical retriever whose query shares no term with any candidate scores
    every one of them zero. That is a real outcome and the impression stays in
    the population -- dropping it would let each cell be measured on a
    different set of impressions, so the cells would not be comparable."""
    scores, widths, labels = flat([([1, 0, 0], [0.0, 0.0, 0.0])])

    assert list(lexical_sweep.per_impression_auc(scores, widths, labels)) == [0.5]


def test_a_degenerate_impression_is_left_out():
    """AUC is undefined with no positive candidate or with every candidate
    positive; the harness counts those out rather than scoring them zero."""
    scores, widths, labels = flat(
        [([0, 0], [1.0, 0.5]), ([1, 1], [1.0, 0.5])]
    )

    assert len(lexical_sweep.per_impression_auc(scores, widths, labels)) == 0


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
