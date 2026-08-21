"""The two questions the grid cannot answer: recall, and the query side."""

import dataclasses

import pandas as pd

from pipeline import evaluate, lexical_ablation
from pipeline.datasets import DATASETS, LexicalSpec

MIND = DATASETS["mind"]


def ranked(rows):
    """rows: (ranked_ids, scores) — best first, as score_candidates returns."""
    return pd.DataFrame(
        {"ranked_ids": [r[0] for r in rows], "scores": [r[1] for r in rows]}
    )


def row(name, auc, recall=0.5, against=None, gap=(0.0, -0.01, 0.01),
        with_abstract=False, lexical=None):
    """One reported setting. `gap` is the paired difference from `against` and
    its interval, which is what the document draws its conclusions from."""
    spec = lexical or lexical_ablation.INHERITED
    built = {
        "setting": name, "k1": spec.k1, "b": spec.b,
        "title_weight": spec.title_weight, "with_abstract": with_abstract,
        "against": against,
        "auc": auc, "auc_lo": auc - 0.003, "auc_hi": auc + 0.003,
        "mrr": 0.3, "ndcg@5": 0.3, "ndcg@10": 0.35,
        "recall@50": recall / 2, "recall@100": recall, "recall@200": recall,
    }
    if against is not None:
        built["auc_gap"], built["auc_gap_lo"], built["auc_gap_hi"] = gap
        built["ndcg@10_gap"] = 0.0
    return built


# --- the ranked-order trap, which this module still walks into --------------


def test_labels_are_read_through_the_ranking_not_the_candidate_order():
    """Unlike the sweep, this module scores through `score_candidates`, because
    nDCG needs an order. That path returns its scores sorted best-first and
    aligned to `ranked_ids`, so the labels have to be looked up by id — zipping
    them against the candidate order is a random pairing, and it scored a whole
    grid at 0.4985 before anyone noticed.

    Here the clicked article `c` is ranked first, so every metric is perfect.
    """
    values = evaluate.per_impression_metrics(
        ranked([(["c", "a", "b"], [9.0, 2.0, 1.0])]), [{"a": 0, "b": 0, "c": 1}]
    )

    assert list(values["auc"]) == [1.0]
    assert list(values["mrr"]) == [1.0]
    assert list(values["ndcg@5"]) == [1.0]


def test_a_degenerate_impression_is_left_out_of_every_metric():
    """No positive candidate or every candidate positive: AUC, MRR and nDCG are
    all undefined, and the harness counts those out rather than scoring zero."""
    values = evaluate.per_impression_metrics(
        ranked([(["a", "b"], [1.0, 0.5]), (["a", "b"], [1.0, 0.5])]),
        [{"a": 0, "b": 0}, {"a": 1, "b": 1}],
    )

    assert all(len(scores) == 0 for scores in values.values())


# --- which settings get run -------------------------------------------------


def test_an_untuned_registry_is_not_compared_against_itself():
    """Before the sweep promotes anything, the registry still holds SPEC.md's
    values, and a table with the same row twice under two names invites reading
    the rounding difference between them as an effect."""
    untuned = dataclasses.replace(MIND, lexical=lexical_ablation.INHERITED)

    names = [setting.name for setting in lexical_ablation.settings(untuned)]

    assert names == ["inherited", "tuned + abstract query"]


def test_the_query_ablation_holds_the_index_parameters_fixed():
    """It is an ablation of one thing. If the abstract row also moved k1, b or
    the title weight, its difference would not be attributable to the query."""
    tuned = dataclasses.replace(
        MIND, lexical=LexicalSpec(k1=0.9, b=0.3, title_weight=3)
    )

    rows = {setting.name: setting for setting in lexical_ablation.settings(tuned)}

    assert rows["tuned + abstract query"].lexical == rows["tuned"].lexical
    assert rows["tuned + abstract query"].with_abstract
    assert not rows["tuned"].with_abstract


def test_the_chain_moves_one_thing_per_row_whatever_the_registry_chose():
    """Each row is a difference from the previous one in exactly one respect,
    so promoting the abstract query must not make the `tuned` row differ from
    `inherited` in the parameters *and* the query at once -- that would leave
    neither difference attributable."""
    chose_abstracts = dataclasses.replace(
        MIND, lexical=LexicalSpec(k1=0.9, b=0.3, query_abstract=True)
    )

    rows = {s.name: s for s in lexical_ablation.settings(chose_abstracts)}

    assert not rows["tuned"].with_abstract
    assert rows["tuned + abstract query"].with_abstract
    assert rows["tuned + abstract query"].against == "tuned"


# --- what the document is willing to claim ----------------------------------


def test_the_document_says_when_the_two_paths_disagree():
    """The reason both are reported: repeating the title lengthens the document,
    and a corpus search feels that where a fifteen-candidate re-rank need not."""
    text = lexical_ablation.document(
        [
            row("inherited", 0.5600, recall=0.40),
            row(
                "tuned", 0.5700, recall=0.35, against="inherited",
                gap=(0.0100, 0.0060, 0.0140),
            ),
        ],
        "mind",
        "tune",
    )

    assert "The two paths disagree" in text
    assert "+0.0100 auc" in text
    assert "-0.0500 recall@200" in text


def test_agreeing_paths_are_not_reported_as_a_disagreement():
    text = lexical_ablation.document(
        [
            row("inherited", 0.5600, recall=0.40),
            row(
                "tuned", 0.5700, recall=0.45, against="inherited",
                gap=(0.0100, 0.0060, 0.0140),
            ),
        ],
        "mind",
        "tune",
    )

    assert "The two paths disagree" not in text


def test_a_paired_interval_clear_of_zero_is_an_established_difference():
    """The point of pairing: a gain of 0.01 with a paired interval of [0.006,
    0.014] is real, even though the two settings' own intervals -- ±0.003 on
    either side of 0.5600 and 0.5700 -- say nothing about each other."""
    text = lexical_ablation.document(
        [
            row("inherited", 0.5600),
            row("tuned", 0.5700, against="inherited", gap=(0.0100, 0.0060, 0.0140)),
        ],
        "mind",
        "tune",
    )

    assert "**is**" in text
    assert "excludes" in text


def test_a_paired_interval_containing_zero_is_not_called_established():
    """A gain of 0.0010 whose paired interval straddles zero is a setting that
    was not shown to differ, and the document has to say so rather than report
    the argmax as a finding."""
    text = lexical_ablation.document(
        [
            row("tuned", 0.5700),
            row(
                "tuned + abstract query", 0.5710, against="tuned",
                gap=(0.0010, -0.0020, 0.0040), with_abstract=True,
            ),
        ],
        "mind",
        "tune",
    )

    assert "**is not**" in text
    assert "contains" in text
