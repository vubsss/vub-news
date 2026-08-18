"""Per-competition line formats for the prediction file.

The counterpart to `sources`: that module knows how each dataset's raw files
parse, this one knows how each competition's leaderboard file is written. Both
are referenced from the registry and nothing else imports them by name, so
`predict` never learns which competition it is submitting to.
"""

from __future__ import annotations


def mind_line(impression_id: str, ranks: list[int]) -> str:
    """`24481 [4,1,3,2]` -- the impression, then a rank per candidate.

    The ranks are positional against the candidate list the competition gave:
    the k-th number is where the k-th candidate placed, 1 for the top of the
    list. Not the reordered candidate ids, which is the natural thing to write
    and would score as though every impression had been ranked at random.
    """
    return f"{impression_id} [{','.join(str(rank) for rank in ranks)}]"


def ebnerd_line(impression_id: str, ranks: list[int]) -> str:
    """`237 [4,1,3,2]` -- the same shape MIND's leaderboard reads.

    Not a guess from the resemblance: it is what the challenge's own
    `ebrec.utils._python.write_submission_file` writes, which joins the
    impression id and `"[" + ",".join(...) + "]"` with a space, over the output
    of `rank_predictions_by_score` -- 1-based ranks in candidate order, not
    scores and not the reordered ids.

    Kept as its own function rather than pointing the registry at `mind_line`,
    because the two competitions agreeing today is a coincidence of format: one
    of them changing must not silently change the file the other submits.
    """
    return f"{impression_id} [{','.join(str(rank) for rank in ranks)}]"
