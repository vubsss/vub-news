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
