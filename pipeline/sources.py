"""Per-dataset adapters: raw source frame -> unified schema frame.

This is where MIND's and EB-NeRD's shapes genuinely differ, so it is the one
module that knows about either. Everything downstream, ingest included, works
off the unified frames these produce. A third dataset means a new adapter and a
new registry entry, and no change to any stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def mind_behaviors(raw: pd.DataFrame) -> pd.DataFrame:
    # "N3-1 N4-0" — candidate article and whether it was clicked, in one field.
    pairs = raw["impressions"].str.split()
    return pd.DataFrame(
        {
            "impression_id": raw["impression_id"].astype(str),
            "user_id": raw["user_id"].astype(str),
            "impression_time": pd.to_datetime(
                raw["time"], format=MIND_TIME_FORMAT
            ),
            "candidate_ids": pairs.map(
                lambda items: [item.rsplit("-", 1)[0] for item in items]
            ),
            "labels": pairs.map(
                lambda items: [int(item.rsplit("-", 1)[1]) for item in items]
            ),
        }
    )


def mind_history(raw: pd.DataFrame) -> pd.DataFrame:
    """MIND carries each user's history on the impression row itself."""
    return pd.DataFrame(
        {
            "impression_id": raw["impression_id"].astype(str),
            "user_id": raw["user_id"].astype(str),
            "click_history": raw["history"].fillna("").str.split(),
        }
    )


def _first(values) -> str | None:
    """First element of a list-valued column, as a string."""
    if values is None or len(values) == 0:
        return None
    return str(values[0])


def ebnerd_articles(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": raw["article_id"].astype(str),
            "title": raw["title"],
            # EB-NeRD has no abstract; the subtitle plays that role.
            "abstract": raw["subtitle"],
            "body": raw["body"],
            "category": raw["category_str"],
            # Only the first subcategory, for parity with MIND's single value.
            "subcategory": raw["subcategory"].map(_first),
            "published_time": raw["published_time"],
        }
    )


def mind_articles(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": raw["news_id"].astype(str),
            "title": raw["title"],
            "abstract": raw["abstract"],
            "category": raw["category"],
            "subcategory": raw["subcategory"],
        }
    )


def ebnerd_behaviors(raw: pd.DataFrame) -> pd.DataFrame:
    candidates = raw["article_ids_inview"].map(lambda ids: [str(i) for i in ids])
    clicked = raw["article_ids_clicked"].map(lambda ids: {str(i) for i in ids})
    return pd.DataFrame(
        {
            "impression_id": raw["impression_id"].astype(str),
            "user_id": raw["user_id"].astype(str),
            "impression_time": raw["impression_time"],
            "candidate_ids": candidates,
            # EB-NeRD gives the clicked ids rather than a label per candidate.
            "labels": [
                [int(candidate in seen) for candidate in row]
                for row, seen in zip(candidates, clicked)
            ],
        }
    )


# How many of a user's most recent clicks the engagement arrays keep.
#
# EB-NeRD stores one history row per user and ingest joins it onto every
# impression, which for this dataset is a 31x duplication -- 15,143 users
# become 232,887 impressions, each carrying its own copy. At a mean 160 clicks
# a user, three untruncated arrays are 229 million values per split, and the
# first attempt at this was OOM-killed at 10 GB.
#
# Truncating before the join bounds it. Every consumer takes the last
# `history_k` clicks, so keeping the last ENGAGEMENT_WINDOW loses nothing a
# window inside that would have seen -- and the consumer refuses a window
# larger than this rather than silently pairing arrays that no longer line up.
# Raising it past 100 means re-running ingest.
ENGAGEMENT_WINDOW = 100

def _tail(values, dtype: str) -> np.ndarray:
    """The last ENGAGEMENT_WINDOW entries of one history row's parallel array.

    Truncated *here*, on the per-user frame, rather than after ingest joins it
    onto impressions: the join is a 31x duplication on this dataset, so an
    untruncated array is copied 31 times and the first version of this was
    OOM-killed at 10 GB.

    Kept as a typed numpy array rather than a list of Python objects for the
    same reason — a Python float costs 32 bytes against float32's 4, which on
    229 million values is the difference between 7 GB and 900 MB. `scroll` is
    10.8% null within its arrays, which float32 carries as NaN and a list of
    Nones would not.
    """
    if values is None or (np.isscalar(values) and pd.isna(values)):
        return np.empty(0, dtype=dtype)
    return np.asarray(values[-ENGAGEMENT_WINDOW:], dtype=dtype)


def ebnerd_history(raw: pd.DataFrame) -> pd.DataFrame:
    """EB-NeRD keeps one history row per user, with no impression attached.

    The three `_fixed` columns beside the ids are parallel arrays over the same
    clicks: when each was read, for how long, and how far down the page. They
    come through aligned to the **last ENGAGEMENT_WINDOW** entries of
    `click_history`, so `array[-k:]` corresponds to `click_history[-k:]` for
    any k inside that window. MIND has no equivalent and carries them as null.
    """
    return pd.DataFrame(
        {
            "user_id": raw["user_id"].astype(str),
            "click_history": raw["article_id_fixed"].map(
                lambda ids: [str(i) for i in ids]
            ),
            "click_times": raw["impression_time_fixed"].map(
                lambda v: _tail(v, "datetime64[us]")
            ),
            "click_read_times": raw["read_time_fixed"].map(
                lambda v: _tail(v, "float32")
            ),
            "click_scroll": raw["scroll_percentage_fixed"].map(
                lambda v: _tail(v, "float32")
            ),
        }
    )


def mind_test_impressions(raw: pd.DataFrame) -> pd.DataFrame:
    """The competition's test impressions: candidates and history, no labels.

    Deliberately not `mind_behaviors`. That one splits "N3-1" on its trailing
    label, which the test file does not carry — it would raise on every row —
    and the history it needs comes from the same file, so one pass produces
    both rather than joining two adapters back together per chunk.

    The timestamp comes through for parity with the EB-NeRD adapter below,
    which reads one from its own file. Neither retriever looks at it: it costs
    one column and keeps both competitions' test impressions the same shape.
    """
    return pd.DataFrame(
        {
            "impression_id": raw["impression_id"].astype(str),
            "user_id": raw["user_id"].astype(str),
            "impression_time": pd.to_datetime(raw["time"], format="%m/%d/%Y %I:%M:%S %p"),
            "candidate_ids": raw["impressions"].str.split(),
            "click_history": raw["history"].fillna("").str.split(),
        }
    )


def ebnerd_test_impressions(raw: pd.DataFrame) -> pd.DataFrame:
    """The competition's test impressions: candidates, and no labels.

    Deliberately not `ebnerd_behaviors`: that one reads `article_ids_clicked`
    to build the label vector, and the test file does not carry the column --
    it is what the leaderboard is holding back. No history here either, unlike
    MIND's test adapter, because EB-NeRD ships it as a separate table that the
    submission spec names and `predict` joins on.
    """
    return pd.DataFrame(
        {
            "impression_id": raw["impression_id"].astype(str),
            "user_id": raw["user_id"].astype(str),
            "impression_time": pd.to_datetime(raw["impression_time"]),
            "candidate_ids": raw["article_ids_inview"].map(
                lambda ids: [str(i) for i in ids]
            ),
        }
    )
