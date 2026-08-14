"""Per-dataset adapters: raw source frame -> unified schema frame.

This is where MIND's and EB-NeRD's shapes genuinely differ, so it is the one
module that knows about either. Everything downstream, ingest included, works
off the unified frames these produce. A third dataset means a new adapter and a
new registry entry, and no change to any stage.
"""

from __future__ import annotations

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


def ebnerd_history(raw: pd.DataFrame) -> pd.DataFrame:
    """EB-NeRD keeps one history row per user, with no impression attached."""
    return pd.DataFrame(
        {
            "user_id": raw["user_id"].astype(str),
            "click_history": raw["article_id_fixed"].map(
                lambda ids: [str(i) for i in ids]
            ),
        }
    )


def mind_test_impressions(raw: pd.DataFrame) -> pd.DataFrame:
    """The competition's test impressions: candidates and history, no labels.

    Deliberately not `mind_behaviors`. That one splits "N3-1" on its trailing
    label, which the test file does not carry — it would raise on every row —
    and the history it needs comes from the same file, so one pass produces
    both rather than joining two adapters back together per chunk.
    """
    return pd.DataFrame(
        {
            "impression_id": raw["impression_id"].astype(str),
            "user_id": raw["user_id"].astype(str),
            "candidate_ids": raw["impressions"].str.split(),
            "click_history": raw["history"].fillna("").str.split(),
        }
    )
