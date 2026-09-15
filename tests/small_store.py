"""A whole small pipeline on disk, for the tests that need one.

Three test modules need the same thing: a feature store over a handful of
articles, the indexes and counters built from it, an NRMS checkpoint and a
re-ranker fitted on it. Building it takes a second and describing it takes
seventy lines, so it is described once here rather than three times -- two
copies would drift, and a test that passes against a store subtly unlike the
one another test uses is a test whose failure means something else.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import bm25_index, counters, embed, features, nrms, paths, retrieval
from pipeline.datasets import DATASETS, NrmsSpec, RerankSpec

MIND = DATASETS["mind"]

T0 = pd.Timestamp("2019-11-11 06:00:00")

ARTICLES = (
    ("a1", "sports", "s1", (1.0, 0.0)),
    ("a2", "sports", "s1", (0.87, 0.5)),
    ("a3", "finance", "f1", (0.0, 1.0)),
    ("a4", "finance", "f1", (0.5, 0.87)),
    ("a5", "tech", "t1", (2.0**-0.5, 2.0**-0.5)),
    ("a6", "tech", "t1", (-1.0, 0.0)),
)

# Small enough to train in a moment, and the same model either way.
SMALL_NRMS = NrmsSpec(
    # The registry's window, not a smaller one. `NrmsSpec.history_length`
    # defaults to `retrieval.HISTORY_K`, and everything that scores NRMS
    # through the harness -- `evaluate`, and so `final` -- passes that default
    # and is refused by a checkpoint fitted at any other. Shrinking it here
    # would make the fixture disagree with the registry about a number the
    # checkpoint carries in its weights.
    history_length=retrieval.HISTORY_K,
    heads=2,
    head_dim=4,
    attention_dim=8,
    dropout=0.0,
    negatives=2,
    epochs=1,
    batch_size=8,
    score_batch=8,
)
SMALL_RERANK = RerankSpec(
    leaves=4, rounds=12, min_data_in_leaf=1, early_stopping=5, learning_rate=0.2
)


def impressions_of(count=16):
    """A train split over distinct hours, then tune and validation after it."""
    rows = []
    for i in range(count):
        sporty = i % 2 == 0
        rows.append(
            (
                f"t{i}",
                f"u{i % 3}",
                i,
                ["a1", "a3", "a5"] if sporty else ["a2", "a4", "a6"],
                [1, 0, 0] if sporty else [0, 1, 0],
                "train",
            )
        )
    for i in range(4):
        rows.append(
            (
                f"n{i}",
                f"u{i % 3}",
                count + i,
                ["a1", "a4", "a6"],
                [1, 0, 0] if i % 2 == 0 else [0, 1, 0],
                "tune",
            )
        )
    for i in range(4):
        rows.append(
            (
                f"v{i}",
                f"u{i % 3}",
                count + 8 + i,
                ["a2", "a3", "a5"],
                [1, 0, 0] if i % 2 == 0 else [0, 0, 1],
                "validation",
            )
        )
    # After validation in time, so a causal read at a validation moment cannot
    # see them: adding a held-out split must not move any number measured on
    # the splits before it.
    for i in range(4):
        rows.append(
            (
                f"s{i}",
                f"u{i % 3}",
                count + 16 + i,
                ["a1", "a4", "a5"],
                [1, 0, 0] if i % 2 == 0 else [0, 1, 0],
                "test",
            )
        )
    return pd.DataFrame(
        {
            "impression_id": pd.Series([r[0] for r in rows], dtype="string"),
            "user_id": pd.Series([r[1] for r in rows], dtype="string"),
            "source": pd.Series(["train"] * len(rows), dtype="string"),
            "impression_time": pd.Series(
                [T0 + pd.Timedelta(hours=r[2]) for r in rows], dtype="datetime64[us]"
            ),
            "session_id": pd.Series([None] * len(rows), dtype="string"),
            "candidate_ids": [list(r[3]) for r in rows],
            "labels": [list(r[4]) for r in rows],
            "split": pd.Series([r[5] for r in rows], dtype="string"),
        }
    )


def write_store(config=MIND):
    """The whole pipeline's outputs for a small log: the three tables, the
    vectors, the BM25 index, the counters, and the three feature frames."""
    store = config.feature_store_dir
    store.mkdir(parents=True, exist_ok=True)
    behaviors = impressions_of()
    behaviors.to_parquet(store / "behaviors.parquet", index=False)
    pd.DataFrame(
        {
            "article_id": pd.Series([a[0] for a in ARTICLES], dtype="string"),
            "title": pd.Series([f"headline {a[1]}" for a in ARTICLES], dtype="string"),
            "abstract": pd.Series([""] * len(ARTICLES), dtype="string"),
            "lexical_text": pd.Series(
                [f"headline {a[1]} {a[0]}" for a in ARTICLES], dtype="string"
            ),
            "category": pd.Series([a[1] for a in ARTICLES], dtype="string"),
            "subcategory": pd.Series([a[2] for a in ARTICLES], dtype="string"),
            "published_time": pd.Series([pd.NaT] * len(ARTICLES), dtype="datetime64[us]"),
        }
    ).to_parquet(store / "articles.parquet", index=False)
    users = {"u0": ["a1", "a3"], "u1": ["a2", "a4"], "u2": ["a5"]}
    pd.DataFrame(
        {
            "user_id": pd.Series(list(users), dtype="string"),
            "source": pd.Series(["train"] * len(users), dtype="string"),
            "click_history": [list(clicks) for clicks in users.values()],
            "n_clicks": [len(clicks) for clicks in users.values()],
        }
    ).to_parquet(store / "history.parquet", index=False)

    embed.Embeddings(
        vectors=np.array([a[3] for a in ARTICLES], dtype="float32"),
        article_ids=np.array([a[0] for a in ARTICLES], dtype=object),
    ).save(embed.output_dir(config))
    bm25_index.run(config)
    counters.build(behaviors).save(config.artifacts_dir / counters.DIRECTORY)
    for split in features.SPLITS:
        features.build(config, split)
    model, _ = nrms.train(config, SMALL_NRMS)
    nrms.save(model, SMALL_NRMS, 2, nrms.checkpoint_path(config, SMALL_NRMS))
    return behaviors


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path


@pytest.fixture
def small(monkeypatch):
    """The registry's two specs, shrunk, since `run` and `rank_candidates` read
    them off the config rather than taking them as arguments."""
    monkeypatch.setattr(
        MIND.__class__, "nrms", property(lambda self: SMALL_NRMS), raising=False
    )
    monkeypatch.setattr(
        MIND.__class__, "rerank", property(lambda self: SMALL_RERANK), raising=False
    )
