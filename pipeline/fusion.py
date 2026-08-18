"""The third retriever: one model over the lexical, semantic and behavioural signals.

`bm25` and `ann` each answer one question — does this article read like what the
user has read — and on EB-NeRD the answer turns out to be almost independent of
whether they clicked it (AUC 0.505 and 0.498, either side of the coin flip). What
they are missing is not a better encoder. It is that a click on a news site is
mostly about what is being read *right now*, which no amount of article text can
say. `features` builds those columns; this module is what turns them into an
order.

It is a **re-ranker**, not a third way of searching the catalogue. The candidates
are given — by the impression locally, by the competition on the test files — and
every one of them comes back scored, so the model's job is to combine signals
rather than to find documents. Where a corpus ranking is asked for (`sweep`'s
recall@K) it reranks the pool the other two retrieve, which is what a two-stage
system would actually serve.

Two variants are registered, and the difference between them is the assignment's
"with and without features unavailable at serving time":

    fusion          every feature, including how often each candidate was
                    clicked in the hours before the impression
    fusion-serving  only what a competition test file can supply — content,
                    metadata, and how often each candidate was *shown*

Both train on the train split only and are scored on validation, which is later
in time; nothing here ever sees a label from the split it is scored on, and the
popularity counters are queried strictly before each impression's own timestamp.

The model is a gradient-boosted tree ensemble. Boosting rather than the logistic
regression the fusion could have been, because the useful reading of a feature is
conditional — a high three-hour CTR means something different on an article
published twenty minutes ago than on a week-old one — and that is exactly the
interaction a linear blend cannot hold. It costs about 0.02 AUC on both datasets.
sklearn's own implementation rather than LightGBM or XGBoost, because it is
already a dependency and at this feature count the difference is not measurable.
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline import ann_index, bm25_index, embed, features, paths, retrieval
from pipeline.datasets import DATASETS, DatasetConfig

# Where a fitted model is cached, under the dataset's artifacts directory.
FUSION_DIR = "fusion"

# Boosting parameters. Swept in `--tune`; these are what it chose on both
# datasets, and the surface is flat enough that the third decimal of AUC does
# not move between neighbouring settings.
LEARNING_RATE = 0.05
MAX_ITER = 400
MAX_LEAF_NODES = 31
MIN_SAMPLES_LEAF = 40
L2 = 1.0

# Impressions sampled from the train split to fit on. EB-NeRD's train split
# holds 292,018 of them and 3.25M candidate rows; the tenth of that this takes
# fits a fourteen-feature model to within noise of the whole, in a minute
# rather than a quarter of an hour.
TRAIN_IMPRESSIONS = 40_000
TRAIN_SEED = 0


class NotTrainedError(RuntimeError):
    """A model was asked for before the train split had been fitted."""


@dataclass(frozen=True)
class Model:
    """A fitted booster and the exact column list it was fitted on.

    The columns travel with the model because the two variants differ only in
    which columns they were shown, and a model handed the wrong ones would
    score every impression without complaining.
    """

    booster: HistGradientBoostingClassifier
    columns: list[str]
    variant: str
    history_k: int

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(
            frame[self.columns].to_numpy("float32")
        )[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)


def model_path(config: DatasetConfig, variant: str, history_k: int) -> Path:
    return config.artifacts_dir / FUSION_DIR / f"{variant}-k{history_k}.pkl"


def _sources(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store = config.feature_store_dir
    return (
        pd.read_parquet(store / "behaviors.parquet"),
        pd.read_parquet(store / "history.parquet"),
        pd.read_parquet(store / "articles.parquet"),
    )


def _aligned_history(history: pd.DataFrame, behaviors: pd.DataFrame) -> pd.DataFrame:
    """The history rows for these impressions, in the behaviours' own order.

    Both base retrievers pair a query to a candidate list by position at some
    point inside `score_candidates`, so the two frames have to agree about
    order before either is called — a mismatch produces a well-formed ranking
    of another user's candidates.
    """
    wanted = behaviors["impression_id"].astype(str)
    rows = history.set_index(history["impression_id"].astype(str))
    return rows.loc[wanted].reset_index(drop=True)


def counters(
    config: DatasetConfig, behaviors: pd.DataFrame, columns
) -> features.Popularity:
    """Fit the popularity counters on everything a server would have seen.

    Every impression up to and including the split being scored goes in — the
    counters are then *queried* strictly before each impression's timestamp, so
    what an impression can see is its own past and nothing else. Fitting them
    on the train split alone would instead answer "how popular was this article
    last week", which is a different and much weaker feature: on EB-NeRD a
    static train-split CTR ranks at 0.567 and the same counter read three hours
    back ranks at 0.716.

    When the variant is serving-only the labels are dropped before fitting, so
    the counter cannot produce a click column even by accident.
    """
    frame = features.explode(behaviors, labelled=True)
    if not any(column in columns for column in features.CLICKED):
        frame = frame.drop(columns="y")
    return features.Popularity.fit(frame)


def build_frame(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    columns,
    history_k: int,
    sources: tuple | None = None,
) -> pd.DataFrame:
    """Features for these impressions, with the counters fitted appropriately."""
    all_behaviors, history, articles = sources or _sources(config)
    # The counter sees every impression this pipeline holds; `Popularity.attach`
    # is what keeps each row to its own past.
    popularity = counters(config, all_behaviors, columns)
    content = features.Content.load(config, articles)
    return features.build(
        config,
        behaviors,
        _aligned_history(history, behaviors),
        articles,
        popularity,
        content,
        history_k,
    )


def train(
    config: DatasetConfig,
    columns=features.ALL,
    variant: str = "fusion",
    history_k: int = retrieval.HISTORY_K,
    force: bool = False,
) -> Model:
    """Fit on the train split, cache under artifacts, return the model.

    The train split is the only labelled data the model is allowed: validation
    is what it is scored on and test is held back, and a booster fitted on
    either would be reporting how well it memorised them.
    """
    path = model_path(config, variant, history_k)
    if path.exists() and not force:
        with path.open("rb") as handle:
            return pickle.load(handle)

    behaviors, history, articles = _sources(config)
    train_split = behaviors[behaviors["split"] == "train"]
    if len(train_split) > TRAIN_IMPRESSIONS:
        train_split = train_split.sample(
            TRAIN_IMPRESSIONS, random_state=TRAIN_SEED
        ).sort_values("impression_time")

    started = time.perf_counter()
    frame = build_frame(
        config,
        train_split,
        columns,
        history_k,
        sources=(behaviors, history, articles),
    )
    model_features = features.model_columns(columns)
    booster = HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITER,
        max_leaf_nodes=MAX_LEAF_NODES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        l2_regularization=L2,
        random_state=TRAIN_SEED,
    )
    booster.fit(frame[model_features].to_numpy("float32"), frame["y"].to_numpy())

    model = Model(
        booster=booster, columns=model_features, variant=variant, history_k=history_k
    )
    model.save(path)
    print(
        f"    {variant} fitted on {len(train_split):,} train impressions "
        f"({len(frame):,} candidates, {len(model_features)} features) in "
        f"{time.perf_counter() - started:.0f} s"
    )
    return model


def _order(
    frame: pd.DataFrame, scores: np.ndarray, impression_ids: np.ndarray
) -> pd.DataFrame:
    """Sort each impression's candidates by score, best first.

    Stable, so an impression the model scores flat comes back in the order it
    arrived — the same convention both base retrievers use for a cold user, and
    what makes the harness's `all_scores_tied` count mean the same thing here.

    `frame["imp"]` holds row positions rather than impression ids, so that a
    competition file which repeats an id does not have those impressions folded
    into one ranking; `impression_ids` is what turns each position back into
    the id the submission has to carry. It is the behaviours' own id column, so
    one row comes back per row given, repeated ids included.
    """
    ranked_ids: list[list[str]] = []
    ranked_scores: list[list[float]] = []
    impressions: list[str] = []

    ids = frame["aid"].to_numpy()
    start = 0
    for position, size in zip(*features._runs(frame["imp"].to_numpy())):
        block = slice(start, start + size)
        start += size
        order = np.argsort(-scores[block], kind="stable")
        impressions.append(impression_ids[position])
        ranked_ids.append(list(ids[block][order]))
        ranked_scores.append([float(s) for s in scores[block][order]])

    return pd.DataFrame(
        {
            "impression_id": impressions,
            "ranked_ids": ranked_ids,
            "scores": ranked_scores,
        }
    )


def rank_candidates(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
    columns=features.ALL,
    variant: str = "fusion",
) -> pd.DataFrame:
    """Score each impression's own candidates. The harness's only entry here."""
    model = train(config, columns, variant, history_k)
    all_behaviors, _, articles = _sources(config)
    frame = features.build(
        config,
        behaviors,
        _aligned_history(history, behaviors),
        articles,
        counters(config, all_behaviors, columns),
        features.Content.load(config, articles),
        history_k,
        labelled="labels" in behaviors,
    )
    return _order(
        frame, model.score(frame), behaviors["impression_id"].astype(str).to_numpy()
    )


def retrieve_corpus(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
    depth: int = max(retrieval.DEPTHS),
    columns=features.ALL,
    variant: str = "fusion",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rerank what the two base retrievers surface: recall@K for the pair fused.

    A re-ranker has no candidate generator of its own, so its recall is the
    recall of the pool it is given — the union of BM25's and the semantic
    index's top `depth`. Reranking cannot add a clicked article the pool
    missed, so this number is bounded above by the better of the two and says
    what a two-stage system would actually retrieve rather than what a third
    index would.
    """
    model = train(config, columns, variant, history_k)
    all_behaviors, _, articles = _sources(config)
    aligned = _aligned_history(history, behaviors)

    lexical, report = bm25_index.retrieve_corpus(
        config, behaviors, aligned, history_k, depth
    )
    semantic, _ = ann_index.retrieve_corpus(
        config, behaviors, aligned, history_k, depth
    )
    pool = _pool(behaviors, lexical, semantic)

    frame = features.build(
        config,
        pool,
        _aligned_history(history, pool),
        articles,
        counters(config, all_behaviors, columns),
        features.Content.load(config, articles),
        history_k,
        labelled=False,
    )
    ranked = _order(
        frame, model.score(frame), pool["impression_id"].astype(str).to_numpy()
    )
    ranked["ranked_ids"] = [ids[:depth] for ids in ranked["ranked_ids"]]
    ranked["scores"] = [values[:depth] for values in ranked["scores"]]
    return ranked, report


def _pool(
    behaviors: pd.DataFrame, lexical: pd.DataFrame, semantic: pd.DataFrame
) -> pd.DataFrame:
    """The two retrievers' rankings, unioned per impression, as a behaviours frame.

    Reshaped into a behaviours frame rather than a list of ids because that is
    what `features.build` reads, and because it keeps the pool path and the
    candidate path on exactly one feature implementation.
    """
    from_lexical = dict(zip(lexical["impression_id"].astype(str), lexical["ranked_ids"]))
    from_semantic = dict(
        zip(semantic["impression_id"].astype(str), semantic["ranked_ids"])
    )
    pooled = []
    for impression in behaviors["impression_id"].astype(str):
        seen = dict.fromkeys(from_lexical.get(impression, []))
        seen.update(dict.fromkeys(from_semantic.get(impression, [])))
        pooled.append(list(seen))
    return pd.DataFrame(
        {
            "impression_id": behaviors["impression_id"].astype(str).to_numpy(),
            "impression_time": behaviors["impression_time"].to_numpy(),
            "candidate_ids": pooled,
        }
    )


@dataclass(frozen=True)
class Ranker:
    """Scores a supplied candidate list against a supplied catalogue.

    The submission path's entry. Unlike the two base retrievers this one needs
    a counter over the competition's own impressions, which `predict` fits from
    the candidate lists in the test file — no labels exist there, so only the
    serving variant can be submitted, and `for_submission` refuses the other
    rather than silently scoring a column of zeros.
    """

    model: Model
    content: features.Content
    popularity: features.Popularity
    articles: pd.DataFrame
    config: DatasetConfig
    history_k: int

    def rank(self, history: pd.DataFrame, candidates: list[list[str]]) -> pd.DataFrame:
        if "impression_time" not in history:
            # Every popularity feature is read at this timestamp. Filling one
            # in would produce a ranking that looks fine and is built from the
            # wrong hour of the news cycle, so the adapter has to supply it.
            raise NotTrainedError(
                f"{self.config.name}'s impression adapter returns no "
                "impression_time, which every popularity feature is read at"
            )
        behaviors = pd.DataFrame(
            {
                "impression_id": history["impression_id"].astype(str).to_numpy(),
                "impression_time": history["impression_time"].to_numpy(),
                "candidate_ids": candidates,
            }
        )
        frame = features.build(
            self.config,
            behaviors,
            history,
            self.articles,
            self.popularity,
            self.content,
            self.history_k,
            labelled=False,
        )
        return _order(
            frame, self.model.score(frame), behaviors["impression_id"].to_numpy()
        )


# Chunks of the competition's impression file aggregated before the running
# exposure table is folded down again. Every fold is a sort over the whole
# table, so folding per chunk would spend the run sorting; letting it grow
# unbounded would spend it in memory.
FOLD_EVERY = 25


def exposure(
    config: DatasetConfig,
    workdir: Path,
    chunk_size: int,
    history_k: int,
    force: bool = False,
) -> features.Popularity:
    """The counter for a competition file, read off its own candidate lists.

    This is what the user's message about the submission comes down to. The
    test file ships no labels, so there is no click history of the test period
    to count — but it ships thirteen million candidate lists, and *what an
    editor put in front of people* is most of what popularity was measuring in
    the first place. On EB-NeRD's validation split, exposure counted this way
    ranks at 0.645 on its own, against 0.505 for BM25 over the same
    impressions.

    Counting it needs a pass over the file before any of it can be ranked,
    because an impression's feature is the exposure of its candidates *up to
    that moment* and the moments arrive interleaved. The pass is cheap next to
    the ranking one — no vectors, no index, three columns — and its result is
    cached, so a resumed or repeated submission pays for it once.

    Bucketed and folded down as it goes: the raw event stream is 206M rows for
    EB-NeRD, and the table that answers every question anyone asks of it is the
    (article x five minutes) grid underneath, which is two orders of magnitude
    smaller.
    """
    from pipeline import predict  # circular at module scope: predict picks the retriever

    cached = workdir / FUSION_DIR / "exposure.parquet"
    if cached.exists() and not force:
        return features.Popularity.from_counts(
            pd.read_parquet(cached), labelled=False, resolution=features.RESOLUTION
        )

    started = time.perf_counter()
    pending: list[pd.DataFrame] = []
    table: pd.DataFrame | None = None
    rows = 0
    stream = predict.impressions(config, chunk_size, history_k, with_history=False)
    for number, chunk in enumerate(stream, 1):
        pending.append(_bucket(chunk))
        rows += len(chunk)
        if number % FOLD_EVERY == 0:
            table, pending = _fold(table, pending), []
    table = _fold(table, pending)

    cached.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cached, index=False)
    print(
        f"    exposure counted over {rows:,} impressions into {len(table):,} "
        f"(article x {features.RESOLUTION}) buckets in "
        f"{time.perf_counter() - started:.0f} s"
    )
    return features.Popularity.from_counts(
        table, labelled=False, resolution=features.RESOLUTION
    )


def _bucket(chunk: pd.DataFrame) -> pd.DataFrame:
    """One chunk's candidate rows, counted per (article, bucket)."""
    frame = features.explode(chunk, labelled=False)
    return (
        pd.DataFrame(
            {
                "aid": frame["aid"].to_numpy(),
                "t": features._floor(frame["t"].to_numpy(), features.RESOLUTION),
                "shows": 1,
            }
        )
        .groupby(["aid", "t"], sort=False, as_index=False)
        .sum()
    )


def _fold(
    table: pd.DataFrame | None, pending: list[pd.DataFrame]
) -> pd.DataFrame | None:
    if not pending:
        return table
    parts = pending if table is None else [table, *pending]
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(["aid", "t"], sort=False, as_index=False)
        .sum()
    )


def ranker(
    articles: pd.DataFrame,
    config: DatasetConfig,
    workdir: Path,
    history_k: int = retrieval.HISTORY_K,
    columns=features.SERVING,
    variant: str = "fusion-serving",
) -> Ranker:
    """A ranker over the competition's catalogue and its own exposure counts.

    The signature `predict` calls on every retriever. The two extra things this
    one needs — vectors for a catalogue the stored artifact only half covers,
    and the exposure pass above — are both built here and both cached under
    `workdir`, so `predict` still knows nothing about which retriever it holds.
    """
    if any(column in columns for column in features.CLICKED):
        raise NotTrainedError(
            f"{variant} scores a click-count feature, and a competition test "
            "file ships no clicks to count. Submit fusion-serving, whose "
            "features are exactly the ones the test file can supply."
        )
    model = train(config, columns, variant, history_k)
    lexical = bm25_index.ranker(articles, config, workdir, history_k).index
    embeddings, report = embed.for_corpus(
        articles, config, workdir / embed.EMBED_DIR
    )
    print(
        f"    {report['articles']:,} article vectors "
        f"({'cached' if report['cached'] else 'built'})"
    )
    from pipeline import predict

    return Ranker(
        model=model,
        content=features.Content(
            config=config, articles=articles, lexical=lexical, embeddings=embeddings
        ),
        popularity=exposure(config, workdir, predict.CHUNK, history_k),
        articles=articles,
        config=config,
        history_k=history_k,
    )


def _variant(name: str, columns) -> SimpleNamespace:
    """One entry for `evaluate.RETRIEVERS`, with its feature set bound in.

    A namespace rather than a module because the two variants differ only in a
    column list, and a second file whose whole content is that list would be a
    worse way of saying so.
    """
    return SimpleNamespace(
        rank_candidates=partial(rank_candidates, columns=columns, variant=name),
        retrieve_corpus=partial(retrieve_corpus, columns=columns, variant=name),
        ranker=partial(ranker, columns=columns, variant=name),
        train=partial(train, columns=columns, variant=name),
        columns=columns,
        variant=name,
    )


FULL = _variant("fusion", features.ALL)
SERVING = _variant("fusion-serving", features.SERVING)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--history-k", type=int, default=retrieval.HISTORY_K)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths.load_env_file()
    for name in args.dataset or sorted(DATASETS):
        config = DATASETS[name]
        print(f"{name}:")
        for variant in (FULL, SERVING):
            variant.train(config, history_k=args.history_k, force=args.force)


if __name__ == "__main__":
    main()
