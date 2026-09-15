"""The re-ranker: a LightGBM model over the feature frame and the NRMS score.

This is the improvement the assignment asks for, and the number that says so is
`rerank - nrms` on `validation`, paired by impression, with a bootstrap
interval. It is a fifth entry in `evaluate.RETRIEVERS`, so that claim comes out
of the same code path as every other number in the project rather than out of a
script written to produce it.

**Where it is fitted.** On the *later* chronological half of `train`, which is
the half NRMS did not see (`nrms.halves`). So the NRMS score the trees learn to
weight is a prediction about impressions the NRMS was not fitted on -- the
same kind of number it will produce in production. Fitting both on the same
rows would teach the trees to trust a feature that is partly memory.

**What an arm is.** Every ablation of ticket 07 is this module with a different
`RerankSpec`: a tier dropped, the NRMS column withheld, a counter window
chosen, the leaky frame read instead of the causal one, a literal top-K cut
applied. Dropping a tier is a *projection* -- the columns never leave the disk
-- because a column of zeros is still a column a tree can split on and still
a column the reader pays for.

**Two paths, on purpose.** Training reads the materialised frame sequentially,
row group by row group, with the arm's column projection; it is the largest
read in the project and the one place where the tier structure buys memory.
Scoring *computes* the frame per chunk through `features.frame_for`, which is
what the submission does and what a server would do -- so the per-impression
milliseconds in the ledger are the served number and not a read off a table
that was prepared earlier.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from pipeline import (
    counters,
    features,
    ingest,
    ledger,
    nrms,
    retrieval,
    timings,
)

# `evaluate` is imported inside the functions that need it, not here: it holds
# the retriever table this module is an entry in, so importing it at module
# level is a cycle. The two modules that already do this -- `embed_compare` for
# its bootstrap, `bench` for its percentiles -- are imported the same way.
from pipeline.datasets import DATASETS, DEFAULT_DATASETS, DatasetConfig, RerankSpec

STAGE = "rerank"
DIRECTORY = "rerank"
MODEL = "model.txt"
IMPORTANCE = "importance.json"

FIT_SPLIT = "train"
TUNE_SPLIT = "tune"

# The column the NRMS score arrives as. Not in `features.FEATURE_GROUPS`: it is
# not a tier of availability but a second model's output, and ticket 07's
# `-NRMS` arm drops it by name.
NRMS_COLUMN = "nrms_score"

# Every counter column carries its window as a suffix. `all` keeps them all,
# which is the row that says whether the windows add up.
ALL_WINDOWS = "all"

OBJECTIVES = ("binary", "lambdarank")

# The grid ticket 06 sweeps on tune, one axis at a time from the registry's
# spec. `top_k` is here because the literal cut is a row in the same table;
# `causal=False` is Q9's arm and is swept from the same place, so the leaky
# model differs from the clean one in one field and nothing else.
GRID = {
    "objective": ("binary", "lambdarank"),
    "leaves": (31, 127),
    "window": (*counters.WINDOWS, ALL_WINDOWS),
    "nrms": (True, False),
    "precision": ("float32", "float16"),
    "top_k": (None, 50, 100, 200),
    "causal": (True, False),
}

# How many rounds the saved snapshots hold, beside the early-stopped best. The
# note's rounds curve: what the last hundred trees buy, and what they cost in
# bytes and in milliseconds per request.
SNAPSHOTS = (50, 200)


class RerankError(RuntimeError):
    """The re-ranker was asked for something its spec or its frame cannot give."""


# ---------------------------------------------------------------------------
# The columns an arm reads.


def window_columns(window: str) -> tuple[str, ...]:
    """The counter columns of one window, or all of them.

    Selected by the suffix the frame already names them with, so adding a
    window to `counters.WINDOWS` adds it here and nowhere else.
    """
    if window == ALL_WINDOWS:
        return ()
    if window not in counters.WINDOWS:
        raise RerankError(
            f"unknown counter window {window!r}; use one of "
            f"{', '.join([*counters.WINDOWS, ALL_WINDOWS])}"
        )
    return tuple(
        name
        for name in features.FEATURES
        if name.endswith(f"_{window}")
    )


def feature_columns(spec: RerankSpec) -> tuple[str, ...]:
    """Exactly the columns this arm trains and scores on, in one order.

    One function, used by the training read, the scoring path and the
    importance table, so a model cannot be fitted on one column order and
    scored on another -- which LightGBM would not notice and which would be
    wrong in a way no metric could show.
    """
    chosen = [
        name
        for name in features.columns_for(spec.groups)
        if name not in features.KEY_COLUMNS
    ]
    if spec.window != ALL_WINDOWS:
        kept = set(window_columns(spec.window))
        windowed = {
            name
            for window in counters.WINDOWS
            for name in window_columns(window)
        }
        chosen = [name for name in chosen if name not in windowed or name in kept]
    if spec.nrms:
        chosen.append(NRMS_COLUMN)
    return tuple(chosen)


def variant_of(spec: RerankSpec) -> str:
    """What the ledger calls this arm: the fields that make it a different
    model, and none that do not."""
    dropped = [
        group for group in features.FEATURE_GROUPS if group not in spec.groups
    ] or ["none"]
    return (
        f"{spec.objective}-l{spec.leaves}-{spec.window}"
        f"-drop:{'+'.join(dropped)}"
        f"{'' if spec.nrms else '-nonrms'}"
        f"{'' if spec.causal else '-leaky'}"
        f"{'' if spec.top_k is None else f'-cut{spec.top_k}'}"
        f"-{spec.precision}"
    )


# ---------------------------------------------------------------------------
# The NRMS column.


def scores_path(config: DatasetConfig, split: str) -> Path:
    return (
        config.artifacts_dir
        / nrms.DIRECTORY
        / f"scores-{split}-{nrms.variant_of(config.nrms)}.parquet"
    )


def nrms_scores(
    config: DatasetConfig,
    split: str,
    impressions: pd.DataFrame | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """One NRMS score per (impression, article), cached per split.

    Cached because every arm of the ablation reads the same column and scoring
    a split through a torch model is minutes, not seconds -- and because the
    column has to be *identical* across arms, or a difference between two arms
    would be partly a difference in their features.
    """
    path = scores_path(config, split)
    if path.exists() and not force:
        return pd.read_parquet(path)

    store = config.feature_store_dir
    if impressions is None:
        behaviors = pd.read_parquet(store / "behaviors.parquet")
        impressions = behaviors[behaviors["split"] == split]
    history = ingest.history_for(config, impressions)
    ranked = nrms.rank_candidates(
        config, impressions, history, config.nrms.history_length
    )

    rows = {"impression_id": [], "article_id": [], NRMS_COLUMN: []}
    for impression, articles, scores in zip(
        ranked["impression_id"], ranked["ranked_ids"], ranked["scores"]
    ):
        rows["impression_id"].extend([impression] * len(articles))
        rows["article_id"].extend(articles)
        rows[NRMS_COLUMN].extend(scores)
    frame = pd.DataFrame(rows)
    frame["impression_id"] = frame["impression_id"].astype("string")
    frame["article_id"] = frame["article_id"].astype("string")
    frame[NRMS_COLUMN] = frame[NRMS_COLUMN].astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def with_nrms(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """The long frame plus the NRMS column, joined by (impression, article).

    By the pair rather than by position: the NRMS frame comes back in *ranked*
    order, and a positional join would give every row another candidate's
    score -- the same mistake `features.read_back` exists to avoid, one stage
    later.
    """
    joined = frame.merge(scores, on=["impression_id", "article_id"], how="left")
    if len(joined) != len(frame):
        raise RerankError(
            f"{len(frame):,} rows joined to {len(joined):,}: the NRMS scores "
            f"are not one per (impression, candidate)"
        )
    return joined


# ---------------------------------------------------------------------------
# Reading the frame for training.


def group_sizes(impression_ids: np.ndarray) -> np.ndarray:
    """How many rows each impression has, in the order they arrive.

    The long frame is the candidate list exploded per impression, so an
    impression's rows are contiguous -- which is what lets every consumer here
    group by run length rather than by a hash, and what `lambdarank` needs to
    draw its group boundaries. Checked rather than assumed: a boundary in the
    wrong place would train the ranker to order one impression's candidates
    against another's, and nothing downstream could see it.
    """
    ids = np.asarray(impression_ids)
    if not len(ids):
        return np.zeros(0, dtype="int32")
    changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    sizes = np.diff([0, *changes, len(ids)]).astype("int32")
    if len(np.unique(ids)) != len(sizes):
        raise RerankError(
            "an impression's rows are not contiguous in the frame, so a group "
            "boundary would fall inside one"
        )
    return sizes


def group_spans(impression_ids: np.ndarray):
    """(impression id, slice) per impression, in arrival order."""
    at = 0
    for size in group_sizes(impression_ids):
        yield impression_ids[at], slice(at, at + size)
        at += size


@dataclass
class Rows:
    """One arm's training matrix, and what groups its rows into impressions."""

    matrix: np.ndarray
    labels: np.ndarray
    impression_ids: np.ndarray
    columns: tuple[str, ...]

    @property
    def groups(self) -> np.ndarray:
        return group_sizes(self.impression_ids)


def read_rows(
    config: DatasetConfig,
    split: str,
    spec: RerankSpec,
    keep: set[str] | None = None,
    scores: pd.DataFrame | None = None,
) -> Rows:
    """The arm's rows, read a row group at a time with its projection.

    Never `read_parquet` of the whole split: the frame is the largest thing A2
    materialises, the projection is the point of the tier structure, and a
    reader that pulled every column into pandas first would make the ablation's
    memory column a fiction. `keep` filters to a set of impressions -- the
    later half of `train` -- as each batch arrives, so the rows that are not
    this arm's never accumulate.
    """
    projection = [
        *features.KEY_COLUMNS,
        *(name for name in feature_columns(spec) if name != NRMS_COLUMN),
    ]
    path = features.path_for(config, split, spec.causal, spec.precision)
    if not path.exists():
        raise RerankError(
            f"no feature frame at {path}. Build it with "
            f"`python -m pipeline.features --dataset {config.name} --split {split}"
            f"{' --leaky' if not spec.causal else ''}`"
        )

    parts: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        batch = parquet.read_row_group(group, columns=projection).to_pandas()
        if keep is not None:
            batch = batch[batch["impression_id"].isin(keep)]
        if len(batch):
            parts.append(batch)
    frame = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=projection)
    )
    if spec.nrms:
        if scores is None:
            scores = nrms_scores(config, split)
        frame = with_nrms(frame, scores)
    if frame.empty:
        raise RerankError(f"no {split} rows for {variant_of(spec)}")

    columns = feature_columns(spec)
    return Rows(
        matrix=frame[list(columns)].to_numpy(dtype="float32"),
        labels=frame["label"].to_numpy(dtype="int32"),
        impression_ids=frame["impression_id"].to_numpy(),
        columns=columns,
    )


def later_half(config: DatasetConfig) -> set[str]:
    """The impressions the re-ranker may fit on: the half NRMS did not see."""
    behaviors = pd.read_parquet(
        config.feature_store_dir / "behaviors.parquet",
        columns=["impression_id", "impression_time", "split"],
    )
    train = behaviors[behaviors["split"] == FIT_SPLIT]
    _, later = nrms.halves(train, config.nrms.train_fraction)
    return set(later["impression_id"])


# ---------------------------------------------------------------------------
# Fitting.


def parameters(spec: RerankSpec) -> dict:
    from pipeline import evaluate

    if spec.objective not in OBJECTIVES:
        raise RerankError(
            f"unknown objective {spec.objective!r}; use one of "
            f"{', '.join(OBJECTIVES)}"
        )
    shared = {
        "objective": spec.objective,
        "num_leaves": spec.leaves,
        "learning_rate": spec.learning_rate,
        "min_data_in_leaf": spec.min_data_in_leaf,
        "num_threads": spec.threads,
        "seed": spec.seed,
        "deterministic": True,
        "verbose": -1,
    }
    if spec.objective == "lambdarank":
        # Grouped by impression, and scored at the depths the harness reports.
        return {**shared, "metric": "ndcg", "ndcg_eval_at": list(evaluate.NDCG_DEPTHS)}
    return {**shared, "metric": "auc"}


def dataset_of(rows: Rows, spec: RerankSpec, reference=None) -> lgb.Dataset:
    data = lgb.Dataset(
        rows.matrix,
        label=rows.labels,
        feature_name=list(rows.columns),
        reference=reference,
        free_raw_data=False,
    )
    if spec.objective == "lambdarank":
        data.set_group(rows.groups)
    return data


def train(
    config: DatasetConfig,
    spec: RerankSpec | None = None,
    fit: Rows | None = None,
    tune: Rows | None = None,
) -> tuple[lgb.Booster, dict]:
    """Fit on the later half of `train`, stop on `tune`, keep the curve.

    The snapshots at `SNAPSHOTS` rounds are the same booster truncated, not
    three trainings: LightGBM scores with `num_iteration`, so the rounds curve
    costs one fit and the note gets AUC against bytes and milliseconds for
    free.
    """
    spec = spec or config.rerank
    scores = nrms_scores(config, FIT_SPLIT) if spec.nrms else None
    fit = fit or read_rows(config, FIT_SPLIT, spec, later_half(config), scores)
    tune = tune or read_rows(config, TUNE_SPLIT, spec)

    evaluated: dict[str, dict] = {}
    started = time.perf_counter()
    booster = lgb.train(
        parameters(spec),
        dataset_of(fit, spec),
        num_boost_round=spec.rounds,
        valid_sets=[dataset_of(tune, spec)],
        valid_names=["tune"],
        callbacks=[
            lgb.early_stopping(spec.early_stopping, verbose=False),
            lgb.record_evaluation(evaluated),
        ],
    )
    curve = [
        {"rounds": round_number + 1, "tune": value}
        for metric in evaluated.get("tune", {})
        for round_number, value in enumerate(evaluated["tune"][metric])
    ]
    return booster, {
        "dataset": config.name,
        "variant": variant_of(spec),
        "spec": dataclasses.asdict(spec),
        "columns": list(fit.columns),
        "fit_rows": len(fit.labels),
        "fit_impressions": int(len(np.unique(fit.impression_ids))),
        "tune_rows": len(tune.labels),
        "rounds": booster.best_iteration or booster.current_iteration(),
        "curve": curve,
        "train_seconds": time.perf_counter() - started,
    }


def importance(booster: lgb.Booster, spec: RerankSpec) -> dict:
    """Gain per column, and summed per availability tier.

    The per-tier sum is what ticket 07's arms are about, and the per-column
    gain is what settles the options ticket 04 left as columns side by side --
    which counter window, which profile, which pooling. Read rather than
    argued.
    """
    gains = dict(
        zip(booster.feature_name(), booster.feature_importance("gain").tolist())
    )
    per_tier: dict[str, float] = {}
    for name, gain in gains.items():
        tier = NRMS_COLUMN if name == NRMS_COLUMN else features.tier_of(name)
        per_tier[tier] = per_tier.get(tier, 0.0) + float(gain)
    return {
        "variant": variant_of(spec),
        "columns": dict(sorted(gains.items(), key=lambda item: -item[1])),
        "tiers": dict(sorted(per_tier.items(), key=lambda item: -item[1])),
    }


def model_path(config: DatasetConfig, spec: RerankSpec | None = None) -> Path:
    spec = spec or config.rerank
    return config.artifacts_dir / DIRECTORY / f"{variant_of(spec)}.txt"


def save(booster: lgb.Booster, config: DatasetConfig, spec: RerankSpec, rounds=None) -> Path:
    path = model_path(config, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path), num_iteration=rounds)
    (path.parent / f"{variant_of(spec)}-{IMPORTANCE}").write_text(
        json.dumps(importance(booster, spec), indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_model(config: DatasetConfig, spec: RerankSpec | None = None) -> lgb.Booster:
    spec = spec or config.rerank
    path = model_path(config, spec)
    if not path.exists():
        raise RerankError(
            f"no re-ranker at {path}. Train one with "
            f"`python -m pipeline.rerank --dataset {config.name}`"
        )
    return lgb.Booster(model_file=str(path))


# ---------------------------------------------------------------------------
# Scoring: the served path.


def cut_outside_k(scores: np.ndarray, ranks: np.ndarray, top_k: int) -> np.ndarray:
    """Push the candidates the retriever did not surface to the bottom.

    The ablation that measures what a real two-stage cut costs: a production
    system re-ranks the top K and never sees the rest, so the rest are ranked
    last rather than scored. Their order among themselves is the stage-one
    order, which is what a server would show if it padded the list out.

    Their scores step down from just below the lowest scored candidate, one
    step per place in the stage-one order, so they stay below everything the
    model looked at and stay separable from each other. A NaN rank is a
    candidate the retriever never ranked at all, which is outside every K and
    sorts last among the cut.
    """
    outside = ~(ranks <= top_k)
    if not outside.any():
        return scores
    floor = np.nanmin(scores) if np.isfinite(scores).any() else 0.0
    # argsort of an argsort is the dense position of each element in the sorted
    # order -- the stage-one order, with the unranked pushed to the end of it.
    order = np.argsort(
        np.argsort(np.where(np.isnan(ranks[outside]), np.inf, ranks[outside]), kind="stable"),
        kind="stable",
    )
    pushed = scores.copy()
    pushed[outside] = floor - 1.0 - order
    return pushed


def score_frame(
    booster: lgb.Booster, frame: pd.DataFrame, spec: RerankSpec
) -> np.ndarray:
    """One `predict` call over a chunk's long frame.

    Per chunk rather than per impression: LightGBM's per-call overhead is
    milliseconds and its per-row cost is microseconds, so a call per impression
    would report the overhead as the model's cost.
    """
    columns = feature_columns(spec)
    missing = [name for name in columns if name not in frame]
    if missing:
        raise RerankError(
            f"the frame is missing {', '.join(missing)}, which this arm trains "
            f"on. A frame built for one arm cannot be scored by another."
        )
    scores = booster.predict(
        frame[list(columns)].to_numpy(dtype="float32"),
        num_threads=spec.threads,
    )
    return np.asarray(scores, dtype="float64")


def ranked_from(frame: pd.DataFrame, scores: np.ndarray, spec: RerankSpec) -> pd.DataFrame:
    """The harness's shape, grouped back out of the long frame.

    The frame is the candidate list exploded in arrival order, so grouping by
    impression and keeping the row order recovers exactly the list the dataset
    gave -- which is what makes the stable sort in `retrieval.ranked_frame`
    mean what it says.
    """
    ids = frame["impression_id"].to_numpy()
    articles = frame["article_id"].to_numpy()
    ranks = (
        frame["ann_rank"].to_numpy(dtype="float64")
        if spec.top_k is not None
        else None
    )

    impression_ids, candidates, per_impression = [], [], []
    for impression, span in group_spans(ids):
        found = scores[span]
        if ranks is not None:
            found = cut_outside_k(found, ranks[span], spec.top_k)
        impression_ids.append(impression)
        candidates.append(list(articles[span]))
        per_impression.append(found)
    return retrieval.ranked_frame(impression_ids, candidates, per_impression)


def rank_candidates(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
    pooling: str = retrieval.POOLING,
) -> pd.DataFrame:
    """Score each impression's own candidates. The harness's only entry here.

    The features are *computed* rather than read: this is the serving path, and
    the submission runs the same function over the competition's chunks. The
    stored frame is for training, where the read is the expensive part.
    """
    spec = config.rerank
    if pooling != retrieval.POOLING:
        raise RerankError(
            f"rerank has no {pooling!r} pooling: it scores a candidate list "
            f"with a tree ensemble. The pooling belongs to the retriever "
            f"features it reads, which are built at {retrieval.POOLING}."
        )
    booster = load_model(config, spec)
    scores = nrms_scores(config, _split_of(behaviors)) if spec.nrms else None

    parts = []
    for frame in frames_for(config, behaviors, history, spec, history_k):
        if spec.nrms:
            frame = with_nrms(frame, scores)
        parts.append(ranked_from(frame, score_frame(booster, frame, spec), spec))
    return pd.concat(parts, ignore_index=True)


def _split_of(behaviors: pd.DataFrame) -> str:
    """Which split these impressions came from, for the cached NRMS column.

    One split per call: the harness scores a split at a time, and a mixture
    would mean a cache keyed by something that is not a fact about the rows.
    """
    found = set(behaviors["split"].dropna().unique()) if "split" in behaviors else set()
    if len(found) != 1:
        raise RerankError(
            f"expected impressions from exactly one split, found {sorted(found)}"
        )
    return str(next(iter(found)))


def frames_for(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    spec: RerankSpec,
    history_k: int,
):
    """The long frame for these impressions, a chunk at a time, computed.

    `features.frame_for` is the one implementation, so the columns a served
    request is scored on are the columns the model was trained on, built by the
    same code in the same order.
    """
    whole = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    loaded = features.load(config, whole)
    by_impression = history.set_index("impression_id")
    for chunk in features.chunks(behaviors, config.features):
        rows = by_impression.reindex(chunk["impression_id"]).reset_index()
        yield features.frame_for(
            config,
            chunk,
            rows,
            loaded,
            causal=spec.causal,
            history_k=history_k,
        )


@dataclass(frozen=True)
class Ranker:
    """Scores a supplied candidate list against a supplied catalogue.

    The submission's entry, and the same two calls `rank_candidates` makes --
    build the chunk's frame, predict over it once -- so the p50 in the ledger
    describes the path the leaderboard file came out of.

    `build_frame(history, candidates)` is what differs between the offline path
    and the submission: offline, the frame is assembled from the feature store
    and the counter store the pipeline built; for the competition it has to be
    assembled over *its* catalogue, from a log the feature store does not
    contain. That assembly is ticket 08's, so it arrives here as a callable
    rather than being half-written in two places.
    """

    booster: lgb.Booster
    spec: RerankSpec
    config: DatasetConfig
    history_k: int
    build_frame: object

    def rank(self, history: pd.DataFrame, candidates: list[list[str]]) -> pd.DataFrame:
        frame = self.build_frame(history, candidates)
        return ranked_from(frame, score_frame(self.booster, frame, self.spec), self.spec)


def ranker(
    articles: pd.DataFrame,
    config: DatasetConfig,
    workdir: Path,
    history_k: int = retrieval.HISTORY_K,
    build_frame=None,
) -> Ranker:
    """The trained arm, ready to score a supplied catalogue's candidates.

    Same name and same shape as `ann_index.ranker` and `nrms.ranker`, which is
    all `predict` knows about a retriever. The one piece it cannot supply for
    itself is `build_frame`: the competition's impressions are a later period
    over its own catalogue, so their counters, freshness and retriever scores
    come from files the feature store does not hold. Ticket 08 builds that and
    hands it in; asked without one, this refuses rather than quietly assembling
    features from the wrong period's log -- which would produce a
    well-formed submission ranked on numbers that describe another week.
    """
    if build_frame is None:
        raise RerankError(
            "rerank.ranker needs a frame builder for the competition's "
            "catalogue: its counters and freshness come from the test log, "
            "which the feature store does not contain. That is ticket 08 "
            "(`ada/predict-rerank.sbatch`); until it lands, submit with "
            "`--retriever ann` or `--retriever nrms`."
        )
    return Ranker(
        booster=load_model(config),
        spec=config.rerank,
        config=config,
        history_k=history_k,
        build_frame=build_frame,
    )


# ---------------------------------------------------------------------------
# The headline, the ledger and the stage.


def labels_for(impressions: pd.DataFrame, ranked: pd.DataFrame) -> list[dict]:
    """Labels by article id, per impression, in the order `ranked` came back."""
    of = {
        impression: dict(zip(candidates, marks, strict=True))
        for impression, candidates, marks in zip(
            impressions["impression_id"],
            impressions["candidate_ids"],
            impressions["labels"],
        )
    }
    return [of[impression] for impression in ranked["impression_id"]]


def headline(
    config: DatasetConfig,
    spec: RerankSpec,
    split: str = retrieval.VALIDATION,
    against: str = "nrms",
    resamples: int | None = None,
) -> dict:
    """`rerank - against` on one split, paired by impression, with an interval.

    Paired because the two score the same impressions in the same order, so
    subtracting per impression cancels the variance between impressions --
    which on this data is most of the width of an unpaired interval and the
    reason two overlapping intervals cannot settle anything.
    """
    from pipeline import embed_compare, evaluate

    resamples = evaluate.BOOTSTRAP_RESAMPLES if resamples is None else resamples
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == split]
    history = ingest.history_for(config, impressions)

    ours = rank_candidates(config, impressions, history)
    theirs = evaluate.RETRIEVERS[against].rank_candidates(
        config,
        impressions,
        history,
        config.nrms.history_length if against == "nrms" else retrieval.HISTORY_K,
    )
    mine = evaluate.per_impression_metrics(ours, labels_for(impressions, ours))
    others = evaluate.per_impression_metrics(theirs, labels_for(impressions, theirs))

    found = {"against": against, "split": split, "n": int(len(mine["auc"]))}
    for metric in evaluate.ACCURACY_METRICS:
        gap = mine[metric] - others[metric]
        low, high = embed_compare.interval(gap, resamples)
        found[metric] = float(gap.mean()) if len(gap) else 0.0
        found[f"{metric}_lo"], found[f"{metric}_hi"] = low, high
    return found


def record(report: dict, cost: dict, delta: dict | None = None) -> dict:
    row = {
        "dataset": report["dataset"],
        "stage": STAGE,
        "variant": report["variant"],
        "split": report.get("split", TUNE_SPLIT),
        "auc": report.get("tune_auc"),
        "model_bytes": report.get("model_bytes"),
        "train_seconds": round(report["train_seconds"], 2),
        "peak_rss_mb": report.get("peak_rss_mb"),
        "p50_ms": cost.get("p50_ms"),
        "p99_ms": cost.get("p99_ms"),
        "rows_per_s": cost.get("rows_per_s"),
        "note": (
            f"{report['rounds']} rounds over {report['fit_rows']:,} rows "
            f"({report['fit_impressions']:,} impressions), "
            f"{len(report['columns'])} columns"
        ),
    }
    if delta is not None:
        row |= {
            "delta_vs": delta["against"],
            "delta": delta["auc"],
            "delta_lo": delta["auc_lo"],
            "delta_hi": delta["auc_hi"],
        }
    return ledger.record(row)


def tune_auc(booster: lgb.Booster, tune: Rows, spec: RerankSpec) -> float:
    """The arm's AUC on tune, the harness's way: per impression, then averaged.

    Not LightGBM's own `auc`, which pools every candidate of every impression
    into one curve and answers a different question -- whether a click can be
    told from a non-click anywhere in the split, rather than whether this
    impression's candidates were ordered.
    """
    scored = np.asarray(booster.predict(tune.matrix, num_threads=spec.threads))
    found = []
    for _, span in group_spans(tune.impression_ids):
        truth = tune.labels[span]
        if truth.sum() in (0, len(truth)):
            continue
        found.append(roc_auc_score(truth, scored[span]))
    return float(np.mean(found)) if found else float("nan")


def prediction_cost(
    booster: lgb.Booster, tune: Rows, spec: RerankSpec, sample: int = 500
) -> dict:
    """What one request's scoring costs, one impression at a time.

    The serving shape rather than the batched one: a request holds one
    impression's candidates, and the p50 of that is the number ticket 09's
    cost-per-1000-queries is built from.
    """
    from pipeline import bench

    seconds, rows = [], 0
    for _, span in list(group_spans(tune.impression_ids))[:sample]:
        matrix = tune.matrix[span]
        started = time.perf_counter()
        booster.predict(matrix, num_threads=spec.threads)
        seconds.append(time.perf_counter() - started)
        rows += matrix.shape[0]
    return {**bench.percentiles(seconds), "rows_per_s": rows / max(sum(seconds), 1e-9)}


def thread_rows(
    booster: lgb.Booster, tune: Rows, spec: RerankSpec, report: dict
) -> list[dict]:
    """The same model measured on one core and on every core.

    Two rows rather than one because ticket 09's cost per 1000 queries is built
    from a *single-core* QPS: a machine serves many requests at once, so the
    per-request number that matters is the one a single core produces, and
    dividing a multi-core measurement by the core count is arithmetic rather
    than measurement. The model is identical -- only `num_threads` differs --
    so the AUC on both rows is the same number and the latency is not.
    """
    found = []
    for threads in (1, 0):
        if threads == spec.threads:
            continue
        measured = prediction_cost(booster, tune, dataclasses.replace(spec, threads=threads))
        row = {
            **report,
            "variant": f"{report['variant']}-t{threads or 'all'}",
            "note": (
                f"{report['rounds']} rounds, scored with "
                f"{'one core' if threads == 1 else 'every core'}; the same "
                f"model as `{report['variant']}`, so the AUC is its AUC"
            ),
        }
        record(row, measured)
        found.append({**row, "cost": measured})
    return found


def projection_rows(config: DatasetConfig, spec: RerankSpec) -> list[dict]:
    """What the arm's column projection buys, in peak RSS and seconds.

    The tier structure's whole engineering claim: an arm that drops a tier does
    not pay to read it. Measured by reading the same rows twice -- once with
    the projection and once with every column -- rather than asserted, because
    parquet's column layout is what makes it true and a change to the writer
    could quietly stop it being so.
    """
    found = []
    for projected in (True, False):
        arm = spec if projected else dataclasses.replace(
            spec, groups=tuple(features.FEATURE_GROUPS), window=ALL_WINDOWS
        )
        with timings.sample() as measured:
            rows = read_rows(config, TUNE_SPLIT, dataclasses.replace(arm, nrms=False))
        row = {
            "dataset": config.name,
            "stage": STAGE,
            "variant": f"{variant_of(spec)}-read{'' if projected else '-all'}",
            "split": TUNE_SPLIT,
            "peak_rss_mb": round(measured["peak_rss_mb"], 1),
            "train_seconds": round(measured["seconds"], 2),
            "rows_per_s": len(rows.labels) / max(measured["seconds"], 1e-9),
            "note": (
                f"{'the arm' if projected else 'every column'}: "
                f"{rows.matrix.shape[1]} columns over {len(rows.labels):,} rows"
            ),
        }
        ledger.record(row)
        found.append(row)
    return found


def fit_one(
    config: DatasetConfig,
    spec: RerankSpec,
    fit: Rows | None = None,
    tune: Rows | None = None,
    snapshots: tuple[int, ...] = (),
) -> dict:
    """Train one arm, save it, measure it, record it -- and its snapshots."""
    tune = tune or read_rows(config, TUNE_SPLIT, spec)
    with timings.sample() as measured:
        booster, report = train(config, spec, fit, tune)
    report["peak_rss_mb"] = round(measured["peak_rss_mb"], 1)
    report["tune_auc"] = tune_auc(booster, tune, spec)
    path = save(booster, config, spec)
    report["model_bytes"] = path.stat().st_size
    report["path"] = str(path)
    report["cost"] = prediction_cost(booster, tune, spec)
    report["importance"] = importance(booster, spec)
    record(report, report["cost"])
    report["threads"] = thread_rows(booster, tune, spec, report)

    # The rounds curve: the same booster truncated, which is what `num_iteration`
    # means -- three points of AUC against bytes and milliseconds for one fit.
    report["snapshots"] = []
    for rounds in snapshots:
        if rounds >= report["rounds"]:
            continue
        text = booster.model_to_string(num_iteration=rounds)
        truncated = lgb.Booster(model_str=text)
        snapshot = {
            **{key: report[key] for key in ("dataset", "rounds", "columns", "fit_rows", "fit_impressions", "train_seconds")},
            "variant": f"{report['variant']}-r{rounds}",
            "rounds": rounds,
            "tune_auc": tune_auc(truncated, tune, spec),
            "model_bytes": len(text.encode()),
            "cost": prediction_cost(truncated, tune, spec),
        }
        record(snapshot, snapshot["cost"])
        report["snapshots"].append(snapshot)
    return report


def grid(config: DatasetConfig, axes: dict | None = None) -> list[dict]:
    """The registry's spec, then one arm per alternative on each axis.

    One axis at a time from a stated default. The tune rows are re-read per
    arm because the projection is part of the arm -- reading every column once
    and slicing would measure a memory cost no arm actually pays.
    """
    axes = GRID if axes is None else axes
    base = config.rerank
    tried: list[dict] = []
    seen: set[str] = set()
    for axis, values in axes.items():
        for value in values:
            spec = dataclasses.replace(base, **{axis: value})
            if variant_of(spec) in seen:
                continue
            seen.add(variant_of(spec))
            tried.append(fit_one(config, spec))
    return tried


def run(config: DatasetConfig, force: bool = False) -> None:
    """The build stage: the registry's arm, trained and saved."""
    path = model_path(config)
    if not force and path.exists():
        print(f"    rerank {variant_of(config.rerank)} is already trained at {path}")
        return
    report = fit_one(config, config.rerank, snapshots=SNAPSHOTS)
    projection_rows(config, config.rerank)
    print(
        f"    {report['variant']}: tune AUC {report['tune_auc']:.4f} at "
        f"{report['rounds']} rounds, {report['train_seconds']:.0f} s, "
        f"{report['model_bytes'] / 1024:.0f} KB"
    )
    for tier, gain in report["importance"]["tiers"].items():
        print(f"      gain {tier:<10} {gain:12,.0f}")
    ledger.render()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.rerank",
        description="Train the LightGBM re-ranker, or sweep the arms that choose it.",
    )
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--grid", action="store_true", help="every cell of GRID on tune")
    parser.add_argument(
        "--headline",
        action="store_true",
        help="score the registry's arm on validation against nrms, paired",
    )
    parser.add_argument("--resamples", type=int, help="bootstrap resamples")
    args = parser.parse_args(argv)

    for name in args.dataset or DEFAULT_DATASETS:
        config = DATASETS[name]
        if args.grid:
            for report in grid(config):
                print(
                    f"  {name}/{report['variant']}: tune AUC "
                    f"{report['tune_auc']:.4f}, {report['train_seconds']:.0f} s"
                )
            continue
        report = fit_one(config, config.rerank, snapshots=SNAPSHOTS)
        if args.headline:
            delta = headline(config, config.rerank, resamples=args.resamples)
            record({**report, "split": retrieval.VALIDATION}, report["cost"], delta)
            print(
                f"  {name}: rerank - {delta['against']} on {delta['split']}: "
                f"AUC {delta['auc']:+.4f} "
                f"[{delta['auc_lo']:+.4f}, {delta['auc_hi']:+.4f}] "
                f"over {delta['n']:,} paired impressions"
            )
        else:
            print(
                f"  {name}/{report['variant']}: tune AUC {report['tune_auc']:.4f}"
            )
    print(f"-> {ledger.render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
