"""One user request through the whole two-stage path, timed stage by stage.

Every other benchmark in this repository measures a *batch*: `bench` times a
thousand queries against an index, `timings` times a rebuild, `predict` times a
submission run over millions of impressions. None of those is what a served
system does. A request arrives on its own, for one user, and has to come back
inside a budget — and the interesting fact about this system is that the
batched numbers do not predict the single-request ones, because the cost moves
from the stage that vectorises well to the stage that does not.

So this module issues requests one at a time and times four stages inside each:

    retrieve   the semantic index over the catalogue, top-K
    features   the forty columns for those K candidates
    nrms       the user encoder, then the dot against each candidate
    gbdt       one LightGBM predict over the long frame

The four are timed inside one request rather than separately, so their
percentiles and the request's own percentile come from the same requests and
can be compared. The sum of the stage medians against the overall median is
reported as a check on the timer itself: they cannot be equal — a median is
not additive — but they cannot be far apart either, and a gap says the timer
is measuring something the stages are not.

What it produces: `artifacts/bench-serve-<dataset>.md`, a ledger row per
variant, and the three numbers the design note's scaling section is built from.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from pipeline import (
    ann_index,
    bench,
    counters,
    embed,
    features,
    ingest,
    ledger,
    paths,
    retrieval,
    rerank,
)
from pipeline.datasets import DATASETS, DEFAULT_DATASETS, DatasetConfig, RerankSpec

DOCUMENT = "bench-serve-{dataset}.md"
RESULTS = "bench-serve-{dataset}.jsonl"

# The four stages of a request, in the order it passes through them.
STAGES = ("retrieve", "features", "nrms", "gbdt")

# Requests per variant. The ticket asks for at least two thousand, which is
# what a p99 needs to mean anything: at a thousand the 99th percentile is the
# tenth-worst sample and moves visibly between runs.
REQUESTS = 2000

# The budget the cost arithmetic is written against. Stated here rather than
# assumed in prose, because the cost per thousand queries is only a number if
# the latency it is allowed is one too.
SLA_MS = 100.0

# What a core costs, per hour, on demand. AWS c7i.xlarge in eu-west-1 is
# $0.17/hour for 4 vCPU as of 2026-09; a vCPU is a hyperthread, so this is the
# price of a thread, not of a physical core, and the QPS below is measured on
# one thread. Quoted in the markdown so a reader can substitute their own.
CORE_HOUR_USD = 0.0425
CORE_HOUR_SOURCE = "AWS c7i.xlarge on-demand, eu-west-1, $0.17/h over 4 vCPU (2026-09)"

# The retrieval depths the K-curve is drawn at. The same three ticket 07's
# literal-cut arms use, so each K has an AUC and a p99 that describe one
# configuration rather than two neighbouring ones.
KS = (50, 100, 200)

# The node the scaling argument is measured against: what this project was
# actually given. Ada's practical ceiling, from ada/README.md.
NODE_RAM_GB = 100
NODE_CORES = 32


class ServeError(RuntimeError):
    """The benchmark cannot describe the path it was asked to measure."""


# ---------------------------------------------------------------------------
# The stores a request touches, and what each costs.


def store_bytes(config: DatasetConfig, spec: RerankSpec, index=None) -> dict:
    """Bytes of every store one request reads from, or None where absent.

    None rather than zero for a store that is not built. A zero here would
    appear in the note's byte table as a store that costs nothing, which is
    the opposite of what a missing file means -- and the sum of a column with
    a false zero in it is a number nobody can check.

    The feature store is measured as the *materialised frames*, which a served
    request does not read: it computes its features. That is deliberate and is
    the point of the row. The number says what the training-time store costs to
    keep, next to the counter store, which the request does read -- so the note
    can say which of the two a server actually has to hold in memory.
    """
    from pipeline import nrms as nrms_module

    counter_dir = config.artifacts_dir / counters.DIRECTORY
    return {
        "index_bytes": bench.index_bytes(index) if index is not None else None,
        "feature_bytes": _bytes_under(config.feature_store_dir),
        "counter_bytes": _bytes_under(counter_dir),
        "nrms_bytes": _file_bytes(nrms_module.checkpoint_path(config)),
        "model_bytes": _file_bytes(rerank.model_path(config, spec)),
    }


def _file_bytes(path: Path) -> int | None:
    return int(path.stat().st_size) if path.exists() else None


def _bytes_under(directory: Path) -> int | None:
    if not directory.exists():
        return None
    return int(sum(file.stat().st_size for file in directory.rglob("*") if file.is_file()))


def resident_bytes(served: Served) -> dict:
    """What the request path holds *in memory*, as opposed to on disk.

    Separate from `store_bytes` because the two differ in exactly the place
    the note wants to make a claim: the index is resident, the counter store is
    resident, and the feature store is not read at all. A byte table that
    conflated them would support the wrong sentence.
    """
    return {
        "index_resident": bench.index_bytes(served.index),
        "counters_resident": served.loaded.counts.nbytes,
        "vectors_resident": int(served.loaded.embeddings.vectors.nbytes),
        "user_cache_resident": served.cache_bytes(),
    }


# ---------------------------------------------------------------------------
# One request.


def index_for(vectors: np.ndarray, kind: str, precision: str = "fp32"):
    """A stage-one index of the named kind over these vectors.

    `flat`, `ivf` and `hnsw` are the three the A1 bench compares in batch; this
    module runs the same three through the whole two-stage path, because the
    question is not which index is fastest but whether a faster one moves the
    request's p99 at all once stage two is behind it.

    The precision is applied to the vectors before the index is built, through
    `bench.quantise`, so the recall a narrower index loses and the latency it
    saves are measured over the same numbers the ablation scored.
    """
    matrix = np.ascontiguousarray(bench.quantise(vectors, precision), dtype=np.float32)
    dim = matrix.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        return index
    if kind == "ivf":
        return bench.ivf_index(matrix)
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(matrix)
        return index
    raise ServeError(f"no {kind!r} index: one of flat, ivf, hnsw")


@dataclass
class Served:
    """Everything a request needs, opened once.

    The whole point of the benchmark is that this exists *before* the loop: a
    served system loads its index, its counters and its two models at startup
    and pays nothing per request for them. A benchmark that opened a file
    inside the loop would be measuring startup and calling it latency, which
    is why `test_the_request_path_opens_no_file` exists.
    """

    config: DatasetConfig
    spec: RerankSpec
    index: faiss.Index
    embeddings: embed.Embeddings
    loaded: features.Loaded
    booster: object
    nrms_model: object
    nrms_spec: object
    history_k: int
    k: int | None
    # One user vector per user id, when the arm caches them. A request from a
    # user already in it skips the user encoder entirely, which is the half of
    # the NRMS cost a cache can actually remove -- the candidate dot is per
    # candidate and stays.
    cache: dict | None = None
    scorers: dict = field(default_factory=dict)

    def cache_bytes(self) -> int:
        """What the per-user cache holds, so the note can price it at the
        dataset's user count rather than at the sample's."""
        if not self.cache:
            return 0
        return int(
            sum(
                vector.numel() * vector.element_size()
                for vector in self.cache.values()
            )
        )


def build(
    config: DatasetConfig,
    spec: RerankSpec | None = None,
    kind: str = "flat",
    precision: str = "fp32",
    k: int | None = None,
    cache_users: bool = False,
    threads: int = 1,
) -> Served:
    """Open every store once, the way a server would at startup."""
    from pipeline import nrms as nrms_module

    spec = spec or config.rerank
    embeddings = embed.load(config)
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    articles = pd.read_parquet(config.feature_store_dir / "articles.parquet")
    model, nrms_spec, _ = nrms_module.load_model(config)
    booster = rerank.load_model(config, spec)
    # The thread count is the model's, not the process's: the cost arithmetic
    # divides by cores, so a model quietly using all of them would price a
    # request at one core and run it on thirty-two.
    booster.params["num_threads"] = threads
    return Served(
        config=config,
        spec=spec,
        index=index_for(embeddings.vectors, kind, precision),
        embeddings=embeddings,
        loaded=features.load(config, behaviors),
        booster=booster,
        nrms_model=model,
        nrms_spec=nrms_spec,
        history_k=nrms_spec.history_length,
        k=k,
        cache={} if cache_users else None,
        scorers=served_scorers(
            config, embeddings, articles, nrms_spec.history_length
        ),
    )


def served_scorers(
    config: DatasetConfig,
    embeddings: embed.Embeddings,
    articles: pd.DataFrame,
    history_k: int,
) -> dict:
    """The two retriever-score features, with their stores opened once.

    `features.module_scorers` calls each module's `rank_candidates`, which
    loads its own index inside the call. That is right for the frame builder,
    which is handed a hundred thousand impressions at a time, and it is the
    *entire* cost of a single request -- MIND's BM25 index is re-read from
    five files per request, and a benchmark built that way would report the
    read as the feature stage's latency.

    So the index is opened here and handed to the same `rank_candidates` the
    harness calls. Not a second scoring path: the argument exists precisely so
    there is one, and `test_the_served_scorers_agree_with_the_harness` compares
    them.
    """
    from pipeline import bm25_index

    opened = {
        "ann_index": ann_index.build(embeddings),
        "bm25_index": bm25_index.load(config.artifacts_dir / "bm25"),
        "articles": articles,
    }
    return {
        name: (
            lambda chunk, history, module=module: module.rank_candidates(
                config, chunk, history, history_k, stores=opened
            )
        )
        for name, module in features.SCORERS.items()
    }


def candidates_for(served: Served, chunk: pd.DataFrame) -> list[str]:
    """Stage one: the catalogue's top-K for this user, or the logged list.

    With no K the impression's own candidates are used, which is what the
    harness scores and is the curve's ceiling -- every candidate the log holds,
    none of them cut. With a K the index is searched and the impression's
    candidates are *not* consulted, because that is what a served system has:
    a user, a catalogue, and no list of things it was going to be shown.
    """
    if served.k is None:
        return list(chunk["candidate_ids"].iloc[0])
    queries, _ = ann_index.build_user_vectors(chunk, served.embeddings)
    vector = queries.vectors[0]
    if not vector.any():
        # A cold user has nothing to search with. Their request still costs the
        # rest of the path, and dropping them here would report a latency for
        # traffic that does not exist.
        return list(chunk["candidate_ids"].iloc[0])
    _, positions = served.index.search(
        np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32),
        min(served.k, len(served.embeddings.article_ids)),
    )
    return [str(article) for article in served.embeddings.article_ids[positions[0]]]


def one(served: Served, chunk: pd.DataFrame) -> dict[str, float]:
    """One request, four stages, wall seconds each.

    `chunk` is a one-row frame carrying the user's history, which is the shape
    every stage below already takes -- so this is the real path at a batch size
    of one rather than a reimplementation of it at that size.
    """
    from pipeline import nrms as nrms_module

    timed: dict[str, float] = {}
    with features.measured(timed, "retrieve"):
        candidates = candidates_for(served, chunk)

    asked = chunk.assign(candidate_ids=[list(candidates)])
    if "n_clicks" not in asked:
        asked = asked.assign(n_clicks=asked["click_history"].map(len))

    with features.measured(timed, "features"):
        frame = features.frame_for(
            served.config,
            asked,
            asked,
            served.loaded,
            causal=served.spec.causal,
            history_k=served.history_k,
            scorers=served.scorers,
        )

    with features.measured(timed, "nrms"):
        frame[rerank.NRMS_COLUMN] = _nrms_scores(served, asked, candidates)

    with features.measured(timed, "gbdt"):
        rerank.score_frame(served.booster, frame, served.spec)

    timed["total"] = sum(timed[stage] for stage in STAGES)
    return timed


def _nrms_scores(served: Served, chunk: pd.DataFrame, candidates: list[str]):
    """The candidate scores, with the user encoding cached where the arm says.

    Written as the two halves rather than as one `rank` call, because only one
    of them is cacheable and the note's claim is about exactly that: the user
    encoder runs once per request and a cache removes it entirely for a repeat
    user; the candidate dot runs once per *candidate* and no cache touches it.
    A single call could only report their sum, which would make a cache look
    like it halves the stage.

    This is `Nrms.forward` split at its own seam -- `user(news(clicked), mask)`
    then the einsum -- not a reimplementation of it. If the model's forward
    changes shape, this stops compiling rather than quietly measuring the old
    one.
    """
    import torch

    from pipeline import nrms as nrms_module

    model = served.nrms_model
    user_id = str(chunk["user_id"].iloc[0])
    vector = None if served.cache is None else served.cache.get(user_id)

    with torch.no_grad():
        if vector is None:
            rows, mask = nrms_module.history_rows(
                chunk, served.embeddings.index, served.nrms_spec.history_length
            )
            clicked = nrms_module.gather(served.embeddings.vectors, rows, mask)
            vector = model.user(model.news(clicked), torch.from_numpy(mask))
            if served.cache is not None:
                served.cache[user_id] = vector
        rows_of, mask_of = nrms_module.candidate_rows(
            [list(candidates)], served.embeddings.index, max(len(candidates), 1)
        )
        matrix = nrms_module.gather(served.embeddings.vectors, rows_of, mask_of)
        scored = torch.einsum("bch,bh->bc", model.news(matrix), vector)
    return [float(value) for value in scored[0][: len(candidates)]]


# ---------------------------------------------------------------------------
# Many requests.


def requests_from(config: DatasetConfig, split: str, count: int) -> list[pd.DataFrame]:
    """`count` single-impression frames, each carrying its own user's history.

    Materialised up front, so the loop that times the requests does no pandas
    slicing between them: a `.iloc` inside the timed region is a millisecond
    of this benchmark's own overhead attributed to the first stage.
    """
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    scored = behaviors[behaviors["split"] == split].head(count).reset_index(drop=True)
    if scored.empty:
        raise ServeError(f"{config.name}: no impressions on {split!r} to serve")
    history = ingest.history_for(config, scored)
    merged = scored.copy()
    for column in history.columns:
        if column not in merged:
            merged[column] = history[column].to_numpy()
    return [merged.iloc[[row]].reset_index(drop=True) for row in range(len(merged))]


def measure(served: Served, requests: list[pd.DataFrame], warmup: int = 5) -> dict:
    """Per-stage and overall percentiles over these requests.

    The first few are dropped: the first request through any of these paths
    pays for a lazily-built lookup somewhere below it, and a p99 over two
    thousand samples is not moved by five of them but a p50 over a hundred
    would be.
    """
    for chunk in requests[:warmup]:
        one(served, chunk)

    timed = [one(served, chunk) for chunk in requests[warmup:]]
    if not timed:
        raise ServeError("every request was a warm-up; ask for more than the warmup")

    summary = {}
    for stage in (*STAGES, "total"):
        for name, value in bench.percentiles([row[stage] for row in timed]).items():
            summary[f"{stage}_{name}"] = round(value, 4)
    summary["requests"] = len(timed)
    # The timer's own sanity check. Medians are not additive, so these cannot
    # be equal; a large gap means the request is spending time somewhere none
    # of the four stages is looking.
    summary["stage_median_sum_ms"] = round(
        sum(summary[f"{stage}_p50_ms"] for stage in STAGES), 4
    )
    summary["unattributed_ms"] = round(
        summary["total_p50_ms"] - summary["stage_median_sum_ms"], 4
    )
    return summary


# ---------------------------------------------------------------------------
# What it costs to run.


def cost_per_1000(
    p99_ms: float,
    qps_per_core: float,
    sla_ms: float = SLA_MS,
    core_hour_usd: float = CORE_HOUR_USD,
) -> dict:
    """Dollars per thousand queries, with the arithmetic kept alongside.

    Two facts and one division, and the reason the arithmetic is returned as a
    string is that a number in a design note that a reader cannot reproduce is
    a number a reader has to trust.

    A variant whose p99 exceeds the SLA does not get a cost: it does not meet
    the budget the price was quoted at, and pricing it anyway would put a
    cheaper row next to a compliant one without saying the cheap one is not
    allowed.
    """
    within = p99_ms <= sla_ms
    core_seconds = 1000 / qps_per_core if qps_per_core > 0 else float("inf")
    usd = core_seconds / 3600 * core_hour_usd
    return {
        "meets_sla": within,
        "sla_ms": sla_ms,
        "p99_ms": round(p99_ms, 3),
        "qps_per_core": round(qps_per_core, 2),
        "core_seconds_per_1000": round(core_seconds, 2),
        "usd_per_1000": usd if within else None,
        "arithmetic": (
            f"1000 queries / {qps_per_core:.2f} q/s = {core_seconds:.2f} core-seconds"
            f" = {core_seconds / 3600:.5f} core-hours x ${core_hour_usd}/core-hour"
            f" = ${usd:.6f}"
            + ("" if within else f"  (not priced: p99 {p99_ms:.1f} ms > {sla_ms:.0f} ms)")
        ),
    }


def scaling(
    bytes_: dict,
    qps_per_core: float,
    users: int,
    target_qps: float = 1000.0,
    node_ram_gb: float = NODE_RAM_GB,
    node_cores: int = NODE_CORES,
) -> list[dict]:
    """The three 10x rows, each with what grows and what it grows into.

    Not a projection of one number ten times. The three axes move different
    stores, which is the whole content of the claim:

    * **10x users** grows the per-user cache and the history table. The index
      does not move: more readers do not make more articles.
    * **10x catalogue** grows the index, the counter store and the candidate
      frame -- everything keyed by article.
    * **10x QPS** grows neither; it grows cores, and it is the only one of the
      three that a bigger node does not fix.

    Each row names what it multiplies and reports the result against the node
    this project was actually given, so "breaks first" is a comparison and not
    an adjective.
    """
    index = bytes_.get("index_resident") or 0
    counter = bytes_.get("counters_resident") or 0
    vectors = bytes_.get("vectors_resident") or 0
    # Per *user*, not the sample's total: the projection is "this many bytes
    # each, times ten times as many users", and a total measured over whichever
    # users the sample happened to contain would scale the sample instead.
    per_user = bytes_.get("cache_bytes_per_user") or 0.0
    by_article = index + vectors + counter

    # Cores needed to serve `target_qps` at the measured per-core throughput,
    # and again at ten times the load.
    cores_now = _cores_for(target_qps, qps_per_core)

    rows = [
        {
            "axis": "10x users",
            "grows": "per-user NRMS cache, history table",
            "gb_after": (by_article + 10 * per_user * users) / 1e9,
            "cores_after": cores_now,
        },
        {
            "axis": "10x catalogue",
            "grows": "stage-one index, article vectors, counter store",
            "gb_after": (10 * by_article + per_user * users) / 1e9,
            "cores_after": cores_now,
        },
        {
            "axis": "10x QPS",
            "grows": "cores only — no store is keyed by request rate",
            "gb_after": (by_article + per_user * users) / 1e9,
            "cores_after": _cores_for(10 * target_qps, qps_per_core),
        },
    ]
    for row in rows:
        row["gb_after"] = round(row["gb_after"], 3)
        row["fits_ram"] = row["gb_after"] <= node_ram_gb
        row["fits_cores"] = row["cores_after"] <= node_cores
        # How many more times over the axis could grow before the node runs
        # out, past the 10x already applied. None where nothing grows with it.
        row["headroom_x"] = (
            round(node_ram_gb / row["gb_after"], 1) if row["gb_after"] > 0 else None
        )
    return rows


def _cores_for(qps: float, qps_per_core: float) -> int:
    """Cores needed to serve `qps` at the measured per-core throughput."""
    if qps_per_core <= 0:
        return 0
    return int(np.ceil(qps / qps_per_core))


def qps_per_core(summary: dict) -> float:
    """Queries per second one core sustains, from the measured median.

    The median rather than the mean: a throughput computed from a mean is
    dragged by the tail the p99 already reports, and the two would then
    disagree about the same run. Measured at `num_threads=1` -- `build` sets it
    -- because the cost arithmetic divides by cores and a multi-core median
    divided by cores is not a per-core number.
    """
    median_ms = summary["total_p50_ms"]
    return 1000 / median_ms if median_ms > 0 else float("inf")


# ---------------------------------------------------------------------------
# The K curve, joined to the arm that scored it.


def k_curve(config: DatasetConfig, rows: list[dict]) -> list[dict]:
    """Each K's p99 from here, beside its AUC from ticket 07's cut arm.

    Read out of the ledger rather than recomputed: the AUC belongs to the
    ablation, which scored it on `validation` against the full model, and a
    second computation here would be a second chance for the two tables to
    disagree about one configuration.

    A K with no ablation row gets None for its AUC and says so. That is the
    state of a checkout where ticket 07 has not been run, and it is better than
    an empty row that reads as "no difference".
    """
    scored = {
        row["variant"]: row
        for row in ledger.load()
        if row["dataset"] == config.name and row["stage"] == "ablation"
    }
    curve = []
    for row in rows:
        if row.get("k") is None:
            continue
        arm = scored.get(f"cut@{row['k']}")
        curve.append(
            {
                "k": row["k"],
                "p50_ms": row["total_p50_ms"],
                "p99_ms": row["total_p99_ms"],
                "auc": None if arm is None else arm.get("auc"),
                "auc_lo": None if arm is None else arm.get("auc_lo"),
                "auc_hi": None if arm is None else arm.get("auc_hi"),
            }
        )
    return curve


# ---------------------------------------------------------------------------
# The variants, the ledger and the markdown.


# Each is one ledger row: a name, and what differs about the path it measures.
# Swept one axis at a time from the same default, like every other grid in this
# project -- the budget is an afternoon, and a full cross would be 54 cells
# whose interesting cases are all on the axes.
VARIANTS = (
    {"name": "flat-fp32-k100", "kind": "flat", "precision": "fp32", "k": 100},
    {"name": "flat-fp32-k50", "kind": "flat", "precision": "fp32", "k": 50},
    {"name": "flat-fp32-k200", "kind": "flat", "precision": "fp32", "k": 200},
    {"name": "flat-fp32-nocut", "kind": "flat", "precision": "fp32", "k": None},
    {"name": "ivf-fp32-k100", "kind": "ivf", "precision": "fp32", "k": 100},
    {"name": "hnsw-fp32-k100", "kind": "hnsw", "precision": "fp32", "k": 100},
    {"name": "flat-fp16-k100", "kind": "flat", "precision": "fp16", "k": 100},
    {"name": "flat-int8-k100", "kind": "flat", "precision": "int8", "k": 100},
    {"name": "flat-fp32-k100-cached", "kind": "flat", "precision": "fp32", "k": 100,
     "cache_users": True},
    {"name": "flat-fp32-k100-threads", "kind": "flat", "precision": "fp32", "k": 100,
     "threads": 0},
)


def run_variant(
    config: DatasetConfig,
    variant: dict,
    requests: list[pd.DataFrame],
    split: str,
    warmup: int = 5,
) -> dict:
    """One variant measured and recorded."""
    served = build(
        config,
        kind=variant["kind"],
        precision=variant["precision"],
        k=variant["k"],
        cache_users=variant.get("cache_users", False),
        threads=variant.get("threads", 1),
    )
    summary = measure(served, requests, warmup)
    row = {
        "dataset": config.name,
        "split": split,
        "variant": variant["name"],
        "k": variant["k"],
        **summary,
        **store_bytes(config, served.spec, served.index),
        **resident_bytes(served),
        # How many users the cache ended up holding, which is the denominator
        # that turns its bytes into a per-user cost.
        "cached_users": 0 if served.cache is None else len(served.cache),
    }
    row["qps_per_core"] = round(qps_per_core(summary), 2)
    row.update(
        {
            f"cost_{name}": value
            for name, value in cost_per_1000(
                summary["total_p99_ms"], row["qps_per_core"]
            ).items()
        }
    )
    record(config, row)
    print(
        f"    {variant['name']:<24} p50 {summary['total_p50_ms']:7.2f} ms  "
        f"p99 {summary['total_p99_ms']:7.2f} ms  "
        f"{row['qps_per_core']:8.1f} q/s/core",
        flush=True,
    )
    return row


def record(config: DatasetConfig, row: dict) -> dict:
    """One ledger row per variant, engineering columns only.

    The functional column comes from ticket 07 and is joined in the K-curve,
    not copied onto this row: a latency benchmark that also carried an AUC
    would be a second place for that AUC to live.
    """
    return ledger.record(
        {
            "dataset": config.name,
            "stage": "serve",
            "variant": row["variant"],
            "split": row["split"],
            "p50_ms": row["total_p50_ms"],
            "p99_ms": row["total_p99_ms"],
            "rows_per_s": row["qps_per_core"],
            "index_bytes": row.get("index_bytes"),
            "feature_bytes": row.get("feature_bytes"),
            "model_bytes": row.get("model_bytes"),
            "note": (
                f"retrieve {row['retrieve_p50_ms']:.2f} / "
                f"features {row['features_p50_ms']:.2f} / "
                f"nrms {row['nrms_p50_ms']:.2f} / "
                f"gbdt {row['gbdt_p50_ms']:.2f} ms at p50; "
                + (
                    f"${row['cost_usd_per_1000']:.4f}/1k queries"
                    if row.get("cost_usd_per_1000") is not None
                    else f"over the {SLA_MS:.0f} ms budget, not priced"
                )
            ),
        }
    )


def results_path(config: DatasetConfig) -> Path:
    return paths.ARTIFACTS_DIR / RESULTS.format(dataset=config.name)


def document(config: DatasetConfig, rows: list[dict], curve: list[dict],
             scale: list[dict]) -> str:
    """The markdown, assembled from the rows rather than written beside them."""
    lines = [
        f"# Serving one request: {config.name}",
        "",
        "Single-user requests through the real two-stage path, timed stage by "
        "stage inside each request. Percentiles are over "
        f"{rows[0]['requests'] if rows else 0:,} requests per variant.",
        "",
        "## Per stage",
        "",
        "| variant | retrieve | features | nrms | gbdt | total p50 | total p99 | q/s/core |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | "
            + " | ".join(
                f"{row[f'{stage}_p50_ms']:.2f}" for stage in STAGES
            )
            + f" | {row['total_p50_ms']:.2f} | {row['total_p99_ms']:.2f} "
            f"| {row['qps_per_core']:.1f} |"
        )
    lines += [
        "",
        "Stage medians do not sum to the total median -- a median is not "
        "additive -- but the gap is the timer's own check: "
        + (
            f"{rows[0]['unattributed_ms']:.3f} ms unattributed at p50 on the "
            f"first row."
            if rows
            else "no rows."
        ),
        "",
        "## Bytes a request touches",
        "",
        "| store | bytes | resident per process |",
        "|---|---|---|",
    ]
    if rows:
        first = rows[0]
        for label, on_disk, resident in (
            ("stage-one index", "index_bytes", "index_resident"),
            ("article vectors", None, "vectors_resident"),
            ("counter store", "counter_bytes", "counters_resident"),
            ("NRMS checkpoint", "nrms_bytes", None),
            ("GBDT model", "model_bytes", None),
            ("feature store (training only)", "feature_bytes", None),
        ):
            lines.append(
                f"| {label} | {ledger.cell(first.get(on_disk), 'bytes') if on_disk else '—'} "
                f"| {ledger.cell(first.get(resident), 'bytes') if resident else '—'} |"
            )
    lines += [
        "",
        f"The feature store is listed because it has to be *kept*, not because "
        f"a request reads it: a served request computes its features. What a "
        f"server holds resident is the index, the vectors and the counters.",
        "",
        "## K against quality and latency",
        "",
        "| K | AUC (ticket 07's cut arm) | p50 ms | p99 ms |",
        "|---|---|---|---|",
    ]
    for point in curve:
        auc = "—" if point["auc"] is None else f"{point['auc']:.4f}"
        lines.append(
            f"| {point['k']} | {auc} | {point['p50_ms']:.2f} | {point['p99_ms']:.2f} |"
        )
    if curve and all(point["auc"] is None for point in curve):
        lines.append("")
        lines.append(
            "The AUC column is outstanding: it comes from `python -m "
            "pipeline.ablation --split validation`, which has not been run "
            "against a built feature store."
        )
    lines += [
        "",
        "## Cost per thousand queries",
        "",
        f"Budget: p99 < {SLA_MS:.0f} ms. Core price: ${CORE_HOUR_USD}/core-hour "
        f"({CORE_HOUR_SOURCE}).",
        "",
        "| variant | q/s/core | meets budget | $/1k queries | arithmetic |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        usd = row.get("cost_usd_per_1000")
        lines.append(
            f"| {row['variant']} | {row['qps_per_core']:.1f} | "
            f"{'yes' if row.get('cost_meets_sla') else 'no'} | "
            f"{'—' if usd is None else f'${usd:.4f}'} | "
            f"{row.get('cost_arithmetic', '')} |"
        )
    lines += [
        "",
        "## Where 10x breaks",
        "",
        f"Against the node this ran on: {NODE_RAM_GB} GB, {NODE_CORES} cores.",
        "",
        "| axis | what grows | GB after | cores after | fits |",
        "|---|---|---|---|---|",
    ]
    for row in scale:
        fits = "yes" if row["fits_ram"] and row["fits_cores"] else "**no**"
        lines.append(
            f"| {row['axis']} | {row['grows']} | {row['gb_after']:.2f} | "
            f"{row['cores_after']} | {fits} |"
        )
    broken = [row["axis"] for row in scale if not (row["fits_ram"] and row["fits_cores"])]
    lines += [
        "",
        (
            f"First to exceed the node: **{broken[0]}**."
            if broken
            else "Nothing exceeds the node at 10x on any of the three axes."
        ),
        "",
    ]
    return "\n".join(lines)


def run(
    config: DatasetConfig,
    split: str = "validation",
    count: int = REQUESTS,
    variants: tuple = VARIANTS,
) -> list[dict]:
    """Every variant, the ledger rows, and the markdown."""
    requests = requests_from(config, split, count)
    print(f"    {len(requests):,} single-user requests from {split}")

    rows = [run_variant(config, variant, requests, split) for variant in variants]

    # The resident bytes come off the rows that were just measured rather than
    # from a fresh `build`: the cache column is only non-zero on the arm that
    # kept one, and a freshly built Served has an empty cache, which would
    # report a per-user cost of nothing.
    base = rows[0]
    cached = next((row for row in rows if row["variant"].endswith("-cached")), base)
    users = int(
        pd.read_parquet(
            config.feature_store_dir / "history.parquet", columns=["user_id"]
        ).shape[0]
    )
    held = cached.get("cached_users") or 0
    scale = scaling(
        {
            "index_resident": base["index_resident"],
            "counters_resident": base["counters_resident"],
            "vectors_resident": base["vectors_resident"],
            "cache_bytes_per_user": (
                cached["user_cache_resident"] / held if held else 0.0
            ),
        },
        base["qps_per_core"],
        users=users,
    )
    curve = k_curve(config, rows)

    results = results_path(config)
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (paths.ARTIFACTS_DIR / DOCUMENT.format(dataset=config.name)).write_text(
        document(config, rows, curve, scale), encoding="utf-8"
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--requests",
        type=int,
        default=REQUESTS,
        help=f"requests per variant (default {REQUESTS:,}; a p99 needs them)",
    )
    args = parser.parse_args(argv)

    for name in args.dataset or DEFAULT_DATASETS:
        print(f"  {name}")
        run(DATASETS[name], args.split, args.requests)
    ledger.render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
