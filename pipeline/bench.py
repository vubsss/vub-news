"""What the tool choices cost: index, backend, serving path, precision.

    python -m pipeline.bench --dataset mind --bench all

Every other comparison in this project chooses a *setting* by AUC. This one
measures the things AUC cannot see, because two of the system's choices were
made by argument rather than by measurement and the arguments read exactly like
evidence:

  * `ann_index.build` takes an exact flat inner-product index, on the grounds
    that "approximate search would buy nothing" at 20-65k articles. Nothing was
    ever timed. `--bench ann` times it, and prices the alternative in build
    seconds, resident bytes, single-query latency and the recall it loses.
  * `SPEC.md` chose `bm25s` because it is "faster than rank_bm25". Nothing was
    ever timed. `--bench lexical` times both and checks they agree.

Two further benchmarks exist because the accuracy tables hide a recurring cost:

  * `--bench serve` measures the path a submission actually runs -- scoring an
    impression's own candidate list -- rather than the batch stages
    `pipeline.timings` already reports. A stage's wall time answers "what does
    a rebuild cost"; this answers "what does one impression cost", which is the
    number that decides whether the system can serve.
  * `--bench precision` prices fp16 and 8-bit vectors against fp32, in bytes
    and milliseconds, against the AUC they lose.

**The index and the ranking path are not the same path.** `rank_candidates`
scores an impression's supplied candidates with a dense product against the
rows it looks up by id; it never queries the FAISS index. The index serves
corpus retrieval -- the recall@K table -- alone. So an approximate index can
cost recall and *cannot* cost AUC, and the two benchmarks report accordingly.
"""

from __future__ import annotations

import argparse
import json
import time

import faiss
import numpy as np
import pandas as pd

from pipeline import ann_index, bm25_index, embed, evaluate, ingest, paths, retrieval
from pipeline.datasets import DATASETS, DatasetConfig

RESULTS = "bench-{kind}-{dataset}.jsonl"
DOCUMENT = "bench-{kind}-{dataset}.md"

# Single-query latency is sampled rather than measured over every impression:
# the distribution is what is wanted and it converges long before the corpus
# does. Batch throughput uses the same sample so the two are comparable.
LATENCY_SAMPLE = 500
RECALL_DEPTH = 10


def percentiles(values: list[float]) -> dict[str, float]:
    """p50/p95/p99 in milliseconds. A mean would hide the tail, and the tail is
    what a serving budget is written against."""
    array = np.asarray(values, dtype=float) * 1000
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def index_bytes(index: faiss.Index) -> int:
    """What the index costs in memory, via the serialised form.

    FAISS does not expose its own footprint, and `sys.getsizeof` sees a SWIG
    pointer. Serialising writes exactly the structure the index holds, so its
    length is the honest answer for every index type at once -- which is the
    property that matters here, since the point is to compare them.
    """
    return int(faiss.serialize_index(index).nbytes)


def user_vectors(config: DatasetConfig, split: str) -> np.ndarray:
    """The queries a real run would issue, on the split being reported."""
    embeddings = embed.load(config)
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    scored = behaviors[behaviors["split"] == split]
    history = ingest.history_for(config, scored)
    queries, _ = ann_index.build_user_vectors(history, embeddings)
    # A cold start has no vector to search with. It is a real share of traffic
    # and it is reported by the harness, but it costs no index time, so
    # including its zero rows here would deflate the latency being measured.
    asked = queries.vectors[np.linalg.norm(queries.vectors, axis=1) > 0]
    return np.ascontiguousarray(asked, dtype=np.float32)


def search_latency(index: faiss.Index, sample: np.ndarray) -> dict[str, float]:
    """One query at a time, which is how a request arrives."""
    timings = []
    for row in sample:
        query = row.reshape(1, -1)
        started = time.perf_counter()
        index.search(query, RECALL_DEPTH)
        timings.append(time.perf_counter() - started)
    return percentiles(timings)


# nlist by the usual sqrt(n) rule and nprobe at about 8% of the lists: the
# setting a deployment would reach for before it tuned anything. One definition,
# because ticket 07 scores the recall this index loses and this module measures
# the latency it saves -- and those two numbers have to describe one index.
NPROBE_SHARE = 12


def ivf_index(vectors: np.ndarray):
    """The approximate index both the scale bench and the ablation use."""
    n, dim = vectors.shape
    nlist = int(np.sqrt(n))
    index = faiss.IndexIVFFlat(
        faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT
    )
    index.train(vectors)
    index.add(vectors)
    index.nprobe = max(1, nlist // NPROBE_SHARE)
    return index


def bench_ann(config: DatasetConfig, split: str) -> list[dict]:
    """Flat, IVF, HNSW and a plain matmul over the same vectors.

    The exact index is built first and its top-`RECALL_DEPTH` kept, so every
    approximate variant is scored against what it was meant to approximate
    rather than against ground truth it was never asked for.
    """
    embeddings = embed.load(config)
    vectors = np.ascontiguousarray(embeddings.vectors, dtype=np.float32)
    n, dim = vectors.shape
    queries = user_vectors(config, split)
    sample = queries[:LATENCY_SAMPLE]
    print(f"    {n:,} articles x {dim}d, {len(sample):,} sampled queries")

    exact = faiss.IndexFlatIP(dim)
    exact.add(vectors)
    _, truth = exact.search(sample, RECALL_DEPTH)

    rows: list[dict] = []
    for name in ("flat", "ivf", "hnsw", "numpy"):
        started = time.perf_counter()
        if name == "flat":
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)
        elif name == "ivf":
            index = ivf_index(vectors)
        elif name == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.add(vectors)
        else:
            index = None
        build_seconds = time.perf_counter() - started

        if index is None:
            # The comparison the flat index is really competing with: no index
            # object at all, one BLAS call, argpartition for the top rows.
            timings = []
            for row in sample:
                started = time.perf_counter()
                scores = vectors @ row
                np.argpartition(-scores, RECALL_DEPTH)[:RECALL_DEPTH]
                timings.append(time.perf_counter() - started)
            latency = percentiles(timings)
            scores = vectors @ sample.T
            found = np.argpartition(-scores, RECALL_DEPTH, axis=0)[:RECALL_DEPTH].T
            footprint = int(vectors.nbytes)
        else:
            latency = search_latency(index, sample)
            _, found = index.search(sample, RECALL_DEPTH)
            footprint = index_bytes(index)

        overlap = [
            len(set(a.tolist()) & set(b.tolist())) / RECALL_DEPTH
            for a, b in zip(truth, found)
        ]
        row = {
            "dataset": config.name,
            "split": split,
            "index": name,
            "articles": int(n),
            "dim": int(dim),
            "build_seconds": round(build_seconds, 2),
            "index_mb": round(footprint / 1e6, 1),
            "recall_vs_exact": float(np.mean(overlap)),
            **{k: round(v, 3) for k, v in latency.items()},
        }
        rows.append(row)
        print(
            f"    {name:6} build {row['build_seconds']:7.2f}s  "
            f"{row['index_mb']:7.1f} MB  p50 {row['p50_ms']:6.3f} ms  "
            f"p95 {row['p95_ms']:6.3f} ms  recall@{RECALL_DEPTH} vs exact "
            f"{row['recall_vs_exact']:.4f}",
            flush=True,
        )
        del index
    return rows


# rank_bm25 is not in environment.yml: it is here to be measured against, not
# to be depended on. Absent, the benchmark reports the backend it has.
try:  # pragma: no cover - the comparison is optional
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

# rank_bm25 scores one query against the whole corpus in Python. At the shipped
# window the query is 80 titles, so a handful of queries is a minute of work
# and is enough to establish an order of magnitude.
LEXICAL_QUERIES = 25


# The catalogue tiled to multiples of itself. Tiling repeats vectors, which
# moves the *distribution* and not the work: a flat scan costs n x d whatever
# the rows say, and this benchmark measures cost. Stopping at 8x is a memory
# limit, not a judgement -- flat, IVF and the source matrix are three copies.
SCALE_STEPS = (1, 2, 4, 8)
SCALE_QUERIES = 200

# What a rerank can spend on retrieval before it stops being a background cost.
# Nothing in the assignment sets a budget; this is the round number the crossing
# is reported against, and the table gives the latencies to read it differently.
BUDGET_MS = 10.0


def bench_scale(config: DatasetConfig, split: str) -> list[dict]:
    """Where the exact index stops being free.

    The shipped choice is exact search, on the argument that at 20-65k articles
    approximation buys nothing worth its recall. That argument has a boundary
    and the note should name it rather than leave it as a property of the two
    catalogues that happened to be in front of us. So the catalogue is tiled to
    multiples of itself and the same three searches are timed at each size.
    """
    embeddings = embed.load(config)
    base = np.ascontiguousarray(embeddings.vectors, dtype=np.float32)
    dim = base.shape[1]
    sample = user_vectors(config, split)[:SCALE_QUERIES]

    rows: list[dict] = []
    for step in SCALE_STEPS:
        vectors = base if step == 1 else np.ascontiguousarray(np.tile(base, (step, 1)))
        n = vectors.shape[0]
        for name in ("flat", "ivf", "numpy"):
            if name == "numpy":
                timings = []
                for row in sample:
                    started = time.perf_counter()
                    scores = vectors @ row
                    np.argpartition(-scores, RECALL_DEPTH)[:RECALL_DEPTH]
                    timings.append(time.perf_counter() - started)
                latency, build_seconds = percentiles(timings), 0.0
            else:
                started = time.perf_counter()
                if name == "flat":
                    index = faiss.IndexFlatIP(dim)
                    index.add(vectors)
                else:
                    index = ivf_index(vectors)
                build_seconds = time.perf_counter() - started
                latency = search_latency(index, sample)
                del index
            row = {
                "dataset": config.name,
                "split": split,
                "step": f"{step}x",
                "articles": int(n),
                "dim": int(dim),
                "index": name,
                "build_seconds": round(build_seconds, 2),
                **{k: round(v, 3) for k, v in latency.items()},
            }
            rows.append(row)
            print(
                f"    {step}x {n:>9,} {name:6} build {row['build_seconds']:7.2f}s  "
                f"p50 {row['p50_ms']:8.3f} ms  p95 {row['p95_ms']:8.3f} ms",
                flush=True,
            )
        if step != 1:
            del vectors
    return rows


def bench_lexical(config: DatasetConfig, split: str) -> list[dict]:
    """bm25s against rank_bm25 on the same corpus and the same queries.

    Both are asked for the same top-`RECALL_DEPTH`, and the overlap is
    reported: a backend that is faster because it scores differently is not a
    substitute, and the timing would be meaningless without that check.
    """
    articles = pd.read_parquet(config.feature_store_dir / "articles.parquet")
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    scored = behaviors[behaviors["split"] == split]
    history = ingest.history_for(config, scored)
    built, _ = bm25_index.build_queries(history, articles, config)
    text = [query for query in built["query"] if query][:LEXICAL_QUERIES]
    corpus = [document.split() for document in articles["lexical_text"].fillna("")]
    print(f"    {len(corpus):,} documents, {len(text):,} queries")

    rows: list[dict] = []
    started = time.perf_counter()
    fast = bm25_index.build(articles, config)
    fast_build = time.perf_counter() - started

    timings, fast_top = [], []
    for query in text:
        started = time.perf_counter()
        positions, _ = fast.bm25.retrieve(
            [query.split()], k=RECALL_DEPTH, show_progress=False
        )
        timings.append(time.perf_counter() - started)
        fast_top.append(set(positions[0].tolist()))
    latency = percentiles(timings)
    rows.append(
        {
            "dataset": config.name,
            "split": split,
            "backend": "bm25s",
            "documents": len(corpus),
            "build_seconds": round(fast_build, 2),
            "queries_per_second": round(1 / np.mean(timings), 3),
            "agreement": 1.0,
            **{k: round(v, 3) for k, v in latency.items()},
        }
    )

    if BM25Okapi is None:
        print("    rank_bm25 not installed; nothing to compare against")
        return rows

    started = time.perf_counter()
    slow = BM25Okapi(corpus, k1=config.lexical.k1, b=config.lexical.b)
    slow_build = time.perf_counter() - started

    timings, agreement = [], []
    for query, top in zip(text, fast_top):
        started = time.perf_counter()
        scores = slow.get_scores(query.split())
        found = np.argpartition(-scores, RECALL_DEPTH)[:RECALL_DEPTH]
        timings.append(time.perf_counter() - started)
        agreement.append(len(top & set(found.tolist())) / RECALL_DEPTH)
    latency = percentiles(timings)
    rows.append(
        {
            "dataset": config.name,
            "split": split,
            "backend": "rank_bm25",
            "documents": len(corpus),
            "build_seconds": round(slow_build, 2),
            "queries_per_second": round(1 / np.mean(timings), 3),
            "agreement": float(np.mean(agreement)),
            **{k: round(v, 3) for k, v in latency.items()},
        }
    )
    for row in rows:
        print(
            f"    {row['backend']:10} build {row['build_seconds']:8.2f}s  "
            f"{row['queries_per_second']:9.1f} q/s  p95 {row['p95_ms']:9.3f} ms  "
            f"top-{RECALL_DEPTH} agreement {row['agreement']:.3f}",
            flush=True,
        )
    return rows


# Two sample sizes, because `rank_candidates` loads its index inside the call:
# a 200 MB matrix amortised over one sample is a large share of a 0.3 ms path,
# and reporting that as the per-impression cost would price the setup as though
# every impression paid it. The difference between the two divides them.
SERVE_SAMPLES = (1000, 4000)


def bench_serve(config: DatasetConfig, split: str) -> list[dict]:
    """What one impression costs on the path a submission runs.

    `pipeline.timings` reports what a rebuild costs per stage; this reports
    what an impression costs -- look up the history, build the query or the
    user vector, score the candidates the impression supplies.

    **marginal** is the cost of one more impression, from the slope between the
    two sample sizes. **setup** is the intercept: loading the index, which a
    served system pays once at startup and a batch submission pays once per
    run. Both come out of the real path rather than a reimplementation of its
    internals, so neither can drift from what `predict` actually does.

    Both windows are timed for the same reason: `rank_candidates` takes
    `history_k`, so the difference between the two rows is exactly the
    history-dependent cost -- which is what phase 4's +0.018 AUC was bought
    with, priced in milliseconds.

    Not to be confused with `pipeline.serve`, which is the A2 module of a
    similar name. This one measures **stage one** in batch and recovers the
    marginal cost by regression; that one issues single requests through the
    **whole two-stage path** and reports their percentiles directly. The two
    answer different questions and neither number is the other.
    """
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    scored = behaviors[behaviors["split"] == split]
    largest = scored.head(max(SERVE_SAMPLES))
    history = ingest.history_for(config, largest)
    candidates = float(np.mean([len(row) for row in largest["candidate_ids"]]))
    print(f"    {len(largest):,} impressions, {candidates:.1f} candidates each")

    rows: list[dict] = []
    for name, module in (("bm25", bm25_index), ("ann", ann_index)):
        for history_k in (10, retrieval.HISTORY_K):
            elapsed = {}
            for size in SERVE_SAMPLES:
                frame = scored.head(size)
                started = time.perf_counter()
                module.rank_candidates(config, frame, history, history_k=history_k)
                elapsed[size] = time.perf_counter() - started
            small, large = SERVE_SAMPLES
            marginal = 1000 * (elapsed[large] - elapsed[small]) / (large - small)
            setup = elapsed[small] - small * marginal / 1000
            row = {
                "dataset": config.name,
                "split": split,
                "retriever": name,
                "history_k": int(history_k),
                "mean_candidates": round(candidates, 1),
                "marginal_ms": round(marginal, 3),
                "impressions_per_second": round(1000 / marginal, 1),
                "setup_seconds": round(setup, 2),
            }
            rows.append(row)
            print(
                f"    {name:5} k={history_k:<3} "
                f"{row['marginal_ms']:8.3f} ms/impression  "
                f"{row['impressions_per_second']:9.1f} impressions/s  "
                f"setup {row['setup_seconds']:5.2f}s",
                flush=True,
            )
    return rows


def quantise(matrix: np.ndarray, precision: str) -> np.ndarray:
    """The matrix as it would come back out of a narrower index.

    Measured as a round trip rather than by reading the index back, so the AUC
    column and the latency column describe the same numbers. `int8` reproduces
    what FAISS' 8-bit scalar quantiser does: a uniform grid per dimension,
    between that dimension's own extremes.
    """
    if precision == "fp32":
        return matrix
    if precision == "fp16":
        return matrix.astype(np.float16).astype(np.float32)
    low, high = matrix.min(axis=0), matrix.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    grid = np.rint((matrix - low) / span * 255)
    return (low + grid / 255 * span).astype(np.float32)


PRECISIONS = (
    ("fp32", None),
    ("fp16", faiss.ScalarQuantizer.QT_fp16),
    ("int8", faiss.ScalarQuantizer.QT_8bit),
)


def bench_precision(config: DatasetConfig, split: str, resamples: int) -> list[dict]:
    """Narrower vectors: what they save, and what they cost in AUC.

    The width finding this pairs with -- 384d to 768d bought +0.0143 AUC and
    doubled the index and the query -- says width is a recurring cost the
    accuracy table hides. Precision is the same trade with the sign reversed,
    and unlike the index choice it *is* on the ranking path, so AUC moves.
    """
    from pipeline import embed_compare

    embeddings = embed.load(config)
    vectors = np.ascontiguousarray(embeddings.vectors, dtype=np.float32)
    dim = vectors.shape[1]
    queries = user_vectors(config, split)
    sample = queries[:LATENCY_SAMPLE]

    exact = faiss.IndexFlatIP(dim)
    exact.add(vectors)
    _, truth = exact.search(sample, RECALL_DEPTH)

    scored = embed_compare.impressions(config, split)
    position = {
        article: row
        for row, article in enumerate(
            pd.read_parquet(
                config.feature_store_dir / "articles.parquet", columns=["article_id"]
            )["article_id"]
        )
    }
    print(f"    {len(scored):,} {split} impressions, {len(sample):,} sampled queries")

    rows: list[dict] = []
    baseline = None
    for name, quantiser in PRECISIONS:
        if quantiser is None:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexScalarQuantizer(
                dim, quantiser, faiss.METRIC_INNER_PRODUCT
            )
            index.train(vectors)
        index.add(vectors)
        latency = search_latency(index, sample)
        _, found = index.search(sample, RECALL_DEPTH)
        overlap = [
            len(set(a.tolist()) & set(b.tolist())) / RECALL_DEPTH
            for a, b in zip(truth, found)
        ]
        values = embed_compare.per_impression_auc(
            quantise(vectors, name), scored, position, retrieval.HISTORY_K
        )
        auc = float(values.mean()) if len(values) else 0.0
        if baseline is None:
            baseline = auc
        row = {
            "dataset": config.name,
            "split": split,
            "precision": name,
            "dim": int(dim),
            "index_mb": round(index_bytes(index) / 1e6, 1),
            "recall_vs_exact": float(np.mean(overlap)),
            "auc": auc,
            "auc_delta": round(auc - baseline, 4),
            **{k: round(v, 3) for k, v in latency.items()},
        }
        rows.append(row)
        print(
            f"    {name:5} {row['index_mb']:7.1f} MB  p50 {row['p50_ms']:6.3f} ms  "
            f"recall {row['recall_vs_exact']:.4f}  auc {auc:.4f} "
            f"({row['auc_delta']:+.4f})",
            flush=True,
        )
        del index
    return rows


HEADERS = {
    "ann": (
        "Vector index: what the alternatives cost",
        "`ann_index.build` takes an exact flat inner-product index. Every "
        "alternative is priced here against it, on the same vectors and the "
        "same queries.\n\n"
        "- **recall vs exact** is the share of the exact top-10 an "
        "approximate index returns. It is not recall against clicks: the "
        "question is what the approximation loses relative to what it "
        "approximates.\n"
        "- **latency** is one query at a time, which is how a request "
        "arrives. Batched search is faster per query and is not what a "
        "serving budget is written against.\n"
        "- `numpy` is no index at all: one BLAS call over the same matrix, "
        "then an argpartition. It is the floor the flat index has to beat to "
        "justify existing.\n"
        "- **The index is not on the ranking path.** `rank_candidates` scores "
        "an impression's own candidates directly, so an approximate index can "
        "cost recall@K and cannot cost AUC.",
    ),
    "scale": (
        "Where exact search stops being free",
        "The catalogue tiled to multiples of itself, with the same searches "
        "timed at each size. Exact search is the shipped choice because "
        "approximation is not worth its recall loss at *this* catalogue; the "
        "point of this table is to say where that stops being true rather "
        "than leave it as a property of the two catalogues we happened to "
        "have.\n\n"
        "- Tiling repeats vectors, which moves the distribution and not the "
        "work: a flat scan costs `n x d` whatever the rows say, and what is "
        "measured here is cost.\n"
        "- 8x is a memory ceiling on the machine that ran it, not a "
        "judgement about where to stop looking.",
    ),
    "lexical": (
        "BM25 backend: bm25s against rank_bm25",
        "`SPEC.md` chose `bm25s` because it is \"faster than `rank_bm25`\". "
        "This is that claim, measured, on the shipped corpus and the shipped "
        "query construction.\n\n"
        "- **agreement** is the overlap of the two backends' top-10 on the "
        "same query. A backend that is quicker because it scores something "
        "else is not an alternative, and the timing would not mean anything "
        "without this column.\n"
        "- Both are given the same `k1` and `b` and the same pre-tokenised "
        "corpus, so what is timed is the scoring, not the cleaning.",
    ),
    "serve": (
        "The ranking path: what one impression costs",
        "`pipeline.timings` reports what a *rebuild* costs per stage. This "
        "reports what an *impression* costs on the path `predict` runs -- "
        "look up the history, build the query or the user vector, score the "
        "candidates the impression supplies.\n\n"
        "- **marginal** is what one more impression costs, from the slope "
        "between two sample sizes; **setup** is the intercept, which a served "
        "system pays once at startup. Amortising the two together would "
        "price a 200 MB index load as though every impression paid it.\n"
        "- The candidate count is the width of the work: both retrievers "
        "score exactly the candidates the impression came with.",
    ),
    "precision": (
        "Vector precision: bytes and milliseconds against AUC",
        "fp16 and 8-bit scalar quantisation against the shipped fp32 "
        "vectors.\n\n"
        "- Unlike the index choice, precision **is** on the ranking path, so "
        "AUC moves and is reported. The AUC is computed on the same "
        "impressions, by the same per-impression definition the harness "
        "uses.\n"
        "- **auc_delta** is against fp32 on the same rows, so it is a paired "
        "difference.",
    ),
}


# How each column reads. Latencies span four orders of magnitude between an
# index and a pure-Python scorer, and a single format either loses the fast
# column or fills the slow one with noise; a share needs four decimals whatever
# its magnitude, because the interesting ones sit just under 1.
SHARES = ("recall_vs_exact", "agreement", "auc", "auc_delta")


def cell(key: str, value: object) -> str:
    """A number at the width its column is worth reading at."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if key in SHARES:
        return f"{value:.4f}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def document(rows: list[dict], kind: str, dataset: str) -> str:
    """The table, from the rows' own keys, so nothing here is typed by hand."""
    title, preamble = HEADERS[kind]
    columns = [
        key for key in rows[0] if key not in ("dataset", "split")
    ]
    lines = [
        f"# {title} — {dataset}",
        "",
        f"Generated by `python -m pipeline.bench --dataset {dataset} "
        f"--bench {kind}`; nothing here was typed by hand.",
        "",
        preamble,
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(
            "---" if isinstance(rows[0][key], str) else "---:" for key in columns
        ) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(key, row[key]) for key in columns) + " |")
    return "\n".join(lines) + "\n"


BENCHES = ("ann", "scale", "lexical", "serve", "precision")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mind")
    parser.add_argument("--bench", choices=(*BENCHES, "all"), default="all")
    parser.add_argument("--split", default=evaluate.TUNE, choices=evaluate.SCORABLE)
    parser.add_argument("--resamples", type=int, default=evaluate.BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)

    config = DATASETS[args.dataset]
    wanted = BENCHES if args.bench == "all" else (args.bench,)
    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for kind in wanted:
        print(f"  {kind} — {config.name}", flush=True)
        if kind == "ann":
            rows = bench_ann(config, args.split)
        elif kind == "scale":
            rows = bench_scale(config, args.split)
        elif kind == "lexical":
            rows = bench_lexical(config, args.split)
        elif kind == "serve":
            rows = bench_serve(config, args.split)
        else:
            rows = bench_precision(config, args.split, args.resamples)

        results = paths.ARTIFACTS_DIR / RESULTS.format(kind=kind, dataset=config.name)
        with results.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        (
            paths.ARTIFACTS_DIR / DOCUMENT.format(kind=kind, dataset=config.name)
        ).write_text(document(rows, kind, config.name), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
