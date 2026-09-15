# 09 — The two-stage serving benchmark, and where 10× breaks

**What to build:** The measured answer to Q4. One user request through the real two-stage path —
semantic retrieval of top-K, feature lookup, NRMS scoring, GBDT scoring — timed per stage over
thousands of requests, with the bytes of every store the request touches, a cost per thousand
queries at a stated SLA, and the K-versus-latency curve that is the project's cleanest
functional-against-engineering trade-off. Then the scaling argument, in numbers: what grows at 10×
load, what grows at 10× catalogue, and which breaks first.

**Blocked by:** 07 — the K-curve pairs the literal-cut arm's AUC with the serving path's p99.

**Status:** ready-for-agent

- [ ] A `serve` benchmark, extending the existing bench module, runs N ≥ 2000 single-user requests and reports p50/p99 overall and per stage (retrieve, features, nrms, gbdt).
- [ ] Bytes reported for: the FAISS index, the feature store, the counter store, the NRMS checkpoint, the GBDT model — each a ledger engineering column on the rerank row.
- [ ] Cost per 1000 queries at p99 < 100 ms, derived from measured QPS per core and a stated $/core-hour, with the arithmetic shown.
- [ ] K ∈ {50, 100, 200}: AUC from the literal-cut arm against p99 from this benchmark, one row per K in the ledger and one plot-ready table in the markdown.
- [ ] The 10× argument written from the measured large-run costs (ticket 08) and the history-schema findings: which store, which stage, and at what multiple it exceeds the node.
- [ ] `artifacts/bench-serve-<dataset>.md` regenerated for both datasets; tests cover the per-stage timer and the bytes accounting.

## Options tried (each a ledger row: p50/p99 per stage, bytes, and the functional column from ticket 07)

The A1 bench already measures flat / IVF / HNSW and fp32 / fp16 / int8 for stage one; this ticket
runs the same options *through the whole two-stage path*, because what matters is whether a faster
stage one moves the request's p99 at all once stage two is behind it.

- [ ] **Stage-one index ∈ {flat, IVF, HNSW}** at the chosen K: retrieve-stage p50/p99, `index_bytes`, and the recall@K and literal-cut AUC from ticket 07's index arm on the same row. The note's claim is expected to be "stage one is not the bottleneck" — the rows have to show it.
- [ ] **Index precision ∈ {fp32, fp16, int8}** for the chosen index type — bytes against retrieve-stage latency against recall@K, reusing `bench.quantise`.
- [ ] **K ∈ {50, 100, 200}** (above), and additionally the **no-cut path** the harness uses (score every logged candidate) so the curve has its ceiling on it.
- [ ] **Feature lookup structure:** pandas index lookup against the flat `np.ndarray` + `searchsorted` layout ticket 03 used for counters, for the user-profile and article-side features. Two rows; the note quotes the feature stage at the faster one and says what the slower one would have cost.
- [ ] **NRMS user vector cached per user** (computed once, reused for every request from that user in the window) against recomputed per request — the cache-hit and cache-miss p99 as two rows, and the bytes a per-user cache costs at the dataset's user count.
- [ ] **GBDT `num_threads` ∈ {1, all}** on the serving path (ticket 06 measured it in batch); the single-thread row is the one the cost arithmetic uses.

## Latency and loading

- [ ] **Cold against warm:** the FAISS index opened with `IO_FLAG_MMAP` against fully loaded — resident bytes, first-request latency and steady-state p99 for both; the note says which a server should do and why.
- [ ] **Request path loads nothing per request** beyond a row lookup: a test asserts that after warm-up the benchmark's per-request path opens no file. Everything it touches is in the bytes table.
- [ ] **Per-stage timers** use `timings.sample()`-style perf counters around each of retrieve / features / nrms / gbdt inside one request, so the four stage p99s and the total are from the same requests; the sum of stage medians against the overall median is reported as a sanity check on the timer.
- [ ] **QPS per core** is measured at `num_threads=1` on one core (`taskset`), and the cost per 1000 queries at p99 < 100 ms is derived from it with the $/core-hour stated in the markdown.
- [ ] **10× argument in three rows:** 10× users (feature store and per-user cache grow), 10× catalogue (index, counters, and the candidate frame grow), 10× QPS (cores grow) — each with the measured bytes or ms from this ticket and ticket 08 multiplied out against the node's RAM and core count, and the first thing that exceeds the node named.
