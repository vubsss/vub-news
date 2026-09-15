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
- [ ] Flat against IVF index at the same K, latency and recall, as a second engineering option tried.
- [ ] The 10× argument written from the measured large-run costs (ticket 08) and the history-schema findings: which store, which stage, and at what multiple it exceeds the node.
- [ ] `artifacts/bench-serve-<dataset>.md` regenerated for both datasets; tests cover the per-stage timer and the bytes accounting.
