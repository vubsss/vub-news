# 01 — Ledger seeded with A1's rows

**What to build:** A single trade-off ledger that every later ticket appends to, rendered as one
markdown table where a reader sees a variant's AUC and its index bytes and its p99 on the same
line. Seeded with the six rows A1 already measured (bm25, ann, hybrid on both datasets, validation),
so every A2 variant has a stage-one row to be a delta against from day one.

Also the housekeeping that must precede everything: the tree is committed clean (the pending README
and `ann_index` edits, the untracked bench module) and `lightgbm` is in the environment.

**Blocked by:** None — can start immediately.

**Status:** done 2026-09-15

- [x] `git status` is clean before the first A2 code lands; `report/` from A1 stays untracked.
- [x] `lightgbm` (CPU wheel) is in `requirements.txt` and `environment.yml`; `import lightgbm` works in the project env.
- [x] A `ledger` module owns the row schema: keys `(dataset, stage, variant, split)`; functional columns `auc, mrr, ndcg@5, ndcg@10, diversity, novelty, coverage` each with `lo, hi`; engineering columns `index_bytes, feature_bytes, model_bytes, train_seconds, peak_rss_mb, p50_ms, p99_ms, rows_per_s`; optional `delta_vs, delta, delta_lo, delta_hi`.
- [x] `ledger.record(row)` appends one JSON line to `artifacts/tradeoffs.jsonl`; recording the same key twice replaces rather than duplicates.
- [x] `ledger.render()` writes `artifacts/tradeoffs.md` grouped by dataset then stage; blank cells are rendered as `—`, never dropped.
- [x] A seed command reads A1's existing evaluate JSON, bench JSONL and build-timings JSONL and records the six retriever rows with both metric families filled.
- [x] Tests: record-then-render round-trips a row; a re-recorded key replaces; render never raises on a row with a missing engineering value.
- [x] `python -m pipeline.ledger` regenerates the markdown; `pytest` is green.
