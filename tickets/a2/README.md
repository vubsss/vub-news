# Tickets — Assignment 2: Learning from Click-Logs

Eleven tickets breaking `../../NEW_PLAN.md` and `A2.pdf` into tracer-bullet slices, each sized to be
picked up cold and finished in one sitting. Each cuts a narrow but complete path — registry,
pipeline stage, harness, test — and is done when it is demoable on its own: a number in the ledger,
a file that scores, a test that fails when the guard it protects is removed.

The decisions these tickets implement were settled in the design interview recorded in
`NEW_PLAN.md`; do not re-litigate them here. Two rules carry over from A1 and bind every ticket:
**chosen on `tune`, reported on `validation`, `test` scored once**, and **no stage branches on a
dataset name** — every MIND/EB-NeRD difference lives in the registry.

## What is new in these tickets

Every ticket that tries an option writes a row to the **trade-off ledger** — functional metrics
(AUC, MRR, nDCG, diversity, novelty, coverage, with CIs) and engineering metrics (bytes, seconds,
peak RSS, p50/p99, rows/s) on the same row. The design note is written from that file, so a
ticket is not done while its row has one side blank.

From ticket 04 on, every ticket closes with two sections that turn that rule into work:

- **Options tried** — the variations compared at that stage, each a ledger row chosen on `tune`,
  so the design note can show a decision as a table with the losers in it. Axes are swept one at a
  time from a stated default, not as a grid; where an axis is *not* a real variation (a monotone
  transform under a GBDT, say) the ticket says so, so nobody adds it later.
- **Latency and loading** — how the stage reads its inputs (parquet row groups, column projection,
  `mmap`, chunks in `impression_time` order, loaded-once per job or per chunk or per request), which
  index method and precision, per-stage timers from `timings.sample()`, and the engineering
  columns those measurements fill. The submission path (08) is where sequential loading is the
  whole point; the serving benchmark (09) is where the index method is chosen with its functional
  column beside it.

## Dependency graph

```mermaid
graph TD
    T01[01 · Ledger seeded with A1's rows]
    T02[02 · session_id in the feature store]
    T03[03 · Causal counters + leakage test]
    T04[04 · Feature frame + forbidden-column test]
    T05[05 · NRMS-DocVec, a fourth retriever]
    T06[06 · GBDT re-ranker, a fifth retriever]
    T07[07 · Ablation arms + recall@K bridge]
    T08[08 · Submission path on Ada]
    T09[09 · Two-stage serving benchmark]
    T10[10 · test scored once]
    T11[11 · Design note + reproduce]

    T01 --> T03
    T01 --> T05
    T02 --> T04
    T03 --> T04
    T04 --> T06
    T05 --> T06
    T06 --> T07
    T06 --> T08
    T07 --> T09
    T07 --> T10
    T08 --> T10
    T09 --> T11
    T10 --> T11
```

**Frontier at the start:** 01 and 02. As soon as 01 lands, 05 runs in parallel with 03 → 04. That
parallelism is what makes five days enough.

## Budget

| by end of | tickets |
|---|---|
| Sept 16 | 01, 02, 03, 05 |
| Sept 17 | 04, 06 |
| Sept 18 | 07, 08 (jobs running) |
| Sept 19 | 09, 10 |
| Sept 20 | 11 |

The options sections add roughly a half-day across 04–06 (mostly extra training runs on `tune` from
frames and loaders that exist anyway) and nothing to 07–09, whose options are the same benchmarks
run with one more flag. If the budget slips, the axes to drop first are named in each ticket's
order: last bullet first.
