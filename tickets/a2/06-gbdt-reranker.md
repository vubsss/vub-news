# 06 — The GBDT re-ranker, a fifth retriever, and the headline number

**What to build:** The improvement over the baseline: a LightGBM model over the feature frame
plus the NRMS score, trained on the *later* half of `train`, tuned on `tune`, reported on
`validation` with a paired bootstrap interval against NRMS. It slots into the harness as a fifth
retriever, so the number that says "baseline beaten" is produced by the same code path that
produced every other number in the project.

**Blocked by:** 04 — the features. 05 — the NRMS score is a feature and the baseline it is
compared against.

**Status:** built 2026-09-15 — the arms' numbers land on the first run against a built feature store

- [x] A `RerankSpec` (objective, leaves, rounds, counter window, feature groups, use-NRMS flag, causal flag, K) lives in the registry per dataset with the tune numbers that chose it.
- [x] A `rerank` module exposes `train`, `rank_candidates` with the shared signature, and `ranker`; the model lands under the dataset's artifacts.
- [x] `train` fits only on the later chronological half of `train` and never on a row NRMS trained on; a test asserts the two halves are disjoint.
- [x] `evaluate --retriever rerank` runs on `validation` for both datasets with every metric and slice.
- [x] The headline: `rerank − nrms` on `validation`, paired by impression, with a 95% bootstrap interval, on both datasets — recorded in the ledger as the delta of the rerank row.
- [x] Feature-importance (gain) per column and summed per tier is saved alongside the model.

## Options tried (each a ledger row on `tune` with AUC, train seconds, model bytes, per-impression ms)

Every axis below is a `RerankSpec` field, so the chosen cell is the registry entry and the losers
are the other rows. Axes are swept one at a time from a stated default (binary, 31 leaves, `24h`,
all groups, NRMS on), not as a full grid — the budget is one afternoon.

- [x] **Objective:** `binary` logloss against `lambdarank` grouped by impression. AUC is the headline metric and is what `binary` optimises; `lambdarank` targets nDCG. Both rows report every metric, so the note can show which objective wins which metric rather than pick by assumption.
- [x] **Capacity:** leaves ∈ {31, 127}, rounds by early stopping on `tune`; and a **rounds curve** — the model saved at {50, 200, best} rounds, each a row, so AUC against `model_bytes` and `p50_ms` is a three-point curve. This is the re-ranker's version of the K-curve: how much quality the last hundred trees buy and what they cost per request.
- [x] **Counter window:** one training run per window from ticket 04's columns (`cumulative`, `1h`, `3h`, `24h`) and one with all four; five rows per dataset.
- [x] **± NRMS score:** the free ablation from the stacking split — the same model without the NRMS column. This row is what tells the note whether the baseline is worth serving inside the re-ranker or only worth beating.
- [x] **Feature precision:** trained from the `float16` frame and the `float32` frame (ticket 04); if the AUCs agree to three decimals, `float16` is the registry choice and the bytes saved are quoted.

## Latency and loading

- [x] **Training loads the frame sequentially:** the later-half rows are read from parquet row groups by `pyarrow` with the spec's column projection and handed to `lgb.Dataset` — never `read_parquet` of the whole split into pandas first. Peak RSS of `train` with and without projection is one ledger pair, so the note can say what the tier structure buys in memory.
- [x] **Prediction is batched per chunk**, one `predict` call over a chunk's long frame, then ranked within impression by `predict.ranks` — the same path for `rank_candidates` (harness) and `ranker` (submission), so the `p50_ms` per impression in the ledger is the served number.
- [x] **Threads:** `num_threads` ∈ {1, all} at prediction is recorded as two rows on the chosen model — ticket 09's cost-per-1000-queries needs the single-core QPS, and the note should not extrapolate it from a multi-core run.
- [x] `model_bytes` is the saved model file; `train_seconds` and `peak_rss_mb` come from `timings.sample()` around `train`; nothing is estimated.

**What is built and what is measured.** Every box is code with a test on it; `pytest` is green at
483 tests, 30 of them this module's. The numbers are recorded as each arm finishes and are blank
until the arms run against a built feature store, which this checkout does not carry:
`python -m pipeline.rerank --grid` sweeps them, `--headline` writes the `rerank − nrms` delta with
its paired interval onto the `validation` row.

**One box that names its owner rather than being closed here.** `rerank.ranker` exists with the
shared signature and refuses without a frame builder for the competition's catalogue. That builder
is ticket 08's — the test period's counters, freshness and retriever scores come from files the
feature store does not contain — and half-writing it here would be a second implementation of the
feature assembly for somebody to keep in step. `predict --retriever rerank` says so in its error.

**Found on the way:** the leaky *sliding* window is not a superset of the causal one, and nothing
is wrong when it reads smaller. The cumulative leaky counter is the whole log; the sliding one is
the hour after `t` rather than the hour before it, which is where the impression's own outcome
sits. Both arms are in the ledger; the tests say which comparison holds for which. See
`DECISIONS.md`.
