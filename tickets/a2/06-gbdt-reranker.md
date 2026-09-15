# 06 — The GBDT re-ranker, a fifth retriever, and the headline number

**What to build:** The improvement over the baseline: a LightGBM model over the feature frame
plus the NRMS score, trained on the *later* half of `train`, tuned on `tune`, reported on
`validation` with a paired bootstrap interval against NRMS. It slots into the harness as a fifth
retriever, so the number that says "baseline beaten" is produced by the same code path that
produced every other number in the project.

**Blocked by:** 04 — the features. 05 — the NRMS score is a feature and the baseline it is
compared against.

**Status:** ready-for-agent

- [ ] A `RerankSpec` (objective, leaves, rounds, counter window, feature groups, use-NRMS flag, causal flag, K) lives in the registry per dataset with the tune numbers that chose it.
- [ ] A `rerank` module exposes `train`, `rank_candidates` with the shared signature, and `ranker`; the model lands under the dataset's artifacts.
- [ ] `train` fits only on the later chronological half of `train` and never on a row NRMS trained on; a test asserts the two halves are disjoint.
- [ ] `evaluate --retriever rerank` runs on `validation` for both datasets with every metric and slice.
- [ ] Options tried on `tune`, each a ledger row with AUC, train seconds, model bytes and per-impression ms: objective ∈ {binary, lambdarank}; leaves ∈ {31, 127}; counter window from ticket 03; rounds by early stopping.
- [ ] The headline: `rerank − nrms` on `validation`, paired by impression, with a 95% bootstrap interval, on both datasets — recorded in the ledger as the delta of the rerank row.
- [ ] Feature-importance (gain) per column and summed per tier is saved alongside the model.
