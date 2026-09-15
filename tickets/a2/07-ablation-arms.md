# 07 — Ablation arms, the Q9 pair, and the stage-one bridge

**What to build:** The evidence that the improvement is understood, not just measured. The same
re-ranker retrained and rescored once per arm, each paired against the full model with a
bootstrap interval; the two anti-gaming rows the assignment asks for; and the recall@K table that
says how often the candidate generator would even surface the click. Plus the side-by-side table
of all five retrievers.

**Blocked by:** 06.

**Status:** ready-for-agent

- [ ] An `ablation` module runs the arms on a named split and writes one paired-CI table per dataset; arms: full; −history; −exposure; −clicked; −session/dwell; −NRMS score; −retriever scores.
- [ ] **Literal top-K cut arm:** candidates outside the semantic retriever's global top-K are ranked last, for K ∈ {50, 100, 200}; reported as an ablation row, never as the headline.
- [ ] **Leaky arm (Q9):** counters read over the whole log instead of causally — one flag, same model — with the gap against the clean arm stated on both datasets.
- [ ] Stage-one recall@K of the semantic retriever at K ∈ {50, 100, 200} is in the same markdown, next to the literal-cut rows it explains.
- [ ] A five-retriever comparison (bm25, ann, hybrid, nrms, rerank) on `validation` with every metric and slice, regenerated from stored JSON.
- [ ] Arms are chosen and inspected on `tune`, then run once on `validation`; nothing is run on `test`.
- [ ] Every arm is a ledger row with its delta and interval and its own model bytes and per-impression ms.
- [ ] Tests: an arm that drops a tier has none of that tier's columns in its model; the leaky arm and the clean arm differ only in the counter's `t`.
