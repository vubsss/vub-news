# 07 — Ablation arms, the Q9 pair, and the stage-one bridge

**What to build:** The evidence that the improvement is understood, not just measured. The same
re-ranker retrained and rescored once per arm, each paired against the full model with a
bootstrap interval; the two anti-gaming rows the assignment asks for; and the recall@K table that
says how often the candidate generator would even surface the click. Plus the side-by-side table
of all five retrievers.

**Blocked by:** 06.

**Status:** built 2026-09-15 — the tables land on the first run against a built feature store

- [x] An `ablation` module runs the arms on a named split and writes one paired-CI table per dataset; arms: full; −history; −exposure; −clicked; −session/dwell; −NRMS score; −retriever scores.
- [x] **Literal top-K cut arm:** candidates outside the semantic retriever's global top-K are ranked last, for K ∈ {50, 100, 200}; reported as an ablation row, never as the headline.
- [x] **Leaky arm (Q9):** counters read over the whole log instead of causally — one flag, same model — with the gap against the clean arm stated on both datasets.
- [x] Stage-one recall@K of the semantic retriever at K ∈ {50, 100, 200} is in the same markdown, next to the literal-cut rows it explains.
- [x] A five-retriever comparison (bm25, ann, hybrid, nrms, rerank) on `validation` with every metric and slice, regenerated from stored JSON.
- [x] Arms are chosen and inspected on `tune`, then run once on `validation`; nothing is run on `test`.
- [x] Every arm is a ledger row with its delta and interval and its own model bytes and per-impression ms.
- [x] Tests: an arm that drops a tier has none of that tier's columns in its model; the leaky arm and the clean arm differ only in the counter's `t`.

## Options tried (each arm is one; the table is the comparison)

The arms above are the functional variations. Two more make the ablation say something about
*serving*, not only about features:

- [x] **Availability-tier arms in serving order:** `content` only → `+history` → `+exposure` → `+clicked` — the cumulative build-up, not only the leave-one-out. Each row's `p50_ms` includes the feature lookup that tier costs (ticket 04's per-stage timing), so the note has a curve of AUC against feature-lookup latency, tier by tier.
- [x] **Stage-one index arm:** the literal-cut arm with candidates from the flat index against IVF at the same K — the recall@K the IVF loses and the AUC it costs, so ticket 09's flat-versus-IVF latency has a functional column beside it.
- [x] **Leaky arm at every window:** the Q9 gap is reported for the chosen window and for `cumulative`, because the leak is largest when the counter runs to the end of the log; two rows keep the note from quoting only the flattering one.

## Latency and loading

- [x] **One frame, loaded once:** the ablation reads the `validation` frame a single time and each arm is a column projection of it (ticket 04's readers); no arm rebuilds features. A test asserts the builder is called once per split for the whole run.
- [x] **Arms retrain from the projected frame sequentially**, row-group by row-group, with the same loader as ticket 06; the whole ablation's wall time, peak RSS and arm count are one ledger row, so the note can say what a full ablation costs to rerun.
- [x] **Literal-cut candidates** come from a single batched `index.search` per split at `K=200` and are sliced to 50 and 100 — three arms, one search; the search's `p50_ms` is recorded once and attributed to all three.
- [x] Per-arm `model_bytes` and `p50_ms` are measured, not copied from the full model — a tier-less model is smaller and faster, and that is part of what the arm shows.

**What is built and what is measured.** Nineteen arms, every one of them a `RerankSpec` and a
choice of index, each paired against `full` by impression with a bootstrap interval; `pytest` is
green at 501 tests, 17 of them this module's. The numbers land on the first run against a built
feature store: `python -m pipeline.ablation --split tune` to inspect the arms, `--split validation`
to report them.

**One place the ticket's wording and the code differ, on purpose.** "Arms retrain from the
projected frame sequentially" is true of every arm that is a different *model*; arms that differ
only in the top-K cut or in which index supplied the ranks share one training, because a cut is
applied when the ranking is produced and retraining would grow identical trees. The saved bytes on
those rows are equal, which is the fact rather than a rounding of it. See `DECISIONS.md`.

**Found on the way:** stopping early on the split the arms are scored on would have selected the
round count on `validation` for every row of the validation table. `run_arms` takes three splits —
fit on the later half of `train`, stop on `tune`, score on the named one — and a test asserts the
stopping rows never intersect the scored ones.
