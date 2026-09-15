# 05 — NRMS-DocVec, reproduced as a fourth retriever

**What to build:** The official baseline, runnable through the same harness as the three A1
retrievers. NRMS-DocVec in PyTorch over the document vectors A1 already chose and corrected
(e5+centre on MIND, word2vec+abtt on EB-NeRD): multi-head self-attention and additive attention
over the user's recent clicks, dot product with the candidate. Trains on the *earlier* half of
`train` (the later half is reserved for the re-ranker in ticket 06), chooses its history length
and epochs on `tune`, reports on `validation`. Registered as one more entry in the harness's
retriever table, so evaluation, slicing, bootstrap CIs and beyond-accuracy metrics come for free.

**Blocked by:** 01 — every grid cell is a ledger row.

**Status:** built 2026-09-15 — the grid's numbers land on the first run against a built feature store

- [x] An `NrmsSpec` (history length, heads, negatives, epochs, learning rate, train fraction) lives in the registry per dataset; the chosen values are recorded there with the tune number that chose them.
- [x] A `nrms` module exposes `train`, `rank_candidates` with the shared retriever signature, and `ranker` for the submission path; the checkpoint lands under the dataset's artifacts.
- [x] `train` fits only on impressions from the earlier chronological half of `train`; a test asserts the boundary.
- [x] Negatives are sampled from the same impression (1:4); a test asserts no negative comes from another impression.
- [x] `evaluate --retriever nrms` runs on `validation` for both datasets and writes the usual JSON with every metric and slice.
- [x] Grid on `tune`: history length ∈ {20, 50, 80}, heads ∈ {8, 16}; each cell a ledger row with AUC, train seconds, model bytes and per-impression scoring ms.
- [ ] The ebnerd-benchmark paper's reported NRMS-DocVec number on `ebnerd_small` is looked up and written next to ours in the artifact markdown, with the gap stated either way.
- [x] Runs on the laptop CPU within an hour per dataset, or the Ada GPU job that runs it is checked in.

## Options tried (each a ledger row on `tune`; the chosen cell is named in `NrmsSpec`)

- [x] **Architecture grid** (above): history length × heads. The history-length axis doubles as an engineering curve — longer history is more attention work per impression, so the row's `p50_ms` moves with its AUC.
- [x] **Input vectors:** A1's *corrected* vectors (centre / abtt) against the *raw* ones from the same encoder. A1 chose the correction for nearest-neighbour retrieval; whether a trained user encoder still wants it is a separate question, answered by two ledger rows per dataset that cost nothing but a second training run.
- [x] **Negatives:** in-impression 1:4 (the paper's setting) against 1:1 and against uniformly random catalogue negatives — `NrmsSpec.negative_source`, swept by `GRID`. The random-negative row is expected to lose; it is there so the note can say the baseline was trained the way its authors trained it *and* that the choice mattered. One difference the arms have beyond difficulty: an impression where every candidate was clicked has no in-impression negative and is skipped, and the catalogue arm can still learn from it.
- [x] **Epochs by early stopping on `tune` AUC**, patience 1; the row records the epoch chosen and the AUC one epoch either side, so the note can show the curve rather than assert the stop.

## Latency and loading

NRMS is the slowest stage-two component; ticket 09 needs its cost split into "per user" and "per
candidate" so the serving breakdown is honest about which part a cache could remove.

- [x] **Scoring path:** the user vector is computed once per impression and dot-producted against all candidates — never one forward pass per `(user, candidate)` pair. A ledger row per dataset records `p50_ms`/`p99_ms` for the user encoder alone and for the candidate dot alone, from `timings.sample()` around each.
- [x] **Batched scoring at evaluation and submission:** impressions are scored in batches (histories padded to the spec's length); batch size ∈ {64, 512} tried once on `tune`, rows/s and peak RSS recorded. `rank_candidates` and `ranker` share the batched path, so the harness and the submission measure the same code.
- [x] **Vectors:** the document matrix is `np.load(mmap_mode="r")` and gathered per batch by int id — no per-user copy of the catalogue. Resident bytes with and without mmap are one ledger row.
- [x] **Checkpoint bytes:** `fp32` state dict against `fp16` for scoring; `model_bytes` and `tune` AUC per precision are two rows, the same shape as A1's precision bench for the index.
- [x] **Training data loading:** click histories come from the history table keyed by user (one read per training run, not one per batch); a test asserts the training loop opens the store once.

**What is built and what is measured.** Every box above is code with a test on it; `pytest` is
green at 451 tests, 20 of them this module's. The grid's numbers — tune AUC per cell, train
seconds, model bytes, the user-encoder and candidate-dot halves of the per-request latency — are
recorded by `nrms.record` as each cell finishes and are blank until the grid is run against a
built feature store, which this checkout does not carry. `sbatch ada/nrms.sbatch --grid` is the
run; `python -m pipeline.nrms --grid` is the same thing on a laptop.

**Two boxes that are not closed by code.** The ebnerd-benchmark paper's NRMS-DocVec figure is
*not* filled in: `nrms.PAPER["auc"]` is `None`, the grid's markdown prints "Outstanding" and names
what has to be read, and the gap is computed the moment the number is set. And "within an hour on
the laptop" is a measurement, not an assertion — the GPU job is checked in either way, which is
the branch the ticket offers.

**Found on the way:** a cold user was being handed article row 0's vector. Padding a history with
row index 0 and marking the position attended (so the softmax has something to normalise over) is
two different facts collapsed into one flag; the fix is a -1 row that `gather` zeroes. It was
invisible in every metric and would have distorted precisely the cold-start slice. See
`DECISIONS.md`.
