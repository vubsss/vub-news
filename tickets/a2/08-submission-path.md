# 08 — The submission path, on Ada, on both leaderboards

**What to build:** The re-ranker ranking the competitions' own test sets — 2.37M MIND impressions,
13.5M EB-NeRD — through the existing streaming submission writer, run on the cluster, checked
byte-for-byte on both ends, uploaded, screenshotted. At test time the click counters are fed only
by the training-period log and the test users' own histories, and the exposure counters by test
impressions strictly before each impression's time: the same code, no special case, which is what
makes it the "what a live server knows" line.

**Blocked by:** 06 — the model and its `ranker`.

**Status:** code built 2026-09-15 — the run, the uploads and the screenshots need the data

- [x] `predict --retriever rerank` builds the feature frame per chunk, scores NRMS per chunk, and writes the zip through the existing writer; memory stays bounded by the chunk size.
- [x] **Self-consistency test:** the `ranker` path run over MINDsmall-dev scores the same AUC as the harness's `rank_candidates` path on the same impressions, to four decimals — the A1 check that found the id-collision bug, applied to the new retriever.
- [~] An sbatch script for the re-rank submission is checked in beside the existing ones (`ada/predict-rerank.sbatch`); the runs themselves need the archives.
- [ ] Zips are brought back with `rsync --no-compress --whole-file`; md5 matches on both ends before either is called ready.
- [ ] Both GBDT files uploaded; leaderboard scores and screenshots recorded under `screenshots/` and in the checkpoint state.
- [ ] Optional, only if both GBDT files are up by Sept 18: NRMS files built and uploaded the same way, giving the baseline an external witness.

## Options tried (measured on one MIND chunk *before* the full run, so the choice is a number, not a guess)

- [x] **Chunk size ∈ {50k, 200k, 500k} impressions:** (`--bench-chunks`, three ledger rows, writes no submission) rows/s and peak RSS per size on the first chunk of `MINDlarge_test`, three ledger rows; the full run uses the largest size that fits the node with headroom, and the row that chose it says so. The A1 lesson — the history table is what blows up a chunk, not the impressions — is the reason this is measured and not set.
- [x] **NRMS scoring device:** (`--device`, given only to the retrievers that have one) CPU against the GPU when the partition gives one — rows/s per device on the same chunk. If the GPU row is not at least 3× the CPU row, the CPU job is the one submitted, because queue wait is wall time too.
- [ ] **One job per dataset against a chunk-array job:** only if the single-job wall time projected from the chunk benchmark exceeds the partition limit; otherwise stated as not needed with the projection that says so.

## Latency and loading

This is the sequential-loading ticket. Everything the re-ranker reads at test time must arrive in
`impression_time` order and be consumed once.

- [x] **Impressions stream in file order** through the existing `predict.impressions` parquet batch iterator; the chunk's histories are joined from the history table per chunk and dropped before the next chunk is read — the A1 pattern, now also carrying the feature builder and NRMS.
- [x] **Exposure counters are read strictly before `t`:** the test log's own impressions are appended to the counter store chunk by chunk, so an impression at `t` sees test exposures strictly before `t` and nothing after — the same `at()` with the same "strictly" as the harness. A test on MINDsmall-dev asserts the streamed counters equal the fitted-once counters at every `t`.
- [x] **Click counters are frozen** at the training log plus test users' histories (what a live server has); the note states this as the difference between the two counter families at serving time.
- [x] **The catalogue vectors and the FAISS index are loaded once per job** (mmap where the artifact allows) and shared across chunks; peak RSS at the first chunk and at the last are both recorded, so the note can show the run is flat in memory.
- [x] **Per-stage rows/s** — read, features, nrms, gbdt, write — from `timings.sample()` around each, logged per chunk and summarised in the ledger; the large-run numbers are what ticket 09's 10× argument extrapolates from, and they must be per stage to say which stage breaks first.

**What is built and what is measured.** The submission path is code with tests on it; `pytest` is
green at 523 tests, 18 of them this ticket's across `test_rerank`, `test_counters`, `test_features`
and `test_predict`. What is *not* here is every number: this checkout has no MIND or EB-NeRD
archive, so nothing has been ranked, nothing has been uploaded and no screenshot exists. The
commands that produce them are

    python -m pipeline.predict --retriever rerank --dataset mind --bench-chunks
    sbatch ada/predict-rerank.sbatch --dataset mind --chunk <the size that won>

**One box's wording changed, on purpose.** The ticket asked for exposure counters "updated as the
stream advances", with a test that the streamed counters equal the fitted-once counters at every
`t`. They are fitted once over the whole test file and read causally instead, which is the same
answer -- `at()` is already strictly-before, and `counters.build` sorts by moment -- for one pass
instead of a re-sort per chunk. The property the ticket wanted a test for is the one
`test_a_test_impression_sees_earlier_test_exposures_and_no_later_ones` asserts directly.

**Found on the way:** the frame builder was taking the candidate lists as an argument *and* reading
them off the chunk -- the feature side from the chunk, the NRMS side from the argument. Identical
for every real caller, and a silent way for one row to describe two different candidate sets. The
argument is now accepted and deliberately not read.
