# 08 — The submission path, on Ada, on both leaderboards

**What to build:** The re-ranker ranking the competitions' own test sets — 2.37M MIND impressions,
13.5M EB-NeRD — through the existing streaming submission writer, run on the cluster, checked
byte-for-byte on both ends, uploaded, screenshotted. At test time the click counters are fed only
by the training-period log and the test users' own histories, and the exposure counters by test
impressions strictly before each impression's time: the same code, no special case, which is what
makes it the "what a live server knows" line.

**Blocked by:** 06 — the model and its `ranker`.

**Status:** ready-for-agent

- [ ] `predict --retriever rerank` builds the feature frame per chunk, scores NRMS per chunk, and writes the zip through the existing writer; memory stays bounded by the chunk size.
- [ ] **Self-consistency test:** the `ranker` path run over MINDsmall-dev scores the same AUC as the harness's `rank_candidates` path on the same impressions, to four decimals — the A1 check that found the id-collision bug, applied to the new retriever.
- [ ] An sbatch script for the re-rank submission is checked in beside the existing ones; runs for both datasets complete on Ada with wall time, rows/s and peak RSS recorded in the ledger.
- [ ] Zips are brought back with `rsync --no-compress --whole-file`; md5 matches on both ends before either is called ready.
- [ ] Both GBDT files uploaded; leaderboard scores and screenshots recorded under `screenshots/` and in the checkpoint state.
- [ ] Optional, only if both GBDT files are up by Sept 18: NRMS files built and uploaded the same way, giving the baseline an external witness.
