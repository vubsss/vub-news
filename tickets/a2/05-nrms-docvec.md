# 05 — NRMS-DocVec, reproduced as a fourth retriever

**What to build:** The official baseline, runnable through the same harness as the three A1
retrievers. NRMS-DocVec in PyTorch over the document vectors A1 already chose and corrected
(e5+centre on MIND, word2vec+abtt on EB-NeRD): multi-head self-attention and additive attention
over the user's recent clicks, dot product with the candidate. Trains on the *earlier* half of
`train` (the later half is reserved for the re-ranker in ticket 06), chooses its history length
and epochs on `tune`, reports on `validation`. Registered as one more entry in the harness's
retriever table, so evaluation, slicing, bootstrap CIs and beyond-accuracy metrics come for free.

**Blocked by:** 01 — every grid cell is a ledger row.

**Status:** ready-for-agent

- [ ] An `NrmsSpec` (history length, heads, negatives, epochs, learning rate, train fraction) lives in the registry per dataset; the chosen values are recorded there with the tune number that chose them.
- [ ] A `nrms` module exposes `train`, `rank_candidates` with the shared retriever signature, and `ranker` for the submission path; the checkpoint lands under the dataset's artifacts.
- [ ] `train` fits only on impressions from the earlier chronological half of `train`; a test asserts the boundary.
- [ ] Negatives are sampled from the same impression (1:4); a test asserts no negative comes from another impression.
- [ ] `evaluate --retriever nrms` runs on `validation` for both datasets and writes the usual JSON with every metric and slice.
- [ ] Grid on `tune`: history length ∈ {20, 50, 80}, heads ∈ {8, 16}; each cell a ledger row with AUC, train seconds, model bytes and per-impression scoring ms.
- [ ] The ebnerd-benchmark paper's reported NRMS-DocVec number on `ebnerd_small` is looked up and written next to ours in the artifact markdown, with the gap stated either way.
- [ ] Runs on the laptop CPU within an hour per dataset, or the Ada GPU job that runs it is checked in.
