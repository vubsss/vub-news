# 02 — `session_id` in the feature store

**What to build:** The behaviours table carries the session an impression belonged to, on the
dataset that ships one. EB-NeRD's raw `session_id` reaches the unified schema through the column
map; MIND, which has no sessions, gets a null column the same way it gets a null `published_time`.
No stage learns which dataset it is serving.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `session_id` is a behaviour column in the registry's column map: mapped from the raw column for EB-NeRD, `None` for MIND.
- [ ] Both feature stores are re-ingested; the split stage and its leakage guard re-run and pass.
- [ ] EB-NeRD: `session_id` is non-null on every row and a session never spans more than one user.
- [ ] MIND: `session_id` is null on every row, and the column exists.
- [ ] Ingest tests cover the new column on both adapters; no test or module branches on the dataset name to handle it.
- [ ] Row counts and every pre-existing column are unchanged by the re-ingest (compare against the previous store before deleting it).
