# 02 — `session_id` in the feature store

**What to build:** The behaviours table carries the session an impression belonged to, on the
dataset that ships one. EB-NeRD's raw `session_id` reaches the unified schema through the column
map; MIND, which has no sessions, gets a null column the same way it gets a null `published_time`.
No stage learns which dataset it is serving.

**Blocked by:** None — can start immediately.

**Status:** done 2026-09-15

- [x] `session_id` is a behaviour column in the registry's column map: mapped from the raw column for EB-NeRD, `None` for MIND.
- [x] Both feature stores are re-ingested; the split stage and its leakage guard re-run and pass.
- [x] EB-NeRD: `session_id` is non-null on every row and a session never spans more than one user.
- [x] MIND: `session_id` is null on every row, and the column exists.
- [x] Ingest tests cover the new column on both adapters; no test or module branches on the dataset name to handle it.
- [x] Row counts and every pre-existing column are unchanged by the re-ingest (compare against the previous store before deleting it).

**Found on the way:** EB-NeRD's `session_id` restarts per source file — 7,386 ids shared across
train/validation under different users — so ingest qualifies it with the file label, as it does
`impression_id`. See `DECISIONS.md`.
