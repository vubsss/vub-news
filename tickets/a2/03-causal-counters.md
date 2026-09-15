# 03 — Causal counters, with the test that proves they are causal

**What to build:** Per-article exposure and click counters that can be asked "as of *t*" and
answer using only what happened strictly before *t*, plus each article's first-seen time in the
log (MIND's stand-in for a publish time). Fitted once over a whole log, read causally — the same
global-structure / causal-read pattern the BM25 index already follows. Asking with `t = +inf`
gives the whole-log answer, which is the leaky arm of the Q9 comparison and must be the same
object, not a second implementation.

**Blocked by:** 01 — the counter's bytes and build time are ledger rows.

**Status:** ready-for-agent

- [ ] A `counters` module builds one counter store from a behaviours frame (`impression_time`, `candidate_ids`, `labels`).
- [ ] `at(article_ids, t)` returns exposures-so-far, clicks-so-far and CTR-so-far, counting only impressions with time strictly less than `t` — an impression at exactly `t` is excluded.
- [ ] `first_seen(article_ids, t)` returns the earliest prior appearance strictly before `t`, or null when the article has not been seen — never a time at or after `t`.
- [ ] Window is a parameter: cumulative, or sliding over a duration; both share the interface.
- [ ] **The leakage test, named for Q9:** an impression never sees its own outcome; two impressions at the same instant do not see each other; removing every row after `t` from the log leaves `at(·, t)` unchanged; the whole-log read is ≥ the causal read at every `t`.
- [ ] Ledger rows: counter-store bytes and build seconds per dataset, for cumulative and each sliding window tried.
