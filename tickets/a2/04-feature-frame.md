# 04 — The feature frame, with the test that keeps the future out

**What to build:** One long frame per split — a row per `(impression, candidate)` with the label
and every behavioural feature the re-ranker will see — built from the feature store, the two
retrievers' scores, the decayed user profile and the causal counters. Features are grouped by
availability tier and the grouping is the module's structure, not a comment: `content`
(retriever scores, category match with history), `history` (click count, decayed-profile cosine,
last-click cosine, past-click dwell and scroll statistics), `exposure` (exposures-so-far,
freshness, impressions and clicks earlier in the same session), `clicked` (clicks-so-far,
CTR-so-far). The ablation in ticket 07 iterates that grouping.

**Blocked by:** 02 — session features need the session column. 03 — exposure and clicked
features need the counters.

**Status:** ready-for-agent

- [ ] A `features` module exposes `FEATURE_GROUPS`, a mapping from tier name to column names, and a builder that returns the long frame for a dataset and split.
- [ ] Retriever scores are read back through `ranked_ids` — both retrievers return candidates in ranked order, not arrival order.
- [ ] Freshness is `impression_time − published_time` where the dataset has one and `impression_time − first_seen` where it does not; the choice comes from the registry, not a dataset check.
- [ ] Recency-weighted history features are computed for more than one half-life and more than one pooling, each its own column, so the GBDT's per-column gain is the comparison.
- [ ] Dwell and scroll features come from past clicks only; on MIND, where those arrays are null, the columns are null and the frame still builds.
- [ ] **The Q9 test:** rebuilding the frame from a log with every row after `t` removed leaves the rows at `t` identical; `next_read_time`, `next_scroll_percentage` and the impression-row `read_time` / `scroll_percentage` are named as forbidden, and the build fails if any of them reaches the frame; every column belongs to exactly one tier in `FEATURE_GROUPS`.
- [ ] Row count equals the sum of candidate counts for the split; `pytest` is green.
- [ ] Ledger rows per dataset and split: frame bytes, rows, build seconds, peak RSS.
