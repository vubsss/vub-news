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

## Options tried (each a ledger row; the choice is written next to the number that made it)

The frame is where most of the re-ranker's quality is decided, so the variations live here as
*columns side by side*, and ticket 06 reads which ones earn gain. Nothing is chosen by looking at a
label in this ticket; the comparison is the GBDT's per-column gain on `tune`.

- [ ] **Counter window as columns, not a switch:** exposures/clicks/CTR-so-far for every window ticket 03 built (`cumulative`, `1h`, `3h`, `24h`) are all in the frame under suffixed names. Ticket 06 trains once per window *and* once with all four, so "which window" is a measured decision and "do windows add up" is a free one.
- [ ] **History pooling × half-life:** the decayed-profile cosine at the A1-chosen half-life, at half and double it, and with `mean` / `max` / `last-click` pooling — nine columns, named `hist_cos_<pool>_<hl>`. The A1 profile-decay sweep is the prior; the gain table is the posterior.
- [ ] **Freshness encoding is *not* a variation** — the GBDT is invariant to monotone transforms, so raw hours is the only column. Say so in the module docstring, so nobody adds `log_hours` later thinking it was missed.
- [ ] **Frame storage:** `float32` versus `float16` feature columns, parquet row-group size ∈ {64k, 512k} rows — the `feature_bytes` and load-seconds columns of the same ledger rows. The `tune` AUC of ticket 06 at both precisions is the check that `float16` loses nothing; if it does, the loser stays in the ledger.

## Latency and loading

The long frame is the biggest thing A2 materialises (candidates × impressions, ~30 columns), and
it is built again per chunk at submission time (ticket 08), so its build path *is* the serving
feature-lookup path and must be measured as one.

- [ ] The builder streams: impressions are read in chunks (`ingest.per_impression` order, sorted by `impression_time`), features computed per chunk, and each chunk appended as a parquet row group — the whole frame is never in memory during the build. Chunk size ∈ {whole, 50k, 200k impressions} is tried once on MIND `train`: peak RSS and build seconds per size are three ledger rows, and the chosen size is the one written into the registry.
- [ ] Counter lookups are batched per chunk, one `at()` call per window over the chunk's flattened `(article, t)` pairs, sorted by `t` — never one call per impression. The per-chunk lookup ms is recorded next to the chunk size.
- [ ] Vectors for the cosine features are read from the A1 embedding artifact with `np.load(mmap_mode="r")`; a ledger row records the resident-bytes difference against a full load, so ticket 09 can quote what the feature stage keeps in RAM.
- [ ] Readers take a column list: an ablation arm (ticket 07) or the serving path (ticket 09) loads only its tier's columns through `pyarrow` projection, and a test asserts the projected read never touches a forbidden column.
- [ ] Per-stage timing inside the builder — retriever scores, profile cosines, counters, session features — via `timings.sample()`, so the note can say which feature family costs the most to compute, not just that the frame took N seconds.
