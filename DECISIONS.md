# Design decisions — Assignment 2

One entry per decision, newest last. Each names what was chosen, what it was chosen over, and
why — the "why" is the part the design note needs. Numbers that back a choice live in
`artifacts/tradeoffs.md`; this file records the reasoning, not the measurement.

The nine decisions from the planning interview are in `NEW_PLAN.md` and are not repeated here.

---

## 2026-09-15 — The bench work is committed as A1's, not A2's

`pipeline.bench` and the README section it produced were uncommitted A1 work. Committed as one
commit under A1's framing before any A2 code, so the A2 diff starts clean and the bench numbers
(IVF 7.7–14.8× faster for 4–8% of exact top-10; `ann` 0.108 ms/impression on MIND) are already
citable as the stage-one engineering baseline.

## 2026-09-15 — `lightgbm==4.7.0`, the CPU wheel, in both env files

Over XGBoost: LightGBM's native handling of NaN features matters here — session and dwell
features are null on MIND by construction (decision 4 in `NEW_PLAN.md`), and the leaf-wise
growth is the faster of the two on wide, sparse-ish tabular data at this size. The GPU build is
not needed: the trees fit in minutes on the laptop and Ada's GPU is reserved for NRMS.

## 2026-09-15 — The ledger is a flat JSONL with a fixed schema, one row per variant

Over extending `timings.py`'s per-stage rows, and over a per-module markdown each: the note needs
a variant's AUC *and* its bytes *and* its p99 on one line, so the row has to carry both metric
families. A fixed schema (rather than free-form dicts) means a row with a blank side is visible
as a blank cell, which is the failure the ledger exists to prevent. Keyed on
`(dataset, stage, variant, split)`; re-recording a key replaces it, so a re-run never leaves a
duplicate for the renderer to pick between.

## 2026-09-15 — Seed rows come from a fresh `evaluate`, not the JSON on disk

MIND's `artifacts/mind/evaluate/*-validation.json` dated 2026-08-22 recorded `ann` at 0.6400 —
MiniLM's number, from before phase 7 promoted e5 (0.6542 in `checkpoint/phase_9.md`). The
vectors on disk are e5's (dated 08-26). So the stale JSON is regenerated before the seed reads it.
The rule this sets: **a ledger row is recorded by the code that computed the number, in the same
run**; the seed is a one-off for A1's rows and is not the pattern.

## 2026-09-15 — A1's ledger rows leave p50/p99 blank rather than borrowing the bench's mean

`bench-serve` recorded a *mean* marginal ms per impression for `bm25` and `ann`, not percentiles,
and only for the ranking path, not a full request. Putting that in the `p99_ms` column would make
the A1 rows look measured on an axis they were not. It goes in the row's `note`; the percentile
columns wait for ticket 09, which measures every retriever through the same serving path. The
`hybrid` rows have no build cost or throughput at all because the hybrid builds nothing — it fuses
two rankings — and the bench never timed the fusion; a blank says so.

## 2026-09-15 — Session ids are qualified with their source file, like impression ids

EB-NeRD's `session_id` restarts per file. Measured on the re-ingested store: **7,386 session ids
appear in both the train and the validation file under two different users**, and within one file
no session has two users. Left raw, a "clicks earlier in this session" feature would read another
user's clicks. Ingest already qualifies `impression_id` with its file label for the same reason,
so `session_id` takes the same prefix in the same place. The alternative — keying sessions by
`(source, session_id)` everywhere downstream — would have moved the rule into every reader.

## 2026-09-15 — Re-ingest is verified against a backup, column by column, before the old store goes

`--force ingest` rewrites all three tables and does not cascade, so `split` and `preprocess` were
forced with it (they write `split` and `lexical_text` back into the store). Every pre-existing
column on all six tables compared equal to the previous store, row counts unchanged, split sizes
unchanged. `history.click_scroll` needs a NaN-aware comparison — it is identical. The downstream
indexes were not rebuilt: they read columns that did not change.

## 2026-09-15 — One counter store, sorted `int64` keys, read by binary search strictly before `t`

Over two implementations (a running counter for the causal arm, a `groupby` for the leaky one)
and over a per-article list of timestamps: every exposure is one key, `article_position × span +
µs_offset`, kept sorted, so "exposures strictly before `t`" is one `searchsorted(side="left")`
and a sliding window is the difference of two. The whole-log read is the same call with
`t = +inf`, which clamps to the end of the article's slot — Q9's two arms differ only in the
moment asked for, and a test that removes every row after `t` and gets the same answer proves the
causal one. "Strictly" is why an impression at exactly `t` is excluded: it does not see its own
outcome, and two impressions in the same second (28% of MIND's rows and 21% of EB-NeRD's carry a
timestamp an earlier row already carries) do not see each other's.

What this costs: 68.7 MB on MIND and 45.9 MB on EB-NeRD, built in ~4 s and ~3 s, and it is the
same bytes for every window — cumulative and sliding both need the timestamps for a causal read;
the "one int per article" cumulative counter is a serving-time structure that could only answer
`now`. Lookups are 0.2–0.35 ms p50 per impression for every window, so the window choice in
ticket 06 is a functional one, not an engineering one.

Two things a reader can lean on: an id the log never showed reads 0 / 0 / NaN (not 0 CTR — the
re-ranker can tell "never shown" from "shown, never clicked"), and the whole-log read with a
window is refused rather than answered, because a window ending at infinity is not a thing.

## 2026-09-15 — The availability tier *is* the module's structure

`features.FEATURE_GROUPS` maps `content` / `history` / `exposure` / `clicked` onto the column
names in each, and `check_columns` refuses a frame whose columns do not partition exactly over
them. Over a comment naming the tiers, and over four separate builders: ticket 07's ablation
drops a tier per arm, so a feature in no tier is one no arm can drop and a feature in two is one
that two arms would each claim to have dropped — and neither failure shows up in a number. The
same check refuses `read_time`, `scroll_percentage`, `next_read_time` and `next_scroll_percentage`
by name *and* as a suffix, so a column called `mean_read_time` cannot carry the impression's own
outcome in under a different name. It runs inside `frame_for`, so it fails the build rather than
being available to it; a test injects a leaking feature family to prove that wiring.

## 2026-09-15 — Recency is columns side by side, not a chosen scheme

A1 picked one weighting scheme per dataset because a retriever ranks by one profile. A tree does
not: it can read seven profiles at once and tell us which earned gain. So the frame carries
`hist_cos_mean_<profile>` and `hist_cos_max_<profile>` for uniform, two position decays, three
time half-lives (6/24/72 h) and engagement — and a profile the dataset cannot express comes out
NaN, decided by `weighting.available` off the registry's ColumnMap rather than by a dataset name.
Ticket 06 reads the gain table; this ticket chooses nothing.

Two non-variations are recorded so nobody adds them later thinking they were missed. Freshness is
raw hours only: a GBDT splits on thresholds and is invariant to monotone transforms, so `log_hours`
would split one column's gain in two and buy nothing. And `last` pooling gets one column rather
than one per profile: it is the similarity to a single click, and a profile's weight on it is one
positive per-impression scalar that says nothing about the candidate. `mean` and `max` do get one
per profile — the weights change the average and change the argmax.

## 2026-09-15 — The leaky arm shifts its window forward rather than widening it

Q9's cumulative arm is `at(ids, WHOLE_LOG)`, which the counter store already answers. A *sliding*
window has no whole-log reading — "the last hour" of a moment past the end of the log is still an
hour — so the leaky arm reads `[t, t + window)` where the causal one reads `[t - window, t)`: the
same width, moved onto the future the server could not have had, and including the impression's
own outcome. Freshness leaks the same way, through the article's first sighting anywhere in the
log instead of its first sighting before `t`. One store, one flag, no second implementation.

## 2026-09-15 — The frame's storage is a registry spec, and its axes are engineering-only

`FeatureSpec(chunk_impressions, row_group, precision)` sits beside the functional specs because
none of its three fields changes a number in the frame. Chunk size is what peak RSS is a function
of — a chunk is built, written as one parquet row group and dropped, so the frame is never whole
in memory — and it is therefore *not* in the ledger's variant name: a name implying otherwise
would invite someone to compare two chunk sizes' AUC. Precision and row-group size are in the
name, because `float16` halves the largest file A2 writes and whether that costs AUC is a real
question, answered in ticket 06 by training on both files.

The cosine features read the embedding matrix with `np.load(mmap_mode="r")` and gather a few dozen
rows per impression. The retriever-score features do not: they come from `rank_candidates`, the
harness's own call, which loads its own copy. Recomputing those scores here off the mmap would be
a second implementation of a published number, which is the trade this records rather than hides.

## 2026-09-15 — NRMS is a fourth entry in `RETRIEVERS`, and `STAGE_ONE` is the other three

The harness reaches a retriever only through `rank_candidates`, so the baseline gets slicing,
bootstrap intervals and beyond-accuracy metrics by being an entry in the table. What that exposed
is that two A1 sweeps were fanning out over *everything the harness can score* — `sweep` retrieves
from the corpus at depth, `profile_sweep` varies the pooling — and neither is a thing a trained
re-ranker has. So `evaluate.STAGE_ONE` names the three that pool a query over a click window and
rank the whole corpus, and the sweeps default to that; `RETRIEVERS` stays the harness's table.
One list, in the module the others already import.

`nrms.rank_candidates` refuses rather than ignores. The click window is inside the trained
weights, so a call at another window raises instead of scoring; `pooling` raises too, because a
learned user encoder has no aggregator and silently ignoring the argument would report a sweep's
cell under a name that never ran.

## 2026-09-15 — The stacking boundary is a timestamp, not a row count

NRMS fits on the earlier half of `train` and the GBDT on the later half, so the GBDT's NRMS
feature is a prediction about impressions the NRMS never saw. `nrms.halves` cuts at the timestamp
at the fraction's position and puts every impression stamped at that instant on one side: a cut by
row count would leave two impressions of the same second in different halves, which is the same
leak in miniature, and a random split would put the same day's popularity in both.

## 2026-09-15 — "Attend here" and "there is a vector here" are two flags

Found by the test rather than by reading. A cold user's history is all padding, and a fully masked
row makes the attention's softmax NaN — so position 0 stays attended. The first implementation left
its row index at 0, which handed that user *article row 0's vector* as their entire history: a
wrong feature, on exactly the users a cold-start slice is about, invisible in every metric. The
padding row is now -1 and `gather` zeroes any position that is unattended **or** has no vector, so
a cold user scores every candidate alike and the ranking falls back to the order the candidates
arrived in — which is what the A1 retrievers do with a user they know nothing about.

## 2026-09-15 — The paper's number is a placeholder, not a guess

`nrms.PAPER["auc"]` is `None` and the grid's markdown says the comparison is outstanding. The
ebnerd-benchmark figure has to be read off the paper; a number typed from memory into a design
note is worse than a stated gap, because it looks like a measurement. Filling it in is one line
and the renderer already prints the gap either way — including when this reproduction lands below
its source, which is a finding rather than something to leave out.
