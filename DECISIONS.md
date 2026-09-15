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

## 2026-09-15 — The re-ranker trains from the stored frame and *computes* it to serve

Two paths on purpose, and they are the two different questions. Training reads the materialised
frame a parquet row group at a time with the arm's column projection: it is the largest read in
the project, and the projection is the whole engineering claim of the tier structure — an arm that
drops a tier does not pay to read it. `projection_rows` measures that by reading the same rows
twice rather than asserting it, because parquet's column layout is what makes it true.

Scoring goes through `features.frame_for` per chunk, which is what the submission does and what a
server would do — so the per-impression milliseconds on the ledger row are the served number and
not a read off a table somebody prepared earlier. A `rank_candidates` that loaded the stored split
would have been faster and would have measured nothing.

## 2026-09-15 — `rerank.ranker` refuses rather than assembling the wrong period's features

`predict` reaches every retriever through `ranker(articles, config, workdir)`. The re-ranker cannot
answer that on its own: the competition's impressions are a later week over its own catalogue, so
their counters, freshness and retriever scores come from files the feature store does not hold.
The `Ranker` therefore takes a `build_frame` callable, and `ranker` without one raises and names
ticket 08. The alternative — assembling features from the feature store's log for impressions a
week later — produces a well-formed submission ranked on numbers that describe another period,
which is the class of failure this project keeps trying to make impossible rather than unlikely.

## 2026-09-15 — The leaky sliding window is the hour *after* `t`, so it is not a superset

Found while testing the Q9 pair through the re-ranker. The cumulative leaky counter is the whole
log and dominates the causal read everywhere; a *sliding* one does not, and should not. The causal
hour is `[t - 1h, t)` and the leaky one is `[t, t + 1h)` — the same width, moved onto the future
the server could not have had, and closed on the left so it contains the impression's own outcome.
So the two arms are compared for difference on the sliding columns and for size on the cumulative
one, and the tests say which is which. A "leaky = bigger number" intuition would have quietly made
the sliding arms look like a bug.

## 2026-09-15 — A cut is scored, not trained, so the cut arms share one model

Every arm that differs only in `top_k`, or in which index supplied the top-K, grows the same trees
from the same rows. The ablation trains once per *training identity* and scores that model several
times. Not a shortcut: retraining would produce identical trees and would invite a reader to think
the difference between two cut rows included a difference in fitting. The saved model bytes are
therefore equal across those rows, which is the fact, and the table shows it.

The cut itself is against the retriever's **corpus** top-K, not against the candidate list's own
order. A production system retrieves K out of the catalogue and re-ranks those, so a logged
candidate stage one would never have surfaced is one the user would never have seen — and that is
the same ranking `retrieval.recall_at_k` measures, which is why the cut rows and the recall table
are printed next to each other. `rerank.ranked_from` refuses a cut without those ranks rather than
falling back to the in-impression rank, which is a weaker claim wearing the same name.

## 2026-09-15 — Early stopping reads `tune` whatever split the arms are reported on

`ablation.run_arms` takes three splits, not two: it fits on the later half of `train`, stops on
`tune`, and scores on whichever split it was asked for. The obvious implementation — stop on the
split being scored — would have selected the number of rounds on `validation` for every arm in the
validation table. A1's rule carries over unchanged, and a test asserts the stopping rows never
intersect the scored ones.

## 2026-09-15 — `bench.ivf_index` is one definition, because two numbers describe one index

The ablation's `cut@K (ivf)` arm reports the AUC an approximate index costs; `bench` reports the
latency it saves. Those are only comparable if they are the same index, so the nlist/nprobe rule
moved into `bench.ivf_index` and both call it. Duplicating four lines of faiss construction would
have let the two tables drift into describing different indexes under one name.

## 2026-09-15 — A submission's clicks come from the histories, and its exposures from the log

`counters.ServingCounters` is three sources behind the harness's two methods, and which three is
the honest answer to "what does a live server know at test time". The training log is read whole,
because all of it precedes the test period. The test log's own impressions count as **exposures**,
read strictly before `t` with the same strictness the offline frame uses. And the only **clicks**
the test period contributes are the ones the competition's own user histories reveal.

No click of a test impression is ever counted, because the leaderboard is holding those back —
that is the thing it is scoring. A real server would have them, we do not, and the note says so
rather than the code quietly inferring a click from an exposure.

The history clicks count **cumulatively and in no window**. MIND's histories carry no timestamp,
so placing one in a particular hour would be inventing the moment it happened; dropping it would
be discarding a click the server has. Cumulative is the only window the data supports.

## 2026-09-15 — One frame builder, so the served columns are the trained columns

`features.frame_for` now takes its scorers as an argument. Offline they are the five retrievers
over the feature store's index; for a submission they are rankers over the competition's own
catalogue. Everything else — the forty columns, their order, the causal counter reads — is the
same function.

The alternative was a second assembly for the competition period, which is where this would have
gone wrong: forty columns written twice, drifting apart on the next change, and the failure it
produces is a LightGBM booster fed a positionally-shifted feature vector. That is a well-formed
leaderboard file with a meaningless score, and no test of either half alone would catch it.
`test_the_submission_and_the_harness_are_one_code_path` compares the two paths' scores directly.

## 2026-09-15 — A submission's labels are −1, not 0

`features.UNKNOWN_LABEL` is −1. A zero in that column is the claim that nobody clicked the
candidate, which is a statement about data nobody has. Nothing in scoring reads the column; it is
in the frame because the frame has one schema, and it holds the one value that means "not known".
The same reasoning makes `session_clicks` null on a test chunk while `session_impressions` is a
real count: the server saw the earlier impressions, it did not see their outcomes.

## 2026-09-15 — `--device` is not given to a retriever that has none

Three of the five retrievers have no device to score on. `predict.build_ranker` inspects the
signature and passes `device` only where it means something, rather than widening five interfaces
for two. The failure it avoids is a `--device cuda` that BM25 accepts and ignores, which reads as
"honoured" in a log and in a design note.

## 2026-09-15 — The serving benchmark found that a request was re-reading its index

`test_the_request_path_opens_no_file` is the kind of test that either passes trivially or finds
something serious. It found something serious: every single-user request was opening nine files —
the BM25 index's five, the embedding matrix and its id index, the article catalogue — because
`features.frame_for` gets its retriever scores through each module's `rank_candidates`, and that
function loads its own index inside the call.

That is the right design for the frame builder, which is handed 200,000 impressions at a time and
amortises the load across all of them. For one request it is the whole cost. And it would not have
shown up as a bug: the load happens inside the `features` stage's timer, so the benchmark would
have reported it as feature-lookup latency and the note would have concluded that the feature
stage is the bottleneck and that a faster index would not help.

The fix is one optional `stores` dict on `rank_candidates`, the same argument on all three
retrievers so the harness's single signature stays single, each reading the keys it needs.
`pipeline.serve` opens the index once and passes it in; the scoring arithmetic is untouched, which
`test_the_served_scorers_agree_with_the_harness` asserts score for score. The alternative — a
serving-only scoring function — would have been a second implementation of a published number.

## 2026-09-15 — A variant that misses the SLA is not given a price

`serve.cost_per_1000` returns `None` for the dollar figure when the variant's p99 exceeds the
budget, and says so in the arithmetic string. A cheap row beside a compliant one, with nothing
saying the cheap one is not allowed, is the shape of a table that gets misread — and the cost is
quoted *at* an SLA, so a variant that does not meet it has no cost at that SLA.

The arithmetic travels with the number as a string for the same reason every other figure in this
project traces to a row: `1000 / qps = core-seconds / 3600 x $/core-hour`, with the core price and
its source named in the markdown, so a reader can substitute their own and get their own answer.
