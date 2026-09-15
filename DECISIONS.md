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
