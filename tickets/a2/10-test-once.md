# 10 — `test`, scored once

**What to build:** The numbers the design note leads with. Every arm — the four A1-era retrievers,
NRMS, the full re-ranker, every ablation arm, the literal-cut rows, the leaky arm — scored on the
held-out `test` split in one batch, on one recorded date, after every choice has been made on
`tune` and every comparison read on `validation`. Nothing is selected on it and it is not run
again.

**Blocked by:** 07 — all arms exist. 08 — the submissions are built, so no model changes remain.

**Status:** code built 2026-09-15 — the one run it permits needs the data

- [x] One command runs evaluate and ablation on `test` for both datasets and every arm; its invocation and date are written into the checkpoint state.
- [x] Every arm's `test` row lands in the ledger with the paired interval against the full model, and the full model's against NRMS.
- [x] The three-way check from A1 is repeated: `test` against `validation` against the leaderboard, per dataset, with the sign of each shift stated.
- [~] No code change after this ticket touches a model, a feature or a spec; if one must, the fact that `test` was scored before it is written down, not the number re-run.

## Options tried

None, by design — this ticket tries nothing. Every option was tried on `tune` and every
comparison read on `validation` in tickets 04–09; `test` only confirms the chosen ones. The
markdown states this in its first line so a reader does not look for a sweep.

## Latency and loading

- [x] The batch reuses ticket 07's one-frame-loaded-once path for the `test` split; the run's wall time, peak RSS and arm count are one ledger row alongside the `validation` ablation's, so the two runs can be seen to have cost the same — a cheap check that nothing different ran on `test`.
- [x] The `test` engineering columns (`p50_ms`, `model_bytes`) are copied from the `validation` rows, not re-measured; the ledger row says `measured_on: validation` for them, because a second measurement is a second chance to select.

**What is built and what is measured.** `python -m pipeline.final` is the one command: it builds
the `test` frames the rebuild deliberately does not, scores all five retrievers, runs every
ablation arm, copies the engineering columns off the `validation` rows, and writes
`artifacts/test-scored.json` with the date, the invocation and the commit. `pytest` is green at
567 tests, 19 of them this module's. No number exists yet, because the split does not exist yet.

**The refusal is the feature.** A second run raises `AlreadyScored` and names the date, the commit
and the invocation of the first. `--force` does not overwrite the record — it *appends*, so a
repository where `test` was scored twice says so in a file, and `test-once-<dataset>.md` says
"run **2**" at the top instead of leaving the reader to find it.

**The leaderboard column is a file somebody writes.** Nothing here can compute it; the leaderboard
is holding the labels, which is the point of a leaderboard. `artifacts/leaderboard.json` maps
`<dataset> -> <retriever> -> auc`, and until it exists the three-way table prints **outstanding**
rather than an empty cell — a blank there reads as agreement, which is the one thing it must not
say.

**The last box stays open by its own nature.** "No code change after this ticket touches a model"
is a promise about the future, not a property of a checkout, and the thing that keeps it is the
record: any later change is visible against the commit the record names.

**Found on the way:** `run_arms` reads six frames, not one — train, tune and test, each in both
causalities, because Q9's leaky arms read their own file. A run that built only the causal test
frame would have scored `test`, got most of the way through nineteen arms, failed on the leaky
one, and then been blocked from retrying by this module's own refusal. All six are ensured before
anything is scored.
