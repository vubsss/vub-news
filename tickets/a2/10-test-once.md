# 10 — `test`, scored once

**What to build:** The numbers the design note leads with. Every arm — the four A1-era retrievers,
NRMS, the full re-ranker, every ablation arm, the literal-cut rows, the leaky arm — scored on the
held-out `test` split in one batch, on one recorded date, after every choice has been made on
`tune` and every comparison read on `validation`. Nothing is selected on it and it is not run
again.

**Blocked by:** 07 — all arms exist. 08 — the submissions are built, so no model changes remain.

**Status:** ready-for-agent

- [ ] One command runs evaluate and ablation on `test` for both datasets and every arm; its invocation and date are written into the checkpoint state.
- [ ] Every arm's `test` row lands in the ledger with the paired interval against the full model, and the full model's against NRMS.
- [ ] The three-way check from A1 is repeated: `test` against `validation` against the leaderboard, per dataset, with the sign of each shift stated.
- [ ] No code change after this ticket touches a model, a feature or a spec; if one must, the fact that `test` was scored before it is written down, not the number re-run.

## Options tried

None, by design — this ticket tries nothing. Every option was tried on `tune` and every
comparison read on `validation` in tickets 04–09; `test` only confirms the chosen ones. The
markdown states this in its first line so a reader does not look for a sweep.

## Latency and loading

- [ ] The batch reuses ticket 07's one-frame-loaded-once path for the `test` split; the run's wall time, peak RSS and arm count are one ledger row alongside the `validation` ablation's, so the two runs can be seen to have cost the same — a cheap check that nothing different ran on `test`.
- [ ] The `test` engineering columns (`p50_ms`, `model_bytes`) are copied from the `validation` rows, not re-measured; the ledger row says `measured_on: validation` for them, because a second measurement is a second chance to select.
