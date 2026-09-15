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
