# 11 — The design note, and one-command reproduce

**What to build:** The graded write-up, assembled from the ledger rather than recomputed: what was
built and the options tried at each step with why the chosen one won; baseline against improved
with the ablation table and its intervals; the Q9 pair; serving findings with the per-stage
latency breakdown, byte counts and the K-curve; where 10× breaks. Six pages target, references and
appendices free. And the repository state a grader can reproduce from one command.

**Blocked by:** 09 — the serving section. 10 — the test numbers.

**Status:** ready-for-agent

- [ ] `report/a2-design-note.tex` builds with two `pdflatex` passes into `a2-design-note.pdf`; every number in it traces to a row in `artifacts/tradeoffs.md` or the checkpoint state.
- [ ] Sections: what we built; options tried and chosen; baseline vs improved with ablation + CI; Q9 with-and-without table; serving and scale with the K-curve; where 10× breaks; stated limitations (no position bias on either dataset, no MIND publish time, sessions only where shipped, the literal-cut row measures stage-one recall).
- [ ] `python build.py` from a clean checkpoint runs every stage through `rerank` and `evaluate` with no manual step; README documents it and the separate submission and benchmark commands.
- [ ] `.gitignore` covers `*.zip`, `*.pt`, `*.ckpt`, `__pycache__/`, `data/`; `git status` shows no large file.
- [ ] Leaderboard screenshots for both competitions are in the repo and referenced from the note.
- [ ] The AI-usage log is current to the final commit.

## Options tried — how the note shows them

The note's job is to make every design decision visible as a comparison, so the "options tried"
section is a table per stage, not prose, generated from the ledger:

- [ ] **One decision table per stage** (frame, NRMS, re-ranker, index, serving): columns `option | AUC (CI) | bytes | ms | chosen?`, rows straight from `tradeoffs.md` filtered by `stage`; a `ledger.decisions()` renderer emits the LaTeX so the table and the ledger cannot disagree. The losers are in every table.
- [ ] **Each chosen option has one sentence of why**, citing the row it beat and the axis it won on — and, where it lost on the other axis (a slower model that scored higher, a smaller index that lost recall), that is said in the same sentence.
- [ ] **The four curves** the tickets built are figures: K against AUC and p99 (09); rounds against AUC and model bytes (06); tier build-up against AUC and feature-lookup ms (07); chunk size against rows/s and RSS (08). Plot-ready tables come from the ledger; no number is typed in.

## Latency and loading — how the note reports them

- [ ] **A per-stage serving table** — retrieve / features / nrms / gbdt / total, p50 and p99, bytes resident — from ticket 09, with the cold-against-warm and mmap-against-loaded rows beneath it.
- [ ] **A loading paragraph** that says what is read once per job, once per chunk, and once per request, in the submission path and in the serving path, with the peak-RSS-at-first-and-last-chunk pair from ticket 08 as the evidence that the pipeline is streaming.
- [ ] **The index decision** (flat / IVF / HNSW × precision) is reported with its functional column — recall@K and the literal-cut AUC — in the same table as its latency and bytes, never as a latency-only choice.
- [ ] **The 10× section** is the three rows from ticket 09 and the first thing that breaks, with the multiple at which it does.
