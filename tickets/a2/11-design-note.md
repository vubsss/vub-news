# 11 — The design note, and one-command reproduce

**What to build:** The graded write-up, assembled from the ledger rather than recomputed: what was
built and the options tried at each step with why the chosen one won; baseline against improved
with the ablation table and its intervals; the Q9 pair; serving findings with the per-stage
latency breakdown, byte counts and the K-curve; where 10× breaks. Six pages target, references and
appendices free. And the repository state a grader can reproduce from one command.

**Blocked by:** 09 — the serving section. 10 — the test numbers.

**Status:** structure built 2026-09-15 — the prose and every number wait on the runs

- [x] `report/a2-design-note.tex` builds with two `pdflatex` passes into `a2-design-note.pdf`; every number in it traces to a row in `artifacts/tradeoffs.md` or the checkpoint state.
- [x] Sections: what we built; options tried and chosen; baseline vs improved with ablation + CI; Q9 with-and-without table; serving and scale with the K-curve; where 10× breaks; stated limitations (no position bias on either dataset, no MIND publish time, sessions only where shipped, the literal-cut row measures stage-one recall).
- [x] `python build.py` from a clean checkpoint runs every stage through `rerank` and `evaluate` with no manual step; README documents it and the separate submission and benchmark commands.
- [x] `.gitignore` covers `*.zip`, `*.pt`, `*.ckpt`, `__pycache__/`, `data/`; `git status` shows no large file.
- [~] Leaderboard screenshots for both competitions are in the repo and referenced from the note.
- [~] The AI-usage log is current to the final commit.

## Options tried — how the note shows them

The note's job is to make every design decision visible as a comparison, so the "options tried"
section is a table per stage, not prose, generated from the ledger:

- [x] **One decision table per stage** (frame, NRMS, re-ranker, index, serving): columns `option | AUC (CI) | bytes | ms | chosen?`, rows straight from `tradeoffs.md` filtered by `stage`; a `ledger.decisions()` renderer emits the LaTeX so the table and the ledger cannot disagree. The losers are in every table.
- [~] **Each chosen option has one sentence of why**, citing the row it beat and the axis it won on — and, where it lost on the other axis (a slower model that scored higher, a smaller index that lost recall), that is said in the same sentence.
- [x] **The four curves** the tickets built are figures: K against AUC and p99 (09); rounds against AUC and model bytes (06); tier build-up against AUC and feature-lookup ms (07); chunk size against rows/s and RSS (08). Plot-ready tables come from the ledger; no number is typed in.

## Latency and loading — how the note reports them

- [~] **A per-stage serving table** — retrieve / features / nrms / gbdt / total, p50 and p99, bytes resident — from ticket 09, with the cold-against-warm and mmap-against-loaded rows beneath it.
- [~] **A loading paragraph** that says what is read once per job, once per chunk, and once per request, in the submission path and in the serving path, with the peak-RSS-at-first-and-last-chunk pair from ticket 08 as the evidence that the pipeline is streaming.
- [~] **The index decision** (flat / IVF / HNSW × precision) is reported with its functional column — recall@K and the literal-cut AUC — in the same table as its latency and bytes, never as a latency-only choice.
- [~] **The 10× section** is the three rows from ticket 09 and the first thing that breaks, with the multiple at which it does.

**What is built and what is measured.** `report/a2-design-note.tex` has every section the ticket
names, compiles on its own, and types **no number at all**: each table is `\inputtable{...}`, and
`python -m pipeline.report` generates those files out of `artifacts/tradeoffs.jsonl`. A table whose
run has not happened renders as *"no rows recorded"* rather than as an empty tabular, and a table
the note asks for that the generator does not write renders as a warning in the PDF —
`test_the_note_inputs_a_table_for_every_file_the_generator_writes` keeps the two halves in step.
`pytest` is green at 590 tests, 23 of them this module's.

**Two properties of the generator worth naming.** The \(\checkmark\) follows the *registry*, not
the best row: the chosen option is the one the code will run, and if that is not the one that won
its sweep the table shows it losing. A registry entry with no row at all — a configuration nobody
measured — is flagged in the file as a `WARNING` comment rather than left as a missing cell. And
every variant name is escaped, because they are full of TeX: `cut@100`, `drop:none`,
`content+history`.

**The `[~]` boxes are prose and screenshots, which is the half this checkout cannot do.** "One
sentence of why per chosen option" needs the option to have won something; the serving table, the
loading paragraph, the index decision and the 10× section all have their generators and no rows to
render. The screenshots need two uploads. The AI-usage log is current to the commit that carries
this line and has to be brought forward again at the end.

**One structural decision.** The tables live in `report/tables/` inside the repository rather than
under `artifacts/`, which is gitignored — a note whose tables vanish on a clean checkout is not a
note a grader can build.
