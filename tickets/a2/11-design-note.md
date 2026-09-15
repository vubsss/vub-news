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
