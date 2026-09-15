# Handoff: Assignment 2 is built, and none of it is measured

**Written for:** whoever picks this up on a machine that has the MIND and EB-NeRD data — and for
the Claude Code session helping them.

Read this before running anything. The short version: **every line of A2 is written and tested,
and there is not a single measured number in the repository.** The machine this was built on has
977 MB of free disk and no dataset, so nothing has ever been run against real data. Your job is
almost entirely *running things in the right order and filling in blanks* — not writing modules.

---

## 1. What state this is in

| | |
|---|---|
| Branch | `main`, clean |
| Tests | **590 passed, 11 skipped** — `.venv/bin/python -m pytest -q` |
| Tickets 01–11 | code complete; **04–11 have no numbers** |
| Data present | **none** — no `data/`, no `feature_store/`, no `artifacts/` |
| Last commits | `fd81ff4` submission · `8a05040` serve · `d25ba92` final · `4d1f0e0` report |

Every ticket file in `tickets/a2/` ends with a **"What is built and what is measured"** section
naming exactly which of its boxes are code and which are waiting on a run, plus the command that
fills them. Those sections are the authoritative per-ticket status; this file is the order to do
them in.

### The one convention that matters most

**No number is ever typed by hand, anywhere.** Each module records its own results as a row in
`artifacts/tradeoffs.jsonl` at the moment it computes them, the markdown and the LaTeX are both
generated from that file, and a `—` means *not measured* rather than zero. If you find yourself
about to write a figure into a document, that is the signal that a command has not been run yet.

Corollary: **do not fabricate a number to make a table look finished.** Two places deliberately
print "outstanding" instead of a value, and they should stay that way until someone supplies the
real thing (§5).

---

## 2. Before anything runs

### 2a. The HuggingFace token — required for MIND

MIND is behind a gated mirror. I verified this: every archive URL returns **401** without a token.

```bash
echo 'HF_TOKEN=hf_...' >> .env      # in the repo root; gitignored
```

The token's owner must have accepted the terms at <https://huggingface.co/datasets/yjw1029/MIND>
while logged in — a valid token from an account that has not is still refused. EB-NeRD needs
nothing; its S3 bucket is open.

### 2b. Disk

Verified sizes (I HEAD-requested them):

| archive | bytes | needed for |
|---|---|---|
| `ebnerd_small.zip` | 84 MB | the rebuild |
| 4 × EB-NeRD embedding artifacts | 1.22 GB | the vector comparison |
| `ebnerd_testset.zip` | **1.63 GB** | tickets 08/10 |
| `MINDsmall_{train,dev}.zip` | ~80 MB (gated, unverified) | the rebuild |
| `MINDlarge_test.zip` | ~1 GB (gated, unverified) | tickets 08/10 |

Plus the feature store, which is the big one and is *generated*. Budget **~60 GB** for both
datasets end to end. `VUB_NEWS_DATA_ROOT` moves `data/` and `feature_store/`;
`VUB_NEWS_ROOT` moves `artifacts/`, `predictions/` and `.checkpoints/`. They move independently —
see `pipeline/paths.py` and `ada/README.md`.

### 2c. Environment

`environment.yml` / `requirements.txt` are the real spec. What I used here was a `.venv` on top of
system site-packages because this box has no conda; **do not copy that setup**, use the conda env.

Two things to check on your machine:

- **`gensim` is missing** from what I had. It is needed only for the `word2vec-google-news-300`
  arm of MIND's encoder comparison (and that model is another ~3.6 GB). Everything else imports.
- **pandas.** This box has 3.0.5 against the pinned 2.3.3 and the suite is green on both, but the
  pinned one is what the numbers should come from.

---

## 3. The run order

Each step lists what it fills. Do them in this order; later ones read earlier ones' outputs.

### Step 1 — the rebuild

```bash
python build.py                    # or: sbatch ada/build.sbatch
```

Runs: `acquire ingest split counters preprocess bm25 embed ann features nrms rerank evaluate
predict`. Checkpointed per stage under `.checkpoints/<dataset>/`, so an interrupted run resumes.

**Fills:** the feature store, the indexes, the counter store, the NRMS checkpoint, the re-ranker,
and `validation` evaluations for all five retrievers.

### Step 2 — the leaky frames (easy to miss)

```bash
python -m pipeline.features --leaky --split train --split tune --split validation
```

The ablation reads **six** materialised frames — `train`, `tune` and the scored split, each in a
causal and a leaky copy — and `build.py` builds only the three causal ones. Without this, step 3
fails partway through.

(`pipeline.final` builds its own six, for a specific reason: a run that spends the held-out split
and *then* fails on a missing frame is blocked from retrying by its own refusal.)

### Step 3 — the sweeps that choose things, on `tune`

```bash
python -m pipeline.nrms --grid                     # or sbatch ada/nrms.sbatch --grid
python -m pipeline.rerank --grid
python -m pipeline.ablation --split tune
```

**Fills:** every option-tried row in the ledger. **Then read them and update the registry** —
`NrmsSpec` and `RerankSpec` in `pipeline/datasets.py` currently hold *unmeasured defaults*, and the
whole point of the design is that the registry holds the cell that won its sweep, with the tune
number that chose it written in the comment beside it (follow the style of `HybridSpec` in the MIND
entry, which does this properly for A1).

> If a sweep's winner differs from what the registry holds, `report.py` will tick the registry's
> choice and show it losing. That is intentional — see §6 — but it is a thing to notice and
> resolve, not to leave.

### Step 4 — report on `validation`

```bash
python -m pipeline.rerank --headline               # rerank − nrms, paired
python -m pipeline.ablation --split validation     # the 19 arms + the Q9 pair
```

**Fills:** the headline delta, the ablation table, the recall@K bridge, the five-retriever table.

### Step 5 — the serving benchmark

```bash
python -m pipeline.serve --dataset mind
python -m pipeline.serve --dataset ebnerd
```

**Fills:** per-stage p50/p99, the byte table, the cost per thousand queries, the K-curve, the three
10× rows, `artifacts/bench-serve-<dataset>.md`. Run it **after** step 4, because the K-curve joins
its own latency to the ablation's cut-arm AUC out of the ledger.

Two knobs in `pipeline/serve.py` you should look at before quoting anything:

- `CORE_HOUR_USD = 0.0425` with its source named in `CORE_HOUR_SOURCE`. Substitute your own if you
  prefer; the arithmetic string travels with the number.
- `NODE_RAM_GB = 100`, `NODE_CORES = 32` — Ada's practical ceiling, from `ada/README.md`. If you
  run elsewhere, change these or the 10× conclusion describes the wrong machine.

### Step 6 — the submissions

```bash
# measure first, then run at the size that won
python -m pipeline.predict --retriever rerank --dataset mind --bench-chunks
sbatch ada/predict-rerank.sbatch --dataset mind --chunk <winner>
sbatch ada/predict-rerank.sbatch --dataset ebnerd --chunk <winner>
```

`--bench-chunks` ranks only the first chunk at each of {50k, 200k, 500k}, records three ledger rows
and **writes no submission** — the size is chosen before the run rather than justified after it.

Then: `rsync --no-compress --whole-file` the zips back, check `md5sum` on both ends before either
is called ready, upload, screenshot into `screenshots/`.

### Step 7 — `test`, once

```bash
python -m pipeline.final
```

Scores every retriever and every arm on `test`, copies the engineering columns off the `validation`
rows, and writes `artifacts/test-scored.json` with the date, the invocation and the commit.

**It refuses to run twice.** `--force` appends to the record rather than replacing it, so a
repository where `test` was scored twice says so in a file and on the generated page. Do not run
this until steps 3–6 are done and no model change remains.

### Step 8 — the note

```bash
python -m pipeline.report
cd report && pdflatex a2-design-note.tex && pdflatex a2-design-note.tex
```

`report/a2-design-note.tex` has every section and **no numbers** — each table is an `\inputtable`
that `report.py` generates from the ledger. What remains is the *prose*: one sentence of why per
chosen option, citing the row it beat and the axis it won on, and saying where it lost on the other
axis.

---

## 4. Traps I hit, so you do not

| | |
|---|---|
| **Leaky frames** | Step 2 above. The ablation needs six frames; the rebuild makes three. |
| **`final` refuses twice** | By design. If it fails partway, check `artifacts/test-scored.json` before re-running — the refusal is doing its job. |
| **NRMS window is in the weights** | A checkpoint fitted at history length *k* refuses to score at any other. `NrmsSpec.history_length` defaults to `retrieval.HISTORY_K`; if you change one, change both or retrain. |
| **`--bench-chunks` before the full run** | Not after. A size chosen after the run is a size that was not chosen. |
| **EB-NeRD's `ENGAGEMENT_WINDOW = 100`** | Truncation happens *before* the history join, which is a 31× duplication on that dataset. The first attempt at this was OOM-killed at 10 GB. Raising it past 100 means re-running ingest. |
| **`/scratch` is purged weekly on Ada** | `ada/build.sbatch` stages out in a `trap`, because job 2674571 lost eight stages' worth of feature store to exactly this. |
| **A `—` in the ledger is not a zero** | It is a ticket that is not done. |

---

## 5. Things that need a human, not a command

Three numbers cannot come from this repository and are deliberately left printing "outstanding":

1. **`nrms.PAPER["auc"]` is `None`.** The ebnerd-benchmark paper's reported NRMS-DocVec AUC on
   `ebnerd_small`. Read it off the paper and set it; the grid's markdown computes the gap the
   moment it is set, and prints "Outstanding" and names what has to be read until then.

2. **`artifacts/leaderboard.json`** does not exist. After uploading, write what CodaBench returned:

   ```json
   { "mind": { "rerank": 0.6431, "nrms": 0.5980 },
     "ebnerd": { "rerank": { "auc": 0.71 } } }
   ```

   Either spelling works. Until it exists, the three-way check (validation → test → leaderboard)
   prints **outstanding** for the last step — a blank cell there would read as agreement.

3. **The screenshots.** `screenshots/`, referenced from the note. Two competitions:
   MIND `codabench.org/competitions/13967/`, EB-NeRD `codabench.org/competitions/2469/`.

And one that needs judgment rather than data: **the registry's chosen cells** (§3 step 3).

---

## 6. Conventions to keep

These are load-bearing. Breaking one produces something that looks right and is not.

- **No stage branches on a dataset name.** Every MIND/EB-NeRD difference lives in
  `pipeline/datasets.py`. A dataset lacking a column carries it **null**, never missing. If a stage
  tempts you to write `if dataset == "mind"`, the fix belongs in the registry.
- **Chosen on `tune`, reported on `validation`, `test` scored once.** Early stopping reads `tune`
  whatever split the arms are reported on — see `ablation.run_arms`, which takes three splits.
- **One ledger row per variant, both metric families.** Functional *and* engineering on the same
  row. A blank side is an unfinished ticket, and the renderer shows it as one.
- **Counters are read strictly before `impression_time`.** The leaky arm is the same store read at
  a later moment — one flag, same model. Note that the leaky *sliding* window is `[t, t+1h)` and is
  **not** a superset of the causal `[t-1h, t)`; it can legitimately read smaller.
- **One frame builder.** `features.frame_for` takes its scorers as an argument so the submission and
  the harness share it. A second assembly for the competition period would be forty columns written
  twice, and the failure mode is a booster fed a positionally-shifted feature vector — a well-formed
  leaderboard file with a meaningless score.
- **The ✓ in the note follows the registry, not the best row.** If the chosen option lost its
  sweep, the table must be able to show that.
- **`DECISIONS.md` gets an entry** for anything decided against a real alternative, with the
  alternative named. There are ~20 entries; match the style.
- **`AI_logs/`** needs to be current to the final commit (ticket 11).

---

## 7. Two real bugs the tests caught, as evidence the tests are worth keeping

Both were invisible in every metric, which is the point:

- **A cold NRMS user was being handed article row 0's vector.** Padding a history with row index 0
  *and* marking that position attended collapses two different facts. The fix is a `-1` row that
  `gather` zeroes. It would have distorted precisely the cold-start slice.
- **A served request was re-reading its BM25 index from five files, every request.**
  `test_the_request_path_opens_no_file` found it. `rank_candidates` opens its own index inside the
  call — correct for a 200,000-impression chunk, catastrophic for one request, and it sits *inside
  the feature stage's timer*, so the design note would have concluded that feature lookup is the
  bottleneck. The fix is the uniform `stores` argument on all three retrievers.

If a test in `tests/` fails after a change, read what it is asserting before changing it. Several
of them encode a fact that took a while to establish.

---

## 8. Quick verification

```bash
.venv/bin/python -m pytest -q          # expect: 590 passed, 11 skipped
python build.py --plan                 # the stage table, changes nothing
python -m pipeline.ledger              # renders artifacts/tradeoffs.md
```

`DECISIONS.md` is the why for anything in the code that looks odd. `tickets/a2/*.md` is the what,
per ticket, with its own status section at the bottom. `ada/README.md` is the cluster, including
several things about that account that were measured rather than read off the user guide.
