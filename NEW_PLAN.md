# NEW_PLAN — Assignment 2: Learning from Click-Logs (due 2026-09-20)

## Context

A2 (`ass1/A2.pdf`) extends the A1 pipeline in `vub-news/` with behavioural signals: engineered
click-log features (Q1), a two-stage re-ranker (Q2), an official baseline reproduced then beaten
with an ablation + paired bootstrap CI (Q3), serving/scale analysis (Q4), the full extended
evaluation with slices and CIs (Q5), a 6-page design note (Q6), both Codabench submissions with
screenshots, and an anti-gaming pair (Q9: metrics with and without serving-unavailable features,
plus a test asserting no future-click leakage).

A1 leaves us well placed: `ann` (e5+centre on MIND, word2vec+abtt on EB-NeRD, k=80 decayed
history) is the selected retriever, on both leaderboards (0.6578 / 0.5967); `evaluate.py` already
computes AUC/MRR/nDCG@5/10, diversity, novelty, coverage, cold/warm + head/tail slices, and
1000-resample bootstrap CIs; `timings.py` + `bench.py` already measure per-stage RSS, index bytes
and per-impression latency; `predict.py` streams the large test sets in chunks; `ada/` runs it all
on the cluster. `split/ticket_10.md` already worked out the availability tiers and the Q9 pair.
382 tests pass at `d31a48b`.

Five days. The plan is one coherent track, not two.

## Decisions settled in the interview (do not re-litigate)

| # | decision |
|---|---|
| 1 | **One track (B).** Baseline = NRMS-DocVec reproduced. Improvement = LightGBM re-ranker over {retriever scores, NRMS score, causal behavioural features}. Ablation = drop feature groups, paired bootstrap each. |
| 2 | **Own PyTorch NRMS-DocVec** in `pipeline/nrms.py`, a fourth entry in `evaluate.RETRIEVERS`, over the same vectors A1 chose. No `ebrec`. Compared against the ebnerd-benchmark paper's reported number. |
| 3 | **Two-stage at eval = (c):** re-ranker scores every logged candidate with retriever scores as features (headline); stage-one recall@K ∈ {50,100,200} reported as the bridge; one extra ablation row with a literal top-K cut (outside-K ranked last) to show what a hard cut costs. |
| 4 | **Sessions via the feature store:** `session_id` added to `BEHAVIOR_COLUMNS` through `ColumnMap` (null on MIND, like `published_time`); both stores re-ingested. Session features are EB-NeRD-only; LightGBM takes the NaN. Position bias: unavailable on both, stated not faked. |
| 5 | **Q9 pair:** same GBDT, one flag — article popularity counters read over the *whole log* (future-inclusive, the leaky arm) vs strictly before `impression_time` (the clean arm). Mirrors the RecSys 2024 organisers' leakage ablation. |
| 6 | **Stacking leak:** `train` split chronologically into halves — NRMS on the earlier half, GBDT on the later half. Tune on `tune`, report on `validation`. Free ablation: GBDT ± NRMS feature. |
| 7 | **`test` scored once**, one batch, all arms, on the last day. Nothing selected on it. |
| 8 | **Submissions:** GBDT clean arm to both leaderboards first; NRMS files only if both GBDT files are up by Sept 18. |
| 9 | Partner arrangement is out of scope for this plan; work continues on `main`, linear history. |

Assumptions carried: NRMS trains with in-impression negatives (1:4), epochs/lr chosen on `tune`;
laptop CPU suffices at small scale (≤225k training impressions over precomputed vectors), Ada is
the fallback. GBDT objective is binary logloss (AUC is the headline and leaderboard metric);
`lambdarank` is a single tune-split comparison if time allows, not a plan item. MIND freshness =
`t − first_seen_in_log` (earliest prior appearance as a candidate, computed causally) because MIND
has no `published_time`. New dependency: `lightgbm` (CPU wheel) in `requirements.txt` /
`environment.yml`.

## Operating principles (these shape every phase)

**1. Every choice is a measured comparison, and the losers are kept.** Grading is on ablation
rigour and design-note clarity, so the note must *show* improvement, not assert it: at each
decision point we try ≥2 options on `tune`, pick one, and record all of them. A1's
"chosen on tune, reported on validation" rule carries over. Concretely, each phase below lists
the **options tried** and the artifact that records the comparison.

**2. Engineering metrics are first-class, measured at every stage, not bolted on in Phase 7.**
Index bytes, feature-store bytes, counter-store bytes, model bytes, train wall time, per-request
p50/p99, peak RSS, rows/s at prediction — captured alongside the functional metrics for every
variant, using `timings.measure` and `bench.index_bytes`/`percentiles`.

**3. One trade-off ledger, appended by every phase, is the design note's raw material.**
`artifacts/tradeoffs.jsonl` (+ `tradeoffs.md` regenerated from it), one row per
`(dataset, stage, variant, split)`:

```
functional:  auc, mrr, ndcg@5, ndcg@10, diversity, novelty, coverage  (+ lo/hi CI)
engineering: index_bytes, feature_bytes, model_bytes, train_seconds, peak_rss_mb,
             p50_ms, p99_ms, rows_per_s
delta vs the row it is compared against, paired, with CI
```

`pipeline/ledger.py` owns the row schema and the md renderer; `evaluate`, `ablation`, `nrms`,
`rerank` and `serve` each call `ledger.record(...)`. The note's tables — baseline vs improved,
ablation, serving cost, "what 10× breaks" — are all views over this one file, so writing the
note is selecting rows, not recomputing anything.

## Architecture

New modules, all under `pipeline/`, all reachable through the existing `RETRIEVERS` /
`rank_candidates` / `ranker` interface so `evaluate`, `predict`, bootstrap and slices need no change:

| module | purpose |
|---|---|
| (`sources.py` / `datasets.py`, no new module) | `session_id` in `ColumnMap.behaviors`; EB-NeRD maps raw `session_id`, MIND `None`. |
| `counters.py` | Causal exposure/click counters per article: fitted over all impressions, **read strictly before `t`** (exact match excluded). One `CausalCounter` with `at(article_ids, t)`; the leaky arm is the same object queried with `t = +inf`. Also first-seen time per article (MIND freshness). |
| `features.py` | Feature assembly, grouped by availability tier as module structure: `content` (retriever scores, category match), `history` (n_clicks, decayed-profile cosine, last-click cosine, dwell/scroll stats from past clicks), `exposure` (impressions-so-far, freshness, session counts), `clicked` (clicks-so-far, CTR-so-far). `FEATURE_GROUPS` dict is what the ablation iterates. Emits one long frame `(impression_id, article_id, label, *features)`. |
| `nrms.py` | NRMS-DocVec in PyTorch: candidate/user encoders over frozen doc vectors (multi-head self-attention + additive attention over last-K clicks), dot-product score. `train(config)`, `rank_candidates(...)`, `ranker(...)`. Checkpoint under `artifacts/<ds>/nrms/`. |
| `rerank.py` | LightGBM re-ranker: `train(config, groups=ALL, nrms=True, causal=True)`, `rank_candidates(...)`, `ranker(...)`. Registered as `"rerank"` in `RETRIEVERS`; ablation arms are the same module with a different `RerankSpec`. |
| `ablation.py` | Runs the arms on `tune`/`validation`, pairs by impression against the full model via `evaluate.paired`, writes `artifacts/<ds>/ablation-<split>.{jsonl,md}`. Arms: full; −history; −exposure; −clicked; −NRMS; −retriever scores; literal top-K cut; leaky counters (Q9). |
| `serve.py` (extends `bench.py`) | Single-request path: ann top-K → feature lookup → GBDT over K. p50/p99 latency, index + feature-store bytes, cost/1000 queries at p99<100 ms, 10× argument inputs. |
| `ledger.py` | The trade-off ledger: row schema, `record(...)`, `render()` → `artifacts/tradeoffs.md`. |

Registry changes in `datasets.py`: `ColumnMap.behaviors["session_id"]`, a `RerankSpec` (K, decay
half-life, groups, nrms, causal) and an `NrmsSpec` (history length, heads, epochs, lr, negatives)
per dataset — chosen on `tune`, recorded like every other spec.

`stages.py` gains `nrms`, `features`, `rerank` between `ann` and `evaluate` so `python build.py`
remains the one-command reproduce.

## Phases (each demoable on its own; verify before moving on)

Budget: Sept 15 → 20. Phases 1–3 by the 16th, 4–5 by the 17th, 6 by the 18th, 7 on the 19th, 8 on the 20th.

### Phase 0 — Clean the tree + the ledger (½ day)
Commit the pending `README.md` / `ann_index.py` edits and the untracked `bench.py` (`report/` stays
untracked; it's A1's note). Add `lightgbm` to the env. Write `ledger.py` first, and seed it with
**A1's rows** (bm25 / ann / hybrid on validation: functional from
`artifacts/<ds>/evaluate/*.json`, engineering from `bench-*.jsonl` and `build-timings.jsonl`) so
every later row has a stage-one baseline to be a delta against.
→ verify: `git status` clean, `pytest` green, `artifacts/tradeoffs.md` shows the six A1 rows with
index bytes, p99 and AUC side by side.

### Phase 1 — `session_id` + causal counters (½ day)
- `datasets.py`/`sources.py`: `session_id` through `ColumnMap`; re-ingest both stores (`build.py --force ingest`; the split's leakage guard re-runs).
- `counters.py`: `CausalCounter` over `(impression_time, candidate_ids, labels)`; `at(ids, t)` returns exposures/clicks strictly before `t`; `first_seen(ids, t)`.
- **Options tried:** counter windows — cumulative vs sliding {1h, 3h, 24h}; the window is a `RerankSpec` field chosen on `tune` in Phase 4. Engineering: counter-store bytes and lookup µs per window (a sliding window needs timestamps kept; cumulative needs one int per article).
→ verify: tests — an impression never sees its own outcome; two impressions at the same instant do not see each other; whole-log read (`t=+inf`) ≥ causal read everywhere; MIND `session_id` all-null, EB-NeRD non-null. Ledger: counter bytes + build seconds per dataset.

### Phase 2 — `features.py` (1 day)
Assemble the long frame for a split from: `ingest.per_impression` history, `ann_index`/`bm25_index`
scores via `rank_candidates` (already return `(ranked_ids, scores)` **in ranked order** — read labels
back through `ranked_ids`, per `phase_3.md`), `weighting.weights` for the decayed profile, counters,
article categories/`published_time`.
- **Options tried:** decay half-life for the recency-weighted history {A1's decay, 6h, 24h, 72h}; profile pooling {mean, max, last} (all three already exist in `retrieval.POOLINGS`) — each becomes a feature column, so the GBDT's gain per column *is* the comparison, recorded per group in the ledger.
→ verify: **the Q9 leakage test** — `test_features.py` asserts (a) no feature column changes when
future rows are removed from the log (causality), (b) `next_read_time` / `next_scroll_percentage` /
impression-row `read_time` / `scroll_percentage` are named as forbidden and the build fails if any
reaches the frame, (c) every feature group is a key in `FEATURE_GROUPS`. Row count = Σ candidates.
Ledger: feature-frame bytes, rows, build seconds, peak RSS, per dataset and split.

### Phase 3 — `nrms.py` baseline (1 day)
Train on the earlier half of `train` (MIND: e5 768-d; EB-NeRD: word2vec 300-d, both post-correction
via `embed.load`); register in `RETRIEVERS`; `evaluate` on `validation` with the full harness.
- **Options tried (on `tune`):** history length {20, 50, 80}; heads {8, 16}; negatives {4}; epochs by early stopping. Each cell gets a ledger row: AUC + train seconds + model bytes + per-impression scoring ms — the functional/engineering trade-off of a longer history is the first one the note can show.
→ verify: validation AUC in the neighbourhood of the ebnerd-benchmark paper's NRMS-DocVec on
`ebnerd_small` (look up the exact figure; record it in the artifact md); beats `ann` or the gap is
recorded, not hidden; `predict.ranker` path produces a MINDsmall-dev file that scores the same as
the harness (the A1 self-consistency check from `findings.md` A11/B6).

### Phase 4 — `rerank.py` + `ablation.py` (1 day)
LightGBM on the later half of `train` (features from Phase 2 + NRMS score from Phase 3);
register `"rerank"`; run all ablation arms on `tune`, then once on `validation`. Paired bootstrap
per arm against the full model; the headline claim is `rerank − nrms` with its CI.
- **Options tried (on `tune`):** objective {binary, lambdarank}; leaves {31, 127}; rounds by early stopping; counter window from Phase 1; K for the literal-cut row {50, 100, 200}. **Ablation arms** (each a ledger row with its delta + CI and its own p99/model bytes): full; −history; −exposure; −clicked; −session/dwell; −NRMS; −retriever scores; literal top-K cut; leaky counters (Q9). Feature-importance (gain) table saved alongside.
→ verify: every arm's CI table written; the `−clicked` and `leaky` arms differ from full in the
expected direction; the literal top-K row is below the headline (that *is* the finding); with/without
NRMS feature row exists. Ledger: one row per arm, functional + engineering.

### Phase 5 — Extended evaluation + recall@K (½ day)
`evaluate` already does Q5. Add stage-one recall@K for `ann` at DEPTHS (exists in `retrieval`) into the
ablation md; regenerate `three_way`-style comparison with the four retrievers side by side.
→ verify: `artifacts/<ds>/evaluate/rerank-validation.json` has all metrics × all slices with CIs;
`tradeoffs.md` renders the four-retriever table with AUC, diversity, index bytes and p99 in one row each.

### Phase 6 — Submissions on Ada (1 day wall, mostly waiting)
Extend `predict` only through `rerank.ranker` (counters fed by the train-period log + test users'
histories for clicks, test impressions strictly before `t` for exposures; NRMS scored per chunk).
Jobs: `ada/predict-*.sbatch` (32 CPU / 100 GB; GPU for NRMS scoring if free). `rsync --no-compress
--whole-file` back; **check CRC/md5 both ends** before upload. Upload GBDT files to both boards,
screenshot. NRMS files only if both are up by the 18th.
→ verify: chunk self-consistency test (`test_predict.py` pattern) on MINDsmall-dev scores ≈ harness;
leaderboard numbers recorded in `checkpoint/`. Ledger: full-run wall time, rows/s, peak RSS per
dataset (the measured large-scale cost the 10× argument extrapolates from).

### Phase 7 — Serving & scale (½ day)
`serve.py`: p50/p99 of ann top-K + features + GBDT for one user, N=2000 requests, **broken down per
stage** (retrieve / feature lookup / NRMS / GBDT) so the note can say where the budget goes; bytes of
FAISS index + feature store + counters + models; cost per 1000 queries at p99<100 ms from measured
QPS/core and a stated $/core-hour; 10× argument from `phase_8.md` (the schema and counters are what
grow — say what breaks first, with numbers).
- **Options tried:** K ∈ {50, 100, 200} for the serving path — AUC of the literal-cut arm vs p99 is the cleanest functional-vs-engineering curve in the project, and it costs nothing extra because both axes are already ledger rows. Also flat vs IVF index (`ann_index.py`'s docstring already discusses it; `bench-scale` has the numbers).
→ verify: `artifacts/bench-serve-<ds>.md` regenerated with the two-stage path; `tradeoffs.md` has the K-curve.

### Phase 8 — `test` once, design note, README (1 day)
Score every arm on `test` in one batch (`evaluate --split test`, `ablation --split test`). Write the
6-page note (`report/a2-design-note.tex`, `pdflatex` twice) **from `tradeoffs.md`**: what we built,
the options tried at each step and why the chosen one won, baseline vs improved with ablation + CI,
serving findings with the per-stage latency breakdown and byte counts, the K-curve, where 10×
breaks, the Q9 pair table, the stated unavailability of position bias and MIND publish time.
README: one-command reproduce, AI-usage log pointers, `.gitignore` covers
`*.zip *.pt *.ckpt __pycache__/ data/`.

## Files touched

New: `pipeline/ledger.py`, `pipeline/counters.py`, `pipeline/features.py`, `pipeline/nrms.py`,
`pipeline/rerank.py`, `pipeline/ablation.py`, `pipeline/serve.py`,
`tests/test_{ledger,counters,features,nrms,rerank,ablation}.py`,
`ada/predict-rerank.sbatch`, `report/a2-design-note.tex`.
Modified: `pipeline/datasets.py` (ColumnMap, RerankSpec, NrmsSpec), `pipeline/sources.py`
(session_id), `pipeline/evaluate.py` (two RETRIEVERS entries), `pipeline/stages.py`,
`pipeline/predict.py` (only if the ranker interface needs the counter store handed in),
`requirements.txt`, `environment.yml`, `README.md`, `.gitignore`.

Reused as-is: `evaluate.paired/interval/measure`, `retrieval.recall@K + DEPTHS`, `weighting.weights`,
`ann_index.build_clicks/click_weights`, `embed.load/for_corpus`, `ingest.per_impression`,
`predict.write/impressions/ranks`, `timings.measure`, `bench.percentiles/index_bytes`, `ada/sync.sh`.

## Verification (end to end)

1. `pytest` — all green, including the three new leakage assertions (causal counters, forbidden
   outcome columns, features invariant to future-row removal). Q9's "include a test asserting this".
2. `python build.py --dataset mind` from a clean checkpoint runs acquire → … → rerank → evaluate
   without manual steps (one-command reproduce).
3. `artifacts/<ds>/ablation-validation.md`: full vs nrms CI excludes zero on both datasets, or the
   note says it doesn't and why.
4. Codabench: both GBDT files scored; screenshots in `screenshots/`.
5. `test` scored exactly once; `checkpoint/state.md` records the date.
6. `artifacts/tradeoffs.md` has, for every variant tried in every phase, both a functional and an
   engineering column filled — no row with one side blank. The design note cites only this file.

## Stated limitations (go in the note, not silently)

- Position bias: no positions in either dataset. Dwell/scroll: EB-NeRD past clicks only.
- MIND has no publish time; freshness is first-seen-in-log.
- The literal top-K row measures stage-one recall, which is why it is an ablation row and not the
  headline.
- Session features exist only where the dataset ships sessions.
