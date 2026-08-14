# vub-news

Ranks the candidate articles in an
impression by click likelihood, using the user's click history and article content, on two news
datasets: **MIND-small** (English) and **EB-NeRD small** (Danish).

## Setup

```bash
conda env create -f environment.yml
conda activate vub-news
```

On Colab or Kaggle, where conda is unavailable:

```bash
pip install -r requirements.txt
```

The two files are kept in lockstep — change both together.

### The HuggingFace token

MIND is downloaded from the `yjw1029/MIND` HuggingFace mirror, because the official Microsoft
endpoint returns HTTP 409. That mirror is a **gated repo**, so getting a token is two steps:

1. Open <https://huggingface.co/datasets/yjw1029/MIND> while logged in and accept its terms. A
   valid token belonging to an account that has not accepted them is still refused.
2. Create a read token at <https://huggingface.co/settings/tokens>.

Then put it in a **`.env` file in the repo root** — create it yourself, it is not in the repo:

```bash
echo 'HF_TOKEN=hf_...' > .env
```

`build.py` reads `.env` on startup, so the token survives across shells and you only do this once.
It is listed in `.gitignore` and must never be committed.

Exporting the variable works too and takes precedence over the file:

```bash
export HF_TOKEN=hf_...
```

EB-NeRD downloads from a public S3 bucket and needs no credential.

### MIND article embeddings

EB-NeRD ships precomputed multilingual BERT vectors, which the pipeline reads directly. MIND ships
none, and this machine has integrated graphics only — so its vectors are generated **once** on a
free-tier hosted GPU and downloaded as an artifact afterwards, rather than recomputed on every
machine.

Run `notebooks/generate_mind_embeddings.ipynb` on Colab with a T4 runtime. It clones this repo,
builds the article corpus with the pipeline's own ingest, encodes it with `embed.encode`, and copies
two files — `embeddings.npy` and `article_id_index.parquet` — into a Drive folder. It installs
nothing: Colab already ships `torch` and `transformers`, and pip re-pinning `numpy` under a live
kernel is what makes hosted notebooks fail. Share that folder as *anyone with the link*, then put
its id in `pipeline/datasets.py`, on MIND's `EmbeddingSpec`:

```python
gdrive_file_id="1AbC...xyz",
```

Until that is set, `python build.py --dataset mind` stops at the embed stage and prints these steps.
Once it is, the artifact downloads automatically the first time and is used as it stands after that,
with no further network call. It is a build output, so it is not committed; the notebook that
produces it is.

## Rebuild

One command rebuilds everything from raw files:

```bash
python build.py
```

| Command | Effect |
|---|---|
| `python build.py` | run every stage that is not already done, for both datasets |
| `python build.py --plan` | print the stage table and exit, changing nothing |
| `python build.py --dataset mind` | restrict to one dataset (repeatable) |
| `python build.py --force bm25 ann` | re-run those stages even if their checkpoint exists |
| `python build.py --force all` | rebuild from scratch |

Each stage writes a checkpoint under `.checkpoints/<dataset>/` when it finishes, so an interrupted
run resumes where it stopped rather than redoing completed work. A forced stage redoes its work even
if its outputs are already on disk.

Downloaded archives are kept under `data/raw/<dataset>/_archives/` (445 MB for EB-NeRD, 100 MB for
MIND), so a re-download is never needed. An archive is opened before it is trusted, so a truncated
one is re-fetched rather than extracted, and an interrupted download leaves only a `.part` file that
nothing will mistake for good data.

```bash
pytest
```

## Evaluation

`python build.py` scores every retriever on the **validation** split as part of stage 8. Any single
combination can also be scored on its own:

| Command | Effect |
|---|---|
| `python -m pipeline.evaluate` | every retriever on every dataset, validation split |
| `python -m pipeline.evaluate --dataset mind --retriever bm25` | one combination |
| `python -m pipeline.evaluate --split test` | the held-back split |
| `python -m pipeline.evaluate --resamples 100` | fewer bootstrap resamples, for a fast run |

Every run prints one row per slice per metric, always the same columns in the same order — dataset,
retriever, split, slice, the slice population, how many impressions the metric was defined on, the
metric, and its value with a bootstrap interval. The same rows are written to
`artifacts/<dataset>/evaluate/<retriever>-<split>.json` alongside the run's counts, so later analysis
aggregates from disk rather than re-ranking. Long rows rather than wide ones: three columns per metric
would not fit, and this way the columns never change and the table sorts and greps.

The harness reaches a retriever only through `rank_candidates`, so adding a third one means adding an
entry to `RETRIEVERS` and nothing else.

Two things it will not do. **It refuses the train split** — metrics on the data a retriever was tuned
against measure memorisation, and `--split train` fails with that explanation rather than returning
numbers. And **the build stage never scores test**: a figure regenerated on every rebuild is one that
gets tuned against, so the held-back split takes a deliberate command.

### What is measured

**Ranking:** AUC, MRR, nDCG@5, nDCG@10 — did the retriever order this impression's candidates well.

**Beyond accuracy**, over the top 10 of each ranked list, because a recommender can order perfectly
while showing everyone the same handful of popular articles:

- **diversity** — the share of pairs in the shown list drawn from different categories
- **novelty** — mean self-information of the shown articles, so obscure ones score above obvious ones
- **coverage** — the fraction of the catalogue the slice ever showed

**Slices:** `overall`, `cold` and `warm` users (fewer / at least 5 clicks in history), and `head` and
`tail` impressions — placed by whether the articles the user clicked are among the most-shown fifth of
the articles the split displays. Impressions whose clicks straddle both are in neither, so head and
tail do not sum to the whole. Every slice is one entry in `SLICES`, a predicate over the
per-impression frame; adding a sixth is that line and nothing else.

**Intervals:** bootstrap 95% percentiles over 1000 resamples of impressions, seeded, so two runs on
the same rankings print the same interval. Impressions are the axis because they are what the sample
is of. Lower `--resamples` while developing, not for a number anyone will quote.

### Reading the output

The counts are not decoration. `population` is how many impressions the slice holds and `n` how many
of them the metric was defined on — a metric over eleven users has to be visibly a metric over eleven
users. AUC is undefined for an impression with no positive candidate and no ranking metric is defined
when every candidate is positive; those are counted in the footer and left out of the mean rather than
averaged in as zeros, which would report the share of degenerate impressions rather than anything
retrieval did. `all_scores_tied` counts impressions the retriever scored flat — their rank metrics
read off the order the candidate file supplied, not off the retriever. Diversity and novelty are still
recorded for all of them: they describe a list that was shown whether or not it was clicked.

**Coverage is a fraction of the whole catalogue**, not of the articles that happened to be offered as
candidates, and the footer prints both. It matters: MIND's validation split shows 6144 of its 65238
articles, so 9.4% is the ceiling any retriever could reach on it. And coverage is the one row whose
interval is not a bracket around its value — it is a union over the slice rather than a mean over it,
and a bootstrap resample repeats about 37% of its impressions, so every resample sees fewer distinct
articles. Read that row's interval as a width and compare coverage between retrievers by value.

Popularity — for novelty and for head/tail — is counted on the split being scored, since these
numbers describe the population the report is about. Nothing here reaches a retriever, so it is
description rather than leakage.

## Lexical against semantic

The comparison the assignment asks for — which retriever wins, on which dataset, on which slice — is
generated from the evaluation results already on disk:

```bash
python -m pipeline.compare
```

| Command | Effect |
|---|---|
| `python -m pipeline.compare` | both datasets, validation split |
| `python -m pipeline.compare --dataset mind` | one dataset; the cross-dataset section is then omitted |
| `python -m pipeline.compare --split test` | compare the held-back split, once it has been scored |

It ranks nothing itself. It reads `artifacts/<dataset>/evaluate/<retriever>-<split>.json`, so it
needs a `python -m pipeline.evaluate` run behind it and costs a second rather than a re-run of both
retrievers. The output is one markdown document — both retrievers x both datasets x every metric x
overall plus all four slices, with intervals, then a reading of it — written to
`artifacts/comparison-<split>.md` and printed. Markdown because it goes into the design note; and
generated rather than written by hand, because a sentence naming a winner has to change when the
numbers do.

**A difference is only a difference when the intervals are disjoint.** Where they overlap the table
says `not established` and the reading says so in words, which is not the same as saying the two are
equal: the test is conservative, and deliberately so. It is conservative in one further way worth
knowing — both retrievers rank the same impressions, so a paired test on the per-impression
differences would separate more than this does. The stored reports carry slice means rather than
per-impression values, so that test is not available without re-scoring.

Coverage is the one metric compared by value rather than by interval, because its interval is a width
and not a bracket around its own value (see above). And only the ranking metrics have a better
direction: for diversity, novelty and coverage the tables say which retriever is *higher* and never
which one won.

Two reports are refused rather than compared when they disagree about what they measured — different
impression counts, catalogue, split or resample count. A bm25 report from before a re-ingest against
a fresh ann one would otherwise produce a difference between two populations wearing the clothes of a
difference between two retrievers.

## The ablation sweep

The experimental grid, and what makes the history-window choice defensible rather than picked:

```bash
python -m pipeline.sweep
```

| Command | Effect |
|---|---|
| `python -m pipeline.sweep` | the full grid on validation |
| `python -m pipeline.sweep --quick` | windows 5 and 20 only, 100 resamples — for development |
| `python -m pipeline.sweep --dataset mind --retriever bm25 --window 5` | restrict any axis (repeatable) |
| `python -m pipeline.sweep --restart` | discard the stored cells and run them all again |
| `python -m pipeline.compare --window 20` | rebuild the comparison from that cell of the grid |

**The grid runs as twelve cells, not thirty-six.** The spec's axes are dataset x retriever x history
window x retrieval depth, but depth and the ranking metrics do not meet: depth belongs to corpus
retrieval, where recall@K asks how far down you had to look, and the three depths are prefixes of one
depth-200 search. The ranking metrics come from re-ranking the candidate list an impression already
carries, which is never truncated, so no depth can move them. Running depth over them would write the
same AUC into the file three times and invite a reader to average it. Depth is recorded next to every
recall number and is absent from the ranking metrics, which is where it actually is.

Each cell is one line of `artifacts/sweep-<split>.jsonl` holding its own configuration, its recall at
each depth, and the evaluation report the harness produced. One line per cell is what makes the sweep
resumable — a line is written whole or not at all, so an interrupted run leaves no half-finished cell
and the next run costs only what is left. Runtime is reported both as total grid time and as the time
this run spent, so a resumed sweep tells the truth about both.

**The window is the axis, so it has to reach both retrievers.** They take it through one parameter of
one call, and every cell checks the window the retriever reports back against the one it was asked
for. A test drives the point: with a history whose last five clicks are about a different article
than its last twenty, a retriever that ignored the window would rank the candidates the other way
round, and both retrievers are asserted to move.

The best window is chosen on AUC and **only where the intervals are disjoint** — a grid this size
always has a highest number, and the question is whether it is a finding or the noise floor. Recall
is reported as a point estimate with no interval, because it comes from the corpus-retrieval path the
harness's bootstrap does not run over; the document says so rather than leaving a reader to notice a
missing column.

`python -m pipeline.compare --window K` reads a cell straight out of the grid, so a swept window
reaches the comparison by being named rather than by anyone copying numbers. It writes
`comparison-<split>-k<K>.md`, a separate file from the default comparison, because two documents
holding different numbers must not share a name. Two retrievers swept at different windows are
refused rather than compared.

The sweep is not a `build.py` stage. It is half an hour of compute whose inputs change only when the
retrievers do, and a rebuild that ran it every time is a rebuild people stop running.

## CodaBench submission

```bash
python -m pipeline.predict --dataset mind
```

Writes `predictions/mind_submission.zip`, ready to upload at
<https://www.codabench.org/competitions/13967/>. `python build.py` runs the same thing as its last
stage.

| Command | Effect |
|---|---|
| `python -m pipeline.predict` | build every competition's submission that the registry describes |
| `python -m pipeline.predict --dataset mind` | restrict to one competition (repeatable) |
| `python -m pipeline.predict --retriever bm25` | rank with the lexical retriever instead of the semantic one |
| `python -m pipeline.predict --history-k 5` | build the query from a different click window |
| `python -m pipeline.predict --force` | rebuild even if the archive is already there |

This is **not** the local evaluation task, and the difference decides most of the design:

- **The competition supplies the candidates.** Every impression arrives with its own list and every
  one of them must come back in an order, so this path calls `rank` and never `retrieve`. No
  retrieval depth appears anywhere in it — a candidate list is scored whole.
- **The competition supplies its own catalogue and its own histories.** MIND's open phase is scored
  against `MINDlarge_test`: 2,370,727 impressions over 120,961 articles from a later week than the
  `MINDsmall` feature store covers. So the retriever is built over the competition's articles, not
  the pipeline's, and a user this project has never seen is not a special case — the history the
  ranking is built from arrives on the impression row.
- **The embedding artifact only half covers it.** 60,609 of those 120,961 articles have a vector in
  the artifact ticket 7 generated; the other 60,352 are encoded locally on CPU the first time this
  runs and cached under `artifacts/mind/predict/`. Left at zero instead, the semantic retriever
  would score half the catalogue 0 and submit the candidate file's own order under its name.

The 1.46 GB impression file is streamed in chunks of 100,000 — one user vector per impression at 384
float32 is 3.6 GB before a single candidate is scored — and written in input order, which the
competition requires.

### What is asserted before the file is written

Every check runs on every impression, not on a sample. The file is written to a `.part` and renamed
only once all of them have passed, so a run that fails halfway leaves nothing that looks finished.

- The ranking of an impression holds each of its candidates exactly once and adds none of its own.
- No two candidates claim the same place, which also rejects an input impression that lists the same
  candidate twice.
- The retriever returned rankings for the chunk it was given, in that order — they are zipped back
  on positionally, so a reordering would produce a well-formed file that scores every impression
  against another one's candidates.
- The number of lines written equals the number of impressions counted from the raw file itself.

### Leaderboard against local

The competition scores the same four metrics the harness does, so the two sit side by side. They are
**not** measured on the same thing and are not expected to agree: the local column is 30,269
`MINDsmall` validation impressions from 50,000 sampled users, the leaderboard column is 2,370,727
`MINDlarge_test` impressions from a later week over the whole user base.

| Metric | Local — MIND validation, `ann` | CodaBench — Official Test |
|---|---|---|
| AUC | 0.6252 [0.6219, 0.6286] | *fill in from the leaderboard* |
| MRR | 0.3297 [0.3260, 0.3333] | *fill in from the leaderboard* |
| nDCG@5 | 0.3058 [0.3019, 0.3098] | *fill in from the leaderboard* |
| nDCG@10 | 0.3642 [0.3606, 0.3680] | *fill in from the leaderboard* |

The last run's own account of what it ranked, which is where a gap would be explained from:

- 2,370,727 impressions, 93,115,001 candidates, ranked in 210 s after a 24-minute one-off encode of
  the half of the catalogue the artifact did not cover.
- 29,108 impressions (1.23%) carry no click history and 29,109 (1.23%) scored flat — so all but one
  flat ranking is a genuinely cold user rather than a catalogue miss, and 98.77% of the leaderboard
  score is the retriever's own work.
- 0 articles left without a vector.

Submitting: upload `predictions/mind_submission.zip` under **Participate → Submit**, and save the
resulting leaderboard entry to `screenshots/mind-leaderboard.png` for the design note.

## Layout

```
build.py              one-command entry point
pipeline/
  datasets.py         the dataset registry — the one place MIND and EB-NeRD differ
  stages.py           pipeline stages in dependency order, plus checkpointing
  acquire.py          download and extract raw archives
  ingest.py           raw files -> the unified schema, dataset-agnostic
  sources.py          per-dataset adapters: the only module that knows either shape
  split.py            temporal train/validation/test split and its leakage guards
  preprocess.py       language-parameterised cleaning, for documents and queries alike
  bm25_index.py       BM25 index, click-history queries, recall@K
  embed.py            article vectors, aligned to the catalogue and unit length
  ann_index.py        exact inner-product index, user vectors, recall@K
  retrieval.py        the ranked shape both retrievers emit, and how it is scored
  evaluate.py         ranking and beyond-accuracy metrics, sliced, with bootstrap intervals
  compare.py          lexical against semantic, both datasets, from the stored results
  sweep.py            the ablation grid over history windows, resumable, one file out
  predict.py          the competition's own impressions, ranked and packaged
  submissions.py      per-competition line formats for the prediction file
  paths.py            filesystem layout
tests/
notebooks/            the MIND embedding generation run, for a hosted GPU
```

Generated directories, none of them committed: `data/raw/` (downloads), `feature_store/` (unified
schema tables), `artifacts/` (embeddings and id indices), `predictions/` (submission files),
`.checkpoints/`.

## The dataset registry

The pipeline is **dataset-agnostic**. Stages take a `DatasetConfig` and work for any entry in
`DATASETS`; everything that differs between the two datasets — archive URLs, source column names,
language, embedding source, submission format — lives in `pipeline/datasets.py`.

**If a stage tempts you to write `if dataset == "mind"`, the fix belongs in the registry.**

The unified schema all stages read is declared there too, as `ARTICLE_COLUMNS`, `BEHAVIOR_COLUMNS`
and `HISTORY_COLUMNS`. Both datasets map onto exactly those columns, and a test enforces it.

## Status

The scaffold, the registry, the checkpointed stage runner, raw data acquisition, ingest into the
unified schema, the temporal split, text preprocessing, BM25 lexical retrieval, article embeddings,
semantic retrieval, the full evaluation harness — ranking and beyond-accuracy metrics, population
slices and bootstrap intervals — the lexical-against-semantic comparison, the ablation sweep and the
MIND CodaBench submission exist. EB-NeRD's submission is the one thing the registry still describes
as a competition url and nothing else; `python -m pipeline.predict` says so and skips it. It lands
with ticket 14; see `../tickets/` for the breakdown and the dependency graph.
