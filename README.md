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
MIND, plus 1.63 GB and 605 MB of competition test set that only the submission stage fetches), so a
re-download is never needed. An archive is opened before it is trusted, so a truncated
one is re-fetched rather than extracted, and an interrupted download leaves only a `.part` file that
nothing will mistake for good data.

```bash
pytest
```

### Assignment 2: the two-stage system

`python build.py` builds everything through the re-ranker and evaluates it on `validation` —
`features`, `nrms` and `rerank` are stages like any other. Four things sit outside the rebuild
because each is run deliberately rather than on every build:

| Command | Effect |
|---|---|
| `python -m pipeline.ablation --split validation` | the nineteen arms, each a paired difference from the full model, with the Q9 pair and the recall@K bridge |
| `python -m pipeline.serve` | one request at a time through the whole two-stage path: per-stage p50/p99, the bytes a request touches, the cost per thousand queries, the K-curve and the three 10× rows |
| `python -m pipeline.predict --retriever rerank` | rank a competition's test set with the re-ranker and write the zip (`ada/predict-rerank.sbatch` on the cluster; `--bench-chunks` first, to choose `--chunk`) |
| `python -m pipeline.final` | score `test` — **once**, with the date and commit recorded, and refused thereafter |
| `python -m pipeline.report` | regenerate every table in the design note from the ledger |

The ablation reads six materialised frames — `train`, `tune` and the scored split, each in a causal
and a leaky copy — and `python build.py` builds only the three causal ones. Build the leaky copies
before running it:

```bash
python -m pipeline.features --leaky --split train --split tune --split validation
```

`pipeline.final` does this for itself, because a run that spends the held-out split and *then*
fails on a missing frame is blocked from retrying by its own refusal.

### The trade-off ledger

Every variant any of those commands tries records one row in `artifacts/tradeoffs.jsonl`, carrying
both what it scores (AUC, MRR, nDCG, diversity, novelty, coverage, each with a bootstrap interval)
and what it costs (bytes, seconds, peak RSS, p50/p99, rows/s). `python -m pipeline.ledger` renders
it to `artifacts/tradeoffs.md`.

A `—` in that table is a measurement that has not been taken, not a zero. The design note's tables
are generated from the same file, so the note and the repository cannot disagree about a number.

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

### Four partitions, not three

The split is `train / tune / validation / test`, strictly in time, with the three held-out windows
measured back from the end of the log.

| | train | tune | validation | test |
|---|---:|---:|---:|---:|
| MIND | 95,071 | 31,625 | 30,269 | 73,152 |
| EB-NeRD | 225,110 | 66,908 | 100,981 | 84,535 |

`tune` exists because selecting a parameter and reporting a result are different jobs and one
partition cannot do both. Every choice this project makes — the history window, the embedding
correction, the BM25 constants — is made on **tune**; every number it reports comes from
**validation**; **test** is scored once, at the end. Reverting a tune-split choice because a
validation number disagreed would be selecting on validation, which is the whole thing this
prevents.

The tune window is carved off the *end* of train, so it is adjacent in time to the population it
stands in for. Because the held-out windows are measured from the end of the log, adding it moved
only the train/tune boundary: validation and test cover exactly the days they always did, and a
re-scored report reproduces its predecessor byte for byte.

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

## Embedding geometry

The single largest improvement in this project came from three lines of numpy, and finding it needed
a statistic the pipeline was not computing.

Raw transformer output does not fill the space it lives in — it occupies a narrow cone, so any two
vectors have a high cosine whatever the two articles say, and the signal a retriever needs rides as a
small residual on a large shared offset. The embed stage now prints the symptom for every dataset:

```
anisotropy 0.9491 -> 0.0016 under abtt:3
```

That is the mean cosine between two different articles. EB-NeRD's shipped multilingual BERT vectors
sat at **0.9503** — every article 95% similar to every other — and its semantic retriever scored AUC
0.4984, a coin flip, for that reason and no other. MIND's sentence-trained MiniLM sits at **0.0630**
and needs almost nothing.

Four corrections, chosen per dataset in the registry the way `normalise` already is:

| | what it removes |
|---|---|
| `none` | nothing |
| `centre` | the corpus mean, which is the shared offset |
| `abtt:n` | the mean, then the *n* leading principal directions |
| `whiten` | the whole covariance, decorrelated |

Which one is a measured choice, made on the tune split:

| correction | EB-NeRD auc | anisotropy | | MIND auc | anisotropy |
|---|---:|---:|---|---:|---:|
| none | 0.4877 | +0.9491 | | 0.6250 | +0.0630 |
| centre | 0.5199 | +0.0317 | | **0.6320** | −0.0000 |
| abtt:1 | 0.5409 | +0.0063 | | 0.6213 | −0.0001 |
| **abtt:3** | **0.5477** | +0.0016 | | 0.6061 | −0.0000 |
| abtt:5 | 0.5441 | +0.0008 | | 0.5940 | −0.0001 |
| whiten | 0.5216 | **+0.0001** | | 0.5793 | −0.0000 |

On validation that takes EB-NeRD's semantic retriever from 0.4984 [0.4963, 0.5004] to
**0.5500 [0.5479, 0.5519]**, and recall@200 from 0.0170 to 0.0266.

**Three things in that table are worth more than the headline number.**

*The statistic diagnoses the problem and does not choose the fix.* Anisotropy falls monotonically
down both columns, and `whiten` reaches the best geometry of the six on both datasets while ranking
among the worst. AUC turns over at three components on EB-NeRD and at zero on MIND. Past the turn the
correction is removing signal along with the offset, so the correction has to be swept and cannot be
read off the geometry.

*The same transform helps in proportion to how broken the vectors were.* EB-NeRD gains 0.060 and MIND
0.007, which is the control that makes the EB-NeRD result credible rather than an artefact of the
method — and over-correction costs MIND up to 0.046, because on vectors that were never broken the
leading directions carry signal rather than offset.

*A marginal tune-split effect did not replicate.* MIND's `centre` was chosen on tune by +0.007 and
validation put it at −0.0025 with overlapping intervals, while its other three ranking metrics and
its recall rose. It is kept regardless: reverting on the strength of a validation number is selecting
on validation. Nothing is established either way on MIND, and that is the honest reading.

### Which vectors, not just which correction

EB-NeRD ships **four** sets of article vectors and the pipeline can index one. Which one was never
measured — the multilingual BERT file was taken because it is the one the assignment's download
snippet names. Scoring all four under all six corrections on the tune split says it was the worst of
them, at more than twice the dimensionality of the best:

| source | dim | best correction | auc | anisotropy as shipped |
|---|---:|---|---:|---:|
| **document_vector** (word2vec) | **300** | abtt:1 | **0.5665** | +0.7907 |
| xlm_roberta_base | 768 | abtt:1 | 0.5646 | +0.9990 |
| contrastive_vector | 768 | abtt:1 | 0.5602 | +0.1850 |
| bert_base_multilingual_cased | 768 | abtt:3 | 0.5477 | +0.9491 |

**word2vec, at 300 dimensions, beats three transformer encoders at 768.** On headlines and subtitles
a few dozen words long a bag of trained word vectors is not obviously the weaker representation, and
nothing in this pipeline had ever measured the assumption that it was. It is also three times cheaper
to query — 0.12 ms against 0.35 ms — and half the index.

Two more things that only four sources could show. **Contrastive training really does fix the
geometry**: those vectors ship at 0.1850 while every other source is between 0.79 and 0.999, which is
what contrastive training is for. They still gain from correction (0.5332 to 0.5602) and they still
do not win. And **`whiten` is the worst correction for all four sources** while producing the best
geometry for all four — the clearest possible statement that the statistic is a diagnosis and not a
prescription.

On validation the promoted choice gives auc **0.5672 [0.5652, 0.5692]**, against 0.5500 for corrected
mBERT and 0.4984 for the vectors the pipeline started with. Tune predicted 0.5665 and validation
returned 0.5672, which is the replication MIND's marginal effect did not manage.

Every source is scored under every correction by:

```bash
python -m pipeline.embed_compare --dataset ebnerd
```

Writes `artifacts/embeddings-ebnerd-tune.jsonl` and a markdown table beside it. It chooses nothing: promoting
a winner into the registry stays a hand edit with a commit message, because it is a decision rather
than a computation.

## Lexical tuning

Two things on the lexical side had never been measured: `k1` and `b` were module constants copied
from `SPEC.md`, and the query was built from clicked **titles** on the strength of a comment
reasoning that abstracts would drown the identifying terms. Both are registry fields now, and both
were chosen on **tune**.

```bash
python -m pipeline.lexical_sweep --dataset mind        # 75 cells: k1 x b x title weight
python -m pipeline.lexical_ablation --dataset mind     # the named settings, on both paths
```

The sweep ranks; it does not establish. **Every cell scores the same impressions**, and a per-cell
bootstrap interval is about ±0.003 wide because impressions differ from each other far more than
settings do — while the whole surface spans 0.0034. No two cells could ever be disjoint, so asking
whether their intervals overlap answers the question before it is run. `lexical_ablation` bootstraps
the **paired** per-impression difference instead, which cancels exactly that shared variance, and it
reports both the corpus-retrieval path (recall@K) and the re-ranking path (AUC, MRR, nDCG) because
field weighting can move them in different directions.

What that found, on tune, paired, both datasets independently:

| | tuned `k1`, `b`, title weight | query built from titles **and abstracts** |
|---|---|---|
| MIND | +0.0010 auc [+0.0003, +0.0016] | **+0.0072** auc [+0.0049, +0.0093] |
| EB-NeRD | +0.0009 auc [+0.0005, +0.0013] | **+0.0073** auc [+0.0054, +0.0091] |

The 75-cell grid over the parameters IR practice says to tune is worth about +0.001. The one-line
assumption nobody had tested is worth seven times that, on both languages, replicating to the fourth
decimal. Both grids are flat and rise monotonically to their own boundary on every axis; they were
not extended, because a surface whose entire span is 0.0034 with all 74 other cells overlapping the
best is flat by the ticket's own rule.

Title weighting is an **approximation** of BM25F rather than BM25F: repeating the title raises its
terms' frequency before saturation, which is the mechanism, but it also lengthens the document, which
real per-field length normalisation would not. `bm25s` indexes one field and cannot express the exact
form, and `LexicalSpec` says so rather than glossing it.

## What the choices cost

Every other comparison here chooses a *setting* by AUC. `pipeline.bench` measures what AUC cannot
see, because two tool choices had been made by argument rather than by measurement — the exact FAISS
index ("approximation buys nothing at this size") and `bm25s` over `rank_bm25` ("faster"). Neither
had ever been timed.

```bash
python -m pipeline.bench --dataset mind --bench all    # ann, scale, lexical, serve, precision
```

Writes `artifacts/bench-<kind>-<dataset>.jsonl` and a markdown table beside it, like every other
comparison. The `lexical` bench needs `pip install rank_bm25`; it is deliberately **not** in
`environment.yml`, because it is here to be measured against rather than depended on, and the bench
reports the backend it has if it is absent.

What it found:

- **Approximation was not worthless, the argument was.** IVF is 7.7x (MIND) and 14.8x (EB-NeRD)
  faster at the median for 4–8% of the exact top-10. What actually justifies exact search is that
  **the index is off the ranking path**: `rank_candidates` scores an impression's own candidates by
  direct product against rows looked up by id, and never queries the index — so approximation can
  cost recall@K and cannot cost AUC.
- **FAISS' flat index earns nothing over the matrix it wraps** at this scale: 1.9–3.7x slower than a
  bare `numpy` matmul, which it also duplicates in memory.
- **`bm25s` is 3,385x faster per query** than `rank_bm25` on MIND at 0.992 top-10 agreement, and
  pays for it with a 4x slower index build. The SPEC claim was right; the trade was never stated.
- **`ann` is the cheaper retriever as well as the better one** — 0.108 ms against 2.43 ms per
  impression on MIND, 0.078 against 8.03 on EB-NeRD. EB-NeRD's 124.5 impressions/s reproduces the
  125 quoted in the design note from an unrelated measurement.
- **`HISTORY_K` 10 -> 80 costs 5.8x on the lexical path and 1.6x on the semantic one.** A lexical
  query is text rebuilt from K titles per impression; a semantic one is the mean of K rows.
- **Precision buys memory, not speed.** int8 is 4x smaller at +0.0001 AUC and 1.9x *slower*, because
  the quantiser decompresses per query. Width bought neither.
- **Exact search holds a 10 ms median to about 85M `n x d` elements** on both datasets independently.
  Our offline catalogue is 50M; `MINDlarge_test` at 120,961 x 768 is 92.9M, so the *submission*
  already runs past the point where the shipped choice stops being free.

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
per-impression values, so that test is not available without re-scoring. `lexical_ablation` does run
that paired test, on the settings it compares — see *Lexical tuning* above for what it separates that
overlapping intervals do not.

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
python -m pipeline.predict --dataset ebnerd
```

Writes `predictions/mind_submission.zip` and `predictions/ebnerd_submission.zip`, ready to upload at
<https://www.codabench.org/competitions/13967/> and
<https://www.codabench.org/competitions/2469/>. `python build.py` runs the same thing as its last
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
  `MINDsmall` feature store covers. EB-NeRD's is scored against `ebnerd_testset`: 13,536,710
  impressions over 125,541 articles, with 807,677 users' reading histories in a table of their own.
  So the retriever is built over the competition's articles, not the pipeline's, and a user this
  project has never seen is not a special case — the history the ranking is built from arrives with
  the competition's files, on the impression row for MIND and in that table for EB-NeRD.
- **A vector is reused on its text, not on its article id.** 60,609 of MIND's 120,961 competition
  articles carry an id the artifact also holds — and **60,608 of them are a different article**.
  MIND's ids are per-release local names, so aligning by id alone gave the submission a catalogue
  half of which held some unrelated article's vector: well-formed, unit length, and wrong, which
  cost two void leaderboard uploads before it was found. `embed.stale_rows` now compares the
  corpus's `document_text` against the feature-store catalogue the artifact was encoded from, and
  `for_corpus` re-encodes whatever disagrees — **120,960 of 120,961 for MIND**, on a GPU where one
  is available. EB-NeRD's ids *are* stable: all 20,738 shared ids agree, the shipped vectors cover
  all 125,541 of its competition articles, and nothing is encoded or left at zero. Nothing announces
  which kind of id a dataset has, so the text is checked rather than assumed.

Both impression files are streamed in chunks of 100,000 — one user vector per impression at 384
float32 is 3.6 GB for MIND before a single candidate is scored — and written in input order, which
the competitions require. EB-NeRD's history table is read in chunks too, and only the last
`--history-k` clicks of each user are kept: it holds 116M clicks, of which the retrievers read 8M.

### What each competition asks for

Recorded here so a regeneration does not have to rediscover it. Both CodaBench pages render their
terms client-side, so neither can be quoted from a fetch; EB-NeRD's format below is read off the
challenge's own `ebrec.utils._python.write_submission_file` and `rank_predictions_by_score`, and its
test-set layout off `examples/quick_start/nrms_ebnerd.py`, in `ebanalyse/ebnerd-benchmark`.

| | MIND | EB-NeRD |
|---|---|---|
| Competition | [13967](https://www.codabench.org/competitions/13967/) | [2469](https://www.codabench.org/competitions/2469/) |
| Test archive | `MINDlarge_test.zip`, gated HF mirror | `ebnerd_testset.zip`, 1.63 GB, public S3 |
| Extracted to | `data/raw/mind/test/` | `data/raw/ebnerd/testset/` — its own `articles.parquet` must not land on the feature store's |
| Impressions | 2,370,727 | 13,536,710 |
| Candidates | 93,115,001 | 205,925,868 |
| Click history | on the impression row | `test/history.parquet`, one row per user, joined on `user_id` |
| File in the zip | `prediction.txt` | `predictions.txt` — plural |
| A line | `24481 [4,1,3,2]` | `237 [4,1,3,2]` — the same shape, arrived at independently |
| Impression id | one row each | one row each for 13,336,710 of them; the 200,000 beyond-accuracy impressions are all stamped `0` |

The last row is the only place the two competitions disagree about what a submission *is*. EB-NeRD's
beyond-accuracy impressions — 250-candidate lists it scores for diversity rather than for clicks —
share a single id, so an id is not a row key there and the file is matched by row. The registry says
so in one field, `repeated_impression_id`, and every other id is still held to appearing once.

### What is asserted before the file is written

Every check runs on every impression, not on a sample. The file is written to a `.part` and renamed
only once all of them have passed, so a run that fails halfway leaves nothing that looks finished.

- The ranking of an impression holds each of its candidates exactly once and adds none of its own.
- No two candidates claim the same place, which also rejects an input impression that lists the same
  candidate twice.
- The retriever returned rankings for the chunk it was given, in that order — they are zipped back
  on positionally, so a reordering would produce a well-formed file that scores every impression
  against another one's candidates.
- No impression id appears twice, except the one the registry records the competition as repeating
  on purpose — which narrows the check by a single value rather than turning it off.
- The number of lines written equals the number of impressions counted from the raw file itself.

### Leaderboard against local — MIND

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

- 2,370,727 impressions, 93,115,001 candidates, ranked in 423 s after encoding 120,960 of the
  120,961 articles — everything the text check refused to take from the artifact — in ~9 minutes on
  a GPU. 17m48s end to end.
- 60,608 rows matched an artifact row by id and were rejected on the text.
- 29,108 impressions (1.23%) carry no click history and 29,109 (1.23%) scored flat — so all but one
  flat ranking is a genuinely cold user rather than a catalogue miss, and 98.77% of the leaderboard
  score is the retriever's own work.
- 0 articles left without a vector.
- anisotropy 0.7262 -> -0.0000 under `centre`, applied on the way into the index. The submission
  used to skip the correction the harness applies; it does not now.

Submitting: upload `predictions/mind_submission.zip` under **Participate → Submit**, and save the
resulting leaderboard entry to `screenshots/mind-leaderboard.png` for the design note.

### Leaderboard against local — EB-NeRD

Here the two retrievers do not separate the way they did on MIND, and the choice of which to submit
is not settled by the local numbers alone. On 100,981 `ebnerd_small` validation impressions, `bm25`
leads on AUC by a margin whose bootstrap intervals are disjoint, and the two are indistinguishable
on every other ranking metric — while `ann`'s own AUC interval contains 0.5. AUC is what the
challenge ranks on, so both files are generated and the leaderboard decides:

| Metric | Local — `bm25` | Local — `ann` | CodaBench — `bm25` | CodaBench — `ann` |
|---|---|---|---|---|
| AUC | 0.5051 [0.5034, 0.5067] | 0.4984 [0.4963, 0.5004] | *fill in* | *fill in* |
| MRR | 0.3171 [0.3153, 0.3188] | 0.3200 [0.3182, 0.3217] | *fill in* | *fill in* |
| nDCG@5 | 0.3472 [0.3451, 0.3492] | 0.3474 [0.3453, 0.3494] | *fill in* | *fill in* |
| nDCG@10 | 0.4316 [0.4299, 0.4333] | 0.4326 [0.4309, 0.4342] | *fill in* | *fill in* |

Choosing between two files by leaderboard score is fitting to the test set, and the design note
should say so rather than present the winner as the retriever the offline evaluation picked.

The `ann` run's own account of what it ranked:

- 13,536,710 impressions, 205,925,868 candidates, ranked in 708 s — 13 m 34 s wall including the
  catalogue, the history table and the zip, at a peak of 8.5 GB resident.
- 0 impressions (0.00%) carry no click history and 0 scored flat: every user in the test file has a
  history row, and every ranking is the retriever's own work rather than the candidate file's order.
- 0 articles left without a vector.
- 703 MB of prediction, 227 MB zipped.

Submitting: `predictions/ebnerd_submission-ann.zip` and `predictions/ebnerd_submission-bm25.zip`
under **Participate → Submit**, and save the better entry to `screenshots/ebnerd-leaderboard.png`.
`python -m pipeline.predict --dataset ebnerd [--retriever bm25]` writes whichever of the two is
asked for, always as `predictions/ebnerd_submission.zip`; the suffixed names are the two runs kept
side by side.

## Layout

```
build.py              one-command entry point
pipeline/
  datasets.py         the dataset registry — the one place MIND and EB-NeRD differ
  stages.py           pipeline stages in dependency order, plus checkpointing
  acquire.py          download and extract raw archives
  ingest.py           raw files -> the unified schema, dataset-agnostic
  sources.py          per-dataset adapters: the only module that knows either shape
  split.py            temporal train/tune/validation/test split and its leakage guards
  counters.py         per-article exposure/click counters, fitted once, read strictly before t
  preprocess.py       language-parameterised cleaning, for documents and queries alike
  bm25_index.py       BM25 index, click-history queries, recall@K
  embed.py            article vectors, aligned, unit length, and geometrically corrected
  embed_compare.py    every vector source under every correction, scored on tune
  ann_index.py        exact inner-product index, user vectors, recall@K
  retrieval.py        the ranked shape every retriever emits, and how it is scored
  evaluate.py         ranking and beyond-accuracy metrics, sliced, with bootstrap intervals
  compare.py          lexical against semantic, both datasets, from the stored results
  sweep.py            the ablation grid over history windows, resumable, one file out
  lexical_sweep.py    BM25's k1, b and title weight, swept jointly on tune
  lexical_ablation.py the same retriever on both paths, and the query-side ablation
  predict.py          the competition's own impressions, ranked and packaged
  submissions.py      per-competition line formats for the prediction file
  paths.py            filesystem layout

  ledger.py           the trade-off ledger: one row per variant, both metric families
  features.py         one row per (impression, candidate), in four availability tiers
  nrms.py             NRMS-DocVec, reproduced as a fourth retriever
  rerank.py           the LightGBM re-ranker, a fifth, and the headline delta
  ablation.py         nineteen arms, each a paired difference from the full model
  serve.py            one request at a time, timed stage by stage, and where 10x breaks
  final.py            `test`, scored once, with a record that says it was once
  report.py           the design note's tables, emitted from the ledger
tests/
report/               the design note; its tables are generated, never typed
ada/                  cluster placement — the same commands, with the filesystem moved
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

**Assignment 1** is complete: the scaffold, the registry, the checkpointed stage runner,
acquisition, ingest, the temporal split, preprocessing, BM25, article embeddings, semantic
retrieval, the evaluation harness with slices and bootstrap intervals, the lexical-against-semantic
comparison, the ablation sweep, and both CodaBench submissions.

**Assignment 2** is built as code with tests on it and is **not yet measured**. The feature frame,
NRMS-DocVec, the LightGBM re-ranker, the nineteen ablation arms, the submission path over the
competitions' own periods, the serving benchmark, the scored-once `test` batch and the design
note's table generator all exist; `pytest` is green. What does not exist in this checkout is any
*number*: there is no MIND or EB-NeRD data here, so every ledger cell is blank and every ticket
file in `tickets/a2/` states which command fills its own.

See `HANDOFF.md` for what has to happen next, in order, on a machine that has the data.
