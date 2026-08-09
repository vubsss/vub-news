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
builds the article corpus with the pipeline's own ingest, encodes it, and copies two files —
`embeddings.npy` and `article_id_index.parquet` — into a Drive folder. Share that folder as
*anyone with the link*, then put its id in `pipeline/datasets.py`, on MIND's `EmbeddingSpec`:

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
unified schema, the temporal split, text preprocessing, BM25 lexical retrieval and article
embeddings exist. The remaining stages are declared but not yet implemented — `python build.py`
reports them as `not built` and skips them. They land ticket by ticket; see `../tickets/` for the
breakdown and the dependency graph.
