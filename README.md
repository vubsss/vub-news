# vub-news — lexical & semantic retrieval on MIND and EB-NeRD

CS4.406 Information Retrieval & Extraction, Assignment 1. Ranks the candidate articles in an
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

MIND is downloaded from the `yjw1029/MIND` HuggingFace mirror, which needs a token
(the official Microsoft endpoint returns HTTP 409):

```bash
export HF_TOKEN=hf_...
```

EB-NeRD downloads from a public S3 bucket and needs no credential.

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
run resumes where it stopped rather than redoing completed work.

```bash
pytest
```

## Layout

```
build.py              one-command entry point
pipeline/
  datasets.py         the dataset registry — the one place MIND and EB-NeRD differ
  stages.py           pipeline stages in dependency order, plus checkpointing
  paths.py            filesystem layout
tests/
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

The scaffold, the registry and the checkpointed stage runner exist. The stages themselves are
declared but not yet implemented — `python build.py` reports them as `not built` and skips them.
They land ticket by ticket; see `../tickets/` for the breakdown and the dependency graph.
