# SPEC.md

---

## Initial Details

- **Environment pinning:** `environment.yml` (conda env). Also include `requirements.txt` for Colab/Kaggle compatibility.
- **One-command rebuild:** `python build.py` rebuilds the full pipeline from raw files. Embedding artifact for MIND is auto-downloaded via `gdown` if not present.
- **Code structure:** modular `.py` files under `pipeline/`. No large files committed to Git (`.gitignore` covers `data/`, `*.zip`, `*.npy`, `*.parquet` under raw paths).
- **Data already downloaded:**
  - `MINDsmall` (train + dev zips)
  - `EB-NeRD_small`
  - EB-NeRD precomputed embeddings: `google_bert_base_multilingual_cased` (S3 artifact)

---

## Repository Structure

```
├── build.py                        # one-command entry point
├── environment.yml                 # conda env pin
├── requirements.txt                # pip fallback for Colab/Kaggle
├── README.md
│
├── pipeline/
│   ├── ingest.py                   # raw → unified schema
│   ├── split.py                    # temporal train/val/test split
│   ├── preprocess.py               # text cleaning, Danish stemming
│   ├── feature_store.py            # build and save feature store
│   ├── bm25_index.py               # BM25 inverted index builder + retriever
│   ├── ann_index.py                # FAISS index builder + retriever
│   ├── evaluate.py                 # metrics: AUC, MRR, nDCG, beyond-accuracy
│   └── predict.py                  # generate CodaBench prediction files
│
├── notebooks/
│   └── generate_mind_embeddings.ipynb   # run on Colab, output uploaded to GDrive
│
├── feature_store/
│   ├── mind/
│   │   ├── articles.parquet
│   │   ├── behaviors.parquet
│   │   └── history.parquet
│   └── ebnerd/
│       ├── articles.parquet
│       ├── behaviors.parquet
│       └── history.parquet
│
├── artifacts/
│   ├── mind/
│   │   ├── embeddings.npy           # auto-downloaded from GDrive by build.py
│   │   └── article_id_index.parquet
│   └── ebnerd/
│       ├── embeddings.parquet       # from google_bert_base_multilingual_cased
│       └── article_id_index.parquet
│
└── predictions/
    ├── mind_submission.txt
    └── ebnerd_submission.txt
```

---

## Unified Schema

All downstream components operate on three canonical tables per dataset.
Dataset-specific logic is fully contained in `pipeline/ingest.py`.

### `articles.parquet`
| Column           | MIND source        | EB-NeRD source       | Notes                          |
|------------------|--------------------|----------------------|--------------------------------|
| `article_id`     | `news_id`          | `article_id`         | cast to `str`                  |
| `title`          | `title`            | `title`              |                                |
| `abstract`       | `abstract`         | `subtitle`           | subtitle plays abstract role   |
| `body`           | `None`             | `body`               | null for MIND                  |
| `category`       | `category`         | `category_str`       |                                |
| `subcategory`    | `subcategory`      | `subcategory_ids[0]` | first element for parity       |
| `published_time` | `None`             | `published_time`     | null for MIND                  |
| `lexical_text`   | precomputed        | precomputed          | preprocessed, BM25-ready       |
| `dataset`        | `"mind"`           | `"ebnerd"`           |                                |

### `behaviors.parquet`
| Column             | MIND source               | EB-NeRD source              | Notes                        |
|--------------------|---------------------------|-----------------------------|------------------------------|
| `impression_id`    | `impression_id`           | `impression_id`             |                              |
| `user_id`          | `user_id`                 | `user_id`                   | cast to `str`                |
| `impression_time`  | `time`                    | `impression_time`           | parsed to `datetime`         |
| `candidate_ids`    | parsed from impressions   | `article_ids_inview`        | `list[str]`                  |
| `labels`           | parsed from impressions   | derived from clicked ids    | `list[int]` — 1=click, 0=not |
| `split`            | assigned post-split       | assigned post-split         | `train`/`val`/`test`         |
| `dataset`          | `"mind"`                  | `"ebnerd"`                  |                              |

### `history.parquet`
| Column          | MIND source          | EB-NeRD source            | Notes                        |
|-----------------|----------------------|---------------------------|------------------------------|
| `user_id`       | `user_id`            | `user_id`                 |                              |
| `impression_id` | `impression_id`      | joined from behaviors     |                              |
| `click_history` | `history` split by space | `article_id_fixed`    | `list[str]`                  |
| `n_clicks`      | `len(click_history)` | `len(click_history)`      | for cold/warm slicing        |
| `dataset`       | `"mind"`             | `"ebnerd"`                |                              |

---

## Data Preprocessing

### MIND-small
- Source files: `news.tsv`, `behaviors.tsv` (train + dev)
- Text fields: `title` + `abstract` (fallback to title only if abstract is null — common in MIND)
- Preprocessing for BM25:
  - lowercase
  - punctuation removal
  - English stopword removal (NLTK)
  - no stemming needed for English
- `lexical_text = preprocess_en(title + " " + abstract)`

### EB-NeRD small
- Source files: `articles.parquet`, `train/behaviors.parquet`, `train/history.parquet`
- Text fields: `title` + `subtitle` (not body — body is too long for BM25, causes length norm issues)
- Preprocessing for BM25 — Danish-specific:
  - lowercase
  - punctuation removal
  - Danish stopword removal (NLTK `stopwords.words('danish')`)
  - Snowball stemmer for Danish (`nltk.stem.snowball.SnowballStemmer("danish")`)
  - UTF-8 safe — preserve æ, ø, å (Snowball handles them natively; do not ASCII-normalize)
  - **Apply identical preprocessing to both documents and queries**
- `lexical_text = preprocess_da(title + " " + subtitle)`

---

## Temporal Split

Applied to `behaviors.parquet` impression timestamps. Both MIND and EB-NeRD span ~6 weeks.

```
|── train ──────────────────|── val ──|── test ──|
              week -4 to -2      week -1    last week
```

- **Test:** last 1 week of impressions by `impression_time`
- **Val:** 1 week immediately before test
- **Train:** everything before val

Split is assigned as a `split` column in `behaviors.parquet`. Never random — strictly time-based.
Add an assertion in `split.py` that `max(train_time) < min(val_time) < min(test_time)` to guard against leakage.

---

## Lexical Pipeline (BM25)

**Library:** `bm25s` (faster than `rank_bm25`, numpy-backed)

**Index input:** `lexical_text` column from `articles.parquet`

**Text fields indexed:**
- MIND: `title + abstract`
- EB-NeRD: `title + subtitle`

**Query construction:**
- Take the last `K` clicked articles from `click_history` (configurable, sweep K ∈ {5, 10, 20})
- Concatenate their titles → apply same preprocessing function as indexing
- `query = preprocess(dataset)(concat of last K clicked titles)`

**Parameters:**
- `k1 = 1.5`, `b = 0.75` (defaults, language-agnostic)

**Retrieval:** top-K candidates per impression for K ∈ {50, 100, 200}

**Output:** ranked candidate list per impression → fed to evaluator

---

## Semantic Pipeline (Embeddings + ANN)

### MIND-small

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- English-only, 384-dim, fast
- `normalize_embeddings=True` (unit norm → dot product = cosine similarity)

**Text encoded:** `title + " " + abstract` (fallback to title if abstract null)

**Generation:** run `notebooks/generate_mind_embeddings.ipynb` on Google Colab (T4 GPU)
- Output: `embeddings.npy` (shape: n_articles × 384), `article_id_index.parquet`
- Upload to Google Drive
- `build.py` auto-downloads via `gdown` if `artifacts/mind/embeddings.npy` not present

**User representation:** mean pool of clicked article embeddings
```python
user_vec = mean([embeddings[article_id_to_idx[aid]] for aid in click_history[-K:]])
```
- Same K as BM25 query construction (sweep K ∈ {5, 10, 20})
- Cold-start fallback (empty history): zero vector

### EB-NeRD small

**Embeddings:** precomputed `google_bert_base_multilingual_cased` (already downloaded from S3)
- Multilingual BERT, appropriate for Danish
- Load from `artifacts/ebnerd/embeddings.parquet`

**User representation:** same mean-pool approach as MIND

### ANN Index (both datasets)

**Library:** `faiss-cpu`
**Index type:** `IndexFlatIP` (exact brute-force inner product)
- Sufficient at this scale (42K articles MIND, ~125K EB-NeRD)
- No approximation error — cleaner recall@K numbers
- No GPU needed

**At local eval time:** ANN search over full article corpus → recall@K
**At CodaBench submission time:** score only the provided candidate list per impression (dot product, no ANN search)

---

## Offline Evaluation Harness

Implemented in `pipeline/evaluate.py`. Runs identically on both datasets.

### Ranking Metrics
- AUC
- MRR
- nDCG@5
- nDCG@10

### Beyond-Accuracy Metrics
- **Intra-list diversity:** mean pairwise category dissimilarity in top-K results
- **Novelty:** inverse popularity of retrieved articles (log(1/click_count))
- **Coverage:** fraction of article catalog retrieved at least once across all impressions

### Slicing
- **Cold vs warm users:** cold = `n_clicks < 5`, warm = `n_clicks >= 5`
- **Head vs tail articles:** head = top 20% by impression frequency, tail = bottom 80%

### Confidence Intervals
- Bootstrap 95% CI (n=1000 resamples) for all metrics

### Leakage Guard
- Assertion: no article in `click_history` appears in `candidate_ids` for the same impression
- Assertion: `click_history` only contains articles published/clicked before `impression_time`

---

## CodaBench Submission

**MIND:** https://www.codabench.org/competitions/13967/
**EB-NeRD:** https://www.codabench.org/competitions/2469/

Prediction files generated by `pipeline/predict.py`:
- Input: CodaBench test impression file (candidate lists, no labels)
- Score each candidate via dot product with user vector
- Output ranked list per impression in required format
- Screenshots of leaderboard scores included in design note

---

## build.py Flow

```
1. check artifacts/mind/embeddings.npy → gdown if missing
2. ingest MIND raw files → unified schema → feature_store/mind/
3. ingest EB-NeRD raw files → unified schema → feature_store/ebnerd/
4. temporal split → assign split column in behaviors.parquet
5. preprocess lexical_text for both datasets
6. build BM25 index (mind + ebnerd)
7. build FAISS IndexFlatIP (mind + ebnerd)
8. run evaluation harness on val split → print metrics
9. generate CodaBench prediction files on test split
```

Checkpoint files (`.done` flags) written after each step so partial runs can resume without redoing completed steps.

---

## Anti-Gaming Checklist

- Temporal split enforced strictly — no random splits anywhere
- Behaviour-window boundary enforced: click history only includes interactions before impression time
- Unit test asserting no future-click leakage in `split.py`
- Metrics reported on val and test splits — not on train
- BM25 and ANN both evaluated without features unavailable at serving time

---
