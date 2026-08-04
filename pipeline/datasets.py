"""The dataset registry.

This is the single place where MIND and EB-NeRD are allowed to differ. Every
pipeline stage takes a DatasetConfig and must work for any entry in DATASETS.
If a stage needs to know which dataset it is looking at, add a field here
rather than branching on the name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline import paths

# A unified-schema column with no direct source column: the ingest stage
# (ticket 3) builds it from other source columns.
DERIVED = "<derived>"

# The unified schema. Both datasets map onto exactly these columns.
ARTICLE_COLUMNS = (
    "article_id",
    "title",
    "abstract",
    "body",
    "category",
    "subcategory",
    "published_time",
    "lexical_text",
    "dataset",
)
BEHAVIOR_COLUMNS = (
    "impression_id",
    "user_id",
    "impression_time",
    "candidate_ids",
    "labels",
    "split",
    "dataset",
)
HISTORY_COLUMNS = (
    "user_id",
    "impression_id",
    "click_history",
    "n_clicks",
    "dataset",
)


@dataclass(frozen=True)
class Archive:
    """One downloadable file."""

    url: str
    filename: str


@dataclass(frozen=True)
class RawSpec:
    archives: tuple[Archive, ...]
    # Paths relative to the dataset's raw directory, present after extraction.
    expected_files: tuple[str, ...]
    # Environment variable holding the download credential, or None if public.
    token_env: str | None


@dataclass(frozen=True)
class ColumnMap:
    """Unified schema column -> source column.

    None means the dataset has no such column and ingest fills null. DERIVED
    means ingest builds it from other source columns.
    """

    articles: dict[str, str | None]
    behaviors: dict[str, str | None]
    history: dict[str, str | None]


@dataclass(frozen=True)
class EmbeddingSpec:
    # "generate": produced by a notebook on a hosted GPU, fetched as an
    # artifact. "provided": ships with the dataset.
    kind: str
    model: str
    dim: int
    artifact: str
    # Drive id of the generated artifact. Filled in by ticket 7; None until
    # the artifact exists, and unused when kind == "provided".
    gdrive_file_id: str | None


@dataclass(frozen=True)
class SubmissionSpec:
    competition_url: str
    filename: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    language: str
    raw: RawSpec
    columns: ColumnMap
    embeddings: EmbeddingSpec
    submission: SubmissionSpec

    @property
    def raw_dir(self) -> Path:
        return paths.RAW_DIR / self.name

    @property
    def feature_store_dir(self) -> Path:
        return paths.FEATURE_STORE_DIR / self.name

    @property
    def artifacts_dir(self) -> Path:
        return paths.ARTIFACTS_DIR / self.name


_HF_MIND = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main"
_EBNERD_S3 = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"

MIND = DatasetConfig(
    name="mind",
    language="english",
    raw=RawSpec(
        # The official MIND endpoint is dead (HTTP 409); this HF mirror is the
        # working source and it requires a token.
        archives=(
            Archive(f"{_HF_MIND}/MINDsmall_train.zip", "MINDsmall_train.zip"),
            Archive(f"{_HF_MIND}/MINDsmall_dev.zip", "MINDsmall_dev.zip"),
        ),
        expected_files=(
            "train/news.tsv",
            "train/behaviors.tsv",
            "dev/news.tsv",
            "dev/behaviors.tsv",
        ),
        token_env="HF_TOKEN",
    ),
    # news.tsv and behaviors.tsv are headerless; these names are the ones
    # ingest assigns to the positional columns.
    columns=ColumnMap(
        articles={
            "article_id": "news_id",
            "title": "title",
            "abstract": "abstract",
            "body": None,
            "category": "category",
            "subcategory": "subcategory",
            "published_time": None,
            "lexical_text": DERIVED,
            "dataset": DERIVED,
        },
        behaviors={
            "impression_id": "impression_id",
            "user_id": "user_id",
            "impression_time": "time",
            # Both parsed out of the space-delimited "impressions" field.
            "candidate_ids": DERIVED,
            "labels": DERIVED,
            "split": DERIVED,
            "dataset": DERIVED,
        },
        history={
            "user_id": "user_id",
            "impression_id": "impression_id",
            "click_history": "history",
            "n_clicks": DERIVED,
            "dataset": DERIVED,
        },
    ),
    embeddings=EmbeddingSpec(
        kind="generate",
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        artifact="embeddings.npy",
        gdrive_file_id=None,
    ),
    submission=SubmissionSpec(
        competition_url="https://www.codabench.org/competitions/13967/",
        filename="mind_submission.txt",
    ),
)

EBNERD = DatasetConfig(
    name="ebnerd",
    language="danish",
    raw=RawSpec(
        archives=(
            Archive(f"{_EBNERD_S3}/ebnerd_small.zip", "ebnerd_small.zip"),
            Archive(
                f"{_EBNERD_S3}/artifacts/google_bert_base_multilingual_cased.zip",
                "google_bert_base_multilingual_cased.zip",
            ),
        ),
        expected_files=(
            "articles.parquet",
            "train/behaviors.parquet",
            "train/history.parquet",
            "validation/behaviors.parquet",
            "validation/history.parquet",
        ),
        token_env=None,
    ),
    columns=ColumnMap(
        articles={
            "article_id": "article_id",
            "title": "title",
            # Subtitle plays the abstract role; body exists but is not indexed.
            "abstract": "subtitle",
            "body": "body",
            "category": "category_str",
            # First element of subcategory_ids, for parity with MIND.
            "subcategory": DERIVED,
            "published_time": "published_time",
            "lexical_text": DERIVED,
            "dataset": DERIVED,
        },
        behaviors={
            "impression_id": "impression_id",
            "user_id": "user_id",
            "impression_time": "impression_time",
            "candidate_ids": "article_ids_inview",
            # Derived from the clicked ids against the in-view list.
            "labels": DERIVED,
            "split": DERIVED,
            "dataset": DERIVED,
        },
        history={
            "user_id": "user_id",
            # History is a separate table here; joined onto behaviors.
            "impression_id": DERIVED,
            "click_history": "article_id_fixed",
            "n_clicks": DERIVED,
            "dataset": DERIVED,
        },
    ),
    embeddings=EmbeddingSpec(
        kind="provided",
        model="google_bert_base_multilingual_cased",
        dim=768,
        artifact="embeddings.parquet",
        gdrive_file_id=None,
    ),
    submission=SubmissionSpec(
        competition_url="https://www.codabench.org/competitions/2469/",
        filename="ebnerd_submission.txt",
    ),
)

DATASETS = {config.name: config for config in (MIND, EBNERD)}
