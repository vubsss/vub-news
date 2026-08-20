"""The dataset registry.

This is the single place where MIND and EB-NeRD are allowed to differ. Every
pipeline stage takes a DatasetConfig and must work for any entry in DATASETS.
If a stage needs to know which dataset it is looking at, add a field here
rather than branching on the name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipeline import paths, sources, submissions

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

# The dtype every unified column carries, whichever dataset produced it. Both
# datasets are conformed to these, so no downstream stage has to care that one
# arrived as tab-separated text and the other as parquet.
COLUMN_DTYPES = {
    "article_id": "string",
    "title": "string",
    "abstract": "string",
    "body": "string",
    "category": "string",
    "subcategory": "string",
    "published_time": "datetime64[us]",
    "lexical_text": "string",
    "dataset": "string",
    "impression_id": "string",
    "user_id": "string",
    "impression_time": "datetime64[us]",
    "candidate_ids": "object",
    "labels": "object",
    "split": "string",
    "click_history": "object",
    "n_clicks": "int64",
}


@dataclass(frozen=True)
class Archive:
    """One downloadable file and where its contents belong.

    extract_to is a subdirectory of the dataset's raw directory; "" means the
    raw directory itself. A single wrapping directory inside the archive is
    stripped on extraction, so the layout below extract_to is the archive's
    real content either way.
    """

    url: str
    filename: str
    extract_to: str


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
class TableSource:
    """Where one unified table's raw input lives and how to turn it into one."""

    # Paths under the dataset's raw directory, concatenated in order.
    files: tuple[str, ...]
    format: str  # "tsv" or "parquet"
    # Positional column names, for headerless tsv. None when the file has them.
    header: tuple[str, ...] | None
    # The adapter from pipeline.sources that maps this dataset's raw frame onto
    # the unified schema.
    adapt: Callable[..., object]


@dataclass(frozen=True)
class SourceSpec:
    articles: TableSource
    behaviors: TableSource
    history: TableSource


@dataclass(frozen=True)
class SplitSpec:
    """How many days at the end of the log each held-out partition takes.

    SPEC.md asks for a week each, which the ~6-week full releases support. The
    small distributions shipped here are one week (MIND) and two (EB-NeRD), so
    a week each would leave train empty; the windows are scaled to the span
    each dataset actually covers.

    `tune_days` is carved off the end of train and is where every parameter is
    selected — the history window, the BM25 constants, the embedding
    post-processing. It exists because validation cannot both choose a
    parameter and report the result of that choice: doing both makes the
    reported number a description of the selection rather than of the
    retriever. Taken from the end of train rather than the start because it is
    then adjacent in time to validation, which is the population it is standing
    in for.
    """

    tune_days: int
    val_days: int
    test_days: int


@dataclass(frozen=True)
class EmbeddingSpec:
    # "generate": produced by a notebook on a hosted GPU, fetched as an
    # artifact. "provided": ships with the dataset.
    kind: str
    model: str
    dim: int
    # Where the vectors are read from. Relative to the dataset's raw directory
    # when kind == "provided", and to its artifacts directory when the
    # download puts them there.
    artifact: str
    # Drive id of the generated artifact. Filled in by ticket 7; None until
    # the artifact exists, and unused when kind == "provided".
    gdrive_file_id: str | None
    # Whether the pipeline scales the vectors to unit length itself. False
    # means the source already emits unit vectors and we only verify it --
    # never that non-unit vectors are acceptable, which check_unit_norm
    # rejects for either dataset.
    normalise: bool
    # Tokens the encoder truncates to, when this dataset generates its own
    # vectors. A property of the checkpoint rather than a free choice: it is
    # what all-MiniLM-L6-v2 was trained and published with, and encoding at a
    # different width would produce vectors that are not the model's. None
    # when kind == "provided" and nothing here does the encoding.
    max_tokens: int | None
    # How the vectors' geometry is corrected before they are indexed: "none",
    # "centre", "abtt:n" or "whiten". A registry decision rather than a stage
    # one, exactly as `normalise` is.
    #
    # Raw transformer output occupies a narrow cone, so every pair of articles
    # has a high cosine whatever they say and a retriever's signal rides as a
    # residual on a shared offset. EB-NeRD's shipped mBERT vectors have a mean
    # pairwise cosine of 0.95 and its semantic retriever scores at chance as a
    # direct consequence; MIND's sentence-trained MiniLM sits at 0.06 and wants
    # no correction. Which method to apply is chosen on the tune split -- the
    # statistics themselves are fitted on the corpus, which is not a split.
    postprocess: str = "none"


@dataclass(frozen=True)
class SubmissionSpec:
    """What the competition hands over, and what it will accept back.

    Its test impressions are not the pipeline's test split. The competition
    supplies the candidate list per impression and expects exactly those
    ranked, over a period later than the feature store covers, so it ships its
    own catalogue and its own click histories and is acquired separately from
    the raw spec above. Everything a leaderboard file's shape depends on lives
    here; `pipeline/predict.py` reads it and never a dataset name.

    Every field below `competition_url` is None for a competition whose
    submission is not built yet, and the predict stage says which ticket owns
    it rather than writing a file the leaderboard would reject.
    """

    competition_url: str
    # The archive holding the test impressions, extracted under the dataset's
    # raw directory exactly as the raw spec's archives are.
    archives: tuple[Archive, ...] | None = None
    expected_files: tuple[str, ...] | None = None
    token_env: str | None = None
    # The catalogue the candidates are drawn from. Read from the competition's
    # own files: the pipeline's corpus predates the test period and holds
    # almost none of the articles that appear as candidates in it.
    articles: TableSource | None = None
    # impression_id, user_id, candidate_ids, click_history -- and no labels,
    # which is what makes this a different adapter from the behaviours one.
    impressions: TableSource | None = None
    # Where the click history comes from, for a competition that ships it as
    # its own table keyed by user rather than on the impression row. None when
    # the impressions adapter already returns click_history, as MIND's does.
    history: TableSource | None = None
    # The name the leaderboard requires inside the zip, and the zip itself.
    filename: str | None = None
    bundle: str | None = None
    # One impression's 1-based ranks, in the order the competition listed its
    # candidates, formatted as one line of the prediction file.
    line: Callable[[str, list[int]], str] | None = None
    # An impression id the competition stamps on more than one row on purpose,
    # exempt from the "every impression appears once" guard. Every other id is
    # still held to it, so this weakens the check by exactly one value rather
    # than turning it off.
    repeated_impression_id: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    language: str
    raw: RawSpec
    split: SplitSpec
    sources: SourceSpec
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
            Archive(
                f"{_HF_MIND}/MINDsmall_train.zip", "MINDsmall_train.zip", "train"
            ),
            Archive(f"{_HF_MIND}/MINDsmall_dev.zip", "MINDsmall_dev.zip", "dev"),
        ),
        expected_files=(
            "train/news.tsv",
            "train/behaviors.tsv",
            "dev/news.tsv",
            "dev/behaviors.tsv",
        ),
        token_env="HF_TOKEN",
    ),
    # MINDsmall covers one week, so a week each for validation and test would
    # leave train empty; a day each keeps train the largest partition. One more
    # day for tuning still leaves train four times the size of any other
    # partition.
    split=SplitSpec(tune_days=1, val_days=1, test_days=1),
    sources=SourceSpec(
        articles=TableSource(
            files=("train/news.tsv", "dev/news.tsv"),
            format="tsv",
            header=(
                "news_id",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ),
            adapt=sources.mind_articles,
        ),
        behaviors=TableSource(
            files=("train/behaviors.tsv", "dev/behaviors.tsv"),
            format="tsv",
            header=("impression_id", "user_id", "time", "history", "impressions"),
            adapt=sources.mind_behaviors,
        ),
        # MIND has no separate history file: it is a column on each impression.
        history=TableSource(
            files=("train/behaviors.tsv", "dev/behaviors.tsv"),
            format="tsv",
            header=("impression_id", "user_id", "time", "history", "impressions"),
            adapt=sources.mind_history,
        ),
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
        gdrive_file_id="1tVfeai5eUVvrhRGZdowZQntAwHt2VkQR",
        # embed.encode normalises as it goes, so the uploaded artifact is
        # already unit length. Verified on load rather than redone.
        normalise=False,
        max_tokens=256,
        # Chosen on tune over the same seven settings EB-NeRD was, 30,894
        # scorable impressions:
        #
        #   none 0.6250   centre 0.6320   abtt:1 0.6213   abtt:3 0.6061
        #   abtt:5 0.5940  abtt:10 0.5810  whiten 0.5793
        #
        # The contrast with EB-NeRD is the point. MiniLM is sentence-trained
        # and arrives near-isotropic at 0.0630, so there is barely a cone to
        # remove: correction buys 0.007 here against 0.060 there, and removing
        # more than the mean costs up to 0.046, because on vectors that were
        # never broken the leading directions carry signal rather than offset.
        #
        # The tune preference for `centre` did not replicate. On validation
        # auc goes 0.6252 [0.6219, 0.6286] -> 0.6227 [0.6195, 0.6259] -- down,
        # not up, with the intervals overlapping -- while mrr, ndcg@5, ndcg@10
        # and recall@200 (0.0344 -> 0.0456) all rise. So nothing is
        # established either way on the ranking metrics here, and the honest
        # reading is that MIND's geometry was not broken enough for this to
        # matter.
        #
        # Kept anyway, because it is what the tune split chose. Reverting on
        # the strength of a validation number would be selecting on validation,
        # which is the contamination the tune split exists to prevent -- and a
        # marginal effect that fails to replicate is the ordinary outcome that
        # holding out a reporting split is designed to expose.
        postprocess="centre",
    ),
    submission=SubmissionSpec(
        competition_url="https://www.codabench.org/competitions/13967/",
        # The only phase still open is Official Test, scored against
        # MINDlarge_test -- a later week than MINDsmall, and the reason the
        # submission path indexes the competition's catalogue rather than the
        # feature store's.
        archives=(
            Archive(f"{_HF_MIND}/MINDlarge_test.zip", "MINDlarge_test.zip", "test"),
        ),
        expected_files=("test/news.tsv", "test/behaviors.tsv"),
        token_env="HF_TOKEN",
        articles=TableSource(
            files=("test/news.tsv",),
            format="tsv",
            header=(
                "news_id",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ),
            adapt=sources.mind_articles,
        ),
        impressions=TableSource(
            files=("test/behaviors.tsv",),
            format="tsv",
            header=("impression_id", "user_id", "time", "history", "impressions"),
            adapt=sources.mind_test_impressions,
        ),
        filename="prediction.txt",
        bundle="mind_submission.zip",
        line=submissions.mind_line,
    ),
)

EBNERD = DatasetConfig(
    name="ebnerd",
    language="danish",
    raw=RawSpec(
        archives=(
            Archive(f"{_EBNERD_S3}/ebnerd_small.zip", "ebnerd_small.zip", ""),
            Archive(
                f"{_EBNERD_S3}/artifacts/google_bert_base_multilingual_cased.zip",
                "google_bert_base_multilingual_cased.zip",
                "embeddings",
            ),
        ),
        expected_files=(
            "articles.parquet",
            "train/behaviors.parquet",
            "train/history.parquet",
            "validation/behaviors.parquet",
            "validation/history.parquet",
            "embeddings/bert_base_multilingual_cased.parquet",
        ),
        token_env=None,
    ),
    # ebnerd_small covers two weeks: three days each, train keeps the rest. Two
    # days to tune on, which is the smallest window that still holds enough
    # impressions to separate two configurations.
    split=SplitSpec(tune_days=2, val_days=3, test_days=3),
    sources=SourceSpec(
        articles=TableSource(
            files=("articles.parquet",),
            format="parquet",
            header=None,
            adapt=sources.ebnerd_articles,
        ),
        behaviors=TableSource(
            files=("train/behaviors.parquet", "validation/behaviors.parquet"),
            format="parquet",
            header=None,
            adapt=sources.ebnerd_behaviors,
        ),
        history=TableSource(
            files=("train/history.parquet", "validation/history.parquet"),
            format="parquet",
            header=None,
            adapt=sources.ebnerd_history,
        ),
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
        artifact="embeddings/bert_base_multilingual_cased.parquet",
        gdrive_file_id=None,
        # The shipped vectors are raw BERT output with norms around 12.4, not
        # unit length. Normalised here so an inner product is a cosine, as it
        # is for MIND: ticket 8 indexes both with faiss IndexFlatIP, and on
        # unnormalised vectors that ranks by magnitude as much as by direction,
        # which would make the two datasets' recall figures measure different
        # things and leave magnitude as a free variable in ticket 11's
        # lexical-versus-semantic comparison.
        normalise=True,
        # Nothing encodes for EB-NeRD; the vectors arrive already made.
        max_tokens=None,
        # Chosen on the tune split over {none, centre, abtt:1/3/5/10, whiten},
        # 66,908 impressions. The shipped vectors rank at chance because they
        # occupy a narrow cone; removing the mean and three principal
        # directions is what makes an inner product between two of them mean
        # anything:
        #
        #   none 0.4877   centre 0.5199   abtt:1 0.5409   abtt:3 0.5477
        #   abtt:5 0.5441  abtt:10 0.5322  whiten 0.5216
        #
        # Note the peak is not where the geometry is best. Anisotropy falls
        # monotonically across that row -- whiten reaches 0.0001, the most
        # isotropic of the seven -- while AUC turns over at three components.
        # Past that the correction is removing signal along with the offset,
        # so the statistic diagnoses the problem and does not pick the fix.
        postprocess="abtt:3",
    ),
    submission=SubmissionSpec(
        competition_url="https://www.codabench.org/competitions/2469/",
        # ebnerd_testset covers a later week than ebnerd_small and ships its
        # own catalogue, so it is extracted under `testset/` rather than into
        # the raw directory: its articles.parquet is a different file from the
        # one the feature store is built from and must not land on top of it.
        archives=(
            Archive(f"{_EBNERD_S3}/ebnerd_testset.zip", "ebnerd_testset.zip", "testset"),
        ),
        expected_files=(
            "testset/articles.parquet",
            "testset/test/behaviors.parquet",
            "testset/test/history.parquet",
        ),
        token_env=None,
        articles=TableSource(
            files=("testset/articles.parquet",),
            format="parquet",
            header=None,
            adapt=sources.ebnerd_articles,
        ),
        impressions=TableSource(
            files=("testset/test/behaviors.parquet",),
            format="parquet",
            header=None,
            adapt=sources.ebnerd_test_impressions,
        ),
        # The one shape difference from MIND's submission: history is its own
        # table here, one row per user, joined on by the predict stage.
        history=TableSource(
            files=("testset/test/history.parquet",),
            format="parquet",
            header=None,
            adapt=sources.ebnerd_history,
        ),
        # `predictions.txt`, plural, unlike MIND's -- the name the challenge's
        # own write_submission_file defaults to and the scorer looks for.
        filename="predictions.txt",
        bundle="ebnerd_submission.zip",
        line=submissions.ebnerd_line,
        # The test file's 13,336,710 ordinary impressions all carry distinct
        # ids; its 200,000 beyond-accuracy impressions — the 250-candidate
        # lists the challenge scores for diversity rather than for clicks — are
        # every one of them stamped 0. So id is not a row key here, and the
        # submission's real one-to-one guarantee is that it writes one line per
        # input row in the input's order.
        repeated_impression_id="0",
    ),
)

DATASETS = {config.name: config for config in (MIND, EBNERD)}
