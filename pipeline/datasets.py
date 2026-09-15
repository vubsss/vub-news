"""The dataset registry.

This is the single place where MIND and EB-NeRD are allowed to differ. Every
pipeline stage takes a DatasetConfig and must work for any entry in DATASETS.
If a stage needs to know which dataset it is looking at, add a field here
rather than branching on the name.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
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
    # Which of the dataset's source files this impression came from -- `train`,
    # `dev`, `validation`. Already inside `impression_id`, which is qualified
    # with it because ids are only unique within a file, and materialised here
    # because the history table is keyed by it: parsing it back out of 18M ids
    # costs 5.3 s every time the table is read, and this column costs nothing,
    # being one of two values and dictionary-encoded by parquet.
    "source",
    "user_id",
    "impression_time",
    # The browsing session the impression was served in, for the session
    # features of A2. EB-NeRD ships one; MIND has no notion of a session and
    # carries the column as null, the way it carries `published_time` -- a
    # feature over it is null there, and no stage asks which dataset it holds.
    "session_id",
    "candidate_ids",
    "labels",
    "split",
    "dataset",
)
# The three engagement columns are arrays parallel to the *last*
# `sources.ENGAGEMENT_WINDOW` entries of click_history, one per past click;
# that module truncates them, and says there why. Every one of them describes a click that has already
# happened, so every one is available at serving time -- which is what makes
# them usable at all. MIND ships a bare id list and carries them as null: a
# dataset that has no such column has a null column, never a missing one, so
# no stage has to ask which dataset it is holding.
HISTORY_COLUMNS = (
    "user_id",
    # Keyed by the user and the file the history was read from -- **not** by
    # impression. One row per user per snapshot, joined onto impressions at
    # read time by `ingest.history_for`.
    #
    # Per-impression was 25x redundant on EB-NeRD: 477,534 rows for 18,827
    # users, each user's 292-click list written once per impression they appear
    # in, 454 MB. It is the first table that breaks at 10x scale and the
    # measurement that says so is in phase 8.
    #
    # `source` rather than `user_id` alone, because a user genuinely has more
    # than one history: EB-NeRD ships a train and a validation snapshot and
    # 11,657 of its 18,827 users appear in both with different click lists.
    # Collapsing those would apply the later snapshot to the earlier period's
    # impressions, which is future clicks leaking into past ones. `split` is
    # not fine enough either -- 2,617 (user, split) pairs span both snapshots.
    "source",
    "click_history",
    "click_times",
    "click_read_times",
    "click_scroll",
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
    "source": "string",
    "user_id": "string",
    "impression_time": "datetime64[us]",
    "session_id": "string",
    "candidate_ids": "object",
    "labels": "object",
    "split": "string",
    "click_history": "object",
    "click_times": "object",
    "click_read_times": "object",
    "click_scroll": "object",
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
class LexicalSpec:
    """BM25's parameters, per dataset.

    `k1` controls term-frequency saturation and `b` document-length
    normalisation. Both were module constants copied from SPEC.md and never
    measured, which matters more here than it usually would: these documents
    are a title plus an abstract, and 5% of MIND and 8% of EB-NeRD have no
    abstract at all, so the corpus is far shorter than the ones the defaults
    were chosen against.

    `title_weight` repeats the title's tokens when the indexed text is built.
    Concatenating title and abstract into one bag -- which is what this
    pipeline did -- scores a term in a six-word title exactly as it scores one
    in a forty-word abstract, and in news the title carries most of the signal.
    Repetition is an approximation of BM25F rather than BM25F itself: it
    weights the term frequency before saturation, which is the mechanism, but
    it also lengthens the document, which real per-field normalisation would
    not. Said plainly here because `bm25s` indexes one field and cannot express
    the exact form.

    `query_abstract` is the other side of the same question, on the query
    rather than the document: whether the click history a query is built from
    contributes each article's title only, or its title and abstract. The
    module that built the query asserted titles were better because abstracts
    would drown the identifying terms. That was an argument until phase 3
    measured it, and it is a field here because the answer is per dataset.
    """

    k1: float
    b: float
    title_weight: int = 1
    query_abstract: bool = False


@dataclass(frozen=True)
class HybridSpec:
    """How the lexical and semantic rankings are combined, per dataset.

    `rule` is `rrf` or `linear`. RRF keeps only the order, which is what makes
    it the safer default: BM25 scores are unbounded sums of term weights and
    cosines live in [-1, 1], and nothing in either makes them comparable.

    `k` is RRF's rank constant — 60 by convention, so a sweep centres there
    rather than searching blind. `alpha` weights the *lexical* side of the
    linear rule, so alpha=1 is BM25 alone and alpha=0 is the semantic index
    alone. Only one of the two is read at a time, and the sweep says which.
    """

    rule: str = "rrf"
    k: float = 60.0
    alpha: float = 0.5


@dataclass(frozen=True)
class WeightingSpec:
    """How much each past click counts toward the profile, per dataset.

    `scheme` is one of `weighting.SCHEMES`; which of them a dataset can express
    is derived from the ColumnMap above rather than repeated here, because a
    second list is a second thing to keep true.

    `decay` is the constant the active scheme reads, and means something
    different in each: for `position` it is the per-click multiplier, so 0.9
    means a click counts 10% less than the one after it; for `time` it is the
    half-life in *hours*. `uniform` and `engagement` read neither and leave it
    at its default. One field rather than one per scheme because exactly one
    scheme is active at a time, and a sweep that varied a constant belonging to
    a scheme it was not running would report cells that differ in nothing.
    """

    scheme: str = "uniform"
    decay: float = 1.0


@dataclass(frozen=True)
class FeatureSpec:
    """How the A2 feature frame is materialised, per dataset.

    Nothing here changes a single number in the frame -- the three fields are
    what the build costs and what the file weighs, which is why they sit beside
    the functional specs rather than inside them.

    `chunk_impressions` is how many impressions are turned into rows before a
    row group is written and the memory dropped, so peak RSS is a function of
    it rather than of the split's length. `None` is the whole split at once,
    which is the row the sweep compares the others against.

    `row_group` is how many *rows* parquet keeps together, which is the unit an
    ablation arm's projected read pays for. `precision` is `float32` or
    `float16`; the frame is the largest thing A2 writes and halving it is worth
    measuring, so both are built and ticket 06 trains on each.

    The values here are where the sweep starts. The chosen ones are written
    back with the ledger rows that chose them, the way every other spec in this
    registry records the comparison that settled it.
    """

    chunk_impressions: int | None = 200_000
    row_group: int = 512_000
    precision: str = "float32"


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
    # How a sequence's token vectors become one vector. A property of the
    # checkpoint, not a choice: sentence-transformers publishes it in the
    # model's own `1_Pooling/config.json`, and reading it the other way is
    # silent -- the vectors come out well-formed, unit length, and not the
    # model's. all-MiniLM-L6-v2, all-mpnet-base-v2 and e5 pool the mean;
    # bge-base-en-v1.5 takes the CLS token.
    pooling: str = "mean"
    # Prepended to every document before it is tokenised, for the checkpoints
    # that were trained with an instruction. e5 requires one on *every* input
    # and degrades quietly without it; the sentence-transformers models and
    # bge-*-v1.5 want none on the document side.
    #
    # There is only a document side here. This pipeline never encodes a query:
    # a user profile is the mean of the vectors of articles they clicked, so
    # both sides of every dot product are article embeddings. e5's asymmetric
    # `query: `/`passage: ` split has nothing to attach to, and bge's query
    # instruction has nowhere to go.
    prefix: str = ""
    # What a comparison table calls this variant. Defaults to the checkpoint's
    # name, which is the right label until two variants share a checkpoint --
    # e5 under its two prefix conventions is one model and two vector sources,
    # and a grid that labelled both `intfloat/e5-base-v2` would report two
    # different measurements under one name.
    label: str = ""
    # What produces the vectors. "transformer" runs `embed.encode`; "word2vec"
    # averages pre-trained word vectors and reads none of the fields above
    # about pooling, prefixes or truncation.
    encoder: str = "transformer"

    @property
    def name(self) -> str:
        return self.label or self.model


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
    lexical: LexicalSpec
    weighting: WeightingSpec
    hybrid: HybridSpec
    sources: SourceSpec
    columns: ColumnMap
    embeddings: EmbeddingSpec
    submission: SubmissionSpec
    # Other vector sources for this dataset, keyed by `spec.model`. The
    # pipeline never builds these -- `embeddings` above is what every stage
    # reads. They exist so the choice of vector source is a measured one:
    # `python -m pipeline.embed_compare` scores each on the tune split, and
    # whichever wins is promoted into `embeddings` by hand.
    embedding_variants: tuple[EmbeddingSpec, ...] = ()
    # How the A2 feature frame is written and read. Defaulted rather than
    # spelled out per dataset: both start from the same sweep, and the entry
    # that differs is the one that has been measured.
    features: FeatureSpec = FeatureSpec()

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
    # SPEC.md's values, and never measured -- phase 3 sweeps them on tune.
    lexical=LexicalSpec(k1=2.0, b=0.9, title_weight=3, query_abstract=True),
    weighting=WeightingSpec(),
    # Chosen on tune: +0.0035 AUC [+0.0023, +0.0046] over ann, the better
    # parent. RRF lost at every k from 1 to 300 because it weights both
    # parents equally and cannot say one is 0.05 AUC weaker; alpha can, and
    # 0.3 lexical is where it lands.
    hybrid=HybridSpec(rule="linear", alpha=0.3),
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
            # Qualified onto the id and kept as its own column; see
            # BEHAVIOR_COLUMNS.
            "source": DERIVED,
            "user_id": "user_id",
            "impression_time": "time",
            "session_id": None,
            # Both parsed out of the space-delimited "impressions" field.
            "candidate_ids": DERIVED,
            "labels": DERIVED,
            "split": DERIVED,
            "dataset": DERIVED,
        },
        history={
            "user_id": "user_id",
            "source": DERIVED,
            "click_history": "history",
            # MIND's history is a bare id list: no timestamp on a past click
            # and no engagement with it. Null columns rather than absent ones,
            # so the schema is the same shape for both datasets and the
            # weighting schemes that need them are simply unavailable here.
            "click_times": None,
            "click_read_times": None,
            "click_scroll": None,
            "n_clicks": DERIVED,
            "dataset": DERIVED,
        },
    ),
    embeddings=EmbeddingSpec(
        kind="generate",
        model="intfloat/e5-base-v2",
        dim=768,
        artifact="embeddings-e5-query.npy",
        # Produced by `ada/encode.sbatch` and not fetched from anywhere. The
        # committed script is `pipeline/encode_variants.py`.
        gdrive_file_id=None,
        # embed.encode normalises as it goes, so the artifact is already unit
        # length. Verified on load rather than redone.
        normalise=False,
        # e5's own max_seq_length. Read off the checkpoint, not chosen.
        max_tokens=512,
        # e5 requires a prefix on **every** input and degrades quietly without
        # one. Which prefix was not obvious and was therefore measured: its
        # card says to use `query: `/`passage: ` "for asymmetric tasks such as
        # passage retrieval", and this is not that -- a user profile is the
        # mean of the vectors of articles they clicked, so both sides of every
        # dot product are documents out of one catalogue. There is no query to
        # prefix. The card's advice for everything else is `query: ` throughout,
        # and the tune split agrees: 0.6649 [0.6618, 0.6681] against 0.6568
        # [0.6538, 0.6598] for `passage: `, intervals disjoint.
        prefix="query: ",
        label="e5-base-v2 (query:)",
        # Phase 7, on tune, 31,625 impressions, at the current HISTORY_K of 80:
        #
        #   none 0.6414   centre 0.6649   abtt:1 0.6529   abtt:3 0.6407
        #   abtt:5 0.6246                 whiten 0.6116
        #
        # `centre` by a disjoint interval over every one of the other 35 cells
        # in the grid, and over the all-MiniLM-L6-v2 this replaces: 0.6649
        # [0.6618, 0.6681] against 0.6506 [0.6473, 0.6537].
        #
        # The correction is worth more here than the encoder is. e5 arrives at
        # an anisotropy of +0.7264 -- near the +0.95 that made EB-NeRD's mBERT
        # score at chance -- and centring is worth +0.0235 to it, against
        # +0.0143 to the near-isotropic MiniLM. Phase 7's ticket predicted the
        # opposite, that MIND's geometry was already fine and only capacity
        # was left to buy. See artifacts/embeddings-mind-tune.md.
        postprocess="centre",
    ),
    # The alternatives phase 7 encodes and compares. MIND ships no vectors at
    # all, so unlike EB-NeRD's four shipped artifacts every one of these has to
    # be produced -- `ada/encode.sbatch` runs them through one path, and the
    # comparison reads them back as ordinary sources.
    #
    # `max_tokens` is each checkpoint's own `max_seq_length`, read from the
    # model rather than chosen: encoding at a different width produces vectors
    # that are not the model's.
    embedding_variants=(
        # What the pipeline ran on until phase 7, and still the cheapest thing
        # here at 384 dimensions. Chosen originally because it fitted a
        # free-tier Colab GPU -- a compute constraint rather than a finding,
        # which is what phase 7 went to the cluster to settle. Its own tune
        # grid, at the HISTORY_K of 10 that phase 2 measured under, read
        # none 0.6250 / centre 0.6320; re-measured at 80 it reads
        # none 0.6363 / centre 0.6506. The window moved, so those two sets of
        # numbers are not comparable to each other -- only within a grid.
        EmbeddingSpec(
            kind="generate",
            model="sentence-transformers/all-MiniLM-L6-v2",
            dim=384,
            artifact="embeddings.npy",
            gdrive_file_id="1tVfeai5eUVvrhRGZdowZQntAwHt2VkQR",
            normalise=False,
            max_tokens=256,
            label="all-MiniLM-L6-v2",
        ),
        EmbeddingSpec(
            kind="generate",
            model="sentence-transformers/all-mpnet-base-v2",
            dim=768,
            artifact="embeddings-mpnet.npy",
            gdrive_file_id=None,
            normalise=False,
            max_tokens=384,
            label="all-mpnet-base-v2",
        ),
        EmbeddingSpec(
            kind="generate",
            model="BAAI/bge-base-en-v1.5",
            dim=768,
            artifact="embeddings-bge.npy",
            gdrive_file_id=None,
            normalise=False,
            max_tokens=512,
            # Its own 1_Pooling/config.json says cls, and nothing errors if we
            # read it as mean -- the vectors come out unit length and wrong.
            pooling="cls",
            # v1.5 was released specifically to work without the instruction,
            # and the instruction is a *query* one besides. Nothing here
            # encodes a query.
            label="bge-base-en-v1.5",
        ),
        # e5 twice, because its prefix convention does not map onto this
        # pipeline and the honest way to pick between the two readings is to
        # measure them. Its card requires a prefix on every input and says to
        # use `query: `/`passage: ` "for asymmetric tasks such as passage
        # retrieval"; scoring an article against the mean of a user's clicked
        # articles is not that -- both sides are documents from one catalogue.
        # So one entry reads them as passages, and the other takes the card's
        # own advice for non-asymmetric tasks, which is `query: ` throughout.
        EmbeddingSpec(
            kind="generate",
            model="intfloat/e5-base-v2",
            dim=768,
            artifact="embeddings-e5-passage.npy",
            gdrive_file_id=None,
            normalise=False,
            max_tokens=512,
            prefix="passage: ",
            label="e5-base-v2 (passage:)",
        ),
        EmbeddingSpec(
            kind="generate",
            model="intfloat/e5-base-v2",
            dim=768,
            artifact="embeddings-e5-query.npy",
            gdrive_file_id=None,
            normalise=False,
            max_tokens=512,
            prefix="query: ",
            label="e5-base-v2 (query:)",
        ),
        # The floor, and the assignment names it directly. EB-NeRD's own
        # word2vec artifact beat three transformers at 768 dimensions, which
        # makes this the one variant here whose result is genuinely in doubt.
        EmbeddingSpec(
            kind="generate",
            model="fse/word2vec-google-news-300",
            dim=300,
            artifact="embeddings-word2vec.npy",
            gdrive_file_id=None,
            # A mean of word vectors is not unit length; the pipeline scales it.
            normalise=True,
            max_tokens=None,
            encoder="word2vec",
            label="word2vec-google-news-300",
        ),
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
            # The alternatives `embedding_variants` below describes. Fetched
            # with the rest of the raw data rather than on demand: they are
            # the evidence for which vector source this dataset should use,
            # and a comparison nobody can reproduce is not evidence.
            Archive(
                f"{_EBNERD_S3}/artifacts/Ekstra_Bladet_contrastive_vector.zip",
                "Ekstra_Bladet_contrastive_vector.zip",
                "embeddings",
            ),
            Archive(
                f"{_EBNERD_S3}/artifacts/Ekstra_Bladet_word2vec.zip",
                "Ekstra_Bladet_word2vec.zip",
                "embeddings",
            ),
            Archive(
                f"{_EBNERD_S3}/artifacts/FacebookAI_xlm_roberta_base.zip",
                "FacebookAI_xlm_roberta_base.zip",
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
            "embeddings/contrastive_vector.parquet",
            "embeddings/document_vector.parquet",
            "embeddings/xlm_roberta_base.parquet",
        ),
        token_env=None,
    ),
    # ebnerd_small covers two weeks: three days each, train keeps the rest. Two
    # days to tune on, which is the smallest window that still holds enough
    # impressions to separate two configurations.
    split=SplitSpec(tune_days=2, val_days=3, test_days=3),
    # SPEC.md's values, and never measured -- phase 3 sweeps them on tune.
    lexical=LexicalSpec(k1=2.0, b=0.9, title_weight=2, query_abstract=True),
    # Chosen on tune in phase 4: +0.0025 AUC [+0.0013, +0.0036] over uniform,
    # paired, at k=80. Every recency scheme lost -- position and time decay
    # alike, at every constant swept -- so what carries signal here is not when
    # a click happened but how hard it was read. MIND has no counterpart.
    weighting=WeightingSpec(scheme="engagement"),
    # The tune argmax among genuinely fused cells, and it does **not** beat
    # ann alone: -0.0005 [-0.0011, +0.0002], an interval containing zero. Kept
    # on the same rule as MIND so the contrast between the two is about alpha
    # rather than about which rule ran, and reported as the tie it is.
    hybrid=HybridSpec(rule="linear", alpha=0.15),
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
            "source": DERIVED,
            "user_id": "user_id",
            "impression_time": "impression_time",
            "session_id": "session_id",
            "candidate_ids": "article_ids_inview",
            # Derived from the clicked ids against the in-view list.
            "labels": DERIVED,
            "split": DERIVED,
            "dataset": DERIVED,
        },
        history={
            "user_id": "user_id",
            # Its own table here, one row per user per file — which is now the
            # shape the feature store keeps for both datasets.
            "source": DERIVED,
            "click_history": "article_id_fixed",
            # Parallel arrays over the same clicks, all three describing a
            # click that already happened and so all three available at
            # serving time.
            "click_times": "impression_time_fixed",
            "click_read_times": "read_time_fixed",
            "click_scroll": "scroll_percentage_fixed",
            "n_clicks": DERIVED,
            "dataset": DERIVED,
        },
    ),
    embeddings=EmbeddingSpec(
        kind="provided",
        model="document_vector",
        dim=300,
        artifact="embeddings/document_vector.parquet",
        gdrive_file_id=None,
        # Ships at unit length already: a vector someone expected a cosine to
        # be taken of.
        normalise=False,
        max_tokens=None,
        # Source and correction chosen together on the tune split, over all
        # four artifacts EB-NeRD ships x six corrections. Best per source:
        #
        #   document_vector    300  abtt:1  0.5665   <- this one
        #   xlm_roberta_base   768  abtt:1  0.5646
        #   contrastive_vector 768  abtt:1  0.5602
        #   mbert              768  abtt:3  0.5477   <- what the pipeline used
        #
        # word2vec at 300 dimensions beats three transformer encoders at 768,
        # which is worth stating plainly rather than burying: on headlines and
        # subtitles a few dozen words long, a bag of trained word vectors is
        # not obviously the weaker representation, and nothing here had ever
        # measured the assumption that it was.
        #
        # See artifacts/embeddings-ebnerd-tune.md, regenerate with
        # `python -m pipeline.embed_compare --dataset ebnerd`.
        postprocess="abtt:1",
    ),
    # The three other vector sources EB-NeRD ships, all covering the same
    # 125,541 articles as the one above. Two of them arrive already unit
    # length, which is itself a signal about what they were prepared for.
    embedding_variants=(
        EmbeddingSpec(
            kind="provided",
            model="google_bert_base_multilingual_cased",
            dim=768,
            artifact="embeddings/bert_base_multilingual_cased.parquet",
            gdrive_file_id=None,
            # Raw encoder output, norms around 12.4.
            normalise=True,
            max_tokens=None,
        ),
        EmbeddingSpec(
            kind="provided",
            model="contrastive_vector",
            dim=768,
            artifact="embeddings/contrastive_vector.parquet",
            gdrive_file_id=None,
            # Ships at unit length already.
            normalise=False,
            max_tokens=None,
        ),
        EmbeddingSpec(
            kind="provided",
            model="document_vector",  # word2vec
            dim=300,
            artifact="embeddings/document_vector.parquet",
            gdrive_file_id=None,
            normalise=False,
            max_tokens=None,
        ),
        EmbeddingSpec(
            kind="provided",
            model="xlm_roberta_base",
            dim=768,
            artifact="embeddings/xlm_roberta_base.parquet",
            gdrive_file_id=None,
            # Raw encoder output, norms around 18.7.
            normalise=True,
            max_tokens=None,
        ),
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

# --- the large distributions -------------------------------------------------
# Phase 8. The same pipeline over the full releases rather than the sampled
# ones, which is the assignment's ten-times-scale question asked with data
# rather than prose.
#
# Everything but three fields is shared with the small entry, and that is the
# point: switching scale is a registry change and no stage learns about it. The
# three are the name -- which relocates the raw directory, the feature store and
# the artifacts, so the two scales cannot overwrite each other's results -- the
# archives, and the split.
#
# The **split is unchanged**, which the ticket did not expect. Both large
# releases are more *users* over the same period, not more days: MINDsmall is a
# 50k-user sample of MINDlarge's ~1M over the same six days, and ebnerd_small
# is a user sample of ebnerd_large over the same two weeks. Day counts chosen
# for the small bundles' spans are therefore right for the large ones too, and
# the spec's original "a week each" is still unaffordable for the same reason
# as before. Verified against the ingested timestamps rather than assumed --
# see phase_8.md.


def at_large_scale(
    config: DatasetConfig, archives: tuple[Archive, ...], expected_files: tuple[str, ...]
) -> DatasetConfig:
    """The same dataset, the full release. Name, archives, nothing else."""
    return dataclasses.replace(
        config,
        name=f"{config.name}_large",
        raw=dataclasses.replace(
            config.raw, archives=archives, expected_files=expected_files
        ),
    )


MIND_LARGE = at_large_scale(
    MIND,
    archives=(
        Archive(f"{_HF_MIND}/MINDlarge_train.zip", "MINDlarge_train.zip", "train"),
        Archive(f"{_HF_MIND}/MINDlarge_dev.zip", "MINDlarge_dev.zip", "dev"),
    ),
    expected_files=MIND.raw.expected_files,
)

EBNERD_LARGE = at_large_scale(
    EBNERD,
    # The four embedding artifacts come along unchanged: they cover the whole
    # release, not the sample, so the same files serve both scales.
    archives=(
        Archive(f"{_EBNERD_S3}/ebnerd_large.zip", "ebnerd_large.zip", ""),
        *EBNERD.raw.archives[1:],
    ),
    expected_files=EBNERD.raw.expected_files,
)

DATASETS = {
    config.name: config
    for config in (MIND, EBNERD, MIND_LARGE, EBNERD_LARGE)
}

# The scale the project works at unless a command is told otherwise. The large
# entries are addressable by name everywhere, and reached only on purpose:
# acquiring them is 3.7 GB and building them is phase 8's experiment rather
# than the configuration everything else is reported at. A command that fanned
# out over the whole registry by default would start that download because the
# registry gained two rows.
DEFAULT_DATASETS = (MIND.name, EBNERD.name)
