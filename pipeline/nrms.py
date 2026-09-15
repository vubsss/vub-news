"""NRMS-DocVec: the official baseline, reproduced as a fourth retriever.

The assignment asks for a published baseline reproduced and then beaten. This
is the reproduction -- the RecSys 2024 challenge's own NRMS-DocVec, which is
Wu et al.'s NRMS with the text encoder replaced by the document vectors the
dataset (or A1's embedding stage) already provides. Two encoders and a dot
product:

    news    a frozen document vector -> dropout -> linear -> ReLU
    user    the last-K clicked news vectors -> multi-head self-attention
            -> additive attention -> one vector
    score   user . candidate

Written here rather than taken from `ebrec` for the reason the rest of this
pipeline is written here: the interfaces are the project's own -- one
`rank_candidates(config, behaviors, history, history_k)` and the harness scores
it, slices it and bootstraps it with no change of its own.

**Where it fits, and why it fits there.** The re-ranker of ticket 06 takes this
model's score as a feature, so the two must not be fitted on the same
impressions: a score the GBDT sees for rows the NRMS memorised is a different
feature at training time from the one it will see in production. `train` fits
on the *earlier* half of `train` by time, and ticket 06 fits on the later half.
`halves` is where that boundary lives, and a test asserts it does not overlap.

**What is baked into the checkpoint.** The history window is part of the
trained weights, so `rank_candidates` refuses a window other than the one its
checkpoint was fitted at rather than scoring at it and reporting the number
under the same name. The same goes for `pooling`, which the harness offers the
two A1 retrievers and which has no meaning here: this user encoder is learned,
not pooled, and an aggregator handed to it would be silently ignored.

**What it costs.** The user vector is computed once per impression and dotted
against every candidate -- never a forward pass per (user, candidate) pair,
which is the same arithmetic done `len(candidates)` times. Scoring runs in
batches of `NrmsSpec.score_batch` impressions, and `rank_candidates` and
`ranker` share that path so the harness and the submission measure the same
code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from pipeline import embed, ingest, ledger, paths, retrieval, timings
from pipeline.datasets import DATASETS, DEFAULT_DATASETS, DatasetConfig, NrmsSpec

# Where the arithmetic happens. CPU by default, because that is where every
# other stage of this pipeline runs and where the laptop reproduces it; the
# cluster job passes `cuda`. Only training and batched scoring take it: the
# per-request latency in `scoring_cost` is measured on the CPU on purpose,
# since that is the serving target ticket 09 costs.
DEVICE = "cpu"

STAGE = "nrms"
DIRECTORY = "nrms"
CHECKPOINT = "model.pt"
RESULTS = "nrms-{dataset}-{split}.jsonl"
DOCUMENT = "nrms-{dataset}-{split}.md"

# Where the model is fitted and where it is chosen. Never `validation`, which
# is where it is reported, and never `test`.
FIT_SPLIT = "train"
TUNE_SPLIT = "tune"

# The grid ticket 05 runs on tune. Swept one axis at a time from the registry's
# spec rather than as a product: a full grid is six trainings to answer two
# questions, and the axes are not expected to interact.
GRID = {
    "history_length": (20, 50, 80),
    "heads": (8, 16),
    "negatives": (1, 4),
    "negative_source": ("impression", "catalogue"),
    "corrected": (True, False),
    "precision": ("fp32", "fp16"),
    "score_batch": (64, 512),
}

# The number this reproduction is compared against. **Not filled in**: it is
# the ebnerd-benchmark paper's reported NRMS-DocVec on `ebnerd_small`, and a
# figure typed from memory is exactly the kind of number a design note must not
# carry. `document` prints the gap when this is set and says it is outstanding
# when it is not.
PAPER = {
    "source": "ebnerd-benchmark (RecSys Challenge 2024), NRMS-DocVec on ebnerd_small",
    "auc": None,
}


class NrmsError(RuntimeError):
    """The model was asked for something its checkpoint cannot answer."""


# ---------------------------------------------------------------------------
# The stacking boundary.


def halves(
    behaviors: pd.DataFrame, fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`behaviors` cut in two by time: the NRMS's half, then the re-ranker's.

    By a *timestamp*, not by row count, so the two halves cannot share an
    instant: impressions stamped at the boundary all fall on one side of it.
    A random split would leak -- the same day's popularity would be in both --
    and a per-row cut would leave two impressions of the same second in
    different halves, which is the same leak in miniature.
    """
    if not 0.0 < fraction < 1.0:
        raise NrmsError(f"a stacking fraction must be strictly inside (0, 1), not {fraction}")
    moments = np.sort(behaviors["impression_time"].to_numpy("datetime64[us]"))
    if not len(moments):
        raise NrmsError("nothing to split")
    boundary = moments[min(int(len(moments) * fraction), len(moments) - 1)]
    earlier = behaviors[behaviors["impression_time"] < boundary]
    later = behaviors[behaviors["impression_time"] >= boundary]
    if earlier.empty or later.empty:
        raise NrmsError(
            f"the {fraction:.0%} boundary at {boundary} leaves one half empty: "
            f"{len(earlier)} earlier, {len(later)} later. The log does not span "
            f"enough distinct moments to stack on."
        )
    return earlier, later


# ---------------------------------------------------------------------------
# The model.


class NewsEncoder(nn.Module):
    """A frozen document vector, projected into the attention's width.

    The whole of the DocVec variant: where NRMS runs a word-level attention
    over the title's tokens, this reads the vector A1's embedding stage already
    produced. Dropout sits before the projection rather than after it, so it
    drops input dimensions -- the vector's own features -- rather than the
    projected ones.
    """

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(dim, hidden)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.project(self.drop(vectors)))


class AdditiveAttention(nn.Module):
    """One vector out of many, weighted by a learned query.

    The mask is what keeps a padded history from voting: a padded position is
    set to -inf before the softmax, so it takes exactly zero weight rather than
    a small one. Every row has at least one unmasked position by construction
    (see `history_rows`), so the softmax never sees an all-(-inf) row.
    """

    def __init__(self, hidden: int, dim: int):
        super().__init__()
        self.project = nn.Linear(hidden, dim)
        self.query = nn.Linear(dim, 1, bias=False)

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = self.query(torch.tanh(self.project(encoded))).squeeze(-1)
        weights = weights.masked_fill(~mask, float("-inf"))
        return torch.einsum("bkh,bk->bh", encoded, torch.softmax(weights, dim=1))


class UserEncoder(nn.Module):
    """The user as their recent clicks attending to each other, then pooled."""

    def __init__(self, hidden: int, heads: int, attention_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.additive = AdditiveAttention(hidden, attention_dim)

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            encoded, encoded, encoded, key_padding_mask=~mask, need_weights=False
        )
        return self.additive(attended, mask)


class Nrms(nn.Module):
    """The two encoders and the dot product between them."""

    def __init__(self, dim: int, spec: NrmsSpec):
        super().__init__()
        hidden = spec.heads * spec.head_dim
        self.news = NewsEncoder(dim, hidden, spec.dropout)
        self.user = UserEncoder(hidden, spec.heads, spec.attention_dim)

    def forward(
        self,
        clicked: torch.Tensor,
        mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """(B, C) scores from (B, K, D) clicks and (B, C, D) candidates.

        One user encoding per impression, reused for every candidate. The
        alternative -- one forward pass per (user, candidate) pair -- is the
        same arithmetic done C times and is the difference between a model that
        serves and one that does not.
        """
        user = self.user(self.news(clicked), mask)
        return torch.einsum("bch,bh->bc", self.news(candidates), user)


# ---------------------------------------------------------------------------
# Turning the feature store into tensors.


def history_rows(
    history: pd.DataFrame, index: dict[str, int], history_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Embedding rows of each impression's last-K clicks, padded, with a mask.

    Built once per run rather than per batch: it is `impressions x K` int32,
    which is 24 MB for MIND's train half and the thing a per-batch rebuild
    would spend the whole training run recomputing.

    A cold user's row is all padding, and a fully masked row would make the
    attention's softmax NaN -- so position 0 stays *attended* while its row
    stays -1, which `gather` turns into a zero vector. The two flags are
    separate on purpose: "attend here" and "there is a vector here" are
    different facts, and collapsing them would hand a cold user whatever
    article happens to sit at row 0 of the catalogue, which is a wrong feature
    that nothing downstream could see.

    Such a user scores every candidate the same, the ranking falls back to the
    order the candidates arrived in, and that is what the A1 retrievers do with
    a user they know nothing about.
    """
    rows = np.full((len(history), history_length), -1, dtype="int32")
    mask = np.zeros((len(history), history_length), dtype=bool)
    for position, clicks in enumerate(history["click_history"]):
        found = [index[click] for click in clicks[-history_length:] if click in index]
        if found:
            rows[position, : len(found)] = found
            mask[position, : len(found)] = True
        else:
            mask[position, 0] = True
    return rows, mask


def gather(vectors: np.ndarray, rows: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """(..., D) float32 vectors for `rows`, zeroed wherever there is no vector.

    A position is zero if it is not attended *or* if its row is -1, which is
    how "the catalogue has no vector for this article" is spelled. Gathered per
    batch out of a matrix that stays on disk (`embed.load(mmap)`), so the
    resident cost of the vectors is the batch's rows rather than the catalogue.
    """
    keep = mask & (rows >= 0)
    found = np.asarray(
        vectors[np.where(rows >= 0, rows, 0).reshape(-1)], dtype="float32"
    )
    found = found.reshape(*rows.shape, vectors.shape[1])
    found[~keep] = 0.0
    return torch.from_numpy(found)


def candidate_rows(
    candidates, index: dict[str, int], width: int
) -> tuple[np.ndarray, np.ndarray]:
    """One batch's candidate lists, padded to `width`, with a validity mask.

    A candidate the catalogue has no vector for is padded rather than dropped:
    it must come back ranked, and a zero vector gives it the news encoder's
    bias, which is one constant score for every such candidate. They tie, the
    sort is stable, and they keep the order the dataset listed them in.
    """
    rows = np.full((len(candidates), width), -1, dtype="int32")
    mask = np.zeros((len(candidates), width), dtype=bool)
    for position, articles in enumerate(candidates):
        found = [index.get(article, -1) for article in articles]
        rows[position, : len(found)] = found
        mask[position, : len(found)] = [row >= 0 for row in found]
    return rows, mask


# Where a negative comes from. `impression` is the paper's setting and the
# registry's default; `catalogue` is the arm that measures what it was worth.
NEGATIVE_SOURCES = ("impression", "catalogue")


def examples(
    behaviors: pd.DataFrame,
    negatives: int,
    rng: np.random.Generator,
    source: str = "impression",
    catalogue: np.ndarray | None = None,
) -> list[tuple[int, list[str]]]:
    """One training example per clicked candidate: (impression row, articles).

    The articles are the positive first and then `negatives` negatives. Under
    `impression` -- what the paper trains on -- they are drawn from the same
    impression: an article this user was shown at this moment and did not
    click, so the model learns to separate a click from its own alternatives
    rather than from the average article. Under `catalogue` they are drawn
    uniformly from every article, which is the easier problem and is here to be
    measured rather than to be used.

    With replacement only when the impression has too few negatives to fill the
    slate, which is the one case where a same-impression negative cannot be
    found and where dropping the example instead would quietly drop the
    impressions with the shortest candidate lists.
    """
    if source not in NEGATIVE_SOURCES:
        raise NrmsError(
            f"unknown negative source {source!r}, expected one of "
            f"{', '.join(NEGATIVE_SOURCES)}"
        )
    if source == "catalogue" and (catalogue is None or not len(catalogue)):
        raise NrmsError("catalogue negatives need a catalogue to draw from")

    drawn: list[tuple[int, list[str]]] = []
    for position, (candidates, labels) in enumerate(
        zip(behaviors["candidate_ids"], behaviors["labels"])
    ):
        truth = np.asarray(labels).astype(bool)
        articles = np.asarray(candidates, dtype=object)
        negative = articles[~truth]
        if not truth.any():
            continue
        if source == "impression" and not len(negative):
            continue
        for positive in articles[truth]:
            if source == "catalogue":
                # Uniform over the catalogue, minus this positive: a "negative"
                # that is the article being predicted would be a label error
                # rather than a harder example.
                pool = catalogue[catalogue != positive]
                chosen = rng.choice(pool, size=negatives, replace=len(pool) < negatives)
            else:
                chosen = rng.choice(
                    negative, size=negatives, replace=len(negative) < negatives
                )
            drawn.append((position, [positive, *chosen]))
    return drawn


# ---------------------------------------------------------------------------
# Fitting.


@dataclass(frozen=True)
class Data:
    """Everything a training run reads, read once.

    The history table is opened once per run and joined onto each split in
    memory; a loop that read it per epoch would re-read 18k lists per pass and
    a loop that read it per batch would do it a thousand times. `vectors` is
    the memory-mapped document matrix, gathered per batch.
    """

    vectors: np.ndarray
    index: dict[str, int]
    # Every article with a vector, for the negative sampler's catalogue arm.
    catalogue: np.ndarray
    fit: pd.DataFrame
    fit_rows: np.ndarray
    fit_mask: np.ndarray
    tune: pd.DataFrame
    tune_rows: np.ndarray
    tune_mask: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


def vectors_for(config: DatasetConfig, spec: NrmsSpec) -> embed.Embeddings:
    """The document matrix this spec learns over: A1's corrected vectors, or
    the raw ones from the same encoder.

    The raw path goes through `embed_compare.corpus_matrix`, which is where the
    uncorrected, catalogue-aligned matrix already lives -- so the two rows of
    this comparison differ in the correction and in nothing else.
    """
    if spec.corrected:
        return embed.load(config, mmap=True)

    from pipeline import embed_compare

    articles = pd.read_parquet(
        config.feature_store_dir / "articles.parquet", columns=["article_id"]
    )
    return embed.Embeddings(
        vectors=embed_compare.corpus_matrix(config, config.embeddings),
        article_ids=articles["article_id"].astype("string").to_numpy(dtype=object),
    )


def prepare(config: DatasetConfig, spec: NrmsSpec) -> Data:
    """The two splits, their histories and the vectors, read once each."""
    store = config.feature_store_dir
    behaviors = pd.read_parquet(store / "behaviors.parquet")
    history = pd.read_parquet(store / "history.parquet")

    fit, _ = halves(behaviors[behaviors["split"] == FIT_SPLIT], spec.train_fraction)
    fit = fit.reset_index(drop=True)
    tune = behaviors[behaviors["split"] == TUNE_SPLIT].reset_index(drop=True)
    if tune.empty:
        raise NrmsError(f"{config.name} has no {TUNE_SPLIT} impressions to stop on")

    embeddings = vectors_for(config, spec)
    index = embeddings.index
    fit_history = ingest.per_impression(history, fit)
    tune_history = ingest.per_impression(history, tune)
    fit_rows, fit_mask = history_rows(fit_history, index, spec.history_length)
    tune_rows, tune_mask = history_rows(tune_history, index, spec.history_length)
    return Data(
        vectors=embeddings.vectors,
        index=index,
        catalogue=np.asarray(embeddings.article_ids, dtype=object),
        fit=fit,
        fit_rows=fit_rows,
        fit_mask=fit_mask,
        tune=tune,
        tune_rows=tune_rows,
        tune_mask=tune_mask,
    )


def mean_auc(scores: list[np.ndarray], labels) -> float:
    """Mean per-impression AUC, on the impressions where it is defined.

    The harness's definition -- `pipeline.evaluate` scores exactly this, and
    leaves out the impressions where every candidate is clicked or none is
    rather than scoring them zero. Recomputed here rather than imported because
    this is the number early stopping reads, inside the training loop, and the
    harness is what reports the result afterwards.
    """
    found = []
    for score, truth in zip(scores, labels):
        truth = np.asarray(truth).astype(int)
        if truth.sum() in (0, len(truth)):
            continue
        found.append(roc_auc_score(truth, score))
    return float(np.mean(found)) if found else float("nan")


def score_all(
    model: Nrms,
    data_vectors: np.ndarray,
    rows: np.ndarray,
    mask: np.ndarray,
    candidates,
    index: dict[str, int],
    batch_size: int,
    device: str = DEVICE,
) -> list[np.ndarray]:
    """One score array per impression, in the order the candidates arrived.

    Batched over impressions, padded to the batch's longest candidate list.
    The padding is sliced off before the scores are returned, so a short
    impression is never scored against another's candidates.
    """
    model.eval()
    candidates = list(candidates)
    scored: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            window = candidates[start : start + batch_size]
            width = max(len(articles) for articles in window)
            rows_of, mask_of = candidate_rows(window, index, width)
            keep = mask[start : start + batch_size]
            found = model(
                gather(data_vectors, rows[start : start + batch_size], keep).to(device),
                torch.from_numpy(keep).to(device),
                gather(data_vectors, rows_of, mask_of).to(device),
            )
            found = found.cpu().numpy()
            for position, articles in enumerate(window):
                scored.append(found[position, : len(articles)].astype("float64"))
    return scored


def train(
    config: DatasetConfig,
    spec: NrmsSpec | None = None,
    data: Data | None = None,
    device: str = DEVICE,
) -> tuple[Nrms, dict]:
    """Fit on the earlier half of `train`, stop on `tune`, report the curve.

    Early stopping rather than a fixed epoch count, and the AUC one epoch
    either side of the stop is kept in the report -- so the design note can
    show the curve that chose the epoch instead of asserting that it was the
    right one.
    """
    spec = spec or config.nrms
    data = data or prepare(config, spec)
    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)

    model = Nrms(data.dim, spec).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    loss_of = nn.CrossEntropyLoss()

    curve: list[dict] = []
    best: dict | None = None
    drawn_count = 0
    best_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    started = time.perf_counter()
    for epoch in range(1, spec.epochs + 1):
        model.train()
        # Re-drawn every epoch: the negatives are a sample of the impression's
        # alternatives, not a fixed dataset, and redrawing them is free.
        drawn = examples(
            data.fit, spec.negatives, rng, spec.negative_source, data.catalogue
        )
        order = rng.permutation(len(drawn))
        drawn_count = len(drawn)
        total = 0.0
        for start in range(0, len(order), spec.batch_size):
            batch = [drawn[position] for position in order[start : start + spec.batch_size]]
            positions = np.array([position for position, _ in batch])
            slates = [articles for _, articles in batch]
            rows_of, mask_of = candidate_rows(slates, data.index, spec.negatives + 1)
            keep = data.fit_mask[positions]
            scores = model(
                gather(data.vectors, data.fit_rows[positions], keep).to(device),
                torch.from_numpy(keep).to(device),
                gather(data.vectors, rows_of, mask_of).to(device),
            )
            # The positive is slot 0 of every slate, so the target is 0 and the
            # loss is the softmax over one positive and its own negatives.
            loss = loss_of(
                scores, torch.zeros(len(batch), dtype=torch.long, device=device)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.detach().item() * len(batch)

        auc = mean_auc(
            score_all(
                model,
                data.vectors,
                data.tune_rows,
                data.tune_mask,
                data.tune["candidate_ids"],
                data.index,
                spec.score_batch,
                device,
            ),
            data.tune["labels"],
        )
        curve.append(
            {"epoch": epoch, "loss": total / max(len(drawn), 1), "tune_auc": auc}
        )
        if best is None or auc > best["tune_auc"]:
            best = curve[-1]
            best_state = {
                name: tensor.clone() for name, tensor in model.state_dict().items()
            }
        elif epoch - best["epoch"] >= spec.patience:
            break

    model.load_state_dict(best_state)
    model.to("cpu")
    return model, {
        "dataset": config.name,
        "variant": variant_of(spec),
        "spec": dataclasses.asdict(spec),
        "examples": drawn_count,
        "fit_impressions": len(data.fit),
        "tune_impressions": len(data.tune),
        "epoch": best["epoch"],
        "tune_auc": best["tune_auc"],
        "curve": curve,
        "device": device,
        "train_seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# The checkpoint.


def variant_of(spec: NrmsSpec) -> str:
    """What the ledger calls this cell: every axis the grid sweeps, in a name."""
    return (
        f"k{spec.history_length}-h{spec.heads}-n{spec.negatives}"
        f"{'' if spec.negative_source == 'impression' else '-catneg'}"
        f"-{'corrected' if spec.corrected else 'raw'}-{spec.precision}"
        f"-b{spec.score_batch}"
    )


def checkpoint_path(config: DatasetConfig, spec: NrmsSpec | None = None) -> Path:
    spec = spec or config.nrms
    return config.artifacts_dir / DIRECTORY / f"{variant_of(spec)}.pt"


def save(model: Nrms, spec: NrmsSpec, dim: int, path: Path) -> Path:
    """The weights, the spec they were fitted under, and the width they expect.

    The spec travels with the weights because the history window is part of
    them: a checkpoint that did not carry it could be scored at another window
    and would report a number under this retriever's name that this retriever
    never produced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if spec.precision == "fp16":
        state = {name: tensor.half() for name, tensor in state.items()}
    torch.save(
        {"state": state, "spec": dataclasses.asdict(spec), "dim": dim}, path
    )
    return path


def load_model(config: DatasetConfig, spec: NrmsSpec | None = None) -> tuple[Nrms, NrmsSpec, int]:
    spec = spec or config.nrms
    path = checkpoint_path(config, spec)
    if not path.exists():
        raise NrmsError(
            f"no NRMS checkpoint at {path}. Train one with "
            f"`python -m pipeline.nrms --dataset {config.name}`"
        )
    saved = torch.load(path, weights_only=False)
    stored = NrmsSpec(**saved["spec"])
    model = Nrms(saved["dim"], stored)
    # Back to float32 whatever the file holds: `precision` is the checkpoint's
    # storage, not the arithmetic's, and scoring in half on a CPU is slower
    # rather than faster.
    model.load_state_dict(
        {name: tensor.float() for name, tensor in saved["state"].items()}
    )
    model.eval()
    return model, stored, int(saved["dim"])


# ---------------------------------------------------------------------------
# The retriever interface.


def rank_candidates(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
    pooling: str = retrieval.POOLING,
) -> pd.DataFrame:
    """Score each impression's own candidates. The harness's only entry here.

    The same signature `ann_index` and `bm25_index` expose, so the harness
    scores this retriever without knowing which it holds -- and the two
    arguments that do not apply are refused rather than ignored. `history_k`
    is part of the trained weights, and `pooling` is an aggregator this model
    does not have: its user encoder is learned.
    """
    model, spec, _ = load_model(config)
    if history_k != spec.history_length:
        raise NrmsError(
            f"this checkpoint was fitted at a {spec.history_length}-click "
            f"window and cannot be scored at {history_k}: the window is in the "
            f"weights. Train a checkpoint at {history_k} or score at "
            f"{spec.history_length}."
        )
    if pooling != retrieval.POOLING:
        raise NrmsError(
            f"nrms has no {pooling!r} pooling: its user encoder is learned "
            f"attention over the clicks, not an aggregator over their "
            f"similarities. Sweep pooling on the retrievers that have one."
        )

    embeddings = vectors_for(config, spec)
    wanted = set(behaviors["impression_id"])
    history = history[history["impression_id"].isin(wanted)].reset_index(drop=True)
    rows, mask = history_rows(history, embeddings.index, spec.history_length)

    # Looked up by id rather than zipped: the history frame is filtered from a
    # larger one and need not arrive in the behaviours frame's order, and a
    # positional pairing would score each impression against another's
    # candidates while looking entirely well-formed.
    candidates_of = dict(zip(behaviors["impression_id"], behaviors["candidate_ids"]))
    impression_ids = list(history["impression_id"].astype("string"))
    candidates = [list(candidates_of[impression]) for impression in impression_ids]
    scored = score_all(
        model, embeddings.vectors, rows, mask, candidates, embeddings.index, spec.score_batch
    )
    return retrieval.ranked_frame(impression_ids, candidates, scored)


@dataclass(frozen=True)
class Ranker:
    """Scores a supplied candidate list against a supplied catalogue.

    `rank_candidates` above is the same operation over the feature store; this
    one is handed the corpus, so the submission path can rank against the
    competition's own catalogue -- which the feature store does not contain and
    which the stored embedding artifact barely covers either. `ann_index`
    exposes the same pair of names, and that is all `predict` knows about any
    retriever.
    """

    model: Nrms
    embeddings: embed.Embeddings
    spec: NrmsSpec
    config: DatasetConfig
    history_k: int
    # Where the scoring runs. Carried on the ranker rather than read from a
    # module constant because the submission measures one chunk on each and
    # submits under whichever won -- and a device the weights are not on is a
    # silent copy per batch, which is the thing being measured.
    device: str = DEVICE

    def rank(self, history: pd.DataFrame, candidates: list[list[str]]) -> pd.DataFrame:
        rows, mask = history_rows(history, self.embeddings.index, self.spec.history_length)
        scored = score_all(
            self.model,
            self.embeddings.vectors,
            rows,
            mask,
            candidates,
            self.embeddings.index,
            self.spec.score_batch,
            self.device,
        )
        return retrieval.ranked_frame(
            list(history["impression_id"].astype("string")), candidates, scored
        )


def ranker(
    articles: pd.DataFrame,
    config: DatasetConfig,
    workdir: Path,
    history_k: int = retrieval.HISTORY_K,
    device: str = DEVICE,
) -> Ranker:
    """The checkpoint over vectors for `articles`, corrected as it was trained.

    The correction is the checkpoint's, not the registry's read afresh: a model
    fitted on raw vectors and served corrected ones is being handed a geometry
    it never saw.
    """
    model, spec, _ = load_model(config)
    if history_k != spec.history_length:
        raise NrmsError(
            f"this checkpoint was fitted at a {spec.history_length}-click "
            f"window and cannot serve at {history_k}"
        )
    embeddings, report = embed.for_corpus(articles, config, workdir / embed.EMBED_DIR)
    if spec.corrected:
        embeddings = embed.Embeddings(
            vectors=embed.postprocess(
                embeddings.vectors, config.embeddings.postprocess
            ),
            article_ids=embeddings.article_ids,
        )
    print(
        f"    {report['articles']:,} article vectors, "
        f"{'corrected' if spec.corrected else 'raw'}, for nrms "
        f"{variant_of(spec)}"
    )
    return Ranker(
        model=model.to(device),
        embeddings=embeddings,
        spec=spec,
        config=config,
        history_k=spec.history_length,
        device=device,
    )


# ---------------------------------------------------------------------------
# The grid, the ledger and the stage.


def scoring_cost(
    model: Nrms, data: Data, spec: NrmsSpec, sample: int = 200
) -> dict[str, float]:
    """What a request costs, split into the two halves ticket 09 needs.

    The user encoder runs once per impression and the candidate dot once per
    candidate, so a cache in front of the user side would remove exactly the
    first number and none of the second. Measured one impression at a time,
    which is the serving shape, not the batched one.
    """
    from pipeline import bench

    rows = data.tune_rows[:sample]
    mask = data.tune_mask[:sample]
    candidates = list(data.tune["candidate_ids"][:sample])
    user_seconds, candidate_seconds = [], []
    model.eval()
    with torch.no_grad():
        for position, articles in enumerate(candidates):
            clicked = gather(data.vectors, rows[position : position + 1], mask[position : position + 1])
            keep = torch.from_numpy(mask[position : position + 1])
            started = time.perf_counter()
            user = model.user(model.news(clicked), keep)
            user_seconds.append(time.perf_counter() - started)

            rows_of, mask_of = candidate_rows([articles], data.index, len(articles))
            vectors = gather(data.vectors, rows_of, mask_of)
            started = time.perf_counter()
            torch.einsum("bch,bh->bc", model.news(vectors), user)
            candidate_seconds.append(time.perf_counter() - started)

    whole = [user + candidate for user, candidate in zip(user_seconds, candidate_seconds)]
    return {
        **bench.percentiles(whole),
        "user_p50_ms": bench.percentiles(user_seconds)["p50_ms"],
        "candidate_p50_ms": bench.percentiles(candidate_seconds)["p50_ms"],
        "rows_per_s": sum(len(articles) for articles in candidates) / max(sum(whole), 1e-9),
    }


def record(report: dict, cost: dict, model_bytes: int) -> dict:
    """One ledger row per cell: tune AUC beside what the cell cost to fit and
    what it costs to answer. The history-length axis is the clearest of these
    -- a longer window is more attention work per impression, so `p50_ms` moves
    with the AUC on the same row."""
    return ledger.record(
        {
            "dataset": report["dataset"],
            "stage": STAGE,
            "variant": report["variant"],
            "split": TUNE_SPLIT,
            "auc": report["tune_auc"],
            "model_bytes": model_bytes,
            "train_seconds": round(report["train_seconds"], 2),
            "peak_rss_mb": report.get("peak_rss_mb"),
            "p50_ms": cost["p50_ms"],
            "p99_ms": cost["p99_ms"],
            "rows_per_s": cost["rows_per_s"],
            "note": (
                f"epoch {report['epoch']} of {len(report['curve'])} by early "
                f"stopping, {report['examples']:,} examples over "
                f"{report['fit_impressions']:,} fitted impressions; "
                f"user encoder {cost['user_p50_ms']:.2f} ms, candidate dot "
                f"{cost['candidate_p50_ms']:.2f} ms per impression"
            ),
        }
    )


def fit_one(
    config: DatasetConfig,
    spec: NrmsSpec,
    data: Data | None = None,
    device: str = DEVICE,
) -> dict:
    """Train one cell, save it, measure what it costs to answer, record it.

    The store is read once here and handed to both the training loop and the
    latency measurement -- a second `prepare` would re-read the history table
    and would make the "one read per run" claim false by the time anyone
    checked it.
    """
    data = data or prepare(config, spec)
    with timings.sample() as measured:
        model, report = train(config, spec, data, device)
    report["peak_rss_mb"] = round(measured["peak_rss_mb"], 1)
    path = save(model, spec, data.dim, checkpoint_path(config, spec))
    report["cost"] = scoring_cost(model, data, spec)
    report["model_bytes"] = path.stat().st_size
    report["path"] = str(path)
    record(report, report["cost"], report["model_bytes"])
    return report


def grid(
    config: DatasetConfig, axes: dict | None = None, device: str = DEVICE
) -> list[dict]:
    """The registry's spec, then one cell per alternative on each axis.

    One axis at a time from a stated default rather than a product: the full
    product is 96 trainings to answer six questions the axes are not expected
    to interact on, and a sweep nobody can afford to re-run is not a
    comparison anyone can check.
    """
    axes = GRID if axes is None else axes
    base = config.nrms
    tried: list[dict] = []
    seen: set[str] = set()
    for axis, values in axes.items():
        for value in values:
            spec = dataclasses.replace(base, **{axis: value})
            if variant_of(spec) in seen:
                continue
            seen.add(variant_of(spec))
            tried.append(fit_one(config, spec, prepare(config, spec), device))
    return tried


def document(rows: list[dict], config: DatasetConfig) -> str:
    """The grid as markdown, best first, with the paper's number beside ours."""
    lines = [
        f"# NRMS-DocVec on {config.name}, chosen on `{TUNE_SPLIT}`",
        "",
        "One row per cell. The axes are swept one at a time from the registry's "
        "`NrmsSpec`, so a row differs from the default in exactly one field.",
        "",
        "| variant | tune AUC | epoch | train s | model | p50 ms | user ms | cand ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda row: -row["tune_auc"]):
        lines.append(
            f"| `{row['variant']}` | {row['tune_auc']:.4f} | {row['epoch']} | "
            f"{row['train_seconds']:.0f} | "
            f"{row['model_bytes'] / (1 << 20):.1f} MB | "
            f"{row['cost']['p50_ms']:.2f} | {row['cost']['user_p50_ms']:.2f} | "
            f"{row['cost']['candidate_p50_ms']:.2f} |"
        )
    best = max(rows, key=lambda row: row["tune_auc"]) if rows else None
    lines += ["", "## Against the published baseline", ""]
    if PAPER["auc"] is None:
        lines.append(
            f"**Outstanding:** {PAPER['source']} has not been read into "
            f"`nrms.PAPER` yet, so the gap cannot be stated. It is a number "
            f"from the paper, and typing one from memory is exactly what this "
            f"placeholder exists to prevent."
        )
    elif best is not None:
        gap = best["tune_auc"] - PAPER["auc"]
        lines.append(
            f"{PAPER['source']} reports {PAPER['auc']:.4f}; this reproduction's "
            f"best cell is {best['tune_auc']:.4f} on `{TUNE_SPLIT}`, a gap of "
            f"{gap:+.4f}. Reported either way: a reproduction that lands below "
            f"its source is a finding, not a thing to leave out."
        )
    return "\n".join(lines) + "\n"


def run(config: DatasetConfig, force: bool = False) -> None:
    """The build stage: the registry's cell, trained and saved."""
    path = checkpoint_path(config)
    if not force and path.exists():
        print(f"    nrms {variant_of(config.nrms)} is already trained at {path}")
        return
    report = fit_one(config, config.nrms)
    print(
        f"    {report['variant']}: tune AUC {report['tune_auc']:.4f} at epoch "
        f"{report['epoch']}, {report['train_seconds']:.0f} s, "
        f"{report['model_bytes'] / (1 << 20):.1f} MB"
    )
    ledger.render()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.nrms",
        description="Train NRMS-DocVec, or sweep the grid that chooses its shape.",
    )
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument(
        "--grid", action="store_true", help="train every cell of GRID on tune"
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help=f"torch device to fit on (default: {DEVICE}); the cluster job "
        f"passes cuda",
    )
    args = parser.parse_args(argv)

    for name in args.dataset or DEFAULT_DATASETS:
        config = DATASETS[name]
        reports = (
            grid(config, device=args.device)
            if args.grid
            else [fit_one(config, config.nrms, device=args.device)]
        )
        for report in reports:
            print(
                f"  {name}/{report['variant']}: tune AUC {report['tune_auc']:.4f} "
                f"at epoch {report['epoch']}, {report['train_seconds']:.0f} s"
            )
        if args.grid:
            path = paths.ARTIFACTS_DIR / DOCUMENT.format(
                dataset=name, split=TUNE_SPLIT
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(document(reports, config), encoding="utf-8")
            print(f"  -> {path}")
    print(f"-> {ledger.render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
