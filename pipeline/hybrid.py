"""One ranking from two: the lexical and semantic scores combined.

    rrf     score = sum over parents of 1 / (k + rank)
    linear  score = alpha * bm25 + (1 - alpha) * ann, each min-max normalised
            within the impression first

Both are content-only and both read scores the pipeline already computes, so
neither disturbs the assignment's position on behavioural features. This is not
the `fusion` module deleted at 317a83c — that one ranked on behavioural
popularity counters, which are out of scope. Combining two content rankings is
an answer to "lexical or semantic, which works better?", not a departure from
it.

**Why RRF is the default.** BM25 scores are unbounded sums of term weights and
cosines live in [-1, 1]; the two are not on a common scale and nothing in
either makes them comparable. RRF never has to reconcile that, because it
throws the scores away and keeps only the order. `linear` is here because
`alpha` is interpretable in a way `k` is not — a per-dataset alpha says which
retriever a dataset leans on — and it pays for that with a calibration step
that can go wrong.

**A parent with no opinion is skipped, not averaged in.** An empty BM25 query
or a user with no usable clicks scores every candidate identically. That is not
a ranking, it is the candidate file's own order, and RRF cannot tell the
difference — it would read a meaningless permutation as evidence. So a parent
whose scores are flat for an impression contributes nothing to it and the other
parent decides alone. Both flat leaves the order the competition gave.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import ann_index, bm25_index, retrieval
from pipeline.datasets import DatasetConfig, HybridSpec

RULES = ("rrf", "linear")

# The parents, in the order their scores are reported. `alpha` weights the
# first of them, so the pair and the parameter cannot drift apart.
PARENTS = ("bm25", "ann")


class HybridError(RuntimeError):
    """The two parents cannot be combined as asked."""


def _opinion(scores: np.ndarray) -> bool:
    """Whether this parent ranked the impression or merely returned it.

    A flat score vector is what an empty query produces, and downstream it is
    indistinguishable from a real ranking that happens to tie — except that it
    carries no information, so folding it in adds noise with the authority of a
    retriever behind it.
    """
    return len(scores) > 0 and float(np.ptp(scores)) > 0


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max within the impression, or zeros where there is nothing to scale.

    Within the impression rather than across the split: BM25's scale moves with
    query length, so a corpus-wide normalisation would make an impression with
    a long query outrank one with a short query on every candidate.

    Zeros for a flat parent is what makes `linear` degrade correctly — a
    constant contributes nothing to the ordering, so the other parent decides.
    """
    if not _opinion(scores):
        return np.zeros(len(scores), dtype="float64")
    low, high = float(scores.min()), float(scores.max())
    return (scores - low) / (high - low)


def fuse_one(
    ranked: dict[str, tuple[list[str], np.ndarray]], spec: HybridSpec
) -> tuple[list[str], list[float]]:
    """One impression's candidates, combined into one order.

    `ranked` maps each parent to the ids it returned and the scores beside
    them, in that parent's own ranked order. Every parent covers the same
    candidate set here — `rank_candidates` drops nothing — so the fused score is
    defined for every candidate without asking what a missing one means.
    """
    speaking = {name: pair for name, pair in ranked.items() if _opinion(pair[1])}
    ids = next(iter(ranked.values()))[0]

    if not speaking:
        # Neither parent ranked it. The order it arrived in is the honest
        # answer, and the scores say so.
        return list(ids), [0.0] * len(ids)

    found: dict[str, float] = {article: 0.0 for article in ids}

    if spec.rule == "rrf":
        for _, (order, _) in speaking.items():
            for position, article in enumerate(order):
                found[article] += 1.0 / (spec.k + position + 1)
    elif spec.rule == "linear":
        weight = {PARENTS[0]: spec.alpha, PARENTS[1]: 1.0 - spec.alpha}
        # Re-weighted over the parents that spoke, so an impression only one
        # parent ranked is that parent's ranking rather than a shrunken
        # version of it.
        total = sum(weight[name] for name in speaking) or 1.0
        for name, (order, scores) in speaking.items():
            for article, value in zip(order, _normalise(np.asarray(scores))):
                found[article] += weight[name] * value / total
    else:
        raise HybridError(
            f"unknown fusion rule {spec.rule!r}, expected one of {', '.join(RULES)}"
        )

    values = np.array([found[article] for article in ids], dtype="float64")
    order = np.argsort(-values, kind="stable")
    return [ids[position] for position in order], [
        float(values[position]) for position in order
    ]


def fuse(parents: dict[str, pd.DataFrame], spec: HybridSpec) -> pd.DataFrame:
    """Both parents' rankings, impression by impression, into one frame.

    Aligned on `impression_id` rather than by position: the two parents filter
    the same history frame and need not emit their rows in the same order, and
    a positional pairing would fuse one impression's lexical ranking with
    another's semantic one while looking entirely well-formed.
    """
    if spec.rule not in RULES:
        raise HybridError(
            f"unknown fusion rule {spec.rule!r}, expected one of {', '.join(RULES)}"
        )

    frames = {name: frame.set_index("impression_id") for name, frame in parents.items()}
    first = frames[PARENTS[0]]
    # Compared as sets, not as sequences: the parents filter the same history
    # frame and need not emit their rows in the same order, and every lookup
    # below is by id rather than by position.
    wanted = set(first.index)
    for name, frame in frames.items():
        if set(frame.index) != wanted:
            raise HybridError(
                f"{name} ranked {len(frame)} impressions and "
                f"{PARENTS[0]} ranked {len(first)}; they must be the same set"
            )

    ranked_ids: list[list[str]] = []
    scores: list[list[float]] = []
    for impression in first.index:
        order, value = fuse_one(
            {
                name: (
                    list(frame.at[impression, "ranked_ids"]),
                    np.asarray(frame.at[impression, "scores"], dtype="float64"),
                )
                for name, frame in frames.items()
            },
            spec,
        )
        ranked_ids.append(order)
        scores.append(value)

    return pd.DataFrame(
        {
            "impression_id": list(first.index),
            "ranked_ids": ranked_ids,
            "scores": scores,
        }
    )


def rank_candidates(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
    pooling: str = retrieval.POOLING,
) -> pd.DataFrame:
    """Score each impression's own candidates. The harness's only entry here.

    The same signature bm25_index and ann_index expose, which is the whole of
    what the harness knows about any retriever — so this one is an entry in
    `evaluate.RETRIEVERS` and nothing downstream learns it exists.
    """
    return fuse(
        {
            name: module.rank_candidates(
                config, behaviors, history, history_k, pooling
            )
            for name, module in (
                (PARENTS[0], bm25_index),
                (PARENTS[1], ann_index),
            )
        },
        config.hybrid,
    )


def retrieve_corpus(
    config: DatasetConfig,
    behaviors: pd.DataFrame,
    history: pd.DataFrame,
    history_k: int = retrieval.HISTORY_K,
    depth: int = max(retrieval.DEPTHS),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rank the catalogue by fusing what the two parents retrieved from it.

    Unlike the candidate path, the parents here return *different* article
    sets — each its own top-`depth` — so the union is what gets fused and an
    article only one parent found is ranked on that parent alone. That is the
    ordinary reading of RRF over two result lists, and it is why the pooled
    depth is at least `depth` and usually more.

    Each parent is asked for `depth` so that the fused list can be truncated
    back to `depth` without either parent having been cut short first.
    """
    retrieved = {}
    report: dict[str, int] = {}
    for name, module in ((PARENTS[0], bm25_index), (PARENTS[1], ann_index)):
        ranked, asked = module.retrieve_corpus(
            config, behaviors, history, history_k, depth
        )
        retrieved[name] = ranked.set_index("impression_id")
        # Cold means cold for both parents: an impression neither could query.
        for key, value in asked.items():
            report[key] = min(report[key], value) if key in report else value

    first = retrieved[PARENTS[0]]
    ranked_ids: list[list[str]] = []
    scores: list[list[float]] = []
    for impression in first.index:
        pooled: dict[str, float] = {}
        for name, frame in retrieved.items():
            order = list(frame.at[impression, "ranked_ids"])
            found = np.asarray(frame.at[impression, "scores"], dtype="float64")
            if not _opinion(found):
                continue
            for position, article in enumerate(order):
                pooled[article] = pooled.get(article, 0.0) + 1.0 / (
                    config.hybrid.k + position + 1
                )
        best = sorted(pooled.items(), key=lambda pair: -pair[1])[:depth]
        ranked_ids.append([article for article, _ in best])
        scores.append([value for _, value in best])

    return (
        pd.DataFrame(
            {
                "impression_id": list(first.index),
                "ranked_ids": ranked_ids,
                "scores": scores,
            }
        ),
        report,
    )
