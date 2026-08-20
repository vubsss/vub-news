"""Metrics over the shared ranked-candidates shape, sliced and with intervals.

The harness knows nothing about BM25 or embeddings. It asks a retriever module
for `rank_candidates` and scores whatever comes back, so a third retriever
needs no change here — only an entry in RETRIEVERS.

    python -m pipeline.evaluate                                  every combination
    python -m pipeline.evaluate --dataset mind --retriever bm25  one of them
    python -m pipeline.evaluate --split test                     the held-back split
    python -m pipeline.evaluate --resamples 100                  a fast development run

Every run prints the same columns in the same order and writes one json per
scored run, so tickets 11 and 12 aggregate from disk rather than re-ranking.

Two families of metric are reported. AUC, MRR and nDCG say whether the
retriever ordered an impression's candidates well; diversity, novelty and
coverage say what the resulting lists are made of, which the assignment asks
for because a recommender can order perfectly while showing everyone the same
handful of popular articles. Both families are reported overall and on four
population slices, and every number carries a bootstrap interval — the point of
which is that a difference smaller than its interval is not a difference.

Metrics are computed over an impression's own candidate list, not over the
corpus: recall@K in tickets 6 and 8 asks whether a global retriever surfaces a
clicked article at all, while these ask whether it orders the candidates the
competition supplies. The two answer different questions and neither replaces
the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline import ann_index, bm25_index, paths, retrieval
from pipeline.datasets import DATASETS, DatasetConfig

# The retrievers the harness can score. The values are modules, each exposing
# rank_candidates(config, behaviors, history) -> ranked candidates. The harness
# reaches a retriever only through that call, so a third one is an entry here
# and nothing else.
RETRIEVERS = {
    "bm25": bm25_index,
    "ann": ann_index,
}

# nDCG cut-offs, per SPEC.
NDCG_DEPTHS = (5, 10)

# Metrics are reported on these; train is refused outright.
VALIDATION, TEST = "validation", "test"
SCORABLE = (VALIDATION, TEST)
TRAIN = "train"

# How far down a ranking the beyond-accuracy metrics look. They describe the
# list a user would be shown, so they have to stop where that list stops: the
# diversity of all 295 candidates of MIND's longest impression is a fact about
# the candidate generator, not about the ranking. Ten, to match nDCG@10.
LIST_DEPTH = 10

# Per-impression metrics, averaged over a slice.
ACCURACY_METRICS = ("auc", "mrr", *(f"ndcg@{depth}" for depth in NDCG_DEPTHS))
LIST_METRICS = ("diversity", "novelty")
MEAN_METRICS = (*ACCURACY_METRICS, *LIST_METRICS)

# Coverage is a property of a set of impressions rather than of any one of
# them — the union of what they showed — so it is aggregated differently and
# sits last, after everything that is a mean.
METRICS = (*MEAN_METRICS, "coverage")

# Where each per-impression metric sits in the values matrix.
COLUMN = {name: i for i, name in enumerate(MEAN_METRICS)}

# Slice thresholds, fixed by the spec.
COLD_CLICKS = 5  # fewer clicks in history than this and a user is cold
HEAD_FRACTION = 0.2  # the most-shown fifth of the articles a split displays

# Every slice is a predicate over the per-impression population frame, so a new
# one is a line here and nothing else: the metrics, the aggregation, the
# bootstrap and the table all iterate over this mapping.
#
# Head and tail are article properties, and an impression is placed by the
# articles its user actually clicked: head when every click landed on a
# most-shown article, tail when none did. An impression whose clicks straddle
# both belongs to neither, which is why the two slices do not sum to the whole
# and why every row carries its own population.
SLICES = {
    "overall": lambda pop: np.ones(len(pop), dtype=bool),
    "cold": lambda pop: pop["n_clicks"] < COLD_CLICKS,
    "warm": lambda pop: pop["n_clicks"] >= COLD_CLICKS,
    "head": lambda pop: pop["head_share"] == 1.0,
    "tail": lambda pop: pop["head_share"] == 0.0,
}

# Bootstrap: resamples impressions with replacement and reads the interval off
# the percentiles of the resulting distribution. Impressions are the axis
# because they are what the sample is of — resampling candidates within an
# impression would measure the candidate list's shape instead.
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE = 0.95
# Fixed, so two runs over the same rankings print the same interval and a
# change in the fourth decimal is a change in the data rather than in the draw.
BOOTSTRAP_SEED = 0

# One row per slice per metric. Long rather than wide because the wide form
# would need three columns per metric; this way the columns never change, the
# table sorts and greps, and ticket 11 reads it without a parser.
TEXT_COLUMNS = ("dataset", "retriever", "split", "slice", "metric")
FLOAT_COLUMNS = ("value", "lo", "hi")
COLUMNS = (
    "dataset",
    "retriever",
    "split",
    "slice",
    "population",
    "n",
    "metric",
    *FLOAT_COLUMNS,
)

# Where a scored run is written for tickets 11 and 12 to aggregate.
EVALUATE_DIR = "evaluate"


class EvaluationError(RuntimeError):
    """The harness was asked for something it must not answer."""


def reciprocal_rank(relevance: np.ndarray) -> float:
    """1 / the rank of the first relevant candidate, ranks counted from 1.

    The textbook definition: the first hit, not an average over every hit. It
    is worth being explicit because 29.6% of MIND's validation impressions have
    more than one positive, so a mean-over-all-positives MRR — which some
    leaderboard evaluators use — would give a visibly different number on the
    same predictions.
    """
    hits = np.flatnonzero(relevance)
    return 1.0 / (hits[0] + 1)


def dcg(relevance: np.ndarray, depth: int) -> float:
    """Binary-gain DCG: each relevant candidate contributes 1/log2(rank+1)."""
    top = relevance[:depth]
    return float((top / np.log2(np.arange(2, len(top) + 2))).sum())


def ndcg(relevance: np.ndarray, depth: int) -> float:
    """DCG against the best ordering the same labels could possibly achieve."""
    ideal = dcg(np.sort(relevance)[::-1], depth)
    return dcg(relevance, depth) / ideal if ideal else 0.0


def intra_list_diversity(categories: np.ndarray) -> float:
    """Share of candidate pairs in the list drawn from different categories.

    The spec's mean pairwise category dissimilarity, with dissimilarity 1 for
    a different category and 0 for the same one. Counted from the category
    sizes rather than by walking the pairs: with c of one category among m
    items there are c(c-1)/2 same-category pairs, so the whole sum is one pass.

    Undefined for a list of one, which has no pairs.
    """
    if len(categories) < 2:
        return np.nan
    _, sizes = np.unique(categories, return_counts=True)
    same = (sizes * (sizes - 1)).sum()
    return 1.0 - same / (len(categories) * (len(categories) - 1))


def relevance_of(ranked_ids: list[str], labels: dict[str, int]) -> np.ndarray:
    """The label of each candidate, in the order the retriever ranked them.

    The ranking has to be a permutation of the candidate list: the leaderboard
    scores every candidate, and a retriever that quietly dropped some would
    score better here than it deserves — the dropped ones can only be misses.
    """
    if len(ranked_ids) != len(labels) or set(ranked_ids) != set(labels):
        raise EvaluationError(
            f"ranking is not a permutation of the candidates: "
            f"{len(ranked_ids)} ranked against {len(labels)} candidates"
        )
    return np.array([labels[article_id] for article_id in ranked_ids])


@dataclass(frozen=True)
class Catalogue:
    """The article facts the beyond-accuracy metrics and slices are read off.

    Popularity is measured on the split being scored, not on train: these
    numbers describe the population the report is about, and an article that
    never appears in the split cannot be in any of its lists. Nothing here
    reaches a retriever, so measuring it on the evaluated split is description
    rather than leakage.
    """

    size: int  # articles in the catalogue
    pool: int  # of those, the ones this split ever shows as a candidate
    index_of: dict[str, int]
    category: np.ndarray  # category code per catalogue position
    novelty: np.ndarray  # self-information per catalogue position
    is_head: np.ndarray  # whether each position is a most-shown article
    head_cutoff: int  # impressions an article needs to count as head


def catalogue_of(articles: pd.DataFrame, impressions: pd.DataFrame) -> Catalogue:
    """Category, popularity and head/tail for every article in the catalogue."""
    article_ids = articles["article_id"].tolist()
    index_of = {article_id: i for i, article_id in enumerate(article_ids)}
    category = pd.factorize(articles["category"])[0]

    shown = np.concatenate(impressions["candidate_ids"].to_list())
    clicked = np.concatenate(impressions["labels"].to_list())
    position = pd.Index(article_ids).get_indexer(shown)
    if (position < 0).any():
        stray = sorted(set(shown[position < 0]))
        raise EvaluationError(
            f"{len(stray)} candidate article(s) are not in the catalogue, "
            f"e.g. {stray[:5]}"
        )

    impression_count = np.bincount(position, minlength=len(article_ids))
    click_count = np.bincount(position, weights=clicked, minlength=len(article_ids))

    # Ranked over the articles the split shows rather than over the catalogue.
    # MIND's validation split displays 6144 of its 65238 articles, so a fifth
    # of the catalogue would be mostly articles that appeared nowhere — a head
    # slice defined by articles no impression contains describes nothing.
    pool = [i for i in range(len(article_ids)) if impression_count[i]]
    ranking = sorted(pool, key=lambda i: (-impression_count[i], article_ids[i]))
    head_size = max(1, round(len(pool) * HEAD_FRACTION)) if pool else 0
    is_head = np.zeros(len(article_ids), dtype=bool)
    is_head[ranking[:head_size]] = True

    # Self-information, add-one smoothed over the pool: an article nobody
    # clicked is the most novel thing the system can show, not an infinity.
    total = click_count[pool].sum() + len(pool) if pool else 1.0
    novelty = -np.log2((click_count + 1) / total)

    return Catalogue(
        size=len(article_ids),
        pool=len(pool),
        index_of=index_of,
        category=category,
        novelty=novelty,
        is_head=is_head,
        head_cutoff=int(impression_count[ranking[head_size - 1]]) if pool else 0,
    )


def paired(
    ranked: pd.DataFrame, behaviors: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """One row per impression: its truth, its ranking and its user's history.

    Every downstream array is positional in this frame, so joining once here
    is what makes them line up. A retriever that returned fewer impressions
    than it was given is an error rather than a shorter table: the missing ones
    would silently leave the averages, and the ones that leave are exactly the
    hard ones — the users with no history to build a query from.
    """
    frame = (
        behaviors[["impression_id", "candidate_ids", "labels"]]
        .merge(ranked[["impression_id", "ranked_ids", "scores"]], on="impression_id")
        .merge(history[["impression_id", "n_clicks"]], on="impression_id")
    )
    if len(frame) != len(behaviors):
        raise EvaluationError(
            f"{len(behaviors)} impressions were ranked but {len(frame)} came "
            f"back paired with a ranking and a history"
        )
    return frame


def measure(
    frame: pd.DataFrame, catalogue: Catalogue
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, int]]:
    """Per-impression metric values, shown lists, slice keys and the leftovers.

    Returns a (impressions x MEAN_METRICS) matrix carrying NaN wherever a
    metric is undefined for that impression, the top-of-list article positions
    coverage is the union of, the columns the slice predicates read, and the
    counts of impressions no ranking metric is defined on.

    AUC is computed from the scores and MRR and nDCG from the order, which is
    the standard split: AUC is defined on score values and handles ties by
    construction, while a rank metric has to break them somehow. Ties are left
    in the order the retriever emitted, which for a stable sort is the order
    the candidates arrived in — never reordered to put a positive first.

    Diversity and novelty are recorded for every impression, including the ones
    with no positive label: they describe the list that was shown, which exists
    whether or not the user clicked anything in it.
    """
    values = np.full((len(frame), len(MEAN_METRICS)), np.nan)
    # The padding slot is one past the catalogue, so an unfilled cell scatters
    # somewhere harmless and coverage needs no mask.
    shown = np.full((len(frame), LIST_DEPTH), catalogue.size, dtype=np.int64)
    head_share = np.full(len(frame), np.nan)
    degenerate = {"no_positive": 0, "all_positive": 0, "all_scores_tied": 0}

    columns = ["candidate_ids", "labels", "ranked_ids", "scores"]
    for row, (candidates, labels, ranked_ids, scores) in enumerate(
        frame[columns].itertuples(index=False)
    ):
        label_of = dict(zip(candidates, labels, strict=True))
        relevance = relevance_of(ranked_ids, label_of)

        top = np.array(
            [catalogue.index_of[a] for a in ranked_ids[:LIST_DEPTH]], dtype=np.int64
        )
        shown[row, : len(top)] = top
        values[row, COLUMN["diversity"]] = intra_list_diversity(catalogue.category[top])
        values[row, COLUMN["novelty"]] = catalogue.novelty[top].mean()

        clicked = [a for a, label in zip(candidates, labels) if label]
        if clicked:
            head_share[row] = np.mean(
                [catalogue.is_head[catalogue.index_of[a]] for a in clicked]
            )

        positives = int(relevance.sum())
        if positives == 0:
            # AUC, MRR and nDCG are all undefined; averaging a zero in would
            # report the share of such impressions rather than any ranking.
            degenerate["no_positive"] += 1
            continue
        if positives == len(relevance):
            degenerate["all_positive"] += 1
            continue
        if len(set(scores)) == 1:
            # Every candidate scored the same, so the order is the one the
            # candidates arrived in. The rank metrics still get a number, but
            # it is a property of the competition's file, not the retriever.
            degenerate["all_scores_tied"] += 1

        values[row, COLUMN["auc"]] = roc_auc_score(relevance, scores)
        values[row, COLUMN["mrr"]] = reciprocal_rank(relevance)
        for depth in NDCG_DEPTHS:
            values[row, COLUMN[f"ndcg@{depth}"]] = ndcg(relevance, depth)

    population = pd.DataFrame(
        {"n_clicks": frame["n_clicks"].to_numpy(), "head_share": head_share}
    )
    return values, shown, population, degenerate


def summarise(
    rows: np.ndarray, values: np.ndarray, shown: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every metric over a set of impression rows, and how many fed each one.

    The means skip the impressions a metric is undefined on rather than
    counting them as zero, and report how many were left — a slice of eleven
    users has to be visibly a slice of eleven users. Coverage is the union of
    the lists over the whole slice, divided by the full catalogue and not by
    the articles that happened to be offered as candidates.
    """
    taken = values[rows]
    defined = ~np.isnan(taken)
    counts = defined.sum(axis=0)
    totals = np.where(defined, taken, 0.0).sum(axis=0)
    means = np.divide(
        totals, counts, out=np.full(len(MEAN_METRICS), np.nan), where=counts > 0
    )

    if len(rows):
        seen = np.zeros(size + 1, dtype=bool)
        seen[shown[rows].ravel()] = True
        coverage = float(seen[:size].sum()) / size
    else:
        coverage = np.nan

    return np.append(means, coverage), np.append(counts, len(rows))


def interval(
    rows: np.ndarray,
    values: np.ndarray,
    shown: np.ndarray,
    size: int,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap percentile interval per metric, over resampled impressions.

    Impressions are the axis that is resampled. That is the whole content of
    the method here: the sample the report generalises from is a sample of
    impressions, so a metric that is stable across impressions gets a narrow
    interval and one carried by a handful of them gets a wide one. Resampling
    anything else — candidates within an impression, say — would produce
    intervals that look plausible and mean nothing.

    Coverage is the one metric this does not bracket. It is a union over the
    slice, and a resample repeats about 37% of its draws, so every resample
    sees fewer distinct articles than the slice itself did and the interval
    lands below the point estimate. That is the naive bootstrap's known
    behaviour on distinct-count statistics rather than a fault here; the width
    is still the spread of the statistic, and `notes` says so in the output so
    the row is not read as a bracket.
    """
    empty = np.full(len(METRICS), np.nan)
    if not len(rows) or resamples < 1:
        return empty, empty.copy()

    draws = np.empty((resamples, len(METRICS)))
    for i in range(resamples):
        draw = rows[rng.integers(0, len(rows), len(rows))]
        draws[i] = summarise(draw, values, shown, size)[0]

    tail = (1 - CONFIDENCE) / 2 * 100
    lo, hi = np.full(len(METRICS), np.nan), np.full(len(METRICS), np.nan)
    for metric in range(len(METRICS)):
        column = draws[:, metric]
        column = column[~np.isnan(column)]
        if column.size:
            lo[metric], hi[metric] = np.percentile(column, [tail, 100 - tail])
    return lo, hi


def reported(value: float) -> float | None:
    """A metric value, or None where it is undefined.

    None rather than NaN because these rows are written as json and read back
    by tickets 11 and 12: `NaN` is not valid json, and a null a consumer has to
    handle is safer here than a NaN that would propagate silently through a
    comparison and come out looking like a finding.
    """
    return None if np.isnan(value) else float(value)


def results(
    values: np.ndarray,
    shown: np.ndarray,
    population: pd.DataFrame,
    catalogue: Catalogue,
    resamples: int,
) -> list[dict]:
    """Every slice x every metric, each with its population and its interval."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict] = []
    for name, belongs in SLICES.items():
        members = np.flatnonzero(np.asarray(belongs(population), dtype=bool))
        point, counts = summarise(members, values, shown, catalogue.size)
        lo, hi = interval(members, values, shown, catalogue.size, resamples, rng)
        rows.extend(
            {
                "slice": name,
                "population": len(members),
                "n": int(counts[i]),
                "metric": metric,
                "value": reported(point[i]),
                "lo": reported(lo[i]),
                "hi": reported(hi[i]),
            }
            for i, metric in enumerate(METRICS)
        )
    return rows


def flatten(reports: list[dict]) -> list[dict]:
    """Reports as one row per slice per metric, ready for the table."""
    return [
        {**{c: report[c] for c in ("dataset", "retriever", "split")}, **result}
        for report in reports
        for result in report["results"]
    ]


def table(reports: list[dict]) -> str:
    """The flattened reports as one fixed-column table, floats to four places.

    Columns are padded to the widest cell rather than to a fixed width, so a
    table stays readable whatever the impression counts are, while the columns
    themselves and their order never change. A metric undefined on a slice —
    AUC where no impression admits it, anything at all on an empty slice —
    prints as a dash rather than as a number nobody should read.
    """
    def cell(row: dict, column: str) -> str:
        if column not in FLOAT_COLUMNS:
            return str(row[column])
        return "-" if row[column] is None else f"{row[column]:.4f}"

    rows = [[cell(row, c) for c in COLUMNS] for row in flatten(reports)]
    widths = [
        max(len(name), *(len(row[i]) for row in rows))
        for i, name in enumerate(COLUMNS)
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) if name in TEXT_COLUMNS else cell.rjust(width)
            for name, cell, width in zip(COLUMNS, cells, widths)
        ).rstrip()

    return "\n".join([line(list(COLUMNS)), *(line(row) for row in rows)])


def notes(report: dict) -> str:
    """What the numbers of one run have to be read against.

    The degenerate counts, the two facts a coverage figure is meaningless
    without — what it is a fraction of, and how much of that the split could
    possibly have reached — and the one place the bootstrap does not behave.
    """
    reachable = 100 * report["candidate_pool"] / report["catalogue"]
    return (
        f"  {report['dataset']}/{report['retriever']}/{report['split']}: "
        f"{report['impressions']} impressions, "
        f"{report['no_positive']} with no positive candidate, "
        f"{report['all_positive']} with every candidate positive, "
        f"{report['all_scores_tied']} scored flat\n"
        f"    coverage is against the full catalogue of {report['catalogue']} "
        f"articles, of which {report['candidate_pool']} ({reachable:.1f}%) ever "
        f"appear as a candidate on this split — no ranking of the offered "
        f"candidates can exceed that ceiling\n"
        f"    head = the most-shown fifth of those {report['candidate_pool']} "
        f"(at least {report['head_cutoff']} impressions); cold = fewer than "
        f"{COLD_CLICKS} clicks in history; lists are read to depth "
        f"{report['list_depth']}\n"
        f"    lo/hi are bootstrap {report['confidence']:.0%} percentiles over "
        f"{report['resamples']} resamples of impressions\n"
        f"    coverage's interval sits below its own value and is not a "
        f"bracket around it: coverage is a union over the slice rather than a "
        f"mean over it, and a resample repeats about 37% of its impressions, "
        f"so it always shows fewer distinct articles. Read that row's lo/hi as "
        f"a width, and compare coverage between retrievers by value"
    )


def evaluate(
    config: DatasetConfig,
    retriever: str,
    split: str = VALIDATION,
    resamples: int = BOOTSTRAP_RESAMPLES,
    history_k: int = retrieval.HISTORY_K,
) -> dict:
    """Score one retriever on one dataset and split, sliced and with intervals.

    history_k is the click window the retriever builds its query from. It is
    recorded in the report because two retrievers given different windows are
    not comparable, and a report that did not carry the window would let that
    comparison be made without anything noticing — ticket 12 sweeps it.
    """
    if split == TRAIN:
        raise EvaluationError(
            f"refusing to score the {TRAIN} split. Metrics on data the "
            f"retriever was tuned against measure memorisation, and reporting "
            f"them is the self-deception the assignment's anti-gaming section "
            f"is about. Use one of: {', '.join(SCORABLE)}."
        )
    if split not in SCORABLE:
        raise EvaluationError(
            f"unknown split {split!r}; use one of: {', '.join(SCORABLE)}"
        )
    if retriever not in RETRIEVERS:
        raise EvaluationError(
            f"unknown retriever {retriever!r}; use one of: "
            f"{', '.join(RETRIEVERS)}"
        )

    store = config.feature_store_dir
    articles = pd.read_parquet(store / "articles.parquet")
    behaviors = pd.read_parquet(store / "behaviors.parquet")
    history = pd.read_parquet(store / "history.parquet")
    impressions = behaviors[behaviors["split"] == split]

    ranked = RETRIEVERS[retriever].rank_candidates(
        config, impressions, history, history_k
    )
    catalogue = catalogue_of(articles, impressions)
    values, shown, population, degenerate = measure(
        paired(ranked, impressions, history), catalogue
    )

    return {
        "dataset": config.name,
        "retriever": retriever,
        "split": split,
        "history_k": history_k,
        "impressions": len(impressions),
        **degenerate,
        "catalogue": catalogue.size,
        "candidate_pool": catalogue.pool,
        "coverage_of": "catalogue",
        "head_cutoff": catalogue.head_cutoff,
        "list_depth": LIST_DEPTH,
        "confidence": CONFIDENCE,
        "resamples": resamples,
        "results": results(values, shown, population, catalogue, resamples),
    }


def save(report: dict, config: DatasetConfig) -> Path:
    """One json per scored run, for tickets 11 and 12 to aggregate."""
    directory = config.artifacts_dir / EVALUATE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report['retriever']}-{report['split']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def report_on(paths_written: list[Path], reports: list[dict]) -> None:
    """The table, what it has to be read against, then where it was written."""
    print("\n" + table(reports) + "\n")
    for report in reports:
        print(notes(report))
    print()
    for path in paths_written:
        print(f"  -> {path.relative_to(paths.ARTIFACTS_DIR.parent)}")
    print()


def run(config: DatasetConfig, force: bool = False) -> None:
    """The build stage: every retriever on the validation split.

    Test is not scored here. It is scored once, deliberately, by the command
    below — a split that is rebuilt into every `python build.py` is one that
    gets looked at repeatedly, which is how a held-back split stops being one.
    """
    reports = [evaluate(config, retriever, VALIDATION) for retriever in RETRIEVERS]
    report_on([save(report, config) for report in reports], reports)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.evaluate",
        description="Score a retriever on a dataset and split.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="restrict to one dataset (repeatable); default is all of them",
    )
    parser.add_argument(
        "--retriever",
        action="append",
        choices=sorted(RETRIEVERS),
        help="restrict to one retriever (repeatable); default is all of them",
    )
    # Deliberately not an argparse choice: `--split train` has to reach
    # evaluate() and be turned down with its reason, not with "invalid choice".
    parser.add_argument(
        "--split",
        default=VALIDATION,
        help=f"which split to score: {' or '.join(SCORABLE)} "
        f"(default: {VALIDATION})",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
        help="bootstrap resamples behind every interval "
        f"(default: {BOOTSTRAP_RESAMPLES}); lower it for a fast run while "
        "developing, but not for a number anyone will quote",
    )
    args = parser.parse_args(argv)

    configs = [DATASETS[name] for name in (args.dataset or sorted(DATASETS))]
    retrievers = args.retriever or sorted(RETRIEVERS)

    reports: list[dict] = []
    written: list[Path] = []
    try:
        for config in configs:
            for retriever in retrievers:
                report = evaluate(config, retriever, args.split, args.resamples)
                reports.append(report)
                written.append(save(report, config))
    except EvaluationError as error:
        print(f"\nerror: {error}\n", file=sys.stderr)
        return 2

    report_on(written, reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
