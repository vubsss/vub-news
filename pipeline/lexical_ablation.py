"""What the lexical sweep's grid cannot say: recall, and the query side.

`lexical_sweep` scores 75 cells on the re-ranking path only, because that is
the one that is cheap enough to run 75 times. Two questions are left over, and
both need the retrieval path as well:

1. Field weighting can move retrieval and re-ranking in different directions.
   Repeating the title lengthens the document, and the corpus search is where
   that shows up — a term the query shares with a headline now competes against
   65,000 other documents rather than against fifteen candidates.
2. The query is built from clicked *titles*, with a comment arguing that
   abstracts would drown the terms that identify what a user reads. That is an
   argument, not a measurement.

So this runs a handful of named settings rather than a grid, and reports both
paths for each: recall@K over the corpus, and AUC, MRR and nDCG over the
impression's own candidates. On `tune`, like every other choice.

Differences are bootstrapped **paired by impression**, which the grid's cells
are not. Two settings scoring the same impressions share all of the variance
that comes from the impressions themselves -- an impression with two clicked
candidates out of forty is hard under both -- and on this data that variance is
most of the width of a per-setting interval. Comparing settings by whether
those intervals overlap would find nothing separable at any effect size the
lexical side can produce.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline import bm25_index, embed_compare, evaluate, paths, preprocess, retrieval
from pipeline.datasets import DATASETS, DatasetConfig, LexicalSpec

# What the parameters were before phase 3 measured them: SPEC.md's k1 and b,
# and title and abstract concatenated into one bag.
INHERITED = LexicalSpec(k1=1.5, b=0.75, title_weight=1)

RESULTS = "lexical-ablation-{suffix}.jsonl"
DOCUMENT = "lexical-ablation-{suffix}.md"


@dataclass(frozen=True)
class Setting:
    """One row of the table: an index configuration, a query construction, and
    the row this one is a difference *from*.

    `against` is what makes the row an ablation rather than a number: each
    setting names the one it differs from in exactly one respect, and the
    document reports that paired difference.
    """

    name: str
    lexical: LexicalSpec
    with_abstract: bool
    against: str | None


def settings(config: DatasetConfig) -> list[Setting]:
    """The inherited setting, the tuned one, and the tuned one asked a
    different question. Each row differs from the one it names in one thing:
    tuned against inherited isolates k1, b and the title weight, and the last
    row against tuned isolates the query."""
    rows = [Setting("inherited", INHERITED, with_abstract=False, against=None)]
    if config.lexical != INHERITED:
        rows.append(
            Setting("tuned", config.lexical, with_abstract=False, against="inherited")
        )
    # Always the abstract query last, whichever one the registry ended up
    # choosing: the chain is inherited -> tuned parameters -> tuned parameters
    # asked a different question, and each link has to move one thing. Which
    # row is the pipeline as it stands is marked in the table instead.
    rows.append(
        Setting(
            "tuned + abstract query",
            config.lexical,
            with_abstract=True,
            against=rows[-1].name,
        )
    )
    return rows


def run(config: DatasetConfig, split: str, resamples: int) -> list[dict]:
    articles = pd.read_parquet(config.feature_store_dir / "articles.parquet")
    scored = embed_compare.impressions(config, split)
    print(f"  {len(scored):,} {split} impressions", flush=True)

    clicks = pd.DataFrame(
        {
            "impression_id": scored["impression_id"],
            "click_history": scored["click_history"],
        }
    )
    candidates = list(scored["candidate_ids"])
    label_of = [
        dict(zip(ids, marks))
        for ids, marks in zip(scored["candidate_ids"], scored["labels"])
    ]

    rows: list[dict] = []
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for setting in settings(config):
        started = time.perf_counter()
        tuned = dataclasses.replace(config, lexical=setting.lexical)
        corpus = articles.copy()
        corpus["lexical_text"], _ = preprocess.build_lexical_text(articles, tuned)
        index = bm25_index.build(corpus, tuned)
        queries, _ = bm25_index.build_queries(
            clicks,
            articles,
            tuned,
            retrieval.HISTORY_K,
            with_abstract=setting.with_abstract,
        )

        values = evaluate.per_impression_metrics(
            index.score_candidates(queries, candidates), label_of
        )
        found = retrieval.recall_at_k(
            index.retrieve(queries, depth=max(retrieval.DEPTHS)),
            scored,
            retrieval.DEPTHS,
        )
        vectors[setting.name] = values

        row = {
            "dataset": config.name, "split": split, "setting": setting.name,
            "k1": setting.lexical.k1, "b": setting.lexical.b,
            "title_weight": setting.lexical.title_weight,
            "with_abstract": setting.with_abstract,
            "against": setting.against,
            # Which of these rows the pipeline is actually configured as.
            "current": setting.lexical == config.lexical
            and setting.with_abstract == config.lexical.query_abstract,
            "n": int(len(values["auc"])),
            "seconds": round(time.perf_counter() - started, 1),
        }
        for metric, scores in values.items():
            low, high = embed_compare.interval(scores, resamples)
            row[metric] = float(scores.mean()) if len(scores) else 0.0
            row[f"{metric}_lo"], row[f"{metric}_hi"] = low, high
            if setting.against is None:
                continue
            # Both settings scored the same impressions, in the same order, so
            # the difference is paired. Bootstrapping it directly rather than
            # the two means apart cancels the between-impression variance,
            # which here is most of the width: an impression with two clicked
            # candidates out of forty is hard under every setting.
            gap = scores - vectors[setting.against][metric]
            low, high = embed_compare.interval(gap, resamples)
            row[f"{metric}_gap"] = float(gap.mean()) if len(gap) else 0.0
            row[f"{metric}_gap_lo"], row[f"{metric}_gap_hi"] = low, high
        for depth in retrieval.DEPTHS:
            row[f"recall@{depth}"] = found[f"recall@{depth}"]
        rows.append(row)

        print(
            f"    {setting.name:<24} auc {row['auc']:.4f} "
            f"[{row['auc_lo']:.4f}, {row['auc_hi']:.4f}]  "
            f"recall@200 {row['recall@200']:.4f}",
            flush=True,
        )
    return rows


def document(rows: list[dict], name: str, split: str) -> str:
    lines = [
        f"# BM25's other two questions — {name}, {split} split",
        "",
        f"Named settings rather than a grid, on the **{split}** split, each "
        f"reported on both paths. Generated by `python -m "
        f"pipeline.lexical_ablation`.",
        "",
        "- **recall@K** searches the whole corpus; **auc**, **mrr** and **ndcg** "
        "re-rank the impression's own candidates. Field weighting can move the "
        "two in different directions, which is why both are here.",
        "- The column intervals are the ordinary per-setting ones. What each "
        "change was worth is the **paired** difference under *Reading*, which "
        "is the comparison these rows exist to make.",
        "",
        "| setting | k1 | b | title weight | query | auc | 95% interval | mrr | "
        "ndcg@5 | ndcg@10 | recall@50 | recall@100 | recall@200 |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | "
        "---: | ---: | ---: |",
    ]
    by_name = {row["setting"]: row for row in rows}
    for row in rows:
        lines.append(
            f"| {row['setting']}{' — **current**' if row.get('current') else ''} "
            f"| {row['k1']} | {row['b']} | "
            f"{row['title_weight']} | "
            f"{'titles + abstracts' if row['with_abstract'] else 'titles'} | "
            f"{row['auc']:.4f} | [{row['auc_lo']:.4f}, {row['auc_hi']:.4f}] | "
            f"{row['mrr']:.4f} | {row['ndcg@5']:.4f} | {row['ndcg@10']:.4f} | "
            f"{row['recall@50']:.4f} | {row['recall@100']:.4f} | "
            f"{row['recall@200']:.4f} |"
        )

    lines += ["", "## Reading", ""]
    lines.append(
        "Each row below is a difference from the row it names, bootstrapped "
        "**paired by impression** rather than as two means apart. The settings "
        "score the same impressions, and an impression with two clicked "
        "candidates out of forty is hard under every one of them, so the "
        "between-impression variance that dominates the column intervals above "
        "cancels in the difference. Overlapping column intervals therefore say "
        "nothing either way; these say what the change was worth."
    )
    lines.append("")
    for row in rows:
        if row.get("against") is None:
            continue
        established = row["auc_gap_lo"] > 0 or row["auc_gap_hi"] < 0
        lines.append(
            f"- **{row['setting']}** against *{row['against']}*: "
            f"{row['auc_gap']:+.4f} auc "
            f"[{row['auc_gap_lo']:+.4f}, {row['auc_gap_hi']:+.4f}], "
            f"{row['ndcg@10_gap']:+.4f} ndcg@10, and "
            f"{row['recall@200'] - by_name[row['against']]['recall@200']:+.4f} "
            f"recall@200. The AUC difference **is"
            f"{'' if established else ' not'}** established: the paired "
            f"interval {'excludes' if established else 'contains'} zero."
        )

    tuned = by_name.get("tuned")
    base = by_name.get("inherited")
    if tuned and base and (
        (tuned["auc"] > base["auc"]) != (tuned["recall@200"] > base["recall@200"])
    ):
        lines.append(
            "- **The two paths disagree**: the setting that re-ranks better "
            "retrieves worse. Repeating the title lengthens the document, which "
            "a corpus search feels and a fifteen-candidate re-rank need not. "
            "Which one to prefer depends on whether the candidates are given, "
            "and in this assignment they are."
        )
    lines.append(
        "- The module that builds the query asserted **titles**, on the "
        "argument that abstracts would drown the terms identifying what a user "
        "reads. The last row is what that argument is worth; the row marked "
        "*current* is what the registry does with the answer."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mind")
    parser.add_argument("--split", default=evaluate.TUNE, choices=evaluate.SCORABLE)
    parser.add_argument("--resamples", type=int, default=evaluate.BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)

    config = DATASETS[args.dataset]
    rows = run(config, args.split, args.resamples)

    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{config.name}-{args.split}"
    with (paths.ARTIFACTS_DIR / RESULTS.format(suffix=suffix)).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    text = document(rows, config.name, args.split)
    (paths.ARTIFACTS_DIR / DOCUMENT.format(suffix=suffix)).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
