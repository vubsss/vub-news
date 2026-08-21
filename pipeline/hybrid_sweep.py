"""Choosing how to combine the two retrievers, on tune.

    python -m pipeline.hybrid_sweep --dataset mind

Two rules and one parameter each: RRF's `k`, centred on the conventional 60
rather than searched blind, and the linear rule's `alpha`, which weights the
lexical side. Both on `tune`, both paired by impression against the **better
parent** — because the question a hybrid has to answer is not "does fusing beat
the worse retriever" but "is it worth having at all".

The parents are ranked **once** and fused many times. `bm25.rank_candidates`
scores the whole corpus per impression and takes ten minutes on EB-NeRD at
k=80; re-running it for every cell would cost hours to compute the same numbers
repeatedly. Fusion is arithmetic over two frames that are already in memory.

Calibrate the expectation before reading the output. Published hybrids report a
lift over the *better* parent, not the worse one — on WANDS a tuned hybrid
reaches 0.7497 nDCG against 0.6983 and 0.6953 for its parents, about 7.4%. A
couple of points over the stronger parent is the normal outcome and is worth
writing up as one; a hybrid that ties is a result too, and this document says so
rather than presenting its argmax as a win.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import pandas as pd

from pipeline import (
    ann_index,
    bm25_index,
    embed_compare,
    evaluate,
    hybrid,
    paths,
    retrieval,
)
from pipeline.datasets import DATASETS, DatasetConfig, HybridSpec

# Centred on the conventional default rather than searched blind, and spread
# wide enough on either side to show the curve turning over if it does.
KS = (1.0, 5.0, 10.0, 20.0, 60.0, 120.0, 300.0)

# alpha weights the lexical side: 0 is the semantic index alone, 1 is BM25
# alone. The endpoints are in the grid deliberately -- they are the parents,
# so a grid whose best cell is an endpoint is telling you not to fuse.
ALPHAS = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)

RESULTS = "hybrid-{suffix}.jsonl"
DOCUMENT = "hybrid-{suffix}.md"


def cells() -> list[HybridSpec]:
    return [HybridSpec(rule="rrf", k=k) for k in KS] + [
        HybridSpec(rule="linear", alpha=a) for a in ALPHAS
    ]


def parents(
    config: DatasetConfig, split: str
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Both parents' rankings for the split, computed once."""
    store = config.feature_store_dir
    behaviors = pd.read_parquet(store / "behaviors.parquet")
    history = pd.read_parquet(store / "history.parquet")
    impressions = behaviors[behaviors["split"] == split]
    history = history[history["impression_id"].isin(set(impressions["impression_id"]))]
    print(f"  {len(impressions):,} {split} impressions", flush=True)

    ranked = {}
    for name, module in (("bm25", bm25_index), ("ann", ann_index)):
        started = time.perf_counter()
        ranked[name] = module.rank_candidates(config, impressions, history)
        print(f"    {name} ranked in {time.perf_counter() - started:.0f}s", flush=True)
    return ranked, impressions


def labels_for(impressions: pd.DataFrame, ranked: pd.DataFrame) -> list[dict[str, int]]:
    of = {
        impression: dict(zip(candidates, marks, strict=True))
        for impression, candidates, marks in zip(
            impressions["impression_id"],
            impressions["candidate_ids"],
            impressions["labels"],
        )
    }
    return [of[impression] for impression in ranked["impression_id"]]


def run(config: DatasetConfig, split: str, resamples: int) -> list[dict]:
    ranked, impressions = parents(config, split)

    measured = {
        name: evaluate.per_impression_metrics(frame, labels_for(impressions, frame))
        for name, frame in ranked.items()
    }
    rows = [
        {
            "dataset": config.name, "split": split, "rule": name, "k": None,
            "alpha": None, "is_parent": True,
            "n": int(len(measured[name]["auc"])),
            **{m: float(measured[name][m].mean()) for m in evaluate.ACCURACY_METRICS},
        }
        for name in ranked
    ]

    # The comparison every cell is measured against: the better parent, chosen
    # on this split. Beating the worse one is not a reason to fuse.
    better = max(ranked, key=lambda name: measured[name]["auc"].mean())
    baseline = measured[better]
    print(f"  better parent on {split}: {better} "
          f"({baseline['auc'].mean():.4f})", flush=True)

    for spec in cells():
        started = time.perf_counter()
        fused = hybrid.fuse(ranked, spec)
        values = evaluate.per_impression_metrics(
            fused, labels_for(impressions, fused)
        )
        row = {
            "dataset": config.name, "split": split, "rule": spec.rule,
            "k": spec.k if spec.rule == "rrf" else None,
            "alpha": spec.alpha if spec.rule == "linear" else None,
            "is_parent": False, "against": better,
            "n": int(len(values["auc"])),
            "seconds": round(time.perf_counter() - started, 1),
        }
        for metric in evaluate.ACCURACY_METRICS:
            row[metric] = float(values[metric].mean())
            gap = values[metric] - baseline[metric]
            low, high = embed_compare.interval(gap, resamples)
            row[f"{metric}_gap"] = float(gap.mean())
            row[f"{metric}_gap_lo"], row[f"{metric}_gap_hi"] = low, high
        rows.append(row)
        setting = f"k={spec.k:g}" if spec.rule == "rrf" else f"alpha={spec.alpha:g}"
        print(
            f"    {spec.rule:<6} {setting:<11} auc {row['auc']:.4f}  "
            f"vs {better} {row['auc_gap']:+.4f} "
            f"[{row['auc_gap_lo']:+.4f}, {row['auc_gap_hi']:+.4f}]",
            flush=True,
        )
    return rows


def document(rows: list[dict], name: str, split: str) -> str:
    parent_rows = [r for r in rows if r["is_parent"]]
    fused = [r for r in rows if not r["is_parent"]]
    better = fused[0]["against"] if fused else "?"

    lines = [
        f"# Hybrid retrieval — {name}, {split} split",
        "",
        f"BM25 and the semantic index combined, on the **{split}** split. "
        f"Generated by `python -m pipeline.hybrid_sweep`.",
        "",
        f"- Every cell is paired by impression against the **better parent** "
        f"(`{better}`), because the question is whether fusing is worth having "
        f"at all — beating the weaker retriever is not a reason to ship one.",
        "- `alpha` weights the **lexical** side, so `alpha=0` is the semantic "
        "index alone and `alpha=1` is BM25 alone. Those endpoints are in the "
        "grid on purpose: a winner there is the grid saying not to fuse.",
        "",
        "| retriever | auc | mrr | ndcg@5 | ndcg@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(parent_rows, key=lambda r: -r["auc"]):
        lines.append(
            f"| {row['rule']} | {row['auc']:.4f} | {row['mrr']:.4f} | "
            f"{row['ndcg@5']:.4f} | {row['ndcg@10']:.4f} |"
        )

    lines += [
        "",
        "| rule | setting | auc | ndcg@10 | vs better parent (paired) |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in fused:
        setting = f"k={row['k']:g}" if row["rule"] == "rrf" else f"alpha={row['alpha']:g}"
        lines.append(
            f"| {row['rule']} | {setting} | {row['auc']:.4f} | "
            f"{row['ndcg@10']:.4f} | {row['auc_gap']:+.4f} "
            f"[{row['auc_gap_lo']:+.4f}, {row['auc_gap_hi']:+.4f}] |"
        )

    lines += ["", "## Reading", ""]
    if not fused:
        return "\n".join(lines) + "\n"

    best = max(fused, key=lambda r: r["auc"])
    setting = f"k={best['k']:g}" if best["rule"] == "rrf" else f"alpha={best['alpha']:g}"
    wins = [r for r in fused if r["auc_gap_lo"] > 0]
    lines.append(
        f"- Best cell: **{best['rule']}, {setting}** (auc {best['auc']:.4f}), "
        f"{best['auc_gap']:+.4f} against `{better}`."
    )
    lines.append(
        f"- {len(wins)} of {len(fused)} cells beat the better parent by a paired "
        f"interval clear of zero."
        if wins
        else f"- **No cell beats `{better}`** by a paired interval clear of zero. "
        f"On this dataset the hybrid is not worth having, and the honest "
        f"report is that the better parent stands alone."
    )
    endpoint = [
        r for r in fused
        if r["rule"] == "linear" and r["alpha"] in (0.0, 1.0)
        and r["auc"] >= best["auc"] - 1e-12
    ]
    if endpoint:
        lines.append(
            "- The best linear cell is an **endpoint**, which is one of the "
            "parents wearing the hybrid's name. Read that as the grid declining "
            "to fuse rather than as a hybrid result."
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
