"""Which vector source, and which geometry correction, this dataset should use.

EB-NeRD ships four sets of article vectors and the pipeline can only index one.
Choosing between them by reputation -- BERT is newer than word2vec, so BERT --
is how this project ended up with a semantic retriever that ranked at chance:
the mBERT vectors occupy a narrow cone and a cosine between two of them says
almost nothing. So the choice is made by measurement, on the tune split, over
both axes at once: which artifact, and which correction applied to it.

Scored here rather than through `pipeline.evaluate` because that reads the one
matrix the embed stage put on disk, and a cell of this grid is a matrix that
was never built. AUC and its bootstrap use the same definition, resample count
and seed as the harness, so a number here is comparable with one from there.

Nothing in this module chooses anything. It writes the table; the winner is
promoted into the registry by hand, which is a decision with a commit message.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline import embed, evaluate, ingest, paths, retrieval
from pipeline.datasets import DATASETS, DatasetConfig, EmbeddingSpec

# The corrections every variant is tried under. `abtt:10` is dropped from the
# earlier seven: both datasets were already declining by five components, so a
# tenth only costs a cell.
METHODS = ("none", "centre", "abtt:1", "abtt:3", "abtt:5", "whiten")

# Named for the dataset as well as the split, like every other comparison this
# project writes. Without the dataset in it, running the grid on MIND would
# overwrite EB-NeRD's -- silently, and with a table that looks entirely
# well-formed under the other dataset's name.
RESULTS = "embeddings-{dataset}-{split}.jsonl"
DOCUMENT = "embeddings-{dataset}-{split}.md"


def variants(config: DatasetConfig) -> tuple[EmbeddingSpec, ...]:
    """Every vector source for this dataset, the active one first, once each.

    Deduplicated by artifact, because promoting a variant into the active slot
    leaves it in both lists -- which is the right thing for the registry, since
    the grid it won is part of its record. Scored twice it would put two
    identical rows in the table under one name, and a reader would reasonably
    wonder which of them was the real one.
    """
    seen: set[str] = set()
    found: list[EmbeddingSpec] = []
    for spec in (config.embeddings, *config.embedding_variants):
        if spec.artifact not in seen:
            seen.add(spec.artifact)
            found.append(spec)
    return tuple(found)


def corpus_matrix(config: DatasetConfig, spec: EmbeddingSpec) -> np.ndarray:
    """One variant's vectors, aligned to the catalogue and unit length.

    Deliberately without post-processing: that is the other axis of the grid,
    and applying it here would fix it.
    """
    articles = pd.read_parquet(
        config.feature_store_dir / "articles.parquet", columns=["article_id"]
    )
    source_ids, source_vectors = embed.read_source(
        dataclasses.replace(config, embeddings=spec)
    )
    matrix, _ = embed.align(
        source_ids, source_vectors, articles["article_id"], spec.dim
    )
    del source_ids, source_vectors
    return embed.normalise(matrix)


def impressions(config: DatasetConfig, split: str) -> pd.DataFrame:
    behaviors = pd.read_parquet(config.feature_store_dir / "behaviors.parquet")
    impressions = behaviors[behaviors["split"] == split]
    history = ingest.history_for(config, impressions, columns=("click_history",))
    return impressions.merge(history[["impression_id", "click_history"]], on="impression_id")


def per_impression_auc(
    matrix: np.ndarray, rows: pd.DataFrame, index: dict[str, int], history_k: int
) -> np.ndarray:
    """AUC per impression, on the ones where it is defined.

    Undefined where every candidate is clicked or none is -- the harness counts
    those and leaves them out rather than scoring them zero, and so does this.
    An impression whose user has no usable click history is left out too: it
    would score the candidate file's own order under the variant's name.
    """
    scores = []
    for clicks, candidates, labels in zip(
        rows["click_history"], rows["candidate_ids"], rows["labels"]
    ):
        truth = np.asarray(labels)
        if truth.sum() == 0 or truth.sum() == len(truth):
            continue
        history = [index[c] for c in clicks[-history_k:] if c in index]
        if not history:
            continue
        user = matrix[history].mean(axis=0)
        length = np.linalg.norm(user)
        if length == 0:
            continue
        user = user / length
        candidate_rows = [index.get(c, -1) for c in candidates]
        against = np.array(
            [matrix[r] @ user if r >= 0 else 0.0 for r in candidate_rows]
        )
        scores.append(roc_auc_score(truth, against))
    return np.asarray(scores)


def interval(values: np.ndarray, resamples: int) -> tuple[float, float]:
    """The harness's bootstrap, over the same axis and from the same seed."""
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(evaluate.BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    edge = (1 - evaluate.CONFIDENCE) / 2
    return float(np.quantile(means, edge)), float(np.quantile(means, 1 - edge))


def document(rows: list[dict], config_name: str, split: str) -> str:
    """The grid as markdown, best first within each variant."""
    lines = [
        f"# Embedding source and geometry — {config_name}, {split} split",
        "",
        f"Every vector source this dataset ships, under every correction, "
        f"scored on the **{split}** split. Generated by "
        f"`python -m pipeline.embed_compare`; nothing here was typed by hand.",
        "",
        "- **anisotropy** is the mean cosine between two different articles. "
        "Near 0 means a cosine between two vectors is informative; near 1 "
        "means every pair looks alike whatever the articles say.",
        "- A difference is a difference only where the intervals are disjoint.",
        "- The correction is applied to the *articles*; the user vector is the "
        "mean of their corrected clicked articles, as `ann` builds it.",
        "",
        "| variant | dim | correction | anisotropy | auc | 95% interval |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['dim']} | {row['method']} | "
            f"{row['anisotropy']:+.4f} | {row['auc']:.4f} | "
            f"[{row['lo']:.4f}, {row['hi']:.4f}] |"
        )

    best = max(rows, key=lambda r: r["auc"])
    contested = [
        r for r in rows
        if r is not best and r["hi"] >= best["lo"] and r["lo"] <= best["hi"]
    ]
    lines += [
        "",
        "## Reading",
        "",
        f"- Best on this split: **{best['variant']} under {best['method']}** "
        f"(auc {best['auc']:.4f} [{best['lo']:.4f}, {best['hi']:.4f}], "
        f"anisotropy {best['anisotropy']:+.4f}).",
    ]
    if contested:
        names = ", ".join(f"{r['variant']}/{r['method']}" for r in contested[:5])
        lines.append(
            f"- Not separated from it by disjoint intervals: {names}. Those are "
            f"not established as worse, and the cheaper of them is the better "
            f"choice at equal evidence."
        )
    else:
        lines.append(
            "- Every other cell sits below it by a disjoint interval, so the "
            "choice is established on this split rather than picked."
        )
    return "\n".join(lines) + "\n"


def run(config: DatasetConfig, split: str, resamples: int) -> list[dict]:
    rows: list[dict] = []
    scored = impressions(config, split)
    print(f"  {len(scored):,} {split} impressions", flush=True)

    for spec in variants(config):
        matrix = corpus_matrix(config, spec)
        index = {
            article: row
            for row, article in enumerate(
                pd.read_parquet(
                    config.feature_store_dir / "articles.parquet",
                    columns=["article_id"],
                )["article_id"]
            )
        }
        for method in METHODS:
            started = time.perf_counter()
            corrected = embed.postprocess(matrix, method)
            values = per_impression_auc(
                corrected, scored, index, retrieval.HISTORY_K
            )
            low, high = interval(values, resamples)
            row = {
                "dataset": config.name,
                "split": split,
                "variant": spec.name,
                "dim": corrected.shape[1],
                "method": method,
                "anisotropy": embed.anisotropy(corrected),
                "auc": float(values.mean()) if len(values) else 0.0,
                "n": int(len(values)),
                "seconds": round(time.perf_counter() - started, 1),
            }
            rows.append(row)
            print(
                f"    {spec.name:38} {method:8} auc {row['auc']:.4f} "
                f"[{low:.4f}, {high:.4f}]  anisotropy {row['anisotropy']:+.4f}",
                flush=True,
            )
            row["lo"], row["hi"] = low, high
        del matrix
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ebnerd")
    parser.add_argument("--split", default=evaluate.TUNE, choices=evaluate.SCORABLE)
    parser.add_argument("--resamples", type=int, default=evaluate.BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)

    config = DATASETS[args.dataset]
    rows = run(config, args.split, args.resamples)

    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results = paths.ARTIFACTS_DIR / RESULTS.format(
        dataset=config.name, split=args.split
    )
    with results.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    text = document(rows, config.name, args.split)
    (
        paths.ARTIFACTS_DIR
        / DOCUMENT.format(dataset=config.name, split=args.split)
    ).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
