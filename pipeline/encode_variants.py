"""Encode a dataset's alternate embedding sources, one at a time, one path.

MIND ships no vectors, so every variant compared in phase 7 has to be produced
here — where EB-NeRD's four arrived in its release and only had to be read.

What comes out is an ordinary source. Each variant writes one `.npy` beside
the artifact the pipeline already loads, in the order of the shared
`article_id_index.parquet`, so `embed.read_source` pairs the two without
knowing which model produced either and `pipeline.embed_compare` reads a
generated variant exactly as it reads a shipped one. Nothing downstream
branches on the checkpoint.

Two properties of a checkpoint are read from the registry rather than assumed,
because both fail silently: the **pooling** (`bge-*` reads the CLS token where
the sentence-transformers models read the mean) and the **prefix** (e5 requires
one on every input). Vectors encoded the wrong way are well-formed, unit
length, and not the model's — and the experiment then reports a good encoder as
a bad one.

    python -m pipeline.encode_variants --dataset mind --device cuda
    python -m pipeline.encode_variants --dataset mind --variant bge-base-en-v1.5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import embed, paths
from pipeline.datasets import DATASETS, DatasetConfig, EmbeddingSpec

# Where the word2vec vectors are cached. A gensim KeyedVectors pair, fetched
# from the Hub rather than through `gensim.downloader`, which pulls from GitHub
# — unreachable from Ada, where this runs.
WORD2VEC_FILE = "word2vec-google-news-300.model"


def texts_for(config: DatasetConfig) -> list[str]:
    """The text each article is encoded from, in the id index's own order.

    The order is the whole contract. `embed.read_source` reads one `.npy` and
    one `article_id_index.parquet` and pairs them by position, so a variant
    encoded in any other order would align to the catalogue perfectly and mean
    something else in every row.

    A dataset that has never been embedded has no index yet, which is the state
    every new one starts in -- `mind_large` reached it first. The index *is* the
    catalogue's own order (checked on MIND: same ids, same positions), so it is
    written from the catalogue here rather than being a thing you must already
    have in order to make it.
    """
    articles = pd.read_parquet(
        config.feature_store_dir / "articles.parquet",
        columns=["article_id", "title", "abstract"],
    )
    stored = config.artifacts_dir / embed.ID_INDEX
    if not stored.exists():
        stored.parent.mkdir(parents=True, exist_ok=True)
        articles[["article_id"]].astype({"article_id": "string"}).to_parquet(
            stored, index=False
        )
        print(f"  wrote {embed.ID_INDEX} from the catalogue, {len(articles):,} rows")
    index = pd.read_parquet(stored)
    ordered = (
        index[["article_id"]]
        .astype({"article_id": "string"})
        .merge(articles.astype({"article_id": "string"}), on="article_id", how="left")
    )
    if len(ordered) != len(index):
        raise embed.EmbeddingError(
            f"{config.name}: the id index has {len(index):,} rows and the join "
            f"onto the catalogue produced {len(ordered):,}"
        )
    return list(embed.document_text(ordered))


def encode_variant(
    config: DatasetConfig,
    spec: EmbeddingSpec,
    texts: list[str],
    device: str,
    batch_size: int,
) -> dict:
    """One variant onto disk, with what it cost to put there."""
    import torch

    destination = config.artifacts_dir / spec.artifact
    destination.parent.mkdir(parents=True, exist_ok=True)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    if spec.encoder == "word2vec":
        from huggingface_hub import snapshot_download

        cached = snapshot_download(repo_id=spec.model)
        matrix = embed.encode_word2vec(texts, Path(cached) / WORD2VEC_FILE, spec.dim)
    elif spec.encoder == "transformer":
        matrix = embed.encode(
            texts, config, device=device, batch_size=batch_size, spec=spec
        )
    else:
        raise embed.EmbeddingError(
            f"unknown encoder {spec.encoder!r} on {spec.name}"
        )

    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated() / 2**30
        if device.startswith("cuda")
        else 0.0
    )

    if matrix.shape != (len(texts), spec.dim):
        raise embed.EmbeddingError(
            f"{spec.name}: encoded {matrix.shape}, registry declares "
            f"({len(texts)}, {spec.dim})"
        )
    np.save(destination, matrix)

    # A row of zeros is a document the encoder had nothing to say about —
    # word2vec's out-of-vocabulary case. Reported rather than hidden, because
    # it is the one number that separates a floor from a broken run.
    empty = int((~matrix.any(axis=1)).sum())
    return {
        "variant": spec.name,
        "dim": spec.dim,
        "articles": len(texts),
        "empty_rows": empty,
        "seconds": round(elapsed, 1),
        "peak_vram_gib": round(peak, 2),
        "device": device,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mind")
    parser.add_argument(
        "--variant", action="append",
        help="encode only this variant, by its registry label; repeatable",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--force", action="store_true",
        help="re-encode a variant whose artifact is already on disk",
    )
    args = parser.parse_args(argv)

    config = DATASETS[args.dataset]
    wanted = set(args.variant or [])
    specs = [
        spec for spec in config.embedding_variants
        if not wanted or spec.name in wanted
    ]
    unknown = wanted - {spec.name for spec in config.embedding_variants}
    if unknown:
        raise SystemExit(
            f"no such variant: {', '.join(sorted(unknown))}. "
            f"Known: {', '.join(s.name for s in config.embedding_variants)}"
        )

    print(f"  root      {paths.ROOT}")
    print(f"  device    {args.device}")
    texts = texts_for(config)
    print(f"  articles  {len(texts):,}\n")

    for spec in specs:
        if (config.artifacts_dir / spec.artifact).exists() and not args.force:
            print(f"  have {spec.name}")
            continue
        print(f"  encoding {spec.name} ...", flush=True)
        report = encode_variant(config, spec, texts, args.device, args.batch_size)
        print(
            f"    {report['seconds']:8.1f} s  {report['dim']:4}d  "
            f"peak VRAM {report['peak_vram_gib']:.2f} GiB  "
            f"{report['empty_rows']:,} empty rows",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
