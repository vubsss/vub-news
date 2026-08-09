#!/usr/bin/env python3
"""One-command rebuild of the MIND + EB-NeRD retrieval pipeline.

    python build.py                    rebuild everything that is not done
    python build.py --plan             show what would run, change nothing
    python build.py --dataset mind     limit to one dataset
    python build.py --force bm25 ann   re-run these stages even if done
    python build.py --force all        rebuild from scratch

Stages run in the order declared in pipeline/stages.py. Each one writes a
checkpoint when it finishes, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import os
import sys

from pipeline import paths, stages
from pipeline.acquire import AcquisitionError
from pipeline.datasets import DATASETS, DatasetConfig
from pipeline.embed import EmbeddingError
from pipeline.stages import STAGES, Stage

DONE = "done"
PENDING = "pending"
NOT_BUILT = "not built"


def load_env_file() -> None:
    """Read credentials from .env, so they survive across shells.

    An already-exported variable wins, so `HF_TOKEN=... python build.py` still
    does what it looks like it does.
    """
    env_file = paths.ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def status(stage: Stage, dataset: DatasetConfig, forced: set[str]) -> str:
    if stage.run is None:
        return NOT_BUILT
    if stage.name in forced:
        return PENDING
    return DONE if stages.is_done(stage, dataset) else PENDING


def print_plan(datasets: list[DatasetConfig], forced: set[str]) -> None:
    width = max(len(s) for s in (*DATASETS, DONE, PENDING, NOT_BUILT)) + 3
    header = "".join(f"{d.name:<{width}}" for d in datasets)
    print(f"\n  {'#':<3}{'stage':<12}{header}what it does")
    for i, stage in enumerate(STAGES, start=1):
        cells = "".join(f"{status(stage, d, forced):<{width}}" for d in datasets)
        print(f"  {i:<3}{stage.name:<12}{cells}{stage.description}")
    print()


def run(datasets: list[DatasetConfig], forced: set[str]) -> None:
    ran = skipped = not_built = 0
    for stage in STAGES:
        for dataset in datasets:
            state = status(stage, dataset, forced)
            if state == NOT_BUILT:
                not_built += 1
                continue
            if state == DONE:
                skipped += 1
                continue
            print(f"  running {stage.name} [{dataset.name}] ...")
            stage.run(dataset, stage.name in forced)
            stages.mark_done(stage, dataset)
            ran += 1

    print(
        f"\n  {ran} ran, {skipped} already done, "
        f"{not_built} not built yet (see ../tickets/).\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="restrict to one dataset (repeatable); default is all of them",
    )
    parser.add_argument(
        "--force",
        nargs="+",
        default=[],
        metavar="STAGE",
        help="re-run these stages even if their checkpoint exists, "
        "or 'all' for every stage",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print the stage table and exit without running anything",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file()

    datasets = [DATASETS[name] for name in (args.dataset or sorted(DATASETS))]

    forced = set(args.force)
    if "all" in forced:
        forced = {stage.name for stage in STAGES}
    unknown = forced - set(stages.STAGES_BY_NAME)
    if unknown:
        known = ", ".join(stages.STAGES_BY_NAME)
        print(
            f"error: unknown stage(s): {', '.join(sorted(unknown))}\n"
            f"known stages: {known}, all",
            file=sys.stderr,
        )
        return 2

    print_plan(datasets, forced)
    if args.plan:
        return 0

    try:
        run(datasets, forced)
    except (AcquisitionError, EmbeddingError) as error:
        # Something the user has to fix. A traceback would only bury it.
        print(f"\nerror: {error}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
