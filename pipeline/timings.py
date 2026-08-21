"""What each stage cost, in wall time and in memory.

Phase 9's scale analysis has to cite measurements rather than estimates, and
`/usr/bin/time -v` wrapped around the whole job cannot supply them: it reports
one peak for nine stages, so a stage that allocates 8 GB and frees it is
indistinguishable from one that never allocated anything. The number that
decides how many CPUs a SLURM job must buy — memory is bought at 3000 MB per
CPU on Ada — is the per-stage one.

So each stage is sampled while it runs. `/proc/self/status` carries the
process's current `VmRSS`; a thread reads it every 100 ms and keeps the
largest value seen between entering the stage and leaving it. That is Linux
only, which both machines this runs on are.

Two things the sampler cannot see, and neither matters here: memory held by a
child process (every stage runs in-process), and a spike shorter than the
sampling interval (a stage's peak is held by an array that lives for the whole
stage, not by a transient).

Rows accumulate in `artifacts/build-timings.jsonl` across runs, because a
build skips the stages it has already done and no single invocation sees all
nine. The document reports the most recent row per stage and dataset.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pipeline import paths

RESULTS = "build-timings.jsonl"
DOCUMENT = "build-timings.md"

INTERVAL = 0.1
STATUS = Path("/proc/self/status")


def resident_mb() -> float:
    for line in STATUS.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError(f"no VmRSS in {STATUS}")


class _Sampler(threading.Thread):
    """The high-water mark of VmRSS over the sampler's lifetime."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = resident_mb()
        # Not `_stop`: Thread already has one, and overwriting it makes join()
        # raise from inside the standard library.
        self._finished = threading.Event()

    def run(self) -> None:
        while not self._finished.wait(INTERVAL):
            self.peak = max(self.peak, resident_mb())

    def stop(self) -> float:
        self._finished.set()
        self.join()
        return max(self.peak, resident_mb())


@contextmanager
def measure(stage: str, dataset: str, run: str):
    """Time and sample one stage, appending a row when it finishes.

    A stage that raises writes nothing: the build does not mark it done
    either, and a cost for work that did not complete is worse than no number.
    """
    sampler = _Sampler()
    sampler.start()
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        peak = sampler.stop()

    record(
        {
            "run": run,
            "host": socket.gethostname(),
            "cpus": len(os.sched_getaffinity(0)),
            "dataset": dataset,
            "stage": stage,
            "seconds": round(elapsed, 2),
            "peak_rss_mb": round(peak, 1),
        }
    )


def record(row: dict) -> None:
    path = paths.ARTIFACTS_DIR / RESULTS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")


def latest() -> list[dict]:
    """The most recent row per stage and dataset, in the order stages run.

    A rebuild re-measures some stages and skips others, so the file holds
    several rows for the same cell and the newest is the one that describes
    the pipeline as it stands.
    """
    path = paths.ARTIFACTS_DIR / RESULTS
    if not path.exists():
        return []

    from pipeline.stages import STAGES

    rows: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(row["dataset"], row["stage"])] = row

    order = {stage.name: i for i, stage in enumerate(STAGES)}
    return sorted(
        rows.values(), key=lambda row: (row["dataset"], order.get(row["stage"], 99))
    )


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def memory(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def document(rows: list[dict]) -> str:
    lines = [
        "# What each stage costs",
        "",
        "Wall time and peak resident memory per stage, appended by `python "
        "build.py` as each stage finishes and reported here newest-first-wins. "
        "Regenerate with `python -m pipeline.timings`.",
        "",
        "- **peak RSS** is `VmRSS` sampled every 100 ms *while that stage "
        "runs*, so it is the stage's own high-water mark. `/usr/bin/time -v` "
        "around the whole job reports one number for all nine and cannot "
        "separate them.",
        "- A stage is measured on whichever machine last ran it, which is why "
        "the host and CPU count are columns rather than a heading. Wall times "
        "from different hosts are not comparable; the memory is.",
        "",
        "| dataset | stage | wall | peak RSS | host | cpus | measured |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['stage']} | "
            f"{duration(row['seconds'])} | {memory(row['peak_rss_mb'])} | "
            f"{row['host']} | {row['cpus']} | {row['run'][:10]} |"
        )

    lines += ["", "## Reading", ""]
    if not rows:
        lines.append("Nothing measured yet.")
        return "\n".join(lines) + "\n"

    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset, group in by_dataset.items():
        heaviest = max(group, key=lambda row: row["peak_rss_mb"])
        slowest = max(group, key=lambda row: row["seconds"])
        lines.append(
            f"- **{dataset}**: {duration(sum(row['seconds'] for row in group))} "
            f"across {len(group)} stage{'s' if len(group) != 1 else ''}, with "
            f"**{slowest['stage']}** the longest at "
            f"{duration(slowest['seconds'])} and **{heaviest['stage']}** the "
            f"largest at {memory(heaviest['peak_rss_mb'])}."
        )

    ceiling = max(row["peak_rss_mb"] for row in rows)
    lines.append(
        f"- The whole pipeline's ceiling is **{memory(ceiling)}**, set by "
        f"`{max(rows, key=lambda row: row['peak_rss_mb'])['stage']}`. Ada sells "
        f"memory at 3000 MB per CPU, so that is "
        f"`-c {max(1, -(-int(ceiling) // 3000))}` at the sizes measured here."
    )
    return "\n".join(lines) + "\n"


def write_document() -> Path:
    path = paths.ARTIFACTS_DIR / DOCUMENT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document(latest()), encoding="utf-8")
    return path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    print(write_document())
