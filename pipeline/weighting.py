"""How much each past click counts toward the profile it feeds.

Every scheme here weights the *clicks*, not the candidates, and every one of
them reads a column describing a click that has already happened — so every one
is available at serving time. That is the line this module stays on: recency
and engagement are behavioural signals about the past, which the assignment
allows, and none of them looks at the impression's own outcome.

    uniform      every click counts the same. What the pipeline shipped with.
    position     w = decay ** (how many clicks ago), so the newest counts 1.
    time         w = exp(-hours since the click / decay). News decays in real
                 time rather than in click count, which is why this is worth
                 separating from position at all.
    engagement   w from how long the click was read and how far it was
                 scrolled. A headline someone opened and abandoned says less
                 about them than one they read to the end.

`time` and `engagement` need columns MIND does not have, so they are available
on EB-NeRD alone. That asymmetry is read off the registry's ColumnMap rather
than listed again here — a dataset supports a scheme exactly when it has the
column the scheme reads, and a second list would be a second thing to keep
true.
"""

from __future__ import annotations

import numpy as np

from pipeline import sources
from pipeline.datasets import DatasetConfig

SCHEMES = ("uniform", "position", "time", "engagement")

# What each scheme reads beside the click ids. One table, because three things
# now need the answer: which schemes a dataset supports, which columns the
# submission path has to stream, and which it can leave behind. `position`
# reads nothing -- a click's place in the window is the window's own shape.
READS: dict[str, tuple[str, ...]] = {
    "uniform": (),
    "position": (),
    "time": ("click_times",),
    "engagement": ("click_read_times", "click_scroll"),
}

# What `read_time` and `scroll_percentage` are turned into. read_time is
# clipped at 1800 seconds with a median of 14, so it is a long tail that raw
# would let one 30-minute read outvote twenty ordinary ones; log1p damps it to
# a ratio. scroll is already a percentage and needs no shaping.
SCROLL_FULL = 100.0


class WeightingError(RuntimeError):
    """A scheme was asked for that this dataset's history cannot support."""


def available(config: DatasetConfig) -> tuple[str, ...]:
    """The schemes this dataset has the columns for.

    Derived from the ColumnMap, so a dataset that carries a column supports the
    scheme that reads it and one that does not, does not. MIND's history is a
    bare id list: it has positions but no timestamps and no engagement.
    """
    history = config.columns.history
    return tuple(
        scheme
        for scheme in SCHEMES
        if all(history.get(column) for column in READS[scheme])
    )


def columns_for(scheme: str) -> tuple[str, ...]:
    """The history columns this scheme reads, beside the click ids.

    The submission path streams 808k histories and cannot carry columns
    nothing will read, so it asks here rather than carrying all three. An
    unknown scheme is left to `check` to reject with a useful message.
    """
    return READS.get(scheme, ())


def check(config: DatasetConfig, scheme: str, history_k: int | None = None) -> None:
    """That this dataset can express the scheme, before anything is computed.

    Loudly, because the alternative is a run that silently falls back to
    uniform and reports its numbers under the name of a scheme that never ran.
    """
    if scheme not in SCHEMES:
        raise WeightingError(
            f"unknown weighting {scheme!r}, expected one of {', '.join(SCHEMES)}"
        )
    if scheme not in available(config):
        raise WeightingError(
            f"{config.name} cannot weight by {scheme!r}: its history carries no "
            f"such column. Available here: {', '.join(available(config))}"
        )
    # The engagement arrays are truncated at ingest, so past that point
    # `times[-k:]` and `clicks[-k:]` are different lengths and every weight
    # lands on the wrong click. Refused rather than clipped, because clipping
    # would silently weight a 160-click window as if it were a 100-click one.
    if (
        history_k is not None
        and scheme in ("time", "engagement")
        and history_k > sources.ENGAGEMENT_WINDOW
    ):
        raise WeightingError(
            f"{scheme!r} weighting cannot reach back {history_k} clicks: the "
            f"engagement arrays keep the last {sources.ENGAGEMENT_WINDOW}. "
            f"Raise sources.ENGAGEMENT_WINDOW and re-run ingest."
        )


def weights(
    scheme: str,
    decay: float,
    count: int,
    *,
    times: np.ndarray | None = None,
    read_times: np.ndarray | None = None,
    scroll: np.ndarray | None = None,
    at: np.datetime64 | None = None,
) -> np.ndarray:
    """One weight per click in the window, oldest first.

    Oldest first because that is the order the history is stored in and the
    order every consumer slices — `[-k:]` is a suffix, so the *last* entry is
    the most recent click. A scheme that read it the other way round would
    weight the stalest click most and still look plausible.

    All-zero weights are returned as uniform rather than as zeros: a weighted
    mean would divide by zero, and an impression whose every click scores zero
    engagement is one this scheme has nothing to say about, not one whose
    profile is empty.
    """
    if count == 0:
        return np.empty(0, dtype="float64")

    if scheme == "uniform":
        return np.ones(count, dtype="float64")

    if scheme == "position":
        # Most recent click is the last entry, so it gets decay ** 0 == 1.
        return decay ** np.arange(count - 1, -1, -1, dtype="float64")

    if scheme == "time":
        if times is None or at is None:
            raise WeightingError("time weighting needs click times and an as-of")
        hours = (
            (np.datetime64(at) - np.asarray(times, dtype="datetime64[us]"))
            / np.timedelta64(1, "h")
        ).astype("float64")
        # A click stamped after the impression is a clock artifact rather than
        # the future leaking in; it weights as if it had just happened.
        return np.exp(-np.maximum(hours, 0.0) / decay)

    if read_times is None or scroll is None:
        raise WeightingError("engagement weighting needs read times and scroll")
    # scroll is 10.8% null *inside* its arrays, so a missing value is neutral
    # rather than zero -- the click happened, only the measurement is absent.
    depth = np.asarray(scroll, dtype="float64") / SCROLL_FULL
    depth = np.where(np.isnan(depth), 1.0, depth)
    read = np.log1p(np.nan_to_num(np.asarray(read_times, dtype="float64"), nan=0.0))
    found = read * depth
    return found if found.sum() > 0 else np.ones(count, dtype="float64")


def normalise(found: np.ndarray) -> np.ndarray:
    """Weights summing to one, or uniform where they sum to nothing."""
    total = found.sum()
    if total <= 0:
        return np.full(len(found), 1.0 / len(found)) if len(found) else found
    return found / total
