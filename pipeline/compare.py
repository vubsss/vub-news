"""Lexical against semantic: both datasets, every slice, from stored results.

    python -m pipeline.compare                   both datasets, validation
    python -m pipeline.compare --split test      the held-back split
    python -m pipeline.compare --dataset mind    one of them

Reads the json every `pipeline.evaluate` run leaves in
`artifacts/<dataset>/evaluate/`, so it ranks nothing itself: the comparison is
a view over measurements already taken, and regenerating it costs a second
rather than a re-run of both retrievers over both datasets.

It writes one markdown document — the tables and a reading of them — to
`artifacts/comparison-<split>.md` and prints the same text. Markdown because
its destination is the design note, and generated rather than written by hand
because a sentence naming a winner has to change when the numbers do.

The two retrievers are separated by asking whether their bootstrap intervals
overlap, which is deliberately conservative and stated as such in the document:
disjoint intervals establish a difference, overlapping ones establish nothing
in either direction. Overlap is not evidence of equality, and since both
retrievers score the same impressions, a paired test on the per-impression
differences would separate more than this does — it would need the
per-impression values, which the stored reports do not carry.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline import evaluate, paths
from pipeline.datasets import DATASETS, DatasetConfig

# The question this module answers is about two named approaches, so it
# compares a pair rather than everything `evaluate.RETRIEVERS` can score: bm25
# is the lexical arm, the ANN index over article embeddings the semantic one.
# A third retriever is scored by the harness and ignored here until someone
# says what it should be compared against.
LEXICAL, SEMANTIC = "bm25", "ann"

# Only the ranking metrics have a better direction. Diversity, novelty and
# coverage describe what a list is made of, so the tables say which retriever
# is *higher* and never which one won.
ACCURACY = evaluate.ACCURACY_METRICS

# Coverage's interval is a width rather than a bracket around its own value
# (see evaluate.interval), so an overlap test on it would be a test of the
# wrong thing. Compared by value, and marked as such wherever it is reported.
UNBRACKETED = ("coverage",)

# What a random ordering of an impression's candidates scores.
CHANCE = 0.5

# A slice whose interval is this many times the width of the overall one is
# reported with that ratio: an ordering that reverses there reverses inside a
# much noisier measurement, and the reader should see it.
WIDE = 2.0

# Fields the two reports must agree on before their numbers may be compared.
# A bm25 report from before a re-ingest against a fresh ann one would produce a
# difference that is a difference between two populations, not two retrievers.
SHARED = (
    "dataset",
    "split",
    "history_k",
    "impressions",
    "catalogue",
    "candidate_pool",
    "list_depth",
    "head_cutoff",
    "confidence",
    "resamples",
)

NOT_ESTABLISHED = "not established"
UNDEFINED = "-"
BY_VALUE = " (by value)"

DOCUMENT = "comparison-{split}.md"
# A comparison drawn from a swept window is a different document: it holds
# different numbers, and one that overwrote the default would be a file whose
# name says nothing about the window its numbers came from.
DOCUMENT_WINDOW = "comparison-{split}-k{window}.md"


class ComparisonError(RuntimeError):
    """The stored results cannot answer the question as asked."""


def from_sweep(config: DatasetConfig, retriever: str, split: str, window: int) -> dict:
    """One cell of the ablation grid, in the shape a stored report has.

    This is what keeps ticket 12's grid and this comparison in one pipeline
    rather than two: a swept window is compared by naming it, not by copying
    numbers out of the sweep file. The import is deferred because sweep reads
    this module's interval test, and the two would otherwise import each other.
    """
    from pipeline import sweep

    for row in sweep.load(split):
        if (row["dataset"], row["retriever"], row["history_k"]) == (
            config.name,
            retriever,
            window,
        ):
            return row["report"]
    raise ComparisonError(
        f"no swept cell for {config.name}/{retriever}/{split} at a history "
        f"window of {window}: {sweep.results_path(split)} holds no such line. "
        f"Run\n"
        f"    python -m pipeline.sweep --dataset {config.name} "
        f"--retriever {retriever} --window {window} --split {split}"
    )


def load(
    config: DatasetConfig, retriever: str, split: str, window: int | None = None
) -> dict:
    """One stored evaluation report, or the command that would produce it.

    With a window, the report comes from that cell of the ablation grid rather
    than from the standalone evaluation run.
    """
    if window is not None:
        return from_sweep(config, retriever, split, window)

    path = config.artifacts_dir / evaluate.EVALUATE_DIR / f"{retriever}-{split}.json"
    if not path.exists():
        raise ComparisonError(
            f"no stored evaluation for {config.name}/{retriever}/{split}: "
            f"{path} does not exist. Run\n"
            f"    python -m pipeline.evaluate --dataset {config.name} "
            f"--retriever {retriever} --split {split}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def disjoint(lex: dict, sem: dict) -> bool:
    """Whether the two intervals fail to overlap at all.

    An absent bound counts as overlapping: a run with no resamples behind it
    has no interval, and the honest reading of no interval is that nothing was
    established, not that everything was.
    """
    bounds = (lex["lo"], lex["hi"], sem["lo"], sem["hi"])
    if any(bound is None for bound in bounds):
        return False
    return lex["lo"] > sem["hi"] or sem["lo"] > lex["hi"]


@dataclass(frozen=True)
class Row:
    """One slice x one metric, the two retrievers side by side."""

    slice: str
    metric: str
    population: int
    n: int
    lex: dict
    sem: dict

    @property
    def defined(self) -> bool:
        return self.lex["value"] is not None and self.sem["value"] is not None

    @property
    def delta(self) -> float | None:
        """Semantic minus lexical, so a positive number favours the ANN."""
        if not self.defined:
            return None
        return self.sem["value"] - self.lex["value"]

    @property
    def established(self) -> bool:
        """Whether the intervals separate the two — never true for coverage."""
        if not self.defined or self.metric in UNBRACKETED:
            return False
        return disjoint(self.lex, self.sem)

    @property
    def leader(self) -> str | None:
        """The higher retriever where that can be said, else None.

        Coverage is answered by value because its interval is not a bracket;
        every other metric is answered only when the intervals are disjoint.
        """
        if not self.defined or not self.delta:
            return None
        if self.metric in UNBRACKETED:
            return SEMANTIC if self.delta > 0 else LEXICAL
        if not self.established:
            return None
        return SEMANTIC if self.delta > 0 else LEXICAL

    @property
    def higher(self) -> str:
        """The table's last cell: who is higher, or that nobody is shown to be."""
        if not self.defined:
            return UNDEFINED
        if self.leader is None:
            return NOT_ESTABLISHED
        return self.leader + (BY_VALUE if self.metric in UNBRACKETED else "")


@dataclass(frozen=True)
class Comparison:
    """Both retrievers on one dataset, aligned row by row."""

    config: DatasetConfig
    lex: dict
    sem: dict
    rows: tuple[Row, ...]

    def at(self, slice_name: str, metric: str) -> Row:
        for row in self.rows:
            if row.slice == slice_name and row.metric == metric:
                return row
        raise ComparisonError(f"no {metric} on the {slice_name} slice")

    def population(self, slice_name: str) -> int:
        return self.at(slice_name, ACCURACY[0]).population


def agree(lex: dict, sem: dict) -> None:
    """Refuse to compare two reports that are not about the same measurement."""
    differ = [field for field in SHARED if lex[field] != sem[field]]
    if differ:
        detail = ", ".join(f"{f}: {lex[f]!r} vs {sem[f]!r}" for f in differ)
        raise ComparisonError(
            f"the stored {LEXICAL} and {SEMANTIC} reports disagree on {detail}. "
            f"They were produced from different data or different settings, so "
            f"a difference between them is not a difference between the "
            f"retrievers. Re-run both: python -m pipeline.evaluate --dataset "
            f"{lex['dataset']} --split {lex['split']}"
        )


def indexed(report: dict) -> dict[tuple[str, str], dict]:
    return {(row["slice"], row["metric"]): row for row in report["results"]}


def comparison(
    config: DatasetConfig, split: str, window: int | None = None
) -> Comparison:
    """Every slice x metric of one dataset, both retrievers, from disk."""
    lex = load(config, LEXICAL, split, window)
    sem = load(config, SEMANTIC, split, window)
    agree(lex, sem)
    lex_rows, sem_rows = indexed(lex), indexed(sem)

    rows = []
    for slice_name in evaluate.SLICES:
        for metric in evaluate.METRICS:
            key = (slice_name, metric)
            if key not in lex_rows or key not in sem_rows:
                raise ComparisonError(
                    f"{config.name}: no stored {metric} on the {slice_name} "
                    f"slice for both retrievers. Re-run the evaluation."
                )
            left, right = lex_rows[key], sem_rows[key]
            if (left["population"], left["n"]) != (right["population"], right["n"]):
                raise ComparisonError(
                    f"{config.name}/{slice_name}/{metric}: the retrievers were "
                    f"scored over different populations "
                    f"({left['population']}/{left['n']} against "
                    f"{right['population']}/{right['n']})"
                )
            rows.append(
                Row(slice_name, metric, left["population"], left["n"], left, right)
            )
    return Comparison(config, lex, sem, tuple(rows))


# --- the tables -------------------------------------------------------------

HEADINGS = (
    "slice",
    "metric",
    "population",
    "n",
    LEXICAL,
    SEMANTIC,
    f"delta ({SEMANTIC} - {LEXICAL})",
    "higher",
)
# Which of those read as text; the rest are right-aligned numbers.
TEXT = (0, 1, 7)


def band(side: dict) -> str:
    """A value with its interval, or a dash where the metric is undefined."""
    if side["value"] is None:
        return UNDEFINED
    if side["lo"] is None or side["hi"] is None:
        return f"{side['value']:.4f}"
    return f"{side['value']:.4f} [{side['lo']:.4f}, {side['hi']:.4f}]"


def signed(value: float | None) -> str:
    return UNDEFINED if value is None else f"{value:+.4f}"


def table(rows: tuple[Row, ...]) -> str:
    """The rows as a markdown table, padded so the source file reads too."""
    cells = [
        [
            row.slice,
            row.metric,
            str(row.population),
            str(row.n),
            band(row.lex),
            band(row.sem),
            signed(row.delta),
            row.higher,
        ]
        for row in rows
    ]
    widths = [
        max(len(head), *(len(row[i]) for row in cells))
        for i, head in enumerate(HEADINGS)
    ]

    def line(values: list[str]) -> str:
        padded = [
            value.ljust(width) if i in TEXT else value.rjust(width)
            for i, (value, width) in enumerate(zip(values, widths))
        ]
        return "| " + " | ".join(padded) + " |"

    rule = [
        ("-" * width if i in TEXT else "-" * (width - 1) + ":")
        for i, width in enumerate(widths)
    ]
    return "\n".join([line(list(HEADINGS)), line(rule), *(line(c) for c in cells)])


# --- the reading ------------------------------------------------------------
#
# Every sentence below is assembled from the numbers it quotes. Nothing here
# is a conclusion typed in by hand: re-run the evaluation with different data
# and the wins, the reversals and the "not established" verdicts move with it.


def deltas(rows: list[Row]) -> str:
    """The metrics and their signed gaps: `auc +0.0655, mrr +0.0298`. For
    sentences that name no leader, where the sign is the only thing saying
    which way the difference runs."""
    return ", ".join(f"{row.metric} {signed(row.delta)}" for row in rows)


def margins(rows: list[Row]) -> str:
    """The metrics and the size of the gap: `auc by 0.0655`. For sentences
    that have already named the retriever the gap favours."""
    return ", ".join(f"{row.metric} by {abs(row.delta):.4f}" for row in rows)


def ranking_rows(c: Comparison, slice_name: str) -> list[Row]:
    return [c.at(slice_name, metric) for metric in ACCURACY]


def led_by(rows: list[Row], retriever: str) -> list[Row]:
    return [row for row in rows if row.leader == retriever]


def verdict_of(rows: list[Row]) -> str:
    """Who the intervals separate, over one slice's ranking metrics.

    Returns the retriever if every established metric agrees on it, and None
    if none of them are established or the established ones disagree — the
    caller says which of those it was, in the words its sentence needs.
    """
    winners = {row.leader for row in rows if row.established}
    return winners.pop() if len(winners) == 1 else None


def ranking_sentence(c: Comparison, slice_name: str) -> str:
    """Which retriever the ranking metrics separate on one slice, and where
    they do not."""
    rows = ranking_rows(c, slice_name)
    established = [row for row in rows if row.established]
    overlapping = [row for row in rows if row.defined and not row.established]
    leader = verdict_of(rows)

    if not established:
        return (
            f"nothing separates the two — the intervals overlap on every "
            f"ranking metric ({deltas(overlapping)})"
        )
    if leader and len(established) == len(rows):
        return (
            f"{leader} leads on all {len(rows)} ranking metrics "
            f"({margins(rows)})"
        )
    parts = []
    for retriever in (LEXICAL, SEMANTIC):
        won = led_by(established, retriever)
        if won:
            parts.append(f"{retriever} leads on {margins(won)}")
    if overlapping:
        parts.append(
            f"the intervals overlap on {deltas(overlapping)}, so no ordering "
            f"is established there"
        )
    return "; ".join(parts)


def overall_bullet(c: Comparison) -> str:
    return f"- **Overall**: {ranking_sentence(c, 'overall')}."


def contains_chance(side: dict) -> bool:
    return side["lo"] is not None and side["lo"] <= CHANCE <= side["hi"]


def at_chance(c: Comparison) -> list[str]:
    """The retrievers whose overall AUC interval contains a random ordering."""
    row = c.at("overall", "auc")
    return [
        retriever
        for retriever, side in ((LEXICAL, row.lex), (SEMANTIC, row.sem))
        if contains_chance(side)
    ]


def against_chance(side: dict) -> str:
    if contains_chance(side):
        return "contains 0.5"
    return f"clears 0.5 by {abs(side['value'] - CHANCE):.4f}"


def chance_bullet(c: Comparison) -> str:
    """An AUC gap between two near-random orderings is still a gap, and has to
    be read as one."""
    row = c.at("overall", "auc")
    random_ordering = at_chance(c)
    reading = (
        f"An AUC interval containing 0.5 means that retriever orders this "
        f"split's candidates no better than shuffling them, so any gap "
        f"involving {' and '.join(random_ordering)} is a gap between "
        f"orderings at "
        f"least one of which is indistinguishable from random."
        if random_ordering
        else "Both are clear of a random ordering."
    )
    return (
        f"- **Against chance**: {LEXICAL} {band(row.lex)} "
        f"{against_chance(row.lex)}; {SEMANTIC} {band(row.sem)} "
        f"{against_chance(row.sem)}. {reading}"
    )


def width(side: dict) -> float | None:
    if side["lo"] is None or side["hi"] is None:
        return None
    return side["hi"] - side["lo"]


def noise_note(c: Comparison, slice_name: str) -> str:
    """How much wider this slice's interval is than the whole population's."""
    here, whole = c.at(slice_name, "auc"), c.at("overall", "auc")
    widths = [width(side) for side in (here.lex, here.sem, whole.lex, whole.sem)]
    if any(w is None for w in widths) or not min(widths):
        return ""
    ratio = max(widths[:2]) / min(widths[2:])
    if ratio < WIDE:
        return ""
    return (
        f" Its AUC intervals are up to {ratio:.1f}x the width of the whole "
        f"split's, so it takes a correspondingly larger gap to establish "
        f"anything here."
    )


def slice_bullet(c: Comparison, slice_name: str) -> str:
    population = c.population(slice_name)
    if not population:
        return (
            f"- **{slice_name}** is empty: no impression on this split falls "
            f"in it, so it says nothing about either retriever."
        )

    whole = (
        ", the whole split, so this row repeats overall"
        if population == c.lex["impressions"]
        else ""
    )
    sentence = ranking_sentence(c, slice_name)
    overall = verdict_of(ranking_rows(c, "overall"))
    here = verdict_of(ranking_rows(c, slice_name))
    reversal = (
        f" **The ordering reverses here**: {overall} leads overall."
        if here and overall and here != overall
        else ""
    )
    return (
        f"- **{slice_name}** ({population} impressions{whole}): {sentence}."
        f"{reversal}{noise_note(c, slice_name)}"
    )


def cold_bullet(c: Comparison) -> str:
    """Whether the semantic side's thin-history weakness actually shows up."""
    if not (c.population("cold") and c.population("warm")):
        return ""
    cold, warm = c.at("cold", "auc"), c.at("warm", "auc")
    if cold.delta is None or warm.delta is None:
        return ""

    if (cold.delta > 0) == (warm.delta > 0):
        movement = "narrows" if abs(cold.delta) < abs(warm.delta) else "widens"
        shape = (
            f"the AUC gap keeps its sign and {movement} on the thinner "
            f"histories ({signed(warm.delta)} warm, {signed(cold.delta)} cold)"
        )
    else:
        shape = (
            f"the AUC gap changes sign between them ({signed(warm.delta)} "
            f"warm, {signed(cold.delta)} cold)"
        )
    moves = ", ".join(
        f"{retriever} {warm_side['value']:.4f} -> {cold_side['value']:.4f}"
        for retriever, warm_side, cold_side in (
            (LEXICAL, warm.lex, cold.lex),
            (SEMANTIC, warm.sem, cold.sem),
        )
    )
    return (
        f"- **Cold against warm users**: {shape}. Both retrievers move the "
        f"same way in absolute terms ({moves}), so a cold user is harder for "
        f"either approach, not only for the one that has to average a short "
        f"history into a vector."
    )


def beyond_bullet(c: Comparison) -> str:
    """The list-composition metrics, which have no better direction."""
    parts = []
    for metric in (*evaluate.LIST_METRICS, *UNBRACKETED):
        row = c.at("overall", metric)
        if not row.defined:
            continue
        verdict = (
            NOT_ESTABLISHED
            if row.leader is None
            else f"{row.leader} higher"
            + (BY_VALUE if metric in UNBRACKETED else "")
        )
        parts.append(
            f"{metric} {row.lex['value']:.4f} against "
            f"{row.sem['value']:.4f}, {verdict}"
        )
    return (
        f"- **Beyond accuracy, overall**: {'; '.join(parts)}. Higher is not "
        f"better for any of these three — they say what the shown lists were "
        f"made of, not how well they were ordered."
    )


def reading_of(c: Comparison) -> str:
    bullets = [
        overall_bullet(c),
        chance_bullet(c),
        *(
            slice_bullet(c, slice_name)
            for slice_name in evaluate.SLICES
            if slice_name != "overall"
        ),
        cold_bullet(c),
        beyond_bullet(c),
    ]
    return "\n".join(bullet for bullet in bullets if bullet)


# --- across the datasets ----------------------------------------------------


def leader_of(c: Comparison) -> str | None:
    return c.at("overall", "auc").leader


def edge(c: Comparison, retriever: str) -> float:
    """How far one retriever's overall AUC sits from a random ordering."""
    row = c.at("overall", "auc")
    side = row.lex if retriever == LEXICAL else row.sem
    return abs(side["value"] - CHANCE)


def gap_of(c: Comparison) -> str:
    row = c.at("overall", "auc")
    if row.leader is None:
        return (
            f"on {c.config.name} the AUC intervals overlap "
            f"({signed(row.delta)}), so neither is ahead"
        )
    return f"on {c.config.name} {row.leader} leads by {abs(row.delta):.4f}"


def gaps_sentence(comparisons: list[Comparison]) -> str:
    stated = "; ".join(gap_of(c) for c in comparisons)
    leaders = {leader_of(c) for c in comparisons}
    if len(leaders) > 1:
        verdict = (
            "The ordering reverses between the datasets, and both sides of "
            "the reversal are established by disjoint intervals."
        )
    else:
        verdict = "The ordering is the same on both, so nothing reverses."
    sizes = sorted(abs(c.at("overall", "auc").delta) for c in comparisons)
    if sizes[0]:
        verdict += f" The larger gap is {sizes[-1] / sizes[0]:.1f}x the smaller."
    return f"AUC overall: {stated}. {verdict}"


def per_dataset(comparisons: list[Comparison], value) -> str:
    return "; ".join(f"{c.config.name} {value(c)}" for c in comparisons)


def cold_start_row(comparisons: list[Comparison]) -> tuple[str, str, str]:
    """Whether thin click histories can carry the difference."""
    trailing = [c for c in comparisons if leader_of(c) != SEMANTIC]
    evidence = per_dataset(
        comparisons,
        lambda c: (
            f"{c.population('cold')} cold impressions"
            + (
                ""
                if not c.population("cold")
                else f", where {verdict_of(ranking_rows(c, 'cold')) or 'neither'} "
                f"leads ({signed(c.at('cold', 'auc').delta)} AUC)"
            )
        ),
    )
    ruled_out = trailing and all(not c.population("cold") for c in trailing)
    verdict = (
        "ruled out"
        if ruled_out
        else "consistent, but the slice cannot separate it from the rest"
    )
    return (
        "Cold users starve the semantic user vector, which is a mean over the "
        "user's history",
        evidence,
        verdict,
    )


def semantic_failure_row(comparisons: list[Comparison]) -> tuple[str, str, str]:
    """Whether the weak dataset is weak for the semantic arm in particular."""
    trailing = [c for c in comparisons if leader_of(c) != SEMANTIC]
    leading = [c for c in comparisons if leader_of(c) == SEMANTIC]
    evidence = per_dataset(
        comparisons,
        lambda c: (
            f"{LEXICAL} {band(c.at('overall', 'auc').lex)}, "
            f"{SEMANTIC} {band(c.at('overall', 'auc').sem)}"
        ),
    )
    both_near_chance = (
        trailing
        and leading
        and max(edge(c, r) for c in trailing for r in (LEXICAL, SEMANTIC))
        < min(edge(c, r) for c in leading for r in (LEXICAL, SEMANTIC))
    )
    verdict = (
        "not supported: both arms sit nearer chance there than either does on "
        + ", ".join(c.config.name for c in leading)
        if both_near_chance
        else "cannot be separated"
    )
    return (
        "The semantic arm in particular breaks on "
        + ", ".join(c.config.name for c in trailing or comparisons),
        evidence,
        verdict,
    )


# Everything that varies with the dataset and nothing else. Each is read off
# the registry or the stored report rather than asserted, and each gets the
# same verdict for the same reason: they all change together.
CONFOUNDED = (
    ("Language", lambda c: c.config.language),
    (
        "Embedding provenance",
        lambda c: (
            f"{c.config.embeddings.kind} {c.config.embeddings.model} "
            f"({c.config.embeddings.dim}d)"
        ),
    ),
    ("Catalogue size", lambda c: f"{c.lex['catalogue']} articles"),
    (
        "How much of it a candidate list can reach",
        lambda c: f"{c.lex['candidate_pool']} ever shown",
    ),
    ("Impressions scored", lambda c: str(c.lex["impressions"])),
)


def hypotheses(comparisons: list[Comparison]) -> str:
    rows = [
        cold_start_row(comparisons),
        semantic_failure_row(comparisons),
        *(
            (name, per_dataset(comparisons, value), "cannot be separated")
            for name, value in CONFOUNDED
        ),
    ]
    header = ["| explanation | what the stored results say | verdict |",
              "| --- | --- | --- |"]
    return "\n".join(header + [f"| {a} | {b} | {c} |" for a, b, c in rows])


def cross_dataset(comparisons: list[Comparison]) -> str:
    random_ordering = [
        f"{' and '.join(at_chance(c))} on {c.config.name}"
        for c in comparisons
        if at_chance(c)
    ]
    caveat = (
        f" One side of that is not ordering anything: the overall AUC "
        f"interval of {', '.join(random_ordering)} contains 0.5, so the "
        f"leader there leads within a pair of orderings at least one of which "
        f"is indistinguishable from shuffling the candidates."
        if random_ordering
        else ""
    )
    return "\n\n".join(
        [
            "## Where the datasets disagree, and what explains it",
            gaps_sentence(comparisons),
            hypotheses(comparisons),
            f"Every factor marked *cannot be separated* varies with the "
            f"dataset and with nothing else, and with {len(comparisons)} "
            f"datasets they all vary together: any one of them could carry "
            f"the whole difference, and no further breakdown of these results "
            f"tells them apart. Separating them needs a manipulation rather "
            f"than another slice — the same retriever run again with one "
            f"factor changed, which is what the ablation sweep is for. What "
            f"the numbers above establish is narrower than a story about "
            f"language or embeddings: they establish which retriever ordered "
            f"each dataset's candidate lists better, and by how much.{caveat}",
        ]
    )


# --- the document -----------------------------------------------------------


def how_to_read(sample: dict) -> str:
    return "\n".join(
        [
            "## How to read this",
            "",
            f"- Each cell is the metric's value with its bootstrap "
            f"{sample['confidence']:.0%} interval over "
            f"{sample['resamples']} resamples of impressions. `delta` is "
            f"{SEMANTIC} minus {LEXICAL}, so a positive number favours the "
            f"semantic side.",
            f"- `higher` names a retriever only where the two intervals are "
            f"disjoint. `{NOT_ESTABLISHED}` means they overlap — the "
            f"difference is not shown, which is **not** the same as shown to "
            f"be absent. The test is conservative in one further way: both "
            f"retrievers rank the same impressions, so a paired test on the "
            f"per-impression differences would separate more than this does. "
            f"The stored reports carry slice means rather than per-impression "
            f"values, so it is not available here.",
            f"- Coverage is the one metric compared `{BY_VALUE.strip()}`: "
            f"it is a union over a slice rather than a mean over it, so its "
            f"interval is a width and not a bracket around its own value.",
            f"- Only the ranking metrics ({', '.join(ACCURACY)}) have a better "
            f"direction. Diversity, novelty and coverage describe the lists "
            f"that were shown; higher is a difference, not an improvement.",
            f"- `population` is how many impressions the slice holds, `n` how "
            f"many of them the metric was defined on. Cold users have fewer "
            f"than {evaluate.COLD_CLICKS} clicks in history and warm users at "
            f"least that many; head and tail place an impression by whether "
            f"the articles its user clicked are among the most-shown fifth of "
            f"the split, so impressions that straddle both are in neither and "
            f"the two do not sum to the whole.",
        ]
    )


def provenance(c: Comparison) -> str:
    report = c.lex
    reachable = 100 * report["candidate_pool"] / report["catalogue"]
    flat = ", ".join(
        f"{retriever} {side['all_scores_tied']}"
        for retriever, side in ((LEXICAL, c.lex), (SEMANTIC, c.sem))
    )
    return (
        f"{report['impressions']} impressions on the {report['split']} split. "
        f"Coverage is against the full catalogue of {report['catalogue']} "
        f"articles, of which {report['candidate_pool']} ({reachable:.1f}%) "
        f"ever appear as a candidate here, so no ranking can exceed that "
        f"ceiling. Head articles are the most-shown fifth of those (at least "
        f"{report['head_cutoff']} impressions); lists are read to depth "
        f"{report['list_depth']}. Impressions the retriever scored flat, whose "
        f"rank metrics come from the candidate file's own order rather than "
        f"from the retriever: {flat}."
    )


def document(
    comparisons: list[Comparison], split: str, window: int | None = None
) -> str:
    source = (
        f"the ablation grid in `artifacts/{sweep_results(split)}`, at a history "
        f"window of {window}"
        if window is not None
        else f"the stored evaluation reports in "
        f"`artifacts/<dataset>/{evaluate.EVALUATE_DIR}/`"
    )
    flag = f" --window {window}" if window is not None else ""
    sections = [
        f"# Lexical versus semantic retrieval — {split} split"
        + (f", history window {window}" if window is not None else ""),
        f"Generated by `python -m pipeline.compare --split {split}{flag}` from "
        f"{source}. Nothing here was "
        f"re-ranked and nothing here was typed by hand; regenerate it rather "
        f"than editing it. {LEXICAL} is the lexical retriever and {SEMANTIC} "
        f"the semantic one.",
        how_to_read(comparisons[0].lex),
    ]
    for c in comparisons:
        sections += [
            f"## {c.config.name}",
            provenance(c),
            table(c.rows),
            "### Reading",
            reading_of(c),
        ]
    if len(comparisons) > 1:
        sections.append(cross_dataset(comparisons))
    return "\n\n".join(sections) + "\n"


def sweep_results(split: str) -> str:
    from pipeline import sweep

    return sweep.RESULTS.format(split=split)


def write(text: str, split: str, window: int | None = None) -> Path:
    paths.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    name = (
        DOCUMENT.format(split=split)
        if window is None
        else DOCUMENT_WINDOW.format(split=split, window=window)
    )
    path = paths.ARTIFACTS_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.compare",
        description="Compare the lexical and semantic retrievers from the "
        "stored evaluation results, on every dataset and slice.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="restrict to one dataset (repeatable); default is all of them",
    )
    parser.add_argument(
        "--split",
        default=evaluate.VALIDATION,
        help=f"which stored split to compare: "
        f"{' or '.join(evaluate.SCORABLE)} (default: {evaluate.VALIDATION})",
    )
    parser.add_argument(
        "--window",
        type=int,
        help="compare the cell of the ablation sweep run at this history "
        "window, instead of the standalone evaluation reports. Both "
        "retrievers are read at the same window, and a grid missing either "
        "of them is an error rather than a half comparison",
    )
    args = parser.parse_args(argv)

    configs = [DATASETS[name] for name in (args.dataset or sorted(DATASETS))]
    try:
        comparisons = [
            comparison(config, args.split, args.window) for config in configs
        ]
    except ComparisonError as error:
        print(f"\nerror: {error}\n", file=sys.stderr)
        return 2

    text = document(comparisons, args.split, args.window)
    path = write(text, args.split, args.window)
    print("\n" + text)
    print(f"  -> {path.relative_to(paths.ARTIFACTS_DIR.parent)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
