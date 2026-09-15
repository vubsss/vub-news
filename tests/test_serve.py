"""The serving benchmark is tested where a latency number can be wrong while
looking right: a stage timer that attributes work to the wrong stage, a byte
table that reports a missing store as free, a request path that loads something
per request and calls the loading latency, and a cost that is quoted for a
variant which never met the budget it was priced at."""

import numpy as np
import pandas as pd
import pytest

from pipeline import features, ledger, paths, rerank, serve

from small_store import (  # noqa: F401 -- `store` and `small` are fixtures
    MIND,
    SMALL_RERANK,
    small,
    store,
    write_store,
)


@pytest.fixture
def served(store, small):
    """A whole pipeline on disk, and one request path built over it."""
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    return serve.build(MIND, k=3)


def requests_of(count=4):
    return serve.requests_from(MIND, "validation", count)


# --- the per-stage timer ----------------------------------------------------


def test_a_request_is_timed_in_four_stages_that_make_up_its_total(served):
    timed = serve.one(served, requests_of(1)[0])

    assert set(timed) == {*serve.STAGES, "total"}
    assert all(timed[stage] > 0 for stage in serve.STAGES)
    # The total is the four, by construction rather than by a second clock:
    # two clocks around one request is how a breakdown starts disagreeing with
    # the thing it breaks down.
    assert timed["total"] == pytest.approx(sum(timed[s] for s in serve.STAGES))


def test_the_summary_reports_what_the_stage_medians_leave_unattributed(served):
    summary = serve.measure(served, requests_of(4), warmup=1)

    assert summary["requests"] == 3
    for stage in (*serve.STAGES, "total"):
        assert summary[f"{stage}_p50_ms"] <= summary[f"{stage}_p99_ms"]
    # Medians are not additive, so this is a check on the timer rather than an
    # identity: it exists so a request spending time outside all four stages
    # shows up as a number instead of as nothing.
    assert "unattributed_ms" in summary
    assert summary["stage_median_sum_ms"] > 0


def test_every_request_is_measured_except_the_warm_ups(served):
    assert serve.measure(served, requests_of(4), warmup=3)["requests"] == 1
    with pytest.raises(serve.ServeError, match="warm-up"):
        serve.measure(served, requests_of(2), warmup=2)


def test_the_request_path_opens_no_file(served, monkeypatch):
    """Everything a request touches is in the bytes table, which is only true
    if the request opens nothing. A benchmark that loaded an index inside the
    loop would report startup as latency -- A1's `bench_serve` had to measure
    two sample sizes precisely to subtract that, and this path is built so
    there is nothing to subtract."""
    chunk = requests_of(1)[0]
    serve.one(served, chunk)  # warm up whatever is lazy

    opened = []
    real = open

    def watched(file, *arguments, **keywords):
        opened.append(str(file))
        return real(file, *arguments, **keywords)

    monkeypatch.setattr("builtins.open", watched)
    serve.one(served, chunk)

    assert not opened, f"the request path opened {opened}"


# --- stage one --------------------------------------------------------------


def test_a_cut_request_retrieves_from_the_catalogue_not_the_logged_list(served):
    """The difference between a served request and a harness one. A server has
    a user and a catalogue; it does not have the list of candidates the log
    happens to record, and scoring that list would be measuring the harness."""
    chunk = requests_of(1)[0]
    logged = set(chunk["candidate_ids"].iloc[0])
    found = serve.candidates_for(served, chunk)

    assert len(found) == 3
    assert set(found) - logged, "every candidate came from the logged list"


def test_the_no_cut_path_scores_exactly_what_the_log_recorded(store, small):
    """The curve's ceiling: no cut at all, which is what the harness reports
    and what every AUC in the note was measured on."""
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    uncut = serve.build(MIND, k=None)
    chunk = requests_of(1)[0]

    assert serve.candidates_for(uncut, chunk) == list(chunk["candidate_ids"].iloc[0])


def test_an_index_it_does_not_have_is_refused():
    with pytest.raises(serve.ServeError, match="no 'annoy' index"):
        serve.index_for(np.eye(4, dtype="float32"), "annoy")


@pytest.mark.parametrize("kind", ["flat", "ivf", "hnsw"])
def test_every_index_kind_answers_the_same_shape(store, small, kind):
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    built = serve.build(MIND, kind=kind, k=2)

    found = serve.candidates_for(built, requests_of(1)[0])
    assert len(found) == 2


def test_the_served_scorers_agree_with_the_harness(served):
    """The reason `rank_candidates` grew an `index` argument instead of this
    module growing its own scoring. Hoisting the load out of the call must not
    change a single score -- if it did, every retriever-score feature a served
    request sees would differ from the one the model was trained on."""
    from pipeline import ann_index, bm25_index

    chunk = requests_of(1)[0]
    for name, module in (("ann", ann_index), ("bm25", bm25_index)):
        hoisted = served.scorers[name](chunk, chunk)
        loaded = module.rank_candidates(MIND, chunk, chunk, served.history_k)

        assert list(hoisted["impression_id"]) == list(loaded["impression_id"])
        for mine, theirs in zip(hoisted["scores"], loaded["scores"]):
            assert list(mine) == pytest.approx(list(theirs))


# --- the bytes --------------------------------------------------------------


def test_a_store_that_is_not_built_reads_as_unknown_rather_than_free(store, small):
    """A zero in the byte table is a store that costs nothing. A store that was
    never built costs nothing *here*, which is a different sentence, and the
    sum of a column containing a false zero is a number nobody can check."""
    counted = serve.store_bytes(MIND, SMALL_RERANK)

    assert counted["index_bytes"] is None  # no index handed in
    assert counted["model_bytes"] is None  # nothing trained yet
    assert counted["feature_bytes"] is None


def test_every_store_a_request_reads_is_counted(served):
    counted = serve.store_bytes(MIND, SMALL_RERANK, served.index)
    resident = serve.resident_bytes(served)

    assert counted["index_bytes"] > 0
    assert counted["model_bytes"] > 0
    assert counted["nrms_bytes"] > 0
    assert counted["counter_bytes"] > 0
    assert resident["counters_resident"] > 0
    assert resident["vectors_resident"] > 0


def test_the_user_cache_is_only_charged_where_the_arm_keeps_one(store, small):
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    plain, cached = serve.build(MIND, k=3), serve.build(MIND, k=3, cache_users=True)
    chunk = requests_of(1)[0]

    serve.one(plain, chunk)
    serve.one(cached, chunk)

    assert plain.cache_bytes() == 0
    assert cached.cache_bytes() > 0
    assert len(cached.cache) == 1


def test_a_cached_user_is_encoded_once_and_scored_twice(store, small):
    """What the cache actually removes. The user encoder runs per request and a
    hit skips it; the candidate dot runs per candidate and a hit does not."""
    write_store()
    rerank.fit_one(MIND, SMALL_RERANK)
    cached = serve.build(MIND, k=3, cache_users=True)
    chunk = requests_of(1)[0]

    first = serve._nrms_scores(cached, chunk, ["a1", "a2"])
    second = serve._nrms_scores(cached, chunk, ["a1", "a2"])

    assert len(cached.cache) == 1
    assert first == pytest.approx(second)


# --- the cost arithmetic ----------------------------------------------------


def test_the_cost_shows_the_division_it_made():
    priced = serve.cost_per_1000(p99_ms=40.0, qps_per_core=50.0)

    assert priced["meets_sla"]
    assert priced["core_seconds_per_1000"] == 20.0
    assert priced["usd_per_1000"] == pytest.approx(20 / 3600 * serve.CORE_HOUR_USD)
    assert "1000 queries / 50.00 q/s" in priced["arithmetic"]


def test_a_variant_that_misses_the_budget_is_not_priced():
    """Pricing it would put a cheap row beside a compliant one without saying
    the cheap one is not allowed to be used."""
    priced = serve.cost_per_1000(p99_ms=250.0, qps_per_core=8.0)

    assert not priced["meets_sla"]
    assert priced["usd_per_1000"] is None
    assert "not priced" in priced["arithmetic"]


def test_throughput_comes_from_the_median_not_the_mean():
    assert serve.qps_per_core({"total_p50_ms": 25.0}) == 40.0


# --- the 10x argument -------------------------------------------------------


BYTES = {
    "index_resident": 200_000_000,
    "vectors_resident": 190_000_000,
    "counters_resident": 50_000_000,
    "cache_bytes_per_user": 3_072,
}


def test_the_three_axes_grow_different_stores():
    rows = {row["axis"]: row for row in serve.scaling(BYTES, 40.0, users=50_000)}

    assert set(rows) == {"10x users", "10x catalogue", "10x QPS"}
    # Ten times the readers does not make more articles, so the article-keyed
    # stores are the same size on that row as on the QPS one.
    assert rows["10x users"]["gb_after"] > rows["10x QPS"]["gb_after"]
    assert rows["10x catalogue"]["gb_after"] > rows["10x users"]["gb_after"]
    # And rate keys no store at all: only the core count moves.
    assert rows["10x QPS"]["cores_after"] == 10 * rows["10x users"]["cores_after"]


def test_the_row_that_exceeds_the_node_says_so():
    rows = serve.scaling(BYTES, qps_per_core=40.0, users=50_000, node_cores=32)
    qps = next(row for row in rows if row["axis"] == "10x QPS")

    # 1000 q/s at 40 per core is 25 cores; ten times that is 250, and the node
    # has 32.
    assert qps["cores_after"] == 250
    assert not qps["fits_cores"]


def test_a_catalogue_that_does_not_fit_is_reported_as_not_fitting():
    huge = dict(BYTES, index_resident=20_000_000_000)
    rows = serve.scaling(huge, 40.0, users=1_000, node_ram_gb=100)
    catalogue = next(row for row in rows if row["axis"] == "10x catalogue")

    assert catalogue["gb_after"] > 100
    assert not catalogue["fits_ram"]


# --- the K curve ------------------------------------------------------------


def test_the_k_curve_reads_its_auc_from_the_ablations_row(store, monkeypatch):
    """The AUC belongs to ticket 07, which scored it on validation against the
    full model. Recomputing it here would be a second chance for two tables to
    disagree about one configuration."""
    ledger.record(
        {
            "dataset": "mind",
            "stage": "ablation",
            "variant": "cut@100",
            "split": "validation",
            "auc": 0.6123,
            "auc_lo": 0.61,
            "auc_hi": 0.615,
        }
    )
    curve = serve.k_curve(
        MIND,
        [
            {"k": 100, "total_p50_ms": 12.0, "total_p99_ms": 30.0},
            {"k": 200, "total_p50_ms": 15.0, "total_p99_ms": 38.0},
        ],
    )

    assert [point["k"] for point in curve] == [100, 200]
    assert curve[0]["auc"] == 0.6123
    # No arm for K=200 yet: outstanding, not zero.
    assert curve[1]["auc"] is None


def test_the_no_cut_row_is_not_a_point_on_the_k_curve(store):
    """It is the ceiling, and it has no K to plot against."""
    curve = serve.k_curve(MIND, [{"k": None, "total_p50_ms": 9.0, "total_p99_ms": 20.0}])
    assert curve == []


# --- the ledger and the markdown --------------------------------------------


def test_a_variant_records_its_engineering_columns_and_no_auc(served, store):
    row = serve.run_variant(
        MIND,
        {"name": "flat-fp32-k3", "kind": "flat", "precision": "fp32", "k": 3},
        requests_of(3),
        "validation",
        warmup=1,
    )
    recorded = [
        entry
        for entry in ledger.load()
        if entry["stage"] == "serve" and entry["variant"] == "flat-fp32-k3"
    ]

    assert len(recorded) == 1
    assert recorded[0]["p50_ms"] > 0 and recorded[0]["p99_ms"] > 0
    # The functional column is ticket 07's and is joined in the K-curve; a
    # latency bench carrying its own AUC would be a second place for it to live.
    assert recorded[0]["auc"] is None
    assert "retrieve" in recorded[0]["note"]
    assert row["cached_users"] == 0


def test_the_markdown_says_which_numbers_are_outstanding(served, store):
    row = serve.run_variant(
        MIND,
        {"name": "flat-fp32-k3", "kind": "flat", "precision": "fp32", "k": 3},
        requests_of(3),
        "validation",
        warmup=1,
    )
    curve = serve.k_curve(MIND, [row])
    scale = serve.scaling(BYTES, row["qps_per_core"], users=3)
    written = serve.document(MIND, [row], curve, scale)

    assert "Where 10x breaks" in written
    assert "Cost per thousand queries" in written
    # Ticket 07 has not been run here, so the AUC column has to say so rather
    # than render an empty cell that reads as "no difference".
    assert "outstanding" in written.lower()
