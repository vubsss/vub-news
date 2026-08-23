"""The submission path is tested at five seams: the rank vector the leaderboard
actually reads, the guards that stand between a ranking and the file, the
competition's own impression adapter, the end-to-end write and its zip, and the
one thing the semantic side needs that the pipeline's stored artifact cannot
give it — vectors for a catalogue that artifact predates."""

import dataclasses
import zipfile

import numpy as np
import pandas as pd
import pytest

from pipeline import acquire, embed, paths, predict, sources, submissions
from pipeline.datasets import DATASETS, SubmissionSpec

MIND = DATASETS["mind"]


# The example the competition's own submission guidelines work through:
# impression 24481 over N125045 N87192 N73556 N20417, ranked N87192, N20417,
# N73556, N125045, is written `24481 [4,1,3,2]`.
GUIDELINE_CANDIDATES = ["N125045", "N87192", "N73556", "N20417"]
GUIDELINE_RANKING = ["N87192", "N20417", "N73556", "N125045"]


def test_ranks_match_the_competitions_worked_example():
    assert predict.ranks(GUIDELINE_CANDIDATES, GUIDELINE_RANKING) == [4, 1, 3, 2]


def test_the_line_is_written_as_the_leaderboard_reads_it():
    ranks = predict.ranks(GUIDELINE_CANDIDATES, GUIDELINE_RANKING)
    assert submissions.mind_line("24481", ranks) == "24481 [4,1,3,2]"


def test_a_ranking_that_is_the_candidate_order_is_the_identity():
    """What a cold user gets: every candidate scores 0, the sort is stable, and
    the file says the competition's own order rather than dropping the row."""
    assert predict.ranks(["a", "b", "c"], ["a", "b", "c"]) == [1, 2, 3]


@pytest.mark.parametrize(
    "candidates, ranked",
    [
        (["a", "b", "c"], ["a", "b"]),          # one dropped
        (["a", "b"], ["a", "b", "c"]),          # one added
        (["a", "b"], ["a", "a"]),               # one ranked twice
        (["a", "b"], ["a", "c"]),               # one swapped for a stranger
    ],
)
def test_a_ranking_that_is_not_the_candidate_set_is_refused(candidates, ranked):
    with pytest.raises(predict.SubmissionError):
        predict.ranks(candidates, ranked)


def test_a_duplicated_candidate_has_no_well_defined_ranking():
    """Not a check on the retriever but on the input: two identical candidates
    cannot both be given a place, and silently giving them the same one is a
    line the leaderboard would accept and misread."""
    with pytest.raises(predict.SubmissionError):
        predict.ranks(["a", "a"], ["a", "b"])


def test_test_impressions_parse_without_labels():
    """The competition's file carries no `-1` suffix and its history may be
    empty; both are what separates this adapter from the behaviours one."""
    raw = pd.DataFrame(
        {
            "impression_id": ["1", "2"],
            "user_id": ["U1", "U2"],
            "time": ["11/15/2019 9:00:00 AM"] * 2,
            "history": ["N1 N2", None],
            "impressions": ["N3 N4", "N5"],
        }
    )
    parsed = sources.mind_test_impressions(raw)
    assert list(parsed["candidate_ids"]) == [["N3", "N4"], ["N5"]]
    assert list(parsed["click_history"]) == [["N1", "N2"], []]


def test_the_behaviours_adapter_cannot_read_the_test_file():
    """Why a second adapter exists at all. If MIND's labelled parser ever
    starts accepting an unlabelled file, it does so by inventing labels."""
    raw = pd.DataFrame(
        {
            "impression_id": ["1"],
            "user_id": ["U1"],
            "time": ["11/15/2019 9:00:00 AM"],
            "history": ["N1"],
            "impressions": ["N3 N4"],
        }
    )
    with pytest.raises(IndexError):
        sources.mind_behaviors(raw)


NEWS = [
    ("N1", "sports", "soccer", "football league final tonight", "the final"),
    ("N2", "foodanddrink", "recipes", "chocolate cake recipe", "how to bake"),
    ("N3", "finance", "markets", "stock market rally continues", "shares rose"),
]


@pytest.fixture
def competition(tmp_path, monkeypatch):
    """A three-article competition test set on disk, where the registry says."""
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "PREDICTIONS_DIR", tmp_path / "predictions")

    test_dir = MIND.raw_dir / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "news.tsv").write_text(
        "".join(
            f"{i}\t{c}\t{s}\t{t}\t{a}\thttps://example.com\t[]\t[]\n"
            for i, c, s, t, a in NEWS
        ),
        encoding="utf-8",
    )
    (test_dir / "behaviors.tsv").write_text(
        # A user who has read about cake, then one with no history at all.
        "1\tU1\t11/15/2019 9:00:00 AM\tN2\tN1 N2 N3\n"
        "2\tU2\t11/15/2019 9:05:00 AM\t\tN3 N1\n",
        encoding="utf-8",
    )
    return tmp_path


def submitted(tmp_path) -> list[str]:
    archive = tmp_path / "predictions" / MIND.submission.bundle
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.namelist() == [MIND.submission.filename]
        return zipped.read(MIND.submission.filename).decode().splitlines()


def test_the_submission_ranks_the_candidates_it_was_given(competition, monkeypatch):
    """End to end over the lexical retriever, which needs no artifact.

    U1's only click is the cake article, so it takes first place among its own
    candidates and the other two keep the order the file listed them in. U2 has
    no history, scores every candidate 0, and still gets a line."""
    monkeypatch.setattr(acquire, "_fetch", lambda *args: pytest.fail("downloaded"))

    predict.submit(MIND, retriever="bm25")

    assert submitted(competition) == ["1 [2,1,3]", "2 [1,2]"]


def test_every_impression_is_written_once_whatever_the_chunking(competition):
    """The file is streamed, so the row count is a property of the chunk size
    until something asserts otherwise."""
    predict.submit(MIND, retriever="bm25", chunk_size=1)

    assert submitted(competition) == ["1 [2,1,3]", "2 [1,2]"]


def test_a_ranking_for_the_wrong_impressions_never_reaches_the_file(
    competition, monkeypatch
):
    """The rankings are zipped back onto the chunk positionally. A retriever
    that reordered them would produce a well-formed file that scores every
    impression against another one's candidates."""

    lexical = predict.evaluate.RETRIEVERS["bm25"]
    honest = lexical.Ranker.rank

    def reversed_rank(self, history, candidates):
        return honest(self, history, candidates).iloc[::-1].reset_index(drop=True)

    monkeypatch.setattr(lexical.Ranker, "rank", reversed_rank)

    with pytest.raises(predict.SubmissionError, match="different order"):
        predict.submit(MIND, retriever="bm25")
    assert not (competition / "predictions" / MIND.submission.bundle).exists()


def test_a_short_file_is_not_left_behind_as_a_submission(competition, monkeypatch):
    """A run that ranks fewer impressions than the competition sent has to fail
    loudly: the leaderboard reads its gold labels by row, so a file missing one
    line scores every line after it against the wrong impression."""
    monkeypatch.setattr(predict, "source_rows", lambda config: 3)

    with pytest.raises(predict.SubmissionError, match="hold 3 impressions"):
        predict.submit(MIND, retriever="bm25")
    prediction = competition / "predictions" / MIND.name / MIND.submission.filename
    assert not prediction.exists()
    assert not prediction.with_name(prediction.name + ".part").exists()


def test_a_dataset_with_no_submission_built_earns_no_checkpoint():
    """A competition described by its url and nothing else — the state both
    entries were in before their submission ticket landed, and the state a
    third dataset would be added in.

    It has to raise rather than return quietly: `build.py` writes a checkpoint
    for any stage that returns, and one written here would have every later
    rebuild skip the submission that is still to be built."""
    unbuilt = dataclasses.replace(
        MIND, submission=SubmissionSpec(competition_url=MIND.submission.competition_url)
    )
    with pytest.raises(predict.NotBuilt, match="archives"):
        predict.run(unbuilt)


def test_the_competitions_catalogue_is_encoded_where_the_artifact_is_empty(
    competition, monkeypatch
):
    """The whole reason the semantic side can submit at all.

    MIND's stored artifact covers MINDsmall, a week the competition's test
    articles are not in, so aligning it onto this catalogue leaves every row
    zero. Left there, the semantic retriever would score every candidate 0 and
    submit the candidate file's own order under its name."""
    # The width comes from the registry rather than a literal: MIND's encoder
    # has already changed once (384-wide MiniLM to 768-wide e5), and a test
    # that hard-codes the old number fails the next time it changes for a
    # reason that has nothing to do with what it is testing.
    width = MIND.embeddings.dim
    monkeypatch.setattr(
        embed,
        "read_source",
        lambda config: (np.array(["N1"], dtype=object),
                        np.eye(1, width, dtype="float32")),
    )
    encoded = []

    def fake_encode(texts, config, **kwargs):
        encoded.extend(texts)
        rows = np.zeros((len(texts), width), dtype="float32")
        rows[:, 1] = 1.0
        return rows

    monkeypatch.setattr(embed, "encode", fake_encode)

    articles = predict.catalogue(MIND)
    vectors, report = embed.for_corpus(articles, MIND, competition / "work")

    assert report["from_artifact"] == 1
    assert report["encoded"] == 2
    assert encoded == ["chocolate cake recipe. how to bake",
                       "stock market rally continues. shares rose"]
    assert vectors.vectors.any(axis=1).all()


def test_a_cached_matrix_is_only_reused_for_the_corpus_it_was_built_for(
    competition, monkeypatch
):
    """A matrix is positional. One built before the catalogue gained an article
    pairs every row after it with the wrong article and loads without
    complaint, which is a submission that is wrong and looks fine."""
    width = MIND.embeddings.dim
    monkeypatch.setattr(
        embed, "read_source", lambda config: (np.array([], dtype=object),
                                              np.zeros((0, width), dtype="float32"))
    )
    monkeypatch.setattr(
        embed, "encode",
        lambda texts, config, **kwargs: np.tile(
            np.eye(1, width, dtype="float32"), (len(texts), 1)
        ),
    )

    articles = predict.catalogue(MIND)
    directory = competition / "work"
    embed.for_corpus(articles, MIND, directory)

    _, again = embed.for_corpus(articles, MIND, directory)
    assert again["cached"] == 1

    _, rebuilt = embed.for_corpus(articles.iloc[:2], MIND, directory)
    assert rebuilt["cached"] == 0


# --- EB-NeRD: the same five seams, for the competition whose test set ships
# --- its candidates in parquet and its click histories in a table of their own.

EBNERD = DATASETS["ebnerd"]

DANISH = [
    (1, "fodboldfinalen spilles i aften", "kampen er udsolgt", "sport"),
    (2, "opskrift på chokoladekage", "sådan bager du den", "mad"),
    (3, "aktiemarkedet stiger fortsat", "kurserne steg", "okonomi"),
]


@pytest.fixture
def ebnerd_competition(tmp_path, monkeypatch):
    """EB-NeRD's test set on disk, where its registry entry says it lives."""
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "PREDICTIONS_DIR", tmp_path / "predictions")

    test_dir = EBNERD.raw_dir / "testset" / "test"
    test_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "article_id": [article for article, _, _, _ in DANISH],
            "title": [title for _, title, _, _ in DANISH],
            "subtitle": [subtitle for _, _, subtitle, _ in DANISH],
            "body": ["" for _ in DANISH],
            "category_str": [category for _, _, _, category in DANISH],
            "subcategory": [[10] for _ in DANISH],
            "published_time": pd.to_datetime(["2023-06-01"] * len(DANISH)),
        }
    ).to_parquet(EBNERD.raw_dir / "testset" / "articles.parquet", index=False)
    # A user who has read about cake, then one the history table never mentions.
    pd.DataFrame(
        {
            "impression_id": [1, 2],
            "user_id": [11, 22],
            "impression_time": pd.to_datetime(["2023-06-02 07:00", "2023-06-02 08:00"]),
            "article_ids_inview": [[1, 2, 3], [3, 1]],
        }
    ).to_parquet(test_dir / "behaviors.parquet", index=False)
    pd.DataFrame(
        {"user_id": [11], **_engagement([[2]])}
    ).to_parquet(test_dir / "history.parquet", index=False)
    return tmp_path


def _engagement(histories):
    """The click ids plus the three arrays that run parallel to them.

    Every EB-NeRD history file carries all four — train, validation and the
    competition's own test set alike, checked on the real files — so a fixture
    with only the ids is a shape the pipeline never actually meets.
    """
    stamp = pd.Timestamp("2023-06-01 09:00:00")
    return {
        "article_id_fixed": histories,
        "impression_time_fixed": [
            [stamp + pd.Timedelta(hours=i) for i in range(len(clicks))]
            for clicks in histories
        ],
        "read_time_fixed": [[12.0] * len(clicks) for clicks in histories],
        "scroll_percentage_fixed": [[60.0] * len(clicks) for clicks in histories],
    }


def ebnerd_submitted(tmp_path) -> list[str]:
    archive = tmp_path / "predictions" / EBNERD.submission.bundle
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.namelist() == [EBNERD.submission.filename]
        return zipped.read(EBNERD.submission.filename).decode().splitlines()


def test_the_ebnerd_line_is_what_the_challenges_own_writer_produces():
    """`write_submission_file` joins the impression id and the bracketed ranks
    with a single space, over `rank_predictions_by_score` output — 1-based
    ranks in candidate order. A file of scores would parse and score as
    nothing."""
    assert submissions.ebnerd_line("237", [4, 1, 3, 2]) == "237 [4,1,3,2]"


def test_ebnerd_test_impressions_parse_without_labels():
    """The in-view ids arrive as integers and the clicked ids are not there at
    all — that column is what the leaderboard is holding back."""
    raw = pd.DataFrame(
        {
            "impression_id": [1, 2],
            "user_id": [11, 22],
            "impression_time": pd.to_datetime(["2023-06-01", "2023-06-01"]),
            "article_ids_inview": [[3, 4], [5]],
        }
    )
    parsed = sources.ebnerd_test_impressions(raw)
    assert list(parsed["candidate_ids"]) == [["3", "4"], ["5"]]
    assert "click_history" not in parsed
    assert "labels" not in parsed


def test_the_ebnerd_behaviours_adapter_cannot_read_the_test_file():
    """Why a second adapter exists. The labelled parser reads a column the test
    file does not have; if it ever stops raising, it does so by inventing
    labels."""
    raw = pd.DataFrame(
        {
            "impression_id": [1],
            "user_id": [11],
            "impression_time": pd.to_datetime(["2023-06-01"]),
            "article_ids_inview": [[3, 4]],
        }
    )
    with pytest.raises(KeyError):
        sources.ebnerd_behaviors(raw)


def test_the_history_table_is_joined_onto_the_impressions(ebnerd_competition):
    """EB-NeRD keeps histories in their own file, keyed by user. A user it has
    no row for is a cold start, not a missing line."""
    chunks = list(predict.impressions(EBNERD, chunk_size=10))

    assert len(chunks) == 1
    assert list(chunks[0]["click_history"]) == [["2"], []]


def test_only_the_clicks_a_retriever_reads_are_kept(ebnerd_competition):
    """Both retrievers read `clicks[-history_k:]` and nothing before it. The
    tail is taken as each chunk of the history table is read, because the whole
    of EB-NeRD's test histories does not fit in memory alongside the run."""
    path = EBNERD.raw_dir / "testset" / "test" / "history.parquet"
    pd.DataFrame(
        {"user_id": [11], **_engagement([[1, 2, 3, 1, 2]])}
    ).to_parquet(path, index=False)

    streamed = predict._histories(EBNERD, history_k=3)

    assert streamed.clicks == {"11": ["3", "1", "2"]}


def test_the_streamed_history_carries_what_the_weighting_reads(
    ebnerd_competition,
):
    """EB-NeRD weights clicks by engagement, and the arrays that describes it
    with run parallel to the ids. Truncated to the same suffix, or every weight
    lands on the wrong click; and truncated to *only* what the scheme reads,
    because `click_times` at 808k users is 500 MB spent on a column engagement
    never looks at."""
    path = EBNERD.raw_dir / "testset" / "test" / "history.parquet"
    pd.DataFrame(
        {"user_id": [11], **_engagement([[1, 2, 3, 1, 2]])}
    ).to_parquet(path, index=False)

    streamed = predict._histories(EBNERD, history_k=3)

    assert set(streamed.columns) == {"click_read_times", "click_scroll"}
    for column in streamed.columns.values():
        assert len(column["11"]) == 3, "aligned with the three ids kept"


def test_a_user_the_history_never_mentions_weights_nothing(ebnerd_competition):
    """A cold start is an empty window, not a missing key that raises on the
    chunk that happens to contain them."""
    chunk = next(predict.impressions(EBNERD, chunk_size=10))

    cold = chunk.index[chunk["click_history"].map(len) == 0][0]
    assert len(chunk["click_read_times"][cold]) == 0
    assert len(chunk["click_scroll"][cold]) == 0


def test_the_ebnerd_submission_ranks_the_candidates_it_was_given(
    ebnerd_competition, monkeypatch
):
    """End to end over the lexical retriever, which needs no artifact.

    User 11's only click is the cake article, so it takes first place among its
    own candidates and the other two keep the order the file listed them in.
    User 22 has no history row, scores every candidate 0, and still gets a
    line."""
    monkeypatch.setattr(acquire, "_fetch", lambda *args: pytest.fail("downloaded"))

    predict.submit(EBNERD, retriever="bm25")

    assert ebnerd_submitted(ebnerd_competition) == ["1 [2,1,3]", "2 [1,2]"]


def test_every_ebnerd_impression_is_written_once_whatever_the_chunking(
    ebnerd_competition,
):
    """The parquet is batched off the file rather than sliced out of a frame
    read whole, so the row count is a property of the batch size until
    something asserts otherwise."""
    predict.submit(EBNERD, retriever="bm25", chunk_size=1)

    assert ebnerd_submitted(ebnerd_competition) == ["1 [2,1,3]", "2 [1,2]"]


def test_the_id_the_competition_repeats_on_purpose_is_written_anyway(
    ebnerd_competition,
):
    """EB-NeRD stamps id 0 on all 200,000 of its beyond-accuracy impressions.
    Refusing them as duplicates would drop the whole diversity half of the
    leaderboard's task; they are matched by row instead."""
    pd.DataFrame(
        {
            "impression_id": [0, 0, 5],
            "user_id": [11, 22, 11],
            "impression_time": pd.to_datetime(["2023-06-02 07:00"] * 3),
            "article_ids_inview": [[1, 2], [2, 3], [3, 2]],
        }
    ).to_parquet(
        EBNERD.raw_dir / "testset" / "test" / "behaviors.parquet", index=False
    )

    predict.submit(EBNERD, retriever="bm25")

    assert ebnerd_submitted(ebnerd_competition) == ["0 [2,1]", "0 [1,2]", "5 [2,1]"]


def test_an_ordinary_id_is_still_refused_twice(ebnerd_competition):
    """The exemption is one value wide. Every other id is still a row key, so a
    reader that handed the same chunk over twice is caught."""
    pd.DataFrame(
        {
            "impression_id": [5, 5],
            "user_id": [11, 11],
            "impression_time": pd.to_datetime(["2023-06-02 07:00"] * 2),
            "article_ids_inview": [[1, 2], [1, 2]],
        }
    ).to_parquet(
        EBNERD.raw_dir / "testset" / "test" / "behaviors.parquet", index=False
    )

    with pytest.raises(predict.SubmissionError, match="appears twice"):
        predict.submit(EBNERD, retriever="bm25")
