"""Embedding artifacts are tested at four seams: align, normalise, load and run."""

import dataclasses
import sys
import types

import numpy as np
import pandas as pd
import pytest

from pipeline import embed, paths
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]
EBNERD = DATASETS["ebnerd"]


def corpus(ids):
    return pd.Series(ids, dtype="string")


def test_align_puts_every_corpus_article_on_its_own_row():
    """The corpus is the authority on which rows exist, not the embedding
    artifact. EB-NeRD ships vectors for 125,541 articles while ebnerd_small's
    catalogue holds 20,738, and the source order is not the corpus order — so
    a positional load would silently pair articles with other articles'
    vectors and every semantic recall number after it would be wrong while
    looking entirely reasonable."""
    source_ids = np.array(["a2", "a9", "a1"], dtype=object)
    source_vectors = np.array(
        [[2.0, 2.0], [9.0, 9.0], [1.0, 1.0]], dtype="float32"
    )

    matrix, missing = embed.align(
        source_ids, source_vectors, corpus(["a1", "a2"]), dim=2
    )

    assert matrix.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert missing == 0


def test_an_article_with_no_vector_keeps_its_row_and_is_counted():
    """Ticket 7 asks for missing vectors to be reported rather than dropped.
    Dropping would be the quiet failure: the matrix would no longer be
    row-aligned to the corpus, so every article after the gap would answer
    with its neighbour's vector. A zero row scores zero inner product against
    every query, so the article is simply never retrieved — which is the
    honest behaviour for an article we hold no evidence about."""
    source_ids = np.array(["a1"], dtype=object)
    source_vectors = np.array([[1.0, 1.0]], dtype="float32")

    matrix, missing = embed.align(
        source_ids, source_vectors, corpus(["a1", "gone", "a1"]), dim=2
    )

    assert matrix.shape == (3, 2)
    assert matrix[1].tolist() == [0.0, 0.0]
    assert missing == 1


def test_an_article_with_no_abstract_is_encoded_from_its_title_alone():
    """3,415 of MIND's 65,238 articles have no abstract. The text a vector
    means is defined here and nowhere else -- the generation notebook imports
    this rather than reproducing it, because a notebook that drifted would
    encode something the pipeline does not think it holds."""
    articles = pd.DataFrame(
        {
            "title": pd.Series(["Cats win", "Dr. Who"], dtype="string"),
            "abstract": pd.Series(["They did", None], dtype="string"),
        }
    )

    assert list(embed.document_text(articles)) == ["Cats win. They did", "Dr. Who"]


def test_mean_pool_averages_only_the_real_tokens():
    """A batch is padded to its longest text, so pooling over the padding too
    would shrink a short document's vector by a factor set by whatever got
    batched beside it -- the same article would embed differently at a
    different batch boundary, and every vector would still look plausible.
    Here the padded position holds an absurd value: if the mask were ignored,
    the first row would come out near (50, 49.5) rather than (1, 0)."""
    torch = pytest.importorskip("torch")
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [99.0, 99.0]],  # one real token, one padded
            [[1.0, 0.0], [3.0, 4.0]],  # two real tokens
        ]
    )
    mask = torch.tensor([[1, 0], [1, 1]])

    pooled = embed.mean_pool(hidden, mask)

    assert pooled[0].tolist() == [1.0, 0.0]
    assert pooled[1].tolist() == [2.0, 2.0]


def test_normalise_scales_rows_to_unit_length_and_leaves_zero_rows_alone():
    """A 3-4-5 triangle, so the expectation comes from geometry rather than
    from rerunning the code's own arithmetic. Zero rows are the articles align
    found no vector for: they have no direction to scale to length one, and
    dividing by their norm would put NaN into the matrix, which would poison
    every score in the index rather than simply never being retrieved."""
    matrix = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 5.0]], dtype="float32")

    unit = embed.normalise(matrix)

    assert unit[0].tolist() == pytest.approx([0.6, 0.8])
    assert unit[1].tolist() == [0.0, 0.0]
    assert unit[2].tolist() == pytest.approx([0.0, 1.0])
    assert not np.isnan(unit).any()


def test_vectors_that_are_not_unit_length_are_an_error():
    """Ticket 8 indexes these with an inner product and calls the result a
    cosine similarity. That equivalence holds only for unit vectors, and
    nothing downstream can detect its absence — the scores stay finite,
    ordered and plausible, they just rank partly by vector magnitude. So the
    property is asserted here, at the one place that can still see it. The
    offending norm is named, because 12.4 points at raw BERT output and 0.5
    points at something having been scaled twice."""
    raw = np.array([[3.0, 4.0], [1.0, 0.0]], dtype="float32")

    with pytest.raises(embed.EmbeddingError, match="5"):
        embed.check_unit_norm(raw, MIND)

    # A zero row is a known-missing article, not a normalisation failure.
    embed.check_unit_norm(
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype="float32"), MIND
    )


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The three directories a dataset owns, relocated under tmp_path."""
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path


def write_provided(vectors=((1.0, 2.0), (3.0, 4.0)), ids=(3001353, 3003065)):
    """EB-NeRD's artifact as it actually ships: int32 ids, vectors as a list
    column named after the model."""
    path = EBNERD.raw_dir / EBNERD.embeddings.artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "article_id": np.array(ids, dtype="int32"),
            "google-bert/bert-base-multilingual-cased": [list(v) for v in vectors],
        }
    ).to_parquet(path, index=False)
    return path


def test_provided_vectors_arrive_with_their_ids_as_strings(tree):
    """EB-NeRD's artifact keys its vectors by int32 article_id while the
    unified corpus keys articles by string. Left unconverted, align matches
    nothing at all: every one of the 20,738 articles takes a zero row, the
    unit-norm check passes because zero rows are exempt, and semantic recall
    comes out a clean 0.0000 that looks like a weak retriever rather than a
    dtype mismatch."""
    write_provided()

    ids, vectors = embed.read_source(EBNERD)

    assert list(ids) == ["3001353", "3003065"]
    assert vectors.dtype == np.dtype("float32")
    assert vectors.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def vec(*values):
    """A row of the width the registry declares, with `values` at the front.

    Full width rather than a convenient two, so build sees the dimensionality
    its own check is about.
    """
    row = np.zeros(EBNERD.embeddings.dim, dtype="float32")
    row[: len(values)] = values
    return row


def test_build_puts_the_corpus_in_charge_of_rows_and_hands_back_one_interface(tree):
    """The interface ticket 7 asks for: vectors plus an article-id-to-row
    index, identical in shape whichever way the dataset came by its vectors.
    The artifact here holds a1 and a2 in that order and an a9 the corpus never
    heard of; the corpus asks for a2 first. Row order following the corpus is
    what lets ticket 8 treat a row number and a catalogue position as the same
    thing."""
    write_provided(vectors=(vec(3.0, 4.0), vec(0.0, 5.0), vec(9.0)),
                   ids=(1, 2, 9))
    articles = pd.DataFrame({"article_id": pd.Series(["2", "1"], dtype="string")})

    embeddings, report = embed.build(articles, EBNERD)

    assert list(embeddings.article_ids) == ["2", "1"]
    assert embeddings.index == {"2": 0, "1": 1}
    # a2 was (0,5) -> unit (0,1); a1 was (3,4) -> unit (0.6,0.8).
    assert embeddings.vectors[0][:2].tolist() == pytest.approx([0.0, 1.0])
    assert embeddings.vectors[1][:2].tolist() == pytest.approx([0.6, 0.8])
    assert report["missing"] == 0
    assert report["articles"] == 2
    assert report["dim"] == EBNERD.embeddings.dim


def test_a_missing_mind_artifact_with_no_drive_id_says_what_to_do(tree):
    """The state the repo is in until the Colab notebook has been run and its
    output uploaded. The stage cannot invent the vectors and must not pretend
    to, so it stops with the two things the user has to do — run the notebook,
    put the id in the registry — rather than a FileNotFoundError naming a path
    nobody ever created."""
    MIND.artifacts_dir.mkdir(parents=True)

    with pytest.raises(embed.EmbeddingError) as failure:
        embed.ensure_artifact(MIND)

    message = str(failure.value)
    assert embed.NOTEBOOK in message
    assert "gdrive_file_id" in message


def test_an_artifact_already_on_disk_is_used_without_touching_the_network(
    tree, monkeypatch
):
    """Ticket 7 asks for the download to happen only when the artifact is
    missing. Re-fetching a 100 MB file on every build would be slow enough to
    notice but not slow enough to investigate, and it would make the pipeline
    need the network to rebuild something it already has. A gdown that
    explodes if it is called at all is the only way to assert the negative."""
    exploding = types.SimpleNamespace(
        download_folder=lambda **kwargs: pytest.fail(
            "downloaded an artifact that was already on disk"
        )
    )
    monkeypatch.setitem(sys.modules, "gdown", exploding)

    # A drive id is set, so nothing but the files on disk can stop the fetch.
    config = dataclasses.replace(
        MIND,
        embeddings=dataclasses.replace(MIND.embeddings, gdrive_file_id="folder-id"),
    )
    config.artifacts_dir.mkdir(parents=True)
    (config.artifacts_dir / config.embeddings.artifact).touch()
    (config.artifacts_dir / embed.ID_INDEX).touch()

    embed.ensure_artifact(config)


def test_a_reloaded_matrix_holds_the_same_vectors_against_the_same_ids(tree):
    """The stage's output is the aligned, unit-length matrix, not the vendor
    file it was built from — EB-NeRD's is 397 MB covering 125,541 articles for
    a corpus of 20,738, and ticket 8 should not pay to re-derive that on every
    run. Saving is only safe if the ids come back attached to the same rows,
    since the ids are the whole reason a row means anything."""
    write_provided(vectors=(vec(3.0, 4.0), vec(0.0, 5.0)), ids=(1, 2))
    articles = pd.DataFrame({"article_id": pd.Series(["2", "1"], dtype="string")})
    built, _ = embed.build(articles, EBNERD)

    built.save(embed.output_dir(EBNERD))
    reloaded = embed.load(EBNERD)

    assert list(reloaded.article_ids) == list(built.article_ids)
    assert reloaded.index == built.index
    np.testing.assert_array_equal(reloaded.vectors, built.vectors)


def write_generated(first_values, ids):
    """MIND's artifact as the notebook uploads it: a matrix at the width the
    registry declares and, beside it, the ids in encode order, which is not
    the corpus order."""
    matrix = np.zeros((len(ids), MIND.embeddings.dim), dtype="float32")
    matrix[:, 0] = first_values
    MIND.artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(MIND.artifacts_dir / MIND.embeddings.artifact, matrix)
    pd.DataFrame({"article_id": pd.Series(ids, dtype="string")}).to_parquet(
        MIND.artifacts_dir / embed.ID_INDEX, index=False
    )


def write_corpus(config, ids):
    config.feature_store_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"article_id": pd.Series(ids, dtype="string")}).to_parquet(
        config.feature_store_dir / "articles.parquet", index=False
    )


def test_saving_leaves_the_downloaded_id_index_where_it_was(tree):
    """The stage's output is in corpus order and the downloaded index is in
    the notebook's encode order, and read_source pairs that index positionally
    with embeddings.npy. Written to one path they would be the same file, so
    the first run would replace the notebook's ordering with the corpus's while
    the vectors kept the notebook's -- and the next `--force embed` would hand
    every article another article's vector, with ensure_artifact declining to
    re-download because both files are present. Nothing downstream can see
    that: the scores stay finite, ordered and plausible."""
    # MIND declares normalise=False, so its source is already unit length.
    write_generated(first_values=(1.0, 1.0), ids=["n2", "n1"])
    write_corpus(MIND, ["n1", "n2"])

    built, _ = embed.build(pd.read_parquet(
        MIND.feature_store_dir / "articles.parquet"), MIND)
    built.save(embed.output_dir(MIND))

    ids, _ = embed.read_source(MIND)
    assert list(ids) == ["n2", "n1"]


def test_run_writes_the_matrix_and_reports_what_it_holds(tree, capsys):
    """Ticket 7 asks for dimensionality and article count per dataset. Until
    run existed there was nowhere for either to be printed."""
    write_provided(vectors=(vec(3.0, 4.0), vec(0.0, 5.0)), ids=(1, 2))
    write_corpus(EBNERD, ["2", "1", "absent"])

    embed.run(EBNERD)

    printed = capsys.readouterr().out
    assert "3 articles" in printed
    assert str(EBNERD.embeddings.dim) in printed
    # The third article has no vector; reported, not dropped.
    assert "1 article(s) have no vector" in printed
    assert list(embed.load(EBNERD).article_ids) == ["2", "1", "absent"]


def test_a_second_run_reuses_the_saved_matrix_instead_of_the_source(tree, capsys):
    """EB-NeRD's source parquet is 397 MB covering 125,541 articles for a
    corpus of 20,738. Re-deriving it on every build is the cost the saved
    matrix exists to avoid, and deleting the source is the only way to assert
    it is not being read."""
    source = write_provided(vectors=(vec(3.0, 4.0),), ids=(1,))
    write_corpus(EBNERD, ["1"])
    embed.run(EBNERD)
    source.unlink()

    embed.run(EBNERD)

    assert "loaded from" in capsys.readouterr().out
    assert embed.load(EBNERD).vectors[0][:2].tolist() == pytest.approx([0.6, 0.8])


def test_forcing_the_stage_rederives_the_matrix_it_would_otherwise_reuse(tree):
    """`build.py --force embed` has to do the work again, or a re-run after a
    corpus change would quietly keep vectors aligned to the old catalogue."""
    write_provided(vectors=(vec(3.0, 4.0),), ids=(1,))
    write_corpus(EBNERD, ["1"])
    embed.run(EBNERD)

    write_provided(vectors=(vec(0.0, 5.0),), ids=(1,))
    embed.run(EBNERD, force=True)

    assert embed.load(EBNERD).vectors[0][:2].tolist() == pytest.approx([0.0, 1.0])
