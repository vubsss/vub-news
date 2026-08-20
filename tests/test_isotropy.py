"""Embedding geometry: the statistic that measures it, and the transforms that fix it.

EB-NeRD's shipped mBERT vectors have a mean pairwise cosine of 0.95 -- every
article looks 95% like every other one -- and its semantic retriever scores at
chance as a direct result. These are the seams that measure and correct that.
"""

import numpy as np
import pytest

from pipeline import embed


def cone(rows, width, spread, seed=0):
    """Rows clustered tightly around one direction: an anisotropic cloud.

    This is what raw BERT output looks like geometrically -- a narrow cone
    rather than a ball -- so it is what the correction has to work on.
    """
    rng = np.random.default_rng(seed)
    axis = np.zeros(width, dtype="float32")
    axis[0] = 1.0
    # Scaled by sqrt(width) so `spread` is the noise-to-signal ratio at any
    # dimensionality: gaussian noise over `width` dimensions has norm
    # proportional to sqrt(width), so a fixed coefficient makes a tight cone at
    # 16 dimensions and a diffuse ball at 768.
    noise = rng.standard_normal((rows, width)).astype("float32") / np.sqrt(width)
    return embed.normalise(axis + spread * noise)


# --- the statistic ---------------------------------------------------------


def test_orthogonal_rows_have_no_pairwise_similarity():
    """The identity matrix is the isotropic extreme: every distinct pair is at
    right angles, so every off-diagonal cosine is exactly zero."""
    assert embed.anisotropy(np.eye(8, dtype="float32")) == pytest.approx(0.0)


def test_a_narrow_cone_is_near_one():
    """The anisotropic extreme, and the shape the statistic exists to catch."""
    assert embed.anisotropy(cone(200, 16, spread=0.05)) > 0.95


def test_the_statistic_ignores_the_diagonal():
    """A row is identical to itself, so including the diagonal would report a
    floor of 1/n no matter how the cloud is actually shaped."""
    orthogonal = embed.anisotropy(np.eye(4, dtype="float32"))
    assert orthogonal == pytest.approx(0.0), (
        "with the diagonal counted this would be 0.25, not 0"
    )


def test_the_same_matrix_always_gives_the_same_number():
    """It samples on a large corpus, so the sample has to be seeded: a
    statistic printed in a build log that moved between runs would read as the
    vectors having changed."""
    vectors = cone(5000, 32, spread=0.4)

    assert embed.anisotropy(vectors) == embed.anisotropy(vectors)


# --- the transforms --------------------------------------------------------


def test_no_postprocessing_returns_the_vectors_unchanged():
    """`none` has to be exactly the identity, or the MIND path -- which is
    already near-isotropic and wants no correction -- silently changes."""
    vectors = cone(50, 8, spread=0.5)

    np.testing.assert_array_equal(embed.postprocess(vectors, "none"), vectors)


@pytest.mark.parametrize("method", ["none", "centre", "abtt:3", "whiten"])
def test_every_method_returns_unit_length_rows(method):
    """ann_index takes an inner product of these and calls it a cosine, so a
    transform that returned unscaled rows would break that silently."""
    result = embed.postprocess(cone(100, 16, spread=0.3), method)

    assert np.linalg.norm(result, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_centring_reduces_the_mean_pairwise_cosine():
    """The narrow cone is a cloud plus a large shared offset. Removing the
    offset is what leaves the part that differs between articles."""
    vectors = cone(500, 32, spread=0.05)

    assert embed.anisotropy(vectors) > 0.9
    assert embed.anisotropy(embed.postprocess(vectors, "centre")) < 0.5


def test_abtt_removes_the_directions_it_was_asked_to():
    """all-but-the-top removes the mean and then the n leading principal
    directions, so the corrected cloud has almost no variance left along them.

    Deliberately not asserted through `anisotropy`: that statistic is a mean
    over pairs and therefore sees the shared offset, which plain centring
    already removes. What removing further components buys is second-moment
    and shows up in retrieval quality, not in a mean cosine -- so the property
    is checked where it actually lives, on the subspace itself.
    """
    vectors = cone(500, 32, spread=0.3)
    centred = vectors - vectors.mean(axis=0)
    _, _, directions = np.linalg.svd(centred, full_matrices=False)
    top_three = directions[:3]

    def variance_along(matrix):
        return float((((matrix - matrix.mean(axis=0)) @ top_three.T) ** 2).mean())

    before = variance_along(embed.postprocess(vectors, "centre"))
    after = variance_along(embed.postprocess(vectors, "abtt:3"))

    assert after < before / 100, (
        f"abtt:3 left {after:.2e} variance along the three directions it was "
        f"asked to remove, against {before:.2e} for plain centring"
    )


def test_an_article_with_no_vector_keeps_its_zero_row():
    """align leaves a zero row where the artifact had no vector for an article,
    and a zero row never wins a comparison. Centring one would hand it the
    negated mean -- a real direction, pointing away from everything, which is
    a claim about an article we have no information about."""
    vectors = cone(20, 8, spread=0.4)
    vectors[3] = 0.0

    for method in ("centre", "abtt:2", "whiten"):
        result = embed.postprocess(vectors, method)
        assert np.linalg.norm(result[3]) == 0.0, method


def test_an_unknown_method_is_refused_by_name():
    """A typo in the registry must not fall through to "no correction", which
    would look like the correction simply not helping."""
    with pytest.raises(embed.EmbeddingError, match="whitening"):
        embed.postprocess(cone(10, 4, spread=0.5), "whitening")


# --- the stage -------------------------------------------------------------
#
# These reuse test_embed's fixtures: the point here is that the registry choice
# reaches the matrix on disk, not that the artifact parses.

import dataclasses  # noqa: E402

import pandas as pd  # noqa: E402

from pipeline import paths  # noqa: E402
from pipeline.datasets import EBNERD  # noqa: E402


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path


def write_cone_artifact(rows=400):
    """EB-NeRD's artifact shape, holding an anisotropic cloud -- which is what
    the real one holds."""
    vectors = cone(rows, EBNERD.embeddings.dim, spread=0.08)
    path = EBNERD.raw_dir / EBNERD.embeddings.artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "article_id": np.arange(rows, dtype="int32"),
            EBNERD.embeddings.model.replace("_", "-"): list(vectors),
        }
    ).to_parquet(path, index=False)
    ids = pd.Series([str(i) for i in range(rows)], dtype="string")
    EBNERD.feature_store_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"article_id": ids}).to_parquet(
        EBNERD.feature_store_dir / "articles.parquet", index=False
    )
    return ids


def configured(method):
    return dataclasses.replace(
        EBNERD, embeddings=dataclasses.replace(EBNERD.embeddings, postprocess=method)
    )


def test_the_registry_choice_reaches_the_matrix(tree):
    """The whole point of the field: correcting the geometry is a registry
    decision, the way normalising already is, and no stage branches on it."""
    write_cone_artifact()

    raw, _ = embed.build(
        pd.read_parquet(EBNERD.feature_store_dir / "articles.parquet"),
        configured("none"),
    )
    corrected, _ = embed.build(
        pd.read_parquet(EBNERD.feature_store_dir / "articles.parquet"),
        configured("abtt:3"),
    )

    assert embed.anisotropy(raw.vectors) > 0.9
    assert embed.anisotropy(corrected.vectors) < 0.5


def test_correcting_the_geometry_does_not_disturb_the_alignment(tree):
    """ann_index treats a row number and a catalogue position as the same
    thing. A transform that reordered or dropped rows would keep passing every
    geometric check while handing each article another article's vector."""
    ids = write_cone_artifact()
    articles = pd.DataFrame({"article_id": ids[::-1].reset_index(drop=True)})

    raw, _ = embed.build(articles, configured("none"))
    corrected, _ = embed.build(articles, configured("whiten"))

    assert list(corrected.article_ids) == list(raw.article_ids)
    assert corrected.index == raw.index
    assert corrected.vectors.shape[0] == len(articles)


def test_the_stage_prints_the_geometry_before_and_after(tree, capsys):
    """The mechanism has to be visible in the build log. A retriever that
    ranks at chance because its vectors occupy a cone looks exactly like a
    retriever that is simply bad, and the difference is this number."""
    write_cone_artifact()

    embed.run(configured("abtt:3"))

    printed = capsys.readouterr().out
    assert "anisotropy" in printed
    assert "abtt:3" in printed
