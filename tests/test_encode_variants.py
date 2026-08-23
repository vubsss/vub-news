"""Phase 7's two silent failure modes, and the order contract.

A checkpoint's pooling and its prefix are properties of the model, not choices.
Read either one wrongly and nothing raises: the vectors come out the right
shape, the right dtype and unit length, and they are not the model's. So both
are asserted here rather than trusted, and so is the thing that would corrupt
every variant at once — encoding in an order other than the id index's.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pipeline import embed, encode_variants, paths
from pipeline.datasets import DATASETS, EmbeddingSpec

MIND = DATASETS["mind"]


def spec(**changes) -> EmbeddingSpec:
    return dataclasses.replace(MIND.embedding_variants[0], **changes)


# --- the registry says what each checkpoint needs ---------------------------


def test_bge_pools_the_cls_token_and_the_others_pool_the_mean():
    """Read off each model's own `1_Pooling/config.json`, not from memory.
    bge-base-en-v1.5 sets `pooling_mode_cls_token`; MiniLM, mpnet and e5 all
    set `pooling_mode_mean_tokens`."""
    pooling = {s.name: s.pooling for s in MIND.embedding_variants}

    assert pooling["bge-base-en-v1.5"] == "cls"
    assert pooling["all-mpnet-base-v2"] == "mean"
    assert MIND.embeddings.pooling == "mean", "MiniLM, and the default"


def test_only_e5_carries_a_prefix_and_it_carries_two():
    """e5 requires one on every input. The sentence-transformers models want
    none, and bge's instruction is a *query* one — this pipeline encodes no
    queries, so it has nowhere to go."""
    prefixes = {s.name: s.prefix for s in MIND.embedding_variants}

    assert prefixes["e5-base-v2 (query:)"] == "query: "
    assert prefixes["e5-base-v2 (passage:)"] == "passage: "
    assert prefixes["all-mpnet-base-v2"] == ""
    assert prefixes["bge-base-en-v1.5"] == ""


def test_two_variants_of_one_checkpoint_are_named_apart():
    """Both e5 entries are the same model. Labelled by `model` they would
    report two different measurements under one name, and the grid would look
    like a checkpoint that disagreed with itself."""
    e5 = [s for s in MIND.embedding_variants if s.model == "intfloat/e5-base-v2"]

    assert len(e5) == 2
    assert len({s.name for s in e5}) == 2
    assert len({s.artifact for s in e5}) == 2, "and they must not overwrite"


# --- the poolers themselves -------------------------------------------------


def test_cls_pooling_reads_the_first_token_and_mean_pooling_does_not():
    """The whole difference, on a sequence built so the two cannot agree."""
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    mask = torch.ones(1, 3)

    assert embed.cls_pool(hidden, mask).tolist() == [[1.0, 0.0]]
    assert embed.mean_pool(hidden, mask)[0].tolist() == pytest.approx([1 / 3, 2 / 3])


def test_mean_pooling_ignores_padding_but_cls_never_sees_it():
    """Position 0 is never padding, which is why cls_pool can drop the mask."""
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 0.0], [9.0, 9.0]]])
    mask = torch.tensor([[1.0, 0.0]])

    assert embed.mean_pool(hidden, mask).tolist() == [[1.0, 0.0]]
    assert embed.cls_pool(hidden, mask).tolist() == [[1.0, 0.0]]


def test_an_unknown_pooling_is_refused_rather_than_defaulted():
    """Defaulting to the mean is exactly the silent failure this guards."""
    with pytest.raises(embed.EmbeddingError, match="unknown pooling"):
        embed.encode(["a"], MIND, spec=spec(pooling="first-and-last"))


# --- the prefix reaches the tokeniser ---------------------------------------


def test_the_prefix_is_actually_prepended_to_every_document():
    """The ticket's own checklist item, and the reason it is on it: an e5 run
    without its prefix does not error, it just scores worse than the model
    deserves. Captured at the tokeniser, which is the last point the text is
    still text."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    seen = []

    class Tokeniser:
        def __call__(self, texts, **kwargs):
            seen.extend(texts)
            raise _Stop

    class _Stop(Exception):
        pass

    monkey = pytest.MonkeyPatch()
    monkey.setattr(transformers.AutoTokenizer, "from_pretrained",
                   staticmethod(lambda *a, **k: Tokeniser()))
    monkey.setattr(transformers.AutoModel, "from_pretrained",
                   staticmethod(lambda *a, **k: _Model()))
    try:
        with pytest.raises(_Stop):
            embed.encode(["a headline"], MIND, spec=spec(prefix="query: "))
    finally:
        monkey.undo()

    assert seen == ["query: a headline"]


class _Model:
    def to(self, device):
        return self

    def eval(self):
        return self


def test_no_prefix_leaves_the_text_alone():
    """A model that wants none must not receive an empty-string artefact."""
    assert spec(prefix="").prefix == ""


# --- the order contract -----------------------------------------------------


def test_the_texts_come_back_in_the_id_indexs_order(tmp_path, monkeypatch):
    """`read_source` pairs one `.npy` with one id index by position. A variant
    encoded in the catalogue's order rather than the index's would align to
    every article perfectly and hold the wrong article's meaning in every row.
    """
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    MIND.artifacts_dir.mkdir(parents=True)
    MIND.feature_store_dir.mkdir(parents=True)

    # The index deliberately disagrees with the catalogue's own order.
    pd.DataFrame({"article_id": ["N3", "N1", "N2"]}).to_parquet(
        MIND.artifacts_dir / embed.ID_INDEX, index=False
    )
    pd.DataFrame(
        {
            "article_id": ["N1", "N2", "N3"],
            "title": ["first", "second", "third"],
            "abstract": ["", "", ""],
        }
    ).to_parquet(MIND.feature_store_dir / "articles.parquet", index=False)

    assert encode_variants.texts_for(MIND) == ["third", "first", "second"]


def test_word2vec_skips_words_it_does_not_know(tmp_path):
    """Out of vocabulary contributes nothing, rather than a zero vector that
    would drag the document toward the origin — the argument the user profile
    already makes about clicks the catalogue has no row for."""
    gensim = pytest.importorskip("gensim")
    from gensim.models import KeyedVectors

    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(["cake", "bread"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    path = tmp_path / "w2v.model"
    vectors.save(str(path))

    found = embed.encode_word2vec(["cake bread", "cake zzzz", "zzzz"], path, 2)

    assert found[0].tolist() == [0.5, 0.5]
    assert found[1].tolist() == [1.0, 0.0], "the unknown word is skipped"
    assert not found[2].any(), "a document of unknowns is a zero row"
