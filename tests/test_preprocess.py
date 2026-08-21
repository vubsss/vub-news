"""Preprocessing is tested at four seams: cleaner, build_lexical_text, run,
and the articles table on disk."""

import dataclasses

import pandas as pd
import pytest

from pipeline import paths, preprocess
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]
EBNERD = DATASETS["ebnerd"]


def test_danish_keeps_ae_oe_and_aa_and_stems_as_danish():
    """æ, ø and å are letters here, not decorated ASCII. Folding them to
    ae/oe/aa would cut every Danish token off from its own stem, so the
    expected string carries all three through to the output. The stems are
    the Snowball Danish algorithm's; the English one leaves 'æblerne' and
    'fløde' untouched, which is what this would look like if the language
    came from anywhere but the registry."""
    clean = preprocess.cleaner(EBNERD)

    text = "Æblerne på øen er gode, og fløde året rundt!"

    assert clean(text) == "æbl øen god flød året rund"


def test_english_is_lowercased_and_destopworded_but_not_stemmed():
    """SPEC asks for no English stemming, so 'running' and 'sharks' have to
    come through whole — a stemmer applied to both languages by accident would
    show up here as 'run' and 'shark'. Digits stay: a year is a strong BM25
    term."""
    clean = preprocess.cleaner(MIND)

    text = "Breaking: The Sharks' 2019 season was running late!"

    assert clean(text) == "breaking sharks 2019 season running late"


def test_a_third_language_is_an_entry_not_a_code_change(monkeypatch):
    """The acceptance criterion for the registry: a dataset in a language
    nothing has been written for yet gets cleaned correctly by the same
    call, with no branch anywhere naming it."""
    monkeypatch.setitem(
        preprocess.LANGUAGES,
        "norwegian",
        preprocess.Language(stopwords_from="norwegian", stemmer="norwegian"),
    )
    config = dataclasses.replace(MIND, language="norwegian")

    clean = preprocess.cleaner(config)

    # "over" and "og" are Norwegian stopwords; "kampen" stems to "kamp".
    assert clean("Fotballspillerne løp over broen og vant kampen!") == (
        "fotballspillern løp broen vant kamp"
    )


def test_a_language_with_no_rules_says_so_rather_than_cleaning_badly():
    """The other half of that criterion. Falling back to English rules for an
    unknown language would silently stem nothing and strip the wrong words,
    and the index would just be quietly worse."""
    config = dataclasses.replace(MIND, language="klingon")

    with pytest.raises(preprocess.LanguageError, match="klingon"):
        preprocess.cleaner(config)


def articles(rows):
    """rows: (article_id, title, abstract)"""
    return pd.DataFrame(
        {
            "article_id": pd.Series([row[0] for row in rows], dtype="string"),
            "title": pd.Series([row[1] for row in rows], dtype="string"),
            "abstract": pd.Series([row[2] for row in rows], dtype="string"),
        }
    )


def test_an_article_with_no_abstract_falls_back_to_its_title():
    """3,415 of MIND's 65,238 articles have no abstract. Pasting a null onto
    the title would give them the empty string and drop them out of the index
    entirely, so the title has to stand on its own."""
    frame = articles(
        [
            ("a1", "Sharks beat the Bears", "A late goal decided the game."),
            ("a2", "Sharks beat the Bears", None),
        ]
    )

    text, report = preprocess.build_lexical_text(frame, MIND, title_weight=1)

    assert list(text) == [
        "sharks beat bears late goal decided game",
        "sharks beat bears",
    ]
    assert report["missing_abstract"] == 1


def test_a_blank_abstract_counts_as_a_missing_one():
    """EB-NeRD writes an absent subtitle as the empty string, not as null, so
    counting only nulls reports it as a dataset with no gaps at all — 0 rather
    than the 1,709 it really has. Blank falls back to the title exactly as
    null does, so it has to be counted the same way."""
    frame = articles(
        [
            ("a1", "Sharks beat the Bears", "A late goal decided the game."),
            ("a2", "Sharks beat the Bears", None),
            ("a3", "Sharks beat the Bears", ""),
            ("a4", "Sharks beat the Bears", "   "),
        ]
    )

    text, report = preprocess.build_lexical_text(frame, MIND, title_weight=1)

    assert list(text)[1:] == ["sharks beat bears"] * 3
    assert report["missing_abstract"] == 3


def test_articles_left_with_nothing_after_cleaning_are_counted_not_dropped():
    """An all-stopword headline cleans down to nothing. It stays as an empty
    string on its own row so the column still lines up with the article table
    the index is built from, and the count is reported so an unusable share of
    the catalogue cannot pass unnoticed."""
    frame = articles(
        [
            ("a1", "Sharks beat the Bears", None),
            ("a2", "The and of a", "It was so."),
            ("a3", None, None),
        ]
    )

    text, report = preprocess.build_lexical_text(frame, MIND, title_weight=1)

    assert list(text) == ["sharks beat bears", "", ""]
    assert report["articles"] == 3
    assert report["empty_after_cleaning"] == 2


@pytest.mark.parametrize("config", list(DATASETS.values()), ids=lambda c: c.name)
def test_a_document_and_a_query_go_through_the_one_callable(config):
    """cleaner(config) is the whole of the cleaning: the indexed field is
    exactly what it returns for the article's own text. Ticket 6 builds its
    queries by calling the same function, so the two sides cannot drift — a
    build_lexical_text that cleaned documents its own way would fail here
    however reasonable its output looked.

    At title weight 1, because that is the composition this equality is about.
    The registry's weight is a tuned value and repeating the title is exactly
    what it does; what must not drift is the cleaning, which is shared."""
    title, abstract = "Æblerne på øen er gode", "Fløde året rundt!"
    frame = articles([("a1", title, abstract)])

    text, _ = preprocess.build_lexical_text(frame, config, title_weight=1)
    query = preprocess.cleaner(config)(f"{title} {abstract}")

    assert text[0] == query


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    MIND.feature_store_dir.mkdir(parents=True)
    return MIND.feature_store_dir


def catalogue(store):
    """Three articles: one complete, one with no abstract, one all stopwords."""
    frame = articles(
        [
            ("a1", "Sharks beat the Bears", "A late goal decided the game."),
            ("a2", "Sharks beat the Bears", None),
            ("a3", "The and of a", None),
        ]
    )
    frame["lexical_text"] = pd.Series([None] * len(frame), dtype="string")
    frame.to_parquet(store / "articles.parquet", index=False)


def test_run_fills_lexical_text_for_every_article_and_reports_the_gaps(store, capsys):
    catalogue(store)

    preprocess.run(MIND)

    written = pd.read_parquet(store / "articles.parquet")
    assert not written["lexical_text"].isna().any()
    # Composed at the registry's title weight rather than at a literal, because
    # what the stage owes is the catalogue the registry describes -- a run that
    # ignored a tuned weight would write a corpus no measured number came from.
    headline = " ".join(["sharks beat bears"] * MIND.lexical.title_weight)
    assert list(written["lexical_text"]) == [
        f"{headline} late goal decided game",
        headline,
        "",
    ]

    # Both counts the ticket asks to measure: the title-only fallbacks and the
    # articles nothing survived, so a catalogue that is mostly unusable cannot
    # reach the index build looking healthy.
    printed = capsys.readouterr().out
    assert "2 of 3" in printed
    assert "1 of 3" in printed


def test_running_preprocess_twice_changes_nothing(store):
    catalogue(store)

    preprocess.run(MIND)
    first = pd.read_parquet(store / "articles.parquet")
    preprocess.run(MIND)
    second = pd.read_parquet(store / "articles.parquet")

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("config", list(DATASETS.values()), ids=lambda c: c.name)
def test_the_real_catalogue_has_lexical_text_for_every_article(config):
    """The artifact on disk, not a fixture: the next ticket indexes this
    column, and a null in it is a row BM25 cannot tokenise."""
    path = config.feature_store_dir / "articles.parquet"
    if not path.exists():
        pytest.skip(f"{config.name} feature store not built")

    articles = pd.read_parquet(path, columns=["lexical_text"])
    if articles["lexical_text"].isna().all():
        pytest.skip(f"{config.name} preprocess stage has not run")

    assert not articles["lexical_text"].isna().any()
