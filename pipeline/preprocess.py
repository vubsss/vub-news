"""Language-parameterised text cleaning.

The rules come from the language the registry declares, never from the dataset
name, and `cleaner` is the only way to get at them — so the text that goes into
the index and the text a query is built from cannot drift apart. A third
language is a new LANGUAGES entry and a `language=` in the registry; no call
site changes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from pipeline.datasets import DatasetConfig

# Everything that is not a word character or whitespace. \w is unicode-aware,
# so æ, ø and å are word characters and survive; only punctuation goes.
PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)


class LanguageError(RuntimeError):
    """A dataset declares a language nothing knows how to clean."""


@dataclass(frozen=True)
class Language:
    """How one language is cleaned.

    stopwords_from and stemmer are NLTK's names for the language, which are
    not always the registry's; stemmer is None where the language needs no
    stemming.
    """

    stopwords_from: str
    stemmer: str | None


LANGUAGES = {
    "english": Language(stopwords_from="english", stemmer=None),
    "danish": Language(stopwords_from="danish", stemmer="danish"),
}


@cache
def _rules(language: str) -> tuple[frozenset[str], Callable[[str], str] | None]:
    """The stopword set and stemming function for a language, built once.

    The stopword corpus is a few kilobytes NLTK manages itself rather than a
    dataset archive, so it is fetched here on first use instead of in the
    acquire stage. Only when it is actually missing: nltk.download re-fetches
    its package index over the network every call, which would put a download
    on the path of every build.
    """
    if language not in LANGUAGES:
        raise LanguageError(
            f"no cleaning rules for {language!r}; add an entry to "
            f"preprocess.LANGUAGES (have: {', '.join(sorted(LANGUAGES))})"
        )
    spec = LANGUAGES[language]
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    stop = frozenset(stopwords.words(spec.stopwords_from))
    stem = SnowballStemmer(spec.stemmer).stem if spec.stemmer else None
    return stop, stem


def cleaner(config: DatasetConfig) -> Callable[[str], str]:
    """The one cleaning callable for this dataset, for documents and queries.

    Lowercase, strip punctuation, drop the language's stopwords, then stem if
    the language asks for it — stopword lists are unstemmed, so removal has to
    come first or the list stops matching.
    """
    stop, stem = _rules(config.language)

    def clean(text: str) -> str:
        tokens = PUNCTUATION.sub(" ", text.lower()).split()
        kept = [token for token in tokens if token not in stop]
        return " ".join(stem(token) for token in kept) if stem else " ".join(kept)

    return clean


def build_lexical_text(
    articles: pd.DataFrame, config: DatasetConfig, title_weight: int | None = None
) -> tuple[pd.Series, dict[str, int]]:
    """Title and abstract, cleaned, as one retrieval-ready field per article.

    An article with no abstract leaves its title standing alone rather than
    emptying the field — 5% of MIND and 8% of EB-NeRD have none, and those
    articles still have to be retrievable. "No abstract" means blank as well
    as null: EB-NeRD writes an absent subtitle as the empty string. Both that
    and the articles left with nothing at all after cleaning are counted for
    the caller to report.

    `title_weight` repeats the cleaned title, which raises the term frequency
    of everything in it before BM25 saturates it — the mechanism BM25F uses to
    say that a term in a headline means more than the same term buried in an
    abstract. Defaults to the registry's value; passed explicitly by the sweep,
    which varies it without rewriting the feature store.

    At weight 1 the result is character-for-character what concatenating the
    two fields and cleaning the whole produced, because cleaning is token-wise.
    A test pins that: every BM25 number on record was produced the old way.
    """
    if title_weight is None:
        title_weight = config.lexical.title_weight
    clean = cleaner(config)
    title = articles["title"].fillna("").str.strip().map(clean)
    abstract = articles["abstract"].fillna("").str.strip().map(clean)

    repeated = title if title_weight == 1 else title.map(
        lambda words: " ".join([words] * title_weight) if words else words
    )
    text = (repeated + " " + abstract).str.strip().astype("string")

    report = {
        "articles": len(articles),
        # Counted on the raw field: an abstract that existed but cleaned away
        # to nothing is a different fact from one that was never there.
        "missing_abstract": int(
            (articles["abstract"].fillna("").str.strip() == "").sum()
        ),
        "empty_after_cleaning": int((text == "").sum()),
    }
    return text, report


def run(config: DatasetConfig, force: bool = False) -> None:
    """Fill the article catalogue's lexical_text column.

    Always recomputed rather than skipped when the column is already there:
    like the split, the field is a pure function of columns that do not change,
    so a second run writes exactly what the first one did.
    """
    path = config.feature_store_dir / "articles.parquet"
    articles = pd.read_parquet(path)

    text, report = build_lexical_text(articles, config)
    articles["lexical_text"] = text
    articles.to_parquet(path, index=False)

    total = report["articles"]
    for count, what in (
        (report["missing_abstract"], "have no abstract and fall back to the title"),
        (report["empty_after_cleaning"], "are empty after cleaning"),
    ):
        share = 100 * count / total if total else 0.0
        print(f"    {count:,} of {total:,} articles ({share:.2f}%) {what}")
