"""The registry is the contract every stage reads from, so it is tested."""

import dataclasses

import pytest

from pipeline.datasets import (
    ARTICLE_COLUMNS,
    BEHAVIOR_COLUMNS,
    DATASETS,
    HISTORY_COLUMNS,
)

CONFIGS = list(DATASETS.values())

# Fields that are legitimately unset until a later ticket fills them in.
INCOMPLETE = {"gdrive_file_id", "token_env"}

# Fields that apply to some datasets and not others. Blanket "must be set" says
# nothing useful about these; the rule each one really obeys is its own test.
CONDITIONAL = {"max_tokens"}


def test_every_config_exposes_the_same_fields():
    field_sets = [
        {f.name for f in dataclasses.fields(config)} for config in CONFIGS
    ]
    assert all(fields == field_sets[0] for fields in field_sets)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
def test_every_field_is_populated(config):
    """Catches adding a field and filling it in for only one dataset."""
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        assert value is not None, f"{config.name}.{field.name} is unset"
        assert value != "", f"{config.name}.{field.name} is empty"


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "table,expected",
    [
        ("articles", ARTICLE_COLUMNS),
        ("behaviors", BEHAVIOR_COLUMNS),
        ("history", HISTORY_COLUMNS),
    ],
)
def test_column_map_covers_the_unified_schema(config, table, expected):
    mapping = getattr(config.columns, table)
    assert set(mapping) == set(expected)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
def test_raw_spec_is_usable(config):
    filenames = [archive.filename for archive in config.raw.archives]
    assert len(filenames) == len(set(filenames))
    assert config.raw.expected_files

    for spec in (config.raw, config.embeddings, config.submission):
        for field in dataclasses.fields(spec):
            if field.name in INCOMPLETE | CONDITIONAL:
                continue
            value = getattr(spec, field.name)
            assert value is not None, f"{config.name}: {field.name} is unset"


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
def test_a_dataset_that_encodes_declares_the_width_it_encodes_at(config):
    """`max_tokens` reaches the tokenizer as `max_length`, and None there means
    the tokenizer's own default -- 512 for BERT, where all-MiniLM-L6-v2 was
    published and trained at 256. Nothing downstream could tell: the vectors
    would still be finite, unit length and 384 wide, just not the model's."""
    spec = config.embeddings
    if spec.kind == "generate":
        assert spec.max_tokens, f"{config.name} encodes but declares no width"
    else:
        assert spec.max_tokens is None, (
            f"{config.name} does not encode, so a width is misleading"
        )


def test_datasets_write_to_distinct_directories():
    for attr in ("raw_dir", "feature_store_dir", "artifacts_dir"):
        dirs = {getattr(config, attr) for config in CONFIGS}
        assert len(dirs) == len(CONFIGS)
