"""The three filesystem roots, and what each of them is for.

Nothing here touches the real tree: the roots are read from the environment at
import time, so the module is reloaded under a patched environment rather than
mutated in place.
"""

import importlib

import pytest

from pipeline import paths


def reloaded(monkeypatch, **environment):
    for key in ("VUB_NEWS_ROOT", "VUB_NEWS_DATA_ROOT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(paths)


@pytest.fixture(autouse=True)
def restore():
    """Whatever a test did to the environment, the module the rest of the suite
    imported has to come back describing the repo it actually lives in."""
    yield
    importlib.reload(paths)


def test_an_unset_environment_puts_everything_under_the_checkout(monkeypatch):
    """The default has to be exactly what it was before the roots existed, or
    every existing invocation changes behaviour."""
    module = reloaded(monkeypatch)

    assert module.ROOT == module.REPO_ROOT
    assert module.RAW_DIR == module.REPO_ROOT / "data" / "raw"
    assert module.FEATURE_STORE_DIR == module.REPO_ROOT / "feature_store"
    assert module.CHECKPOINT_DIR == module.REPO_ROOT / ".checkpoints"


def test_one_root_moves_the_whole_tree(monkeypatch, tmp_path):
    module = reloaded(monkeypatch, VUB_NEWS_ROOT=str(tmp_path))

    assert module.FEATURE_STORE_DIR == tmp_path / "feature_store"
    assert module.CHECKPOINT_DIR == tmp_path / ".checkpoints"
    assert module.ARTIFACTS_DIR == tmp_path / "artifacts"


def test_the_data_root_moves_only_the_large_directories(monkeypatch, tmp_path):
    """The case Ada forces: the feature store on a 2 TB disk that is purged
    weekly, the checkpoints on the 25 GB one that is not. A checkpoint records
    work worth hours and must not sit on the disk that empties."""
    small, large = tmp_path / "home", tmp_path / "scratch"

    module = reloaded(
        monkeypatch, VUB_NEWS_ROOT=str(small), VUB_NEWS_DATA_ROOT=str(large)
    )

    assert module.RAW_DIR == large / "data" / "raw"
    assert module.FEATURE_STORE_DIR == large / "feature_store"
    assert module.CHECKPOINT_DIR == small / ".checkpoints"
    assert module.ARTIFACTS_DIR == small / "artifacts"
    assert module.PREDICTIONS_DIR == small / "predictions"


def test_the_credential_stays_with_the_checkout(monkeypatch, tmp_path):
    """.env belongs beside the code, not beside wherever this run was told to
    write. Reading it from a relocated root would mean a token that works on a
    laptop and silently does not on a compute node."""
    module = reloaded(
        monkeypatch, VUB_NEWS_ROOT=str(tmp_path), VUB_NEWS_DATA_ROOT=str(tmp_path)
    )

    assert module.REPO_ROOT != tmp_path
    (tmp_path / ".env").write_text("HF_TOKEN=from_the_wrong_place\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    module.load_env_file()

    assert module.os.environ.get("HF_TOKEN") != "from_the_wrong_place"
