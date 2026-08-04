"""Acquisition is tested without touching the network."""

import dataclasses
import zipfile

import pytest

from pipeline import acquire, paths
from pipeline.acquire import AcquisitionError
from pipeline.datasets import DATASETS

MIND = DATASETS["mind"]
EBNERD = DATASETS["ebnerd"]


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path)
    return tmp_path


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_missing_token_names_the_variable(monkeypatch):
    monkeypatch.delenv(MIND.raw.token_env, raising=False)

    with pytest.raises(AcquisitionError, match=MIND.raw.token_env):
        acquire._token(MIND)


def test_blank_token_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv(MIND.raw.token_env, "   ")

    with pytest.raises(AcquisitionError, match=MIND.raw.token_env):
        acquire._token(MIND)


def test_public_source_needs_no_token():
    assert acquire._token(EBNERD) is None


def test_truncated_archive_is_not_trusted(tmp_path):
    good = make_zip(tmp_path / "good.zip", {"a.txt": "hello"})
    truncated = tmp_path / "truncated.zip"
    truncated.write_bytes(good.read_bytes()[:20])

    assert acquire._readable_zip(good)
    assert not acquire._readable_zip(truncated)


def test_wrapping_directory_is_stripped(raw):
    archive = dataclasses.replace(
        EBNERD.raw.archives[0], filename="wrapped.zip", extract_to=""
    )
    path = make_zip(
        raw / "wrapped.zip",
        {"ebnerd_small/articles.parquet": "x", "ebnerd_small/train/behaviors.parquet": "y"},
    )

    acquire._extract(archive, path, EBNERD)

    assert (EBNERD.raw_dir / "articles.parquet").read_text() == "x"
    assert (EBNERD.raw_dir / "train" / "behaviors.parquet").read_text() == "y"


def test_macos_junk_is_skipped_and_does_not_defeat_stripping(raw):
    """Both EB-NeRD archives ship __MACOSX entries next to the real top level."""
    archive = dataclasses.replace(
        EBNERD.raw.archives[1], filename="junky.zip", extract_to="embeddings"
    )
    path = make_zip(
        raw / "junky.zip",
        {
            "model/vectors.parquet": "v",
            "model/.DS_Store": "junk",
            "__MACOSX/model/._vectors.parquet": "junk",
        },
    )

    acquire._extract(archive, path, EBNERD)

    extracted = EBNERD.raw_dir / "embeddings"
    assert (extracted / "vectors.parquet").read_text() == "v"
    assert [p.name for p in extracted.rglob("*") if p.is_file()] == [
        "vectors.parquet"
    ]


def test_flat_archive_keeps_its_layout(raw):
    archive = dataclasses.replace(
        MIND.raw.archives[0], filename="flat.zip", extract_to="train"
    )
    path = make_zip(raw / "flat.zip", {"news.tsv": "n", "behaviors.tsv": "b"})

    acquire._extract(archive, path, MIND)

    assert (MIND.raw_dir / "train" / "news.tsv").read_text() == "n"
    assert (MIND.raw_dir / "train" / "behaviors.tsv").read_text() == "b"


def test_path_escaping_its_target_is_rejected(raw):
    archive = dataclasses.replace(
        MIND.raw.archives[0], filename="evil.zip", extract_to="train"
    )
    path = make_zip(raw / "evil.zip", {"../../escaped.txt": "no"})

    with pytest.raises(AcquisitionError, match="escapes"):
        acquire._extract(archive, path, MIND)

    assert not (raw.parent / "escaped.txt").exists()


def test_verify_rejects_an_empty_file(raw):
    for name in EBNERD.raw.expected_files:
        target = EBNERD.raw_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
    (EBNERD.raw_dir / EBNERD.raw.expected_files[0]).write_text("")

    with pytest.raises(AcquisitionError, match=EBNERD.raw.expected_files[0]):
        acquire._verify(EBNERD)


def populate(config):
    for name in config.raw.expected_files:
        target = config.raw_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
    acquire._archive_dir(config).mkdir(parents=True, exist_ok=True)


def test_run_downloads_nothing_when_every_file_is_present(raw, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("re-run tried to download")

    monkeypatch.setattr(acquire, "_fetch", fail)
    populate(EBNERD)

    acquire.run(EBNERD)


def test_force_reacquires_even_when_files_are_present(raw, monkeypatch):
    """Otherwise `build.py --force acquire` would silently do nothing."""
    fetched = []
    monkeypatch.setattr(
        acquire, "_fetch", lambda archive, config, token: fetched.append(archive)
    )
    monkeypatch.setattr(acquire, "_extract", lambda *args: None)
    populate(EBNERD)

    acquire.run(EBNERD, force=True)

    assert len(fetched) == len(EBNERD.raw.archives)
