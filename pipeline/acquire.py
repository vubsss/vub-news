"""Download and extract the raw archives a dataset needs.

Driven entirely by the registry: which archives, where they come from, where
their contents belong, and which files must exist afterwards.

Two properties this stage guarantees:

  * a partial download never gets the real filename — bytes land in a .part
    file and are renamed only once the archive is complete and readable, so an
    interrupted run leaves nothing that a later run would mistake for good data
  * an archive already on disk is opened before it is trusted, so a truncated
    file left behind by something else is re-fetched rather than extracted
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from pipeline.datasets import Archive, DatasetConfig

CHUNK_BYTES = 1 << 20
ARCHIVE_DIRNAME = "_archives"


class AcquisitionError(RuntimeError):
    """A failure the user has to act on: missing credential, unreachable source."""


def run(config: DatasetConfig, force: bool = False) -> None:
    if not force and not _missing(config):
        print(f"  all raw files for {config.name} are present")

    raw = config.raw
    ensure(config, raw.archives, raw.expected_files, raw.token_env, force)
    _report(config)


def ensure(
    config: DatasetConfig,
    archives: tuple[Archive, ...],
    expected_files: tuple[str, ...],
    token_env: str | None,
    force: bool = False,
) -> None:
    """Put one archive set on disk under the dataset's raw directory.

    `run` calls it for the registry's raw spec. The submission stage calls it
    for the competition's own test archive, which the pipeline never reads and
    which is far larger than everything the feature store is built from — so
    it stays off the rebuild path and is fetched only when a leaderboard file
    is actually being written.
    """
    if force or _absent(config, expected_files):
        token = _token(config, token_env)
        for archive in archives:
            _extract(archive, _fetch(archive, config, token), config)

    missing = _absent(config, expected_files)
    if missing:
        raise AcquisitionError(
            f"{config.name}: acquisition finished but these files the registry "
            f"declares are missing or empty:\n    " + "\n    ".join(missing)
        )


def _absent(config: DatasetConfig, expected_files: tuple[str, ...]) -> list[str]:
    return [name for name in expected_files if not _present(config.raw_dir / name)]


def _token(config: DatasetConfig, token_env: str | None) -> str | None:
    """The credential a source needs, if any."""
    if token_env is None:
        return None

    token = os.environ.get(token_env, "").strip()
    if not token:
        raise AcquisitionError(
            f"{config.name}: ${token_env} is not set.\n"
            f"  This source is a gated HuggingFace repo. Accept its terms while "
            f"logged in, create a read token, then either write it to the .env "
            f"file in the repo root:\n"
            f"      echo '{token_env}=hf_...' >> .env\n"
            f"  or export it in your shell:\n"
            f"      export {token_env}=hf_..."
        )
    return token


def _archive_dir(config: DatasetConfig) -> Path:
    return config.raw_dir / ARCHIVE_DIRNAME


def _missing(config: DatasetConfig) -> list[str]:
    return _absent(config, config.raw.expected_files)


def _present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _readable_zip(path: Path) -> bool:
    """True if the central directory is intact — cheap truncation check."""
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def _fetch(archive: Archive, config: DatasetConfig, token: str | None) -> Path:
    destination = _archive_dir(config) / archive.filename

    if destination.exists():
        if _readable_zip(destination):
            print(f"  have {archive.filename}")
            return destination
        print(f"  {archive.filename} is truncated or corrupt, re-downloading")
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        with requests.get(
            archive.url, headers=headers, stream=True, timeout=60
        ) as response:
            _check_response(response, archive, config)
            total = int(response.headers.get("content-length", 0))
            with (
                partial.open("wb") as handle,
                tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=f"  {archive.filename}",
                ) as progress,
            ):
                for chunk in response.iter_content(CHUNK_BYTES):
                    handle.write(chunk)
                    progress.update(len(chunk))
    except requests.RequestException as error:
        partial.unlink(missing_ok=True)
        raise AcquisitionError(
            f"{config.name}: downloading {archive.url} failed: {error}"
        ) from error

    if not _readable_zip(partial):
        partial.unlink()
        raise AcquisitionError(
            f"{config.name}: {archive.filename} downloaded but is not a readable "
            f"zip — the source may have returned an error page instead of the file."
        )

    partial.rename(destination)
    return destination


def _check_response(
    response: requests.Response, archive: Archive, config: DatasetConfig
) -> None:
    if response.status_code in (401, 403) and config.raw.token_env:
        raise AcquisitionError(
            f"{config.name}: {archive.url} refused the request "
            f"({response.status_code}).\n"
            f"  ${config.raw.token_env} is set but does not grant access. This is "
            f"a gated repo: visit the dataset page while logged in as the token's "
            f"owner, accept the terms, then retry."
        )
    response.raise_for_status()


def _is_junk(name: str) -> bool:
    """macOS resource forks. Both EB-NeRD archives are full of them."""
    parts = name.split("/")
    return "__MACOSX" in parts or ".DS_Store" in parts


def _common_top_level(names: list[str]) -> str | None:
    """The single directory every entry sits under, if there is one."""
    tops = {name.split("/", 1)[0] for name in names if name.strip("/")}
    if len(tops) != 1:
        return None
    top = tops.pop()
    if not all(name == top or name.startswith(f"{top}/") for name in names):
        return None
    return top


def _extract(archive: Archive, path: Path, config: DatasetConfig) -> None:
    target = config.raw_dir / archive.extract_to
    target.mkdir(parents=True, exist_ok=True)
    resolved_target = target.resolve()

    with zipfile.ZipFile(path) as zipped:
        names = [n for n in zipped.namelist() if not _is_junk(n)]
        strip = _common_top_level(names)
        members = [
            m for m in zipped.infolist() if not m.is_dir() and not _is_junk(m.filename)
        ]

        for member in tqdm(members, desc=f"  extracting {archive.filename}"):
            name = member.filename
            if strip:
                name = name[len(strip) + 1 :]
            if not name:
                continue

            out = target / name
            if not out.resolve().is_relative_to(resolved_target):
                raise AcquisitionError(
                    f"{config.name}: {archive.filename} contains a path that "
                    f"escapes its target directory: {member.filename}"
                )

            out.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, out.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def _report(config: DatasetConfig) -> None:
    """What the raw spec left on disk, and what it cost to keep."""
    total = 0
    for name in config.raw.expected_files:
        size = (config.raw_dir / name).stat().st_size
        total += size
        print(f"    {size / 1e6:9.1f} MB  {name}")

    archives = sum(
        path.stat().st_size for path in _archive_dir(config).glob("*") if path.is_file()
    )
    print(
        f"  {config.name}: {len(config.raw.expected_files)} files, "
        f"{total / 1e6:.0f} MB extracted, {archives / 1e6:.0f} MB of archives kept "
        f"for re-runs"
    )
