# Present so pytest puts the repo root on sys.path and `import pipeline` works
# from tests/ without an editable install.

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def no_artifact_download(monkeypatch):
    """No test may fetch the MIND embedding artifact from Drive.

    Not hypothetical. These tests read MIND straight from the registry, so the
    moment ticket 7's `gdrive_file_id` was filled in with a real folder, two
    tests stopped asserting what they claimed and quietly pulled 100 MB into a
    tmp_path instead — one of them then passing for the wrong reason. A gdown
    that raises if it is called at all makes that impossible rather than
    merely unlikely, for tests not yet written as much as for these.
    """
    monkeypatch.setitem(
        sys.modules,
        "gdown",
        types.SimpleNamespace(
            download_folder=lambda **kwargs: pytest.fail(
                "a test reached the network to download an embedding artifact"
            )
        ),
    )
