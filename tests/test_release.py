"""Independent package versions and documentation must be release-consistent."""

import runpy
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = runpy.run_path(str(ROOT / "scripts/validate_release.py"))


def test_release_versions_agree():
    assert CHECKS["check_versions"](ROOT)


def test_standalone_documentation_links_resolve():
    CHECKS["check_local_links"](ROOT)


def test_source_rebuild_detects_missing_or_changed_wheel_contents(tmp_path):
    """An sdist must retain the code and metadata used by the checkout wheel."""
    original = tmp_path / "original.whl"
    rebuilt = tmp_path / "rebuilt.whl"
    with zipfile.ZipFile(original, "w") as archive:
        archive.writestr("figgie_jax/env.py", "original")
        archive.writestr("metadata", "test-version")
    for files, error in (
        ({"figgie_jax/env.py": "original"}, "file set"),
        ({"figgie_jax/env.py": "changed", "metadata": "test-version"}, "differs"),
    ):
        with zipfile.ZipFile(rebuilt, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        with pytest.raises(AssertionError, match=error):
            CHECKS["check_rebuilt_wheel"](original, rebuilt)
    CHECKS["check_rebuilt_wheel"](original, original)


@pytest.mark.parametrize(
    "name", ["src/.env", "src/.env.local", ".aws/config", "checkpoints/model.pt", "src/key.pem"]
)
def test_distribution_rejects_private_artifacts(name):
    with pytest.raises(AssertionError):
        CHECKS["check_public_files"]([name])
