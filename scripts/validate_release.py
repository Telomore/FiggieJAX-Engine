"""Validate source, examples, distributions, and an isolated installed wheel.

Run with uv from the repository root. This command never publishes artifacts.
Use --with-adapters after `uv sync --locked --extra all` for framework checks.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_versions(root: Path) -> str:
    """Require independent release metadata to identify the same version."""
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    lock = tomllib.loads((root / "uv.lock").read_text())
    locked = next(p for p in lock["package"] if p["name"] == "figgie-jax-engine")
    assert locked["version"] == version, "Lockfile version differs from package version"
    tree = ast.parse((root / "src/figgie_jax/__init__.py").read_text())
    imported_version = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    )
    assert imported_version == version, "Python package version differs from metadata"
    assert f"Version **{version}**" in (root / "README.md").read_text()
    citation = (root / "CITATION.cff").read_text()
    assert re.search(rf'^version: ["\']?{re.escape(version)}["\']?\s*$', citation, re.M)
    return version


def check_local_links(root: Path) -> None:
    """Check local Markdown file links, including assistant setup instructions."""
    broken = []
    paths = [root / "README.md", root / "CONTRIBUTING.md", root / "AGENTS.md"]
    paths += list((root / "docs").rglob("*.md"))
    paths += list((root / ".agents").rglob("*.md"))
    for path in paths:
        for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", path.read_text()):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            local = target.split("#", 1)[0].strip("<>")
            if local and not (path.parent / local).is_file():
                broken.append(f"{path.relative_to(root)}: {target}")
    assert not broken, "Broken local links: " + "; ".join(broken)


def check_distributions(directory: Path, version: str) -> Path:
    """Inspect built contents before installing the wheel outside the source tree."""
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        check_public_files(names)
        metadata_path = next(n for n in names if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        assert metadata["Name"] == "figgie-jax-engine"
        assert metadata["Version"] == version
        assert metadata["License-Expression"] == "MIT"
        for license_file in ("LICENSE", "NOTICE.md"):
            assert any(name.endswith(f".dist-info/licenses/{license_file}") for name in names)
        extras = set(metadata.get_all("Provides-Extra", []))
        assert {"gymnasium", "pettingzoo", "openspiel", "all", "gpu"} <= extras
        for module in ("common", "gymnasium", "pettingzoo", "openspiel"):
            assert f"figgie_jax_adapters/{module}.py" in names
        assert "figgie_jax/env.py" in names
        assert "figgie_jax/learning.py" in names
        assert not any(
            n.startswith(("figgie_jax/training", "figgie_jax/server", "figgie_jax/league"))
            for n in names
        )
    with tarfile.open(sdists[0]) as archive:
        names = ["/".join(Path(n).parts[1:]) for n in archive.getnames()]
        check_public_files(names)
        for required in (
            "uv.lock",
            "README.md",
            "LICENSE",
            "NOTICE.md",
            "Dockerfile",
            "CONTRIBUTING.md",
            "CITATION.cff",
            "examples/rollout.py",
            "examples/train_reinforce.py",
            "examples/train_jax.py",
            "docs/compiled-learning.md",
            "scripts/validate_release.py",
            "docs/rl-tutorial.md",
            "docs/framework-adapters.md",
            "docs/release.md",
            "docs/installation.md",
            "docs/parallel-collection.md",
            ".agents/skills/setup-figgie-engine/SKILL.md",
        ):
            assert required in names, f"Missing sdist file: {required}"
        assert not any(
            Path(n).parts and Path(n).parts[0] in {".git", ".venv", "results", "dist"}
            for n in names
        )
    return wheel


def check_public_files(names: list[str]) -> None:
    """Reject local credentials, browser data, and model artifacts in distributions."""
    private_directories = {
        ".git",
        ".venv",
        ".aws",
        ".ssh",
        ".codex",
        "results",
        "dist",
        "browser-profiles",
        "checkpoints",
        ".auth",
    }
    private_suffixes = {".pem", ".key", ".ckpt", ".pt", ".pth", ".safetensors"}
    for name in names:
        path = Path(name)
        assert not private_directories.intersection(path.parts), f"Private artifact: {name}"
        assert path.name != ".env" and not path.name.startswith(".env."), name
        assert path.suffix not in private_suffixes, f"Private artifact: {name}"


def check_rebuilt_wheel(original: Path, rebuilt: Path) -> None:
    """Require the source archive to reproduce the wheel's complete contents."""
    with zipfile.ZipFile(original) as first, zipfile.ZipFile(rebuilt) as second:
        assert set(first.namelist()) == set(second.namelist()), "Rebuilt wheel file set differs"
        for name in first.namelist():
            assert first.read(name) == second.read(name), f"Rebuilt wheel differs: {name}"


def main() -> None:
    """Run bounded CPU release checks and save a local verification record."""
    import jax

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-adapters", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use uv's existing cache only; all dependency wheels must already be cached.",
    )
    args = parser.parse_args()
    uv = [shutil.which("uv") or "uv"] + (["--offline"] if args.offline else [])
    version = check_versions(ROOT)
    check_local_links(ROOT)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    record_directory = Path(tempfile.mkdtemp(prefix="release-", dir=results))
    commands = []

    def run(arguments: list[str], *, cwd: Path = ROOT) -> None:
        command = uv + arguments
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=cwd, check=True)
        commands.append(arguments)

    if args.with_adapters:
        run(
            [
                "run",
                "--no-sync",
                "python",
                "-c",
                "import gymnasium, pettingzoo, pyspiel; print('All adapter extras available')",
            ]
        )
    run(["lock", "--check"])
    run(["run", "--no-sync", "ruff", "check", "."])
    run(["run", "--no-sync", "ruff", "format", "--check", "."])
    run(["run", "--no-sync", "pytest"])
    documented_examples = ["README.md", "docs/engine-api.md"]
    if args.with_adapters:
        documented_examples += ["docs/framework-adapters.md", "docs/openspiel.md"]
    for name in documented_examples:
        blocks = re.findall(r"```python\n(.*?)```", (ROOT / name).read_text(), re.S)
        assert blocks, f"No runnable Python example found in {name}"
        print(f"Executing documented examples: {name}", flush=True)
        run(["run", "--no-sync", "python", "-c", "\n".join(blocks)])
    for players in (4, 5):
        round_output = record_directory / f"round-{players}.json"
        run(
            [
                "run",
                "--no-sync",
                "python",
                "examples/rollout.py",
                "--players",
                str(players),
                "--duration-ms",
                "1000",
                "--seed",
                "7",
                "--output",
                str(round_output),
            ]
        )
        summary = json.loads(round_output.read_text())
        assert summary["terminal"] and not summary["truncated"], "Example did not settle"
        assert len(summary["terminal_profit"]) == players
        assert all(math.isfinite(value) for value in summary["terminal_profit"])
        truncated_output = record_directory / f"truncated-{players}.json"
        run(
            [
                "run",
                "--no-sync",
                "python",
                "examples/rollout.py",
                "--players",
                str(players),
                "--seed",
                "7",
                "--max-events",
                "1",
                "--output",
                str(truncated_output),
            ]
        )
        summary = json.loads(truncated_output.read_text())
        assert summary["truncated"] and not summary["terminal"], "Example did not truncate"
        assert summary["terminal_profit"] is None, "Truncation must not report terminal profit"
    run(
        [
            "run",
            "--no-sync",
            "python",
            "examples/train_reinforce.py",
            "--episodes",
            "2",
            "--eval-episodes",
            "2",
            "--duration-ms",
            "1000",
            "--seed",
            "7",
            "--output",
            str(record_directory / "learning-smoke.json"),
        ]
    )
    learning_smoke = json.loads((record_directory / "learning-smoke.json").read_text())["training"]
    assert learning_smoke["updates_applied"] > 0, "Learning smoke did not update the policy"
    change = learning_smoke["parameter_l2_change"]
    assert math.isfinite(change) and change > 0, "Learning smoke did not change finite parameters"
    run(
        [
            "run",
            "--no-sync",
            "python",
            "examples/train_jax.py",
            "--backend",
            "cpu",
            "--batch",
            "2",
            "--updates",
            "2",
            "--eval-games",
            "2",
            "--duration-ms",
            "1000",
            "--max-price",
            "20",
            "--output",
            str(record_directory / "compiled-learning-smoke.json"),
        ]
    )
    with tempfile.TemporaryDirectory(prefix="figgie-release-") as temporary:
        workspace = Path(temporary)
        dist = workspace / "dist"
        # The dev lock supplies Hatchling. Avoid resolving an unpinned build environment.
        for target in ("--sdist", "--wheel"):
            run(
                [
                    "build",
                    target,
                    "--no-build-isolation",
                    "--python",
                    sys.executable,
                    "--out-dir",
                    str(dist),
                ]
            )
        wheel = check_distributions(dist, version)
        # Build only from the archive, not from checkout files that may be absent
        # from a published source distribution. Use the same locked build backend.
        unpacked = workspace / "unpacked"
        with tarfile.open(next(dist.glob("*.tar.gz"))) as archive:
            archive.extractall(unpacked, filter="data")
        source_root = next(unpacked.iterdir())
        rebuilt_dist = workspace / "rebuilt"
        run(
            [
                "build",
                "--wheel",
                "--no-build-isolation",
                "--python",
                sys.executable,
                "--out-dir",
                str(rebuilt_dist),
            ],
            cwd=source_root,
        )
        rebuilt_wheel = next(rebuilt_dist.glob("*.whl"))
        check_rebuilt_wheel(wheel, rebuilt_wheel)
        requirements = workspace / "requirements.txt"
        export = [
            "export",
            "--quiet",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--output-file",
            str(requirements),
        ]
        if args.with_adapters:
            export += ["--extra", "all"]
        run(export)
        smoke = (
            "import importlib.metadata as m, importlib.util as u; "
            "import jax, figgie_jax, figgie_jax_adapters; "
            "from figgie_jax.learning import initialize, make_train; "
            "from figgie_jax.env import reset; from figgie_jax.types import Config; "
            f"assert m.version('figgie-jax-engine') == figgie_jax.__version__ == {version!r}; "
            "assert reset(jax.random.key(0), Config(num_players=4)).hands.shape == (4,4); "
            "assert reset(jax.random.key(1), Config(num_players=5)).hands.shape == (5,4); "
            "c = Config(duration_ms=500, max_price=20); "
            "w = initialize(jax.random.key(2), c); "
            "w2, metrics = make_train(c, 2, 1)(w, jax.random.key(3)); "
            "assert float(jax.numpy.linalg.norm(w2-w)) > 0; "
            "assert u.find_spec('figgie_jax.training') is None; "
            "assert u.find_spec('figgie_jax.server') is None; "
            "assert u.find_spec('optax') is None; assert u.find_spec('fastapi') is None; "
        )
        if args.with_adapters:
            smoke += (
                "import figgie_jax_adapters.gymnasium, figgie_jax_adapters.pettingzoo, "
                "figgie_jax_adapters.openspiel; "
            )
        else:
            smoke += (
                "assert u.find_spec('gymnasium') is None; "
                "assert u.find_spec('pettingzoo') is None; assert u.find_spec('pyspiel') is None; "
            )
        smoke += "print('Isolated wheel passed', figgie_jax.__version__, jax.devices())"
        run(
            [
                "run",
                "--isolated",
                "--no-project",
                "--python",
                sys.executable,
                "--with-requirements",
                str(requirements),
                "--with",
                str(rebuilt_wheel),
                "python",
                "-I",
                "-c",
                smoke,
            ],
            cwd=workspace,
        )
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dist.iterdir()}
        hashes[f"rebuilt/{rebuilt_wheel.name}"] = hashlib.sha256(
            rebuilt_wheel.read_bytes()
        ).hexdigest()
    report = record_directory / "validation.json"
    report.write_text(
        json.dumps(
            {
                "version": version,
                "with_adapters": args.with_adapters,
                "commands": commands,
                "artifact_sha256": hashes,
                "status": "passed",
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        ["jax", "jaxlib", "numpy", "pydantic", "pytest", "ruff", "hatchling"]
                        + (["gymnasium", "pettingzoo", "open-spiel"] if args.with_adapters else [])
                    )
                },
                "source_sha256": {
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(
                        list((ROOT / "src").rglob("*.py"))
                        + list((ROOT / "examples").rglob("*.py"))
                        + list((ROOT / "scripts").rglob("*.py"))
                        + list((ROOT / "tests").rglob("*.py"))
                        + list((ROOT / "docs").rglob("*.md"))
                        + [
                            ROOT / name
                            for name in (
                                "pyproject.toml",
                                "uv.lock",
                                "README.md",
                                "AGENTS.md",
                                "Dockerfile",
                                "CITATION.cff",
                                "LICENSE",
                                "NOTICE.md",
                            )
                        ]
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Release checks passed. Local record: {report}")


if __name__ == "__main__":
    main()
