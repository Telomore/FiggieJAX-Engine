# Validate an independent release

This repository owns its package version, dependencies, examples, and Git history.
All release commands run from this checkout with its own dependency lockfile.

## Check the core installation

Use Python 3.12 and the committed uv lockfile:

```bash
uv sync --locked
uv run python scripts/validate_release.py
```

The validator checks version consistency and local documentation links. It runs
Ruff, the tests, complete four- and five-seat examples, and a small learning run.
It also forces event-cap truncations for both player counts and checks that
the examples report no terminal profit for those incomplete games.
The host and compiled learning smoke runs must produce a finite parameter change. They do not need to
improve its simulated evaluation score.

The validator builds a wheel and source archive with the locked Hatchling build
backend. It inspects their contents and rejects local credential files, browser
profiles, and model checkpoints. It extracts the source archive, rebuilds a wheel
using the same locked backend, and compares every wheel member with the checkout
build. It then installs the rebuilt wheel in an isolated uv
environment outside the checkout, using runtime versions exported from `uv.lock`.
Python runs with `-I` so the checkout cannot supply package imports. The core
check requires the optional frameworks, optimizer, and web-server dependencies to be absent.
The isolated installed wheel must also execute a compiled learning update.

Optional tests are skipped when their framework is absent. This is expected for
the core installation. All adapter tests must run in the integration check.

## Check all CPU framework integrations

```bash
uv sync --locked --extra all
uv run python scripts/validate_release.py --with-adapters
```

This mode requires Gymnasium, PettingZoo, and OpenSpiel before it runs tests. Its
isolated wheel installation also includes the three integrations. The `all`
extra does not install CUDA dependencies.

For a clean Docker check, mount the results directory so the record survives
container removal:

```bash
mkdir -p results
docker build --target release-check -t figgie-engine:release-check .
docker run --rm -v "$PWD/results:/app/results" figgie-engine:release-check
```

The core-only image uses the `test` target. To exercise the complete core release
validator in that image, override its default test command:

```bash
docker build --target test -t figgie-engine:test .
docker run --rm -v "$PWD/results:/app/results" --entrypoint uv figgie-engine:test \
  run --no-sync python scripts/validate_release.py
```

Each run writes to a new `results/release-*/` directory. `learning-smoke.json`
records simulated example settings and outcomes. `validation.json` records the
commands, source and artifact hashes, versions, and actual backend. These files
are ignored by Git and remain local. GitHub stores the source code; all
validation is run locally with uv and Docker.

The validator normally uses the network for a fresh wheel installation.
`--offline` works only after all required wheels are available in uv's cache.
Docker build-time cache mounts are not available inside a running container.
The default Docker targets validate Linux CPU execution. They do not validate
CUDA execution or GPU performance. No GitHub Actions or publishing
automation is configured.

## Build deliverable artifacts

After validation, keep a wheel and source archive with:

```bash
uv sync --locked
uv build --no-build-isolation --out-dir dist
```

`dist/` is ignored. Keep release metadata consistent across the package manifest,
lockfile, Python package, README, citation file, and changelog. Validate changes
to `CITATION.cff` against the [Citation File Format schema](https://github.com/citation-file-format/citation-file-format/blob/1.2.0/schema-guide.md).
`SOURCE.json`
describes the original upstream export, not the hashes of subsequent edits.
Use this repository's commits to identify later releases.

These commands build and validate locally. They do not create a remote repository,
push commits, publish a package, or publish a container.

## Inspect documentation diagrams

Keep Mermaid source in the Markdown documents. Render previews on the host with
Docker and the Mermaid CLI. The renderer is optional documentation tooling; it
is not an engine dependency. These commands write only to ignored `results/`:

```bash
mkdir -p results/diagrams
for source in README.md CONTRIBUTING.md docs/engine-api.md docs/parallel-collection.md \
  docs/gpu-throughput.md docs/league-gpu.md; do
  name=$(basename "$source" .md)
  docker run --rm --user "$(id -u):$(id -g)" \
    -v "$PWD:/source:ro" -v "$PWD/results/diagrams:/data" \
    ghcr.io/mermaid-js/mermaid-cli/mermaid-cli:11.12.0 \
    -i "/source/$source" -o "/data/$name.md" -e png -b white
done
```

Open each generated PNG. Check text, arrows, branch labels, and clipping. A
successful parse alone is not a visual inspection. Preview output is not part
of the wheel or source archive.

## Prepare a public source snapshot

Review the tracked files, source archive, and wheel for credentials, private
paths, personal data, conversation links, browser profiles, and generated
artifacts. Report affected paths without printing sensitive values. Ignore rules
only apply to untracked files. Pattern scans have limited coverage.

To publish from a private development repository, export the validated commit
with `git archive` and initialize a separate repository from the extracted
files. Create its first commit with an author and committer identity approved for
public use. Verify that its tracked tree matches the validated source, its commits
contain only the intended history, and its refs point to the intended commits.
Check that the repository and documentation URLs in `pyproject.toml` and the clone
command in [installation](installation.md) match the public destination.

If you plan to change the visibility of an existing GitHub repository, review all
hosted branches, tags, and Actions history first. GitHub exposes existing Actions
logs when a repository becomes public. See [GitHub's visibility guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility).
