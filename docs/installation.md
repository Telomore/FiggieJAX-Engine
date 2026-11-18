# Install the standalone engine

The distribution name is `figgie-jax-engine`. The Python namespaces are
`figgie_jax` and `figgie_jax_adapters`. This repository requires Python 3.12.
The committed lockfile defines the tested dependency versions. Do not upgrade
packages as part of installation.

## Clone and install uv

Install Git first. On Linux or macOS, install uv with the official installer:

```bash
curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh
```

Open a new shell so `uv` is on `PATH`. Version 0.8.22 matches the Docker image.
See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
for installer inspection, other installation methods, and Windows instructions.
The Windows installer does not establish engine or adapter compatibility.

```bash
git clone https://github.com/Telomore/FiggieJAX-Engine.git
cd FiggieJAX-Engine
uv python install 3.12
```

If you already have this checkout, use it. Do not copy files from another clone.
Record `git rev-parse HEAD` with each run. Uncommitted edits also affect results.

## Runtime checkout

Create `.venv` and install only the locked runtime dependencies:

```bash
uv sync --locked --no-dev
JAX_PLATFORMS=cpu uv run --no-sync python examples/rollout.py \
  --players 4 --duration-ms 10000 --seed 7 --output results/round.json
```

`uv sync` creates the environment. Shell activation is optional. `uv run
--no-sync` uses the environment that you just installed. It does not add the
default development group. Runtime dependencies are JAX, NumPy, and Pydantic,
plus their dependencies. No external data, trainer, or checkpoint is required.

The [complete-round example](../examples/rollout.py) splits explicit deal and
policy key streams. It selects the scheduled actor, advances the private
observation to decision time, and passes only that observation to the policy.
It carries the state until `done`. A settled round reports chip profit including
ante and payout. A safety-cap truncation reports `terminal_profit: null`.

Run both player counts and force an incomplete round to check the distinction:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python examples/rollout.py --players 5 --seed 7
JAX_PLATFORMS=cpu uv run --no-sync python examples/rollout.py \
  --players 4 --seed 7 --max-events 1 --output results/truncated-round.json
```

The truncation command completes successfully and reports `truncated: true`.
It does not claim that the game settled. The host loop in this readable example
is separate from the compiled batch collector shown in the README.

## Development and optional adapters

Install the locked development tools, then run local checks:

```bash
uv sync --locked
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
```

Choose an adapter extra during synchronization. Add `--no-dev` for runtime use:

```bash
uv sync --locked --extra gymnasium
uv sync --locked --extra pettingzoo
uv sync --locked --extra openspiel
uv sync --locked --extra all
```

Each command selects that set of extras; these commands are alternatives. `all`
means the three CPU adapters. It does not enable CUDA. Use `uv run --no-sync`
afterward to preserve the selected environment. See the
[adapter guide](framework-adapters.md) and [OpenSpiel restrictions](openspiel.md).
These adapters connect training frameworks to this simulator. No browser adapter
for the official Figgie website, browser login, or live-site agent is included.

## Install into another project

Use a separate Python 3.12 uv project for an application or trainer. From a
sibling application directory, add this checkout as an editable dependency:

```bash
uv init --python 3.12 figgie-consumer
cd figgie-consumer
uv add --editable ../FiggieJAX-Engine
uv run python -c 'import figgie_jax; print(figgie_jax.__version__)'
```

Run the first command from the directory that contains the engine checkout.
The path dependency is local and records a uv source entry. For a portable
application, distribute a verified wheel or use a Git dependency pinned to a
full commit ID. Do not depend on a moving branch for a reproducible run.

After [building a wheel](release.md), an application can instead run
`uv add ../FiggieJAX-Engine/dist/figgie_jax_engine-0.16.1-py3-none-any.whl`.
To add an optional framework to a checkout dependency, use
`uv add --editable ../FiggieJAX-Engine --extra gymnasium`.
The application owns its lockfile. The engine's `uv.lock` is not a transitive
dependency lock. Review the application's resolved JAX version and run its own
integration tests. See [uv dependency sources](https://docs.astral.sh/uv/concepts/projects/dependencies/).

Do not install another package that provides `figgie_jax` in the same environment.
Older combined Figgie projects can use that namespace. Different distribution
names do not prevent Python files from colliding. Create a clean environment
and verify the imported file with
`uv run python -c 'import figgie_jax; print(figgie_jax.__file__)'`.
Keep application optimizers, training frameworks, and checkpoints in the
consumer project. The standalone engine does not require them.

## Platforms and devices

| Environment | Validation scope |
| --- | --- |
| Linux x86-64, Python 3.12, Docker CPU | Local core, adapter, examples, and distribution checks; see the saved release record |
| Linux x86-64, NVIDIA CUDA 13 | Optional engine path; historical RTX 2070 SUPER benchmark is documented separately |
| macOS, native Windows, WSL, Linux ARM64 | Not validated by this release review; no adapter or wheel compatibility claim |
| Other Python versions, AMD GPU, Apple GPU, TPU, multiple GPUs | Outside the tested configuration |

The engine is pure Python, but JAX and adapter wheels are platform-specific.
Do not infer platform support from a `py3-none-any` engine wheel. See
[JAX installation](https://docs.jax.dev/en/latest/installation.html).
The Docker base pins Python 3.12.11 and uv 0.8.22. `.python-version` selects the
3.12 series for native installs. Release records include the actual patch version.

For a Linux CPU check, use the README's runtime and test Docker targets. If a
local Docker bridge cannot resolve package hosts, use `docker build --network
host ...` on Linux. This is an environment workaround, not a dependency change.

For native NVIDIA execution on Linux x86-64:

```bash
uv sync --locked --no-dev --extra gpu
JAX_PLATFORMS=cuda uv run --no-sync python -c 'import jax; print(jax.devices())'
JAX_PLATFORMS=cuda uv run --no-sync figgie-engine-benchmark \
  --backend gpu --batch 32 --steps 64 --repeats 1 --seed 7 \
  --output results/gpu-smoke.json
```

Verify the NVIDIA device before running the workload. The `gpu` extra is limited
to Linux x86-64. A successful install on another platform does not enable CUDA.
See [GPU requirements and Docker commands](gpu-throughput.md) for driver,
container runtime, memory, and batch tuning. CPU tests do not validate GPU work.
