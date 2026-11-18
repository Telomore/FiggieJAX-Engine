---
name: setup-figgie-engine
description: Set up, validate, benchmark, or integrate the independent FiggieJAX engine and optional Gymnasium, PettingZoo, and OpenSpiel adapters. Use for private-agent integration and the small learning example, not a bundled production trainer or web-room setup.
---

# Set up FiggieJAX Engine

Read the target repository's `AGENTS.md` and `pyproject.toml` first. The standalone
distribution is `figgie-jax-engine`; its Python namespace is `figgie_jax`.
Avoid installing conflicting providers of the `figgie_jax` namespace.

For cloning, uv installation, runtime-only setup, consumer projects, and actual
platform scope, read [installation](../../../docs/installation.md). Use
`uv sync --locked --no-dev` for runtime only, and `uv run --no-sync` afterward.
`uv sync --locked` includes development tools. Framework extras are alternatives;
use `--extra all` when checking all three CPU adapters together.

This is an independent repository with its own Git history and release version.
Its `SOURCE.json` records initial source provenance. All setup commands run from
this checkout. If only this skill was installed, locate the user's engine checkout
before running repository-relative commands. Do not create or publish a remote
repository as an incidental setup action.

## Install and verify

Use Python 3.12 and uv. The lockfile contains the tested versions; do not upgrade
dependencies as an incidental setup step. Prefer the repository Docker targets:

```bash
docker build --target test -t figgie-engine:test .
docker run --rm figgie-engine:test
docker build --target runtime -t figgie-engine:runtime .
docker run --rm figgie-engine:runtime
docker build --target release-check -t figgie-engine:release-check .
docker run --rm figgie-engine:release-check
```

If native setup is requested, run `uv sync --locked`, then `uv run pytest`,
`uv run ruff check .`, and `uv run ruff format --check .`. The runtime needs JAX,
NumPy, and Pydantic; it does not need Optax, FastAPI, or training checkpoints.
There are no datasets to download and no notebook execution step. Optional extras
are `gymnasium`, `pettingzoo`, `openspiel`, and `all` for all CPU integrations.
Install only the required extra, for example `uv sync --locked --extra gymnasium`.
Default imports must not require any framework. The `all` extra does not enable CUDA.

For Linux x86-64 NVIDIA execution, build `--target gpu`, run with
`--gpus device=0`, and verify `jax.devices()`. The target selects CUDA explicitly.
Native GPU setup uses `uv sync --locked --extra gpu` and `JAX_PLATFORMS=cuda`.
If CUDA initialization fails, inspect the host driver and NVIDIA container runtime;
report the failure before switching the requested workload to CPU.
CPU validation is not GPU validation. Read the README for complete commands.

## Choose the execution path

Read the compiled JAX workflow and Mermaid diagrams in README.md. Use
`benchmark.make_rollout` for batched `jit`/`vmap`/`lax.scan` simulation. The native
`train_reinforce.py` example and optional framework adapters use host orchestration.
`examples/train_jax.py` compiles complete-round collection and repeated REINFORCE
updates. Read [compiled learning](../../../docs/compiled-learning.md) for Docker
GPU tests, training, held-out evaluation, and saved parameter artifacts. It is
educational, not a production league trainer. Keep all validation local.
Read [parallel collection](../../../docs/parallel-collection.md) for one or more
learning seats, frozen opponents, every-seat rewards, and selective resets.

For GPU prerequisites, batch sweeps, and memory settings, read
[GPU throughput](../../../docs/gpu-throughput.md). For an optional application-owned
trainer, read [the GPU league design](../../../docs/league-gpu.md).
This self-contained guide specifies an integration contract. It does not add a
trainer, optimizer, or checkpoint loader to the engine distribution.
In that design, scripts are JAX policies and neural opponents use an
application-owned frozen parameter bank. Keep measured engine rates separate from complete PPO update rates.

## Connect an agent

Read `docs/engine-api.md`, `docs/rl-tutorial.md`, `docs/observation-actions.md`,
and the relevant parts of `docs/rules.md`. `Config` is a
frozen static configuration. Use explicit JAX random keys, `FiggieEnv`, and `jit`;
use `vmap` over independent tables and `lax.scan` over events. Give decentralized
policies only `observe(state, player, config)`. Full state exposes hidden hands
and the goal. Schedule the actor with `next_actor`; advance the observation's
time to its opportunity before a time-sensitive policy acts.

Do not auto-reset inside `step`. Carry live rounds across rollout boundaries and
reset only done states. Distinguish event-cap truncation from settled completion.
`Transition.rewards` is terminal chip profit, including ante and payout once.
BUY/SELL actions need the displayed quote ID. Quotes never cross automatically.
Cash remains float32 for fractional ties. Added per-seat latency schedules
opportunities; it is not network or inference latency.

Gymnasium uses one learner against configured opponents. PettingZoo AEC selects
each scheduled actor; do not force a fixed seat rotation. Both host adapters live
in `figgie_jax_adapters`, outside compiled simulator transitions. OpenSpiel has
explicit chance handling and its own documented restrictions in `docs/openspiel.md`.
Do not claim information-state, solver, or throughput capabilities beyond that guide.
The examples in `examples/` explain complete rounds and a small REINFORCE update.
They establish execution, not strategic strength.

## Benchmark or hand off

Use `python -m figgie_jax.benchmark` (or `figgie-engine-benchmark` in the standalone
package) with explicit `--backend` and `--output`. Start with the README workload.
Read `docs/engine-performance.md` before comparing rates. The benchmark includes
scripted decisions and reset but no neural inference or optimizer. Use matching
settings and seeds, separate processes, synchronized timings, and exclude warm-up.
Do not launch a long training job as setup validation.

Report tested commands, actual backend, any unavailable dependencies, and artifact
paths. Treat `SOURCE.json` as initial-export provenance, not a hash of later edits.
Keep generated outputs in `results/`; preserve the MIT license and notice.
