# FiggieJAX Engine instructions

This standalone package contains the Figgie simulator, observation-only scripts,
replay support, a Python oracle, and an engine benchmark. Keep the simulator in
`src/figgie_jax/`, optional host adapters in `src/figgie_jax_adapters/`, examples in
`examples/`, tests in `tests/`, and explanations in `docs/`. Read
`.agents/skills/setup-figgie-engine/SKILL.md` for setup. A production trainer
and web service are not bundled.

`src/figgie_jax/learning.py` and `examples/train_jax.py` provide educational
compiled REINFORCE training. Keep it separate from production league claims.
Validate collection, every-seat rewards, masks, truncation exclusion, seeded
updates, and GPU backend selection. `test-gpu` runs the suite with CUDA required.

Use Python 3.12 and uv. `pyproject.toml` and `uv.lock` define dependencies.
Use Docker for validation: `docker build --target test -t figgie-engine:test .`
then `docker run --rm figgie-engine:test`. The test image runs `uv run ruff check .`,
`uv run ruff format --check .`, and `uv run pytest`. Validate the runtime image
after packaging changes. Do not infer GPU correctness or performance from CPU tests.
The `test-adapters` Docker target installs the optional CPU framework extras.
The `release-check` target also builds distribution artifacts and tests the wheel
outside the source tree. Run it before publication. These checks do not publish.
Keep framework dependencies optional. Never import Gymnasium, PettingZoo, or
OpenSpiel from the core engine or the adapters package initializer.
Keep native JAX transitions usable without host adapters. Keep the compiled
JAX workflow and Mermaid source in README.md accurate. GitHub is source storage;
keep validation local and do not add GitHub Actions or publishing automation. Adapter host loops are
not compiled JAX rollouts and do not inherit the engine's throughput claims.

Use Google-style Python docstrings, useful type hints, and Ruff formatting.
Keep JAX functions pure, static-shape, and compatible with `jit`, `vmap`, and
`lax.scan`. Keep configuration static and pass explicit random keys. Never put
host I/O or NumPy conversion inside compiled event processing.

Preserve the rules in `docs/rules.md`: four or five players, single-card trades,
global quote clearing, stale quote rejection, separate seat clocks, and fractional
pot shares. Cash stays floating point. Invalid events consume opportunities;
backward timestamps and completed states are absorbing. A deadline settles once.
An event-cap truncation does not settle. Test both player counts, all layouts,
credit modes, boundaries, parity with the Python oracle, JIT, and batching.
Private actor inputs come from `observe`, never the full privileged `State`.
Zero added latency and jitter remain the defaults. Preserve seeded scheduling.

GPU setup and batch tuning are in `docs/gpu-throughput.md`. The self-contained
league integration design is in `docs/league-gpu.md`. Distinguish implemented
engine commands from application-owned training components. Keep the GPU bank,
scripted policies, host boundaries, and README Mermaid overview accurate. Keep
setup independent of any other repository. Do not add trainer dependencies to
the standalone core as a documentation change.

No external data is required. Results are simulated. Keep generated replays,
benchmarks, and large artifacts in ignored `results/`. Record seeds, configuration,
source identity, package versions, and hardware. Synchronize benchmark samples,
exclude compilation, and compare identical workloads. Do not claim PPO speedups
from the engine-only benchmark. Keep README, skill, and API documentation current.
No notebooks are included; if added, import reusable code and verify clean execution.

Preserve user changes. Make coherent commits with semantic versions and run
relevant checks before committing. Never rewrite history or force-push. Inspect
status, branch, and remote before a push. `SOURCE.json` records initial export
provenance; use Git for subsequent changes. Never commit credentials or checkpoints.
This independent repository owns its version. Keep `pyproject.toml`, `uv.lock`,
`src/figgie_jax/__init__.py`, README, and citation metadata consistent.
Release this package independently.

Read `docs/installation.md` for runtime versus development setup and platform
scope. Read `docs/parallel-collection.md` for learner seat routing and every-seat
reward collection. Preserve current dependency pins unless a change is required.
The release validator must rebuild the wheel from the source archive and test
that wheel in an isolated consumer environment. Keep diagram source in Markdown;
render and inspect previews under ignored `results/diagrams/` as described in
`docs/release.md`. Keep credentials, browser profiles, and checkpoints out of
Git, Docker build contexts, and distributions. A privacy scan cannot prove that
history contains no secrets. Report historical exposures without printing values.
Obtain explicit user authorization before any history rewrite.

Make coherent versioned local commits after validation. Push only at the user's
explicit request. Do not add GitHub Actions, hosted CI, mandatory pull requests,
or branch-protection requirements. No notebooks are currently included.
