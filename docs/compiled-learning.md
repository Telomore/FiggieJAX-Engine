# Train an educational agent entirely in JAX

`examples/train_jax.py` trains a linear, factorized masked policy against the
fixed `heuristic_policy`. Reusable code is in `src/figgie_jax/learning.py`.
It needs only the existing core dependencies. It is a bounded execution test
and learning example. It is not a production PPO or league trainer.

## Docker GPU validation

Use Linux x86-64 with a working NVIDIA driver and Container Toolkit. The GPU
image requires CUDA initialization. The script checks `--backend gpu` before
training, so a CPU fallback cannot count as a GPU result.

```bash
mkdir -p results
docker build --network host --target test-gpu -t figgie-engine:test-gpu .
docker run --rm --gpus device=0 -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  figgie-engine:test-gpu
docker build --network host --target gpu -t figgie-engine:gpu .
docker run --rm --gpus device=0 figgie-engine:gpu
docker run --rm --gpus device=0 -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -v "$PWD/results:/results" figgie-engine:gpu \
  uv run --no-sync python examples/train_jax.py --backend gpu \
  --players 4 --batch 64 --updates 64 --eval-games 256 --duration-ms 10000 \
  --seed 7 --output /results/train-jax-4.json
```

Repeat the training command with `--players 5` and a different output filename.
`--network host` is the documented Linux workaround for Docker bridge DNS
failures. It changes no dependency pins. Use `--user "$(id -u):$(id -g)"` and
`-e UV_CACHE_DIR=/tmp/uv-cache` if host-owned output files are required.

For an intentional CPU check, run the script in the `test` image with
`--backend cpu`. CPU results do not validate GPU execution. The GPU suite also
runs optional host adapters; those APIs remain host operations.

## What executes on the device

One `jit` contains an outer `lax.scan` over optimizer updates. Each update
collects one full round per independent table using an inner `lax.scan` and
`vmap`. The scheduled actor receives only its private observation at its
opportunity time. Kind, conditional suit, and conditional price heads sample
masked actions. BUY and SELL use the observed quote identifier. Each table
assigns the learner a random seat for that round. Other seats use the heuristic.

The collector retains private feature vectors, conditional masks, actions, and
learner decision flags. It records every seat's reward after every event.
It reads the learner's terminal component even if an opponent triggered
settlement. Padded events use absorbing completed states and contribute no
actor loss or duplicate reward. The scan length bounds the number of events
before the deadline using the positive decision interval, plus a settlement
event. A lower safety cap creates a marked truncation and no learning target.
Each optimizer update starts fresh rounds after the prior batch has finished.

The loss multiplies each complete episode's sum of learner log probabilities
by its terminal profit minus a leave-one-out baseline, divided by 100 chips.
The baseline uses other completed games in the batch. A singleton uses zero.
The loss averages over completed games, not padded events. An entropy bonus
averages over active decisions. JAX differentiates the policy loss, clips the
global gradient norm to 1, and applies SGD. Environment samples are fixed data
for differentiation. No gradient passes through game transitions.

Python handles arguments, compilation, calls, synchronization, and file I/O.
There is no Python, NumPy conversion, adapter helper, or host callback per game
event or optimizer update inside the compiled training computation. The engine
benchmark still measures scripted simulation rather than training speed.

## Reproduce and interpret a run

The JSON saves the configuration, seed, key domains, dependency versions,
device, source hashes, gradient norms, losses, completed games, parameter
change, and fixed held-out evaluation before and after training. Training
and evaluation use separate key domains. Initial and final evaluation reuse
the same deal, seat, and sampling keys. These keys give matched random streams;
different actions can still cause different trajectories.

The adjacent `.npz` contains `weights` and `initial_weights`. Its SHA-256 is
recorded in the JSON. Recreate the policy with the recorded `Config` and call
`sample_policy` with the saved weights and a private observation. This is a
parameter artifact, not an optimizer-resume checkpoint. Files remain local in
ignored `results/`. Source and lockfile hashes identify the implementation.

The script compiles, warms the exact training executable, discards warm-up
weights, then times a fresh run from the original weights and keys. It
synchronizes before stopping the clock. The reported training duration excludes
compilation, warm-up, evaluation, and saving files. The script rejects missing
updates, nonfinite arrays, and incomplete training or evaluation games.

These are simulated outcomes. A nonzero parameter change verifies an update.
It does not prove improved play. Shortened rounds, one linear policy, fixed
heuristic opponents, and a single seed limit interpretation. Larger experiments
need independent seeds, suitable uncertainty estimates, and stronger opponents.
No multi-GPU, recurrent, frozen-policy-bank, or production training claim follows
from this example.

## Local validation on 2026-11-17

Docker ran on the native Linux `default` context. JAX reported the NVIDIA
GeForce RTX 2070 SUPER with 8,192 MiB VRAM. The driver was 615.71.09.
Python was 3.12.11. JAX and JAXlib were 0.12.0.1.
The complete suite passed on CUDA: 160 tests, with four PettingZoo observation
format warnings. Ruff lint and format checks passed. The CPU release validator
also passed its tests, examples, source-archive rebuild, isolated wheel install,
and compiled learning update.

The following simulated runs used seed 7, batch 64, 64 SGD updates, and 4,096
training games each. Each initial and final evaluation used 256 held-out games.
All training and evaluation games settled. There were no truncations.
The price cap was 200 chips. Credit was enabled, so losses can exceed the
initial 350-chip bankroll. Mean terminal profits are in chips per game.

| Seats | Round length | Initial mean profit | Final mean profit |
| --- | --- | --- | --- |
| 4 | 10 seconds | -737.86 | -130.94 |
| 5 | 10 seconds | -760.22 | -117.64 |
| 4 | 240 seconds | -11,273.84 | -1,443.59 |
| 5 | 240 seconds | -12,671.54 | -1,804.58 |

These policies still lose against the fixed heuristic opponents. Lower losses
on this one matched evaluation sample do not establish strategic strength.
The standard-round training computations took 14.20 and 18.16 seconds for four
and five seats, respectively. These synchronized timings exclude compilation,
warm-up, evaluation, and output. They apply only to this small linear learner
and hardware. They are not PPO throughput measurements.

Local JSON reports, weights, test logs, and hardware records are under
`results/gpu-validation-20261117/`. The reports record source hashes and exact
settings. Generated artifacts remain outside Git and package distributions.
