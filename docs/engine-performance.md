# Engine optimization — 2026-11-03

The optimized engine measured **1.29 times the throughput** of the previous
engine on the specified RTX 2070 SUPER workload: 37.36 to 48.07 million submitted
opportunities per second. Both runs produced the same final-state SHA-256,
including simulator and policy random keys. These are measured execution times
for simulated games. They do not measure neural training speed or agent quality.

## Changes and invariants

The zero-jitter path no longer generates random integers that are always zero.
It preserves the original key splitting, including the next simulator key.
Nonzero jitter retains the original random draws.

Small state arrays now use broadcast masks instead of indexed scatters for card
transfers, cash, quotes, the trade-history row, and the acting seat's clock.
This changes compiled array operations, not the economic rules. Buyer and seller
are distinct for a valid fill. Cash remains float32. Config, public APIs, quote
IDs, history ordering, observation semantics, and checkpoint formats are unchanged.
The optimization applies wherever the engine is imported, including trainers.
It does not establish an end-to-end training speedup.

## Workload and method

- NVIDIA GeForce RTX 2070 SUPER, 8,192 MiB, Linux x86-64, driver 615.71.09.
  A desktop session remained active. No other training or GPU test ran alongside
  either timing process.
- Python 3.12.11, JAX/JAXlib 0.12.0.1. Exact NumPy version, source hashes, device
  kind, and all configuration values are in the compact record below.
- Same locked GPU image for both implementations:
  `sha256:d548f51482afa22319b6476e4b58f8b54a5b6134a1ece6619288f4d2cd90ff89`.
- Baseline: source revision `220922a` before the event changes. Candidate: masked
  updates and zero-jitter specialization. Benchmark script formatting differs
  between the stored source hashes; its workload and numerical logic are identical.
- 8,192 independent four-seat tables, 240-second rounds, history size 16, price
  cap 200, zero added latency/jitter, negative cash allowed, seed 8.
- Observation-only heuristic decisions, independent per-event policy keys, and
  256-opportunity scans. Finished tables reset to fresh 350-chip bankrolls.
  Counts include deadline opportunities, which settle without executing a trade.
- One initial compilation/warm-up call was excluded. Timing restarted from the
  same initial state and key. Each of 100 measured calls ended after
  `jax.block_until_ready`. Resets, policy decisions, and settlement were included;
  host checks and JSON writing were outside the timer.
- Five-call screens ran baseline then candidate. The longer confirmations ran
  candidate then baseline, each in a fresh process. The full timings below cover
  multiple completed rounds per table.

| Implementation | Opportunities | Trades | Completed rounds | Opportunities/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline confirmation | 209,715,200 | 1,870,205 | 49,152 | 37,363,808 |
| Optimized confirmation | 209,715,200 | 1,870,205 | 49,152 | 48,069,871 |

Rates use total opportunities divided by total measured seconds. Both checks
found conserved cards and cash, nonnegative inventories, finite cash, and zero
safety truncations. Matching state hashes add a deterministic comparison for
this workload; they do not replace conformance tests. See the
[complete compact record](benchmarks/rtx2070-engine-20261103.json).

### Verification for the benchmark figure

A fresh comparison on 2026-11-03 used the same GPU image, driver, batch, policy,
seed, and 100-call workload. It ran baseline then optimized engine in separate
CUDA containers, with no concurrent Figgie training or GPU test.

| Implementation | Opportunities/s | Completed rounds | Safety truncations |
| --- | ---: | ---: | ---: |
| Baseline verification | 39,778,626 | 49,152 | 0 |
| Optimized verification | 49,490,681 | 49,152 | 0 |

The fresh ratio is **1.24× (+24.4%)**. Each engine processed 209,715,200
opportunities and 1,870,205 trades. Both final-state hashes match the original
comparison. The compact record retains this pair under `verification`, including
timing samples, source hashes, image identity, and raw-report hashes.
This verification measures heuristic engine execution only.
The original comparison above remains unchanged. Variation between timing pairs
does not establish an uncertainty interval or a neural-training speedup.

The implementation follows JAX's guidance to
[synchronize benchmark work and separate compilation](https://docs.jax.dev/en/latest/benchmarking.html).
The reset guard sits outside the per-table `vmap`, where
[`lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)
can skip unused reset work.

## Reproduce

In the full repository, build the GPU target, then run:

```bash
mkdir -p results
docker run --rm --gpus device=0 --user "$(id -u):$(id -g)" \
  -e UV_CACHE_DIR=/tmp/uv-cache -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e PYTHONPATH=/app/src -v "$PWD/src:/app/src:ro" \
  -v "$PWD/results:/app/results" --entrypoint uv figgiejax:gpu \
  run --no-sync python -m figgie_jax.benchmark --backend gpu \
  --batch 8192 --steps 256 --repeats 100 --seed 8 \
  --output results/engine-candidate.json
```

To compare a baseline, export its engine modules to a separate directory. Copy
the current `benchmark.py` alongside them, mount that source instead, and run
the identical command in a new process. Do not run comparisons concurrently.
The standalone repository's README supplies equivalent image and CLI commands.
Use `--players 5` and `--jitter-ms` for other explicit workloads. Do not compare
different configurations as an implementation speedup.

## Limits

Only the specified four-seat GPU workload has a recorded speed comparison.
There is no CPU, MacBook, single-table, random-policy, or neural-training speed
claim. This pair of runs gives an observed ratio, not a confidence interval.
Larger histories and different hardware can change the mask/scatter tradeoff.
No long training run or learning-quality experiment was performed.

## Validation

- The full Docker CPU suite passed 281 tests, including web, replay, training,
  export boundaries, and conformance. The existing Starlette/httpx deprecation
  warning remains.
  `notebooks/01_environment.ipynb` also passed clean-kernel execution.
- CUDA passed 85 focused engine tests and 97 training/replay integration tests.
  One CPU-only fallback guard skipped on CUDA as intended. The new batched
  regression compares event outcomes and state with the independent Python oracle,
  and checks the original scheduling keys and delays for zero/nonzero jitter.
- The standalone test image passed all 89 tests and Ruff lint/format checks.
  The installed runtime passed dependency isolation, benchmark CLI, and scalar
  and batched API examples using only this package.
- The setup skill passed the skill-creator validator. Both workflow YAML files,
  documented shell syntax, export links, retained dependency versions, timing
  arithmetic, and current candidate source hashes were checked.

The standalone CUDA image target was not built separately. CUDA numerical checks
and timing used the existing locked GPU images with the current source mounted.
These are local checks.
