"""Synchronized engine-only throughput measurements, without neural policies."""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from figgie_jax.bots import heuristic_policy
from figgie_jax.env import apply_event, next_actor, reset
from figgie_jax.observations import observe
from figgie_jax.types import Config


def make_rollout(config: Config, steps: int):
    """Compile heuristic games, recycling settled or event-cap-truncated tables.

    Each table has an independent policy key stream. The returned counters are
    submitted opportunities, trades, completed rounds, and safety truncations.
    Reset work and policy work are included in the measured workload.
    The third counter includes truncations; subtract the fourth for settlements.
    """
    if steps < 1:
        raise ValueError("steps must be positive")

    def event(state, key):
        actor = next_actor(state)
        at_ms = jnp.maximum(state.time_ms, state.ready_at[actor])
        view = state._replace(time_ms=jnp.minimum(at_ms, config.duration_ms))
        act = heuristic_policy(key, observe(view, actor, config), config)
        return apply_event(state, actor, act, at_ms, config)

    def rollout(states, key):
        def advance(carry, _):
            states, key, totals = carry
            key, policy_key, reset_key = jax.random.split(key, 3)
            result = jax.vmap(event)(states, jax.random.split(policy_key, states.done.size))
            ended = result.state.done
            totals += jnp.array(
                [
                    states.done.size,
                    result.info.traded.sum(),
                    ended.sum(),
                    result.state.truncated.sum(),
                ],
                dtype=jnp.int32,
            )

            def replace_finished():
                fresh = jax.vmap(lambda k: reset(k, config))(
                    jax.random.split(reset_key, ended.size)
                )
                return jax.tree.map(
                    lambda new, old: jnp.where(
                        ended.reshape(ended.shape + (1,) * (old.ndim - 1)), new, old
                    ),
                    fresh,
                    result.state,
                )

            states = jax.lax.cond(jnp.any(ended), replace_finished, lambda: result.state)
            return (states, key, totals), None

        return jax.lax.scan(advance, (states, key, jnp.zeros(4, jnp.int32)), None, length=steps)[0]

    return jax.jit(rollout)


def benchmark(config: Config, batch: int, steps: int, repeats: int, seed: int) -> dict:
    """Measure synchronized steady-state calls; report compilation separately."""
    if min(batch, steps, repeats) < 1:
        raise ValueError("batch, steps, and repeats must be positive")
    key, reset_key = jax.random.split(jax.random.key(seed))
    states = jax.jit(jax.vmap(lambda k: reset(k, config)))(jax.random.split(reset_key, batch))
    jax.block_until_ready(states)
    run = make_rollout(config, steps)
    start = time.perf_counter()
    jax.block_until_ready(run(states, key))
    warmup = time.perf_counter() - start
    samples = []
    totals = np.zeros(4, np.int64)
    for _ in range(repeats):
        start = time.perf_counter()
        states, key, counts = jax.block_until_ready(run(states, key))
        samples.append(time.perf_counter() - start)
        totals += np.asarray(counts)
    np.testing.assert_array_equal(np.asarray(states.hands).sum(1), states.deck)
    assert np.all(np.asarray(states.hands) >= 0)
    assert np.isfinite(np.asarray(states.cash)).all()
    np.testing.assert_allclose(
        np.asarray(states.cash).sum(1) + 200,
        config.num_players * config.initial_cash,
        rtol=0,
        atol=1e-3,
    )
    if totals[3]:
        raise RuntimeError("Benchmark hit the safety event cap")
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves((states, key)):
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        digest.update(np.asarray(leaf).tobytes())
    root = Path(__file__).parent
    return {
        "config": asdict(config),
        "batch": batch,
        "steps": steps,
        "repeats": repeats,
        "seed": seed,
        "policy": "heuristic",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_kind": jax.devices()[0].device_kind,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "numpy": np.__version__,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("env.py", "types.py", "observations.py", "bots.py", "benchmark.py")
        },
        "warmup_seconds": warmup,
        "seconds": samples,
        "opportunities": int(totals[0]),
        "trades": int(totals[1]),
        "completed_rounds": int(totals[2]),
        "truncations": int(totals[3]),
        "opportunities_per_second": float(totals[0] / sum(samples)),
        "final_state_sha256": digest.hexdigest(),
    }


def main():
    """Run one explicit backend and batch; write a machine-readable timing record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--players", type=int, choices=(4, 5), default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--jitter-ms", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != args.backend:
        parser.error(f"Requested {args.backend}; actual backend is {jax.default_backend()}")
    report = benchmark(
        Config(num_players=args.players, jitter_ms=args.jitter_ms),
        args.batch,
        args.steps,
        args.repeats,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
