"""Regression checks for masked event updates and seeded scheduling."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from figgie_jax.benchmark import benchmark, make_rollout
from figgie_jax.env import apply_event, reset
from figgie_jax.reference import transition
from figgie_jax.replay import snapshot
from figgie_jax.types import Config, Kind, action


@pytest.mark.parametrize("players,jitter,credit", [(4, 0, True), (5, 17, False)])
def test_batched_mixed_events_match_python_oracle_and_original_clock(players, jitter, credit):
    config = Config(
        num_players=players,
        jitter_ms=jitter,
        allow_negative_cash=credit,
        history_size=3,
        duration_ms=1000,
        latency_ms=tuple(range(players)),
    )
    keys = jax.random.split(jax.random.key(34), 8)
    states = jax.jit(jax.vmap(partial(reset, config=config)))(keys)
    oracles = [snapshot(jax.tree.map(lambda x, i=i: x[i], states)) for i in range(8)]
    execute = jax.jit(jax.vmap(partial(apply_event, config=config)))
    rng = np.random.default_rng(81)
    for event in range(42):
        acts, actors = [], []
        for i in range(8):
            actor = int(rng.integers(-1, players + 1))
            kind = event % 7
            suit = int(rng.integers(-1, 5))
            # Include repeated actual fills, so ring-buffer wrap is exercised.
            if i < 4:
                actor = (event // 2 + event % 2) % players
                kind = Kind.BID if event % 2 == 0 else Kind.SELL
                suit = 0
            side = int(kind in (Kind.ASK, Kind.BUY, Kind.CANCEL_ASK))
            quote_id = int(states.quote_ids[i, side, np.clip(suit, 0, 3)])
            if i == 7:
                quote_id += 1  # Stale acceptances must still be rejected.
            actors.append(actor)
            acts.append(action(kind, suit, 13, quote_id))
        at = min(event * 26, config.duration_ms)
        old = states
        result = execute(
            states,
            jnp.array(actors),
            jax.tree.map(lambda *xs: jnp.stack(xs), *acts),
            jnp.full(8, at, jnp.int32),
        )
        states = result.state
        for i in range(8):
            previous = oracles[i]
            expected, reason, accepted, traded = transition(
                previous, actors[i], tuple(map(int, acts[i])), at, config
            )
            oracles[i] = expected
            actual = snapshot(jax.tree.map(lambda x, i=i: x[i], states))
            for field, value in expected.items():
                np.testing.assert_allclose(actual[field], value, rtol=1e-6, atol=3e-5)
            assert int(result.info.reason[i]) == reason
            assert bool(result.info.accepted[i]) == accepted
            assert bool(result.info.traded[i]) == traded
            old_key = jax.tree.map(lambda x, i=i: x[i], old.key)
            expected_ready = np.asarray(old.ready_at[i]).copy()
            expected_key = old_key
            if not previous["done"] and at < config.duration_ms:
                delay_key, expected_key = jax.random.split(old_key)
                original_delay = (
                    config.decision_interval_ms
                    + jnp.array(config.latencies)
                    + jax.random.randint(delay_key, (players,), 0, jitter + 1)
                )
                seat = np.clip(actors[i], 0, players - 1)
                expected_ready[seat] = at + int(original_delay[seat])
            np.testing.assert_array_equal(states.ready_at[i], expected_ready)
            np.testing.assert_array_equal(
                jax.random.key_data(states.key[i]), jax.random.key_data(expected_key)
            )
    assert int(states.trade_count.sum()) > 12


def test_engine_benchmark_counts_resets_and_is_reproducible():
    config = Config(duration_ms=500)
    keys = jax.random.split(jax.random.key(3), 4)
    initial = jax.vmap(partial(reset, config=config))(keys)
    run = make_rollout(config, 20)
    final, key, counts = run(initial, jax.random.key(8))
    repeated, repeated_key, repeated_counts = run(initial, jax.random.key(8))
    np.testing.assert_array_equal(counts, repeated_counts)
    np.testing.assert_array_equal(np.asarray(counts)[[0, 2, 3]], [80, 16, 0])
    np.testing.assert_array_equal(final.hands, repeated.hands)
    np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(repeated_key))
    assert not final.done.any()


def test_engine_benchmark_rejects_empty_work():
    with pytest.raises(ValueError, match="positive"):
        benchmark(Config(), 0, 1, 1, 0)
    with pytest.raises(ValueError, match="positive"):
        make_rollout(Config(), 0)
