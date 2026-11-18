"""Randomized independent Python/JAX parity and conservation checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from figgie_jax.bots import random_policy
from figgie_jax.env import apply_event, reset
from figgie_jax.observations import observe
from figgie_jax.reference import transition
from figgie_jax.replay import snapshot
from figgie_jax.types import Config, action


@pytest.mark.parametrize("players,credit", [(4, True), (4, False), (5, True), (5, False)])
def test_randomized_economic_transition_parity(players, credit):
    config = Config(
        num_players=players,
        duration_ms=5000,
        max_price=30,
        history_size=4,
        allow_negative_cash=credit,
    )
    execute = jax.jit(lambda s, p, a, t: apply_event(s, p, a, t, config))
    policy = jax.jit(lambda k, s, p: random_policy(k, observe(s, p, config), config))
    rng = np.random.default_rng(20261027)
    trades = 0
    for episode in range(4):
        state = reset(jax.random.key(episode), config)
        oracle = snapshot(state)
        for i in range(100):
            actor = int(rng.integers(players))
            if i % 7:
                act = policy(jax.random.key(i + episode * 100), state, jnp.int32(actor))
            else:
                act = action(
                    int(rng.integers(9)),
                    int(rng.integers(-1, 5)),
                    int(rng.integers(-1, 33)),
                    int(rng.integers(10)),
                )
            at_ms = i * 51
            oracle, reason, accepted, traded = transition(
                oracle, actor, tuple(map(int, act)), at_ms, config
            )
            result = execute(state, actor, act, at_ms)
            state = result.state
            actual = snapshot(state)
            for field in actual:
                np.testing.assert_allclose(
                    actual[field],
                    oracle[field],
                    rtol=1e-6,
                    atol=3e-5,
                    err_msg=f"{players=} {episode=} {i=} {field=}",
                )
            assert int(result.info.reason) == reason
            assert bool(result.info.accepted) == accepted
            assert bool(result.info.traded) == traded
            trades += traded
            np.testing.assert_array_equal(state.hands.sum(0), state.deck)
            assert bool(jnp.all(state.hands >= 0))
            pot = 0 if bool(state.done) and not bool(state.truncated) else 200
            np.testing.assert_allclose(float(state.cash.sum()) + pot, players * 350, atol=3e-4)
    assert trades > 5
