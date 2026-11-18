"""Verify compiled on-policy updates, private masks, and complete-round rewards."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from figgie_jax.env import reset
from figgie_jax.learning import (
    Decisions,
    Games,
    collect,
    initialize,
    learner_profits,
    loss,
    make_train,
    sample_policy,
)
from figgie_jax.observations import action_masks, observe, price_mask
from figgie_jax.types import Config, Kind


@pytest.mark.parametrize("players,credit", [(4, True), (5, False)])
def test_compiled_complete_games_and_seeded_update(players, credit):
    config = Config(num_players=players, allow_negative_cash=credit, duration_ms=1000, max_price=20)
    weights = initialize(jax.random.key(1), config)
    collector = jax.jit(partial(collect, config=config, batch=4))
    games = collector(weights, jax.random.key(2))
    assert np.asarray(games.states.done).all()
    assert not np.asarray(games.states.truncated).any()
    np.testing.assert_allclose(games.rewards, games.states.cash - games.states.initial_cash)
    np.testing.assert_allclose(np.asarray(games.rewards).sum(1), 0, atol=1e-4)
    np.testing.assert_allclose(
        learner_profits(games), np.asarray(games.rewards)[np.arange(4), games.learner_seats]
    )
    train = make_train(config, batch=4, updates=2)
    updated, metrics = train(weights, jax.random.key(3))
    repeated, repeated_metrics = train(weights, jax.random.key(3))
    assert np.isfinite(np.asarray(updated)).all()
    assert float(jnp.linalg.norm(updated - weights)) > 0
    np.testing.assert_array_equal(metrics["completed_games"], [4, 4])
    np.testing.assert_array_equal(updated, repeated)
    for name in metrics:
        np.testing.assert_array_equal(metrics[name], repeated_metrics[name])
    # No Python callbacks or host I/O may enter the compiled training computation.
    lowered = train.lower(weights, jax.random.key(3)).as_text()
    assert "callback" not in lowered.lower()


def test_truncations_never_produce_a_training_target():
    config = Config(duration_ms=1000, max_events=1, max_price=20)
    weights = initialize(jax.random.key(4), config)
    updated, metrics = make_train(config, batch=4, updates=2)(weights, jax.random.key(5))
    np.testing.assert_array_equal(updated, weights)
    np.testing.assert_array_equal(metrics["truncated_games"], [4, 4])
    np.testing.assert_array_equal(metrics["completed_games"], [0, 0])
    np.testing.assert_array_equal(metrics["update_applied"], [False, False])
    np.testing.assert_array_equal(metrics["gradient_norm"], [0, 0])


def test_compiled_policy_uses_observable_masks_and_quote_ids():
    config = Config(num_players=5, max_price=20, allow_negative_cash=False)
    state = reset(jax.random.key(6), config)
    state = state._replace(
        prices=jnp.full((2, 4), 10),
        owners=jnp.ones((2, 4), jnp.int32),
        quote_ids=jnp.arange(1, 9).reshape(2, 4),
    )
    obs = observe(state, jnp.int32(0), config)
    weights = initialize(jax.random.key(7), config)
    selected, _ = jax.jit(jax.vmap(lambda key: sample_policy(weights, key, obs, config)))(
        jax.random.split(jax.random.key(8), 128)
    )
    kinds, suits = action_masks(obs, config)
    for kind, suit, price, quote in zip(*map(np.asarray, selected), strict=True):
        assert kinds[kind] and suits[kind, suit]
        assert price_mask(obs, kind, suit, config)[price]
        if kind in (Kind.BUY, Kind.SELL):
            assert quote == obs.quote_ids[int(kind == Kind.BUY), suit]


@pytest.mark.parametrize("profit", [-100.0, 100.0])
def test_terminal_reward_moves_policy_in_correct_direction(profit):
    """Reward must drive the update even when the entropy bonus is disabled."""
    config = Config(max_price=2)
    states = jax.tree.map(lambda x: x[None], reset(jax.random.key(9), config))
    states = states._replace(done=jnp.array([True]))
    weights = jnp.zeros((1, 14))
    masks = jnp.zeros((1, 1, 14), bool).at[..., 0:2].set(True)
    masks = masks.at[..., 7].set(True).at[..., 11].set(True)
    games = Games(
        states,
        jnp.array([0]),
        jnp.array([[profit, 0, 0, 0]]),
        Decisions(
            jnp.ones((1, 1, 1)), masks, jnp.zeros((1, 1, 3), jnp.int32), jnp.ones((1, 1), bool)
        ),
    )
    gradient = jax.jit(jax.grad(loss))(weights, games, 0.0)
    updated = weights - 0.1 * gradient
    probability = jax.nn.softmax(updated[0, :2])[0]
    assert (float(probability) - 0.5) * profit > 0
    np.testing.assert_array_equal(updated[:, 2:], weights[:, 2:])


def test_collection_bound_with_latency_and_jitter():
    config = Config(
        num_players=5,
        duration_ms=1000,
        max_price=20,
        latency_ms=(0, 50, 100, 150, 200),
        jitter_ms=50,
    )
    games = jax.jit(partial(collect, config=config, batch=4))(
        initialize(jax.random.key(10), config), jax.random.key(11)
    )
    assert np.asarray(games.states.done).all()
    assert not np.asarray(games.states.truncated).any()
    np.testing.assert_allclose(games.rewards, games.states.cash - games.states.initial_cash)


@pytest.mark.parametrize(
    "batch,updates,rate,entropy",
    [
        (0, 2, 0.01, 0.01),
        (2, 0, 0.01, 0.01),
        (2, 2, -1, 0.01),
        (2, 2, 0.01, float("nan")),
    ],
)
def test_invalid_training_settings(batch, updates, rate, entropy):
    with pytest.raises(ValueError):
        make_train(Config(), batch, updates, rate, entropy)
