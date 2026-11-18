"""Official-rule fixtures, event boundaries, scheduling, and JAX transformations."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from figgie_jax.env import DECK_LAYOUTS, apply_event, next_actor, reset, settle, step
from figgie_jax.observations import action_masks, features, observe, price_mask
from figgie_jax.types import Config, Kind, Reason, action


@pytest.fixture(scope="module", params=[4, 5])
def config(request):
    return Config(num_players=request.param, duration_ms=10_000)


@pytest.fixture(scope="module")
def event(config):
    return jax.jit(partial(apply_event, config=config))


@pytest.fixture
def state(config):
    return reset(jax.random.key(7), config)


@pytest.mark.parametrize("layout", range(12))
def test_all_layouts_and_goal(layout):
    deck = np.asarray(DECK_LAYOUTS[layout])
    assert sorted(deck) == [8, 10, 10, 12]
    goal = int(np.argmax(deck)) ^ 1
    assert deck[goal] in (8, 10)
    assert goal // 2 == int(np.argmax(deck)) // 2
    assert len(np.unique(np.asarray(DECK_LAYOUTS), axis=0)) == 12


def test_prior_weights_are_one_third_short_goal():
    goals = np.argmax(np.asarray(DECK_LAYOUTS), axis=1) ^ 1
    assert sum(int(row[goal]) == 8 for row, goal in zip(DECK_LAYOUTS, goals, strict=True)) == 4


def test_deal_ante_and_reproducibility(state, config):
    np.testing.assert_array_equal(state.hands.sum(0), state.deck)
    np.testing.assert_array_equal(state.hands.sum(1), 40 // config.num_players)
    np.testing.assert_allclose(state.cash, 350 - 200 / config.num_players)
    repeated = reset(jax.random.key(7), config)
    for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(repeated), strict=True):
        if jax.dtypes.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(left, right)


def test_reset_jit_vmap(config):
    states = jax.jit(jax.vmap(partial(reset, config=config)))(
        jax.random.split(jax.random.key(4), 128)
    )
    np.testing.assert_array_equal(states.hands.sum((1, 2)), np.full(128, 40))
    assert states.hands.shape == (128, config.num_players, 4)


def test_quote_and_manual_acceptance(state, event):
    suit = int(jnp.argmax(state.hands[1]))
    result = event(state, 1, action(Kind.ASK, suit, 12), 100)
    assert result.info.accepted and not result.info.traded
    ask = result.state
    crossing = event(ask, 0, action(Kind.BID, suit, 20), 101)
    assert crossing.info.accepted and not crossing.info.traded
    traded = event(crossing.state, 0, action(Kind.BUY, suit, 0, ask.quote_ids[1, suit]), 102)
    assert traded.info.traded
    np.testing.assert_array_equal(traded.state.hands[0], state.hands[0] + jax.nn.one_hot(suit, 4))
    np.testing.assert_allclose(traded.state.cash[0], state.cash[0] - 12)
    np.testing.assert_allclose(traded.state.cash[1], state.cash[1] + 12)
    assert jnp.all(traded.state.owners == -1)
    assert jnp.all(traded.state.quote_ids == 0)
    assert jnp.all(traded.state.prices == 0)
    assert jnp.all(traded.rewards == 0)


def test_sell_transfers_single_card_and_globally_clears(state, event):
    suit = int(jnp.argmax(state.hands[0]))
    s = event(state, 1, action(Kind.BID, suit, 9), 0).state
    for other_suit in range(4):
        s = event(s, 2, action(Kind.BID, other_suit, 8), 1).state
    result = event(s, 0, action(Kind.SELL, suit, 0, s.quote_ids[0, suit]), 2)
    assert result.info.traded
    assert int(result.state.hands[0, suit]) == int(state.hands[0, suit]) - 1
    assert int(result.state.hands[1, suit]) == int(state.hands[1, suit]) + 1
    assert int(result.state.hands.sum()) == 40
    assert jnp.all(result.state.owners == -1)
    np.testing.assert_allclose(result.state.cash.sum(), state.cash.sum())


def test_stale_acceptance_after_price_replacement(state, event):
    suit = int(jnp.argmax(state.hands[1]))
    s = event(state, 1, action(Kind.ASK, suit, 15), 0).state
    old_id = s.quote_ids[1, suit]
    s = event(s, 1, action(Kind.ASK, suit, 5), 1).state
    rejected = event(s, 0, action(Kind.BUY, suit, 0, old_id), 2)
    assert not rejected.info.traded and rejected.info.reason == Reason.STALE_QUOTE
    np.testing.assert_array_equal(rejected.state.hands, state.hands)


def test_quote_improvement_and_own_replacement(state, event):
    s = event(state, 0, action(Kind.BID, 0, 10), 0).state
    assert not event(s, 1, action(Kind.BID, 0, 10), 1).info.accepted
    assert not event(s, 1, action(Kind.BID, 0, 9), 1).info.accepted
    assert event(s, 1, action(Kind.BID, 0, 11), 1).info.accepted
    assert event(s, 0, action(Kind.BID, 0, 1), 1).info.accepted


def test_cancellation_requires_ownership(state, event):
    s = event(state, 0, action(Kind.BID, 0, 10), 0).state
    assert not event(s, 1, action(Kind.CANCEL_BID, 0), 1).info.accepted
    result = event(s, 0, action(Kind.CANCEL_BID, 0), 1)
    assert result.info.accepted and result.state.owners[0, 0] == -1


def test_self_trade_and_empty_seller_rejected(state, event):
    s = event(state, 0, action(Kind.BID, 0, 10), 0).state
    assert not event(s, 0, action(Kind.SELL, 0, 0, s.quote_ids[0, 0]), 1).info.accepted
    empty = s._replace(hands=s.hands.at[1, 0].set(0))
    assert not event(empty, 1, action(Kind.ASK, 0, 10), 1).info.accepted
    assert not event(empty, 1, action(Kind.SELL, 0, 0, s.quote_ids[0, 0]), 1).info.accepted


@pytest.mark.parametrize(
    "kind,suit,price,actor",
    [
        (1, -1, 10, 0),
        (1, 4, 10, 0),
        (1, 0, 0, 0),
        (1, 0, 201, 0),
        (7, 0, 10, 0),
        (1, 0, 10, -1),
        (1, 0, 10, 99),
    ],
)
def test_invalid_inputs_do_not_mutate_economics(kind, suit, price, actor, state, event):
    result = event(state, actor, action(kind, suit, price), 1)
    assert not result.info.accepted
    np.testing.assert_array_equal(result.state.hands, state.hands)
    np.testing.assert_array_equal(result.state.cash, state.cash)
    assert jnp.all(result.state.owners == -1)


def test_exact_deadline_settles_and_is_idempotent(state, config, event):
    suit = int(jnp.argmax(state.hands[1]))
    s = event(state, 1, action(Kind.ASK, suit, 15), 0).state
    result = event(s, 0, action(Kind.BUY, suit, 0, s.quote_ids[1, suit]), config.duration_ms)
    assert result.state.done and not result.info.traded and not result.state.truncated
    np.testing.assert_array_equal(result.state.hands, state.hands)
    np.testing.assert_allclose(result.state.cash.sum(), config.num_players * 350, atol=2e-4)
    np.testing.assert_allclose(result.rewards, result.state.cash - state.initial_cash)
    repeated = event(result.state, 0, action(), config.duration_ms + 1)
    np.testing.assert_array_equal(repeated.state.cash, result.state.cash)
    assert jnp.all(repeated.rewards == 0)
    np.testing.assert_array_equal(settle(result.state).cash, result.state.cash)


def test_backward_time_is_rejected(state, event):
    s = event(state, 0, action(), 10).state
    result = event(s, 0, action(Kind.BID, 0, 20), 9)
    assert result.info.reason == Reason.TIME_REVERSED
    assert result.state.events == s.events and result.state.time_ms == 10


@pytest.mark.parametrize(
    "counts,expected",
    [
        ([1, 3, 5, 1], [10, 30, 150, 10]),
        ([1, 3, 0, 1, 3], [10, 90, 0, 10, 90]),
        ([2, 2, 2, 2], [50, 50, 50, 50]),
        ([3, 3, 3, 1], [30 + 100 / 3, 30 + 100 / 3, 30 + 100 / 3, 10]),
    ],
)
def test_official_and_fractional_payouts(counts, expected):
    cfg = Config(num_players=len(counts))
    s = reset(jax.random.key(0), cfg)
    # Only goal-column counts matter for this isolated settlement fixture.
    s = s._replace(hands=s.hands.at[:, s.goal].set(jnp.array(counts)), cash=jnp.zeros(len(counts)))
    result = jax.jit(settle)(s)
    np.testing.assert_allclose(result.cash, expected, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(result.cash.sum(), 200, atol=3e-5)


def test_safety_cap_is_truncation_without_payout():
    cfg = Config(max_events=1)
    s = reset(jax.random.key(0), cfg)
    result = jax.jit(partial(step, config=cfg))(s, action())
    assert result.state.done and result.state.truncated
    assert jnp.all(result.rewards == 0)
    np.testing.assert_array_equal(result.state.cash, s.cash)


def test_heterogeneous_latency_is_event_driven():
    cfg = Config(latency_ms=(0, 1000, 2000, 3000), decision_interval_ms=100)
    s = reset(jax.random.key(1), cfg)
    execute = jax.jit(partial(step, config=cfg))
    assert next_actor(s) == 0
    for i in range(5):
        assert next_actor(s) == 0
        s = execute(s, action()).state
        assert s.time_ms == (i + 1) * 100
    assert s.ready_at[1] == 1100


def test_zero_latency_progress_and_jitter_reproducibility():
    cfg = Config(jitter_ms=10)
    s = reset(jax.random.key(0), cfg)
    assert jnp.all(s.ready_at >= 250) and jnp.all(s.ready_at <= 260)
    result = jax.jit(partial(step, config=cfg))(s, action())
    assert result.state.time_ms >= 250
    np.testing.assert_array_equal(s.ready_at, reset(jax.random.key(0), cfg).ready_at)


def test_private_observations_do_not_leak_hidden_state(state, config):
    obs = observe(state, 0, config)
    other = state._replace(
        hands=state.hands.at[1:].set(99), deck=jnp.array([1, 2, 3, 4]), goal=(state.goal + 1) % 4
    )
    other_obs = observe(other, 0, config)
    for left, right in zip(obs, other_obs, strict=True):
        np.testing.assert_array_equal(left, right)
    assert jnp.isfinite(features(obs, config)).all()
    kinds, suits = action_masks(obs, config)
    assert kinds[Kind.WAIT] and not kinds[Kind.BUY]
    assert jnp.all(suits[Kind.ASK] == (state.hands[0] > 0))
    assert not price_mask(obs, Kind.BID, 0, config)[0]


def test_credit_policy_is_explicit():
    for credit in (False, True):
        cfg = Config(allow_negative_cash=credit)
        s = reset(jax.random.key(0), cfg)._replace(cash=jnp.full(4, -5.0))
        result = jax.jit(partial(apply_event, config=cfg))(s, 0, action(Kind.BID, 0, 1), 0)
        assert bool(result.info.accepted) == credit


def test_bankroll_carries_across_rounds(config):
    cash = jnp.arange(config.num_players, dtype=jnp.float32) * 100 - 200
    s = reset(jax.random.key(0), config, cash)
    np.testing.assert_allclose(s.initial_cash, cash)
    np.testing.assert_allclose(s.cash, cash - 200 / config.num_players)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 3},
        {"duration_ms": 0},
        {"decision_interval_ms": 0},
        {"latency_ms": (1,)},
        {"latency_ms": (-1, 0, 0, 0)},
        {"initial_cash": float("nan")},
        {"max_price": 10001},
        {"history_size": 0},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_batched_steps_inside_scan(config):
    states = jax.vmap(partial(reset, config=config))(jax.random.split(jax.random.key(0), 8))

    def rollout(states):
        def body(states, _):
            results = jax.vmap(partial(step, config=config))(
                states, jax.tree.map(lambda x: jnp.broadcast_to(x, (8,)), action())
            )
            return results.state, results.rewards

        return jax.lax.scan(body, states, None, length=12)

    final, rewards = jax.jit(rollout)(states)
    assert final.hands.shape == (8, config.num_players, 4)
    np.testing.assert_array_equal(final.hands, states.hands)
    assert jnp.all(rewards == 0)
    assert jnp.all(final.time_ms > 0)


def test_history_wraps_in_chronological_order():
    cfg = Config(history_size=2)
    state = reset(jax.random.key(0), cfg)
    execute = jax.jit(partial(apply_event, config=cfg))
    suit = int(jnp.argmax(state.hands[1]))
    for i in range(3):
        state = execute(state, 1, action(Kind.ASK, suit, 10 + i), i * 10).state
        state = execute(
            state, 0, action(Kind.BUY, suit, 0, state.quote_ids[1, suit]), i * 10 + 1
        ).state
    obs = observe(state, 0, cfg)
    assert int(state.trade_count) == 3
    np.testing.assert_array_equal(obs.history[:, 0], [11, 21])
    np.testing.assert_array_equal(obs.history[:, 4], [11, 12])


@pytest.mark.parametrize("players", [4, 5])
def test_per_agent_latency_survives_batched_reset_and_scheduling(players):
    latencies = tuple(125 * seat for seat in range(players))
    cfg = Config(num_players=players, latency_ms=latencies)
    states = jax.jit(jax.vmap(partial(reset, config=cfg)))(jax.random.split(jax.random.key(7), 8))
    expected = np.asarray(latencies) + cfg.decision_interval_ms
    np.testing.assert_array_equal(states.ready_at, np.broadcast_to(expected, (8, players)))
    updated = jax.jit(jax.vmap(lambda s: step(s, action(), cfg).state))(states)
    assert jnp.all(updated.time_ms == expected[0])
    np.testing.assert_array_equal(updated.ready_at[:, 1:], states.ready_at[:, 1:])
    np.testing.assert_array_equal(updated.ready_at[:, 0], np.full(8, 2 * expected[0]))


@pytest.mark.parametrize("bad_value", [True, False, 0.5, -1, 10_000_000])
def test_invalid_per_agent_latency_values(bad_value):
    with pytest.raises(ValueError, match="latency"):
        Config(latency_ms=(0, bad_value, 0, 0))
