"""Independent payout enumeration and adversarial game-logic conformance."""

from fractions import Fraction
from functools import partial
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from figgie_jax.env import apply_event, reset
from figgie_jax.observations import action_masks, features, observe, relative_features
from figgie_jax.reference import transition
from figgie_jax.types import Action, Config, Kind, Reason, action


def compositions(total, seats):
    """Enumerate every ordered nonnegative holding vector with a fixed total."""
    if seats == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in compositions(total - first, seats - 1):
                yield (first, *rest)


@pytest.mark.parametrize("players", [4, 5])
def test_every_goal_allocation_in_every_deck_settles_exactly_once(players):
    config = Config(num_players=players)
    template = reset(jax.random.key(41), config)
    hands, decks, goals, expected = [], [], [], []
    # Derive decks and color partners independently from the engine constants.
    for deck in sorted(set(permutations((8, 10, 10, 12)))):
        goal = (1, 0, 3, 2)[deck.index(12)]
        for counts in compositions(deck[goal], players):
            hand = np.zeros((players, 4), np.int32)
            hand[0] = deck
            hand[:, goal] = counts
            payouts = [Fraction(10 * count) for count in counts]
            remaining = Fraction(200) - sum(payouts)
            winners = [seat for seat, count in enumerate(counts) if count == max(counts)]
            for seat in winners:
                payouts[seat] += remaining / len(winners)
            assert sum(payouts) == 200
            hands.append(hand)
            decks.append(deck)
            goals.append(goal)
            expected.append([float(value) for value in payouts])
    assert len(hands) == {4: 2948, 5: 9988}[players]
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (len(hands), *x.shape)), template)
    states = states._replace(
        hands=jnp.array(hands),
        deck=jnp.array(decks),
        goal=jnp.array(goals),
        cash=jnp.zeros_like(states.cash),
        initial_cash=jnp.zeros_like(states.cash),
    )
    execute = jax.jit(
        jax.vmap(
            lambda s: apply_event(
                s, jnp.int32(0), action(Kind.BID, 0, 20), jnp.int32(config.duration_ms), config
            )
        )
    )
    result = execute(states)
    np.testing.assert_allclose(result.state.cash, expected, rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result.state.cash).sum(1), 200, rtol=0, atol=3e-5)
    np.testing.assert_array_equal(result.rewards, result.state.cash)
    np.testing.assert_array_equal(result.state.hands, hands)
    assert np.all(result.state.done) and not np.any(result.state.truncated)
    assert np.all(result.info.reason == Reason.ROUND_OVER)
    assert not np.any(result.info.accepted)
    repeated = execute(result.state)
    for left, right in zip(result.state, repeated.state, strict=True):
        if jax.dtypes.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(left, right)
    assert not np.any(repeated.rewards)


@pytest.mark.parametrize("players", [4, 5])
def test_seeded_deals_cover_all_layouts_and_conserve_cards(players):
    config = Config(num_players=players)
    states = jax.jit(jax.vmap(partial(reset, config=config)))(
        jax.random.split(jax.random.key(20261103), 4096)
    )
    decks = np.asarray(states.deck)
    assert set(map(tuple, decks)) == set(permutations((8, 10, 10, 12)))
    np.testing.assert_array_equal(np.asarray(states.hands).sum(1), decks)
    np.testing.assert_array_equal(np.asarray(states.hands).sum(2), 40 // players)
    np.testing.assert_array_equal(states.goal, np.array([1, 0, 3, 2])[decks.argmax(1)])
    np.testing.assert_array_equal(states.cash, 350 - 200 / players)


def adversarial_action(rng, state, index, players):
    """Build inputs from the Python oracle only, including fills and invalid attempts."""
    actor = int(rng.integers(-1, players + 1))
    kind = int(rng.integers(-1, 8))
    suit = int(rng.integers(-1, 5))
    price = int(rng.choice([-1, 0, 1, 9, 10, 11, 30, 31]))
    side = int(kind in (Kind.ASK, Kind.BUY, Kind.CANCEL_ASK))
    quote_id = state["quote_ids"][side][min(max(suit, 0), 3)]
    if index % 16 == 0:
        actor, kind, suit, price, quote_id = 0, Kind.ASK, int(np.argmax(state["hands"][0])), 1, 0
    elif index % 16 == 1:
        actor, kind = players - 1, Kind.BUY
        suit = int(np.argmax(state["quote_ids"][1]))
        quote_id = state["quote_ids"][1][suit]
    elif index % 5 == 0:
        quote_id += 1  # A stale identifier must not execute against a replacement.
    return actor, (kind, suit, price, quote_id)


@pytest.mark.parametrize("players", [4, 5])
@pytest.mark.parametrize("credit", [False, True])
@pytest.mark.parametrize("capped", [False, True])
def test_adversarial_traces_match_oracle_and_conserve_economics(players, credit, capped):
    config = Config(
        num_players=players,
        allow_negative_cash=credit,
        max_price=30,
        duration_ms=1000,
        max_events=37 if capped else 500,
        history_size=3,
        jitter_ms=17,
        latency_ms=tuple(seat * 11 for seat in range(players)),
    )
    cash = jnp.array([-100.25, 0, 25.5, 400.75, 1000][:players], jnp.float32)
    initial = jax.jit(jax.vmap(lambda key: reset(key, config, cash)))(
        jax.random.split(jax.random.key(20261103), 16)
    )
    rng = np.random.default_rng(20261103)
    host = {
        name: np.asarray(value)
        for name, value in initial._asdict().items()
        if name not in ("key", "ready_at")
    }
    inputs, expected_states, expected_infos, expected_rewards = [], [], [], []
    for table in range(16):
        oracle = {name: value[table].tolist() for name, value in host.items()}
        table_inputs, table_states, table_infos, table_rewards = [], [], [], []
        for index in range(256):
            actor, act = adversarial_action(rng, oracle, index, players)
            at_ms = index * 4 if index % 23 else oracle["time_ms"] - 1
            updated, reason, accepted, traded = transition(oracle, actor, act, at_ms, config)
            if traded:
                side = int(act[0] == Kind.BUY)
                owner = oracle["owners"][side][act[1]]
                buyer, seller = (actor, owner) if side else (owner, actor)
                info = [
                    accepted,
                    traded,
                    buyer,
                    seller,
                    act[1],
                    oracle["prices"][side][act[1]],
                    reason,
                ]
            else:
                info = [accepted, traded, -1, -1, -1, -1, reason]
            terminal = not oracle["done"] and updated["done"] and not updated["truncated"]
            reward = (
                np.array(updated["cash"]) - np.array(updated["initial_cash"])
                if terminal
                else np.zeros(players)
            )
            table_inputs.append((actor, *act, at_ms))
            table_states.append(updated)
            table_infos.append(info)
            table_rewards.append(reward)
            oracle = updated
        inputs.append(table_inputs)
        expected_states.append(table_states)
        expected_infos.append(table_infos)
        expected_rewards.append(table_rewards)

    def run(state, events):
        def event(state, row):
            result = apply_event(state, row[0], Action(*row[1:5]), row[5], config)
            return result.state, result

        return jax.lax.scan(event, state, events)[1]

    result = jax.jit(jax.vmap(run))(initial, jnp.array(inputs, jnp.int32))
    for field in host:
        expected = np.array([[s[field] for s in states] for states in expected_states])
        actual = np.asarray(getattr(result.state, field))
        if field in ("cash", "initial_cash"):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
        else:
            np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.stack(result.info, axis=-1), expected_infos)
    np.testing.assert_allclose(result.rewards, expected_rewards, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(np.asarray(result.state.hands).sum(2), result.state.deck)
    assert np.all(np.asarray(result.state.hands) >= 0)
    settled = np.asarray(result.state.done & ~result.state.truncated)
    np.testing.assert_allclose(
        np.asarray(result.state.cash).sum(2) + np.where(settled, 0, 200),
        np.asarray(cash).sum(),
        rtol=0,
        atol=3e-4,
    )
    assert np.all(np.asarray(result.state.done)[:, -1])
    assert np.all(np.asarray(result.state.truncated)[:, -1] == capped)
    assert np.asarray(result.info.traded).sum() >= 16


@pytest.mark.parametrize("players", [4, 5])
def test_all_seats_hide_privileged_information(players):
    config = Config(num_players=players)
    state = reset(jax.random.key(73), config)
    for player in range(players):
        changed = state._replace(
            hands=state.hands + (jnp.arange(players) != player)[:, None] * 99,
            deck=state.deck + 100,
            goal=(state.goal + 1) % 4,
            ready_at=state.ready_at + 1234,
            key=jax.random.key(999),
        )
        original = observe(state, jnp.int32(player), config)
        hidden_changed = observe(changed, jnp.int32(player), config)
        for a, b in zip(original, hidden_changed, strict=True):
            np.testing.assert_array_equal(a, b)
        for transform in (features, relative_features, action_masks):
            for a, b in zip(
                jax.tree.leaves(transform(original, config)),
                jax.tree.leaves(transform(hidden_changed, config)),
                strict=True,
            ):
                np.testing.assert_array_equal(a, b)
