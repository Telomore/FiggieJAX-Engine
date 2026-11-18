"""Tests for the optional OpenSpiel Figgie game integration."""

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

import jax.numpy as jnp  # noqa: E402

import figgie_jax_adapters.openspiel  # noqa: E402, F401, registers python_figgie.


def _game(players: int = 4, duration_ms: int = 750, max_events: int = 100):
    """Load a small deterministic-clock game for API and simulation tests."""
    return pyspiel.load_game(
        "python_figgie("
        f"players={players},duration_ms={duration_ms},max_price=5,"
        f"decision_interval_ms=250,max_events={max_events},history_size=3)"
    )


def _sample_chance(state, rng: np.random.Generator) -> None:
    """Sample one explicit chance outcome from the state's distribution."""
    outcomes = state.chance_outcomes()
    actions, probabilities = zip(*outcomes, strict=True)
    assert np.isclose(sum(probabilities), 1.0)
    state.apply_action(int(rng.choice(actions, p=probabilities)))


def _first_decision(state, rng: np.random.Generator):
    """Resolve the explicit deal and any initial scheduler tie."""
    while state.is_chance_node():
        _sample_chance(state, rng)
    assert not state.is_terminal()
    return state


def test_registered_game_has_parameterized_player_count_and_action_space():
    game = _game(players=5)

    assert game.num_players() == 5
    assert game.get_type().short_name == "python_figgie"
    assert game.get_type().chance_mode == pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC
    assert game.get_type().provides_observation_tensor
    assert not game.get_type().provides_information_state_string
    assert not game.get_type().provides_information_state_tensor
    assert game.num_distinct_actions() == 1 + 8 * 5 + 16


def test_initial_layout_and_deal_are_explicit_chance_nodes():
    game = _game(players=4)
    state = game.new_initial_state()
    outcomes = state.chance_outcomes()

    assert state.is_chance_node()
    assert len(outcomes) == 12
    assert all(np.isclose(probability, 1 / 12) for _, probability in outcomes)

    rng = np.random.default_rng(12)
    state.apply_action(outcomes[0][0])
    assert len(state.chance_outcomes()) == 4
    _first_decision(state, rng)

    dealt = np.asarray(state._sim_state.hands)
    deck = np.asarray(state._sim_state.deck)
    assert dealt.shape == (4, 4)
    assert np.all(dealt.sum(axis=1) == 10)
    assert np.array_equal(dealt.sum(axis=0), deck)
    assert int(state._sim_state.goal) == int(np.argmax(deck)) ^ 1


@pytest.mark.parametrize("players", [4, 5])
def test_random_legal_play_terminates_with_zero_sum_chip_returns(players: int):
    game = _game(players=players)
    state = game.new_initial_state()
    rng = np.random.default_rng(23 + players)

    for _ in range(100):
        if state.is_terminal():
            break
        if state.is_chance_node():
            _sample_chance(state, rng)
            continue
        legal_actions = state.legal_actions()
        assert legal_actions
        state.apply_action(int(rng.choice(legal_actions)))
    else:
        pytest.fail("The short Figgie round did not terminate within the step bound.")

    returns = np.asarray(state.returns(), dtype=np.float64)
    actual_profit = np.asarray(
        state._sim_state.cash - state._sim_state.initial_cash, dtype=np.float64
    )
    assert np.allclose(returns, actual_profit, atol=1e-5)
    assert np.isclose(returns.sum(), 0.0, atol=1e-5)


def test_scheduler_ties_are_uniform_explicit_chance_outcomes():
    state = _first_decision(_game().new_initial_state(), np.random.default_rng(31))
    actor = state.current_player()

    state.apply_action(0)  # WAIT at the actor's 250 ms opportunity.

    assert state.is_chance_node()
    outcomes = state.chance_outcomes()
    players = [action for action, _ in outcomes]
    probabilities = [probability for _, probability in outcomes]
    assert actor not in players
    assert len(players) == 3
    assert np.allclose(probabilities, np.full(3, 1 / 3))


def test_clone_has_independent_economic_state():
    state = _first_decision(_game().new_initial_state(), np.random.default_rng(43))
    clone = state.clone()
    original_events = int(state._sim_state.events)

    clone.apply_action(0)

    assert int(state._sim_state.events) == original_events
    assert int(clone._sim_state.events) == original_events + 1
    assert not state.is_terminal()


def test_observer_omits_opponent_hands_and_goal():
    game = _game()
    state = _first_decision(game.new_initial_state(), np.random.default_rng(59))
    actor = state.current_player()
    observer = game.make_py_observer()
    observer.set_from(state, actor)
    before = observer.tensor.copy()

    hidden_variant = state.clone()
    hands = np.asarray(hidden_variant._sim_state.hands).copy()
    other = (actor + 1) % game.num_players()
    hands[other, 0] += 1
    hidden_variant._sim_state = hidden_variant._sim_state._replace(
        hands=jnp.asarray(hands), goal=(hidden_variant._sim_state.goal + 1) % 4
    )
    observer.set_from(hidden_variant, actor)

    assert np.array_equal(observer.tensor, before)
    assert len(state.observation_tensor(actor)) == game.observation_tensor_size()


def test_hidden_counterparty_inventory_does_not_change_legal_mask():
    state = _first_decision(_game(max_events=10).new_initial_state(), np.random.default_rng(67))
    actor = state.current_player()
    owner = (actor + 1) % 4
    sim = state._sim_state
    hands = np.asarray(sim.hands).copy()
    hands[owner, 0] = 0
    prices = np.asarray(sim.prices).copy()
    owners = np.asarray(sim.owners).copy()
    quote_ids = np.asarray(sim.quote_ids).copy()
    prices[1, 0] = 1
    owners[1, 0] = owner
    quote_ids[1, 0] = int(sim.events) + 1
    state._sim_state = sim._replace(
        hands=jnp.asarray(hands),
        prices=jnp.asarray(prices),
        owners=jnp.asarray(owners),
        quote_ids=jnp.asarray(quote_ids),
    )
    buy_id = 1 + 8 * 5

    assert buy_id in state.legal_actions()
    state.apply_action(buy_id)

    assert int(state._sim_state.events) == 1
    assert int(state._sim_state.trade_count) == 0
    assert int(state._sim_state.owners[1, 0]) == owner


def test_event_cap_that_could_precede_deadline_is_rejected():
    with pytest.raises(ValueError, match="max_events must exceed"):
        _game(duration_ms=750, max_events=8)


def test_perfect_recall_observer_request_is_rejected():
    game = _game()

    with pytest.raises(ValueError, match="perfect-recall"):
        game.make_py_observer(pyspiel.IIGObservationType(perfect_recall=True))
