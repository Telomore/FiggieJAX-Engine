"""Conformance and privacy checks for optional RL framework adapters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from figgie_jax.env import next_actor, reset
from figgie_jax.observations import observe
from figgie_jax.types import Config, Kind
from figgie_jax_adapters.common import (
    action_count,
    decision_observation,
    decode_action,
    legal_action_mask,
    observation_array,
)


def _short_config(num_players: int = 4, **kwargs) -> Config:
    """Build a short deterministic game configuration for adapter checks."""
    return Config(
        num_players=num_players,
        duration_ms=1_200,
        decision_interval_ms=100,
        history_size=4,
        max_events=256,
        **kwargs,
    )


@pytest.mark.parametrize("num_players", [4, 5])
def test_finite_action_mapping_uses_visible_quote_ids(num_players: int) -> None:
    """Discrete actions have stable IDs and accept only displayed quote IDs."""
    config = _short_config(num_players)
    state = reset(jax.random.key(4), config)
    state = state._replace(
        owners=state.owners.at[1, 2].set(1),
        prices=state.prices.at[1, 2].set(17),
        quote_ids=state.quote_ids.at[1, 2].set(91),
    )
    obs = observe(state, jnp.int32(0), config)

    expected_count = 1 + 8 * config.max_price + 16
    assert action_count(config) == expected_count
    mask = legal_action_mask(obs, config)
    assert mask.dtype == np.bool_
    assert mask.shape == (expected_count,)
    assert mask[0]

    buy_id = 1 + 8 * config.max_price + 2
    buy = decode_action(buy_id, obs, config)
    assert int(buy.kind) == Kind.BUY
    assert int(buy.suit) == 2
    assert int(buy.quote_id) == 91
    assert mask[buy_id]

    wait = decode_action(0, obs, config)
    assert int(wait.kind) == Kind.WAIT
    with pytest.raises(ValueError):
        decode_action(expected_count, obs, config)


def test_decision_observation_advances_only_the_current_actor() -> None:
    """Only the seat with the earliest clock sees its upcoming decision time."""
    config = _short_config()
    state = reset(jax.random.key(7), config)
    state = state._replace(
        time_ms=jnp.int32(100),
        ready_at=jnp.asarray([500, 600, 700, 800], dtype=jnp.int32),
    )

    actor_obs = decision_observation(state, 0, config)
    other_obs = decision_observation(state, 1, config)
    assert int(actor_obs.remaining_ms) == config.duration_ms - 500
    assert int(other_obs.remaining_ms) == config.duration_ms - 100


def test_observations_do_not_change_with_other_hidden_cards_or_goal() -> None:
    """Private features exclude other hands and the hidden goal."""
    config = _short_config()
    state = reset(jax.random.key(9), config)
    first = observe(state, jnp.int32(0), config)
    changed_state = state._replace(
        hands=state.hands.at[1].add(jnp.asarray([1, 0, 0, 0], dtype=jnp.int32)),
        goal=(state.goal + 1) % 4,
    )
    second = observe(changed_state, jnp.int32(0), config)
    np.testing.assert_array_equal(
        observation_array(first, config), observation_array(second, config)
    )


@pytest.mark.parametrize("num_players", [4, 5])
def test_gymnasium_checker_and_seeded_episode(num_players: int) -> None:
    """Gymnasium API, timing, and learner reward remain reproducible."""
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from figgie_jax_adapters.gymnasium import FiggieGymEnv

    config = _short_config(num_players)
    env = FiggieGymEnv(config=config, learner_seat=0, opponent="random")
    check_env(env, skip_render_check=True)

    def play_seeded_round() -> tuple[list[tuple[float, int, int]], float]:
        observation, info = env.reset(seed=1234)
        assert observation["action_mask"][0] == 1
        trace = []
        previous_time = info["simulation_time_ms"]
        previous_events = info["event_count"]
        total_reward = 0.0
        for _ in range(64):
            observation, reward, terminated, truncated, info = env.step(0)
            assert info["simulation_time_ms"] >= previous_time
            if not (terminated or truncated):
                assert info["event_count"] > previous_events
            else:
                # Deadline settlement does not consume an additional event.
                assert info["event_count"] >= previous_events
            previous_time = info["simulation_time_ms"]
            previous_events = info["event_count"]
            total_reward += reward
            trace.append((reward, info["simulation_time_ms"], info["event_count"]))
            if terminated or truncated:
                break
        assert terminated or truncated
        assert not truncated
        assert info["episode_reward"] == pytest.approx(total_reward)
        final_profit = float(
            np.asarray(env._state.cash)[0] - np.asarray(env._state.initial_cash)[0]
        )
        assert total_reward == pytest.approx(final_profit, abs=1e-5)
        return trace, total_reward

    first = play_seeded_round()
    second = play_seeded_round()
    assert first == second


def test_gymnasium_event_cap_is_truncation() -> None:
    """Reaching the event safety cap is not reported as game termination."""
    pytest.importorskip("gymnasium")

    from figgie_jax_adapters.gymnasium import FiggieGymEnv

    config = Config(
        num_players=4,
        duration_ms=10_000,
        decision_interval_ms=100,
        max_events=1,
        history_size=4,
    )
    env = FiggieGymEnv(config=config, opponent="heuristic")
    env.reset(seed=71)
    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == 0.0
    assert not terminated
    assert truncated


def test_gymnasium_delivers_reward_if_round_ends_before_first_turn() -> None:
    """Terminal reward during reset-time opponent turns is returned on step."""
    pytest.importorskip("gymnasium")

    from figgie_jax_adapters.gymnasium import FiggieGymEnv

    config = Config(
        num_players=4,
        duration_ms=500,
        decision_interval_ms=100,
        latency_ms=(5_000, 0, 0, 0),
        max_events=100,
        history_size=4,
    )
    env = FiggieGymEnv(config=config, learner_seat=0, opponent="random")
    env.reset(seed=17)
    _, reward, terminated, truncated, info = env.step(0)

    expected_reward = float(np.asarray(env._state.cash)[0] - np.asarray(env._state.initial_cash)[0])
    assert info["round_ended_before_learner_turn"]
    assert terminated
    assert not truncated
    assert reward == pytest.approx(expected_reward, abs=1e-5)


@pytest.mark.parametrize("num_players", [4, 5])
def test_gymnasium_settles_at_deadline_without_a_phantom_turn(num_players: int) -> None:
    """Exact-deadline clocks settle before either the learner or a bot can act."""
    pytest.importorskip("gymnasium")

    from figgie_jax_adapters.gymnasium import FiggieGymEnv

    config = Config(
        num_players=num_players,
        duration_ms=1_000,
        decision_interval_ms=250,
        max_events=256,
        history_size=4,
    )
    env = FiggieGymEnv(config=config, learner_seat=0, opponent="random")
    observation, _ = env.reset(seed=92)
    learner_turns = 0

    for _ in range(4):
        observation, _, terminated, truncated, info = env.step(0)
        learner_turns += 1
        if terminated or truncated:
            break

    assert terminated
    assert not truncated
    assert learner_turns == 3
    assert info["simulation_time_ms"] == config.duration_ms
    assert info["event_count"] == num_players * 3
    assert int(np.count_nonzero(observation["action_mask"])) == 1
    assert observation["action_mask"][0] == 1


@pytest.mark.parametrize("num_players", [4, 5])
@pytest.mark.parametrize("learner_seat", [0, 1])
def test_gymnasium_deadline_settles_before_learner_or_opponent_action(
    num_players: int, learner_seat: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A due learner or opponent clock cannot trigger a policy action at deadline."""
    pytest.importorskip("gymnasium")

    from figgie_jax_adapters.gymnasium import FiggieGymEnv

    config = Config(
        num_players=num_players,
        duration_ms=1_000,
        decision_interval_ms=250,
        max_events=256,
        history_size=4,
    )
    env = FiggieGymEnv(config=config, learner_seat=learner_seat, opponent="random")
    env.reset(seed=29)
    prior_events = int(np.asarray(env._state.events))
    env._state = env._state._replace(
        time_ms=jnp.int32(750),
        ready_at=jnp.asarray(
            [config.duration_ms + seat for seat in range(num_players)], dtype=jnp.int32
        ),
    )
    monkeypatch.setattr(
        env,
        "_opponent_action",
        lambda player: pytest.fail(f"opponent {player} acted at the deadline"),
    )

    reward = env._advance_to_learner()

    assert bool(np.asarray(env._state.done))
    assert int(np.asarray(env._state.events)) == prior_events
    assert int(np.asarray(env._state.time_ms)) == config.duration_ms
    expected_reward = float(
        np.asarray(env._state.cash)[learner_seat]
        - np.asarray(env._state.initial_cash)[learner_seat]
    )
    assert reward == pytest.approx(expected_reward, abs=1e-5)


@pytest.mark.parametrize("num_players", [4, 5])
def test_gymnasium_pre_first_turn_settlement_uses_wait_only_handoff(num_players: int) -> None:
    """A round shorter than the first clock settles on reset with a safe handoff."""
    pytest.importorskip("gymnasium")

    from figgie_jax_adapters.gymnasium import FiggieGymEnv

    config = Config(
        num_players=num_players,
        duration_ms=100,
        decision_interval_ms=250,
        max_events=32,
        history_size=4,
    )
    env = FiggieGymEnv(config=config, learner_seat=0, opponent="random")
    observation, info = env.reset(seed=73)

    assert info["simulation_time_ms"] == config.duration_ms
    assert info["event_count"] == 0
    assert bool(np.asarray(env._state.done))
    expected_mask = np.zeros(action_count(config), dtype=np.int8)
    expected_mask[0] = 1
    np.testing.assert_array_equal(observation["action_mask"], expected_mask)

    _, reward, terminated, truncated, info = env.step(0)
    expected_reward = float(np.asarray(env._state.cash)[0] - np.asarray(env._state.initial_cash)[0])
    assert terminated
    assert not truncated
    assert info["round_ended_before_learner_turn"]
    assert info["event_count"] == 0
    assert reward == pytest.approx(expected_reward, abs=1e-5)


@pytest.mark.parametrize("num_players", [4, 5])
def test_pettingzoo_api_and_terminal_rewards(num_players: int) -> None:
    """AEC conformance preserves asynchronous actors and sparse rewards."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import api_test

    from figgie_jax_adapters.pettingzoo import FiggieAECEnv

    config = _short_config(num_players)
    env = FiggieAECEnv(config=config)
    api_test(env, num_cycles=100, verbose_progress=False)

    env.reset(seed=2026)
    assert env.agent_selection == f"player_{int(np.asarray(next_actor(env._state)))}"

    collected = {agent: 0.0 for agent in env.possible_agents}
    for _ in range(1_000):
        if not env.agents:
            break
        agent = env.agent_selection
        _, reward, terminated, truncated, _ = env.last()
        collected[agent] += reward
        env.step(None if terminated or truncated else 0)
    assert not env.agents
    expected_profits = np.asarray(env._state.cash - env._state.initial_cash)
    np.testing.assert_allclose(
        [collected[agent] for agent in env.possible_agents],
        expected_profits,
        atol=1e-5,
    )


def test_pettingzoo_keeps_independent_seat_clocks() -> None:
    """AEC can select one seat repeatedly when its next clock is earliest."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")

    from figgie_jax_adapters.pettingzoo import FiggieAECEnv

    config = Config(
        num_players=4,
        duration_ms=10_000,
        decision_interval_ms=100,
        latency_ms=(0, 1_000, 2_000, 3_000),
        max_events=256,
        history_size=4,
    )
    env = FiggieAECEnv(config=config)
    env.reset(seed=15)
    assert env.agent_selection == "player_0"
    env.step(0)
    assert env.agent_selection == "player_0"
    np.testing.assert_array_equal(np.asarray(env._state.ready_at), [200, 1100, 2100, 3100])


@pytest.mark.parametrize("num_players", [4, 5])
def test_pettingzoo_settles_at_deadline_before_selecting_another_actor(
    num_players: int,
) -> None:
    """A round settles after its last pre-deadline event, then pays each seat once."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")

    from figgie_jax_adapters.pettingzoo import FiggieAECEnv

    config = Config(
        num_players=num_players,
        duration_ms=1_000,
        decision_interval_ms=250,
        max_events=256,
        history_size=4,
    )
    env = FiggieAECEnv(config=config)
    env.reset(seed=112)

    while not bool(np.asarray(env._state.done)):
        _, _, terminated, truncated, _ = env.last()
        assert not terminated and not truncated
        env.step(0)

    assert not bool(np.asarray(env._state.truncated))
    assert int(np.asarray(env._state.time_ms)) == config.duration_ms
    assert int(np.asarray(env._state.events)) == num_players * 3

    collected = {}
    for _ in range(num_players):
        agent = env.agent_selection
        _, reward, terminated, truncated, _ = env.last()
        assert terminated and not truncated
        collected[agent] = reward
        env.step(None)
    assert not env.agents
    expected = np.asarray(env._state.cash - env._state.initial_cash)
    np.testing.assert_allclose(
        [collected[agent] for agent in env.possible_agents], expected, atol=1e-5
    )


@pytest.mark.parametrize("num_players", [4, 5])
def test_pettingzoo_settles_before_first_opportunity_and_pays_dead_agents(
    num_players: int,
) -> None:
    """A deadline before the first clock is reported through normal AEC removal."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")

    from figgie_jax_adapters.pettingzoo import FiggieAECEnv

    config = Config(
        num_players=num_players,
        duration_ms=100,
        decision_interval_ms=250,
        max_events=32,
        history_size=4,
    )
    env = FiggieAECEnv(config=config)
    env.reset(seed=113)

    assert bool(np.asarray(env._state.done))
    assert not bool(np.asarray(env._state.truncated))
    assert int(np.asarray(env._state.time_ms)) == config.duration_ms
    assert int(np.asarray(env._state.events)) == 0
    collected = {}
    for _ in range(num_players):
        agent = env.agent_selection
        _, reward, terminated, truncated, _ = env.last()
        assert terminated and not truncated
        collected[agent] = reward
        env.step(None)
    assert not env.agents
    expected = np.asarray(env._state.cash - env._state.initial_cash)
    np.testing.assert_allclose(
        [collected[agent] for agent in env.possible_agents], expected, atol=1e-5
    )


@pytest.mark.parametrize("framework", ["gymnasium", "pettingzoo"])
def test_seed_controls_subsequent_unseeded_resets(framework: str) -> None:
    """A single initial seed must reproduce a sequence of fresh rounds."""
    pytest.importorskip(framework)
    if framework == "gymnasium":
        from figgie_jax_adapters.gymnasium import FiggieGymEnv

        first, second = FiggieGymEnv(_short_config()), FiggieGymEnv(_short_config())
    else:
        from figgie_jax_adapters.pettingzoo import FiggieAECEnv

        first, second = FiggieAECEnv(_short_config()), FiggieAECEnv(_short_config())
    first.reset(seed=33)
    second.reset(seed=33)
    for _ in range(3):
        first.reset()
        second.reset()
        np.testing.assert_array_equal(first._state.hands, second._state.hands)
        np.testing.assert_array_equal(first._state.ready_at, second._state.ready_at)
