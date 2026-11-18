"""Optional Gymnasium adapter for one learner and scripted opponents."""

from __future__ import annotations

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from figgie_jax.bots import heuristic_policy, random_policy
from figgie_jax.env import FiggieEnv, next_actor, reset
from figgie_jax.types import Config
from figgie_jax.types import action as make_action
from figgie_jax_adapters.common import (
    action_count,
    decision_observation,
    decode_action,
    legal_action_mask,
    observation_array,
)


class FiggieGymEnv(gym.Env):
    """Play one seat against scripted agents through the Gymnasium API.

    Each call to :meth:`step` applies the learner's action, then advances every
    earlier opponent clock until the learner is next to act or the round ends.
    Observations contain only the learner's private features and legal action
    mask. A true deadline is termination; an engine event cap is truncation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config = Config(),
        learner_seat: int = 0,
        opponent: str = "heuristic",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= learner_seat < config.num_players:
            raise ValueError(f"learner_seat must be in [0, {config.num_players})")
        if opponent not in ("heuristic", "random"):
            raise ValueError("opponent must be 'heuristic' or 'random'")
        if render_mode is not None:
            raise ValueError("FiggieGymEnv does not provide a render mode")

        self.config = config
        self.learner_seat = learner_seat
        self.opponent = opponent
        self.render_mode = render_mode
        self._engine = FiggieEnv(config)
        self._apply_event = jax.jit(self._engine.apply_event)
        self._state = None
        self._completed_unreported = False
        self._pending_initial_reward = 0.0
        self._episode_reward = 0.0
        self.action_space = spaces.Discrete(action_count(config))
        feature_limit = np.finfo(np.float32).max
        observation_size = (
            4
            + 2 * config.num_players
            + 8
            + 8 * (config.num_players + 1)
            + 6 * config.history_size
            + 1
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-feature_limit,
                    high=feature_limit,
                    shape=(observation_size,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(action_count(config)),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Start a seeded round and advance bots to the learner's first turn."""
        del options
        super().reset(seed=seed)
        self._episode_reward = 0.0
        self._completed_unreported = False
        self._pending_initial_reward = 0.0
        self._state = reset(self._random_key(), self.config)

        reward = self._advance_to_learner()
        if bool(np.asarray(self._state.done)):
            # A very short round or extreme per-seat latency can settle before
            # the learner's first opportunity. Deliver that terminal reward on
            # the next Gym step instead of discarding it during reset.
            self._pending_initial_reward = reward
            self._completed_unreported = True
        return self._observation(), self._info()

    def step(self, action_id: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Apply one learner action and run opponents up to the next learner turn."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._completed_unreported:
            self._completed_unreported = False
            reward = self._pending_initial_reward
            self._episode_reward += reward
            terminated, truncated = self._done_flags()
            info = self._info()
            info["round_ended_before_learner_turn"] = True
            info["episode_reward"] = self._episode_reward
            return self._observation(), reward, terminated, truncated, info
        if bool(np.asarray(self._state.done)):
            raise RuntimeError("The round is complete. Call reset() to start another round.")

        obs = decision_observation(self._state, self.learner_seat, self.config)
        engine_action = decode_action(action_id, obs, self.config)
        transition = self._apply_event(
            self._state,
            jnp.asarray(self.learner_seat, jnp.int32),
            engine_action,
            jnp.maximum(self._state.time_ms, self._state.ready_at[self.learner_seat]),
        )
        self._state = transition.state
        reward = float(np.asarray(transition.rewards)[self.learner_seat])
        reward += self._advance_to_learner()
        self._episode_reward += reward
        terminated, truncated = self._done_flags()
        info = self._info()
        if terminated or truncated:
            info["episode_reward"] = self._episode_reward
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> None:
        """Return no frame because this adapter has no rendering mode."""
        return None

    def _random_key(self) -> jax.Array:
        """Draw one deterministic JAX key from Gymnasium's seeded generator."""
        seed = int(self.np_random.integers(0, 2**31 - 1))
        return jax.random.key(seed)

    def _opponent_action(self, player: int):
        obs = decision_observation(self._state, player, self.config)
        policy = heuristic_policy if self.opponent == "heuristic" else random_policy
        return policy(self._random_key(), obs, self.config)

    def _advance_to_learner(self) -> float:
        """Run scripted agents until the learner is next or the round ends."""
        learner_reward = 0.0
        while not bool(np.asarray(self._state.done)):
            if int(np.asarray(self._state.ready_at).min()) >= self.config.duration_ms:
                learner_reward += self._settle_at_deadline()
                break
            actor = int(np.asarray(next_actor(self._state)))
            if actor == self.learner_seat:
                break
            action = self._opponent_action(actor)
            transition = self._apply_event(
                self._state,
                jnp.asarray(actor, jnp.int32),
                action,
                jnp.maximum(self._state.time_ms, self._state.ready_at[actor]),
            )
            learner_reward += float(np.asarray(transition.rewards)[self.learner_seat])
            self._state = transition.state
        return learner_reward

    def _settle_at_deadline(self) -> float:
        """Settle before requesting an action whose scheduled time is expired."""
        transition = self._apply_event(
            self._state,
            jnp.int32(0),
            make_action(),
            jnp.int32(self.config.duration_ms),
        )
        self._state = transition.state
        return float(np.asarray(transition.rewards)[self.learner_seat])

    def _observation(self) -> dict[str, np.ndarray]:
        """Build a fresh private observation and discrete legal-action mask."""
        if self._state is None:
            raise RuntimeError("Call reset() before requesting an observation.")
        obs = decision_observation(self._state, self.learner_seat, self.config)
        action_mask = legal_action_mask(obs, self.config)
        if bool(np.asarray(self._state.done)):
            # A terminal handoff still needs one safe action for policies that
            # sample from masks before they receive Gymnasium's done flags.
            action_mask[:] = False
            action_mask[0] = True
        return {
            "observation": observation_array(obs, self.config).copy(),
            "action_mask": action_mask.astype(np.int8),
        }

    def _done_flags(self) -> tuple[bool, bool]:
        """Return Gymnasium termination and truncation flags for engine state."""
        done = bool(np.asarray(self._state.done))
        truncated = bool(np.asarray(self._state.truncated))
        return done and not truncated, done and truncated

    def _info(self) -> dict:
        """Return public timing metadata without exposing hidden game state."""
        if self._state is None:
            return {}
        return {
            "learner_seat": self.learner_seat,
            "simulation_time_ms": int(np.asarray(self._state.time_ms)),
            "event_count": int(np.asarray(self._state.events)),
        }
