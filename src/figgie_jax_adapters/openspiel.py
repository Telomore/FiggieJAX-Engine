"""OpenSpiel game wrapper for the FiggieJAX engine.

Importing this module registers ``python_figgie`` with PySpiel. OpenSpiel is an
optional dependency. This adapter models the initial deal and equal-priority
scheduler ties as explicit chance nodes. It does not expose perfect-recall
information states.
"""

from __future__ import annotations

import itertools
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyspiel

from figgie_jax.env import apply_event, reset
from figgie_jax.observations import observe
from figgie_jax.types import Config, State
from figgie_jax.types import action as make_action
from figgie_jax_adapters.common import (
    action_count,
    decode_action,
    legal_action_mask,
    observation_array,
)

_DECK_LAYOUTS = tuple(sorted(set(itertools.permutations((8, 10, 10, 12)))))
_SUIT_NAMES = ("spades", "clubs", "hearts", "diamonds")
_DEFAULT_PARAMETERS = {
    "players": 4,
    "duration_ms": 240_000,
    "max_price": 200,
    "decision_interval_ms": 250,
    "initial_cash": 350.0,
    "max_events": 50_000,
    "history_size": 16,
    "allow_negative_cash": True,
}
_GAME_TYPE = pyspiel.GameType(
    short_name="python_figgie",
    long_name="FiggieJAX (Python)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=5,
    min_num_players=4,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMETERS,
)


@cache
def _compiled_reset(config: Config):
    """Return the reset function compiled for one immutable game configuration."""
    return jax.jit(lambda: reset(jax.random.key(0), config))


@cache
def _compiled_apply_event(config: Config):
    """Return the event transition compiled for one immutable configuration."""
    return jax.jit(lambda state, actor, act, at_ms: apply_event(state, actor, act, at_ms, config))


def _feature_size(config: Config) -> int:
    """Return the width of the shared seat-relative observation vector."""
    players = config.num_players
    history = config.history_size
    return 4 + players + players + 8 + 8 * (players + 1) + 6 * history + 1


class FiggieGame(pyspiel.Game):
    """Sequential imperfect-information game for four- or five-seat Figgie."""

    def __init__(self, params: dict[str, Any] | None = None):
        parameters = dict(params or {})
        players = int(parameters.get("players", _DEFAULT_PARAMETERS["players"]))
        self.config = Config(
            num_players=players,
            duration_ms=int(parameters.get("duration_ms", _DEFAULT_PARAMETERS["duration_ms"])),
            max_price=int(parameters.get("max_price", _DEFAULT_PARAMETERS["max_price"])),
            decision_interval_ms=int(
                parameters.get("decision_interval_ms", _DEFAULT_PARAMETERS["decision_interval_ms"])
            ),
            initial_cash=float(parameters.get("initial_cash", _DEFAULT_PARAMETERS["initial_cash"])),
            max_events=int(parameters.get("max_events", _DEFAULT_PARAMETERS["max_events"])),
            history_size=int(parameters.get("history_size", _DEFAULT_PARAMETERS["history_size"])),
            allow_negative_cash=bool(
                parameters.get("allow_negative_cash", _DEFAULT_PARAMETERS["allow_negative_cash"])
            ),
            latency_ms=(0,) * players,
            jitter_ms=0,
        )
        max_decisions = players * (
            (self.config.duration_ms - 1) // self.config.decision_interval_ms
        )
        if self.config.max_events <= max_decisions:
            raise ValueError(
                "max_events must exceed the number of zero-latency opportunities before "
                f"the deadline ({max_decisions})."
            )

        # This bound allows every event to move at most max_price chips for one
        # player, plus the initial bankroll and the fixed settlement pot.
        utility_bound = self.config.max_events * self.config.max_price
        utility_bound += self.config.initial_cash + 200.0
        game_info = pyspiel.GameInfo(
            num_distinct_actions=max(action_count(self.config), len(_DECK_LAYOUTS)),
            max_chance_outcomes=len(_DECK_LAYOUTS),
            num_players=players,
            min_utility=-utility_bound,
            max_utility=utility_bound,
            utility_sum=0.0,
            max_game_length=2 * self.config.max_events + 41,
        )
        super().__init__(_GAME_TYPE, game_info, parameters)

    def new_initial_state(self) -> FiggieState:
        """Create the first explicit chance node for the deck layout."""
        return FiggieState(self)

    def make_py_observer(self, iig_obs_type=None, params=None) -> FiggieObserver:
        """Create an observer that exposes the current actor's private view only."""
        if params:
            raise ValueError(f"Observer parameters are not supported: {params}")
        obs_type = iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False)
        if obs_type.perfect_recall:
            raise ValueError("Figgie does not provide perfect-recall information states.")
        if not obs_type.public_info:
            raise ValueError("The Figgie observation includes public market information.")
        if obs_type.private_info != pyspiel.PrivateInfoType.SINGLE_PLAYER:
            raise ValueError("Only the acting player's private hand is available.")
        return FiggieObserver(self.config)


class FiggieState(pyspiel.State):
    """One explicit-deal trajectory and its asynchronous market events."""

    def __init__(self, game: FiggieGame):
        super().__init__(game)
        self._game_config = game.config
        self._stage = "layout"
        self._layout_index: int | None = None
        self._deck: list[int] | None = None
        self._hands: list[list[int]] | None = None
        self._cards_dealt = 0
        self._sim_state: State | None = None
        self._actor: int | None = None
        self._terminal = False

    def current_player(self) -> int:
        """Return the chance player, scheduled seat, or terminal marker."""
        if self._terminal:
            return pyspiel.PlayerId.TERMINAL
        if self._stage in ("layout", "deal", "schedule"):
            return pyspiel.PlayerId.CHANCE
        if self._actor is None:
            raise RuntimeError("A Figgie decision state must have a scheduled actor.")
        return self._actor

    def _legal_actions(self, player: int) -> list[int]:
        """List actions permitted by the acting player's observation."""
        if self._stage != "decision" or player != self._actor:
            return []
        observation = self._actor_observation(player)
        return np.flatnonzero(legal_action_mask(observation, self._game_config)).tolist()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Return exact probabilities for a layout, card, or scheduler tie."""
        if self._stage == "layout":
            probability = 1.0 / len(_DECK_LAYOUTS)
            return [(index, probability) for index in range(len(_DECK_LAYOUTS))]
        if self._stage == "deal":
            if self._deck is None or self._hands is None:
                raise RuntimeError("The deck layout must be sampled before dealing cards.")
            remaining = np.asarray(self._deck) - np.asarray(self._hands).sum(axis=0)
            total = int(remaining.sum())
            return [
                (suit, float(count) / total) for suit, count in enumerate(remaining) if count > 0
            ]
        if self._stage == "schedule":
            tied = self._earliest_players()
            probability = 1.0 / len(tied)
            return [(player, probability) for player in tied]
        raise RuntimeError("Chance outcomes are available only at a Figgie chance node.")

    def _apply_action(self, action: int) -> None:
        """Apply a chance outcome or process one observable legal action."""
        if self._terminal:
            raise ValueError("Cannot apply an action to a terminal Figgie state.")
        if self._stage == "layout":
            if not 0 <= action < len(_DECK_LAYOUTS):
                raise ValueError(f"Unknown deck layout action: {action}")
            self._layout_index = action
            self._deck = list(_DECK_LAYOUTS[action])
            self._hands = [[0, 0, 0, 0] for _ in range(self._game_config.num_players)]
            self._cards_dealt = 0
            self._stage = "deal"
            return
        if self._stage == "deal":
            self._apply_deal_action(action)
            return
        if self._stage == "schedule":
            tied = self._earliest_players()
            if action not in tied:
                raise ValueError(f"Seat {action} is not eligible in this scheduler tie.")
            self._actor = action
            self._stage = "decision"
            return
        if self._stage != "decision" or self._actor is None:
            raise RuntimeError("The Figgie state is not ready for a player action.")

        legal_actions = self._legal_actions(self._actor)
        if action not in legal_actions:
            raise ValueError(f"Action {action} is not legal from the actor's observation.")
        observation = self._actor_observation(self._actor)
        structured_action = decode_action(action, observation, self._game_config)
        event_time = max(
            int(np.asarray(self._sim_state.time_ms)),
            int(np.asarray(self._sim_state.ready_at[self._actor])),
        )
        transition = _compiled_apply_event(self._game_config)(
            self._sim_state,
            jnp.int32(self._actor),
            structured_action,
            jnp.int32(event_time),
        )
        next_state = transition.state
        if bool(np.asarray(next_state.truncated)):
            raise RuntimeError(
                "The Figgie event cap was reached despite the validated deadline bound."
            )
        self._sim_state = next_state
        self._terminal = bool(np.asarray(next_state.done))
        self._actor = None
        if not self._terminal:
            self._prepare_next_turn()

    def _apply_deal_action(self, suit: int) -> None:
        """Deal the next card from the sampled layout without replacement."""
        if self._deck is None or self._hands is None:
            raise RuntimeError("The deck layout must be sampled before dealing cards.")
        remaining = np.asarray(self._deck) - np.asarray(self._hands).sum(axis=0)
        if not 0 <= suit < 4 or remaining[suit] <= 0:
            raise ValueError(f"Suit {suit} is not available in the remaining deck.")
        seat = self._cards_dealt % self._game_config.num_players
        self._hands[seat][suit] += 1
        self._cards_dealt += 1
        if self._cards_dealt == 40:
            self._start_round()

    def _start_round(self) -> None:
        """Build the simulator state from the explicit chance deal."""
        if self._deck is None or self._hands is None:
            raise RuntimeError("The explicit chance deal is incomplete.")
        # This key is not used to sample a deal or scheduler decision. The deal
        # has already been sampled through OpenSpiel chance outcomes, and tie
        # arbitration is represented by explicit chance nodes below.
        state = _compiled_reset(self._game_config)()
        deck = jnp.asarray(self._deck, dtype=jnp.int32)
        hands = jnp.asarray(self._hands, dtype=jnp.int32)
        goal = jnp.argmax(deck).astype(jnp.int32) ^ 1
        self._sim_state = state._replace(deck=deck, hands=hands, goal=goal)
        self._stage = "schedule"
        self._prepare_next_turn()

    def _earliest_players(self) -> list[int]:
        """Return the seats tied at the smallest scheduled clock time."""
        if self._sim_state is None:
            return []
        ready = np.asarray(self._sim_state.ready_at)
        return np.flatnonzero(ready == ready.min()).tolist()

    def _prepare_next_turn(self) -> None:
        """Create an explicit scheduler chance node or settle at the deadline."""
        if self._sim_state is None:
            raise RuntimeError("The simulator state has not been initialized.")
        if bool(np.asarray(self._sim_state.done)):
            self._terminal = True
            return
        ready = np.asarray(self._sim_state.ready_at)
        if int(ready.min()) >= self._game_config.duration_ms:
            expired = _compiled_apply_event(self._game_config)(
                self._sim_state,
                jnp.int32(0),
                make_action(),
                jnp.int32(self._game_config.duration_ms),
            )
            self._sim_state = expired.state
            self._terminal = True
            self._stage = "terminal"
            self._actor = None
            return
        tied = np.flatnonzero(ready == ready.min()).tolist()
        if len(tied) == 1:
            self._stage = "decision"
            self._actor = tied[0]
        else:
            self._stage = "schedule"
            self._actor = None

    def _actor_observation(self, player: int):
        """Observe the selected actor at its own next clock time."""
        if self._sim_state is None:
            raise RuntimeError("Cards have not been dealt yet.")
        decision_ms = int(np.asarray(self._sim_state.time_ms))
        if player == self._actor:
            decision_ms = max(decision_ms, int(np.asarray(self._sim_state.ready_at[player])))
        decision_ms = min(decision_ms, self._game_config.duration_ms)
        decision_state = self._sim_state._replace(time_ms=jnp.int32(decision_ms))
        return observe(decision_state, jnp.int32(player), self._game_config)

    def is_terminal(self) -> bool:
        """Return whether the round settled at its deadline."""
        return self._terminal

    def returns(self) -> list[float]:
        """Return final chip profit, or zero while the round remains live."""
        if not self._terminal or self._sim_state is None:
            return [0.0] * self._game_config.num_players
        if bool(np.asarray(self._sim_state.truncated)):
            raise RuntimeError("A truncated state is not a completed Figgie result.")
        profit = np.asarray(self._sim_state.cash - self._sim_state.initial_cash)
        return [float(value) for value in profit]

    def _action_to_string(self, player: int, action: int) -> str:
        """Render chance outcomes and public player actions for logs."""
        if player == pyspiel.PlayerId.CHANCE:
            if self._stage == "layout":
                return f"Layout:{action}"
            if self._stage == "deal":
                return f"Deal:{_SUIT_NAMES[action]}"
            if self._stage == "schedule":
                return f"Schedule:seat-{action}"
            return f"Chance:{action}"
        if self._sim_state is None:
            return str(action)
        observation = self._actor_observation(player)
        structured = decode_action(action, observation, self._game_config)
        kind = int(np.asarray(structured.kind))
        if kind == 0:
            return "WAIT"
        suit = int(np.asarray(structured.suit))
        price = int(np.asarray(structured.price))
        quote_id = int(np.asarray(structured.quote_id))
        names = ("WAIT", "BID", "ASK", "BUY", "SELL", "CANCEL_BID", "CANCEL_ASK")
        suffix = f" {_SUIT_NAMES[suit]}"
        if kind in (1, 2):
            suffix += f" {price}"
        elif kind in (3, 4):
            suffix += f" quote={quote_id}"
        return names[kind] + suffix

    def __str__(self) -> str:
        """Return a compact state label without private cards or the goal."""
        if self._sim_state is None:
            return f"Figgie(stage={self._stage}, cards_dealt={self._cards_dealt})"
        if self._terminal:
            status = "truncated" if bool(np.asarray(self._sim_state.truncated)) else "settled"
            return f"Figgie({status})"
        if self._actor is None:
            return f"Figgie(stage={self._stage}, events={int(np.asarray(self._sim_state.events))})"
        return (
            f"Figgie(actor={self._actor}, events={int(np.asarray(self._sim_state.events))}, "
            f"time_ms={int(np.asarray(self._sim_state.time_ms))})"
        )


class FiggieObserver:
    """Flat actor-relative observation backed by the shared adapter features."""

    def __init__(self, config: Config):
        self._config = config
        self.tensor = np.zeros(_feature_size(config), dtype=np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state: FiggieState, player: int) -> None:
        """Fill the tensor with only the requested player's private observation."""
        self.tensor.fill(0.0)
        if state._sim_state is None:
            return
        self.tensor[:] = observation_array(state._actor_observation(player), self._config)

    def string_from(self, state: FiggieState, player: int) -> str:
        """Return a string form of the same actor-private feature vector."""
        self.set_from(state, player)
        return np.array2string(self.tensor, precision=3, separator=",")


pyspiel.register_game(_GAME_TYPE, FiggieGame)
