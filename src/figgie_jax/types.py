"""Fixed-shape simulator types and validated static configuration."""

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

SUITS = ("spades", "clubs", "hearts", "diamonds")


class Kind(IntEnum):
    """Structured action kinds; quotes never execute automatically."""

    WAIT = 0
    BID = 1
    ASK = 2
    BUY = 3
    SELL = 4
    CANCEL_BID = 5
    CANCEL_ASK = 6


class Reason(IntEnum):
    """Public event outcomes for replay and debugging."""

    ACCEPTED = 0
    INVALID = 1
    STALE_QUOTE = 2
    ROUND_OVER = 3
    TIME_REVERSED = 4
    EVENT_LIMIT = 5


@dataclass(frozen=True)
class Config:
    """Static simulation settings; times are integer milliseconds.

    Latency delays decision opportunities, not delivery of a precomputed action.
    An empty latency tuple gives every agent zero added latency during training.
    The positive base interval guarantees progress even with zero extra latency.
    Prices use whole chips. The configurable cap is a simulator constraint.
    """

    num_players: int = 4
    duration_ms: int = 240_000
    initial_cash: float = 350.0
    max_price: int = 200
    decision_interval_ms: int = 250
    latency_ms: tuple[int, ...] = ()
    jitter_ms: int = 0
    max_events: int = 50_000
    history_size: int = 16
    allow_negative_cash: bool = True

    def __post_init__(self):
        if not isinstance(self.num_players, int) or self.num_players not in (4, 5):
            raise ValueError("Figgie requires four or five players, including bots.")
        for name in (
            "duration_ms",
            "max_price",
            "decision_interval_ms",
            "max_events",
            "history_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 < value < 10_000_000:
                raise ValueError(f"{name} must be a positive integer below 10,000,000.")
        if self.max_price > 10_000:
            raise ValueError("max_price must not exceed 10,000.")
        if not 0 <= self.initial_cash <= 1_000_000:
            raise ValueError("initial_cash must be finite and between 0 and 1,000,000.")
        if not isinstance(self.jitter_ms, int) or not 0 <= self.jitter_ms < 10_000_000:
            raise ValueError("jitter_ms must be a nonnegative integer below 10,000,000.")
        if self.latency_ms and len(self.latency_ms) != self.num_players:
            raise ValueError("latency_ms needs one value per player.")
        if any(
            not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < 10_000_000
            for x in self.latency_ms
        ):
            raise ValueError("Each latency must be a nonnegative integer below 10,000,000.")

    @property
    def latencies(self) -> tuple[int, ...]:
        """Return explicit per-player latency values."""
        return self.latency_ms or (0,) * self.num_players


class Action(NamedTuple):
    """An action; BUY/SELL require the displayed quote identifier."""

    kind: jax.Array
    suit: jax.Array
    price: jax.Array
    quote_id: jax.Array


class State(NamedTuple):
    """Full privileged simulator state; never pass this to a decentralized actor."""

    hands: jax.Array
    cash: jax.Array
    initial_cash: jax.Array
    deck: jax.Array
    goal: jax.Array
    prices: jax.Array
    owners: jax.Array
    quote_ids: jax.Array
    history: jax.Array
    trade_count: jax.Array
    time_ms: jax.Array
    events: jax.Array
    done: jax.Array
    truncated: jax.Array
    ready_at: jax.Array
    key: jax.Array


class EventInfo(NamedTuple):
    """Public action outcome; buyer and seller are -1 without a trade."""

    accepted: jax.Array
    traded: jax.Array
    buyer: jax.Array
    seller: jax.Array
    suit: jax.Array
    price: jax.Array
    reason: jax.Array


class Transition(NamedTuple):
    """State, per-player sparse terminal rewards, and public event outcome."""

    state: State
    rewards: jax.Array
    info: EventInfo


def action(kind: int = Kind.WAIT, suit: int = 0, price: int = 0, quote_id: int = 0) -> Action:
    """Construct a scalar JAX action from host values."""
    return Action(*(jnp.asarray(x, jnp.int32) for x in (kind, suit, price, quote_id)))
