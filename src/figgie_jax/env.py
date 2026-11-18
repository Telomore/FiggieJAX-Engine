"""Pure JAX reset, asynchronous event transitions, and settlement.

Suit order is spades, clubs, hearts, diamonds. Paired color suits differ by xor 1.
Only the best displayed quote on each side of each suit is retained.
"""

import itertools
from functools import partial

import jax
import jax.numpy as jnp

from figgie_jax.types import Action, Config, EventInfo, Kind, Reason, State, Transition

DECK_LAYOUTS = jnp.asarray(sorted(set(itertools.permutations((8, 10, 10, 12)))), jnp.int32)


def _delay(key: jax.Array, config: Config) -> jax.Array:
    # Zero jitter is the training default. Keep key splitting at the call site
    # unchanged, but avoid random integer generation for a deterministic delay.
    jitter = (
        jax.random.randint(key, (config.num_players,), 0, config.jitter_ms + 1)
        if config.jitter_ms
        else jnp.zeros(config.num_players, jnp.int32)
    )
    return config.decision_interval_ms + jnp.asarray(config.latencies, jnp.int32) + jitter


def reset(key: jax.Array, config: Config = Config(), cash: jax.Array | None = None) -> State:
    """Deal all forty cards from one of twelve uniformly sampled layouts.

    Args:
        key: Explicit JAX PRNG key.
        config: Static shape, rule, and timing settings.
        cash: Optional pre-ante bankroll vector for a subsequent round.

    Returns:
        A fresh privileged state, with the equal ante already deducted.
    """
    layout_key, deal_key, delay_key, next_key = jax.random.split(key, 4)
    deck = DECK_LAYOUTS[jax.random.randint(layout_key, (), 0, 12)]
    cards = jnp.searchsorted(jnp.cumsum(deck), jnp.arange(40), side="right")
    dealt = jax.random.permutation(deal_key, cards).reshape(config.num_players, -1)
    hands = jnp.sum(jax.nn.one_hot(dealt, 4, dtype=jnp.int32), axis=1)
    initial = (
        jnp.full(config.num_players, config.initial_cash, jnp.float32) if cash is None else cash
    )
    return State(
        hands=hands,
        cash=initial - 200.0 / config.num_players,
        initial_cash=initial,
        deck=deck,
        goal=jnp.argmax(deck).astype(jnp.int32) ^ 1,
        prices=jnp.zeros((2, 4), jnp.int32),
        owners=jnp.full((2, 4), -1, jnp.int32),
        quote_ids=jnp.zeros((2, 4), jnp.int32),
        history=jnp.full((config.history_size, 5), -1, jnp.int32),
        trade_count=jnp.int32(0),
        time_ms=jnp.int32(0),
        events=jnp.int32(0),
        done=jnp.bool_(False),
        truncated=jnp.bool_(False),
        ready_at=_delay(delay_key, config),
        key=next_key,
    )


def next_actor(state: State) -> jax.Array:
    """Select the earliest independent clock; break exact ties with seeded randomness."""
    tied = state.ready_at == jnp.min(state.ready_at)
    priorities = jax.random.uniform(state.key, state.ready_at.shape)
    return jnp.argmax(jnp.where(tied, priorities, -1.0)).astype(jnp.int32)


def settle(state: State) -> State:
    """Distribute exactly the economic pot once, including fractional tied shares."""
    counts = state.hands[:, state.goal]
    winners = counts == jnp.max(counts)
    remainder = 200.0 - 10.0 * jnp.sum(counts)
    payout = 10.0 * counts + winners * remainder / jnp.sum(winners)
    return jax.lax.cond(
        state.done,
        lambda: state,
        lambda: state._replace(
            cash=state.cash + payout,
            prices=jnp.zeros_like(state.prices),
            owners=jnp.full_like(state.owners, -1),
            quote_ids=jnp.zeros_like(state.quote_ids),
            done=jnp.bool_(True),
        ),
    )


def _empty_info(reason: int = Reason.INVALID) -> EventInfo:
    return EventInfo(jnp.bool_(False), jnp.bool_(False), *[jnp.int32(-1)] * 4, jnp.int32(reason))


def _process(
    state: State, actor: jax.Array, act: Action, config: Config
) -> tuple[State, EventInfo]:
    """Process one action after validating indices, inventory, and quote freshness."""
    valid_actor = (actor >= 0) & (actor < config.num_players)
    valid_suit = (act.suit >= 0) & (act.suit < 4)
    player = jnp.clip(actor, 0, config.num_players - 1)
    suit = jnp.clip(act.suit, 0, 3)
    bid = act.kind == Kind.BID
    ask = act.kind == Kind.ASK
    buy = act.kind == Kind.BUY
    sell = act.kind == Kind.SELL
    quoting = bid | ask
    accepting = buy | sell
    canceling = (act.kind == Kind.CANCEL_BID) | (act.kind == Kind.CANCEL_ASK)
    side = jnp.where(ask | buy | (act.kind == Kind.CANCEL_ASK), 1, 0)
    owner = state.owners[side, suit]
    old_price = state.prices[side, suit]
    counterparty = jnp.clip(owner, 0, config.num_players - 1)
    price_valid = (act.price >= 1) & (act.price <= config.max_price)
    improves = jnp.where(bid, act.price > old_price, act.price < old_price)
    quote_valid = price_valid & ((owner == -1) | (owner == player) | improves)
    quote_valid &= (~ask) | (state.hands[player, suit] > 0)
    if not config.allow_negative_cash:
        quote_valid &= (~bid) | (state.cash[player] >= act.price)
    buyer = jnp.where(buy, player, counterparty)
    seller = jnp.where(buy, counterparty, player)
    fresh = (act.quote_id == state.quote_ids[side, suit]) & (act.quote_id > 0)
    trade_valid = (owner >= 0) & (owner != player) & fresh
    trade_valid &= state.hands[seller, suit] > 0
    if not config.allow_negative_cash:
        trade_valid &= state.cash[buyer] >= old_price
    valid = valid_actor & (
        (act.kind == Kind.WAIT)
        | (
            valid_suit
            & (
                (quoting & quote_valid)
                | (accepting & trade_valid)
                | (canceling & (owner == player))
            )
        )
    )
    traded = valid & accepting

    def trade():
        # These tiny fixed arrays are cheaper to update by masks than by batched
        # indexed scatters. Buyer and seller are distinct for every valid trade.
        seats = jnp.arange(config.num_players)
        transfer = (seats == buyer).astype(jnp.int32) - (seats == seller).astype(jnp.int32)
        hands = state.hands + transfer[:, None] * (jnp.arange(4) == suit)
        cash = state.cash - transfer * old_price
        row = jnp.array([state.time_ms, buyer, seller, suit, old_price], jnp.int32)
        return state._replace(
            hands=hands,
            cash=cash,
            prices=jnp.zeros_like(state.prices),
            owners=jnp.full_like(state.owners, -1),
            quote_ids=jnp.zeros_like(state.quote_ids),
            history=jnp.where(
                (jnp.arange(config.history_size) == state.trade_count % config.history_size)[
                    :, None
                ],
                row,
                state.history,
            ),
            trade_count=state.trade_count + 1,
        )

    def no_trade():
        change = valid & (quoting | canceling)
        cell = change & (jnp.arange(2)[:, None] == side) & (jnp.arange(4) == suit)
        return state._replace(
            prices=jnp.where(cell, jnp.where(quoting, act.price, 0), state.prices),
            owners=jnp.where(cell, jnp.where(quoting, player, -1), state.owners),
            quote_ids=jnp.where(cell, jnp.where(quoting, state.events + 1, 0), state.quote_ids),
        )

    updated = jax.lax.cond(traded, trade, no_trade)
    stale = valid_actor & valid_suit & accepting & ~fresh
    reason = jnp.where(valid, Reason.ACCEPTED, jnp.where(stale, Reason.STALE_QUOTE, Reason.INVALID))
    info = EventInfo(
        valid,
        traded,
        jnp.where(traded, buyer, -1),
        jnp.where(traded, seller, -1),
        jnp.where(traded, suit, -1),
        jnp.where(traded, old_price, -1),
        reason,
    )
    return updated, info


def apply_event(
    state: State, actor: jax.Array, act: Action, at_ms: jax.Array, config: Config = Config()
) -> Transition:
    """Apply an externally timestamped event, or settle at the round deadline.

    Invalid actions consume an opportunity. An event at the exact deadline cannot
    trade. The event safety cap truncates without inventing an early payout.
    """

    def active():
        def expire():
            return settle(state._replace(time_ms=jnp.int32(config.duration_ms))), _empty_info(
                Reason.ROUND_OVER
            )

        def execute():
            updated, info = _process(state._replace(time_ms=at_ms), actor, act, config)
            delay_key, next_key = jax.random.split(state.key)
            player = jnp.clip(actor, 0, config.num_players - 1)
            updated = updated._replace(
                events=state.events + 1,
                key=next_key,
                ready_at=jnp.where(
                    jnp.arange(config.num_players) == player,
                    at_ms + _delay(delay_key, config),
                    updated.ready_at,
                ),
            )
            capped = updated.events >= config.max_events
            updated = updated._replace(done=capped, truncated=capped)
            return updated, info._replace(reason=jnp.where(capped, Reason.EVENT_LIMIT, info.reason))

        return jax.lax.cond(at_ms >= config.duration_ms, expire, execute)

    invalid_time = at_ms < state.time_ms
    blocked = state.done | invalid_time
    updated, info = jax.lax.cond(
        blocked,
        lambda: (
            state,
            _empty_info(jnp.where(state.done, Reason.ROUND_OVER, Reason.TIME_REVERSED)),
        ),
        active,
    )
    terminal = ~state.done & updated.done & ~updated.truncated
    rewards = jnp.where(terminal, updated.cash - updated.initial_cash, jnp.zeros_like(state.cash))
    return Transition(updated, rewards, info)


def step(state: State, act: Action, config: Config = Config()) -> Transition:
    """Execute the next asynchronous decision opportunity without Python scheduling."""
    actor = next_actor(state)
    return apply_event(state, actor, act, jnp.maximum(state.time_ms, state.ready_at[actor]), config)


class FiggieEnv:
    """Small static-config facade compatible with jit, vmap, and lax.scan."""

    def __init__(self, config: Config = Config()):
        self.config = config
        self.reset = partial(reset, config=config)
        self.step = partial(step, config=config)
        self.apply_event = partial(apply_event, config=config)
