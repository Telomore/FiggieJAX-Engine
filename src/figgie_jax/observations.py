"""Private actor observations, explicit privileged critic inputs, and action masks."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from figgie_jax.types import Config, Kind, State


class Observation(NamedTuple):
    """Information available to one player, excluding the hidden deal and goal."""

    hand: jax.Array
    cash: jax.Array
    player: jax.Array
    prices: jax.Array
    owners: jax.Array
    quote_ids: jax.Array
    history: jax.Array
    history_valid: jax.Array
    remaining_ms: jax.Array


def observe(state: State, player: jax.Array, config: Config = Config()) -> Observation:
    """Return a private observation; trade history is oldest to newest, padded at end."""
    count = jnp.minimum(state.trade_count, config.history_size)
    start = jnp.maximum(state.trade_count - config.history_size, 0)
    offsets = jnp.arange(config.history_size)
    history = state.history[(start + offsets) % config.history_size]
    valid = offsets < count
    return Observation(
        state.hands[player],
        state.cash,
        player,
        state.prices,
        state.owners,
        state.quote_ids,
        jnp.where(valid[:, None], history, -1),
        valid,
        jnp.maximum(config.duration_ms - state.time_ms, 0),
    )


def features(obs: Observation, config: Config = Config()) -> jax.Array:
    """Normalize public/private information into a fixed actor feature vector."""
    history_scale = jnp.array(
        [config.duration_ms, config.num_players, config.num_players, 4, config.max_price],
        jnp.float32,
    )
    return jnp.concatenate(
        [
            obs.hand / 12.0,
            obs.cash / 350.0,
            jax.nn.one_hot(obs.player, config.num_players),
            (obs.prices / config.max_price).ravel(),
            jax.nn.one_hot(obs.owners + 1, config.num_players + 1).ravel(),
            (jnp.where(obs.history_valid[:, None], obs.history, 0) / history_scale).ravel(),
            obs.history_valid.astype(jnp.float32),
            jnp.reshape(obs.remaining_ms / config.duration_ms, (1,)),
        ]
    ).astype(jnp.float32)


def relative_features(obs: Observation, config: Config = Config()) -> jax.Array:
    """Encode all player references relative to the observer, whose index is zero.

    Cash, quote owners, and public trade participants use the same cyclic seat
    relabeling. Suit identities and the private hand stay unchanged. Padding and
    absent-owner sentinels remain -1. No physical seat identifier is exposed.
    """
    order = (jnp.arange(config.num_players) + obs.player) % config.num_players

    def relative_player(indices):
        return jnp.where(indices >= 0, (indices - obs.player) % config.num_players, -1)

    history = obs.history.at[:, 1:3].set(relative_player(obs.history[:, 1:3]))
    return features(
        obs._replace(
            cash=obs.cash[order],
            player=jnp.int32(0),
            owners=relative_player(obs.owners),
            history=history,
        ),
        config,
    )


def critic_features(state: State, config: Config = Config()) -> jax.Array:
    """Expose hidden information explicitly for a centralized critic only."""
    public = features(observe(state, jnp.int32(0), config), config)
    return jnp.concatenate(
        [public, state.hands.ravel() / 12.0, state.deck / 12.0, jax.nn.one_hot(state.goal, 4)]
    )


def action_masks(obs: Observation, config: Config = Config()) -> tuple[jax.Array, jax.Array]:
    """Return kind [7] and conditional suit [7,4] masks using observable information.

    Inventory at the counterparty and future quote races remain exchange checks.
    """
    owned_bid, owned_ask = obs.owners == obs.player
    other_bid, other_ask = (obs.owners >= 0) & (obs.owners != obs.player)
    bid_room = (obs.owners[0] == -1) | owned_bid | (obs.prices[0] < config.max_price)
    ask_room = (obs.owners[1] == -1) | owned_ask | (obs.prices[1] > 1)
    if not config.allow_negative_cash:
        bid_room &= obs.cash[obs.player] >= jnp.where(
            (obs.owners[0] >= 0) & ~owned_bid, obs.prices[0] + 1, 1
        )
        other_ask &= obs.cash[obs.player] >= obs.prices[1]
    suits = jnp.stack(
        [
            jnp.ones(4, bool),
            bid_room,
            ask_room & (obs.hand > 0),
            other_ask,
            other_bid & (obs.hand > 0),
            owned_bid,
            owned_ask,
        ]
    )
    return jnp.any(suits, axis=1), suits


def price_mask(
    obs: Observation, kind: jax.Array, suit: jax.Array, config: Config = Config()
) -> jax.Array:
    """Mask price choices; irrelevant price heads use a deterministic zero token."""
    prices = jnp.arange(config.max_price + 1)
    side = jnp.where(kind == Kind.ASK, 1, 0)
    other = (obs.owners[side, suit] >= 0) & (obs.owners[side, suit] != obs.player)
    improves = jnp.where(
        kind == Kind.BID, prices > obs.prices[side, suit], prices < obs.prices[side, suit]
    )
    valid = (prices >= 1) & (~other | improves)
    if not config.allow_negative_cash:
        valid &= (kind != Kind.BID) | (prices <= obs.cash[obs.player])
    return jnp.where((kind == Kind.BID) | (kind == Kind.ASK), valid, prices == 0)
