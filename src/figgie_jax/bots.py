"""Reproducible observation-only baseline agents."""

from typing import Literal

import jax
import jax.numpy as jnp

from figgie_jax.observations import Observation, action_masks, price_mask
from figgie_jax.types import Action, Config, Kind

AgentKind = Literal["human", "heuristic", "random", "checkpoint"]


def random_policy(key: jax.Array, obs: Observation, config: Config = Config()) -> Action:
    """Sample a legal structured action without using privileged state."""
    kind_key, suit_key, price_key = jax.random.split(key, 3)
    kinds, suits = action_masks(obs, config)
    kind = jax.random.categorical(kind_key, jnp.where(kinds, 0.0, -1e9)).astype(jnp.int32)
    suit = jax.random.categorical(suit_key, jnp.where(suits[kind], 0.0, -1e9)).astype(jnp.int32)
    prices = price_mask(obs, kind, suit, config)
    price = jax.random.categorical(price_key, jnp.where(prices, 0.0, -1e9)).astype(jnp.int32)
    side = jnp.where(kind == Kind.BUY, 1, 0)
    quote_id = jnp.where((kind == Kind.BUY) | (kind == Kind.SELL), obs.quote_ids[side, suit], 0)
    return Action(kind, suit, price, quote_id)


def heuristic_policy(key: jax.Array, obs: Observation, config: Config = Config()) -> Action:
    """Use a deliberately simple inventory heuristic; no claim of strategic strength.

    The paired suit of the most common private suit is a crude goal guess.
    The bot buys below its heuristic value and sells above it. Other actions
    quote around that value. This is a playable baseline, not Bayesian inference.
    """
    suit_key, side_key = jax.random.split(key)
    guessed_goal = jnp.argmax(obs.hand) ^ 1
    values = jnp.where(jnp.arange(4) == guessed_goal, 18, 5)
    _, suits = action_masks(obs, config)
    buys = suits[Kind.BUY] & (obs.prices[1] <= values)
    sells = suits[Kind.SELL] & (obs.prices[0] >= values)
    buy_suit = jnp.argmax(buys)
    sell_suit = jnp.argmax(sells)
    suit = jax.random.randint(suit_key, (), 0, 4)
    bid = jax.random.bernoulli(side_key)
    kind = jnp.where(bid, Kind.BID, Kind.ASK)
    kind = jnp.where(suits[kind, suit], kind, Kind.WAIT)
    price = jnp.clip(values[suit] + jnp.where(bid, -2, 2), 1, config.max_price)
    candidate = Action(kind, suit, price, jnp.int32(0))
    # The final exchange validates improving prices and stale quotes.
    candidate = jax.lax.cond(
        jnp.any(sells),
        lambda: Action(jnp.int32(Kind.SELL), sell_suit, jnp.int32(0), obs.quote_ids[0, sell_suit]),
        lambda: candidate,
    )
    return jax.lax.cond(
        jnp.any(buys),
        lambda: Action(jnp.int32(Kind.BUY), buy_suit, jnp.int32(0), obs.quote_ids[1, buy_suit]),
        lambda: candidate,
    )
