"""Framework-independent action and observation helpers for RL adapters."""

from __future__ import annotations

import operator

import jax.numpy as jnp
import numpy as np

from figgie_jax.env import next_actor
from figgie_jax.observations import Observation, action_masks, observe, relative_features
from figgie_jax.types import Action, Config, State
from figgie_jax.types import action as make_action


def action_count(config: Config) -> int:
    """Return the size of the canonical finite action space.

    Each bid and ask has one action for every suit and integer price from 1 to
    ``config.max_price``. Each remaining action kind has one action per suit.
    """
    return 1 + 8 * config.max_price + 16


def decode_action(action_id: int, obs: Observation, config: Config) -> Action:
    """Decode a finite action ID into a structured engine action.

    BUY and SELL actions use the quote identifier visible in ``obs``. The
    simulator still validates that identifier when it processes the event.

    Args:
        action_id: Integer ID in ``[0, action_count(config))``.
        obs: Current private observation for the acting player.
        config: Static simulator configuration.

    Returns:
        A scalar JAX ``Action`` suitable for ``FiggieEnv.apply_event``.

    Raises:
        ValueError: If ``action_id`` is outside the finite action space.
        TypeError: If ``action_id`` is not an integer scalar.
    """
    try:
        value = operator.index(action_id)
    except TypeError as exc:
        raise TypeError("action_id must be an integer scalar") from exc

    if not 0 <= value < action_count(config):
        raise ValueError(f"action_id must be in [0, {action_count(config)})")
    if value == 0:
        return make_action()

    price_actions = 4 * config.max_price
    if value <= price_actions:
        offset = value - 1
        suit, price_offset = divmod(offset, config.max_price)
        return make_action(kind=1, suit=suit, price=price_offset + 1)

    ask_start = 1 + price_actions
    if value < ask_start + price_actions:
        offset = value - ask_start
        suit, price_offset = divmod(offset, config.max_price)
        return make_action(kind=2, suit=suit, price=price_offset + 1)

    simple_start = 1 + 2 * price_actions
    simple_offset = value - simple_start
    kind_offset, suit = divmod(simple_offset, 4)
    kind = 3 + kind_offset
    if kind in (3, 4):
        side = 1 if kind == 3 else 0
        quote_id = int(np.asarray(obs.quote_ids)[side, suit])
        return make_action(kind=kind, suit=suit, quote_id=quote_id)
    return make_action(kind=kind, suit=suit)


def legal_action_mask(obs: Observation, config: Config) -> np.ndarray:
    """Return a boolean mask for actions valid from observable information.

    The mask follows the simulator's quote-price, ownership, inventory, and
    credit rules. It cannot know whether a counterparty has a card available,
    and the exchange remains the final authority when it processes an action.
    """
    mask = np.zeros(action_count(config), dtype=np.bool_)
    mask[0] = True

    owners = np.asarray(obs.owners)
    prices = np.asarray(obs.prices)
    quote_ids = np.asarray(obs.quote_ids)
    hand = np.asarray(obs.hand)
    cash = np.asarray(obs.cash)
    player = int(np.asarray(obs.player))
    own_cash = float(cash[player])
    max_price = config.max_price
    bid_base = 1
    ask_base = 1 + 4 * max_price
    buy_base = 1 + 8 * max_price
    sell_base = buy_base + 4
    cancel_bid_base = sell_base + 4
    cancel_ask_base = cancel_bid_base + 4

    for suit in range(4):
        bid_owner = int(owners[0, suit])
        bid_price = int(prices[0, suit])
        bid_start = 1 if bid_owner in (-1, player) else bid_price + 1
        bid_end = max_price
        if not config.allow_negative_cash:
            bid_end = min(bid_end, int(np.floor(own_cash)))
        if bid_start <= bid_end:
            prices_range = np.arange(bid_start, bid_end + 1)
            mask[bid_base + suit * max_price + prices_range - 1] = True

        ask_owner = int(owners[1, suit])
        ask_price = int(prices[1, suit])
        ask_start = 1
        ask_end = max_price if ask_owner in (-1, player) else ask_price - 1
        if hand[suit] <= 0:
            ask_end = 0
        if ask_start <= ask_end:
            prices_range = np.arange(ask_start, ask_end + 1)
            mask[ask_base + suit * max_price + prices_range - 1] = True

        has_other_offer = ask_owner >= 0 and ask_owner != player and quote_ids[1, suit] > 0
        can_buy = has_other_offer and (config.allow_negative_cash or own_cash >= ask_price)
        mask[buy_base + suit] = can_buy

        has_other_bid = bid_owner >= 0 and bid_owner != player and quote_ids[0, suit] > 0
        mask[sell_base + suit] = bool(has_other_bid and hand[suit] > 0)
        mask[cancel_bid_base + suit] = bid_owner == player
        mask[cancel_ask_base + suit] = ask_owner == player

    # Keep the engine's factorized public mask aligned with this flattened one.
    kind_mask, suit_mask = action_masks(obs, config)
    expected_suits = np.zeros((7, 4), dtype=np.bool_)
    expected_suits[0] = True
    for suit in range(4):
        expected_suits[1, suit] = mask[
            bid_base + suit * max_price : bid_base + (suit + 1) * max_price
        ].any()
        expected_suits[2, suit] = mask[
            ask_base + suit * max_price : ask_base + (suit + 1) * max_price
        ].any()
        expected_suits[3, suit] = mask[buy_base + suit]
        expected_suits[4, suit] = mask[sell_base + suit]
        expected_suits[5, suit] = mask[cancel_bid_base + suit]
        expected_suits[6, suit] = mask[cancel_ask_base + suit]
    actual_suits = np.asarray(suit_mask)
    actual_kinds = np.asarray(kind_mask)
    if not np.array_equal(expected_suits, actual_suits) or not np.array_equal(
        expected_suits.any(axis=1), actual_kinds
    ):
        raise RuntimeError("Finite action mask does not match the engine observation mask.")
    return mask


def observation_array(obs: Observation, config: Config) -> np.ndarray:
    """Return seat-relative actor features as a one-dimensional float32 array."""
    return np.asarray(relative_features(obs, config), dtype=np.float32)


def decision_observation(state: State, player: int, config: Config) -> Observation:
    """Observe one player at its valid decision time.

    The current actor sees the market at its upcoming clock, clamped to the
    round deadline. Other seats see the state at its current simulation time.
    This preserves the engine's asynchronous schedule without exposing another
    player's private hand.
    """
    if not 0 <= player < config.num_players:
        raise ValueError(f"player must be in [0, {config.num_players})")

    actor = int(np.asarray(next_actor(state)))
    if player == actor:
        decision_ms = jnp.minimum(
            jnp.maximum(state.time_ms, state.ready_at[actor]), config.duration_ms
        )
        state = state._replace(time_ms=decision_ms)
    return observe(state, jnp.asarray(player, jnp.int32), config)
