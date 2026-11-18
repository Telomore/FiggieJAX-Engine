"""Small compiled episodic REINFORCE example, without trainer dependencies.

This educational learner uses a linear, factorized masked policy and SGD. It is
not a production trainer or a claim of strategic strength. All collection,
gradients, and repeated updates can execute inside one JAX computation.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from figgie_jax.bots import heuristic_policy
from figgie_jax.env import apply_event, next_actor, reset
from figgie_jax.observations import (
    Observation,
    action_masks,
    observe,
    price_mask,
    relative_features,
)
from figgie_jax.types import Action, Config, Kind, State, action


class Decisions(NamedTuple):
    """Fixed-shape private policy records, with time then table batch axes."""

    features: jax.Array
    masks: jax.Array
    choices: jax.Array
    learner_turn: jax.Array


class Games(NamedTuple):
    """One complete or explicitly truncated round per independent table."""

    states: State
    learner_seats: jax.Array
    rewards: jax.Array
    decisions: Decisions


def initialize(key: jax.Array, config: Config) -> jax.Array:
    """Initialize linear kind, suit, and price heads from private features."""
    obs = observe(reset(jax.random.fold_in(key, 0), config), jnp.int32(0), config)
    size = relative_features(obs, config).size
    return 0.01 * jax.random.normal(jax.random.fold_in(key, 1), (size, 12 + config.max_price))


def sample_policy(
    weights: jax.Array, key: jax.Array, obs: Observation, config: Config
) -> tuple[Action, tuple[jax.Array, jax.Array, jax.Array]]:
    """Sample kind, conditional suit, and conditional price using only an observation."""
    feature = relative_features(obs, config)
    logits = feature @ weights
    kind_key, suit_key, price_key = jax.random.split(key, 3)
    kinds, suits = action_masks(obs, config)

    def sample(sample_key, head, mask):
        return jax.random.categorical(sample_key, jnp.where(mask, head, -1e9)).astype(jnp.int32)

    kind = sample(kind_key, logits[:7], kinds)
    # WAIT has no suit decision. A single token avoids a spurious gradient.
    suit_mask = jnp.where(kind == Kind.WAIT, jnp.arange(4) == 0, suits[kind])
    suit = sample(suit_key, logits[7:11], suit_mask)
    prices = price_mask(obs, kind, suit, config)
    price = sample(price_key, logits[11:], prices)
    side = jnp.where(kind == Kind.BUY, 1, 0)
    quote_id = jnp.where((kind == Kind.BUY) | (kind == Kind.SELL), obs.quote_ids[side, suit], 0)
    return Action(kind, suit, price, quote_id), (
        feature,
        jnp.concatenate((kinds, suit_mask, prices)),
        jnp.stack((kind, suit, price)),
    )


def collect(weights: jax.Array, key: jax.Array, config: Config, batch: int) -> Games:
    """Collect full rounds with one randomly assigned learner seat per table.

    Positive clock increments bound opportunities before the deadline, including
    latency and jitter. The extra scan event settles the round. Completed states
    stay absorbing for padding; no live round is discarded or reset mid-round.
    Every seat's reward is accumulated on every event, before any later reset.
    """
    deal_key, seat_key, policy_key = jax.random.split(key, 3)
    states = jax.vmap(lambda k: reset(k, config))(jax.random.split(deal_key, batch))
    seats = jax.random.randint(seat_key, (batch,), 0, config.num_players)
    steps = min(
        config.max_events,
        config.num_players
        * ((config.duration_ms + config.decision_interval_ms - 1) // config.decision_interval_ms)
        + 1,
    )

    def event(state, seat, event_key):
        actor = next_actor(state)
        at_ms = jnp.maximum(state.time_ms, state.ready_at[actor])
        obs = observe(state._replace(time_ms=jnp.minimum(at_ms, config.duration_ms)), actor, config)
        proposed, record = sample_policy(weights, event_key, obs, config)
        learner_turn = (actor == seat) & ~state.done & (at_ms < config.duration_ms)
        selected = jax.lax.cond(
            actor == seat,
            lambda: proposed,
            lambda: heuristic_policy(event_key, obs, config),
        )
        selected = jax.lax.cond(at_ms >= config.duration_ms, lambda: action(), lambda: selected)
        transition = apply_event(state, actor, selected, at_ms, config)
        return transition.state, transition.rewards, Decisions(*record, learner_turn)

    def advance(carry, _):
        states, policy_key, rewards = carry
        policy_key, event_key = jax.random.split(policy_key)
        states, increment, records = jax.vmap(event)(
            states, seats, jax.random.split(event_key, batch)
        )
        return (states, policy_key, rewards + increment), records

    (states, _, rewards), records = jax.lax.scan(
        advance, (states, policy_key, jnp.zeros((batch, config.num_players))), None, length=steps
    )
    return Games(states, seats, rewards, records)


def learner_profits(games: Games) -> jax.Array:
    """Read the learner component even when settlement occurred on an opponent turn."""
    return jnp.take_along_axis(games.rewards, games.learner_seats[:, None], axis=1)[:, 0]


def loss(weights: jax.Array, games: Games, entropy_coefficient: float = 0.01) -> jax.Array:
    """Compute episodic policy loss, excluding safety-cap truncations.

    Each round contributes its sum of learner log probabilities times terminal
    profit in units of 100 chips. The baseline is the mean of OTHER completed
    rounds in this independent batch; it does not use the round's own outcome.
    A batch with one completed round uses zero baseline. Entropy is averaged
    across active learner decisions. Recorded conditional masks are held fixed.
    """
    records = games.decisions
    complete = games.states.done & ~games.states.truncated
    profits = learner_profits(games)
    count = complete.sum()
    others = count - complete.astype(jnp.int32)
    baseline = jnp.where(
        others > 0,
        (jnp.sum(jnp.where(complete, profits, 0.0)) - jnp.where(complete, profits, 0.0))
        / jnp.maximum(others, 1),
        0.0,
    )
    advantages = jax.lax.stop_gradient((profits - baseline) / 100.0)
    logits = records.features @ weights
    masked = jnp.where(records.masks, logits, -1e9)
    log_probability = jnp.zeros(records.learner_turn.shape)
    entropy = jnp.zeros_like(log_probability)
    for head, (start, end) in enumerate(((0, 7), (7, 11), (11, weights.shape[1]))):
        logs = jax.nn.log_softmax(masked[..., start:end], axis=-1)
        log_probability += jnp.take_along_axis(logs, records.choices[..., head, None], -1)[..., 0]
        entropy -= jnp.sum(jnp.exp(logs) * logs, axis=-1)
    active = records.learner_turn & complete[None, :]
    policy_loss = -jnp.sum(jnp.where(active, log_probability * advantages[None, :], 0.0))
    policy_loss /= jnp.maximum(count, 1)
    entropy_bonus = jnp.sum(jnp.where(active, entropy, 0.0)) / jnp.maximum(active.sum(), 1)
    return policy_loss - entropy_coefficient * entropy_bonus


def game_metrics(games: Games) -> dict[str, jax.Array]:
    """Summarize settled simulations separately from incomplete games."""
    complete = games.states.done & ~games.states.truncated
    count = complete.sum()
    profits = learner_profits(games)
    mean = jnp.sum(jnp.where(complete, profits, 0.0)) / jnp.maximum(count, 1)
    variance = jnp.sum(jnp.where(complete, (profits - mean) ** 2, 0.0)) / jnp.maximum(count, 1)
    return {
        "completed_games": count,
        "truncated_games": games.states.truncated.sum(),
        "unfinished_games": (~games.states.done).sum(),
        "learner_decisions": games.decisions.learner_turn.sum(),
        "mean_terminal_profit": mean,
        "population_std_terminal_profit": jnp.sqrt(variance),
        "positive_profit_fraction": jnp.sum(complete & (profits > 0)) / jnp.maximum(count, 1),
    }


def make_train(
    config: Config,
    batch: int,
    updates: int,
    learning_rate: float = 0.01,
    entropy_coefficient: float = 0.01,
):
    """Build one compiled scan of on-policy collection, gradients, and clipped SGD."""
    if batch < 1 or updates < 1:
        raise ValueError("batch and updates must be positive")
    if not 0 < learning_rate < float("inf") or not 0 <= entropy_coefficient < float("inf"):
        raise ValueError("learning rate must be positive and entropy nonnegative, both finite")

    def train(weights, key):
        def update(weights, index):
            games = collect(weights, jax.random.fold_in(key, index), config, batch)
            # The simulator is a sampled environment, not a differentiable model.
            games = jax.tree.map(jax.lax.stop_gradient, games)
            value, gradient = jax.value_and_grad(loss)(weights, games, entropy_coefficient)
            norm = jnp.linalg.norm(gradient)
            weights = weights - learning_rate * gradient / jnp.maximum(norm, 1.0)
            metrics = {
                **game_metrics(games),
                "loss": value,
                "gradient_norm": norm,
                "update_applied": (games.states.done & ~games.states.truncated).any()
                & games.decisions.learner_turn.any(),
            }
            return weights, metrics

        return jax.lax.scan(update, weights, jnp.arange(updates))

    return jax.jit(train)
