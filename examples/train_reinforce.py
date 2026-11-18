"""Train and evaluate a small masked linear policy with native FiggieJAX.

This example demonstrates a real policy-gradient update. It is a learning
exercise, not a benchmark of strategic strength.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from collections.abc import Callable
from dataclasses import asdict
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from figgie_jax import __version__
from figgie_jax.bots import heuristic_policy
from figgie_jax.env import FiggieEnv, next_actor
from figgie_jax.types import Config, action
from figgie_jax_adapters.common import (
    action_count,
    decision_observation,
    decode_action,
    legal_action_mask,
    observation_array,
)


class Episode(NamedTuple):
    """Policy decisions and outcome for one complete game."""

    features: jax.Array
    action_ids: jax.Array
    masks: jax.Array
    profit: float | None
    truncated: bool
    events: int


@lru_cache(maxsize=16)
def _compiled_functions(
    config: Config,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Compile and cache reset, transition, scheduling, and opponent functions."""
    env = FiggieEnv(config)
    return (
        jax.jit(env.reset),
        jax.jit(env.apply_event),
        jax.jit(next_actor),
        jax.jit(partial(heuristic_policy, config=config)),
    )


def _host_int(value: jax.Array) -> int:
    """Copy one scalar from the selected JAX device to Python."""
    return int(jax.device_get(value))


@jax.jit
def _sample_action(
    weights: jax.Array,
    feature: jax.Array,
    mask: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Sample one legal categorical action from the linear policy."""
    logits = feature @ weights
    masked_logits = jnp.where(mask, logits, -1e9)
    return jax.random.categorical(key, masked_logits)


def play_episode(
    config: Config,
    episode_key: jax.Array,
    weights: jax.Array,
    train: bool,
) -> Episode:
    """Play one round with the learner in seat zero and heuristic opponents.

    The learner sees only its private observation. Its terminal chip profit is
    read from the environment reward after settlement, including rewards caused
    by trades on opponents' turns. A truncated game produces no learning target.
    """
    reset_fn, apply_event_fn, next_actor_fn, heuristic_fn = _compiled_functions(config)
    state = reset_fn(jax.random.fold_in(episode_key, 0))
    empty_feature = observation_array(decision_observation(state, jnp.int32(0), config), config)
    policy_key = jax.random.fold_in(episode_key, 1)
    feature_rows: list[jax.Array] = []
    action_ids: list[int] = []
    legal_masks: list[jax.Array] = []
    learner_reward = 0.0

    while not bool(jax.device_get(state.done)):
        actor = next_actor_fn(state)
        actor_id = _host_int(actor)
        at_ms = max(_host_int(state.time_ms), _host_int(state.ready_at[actor_id]))
        if at_ms >= config.duration_ms:
            transition = apply_event_fn(state, actor, action(), jnp.int32(at_ms))
            learner_reward += float(jax.device_get(transition.rewards[0]))
            state = transition.state
            continue

        obs = decision_observation(state, actor, config)
        event_key = jax.random.fold_in(policy_key, _host_int(state.events))
        if actor_id == 0:
            feature = observation_array(obs, config)
            mask = legal_action_mask(obs, config)
            selected_id = _host_int(_sample_action(weights, feature, mask, event_key))
            selected = decode_action(selected_id, obs, config)
            if train:
                feature_rows.append(feature)
                action_ids.append(selected_id)
                legal_masks.append(mask)
        else:
            selected = heuristic_fn(event_key, obs)

        transition = apply_event_fn(state, actor, selected, jnp.int32(at_ms))
        learner_reward += float(jax.device_get(transition.rewards[0]))
        state = transition.state

    truncated = bool(jax.device_get(state.truncated))
    complete = bool(jax.device_get(state.done)) and not truncated
    profit = learner_reward if complete else None
    features = (
        jnp.stack(feature_rows)
        if feature_rows
        else jnp.zeros((0, empty_feature.size), dtype=jnp.float32)
    )
    ids = jnp.asarray(action_ids, jnp.int32)
    masks = (
        jnp.stack(legal_masks)
        if legal_masks
        else jnp.zeros((0, action_count(config)), dtype=jnp.bool_)
    )
    return Episode(features, ids, masks, profit, truncated, _host_int(state.events))


def policy_loss(
    weights: jax.Array,
    features: jax.Array,
    action_ids: jax.Array,
    masks: jax.Array,
    advantages: jax.Array,
    entropy_coefficient: float,
) -> jax.Array:
    """Compute episodic REINFORCE loss with an entropy bonus.

    For decision ``t``, the policy has probability ``pi(a_t | o_t)`` and the
    decision advantage is ``A_t``. The loss is the mean negative
    ``A_t * log pi(a_t | o_t)`` plus an entropy bonus. ``o_t`` is a relative
    private observation, and ``a_t`` is one legal finite action ID.
    """
    logits = features @ weights
    log_probabilities = jax.nn.log_softmax(jnp.where(masks, logits, -1e9), axis=-1)
    chosen_log_probabilities = jnp.take_along_axis(log_probabilities, action_ids[:, None], axis=-1)[
        :, 0
    ]
    probabilities = jnp.exp(log_probabilities)
    entropy = -jnp.sum(jnp.where(masks, probabilities * log_probabilities, 0.0), axis=-1)
    decision_loss = -advantages * chosen_log_probabilities - entropy_coefficient * entropy
    return jnp.mean(decision_loss)


@jax.jit
def _policy_gradient(
    weights: jax.Array,
    features: jax.Array,
    action_ids: jax.Array,
    masks: jax.Array,
    advantages: jax.Array,
    entropy_coefficient: jax.Array,
) -> jax.Array:
    """Differentiate one batch of on-policy episodes."""
    return jax.grad(policy_loss)(
        weights,
        features,
        action_ids,
        masks,
        advantages,
        entropy_coefficient,
    )


def _episode_keys(root_key: jax.Array, domain: int, count: int) -> list[jax.Array]:
    """Derive indexed keys in a domain that is separate for train and evaluation."""
    domain_key = jax.random.fold_in(root_key, domain)
    return [jax.random.fold_in(domain_key, index) for index in range(count)]


def _gradient_update(
    weights: jax.Array,
    features: jax.Array,
    action_ids: jax.Array,
    masks: jax.Array,
    advantages: jax.Array,
    learning_rate: float,
    entropy_coefficient: float,
) -> tuple[jax.Array, float]:
    """Apply one clipped batch SGD step and return its gradient norm."""
    if action_ids.size == 0:
        return weights, 0.0
    gradient = _policy_gradient(
        weights,
        features,
        action_ids,
        masks,
        advantages,
        jnp.float32(entropy_coefficient),
    )
    norm = jnp.sqrt(jnp.sum(jnp.square(gradient)))
    scale = jnp.minimum(1.0, 1.0 / jnp.maximum(norm, 1e-8))
    updated = weights - learning_rate * scale * gradient
    return updated, float(jax.device_get(norm))


def _evaluate(
    config: Config,
    weights: jax.Array,
    episode_keys: list[jax.Array],
) -> dict[str, float | int | None]:
    """Evaluate on fixed seeds that are separate from training episodes."""
    profits: list[float] = []
    truncated = 0
    for episode_key in episode_keys:
        episode = play_episode(config, episode_key, weights, train=False)
        if episode.profit is None:
            truncated += 1
        else:
            profits.append(episode.profit)
    return {
        "games": len(episode_keys),
        "completed_games": len(profits),
        "truncated_games": truncated,
        "mean_terminal_profit": statistics.fmean(profits) if profits else None,
        "population_std_terminal_profit": statistics.pstdev(profits) if len(profits) > 1 else 0.0,
        "win_fraction": sum(profit > 0 for profit in profits) / len(profits) if profits else None,
    }


def train(
    config: Config,
    seed: int,
    episodes: int,
    eval_episodes: int,
    learning_rate: float,
    entropy_coefficient: float,
) -> dict[str, object]:
    """Train a linear actor and evaluate initial and final weights on held-out seeds."""
    root_key = jax.random.key(seed)
    reset_fn, _, _, _ = _compiled_functions(config)
    initial_state = reset_fn(jax.random.fold_in(root_key, 1))
    initial_obs = decision_observation(initial_state, jnp.int32(0), config)
    feature_size = observation_array(initial_obs, config).size
    num_actions = action_count(config)
    initial_weights = (
        jax.random.normal(jax.random.fold_in(root_key, 2), (feature_size, num_actions)) * 0.01
    )
    eval_keys = _episode_keys(root_key, 20, eval_episodes)
    initial_evaluation = _evaluate(config, initial_weights, eval_keys)

    baseline = 0.0
    completed = 0
    skipped_truncated = 0
    training_features: list[jax.Array] = []
    training_action_ids: list[jax.Array] = []
    training_masks: list[jax.Array] = []
    training_advantages: list[jax.Array] = []
    training_keys = _episode_keys(root_key, 10, episodes)
    for episode_index in range(episodes):
        episode = play_episode(config, training_keys[episode_index], initial_weights, train=True)
        if episode.profit is None:
            skipped_truncated += 1
            continue
        normalized_advantage = max(-5.0, min(5.0, (episode.profit - baseline) / 100.0))
        if episode.action_ids.size:
            training_features.append(episode.features)
            training_action_ids.append(episode.action_ids)
            training_masks.append(episode.masks)
            training_advantages.append(
                jnp.full((episode.action_ids.size,), normalized_advantage, dtype=jnp.float32)
            )
        baseline = 0.9 * baseline + 0.1 * episode.profit
        completed += 1

    if training_features:
        features = jnp.concatenate(training_features)
        action_ids = jnp.concatenate(training_action_ids)
        masks = jnp.concatenate(training_masks)
        advantages = jnp.concatenate(training_advantages)
        weights, gradient_norm = _gradient_update(
            initial_weights,
            features,
            action_ids,
            masks,
            advantages,
            learning_rate,
            entropy_coefficient,
        )
        updates_applied = 1
        decision_count = int(action_ids.size)
    else:
        weights = initial_weights
        gradient_norm = 0.0
        updates_applied = 0
        decision_count = 0

    final_evaluation = _evaluate(config, weights, eval_keys)
    return {
        "seed": seed,
        "training_key_derivation": "fold_in(fold_in(key(seed), 10), episode_index)",
        "evaluation_key_derivation": "fold_in(fold_in(key(seed), 20), episode_index)",
        "training_episode_indices": list(range(episodes)),
        "evaluation_episode_indices": list(range(eval_episodes)),
        "training_episodes_requested": episodes,
        "training_episodes_completed": completed,
        "truncated_training_episodes_skipped": skipped_truncated,
        "training_decision_count": decision_count,
        "learning_rate": learning_rate,
        "entropy_coefficient": entropy_coefficient,
        "action_count": num_actions,
        "feature_count": feature_size,
        "parameter_count": feature_size * num_actions,
        "updates_applied": updates_applied,
        "gradient_norm": gradient_norm,
        "parameter_l2_change": float(jax.device_get(jnp.linalg.norm(weights - initial_weights))),
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "results_are_simulated": True,
        "claim": (
            "This short example demonstrates policy updates; it does not measure "
            "strategic strength."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, choices=(4, 5), default=4)
    parser.add_argument("--duration-ms", type=int, default=10_000)
    parser.add_argument("--max-events", type=int, default=50_000)
    parser.add_argument("--max-price", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The summary is always printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    """Run seeded training and separate held-out evaluation games."""
    args = _parse_args()
    config = Config(
        num_players=args.players,
        duration_ms=args.duration_ms,
        max_events=args.max_events,
        max_price=args.max_price,
    )
    summary = {
        "configuration": {
            **asdict(config),
            "reward": "terminal chip profit relative to pre-ante cash",
            "opponents": "fixed heuristic_policy in seats 1 through N-1",
            "actor_observation": "relative private observation; no hidden state",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "figgie_jax_engine": __version__,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "training": train(
            config,
            args.seed,
            args.episodes,
            args.eval_episodes,
            args.learning_rate,
            args.entropy_coefficient,
        ),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
