"""Run one complete native FiggieJAX round with observation-only agents."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from dataclasses import asdict
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from figgie_jax import __version__
from figgie_jax.bots import heuristic_policy, random_policy
from figgie_jax.env import FiggieEnv, next_actor
from figgie_jax.observations import Observation, observe
from figgie_jax.types import Action, Config, action

Policy = Callable[[jax.Array, Observation], Action]


@lru_cache(maxsize=16)
def _compiled_functions(
    config: Config,
) -> tuple[
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
]:
    """Compile and cache simulator and baseline-policy functions per config."""
    env = FiggieEnv(config)
    return (
        jax.jit(env.reset),
        jax.jit(env.apply_event),
        jax.jit(next_actor),
        jax.jit(partial(observe, config=config)),
        jax.jit(partial(heuristic_policy, config=config)),
        jax.jit(partial(random_policy, config=config)),
    )


def _host_int(value: jax.Array) -> int:
    """Copy one scalar from the selected JAX device to Python."""
    return int(jax.device_get(value))


def play_round(config: Config, seed: int, policy_name: str) -> dict[str, object]:
    """Play one round, carrying live state until settlement or event truncation.

    Args:
        config: Static game configuration.
        seed: Seed used for the deal, clock ties, and policy keys.
        policy_name: ``heuristic`` or ``random`` for every seat.

    Returns:
        A JSON-compatible summary. A truncated round has no terminal rewards.
    """
    if policy_name not in ("heuristic", "random"):
        raise ValueError("policy_name must be 'heuristic' or 'random'")
    reset_fn, apply_event_fn, next_actor_fn, observe_fn, heuristic_fn, random_fn = (
        _compiled_functions(config)
    )
    policy_fn = heuristic_fn if policy_name == "heuristic" else random_fn
    root_key = jax.random.key(seed)
    state = reset_fn(jax.random.fold_in(root_key, 0))
    policy_key = jax.random.fold_in(root_key, 1)
    trades = 0
    accepted_actions = 0

    while not bool(jax.device_get(state.done)):
        actor_array = next_actor_fn(state)
        actor = _host_int(actor_array)
        at_ms = max(_host_int(state.time_ms), _host_int(state.ready_at[actor]))

        if at_ms >= config.duration_ms:
            # The engine settles at the deadline and ignores this placeholder action.
            transition = apply_event_fn(state, actor_array, action(), jnp.int32(at_ms))
        else:
            decision_state = state._replace(time_ms=jnp.int32(at_ms))
            obs = observe_fn(decision_state, actor_array)
            event_key = jax.random.fold_in(policy_key, _host_int(state.events))
            selected = policy_fn(event_key, obs)
            transition = apply_event_fn(state, actor_array, selected, jnp.int32(at_ms))

        state = transition.state
        accepted_actions += int(jax.device_get(transition.info.accepted))
        trades += int(jax.device_get(transition.info.traded))

    terminal = bool(jax.device_get(state.done)) and not bool(jax.device_get(state.truncated))
    rewards = jax.device_get(state.cash - state.initial_cash).tolist() if terminal else None
    return {
        "seed": seed,
        "configuration": asdict(config),
        "players": config.num_players,
        "duration_ms": config.duration_ms,
        "max_price": config.max_price,
        "max_events": config.max_events,
        "policy": policy_name,
        "events": _host_int(state.events),
        "accepted_actions": accepted_actions,
        "trades": trades,
        "terminal": terminal,
        "truncated": bool(jax.device_get(state.truncated)),
        "cash_after_round": jax.device_get(state.cash).tolist(),
        "terminal_profit": rewards,
        "runtime": {
            "python": platform.python_version(),
            "figgie_jax_engine": __version__,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "results_are_simulated": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, choices=(4, 5), default=4)
    parser.add_argument("--duration-ms", type=int, default=10_000)
    parser.add_argument("--max-events", type=int, default=50_000)
    parser.add_argument("--max-price", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The summary is always printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured native rollout and print a reproducible summary."""
    args = _parse_args()
    config = Config(
        num_players=args.players,
        duration_ms=args.duration_ms,
        max_events=args.max_events,
        max_price=args.max_price,
    )
    summary = play_round(config, args.seed, args.policy)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
