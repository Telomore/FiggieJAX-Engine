"""Versioned deterministic JSON replays and public/omniscient board views."""

import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from figgie_jax import __version__
from figgie_jax.bots import AgentKind
from figgie_jax.env import apply_event, reset
from figgie_jax.observations import observe
from figgie_jax.types import Action, Config, State


class ReplayEvent(BaseModel):
    """A timestamped submitted event, including rejected attempts."""

    model_config = ConfigDict(extra="forbid", strict=True)
    actor: int = Field(ge=0, le=4)
    at_ms: int = Field(ge=0, le=10_000_000)
    kind: int = Field(ge=0, le=6)
    suit: int = Field(ge=0, le=3)
    price: int = Field(ge=0, le=10_000)
    quote_id: int = Field(ge=0, le=10_000_000)


class Replay(BaseModel):
    """Portable replay inputs; board states are recomputed rather than trusted."""

    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(default=1, ge=1, le=1)
    project_version: str = __version__
    seed: int = Field(ge=0, le=2**32 - 1)
    config: dict
    initial_cash: list[float]
    events: list[ReplayEvent] = Field(max_length=50_001)
    agents: list[AgentKind] = Field(default_factory=list, max_length=5)


def snapshot(state: State) -> dict:
    """Convert economic state to a plain snapshot for tests and trusted offline tools."""
    return {
        name: np.asarray(array).tolist()
        for name, array in state._asdict().items()
        if name not in ("key", "ready_at")
    }


def board(state: State, config: Config, player: int = 0, reveal: bool = False) -> dict:
    """Return a board view, revealing opponents only after settlement or for offline replay."""
    obs = observe(state, jnp.int32(player), config)
    show = reveal or bool(state.done)
    return {
        "hand": np.asarray(obs.hand).tolist(),
        "cash": np.asarray(state.cash).tolist(),
        "prices": np.asarray(state.prices).tolist(),
        "owners": np.asarray(state.owners).tolist(),
        "quote_ids": np.asarray(state.quote_ids).tolist(),
        "history": np.asarray(obs.history)[np.asarray(obs.history_valid)].tolist(),
        "remaining_ms": int(obs.remaining_ms),
        "time_ms": int(state.time_ms),
        "done": bool(state.done),
        "truncated": bool(state.truncated),
        "hands": np.asarray(state.hands).tolist() if show else None,
        "goal": int(state.goal) if show else None,
        "deck": np.asarray(state.deck).tolist() if show else None,
        "pnl": np.asarray(state.cash - state.initial_cash).tolist() if show else None,
        "num_players": config.num_players,
        "max_price": config.max_price,
        "latency_ms": list(config.latencies),
        "decision_interval_ms": config.decision_interval_ms,
        "events": int(state.events),
    }


def replay_config(replay: Replay) -> Config:
    """Validate portable settings and bound replay resource use."""
    settings = dict(replay.config)
    settings["latency_ms"] = tuple(settings.get("latency_ms", ()))
    config = Config(**settings)
    if replay.agents and (len(replay.agents) != config.num_players or "human" in replay.agents[1:]):
        raise ValueError("Replay player types do not match the seats")
    if config.max_events > 50_000 or config.history_size > 128 or config.duration_ms > 1_200_000:
        raise ValueError("Replay settings exceed viewer resource limits")
    if len(replay.initial_cash) != config.num_players or not np.isfinite(replay.initial_cash).all():
        raise ValueError("Replay bankroll vector is invalid")
    if any(abs(c) > 1_000_000 for c in replay.initial_cash):
        raise ValueError("Replay bankroll is outside supported bounds")
    if any(e.actor >= config.num_players for e in replay.events):
        raise ValueError("Replay actor is not seated")
    if any(b.at_ms < a.at_ms for a, b in zip(replay.events, replay.events[1:], strict=False)):
        raise ValueError("Replay timestamps must be nondecreasing")
    return config


def reconstruct(replay: Replay) -> tuple[State, list[dict]]:
    """Re-execute every replay event and produce scrubbable omniscient frames."""
    config = replay_config(replay)
    state = reset(jax.random.key(replay.seed), config, jnp.array(replay.initial_cash, jnp.float32))

    def frame_view(current):
        result = board(current, config, reveal=True)
        result["agents"] = replay.agents
        return result

    frames = [frame_view(state)]
    execute = jax.jit(lambda s, p, a, t: apply_event(s, p, a, t, config))
    for event in replay.events:
        if bool(state.done):
            raise ValueError("Replay contains an event after the round ended")
        act = Action(*(jnp.int32(x) for x in (event.kind, event.suit, event.price, event.quote_id)))
        result = execute(state, jnp.int32(event.actor), act, jnp.int32(event.at_ms))
        state = result.state
        frame = frame_view(state)
        frame["event"] = event.model_dump()
        frame["accepted"] = bool(result.info.accepted)
        frame["traded"] = bool(result.info.traded)
        frame["reason"] = int(result.info.reason)
        frames.append(frame)
    return state, frames


class Recorder:
    """Collect replay inputs without exposing hidden state to live players."""

    def __init__(self, seed: int, config: Config, initial_cash=None):
        self.replay = Replay(
            seed=seed,
            config=asdict(config),
            initial_cash=(
                [config.initial_cash] * config.num_players
                if initial_cash is None
                else list(initial_cash)
            ),
            events=[],
        )

    def append(self, actor: int, act: Action, at_ms: int):
        """Append one scalar submitted action."""
        self.replay.events.append(
            ReplayEvent(
                actor=int(actor),
                at_ms=int(at_ms),
                **dict(zip(Action._fields, map(int, act), strict=True)),
            )
        )

    def save(self, path: Path):
        """Write a replay to an explicit path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.replay.model_dump_json(indent=2) + "\n")


def load_replay(path: Path) -> Replay:
    """Read bounded JSON data; pickle and executable formats are never accepted."""
    if path.stat().st_size > 8_000_000:
        raise ValueError("Replay exceeds 8 MB")
    return Replay.model_validate(json.loads(path.read_text()))
