"""Offline replay conformance without web or training dependencies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from figgie_jax.env import apply_event, reset
from figgie_jax.replay import Recorder, Replay, board, reconstruct, snapshot
from figgie_jax.types import Config, Kind, action


def test_replay_reconstructs_actions_and_payout():
    config = Config(duration_ms=1000, history_size=3)
    state = reset(jax.random.key(7), config)
    recorder = Recorder(7, config)
    suit = int(jnp.argmax(state.hands[1]))
    execute = jax.jit(lambda s, p, a, t: apply_event(s, p, a, t, config))
    for player, act, at in [
        (1, action(Kind.ASK, suit, 10), 0),
        (0, action(Kind.BUY, suit, 0, 1), 100),
        (0, action(), 1000),
    ]:
        recorder.append(player, act, at)
        state = execute(state, player, act, at).state
    replay = Replay.model_validate_json(recorder.replay.model_dump_json())
    replayed, frames = reconstruct(replay)
    assert len(frames) == 4 and frames[-1]["done"]
    for field, expected in snapshot(state).items():
        np.testing.assert_allclose(snapshot(replayed)[field], expected, atol=1e-5)
    assert frames[2]["traded"]


def test_live_board_hides_omniscient_data():
    config = Config()
    state = reset(jax.random.key(0), config)
    private = board(state, config)
    assert private["hands"] is None and private["goal"] is None and private["deck"] is None
    assert board(state, config, reveal=True)["hands"] is not None


def test_replay_rejects_bad_clock_actor_and_oversized_settings():
    recorder = Recorder(0, Config())
    recorder.append(0, action(), 10)
    recorder.append(0, action(), 9)
    with pytest.raises(ValueError, match="nondecreasing"):
        reconstruct(recorder.replay)
    recorder.replay.events = []
    recorder.replay.config["history_size"] = 1000
    with pytest.raises(ValueError, match="resource"):
        reconstruct(recorder.replay)


def test_legacy_replay_without_player_types_remains_supported():
    replay = Recorder(0, Config(duration_ms=1000)).replay.model_dump()
    del replay["agents"]
    restored = Replay.model_validate(replay)
    assert reconstruct(restored)[1][0]["agents"] == []
