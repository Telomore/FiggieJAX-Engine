"""Check that the learning tutorial uses the reward signal and action mask."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/train_reinforce.py"))


@pytest.mark.parametrize("advantage", [-1.0, 1.0])
def test_reward_signal_moves_selected_action_probability(advantage):
    """Positive advantage increases the selected action's probability; negative lowers it."""
    weights = jnp.zeros((3, 3), jnp.float32)
    features = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    selected = jnp.zeros((2,), jnp.int32)
    masks = jnp.array([[True, True, False], [True, True, False]])
    updated, norm = EXAMPLE["_gradient_update"](
        weights, features, selected, masks, jnp.full((2,), advantage), 0.1, 0.0
    )
    probabilities = jax.nn.softmax(jnp.where(masks[0], features[0] @ updated, -1e9))
    assert np.isfinite(norm) and norm > 0
    assert (float(probabilities[0]) - 0.5) * advantage > 0
    assert float(probabilities[2]) == 0.0
    np.testing.assert_array_equal(updated[:, 2], weights[:, 2])


def test_no_decisions_produces_no_parameter_update():
    """A round without learner decisions must not produce a gradient update."""
    weights = jnp.zeros((3, 3), jnp.float32)
    updated, norm = EXAMPLE["_gradient_update"](
        weights,
        jnp.zeros((0, 3)),
        jnp.zeros((0,), jnp.int32),
        jnp.zeros((0, 3), bool),
        jnp.zeros((0,)),
        0.1,
        0.01,
    )
    np.testing.assert_array_equal(updated, weights)
    assert norm == 0.0
