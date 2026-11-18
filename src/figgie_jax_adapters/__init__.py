"""Optional RL adapters for the standalone FiggieJAX engine.

Import a framework-specific adapter from its own module. The common module uses
only the engine's core dependencies and does not import Gymnasium or PettingZoo.
"""

from figgie_jax_adapters.common import (
    action_count,
    decision_observation,
    decode_action,
    legal_action_mask,
    observation_array,
)

__all__ = [
    "action_count",
    "decode_action",
    "decision_observation",
    "legal_action_mask",
    "observation_array",
]
