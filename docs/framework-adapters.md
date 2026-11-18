# Gymnasium and PettingZoo adapters

This guide shows one complete round with each host adapter. The examples use a
one-second simulated round, a price cap of 8, a four-row trade history, seed 7,
and an event cap of 50. They select only actions enabled by each observation's
mask. The adapters keep each actor's hand private.

Install the adapter that you use, or install both:

```bash
uv sync --locked --extra gymnasium
uv sync --locked --extra pettingzoo
# Or install all framework adapters, including OpenSpiel:
uv sync --locked --extra all
```

The commands are alternatives. Run them from the repository root. Each code
example runs one round and does not reset the environment automatically.

## Gymnasium: one learner against scripted opponents

`FiggieGymEnv` controls one selected learner seat. The other seats use the
configured heuristic or random policy. Each `step` applies the learner action
and advances opponents until the learner acts again or the round ends.

The Gymnasium observation is a dictionary:

| Key | Shape and type | Meaning |
| --- | --- | --- |
| `observation` | `(D,)`, NumPy `float32` | Relative private actor features |
| `action_mask` | `(A,)`, NumPy `int8` | `1` for legal actions; terminal Gym observations enable only `WAIT` |

Here, `D = 21 + 10N + 6H`, where `N` is the player count and `H` is the trade
history size. `A = 1 + 8P + 16`, where `P` is `max_price`. See
[observations and actions](observation-actions.md) for the feature layout and
action ID mapping. A mask cannot see hidden counterparty inventory. The engine
checks submitted actions again.

`reset(seed=...)` returns `(observation, info)`. `step(action_id)` returns
`(observation, reward, terminated, truncated, info)`. The reward is the learner's
sum of sparse engine rewards since its prior Gym action. It includes a terminal
reward when settlement happens during an opponent's event. `terminated=True`
marks normal deadline settlement. `truncated=True` marks the event cap. They
cannot both be true. Terminal `info` includes `episode_reward`; info also
includes `learner_seat`, `simulation_time_ms`, and `event_count`.

The adapter settles when the next scheduled clock reaches the deadline. It does
not request a learner or opponent action at that time. If the deadline comes
before the learner's first opportunity, `reset` returns a handoff observation
with only `WAIT` enabled; the next `step` reports the terminal reward and sets
`round_ended_before_learner_turn` in `info`. All terminal observations use the
same `WAIT`-only mask so a masked policy can safely produce its final action.

```python
import numpy as np

from figgie_jax.types import Config
from figgie_jax_adapters.gymnasium import FiggieGymEnv

config = Config(
    num_players=4,
    duration_ms=1000,
    max_price=8,
    history_size=4,
    max_events=50,
)
env = FiggieGymEnv(config=config, learner_seat=0, opponent="heuristic")
rng = np.random.default_rng(7)
observation, info = env.reset(seed=7)
episode_reward = 0.0

while True:
    legal_ids = np.flatnonzero(observation["action_mask"])
    action_id = int(rng.choice(legal_ids))
    observation, reward, terminated, truncated, info = env.step(action_id)
    episode_reward += reward
    if terminated or truncated:
        break

print(
    {
        "terminated": terminated,
        "truncated": truncated,
        "episode_reward": episode_reward,
        "info": info,
    }
)
env.close()
```

Call `env.reset(seed=...)` yourself to start the next round. `step` raises after
the round ends until you reset. The seeded NumPy generator controls the example
learner's action choices. The Gymnasium seed controls the deal and scripted
opponent keys.

## PettingZoo: all seats controlled by agents

`FiggieAECEnv` exposes every seat as an AEC agent. The environment selects the
seat with the next independent decision clock. It does not add scripted
opponents. The loop below controls all seats with one masked random policy.

Each `env.observe(agent)` value has the same `observation` and `action_mask`
keys as the Gymnasium adapter. `env.last()` returns the current agent's
observation, cumulative reward since that agent last acted, termination flag,
truncation flag, and info. Add that reward each time `last()` selects the agent.
This accumulates sparse terminal profit for every seat, including settlement
that occurs on another seat's action. After an agent is terminated or truncated,
call `env.step(None)` to remove that dead agent, as required by PettingZoo's AEC
API.

The adapter settles immediately after the last event before the deadline. It
does not select an agent whose next clock is at or after the deadline. If the
deadline precedes the first clock, reset starts the AEC dead-agent removal flow
with each seat's terminal reward available through `last()`.

```python
import numpy as np

from figgie_jax.types import Config
from figgie_jax_adapters.pettingzoo import FiggieAECEnv

config = Config(
    num_players=4,
    duration_ms=1000,
    max_price=8,
    history_size=4,
    max_events=50,
)
env = FiggieAECEnv(config=config)
env.reset(seed=7)
rng = np.random.default_rng(7)
seat_reward = {agent: 0.0 for agent in env.possible_agents}

for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    seat_reward[agent] += reward
    if terminated or truncated:
        env.step(None)
        continue

    legal_ids = np.flatnonzero(observation["action_mask"])
    action_id = int(rng.choice(legal_ids))
    env.step(action_id)

print({"seat_reward": seat_reward})
env.close()
```

This loop demonstrates a multi-agent rollout. A learner-versus-scripted-opponent
setup uses Gymnasium with `learner_seat` and `opponent`. The AEC example assigns
one policy to each Figgie seat. Call `env.reset(seed=...)` after all agents have
been removed to start another round. The environment does not auto-reset.

## OpenSpiel

The optional OpenSpiel adapter uses explicit chance nodes for the initial deal
and scheduler ties. It does not represent an event-cap truncation as a normal
terminal round. Its game constructor rejects `max_events` values that could
truncate before the deadline. See [the OpenSpiel guide](openspiel.md) for its
action, observation, and support details.
