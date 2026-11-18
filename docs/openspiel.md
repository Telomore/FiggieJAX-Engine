# OpenSpiel adapter

The optional adapter registers Figgie as the OpenSpiel game
`python_figgie`. It uses the engine's transition rules and the shared finite
action encoding. It supports four and five players.

Install the optional dependency with `uv sync --locked --extra openspiel`. Import the
adapter before loading the game so that PySpiel can find the Python game class.

```python
import numpy as np
import pyspiel

import figgie_jax_adapters.openspiel  # Registers python_figgie.

game = pyspiel.load_game("python_figgie(players=4,duration_ms=12000,max_price=200)")
state = game.new_initial_state()
rng = np.random.default_rng(7)

while not state.is_terminal():
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probabilities = zip(*outcomes, strict=True)
        action = rng.choice(actions, p=probabilities)
    else:
        action = rng.choice(state.legal_actions())
    state.apply_action(int(action))

print(state.returns())
```

## Game parameters

The `python_figgie(...)` game string accepts these parameters:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `players` | 4 | Number of seats. Use 4 or 5. |
| `duration_ms` | 240000 | Round duration in milliseconds. |
| `max_price` | 200 | Largest integer quote price. |
| `decision_interval_ms` | 250 | Base time between opportunities for one seat. |
| `initial_cash` | 350 | Starting cash per seat before the ante. |
| `max_events` | 50000 | Maximum player actions. The value must exceed all zero-latency opportunities before the deadline. |
| `history_size` | 16 | Number of recent trades in each actor observation. |
| `allow_negative_cash` | true | Whether a seat may spend more cash than it has. |

The adapter sets added latency and jitter to zero. A round begins with one
uniform chance action over the twelve deck layouts. It then deals forty cards
without replacement. Each card goes to the next seat in order. This produces
the same uniform random assignment of suits to seat hands as shuffling the
multiset and splitting it into equal hands.

The adapter represents an equal-clock tie with a chance node. Each seat at the
earliest clock has equal probability. A single earliest seat acts directly.
This models the tie distribution from independent random priorities. It does
not reproduce the engine's seeded PRNG realization. The wrapper uses
`apply_event` at the selected seat's scheduled time. It does not use the
engine's `next_actor` to override the OpenSpiel chance outcome.

At the deadline, the wrapper settles once before any action at or after the
deadline. The wrapper rejects any `max_events` value that could truncate the
round before its deadline. If the core engine still reports a truncation, the
wrapper raises an error. It does not present an incomplete round as a terminal
result.

## Actions and observations

The finite action count is `1 + 8 * max_price + 16`. Action IDs use the same
encoding as `figgie_jax_adapters.common`:

- `0`: WAIT.
- `1 + suit * P + (price - 1)`: BID at `price`.
- `1 + 4 * P + suit * P + (price - 1)`: ASK at `price`.
- `1 + 8 * P + suit`: BUY the displayed offer.
- `1 + 8 * P + 4 + suit`: SELL to the displayed bid.
- `1 + 8 * P + 8 + suit`: CANCEL_BID.
- `1 + 8 * P + 12 + suit`: CANCEL_ASK.

Here, `P` is `max_price` and suit IDs are spades 0, clubs 1, hearts 2, and
diamonds 3. BUY and SELL bind to the quote identifier visible to the acting
seat. The engine checks that identifier again when it processes the action.

OpenSpiel's observation tensor and string contain the current actor's private
hand, public cash, quotes, retained trade history, and remaining time at that
actor's scheduled opportunity. Player references use seat-relative features.
The observation does not expose other hands, the deck, the goal, or scheduler
state. Deal-stage observations are all zeros until the deal is complete.

Action legality comes from the actor's observation. The mask does not inspect
hidden counterparty inventory. A BUY can therefore be legal from the visible
offer and still be rejected by the exchange if the seller no longer has that
card. Such a rejected attempt consumes one event, as it does in the engine.

The game type declares imperfect information and does not provide information
state strings or tensors. The observer rejects perfect-recall requests. Do not
use this adapter as a perfect-recall or CFR-ready game. The initial chance tree
has forty sequential card-deal steps, and the event tree is large. This adapter
is for integration and sampled interaction. It makes no claim that exhaustive
tree solvers are practical.

OpenSpiel's Python game guide describes the Python game API and its chance and
observer conventions: [Python games](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/games/README.md).
The adapter follows its explicit chance-node pattern for imperfect-information
games, as shown in [Kuhn Poker](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/games/kuhn_poker.py).
OpenSpiel's core API defines explicit chance as enumerated outcomes with
deterministic application; see [`spiel.h`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h).
