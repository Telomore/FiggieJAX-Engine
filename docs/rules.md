# Rule sources and fidelity specification

Source review date: **2026-11-03**. Primary sources are Jane Street's
[How to Play](https://www.figgie.com/how-to-play.html) and
[FAQ](https://www.figgie.com/faqs.html). Architectural discussions are not
rules authorities. No third-party
engine was copied or treated as canonical.

The [game-logic audit](game-logic-audit.md) records checks performed on the
engine before this standalone release. It is historical evidence of
those checks. It does not establish byte-for-byte compatibility with Jane
Street's private server or online matchmaking service. This standalone package
contains the simulator, adapters, examples, and offline replay tools. It does
not contain a web room or PPO training system.

## Published rules and conformance

| Rule | Implementation | Validation |
| --- | --- | --- |
| Four or five seats, including bots | `Config.num_players` | Both counts are covered by engine checks |
| Initial bankroll: 350 chips per player | `reset` | Ante and carry-forward checks |
| Equal contributions to a 200-chip pot | 50 each for four; 40 each for five | Pot conservation before and after settlement |
| Forty cards with counts 8, 10, 10, 12 | Twelve distinct assignments, sampled uniformly | Assignment enumeration and batched deal checks |
| Random equal deal of every card | Uniform permutation, 10 or 8 cards per seat | Suit and player totals |
| Hidden goal is the other suit of the 12-card suit's color | Spades/clubs; hearts/diamonds | Every layout checked |
| Goal has 8 or 10 cards | Implied by the layout | Four layouts have 8 goal cards; eight have 10 |
| Players can act without taking turns | Independent event clocks | Heterogeneous and tied clock checks |
| A trade transfers exactly one card | Explicit buy/sell acceptance | Cash and inventory transfer checks |
| Every trade cancels every quote in all four suits | Global quote and identifier reset | Multi-suit regression fixture |
| Each goal card pays 10 chips | Terminal settlement | Official payout examples |
| Largest goal holding receives the pot remainder; ties split equally | All tied seats receive a share | Two-, three-, and four-way ties |
| Subsequent rounds retain cumulative bankroll | Optional `cash` input to `reset` | Negative and carried bankroll checks |
| Other hands and the goal remain private during play | `observe` and `relative_features` | Observation privacy checks |
| Replays can reconstruct the board after each event | Re-executed JSON events and timeline | Replay reconstruction checks |

The independent Python oracle uses lists and ordinary branches. Randomized traces
compare economic state and event outcomes with JAX after every event. The oracle
is a second implementation of this written specification. It is not independent
validation against the official server.

## Official timing conflict

The How to Play page says that standard trading lasts four minutes. The FAQ says
two minutes. Both pages were available on the source review date. This engine
defaults to **240 seconds**, following How to Play. `Config.duration_ms` uses
milliseconds. The native rollout example accepts `--duration-ms` and records the
value in its JSON summary.

No action executes at or after the deadline. Such an event settles the pot first.
Earlier timestamps are rejected. A completed state is absorbing. Settlement
cannot pay the pot twice.

## Explicit simulation choices

The published rules do not specify all electronic exchange details. These
choices are documented assumptions, not verified official behavior:

- Each suit retains one best bid and one best offer, with an owner and quote ID.
  Another player must improve the displayed price. Equal-price replacements from
  another player are rejected. A player can replace or cancel their own quote.
  Displaced quotes do not remain in a hidden order book.
- A crossing bid and offer do not trade automatically. A player must use `BUY`
  or `SELL`. Acceptance uses the resting quote price and its exact identifier.
- Quoting an offer requires owning a card. Sales cannot create negative inventory.
- Quotes use whole chips from 1 to `max_price`, which defaults to 200. This cap is
  an action-space constraint, not a documented official price bound. Train and
  evaluate with the same cap.
- Negative cash is permitted by default. The FAQ permits negative-chip players
  to continue rounds, but it does not specify intraround borrowing rules.
  `allow_negative_cash=False` is an alternate simulator setting. It validates
  cash at bid placement and again at execution. It does not reserve cash for a
  quote.
- Tied pot shares use floating-point chips without rounding to whole chips or
  cents. Three-way ties can therefore have repeating fractional shares. Tests
  use floating-point tolerances and check that the 200-chip pot is conserved.
- Public cash balances and quote owners are observable. Other inventories, deck
  counts, goal suit, PRNG key, and opportunity clocks are private to the actor.
  The default observation uses the actor's hand and a finite public trade history.
- The finite history is an observation design, not a restriction of the game's
  underlying information. Replay retains every submitted event. An agent can
  keep additional memory outside the engine.

## Event time and optional latency

There is no shared action tick. Each seat has its own next opportunity time. The
simulator selects the smallest absolute time and resolves exact ties with seeded
random priority. A later action at the same timestamp sees the result of the
earlier action. This models arbitration of simultaneous arrivals; it is not a
claim about the official server's scheduler.

Each delay is the sum of a positive base interval, a seat-specific added latency,
and optional uniform integer jitter. The default base interval is 250 ms. Added
latency and jitter default to zero. There are no wall-clock sleeps in the
simulator. Zero added latency still advances simulated time.

**Latency means decision-opportunity delay.** The policy reads the current
market at its opportunity. This is not a queued, precomputed action and is not a
model of observation delay, packet transport, compute time, or network jitter.
Do not interpret latency settings as measurements of a real exchange network.
Quote identifiers protect delayed external acceptances against stale quotes and
global resets caused by another trade.

## Rewards and limits

Rewards are zero during trading. On a true terminal event, each seat receives its
final bankroll minus its pre-ante bankroll. Trading payments, antes, and the pot
are included exactly once. Total round profit is zero up to floating-point error.

`max_events` is a safety limit. Reaching it marks a **truncation** without
settling the pot. A truncated trajectory is not a completed Figgie result. The
native rollout reports truncation without a terminal-profit vector. The learning
example skips a truncated round because it has no terminal reward. A rollout
segment does not end a game; carry the live state into the next segment.

The standalone release does not provide accounts, official ratings, matchmaking,
official bots, a live service, or official-server interoperability. Its replay
support is an offline analysis tool and can reveal every hand.
