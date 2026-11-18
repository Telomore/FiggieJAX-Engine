# Observations and actions

This guide describes the information that a decentralized actor can use and the
finite action IDs used by the optional framework adapters. The native engine
continues to accept structured `Action` values. The adapter helpers live in
`figgie_jax_adapters.common` and need only the core JAX and NumPy dependencies.

## Private actor observations

Call `observe(state, player, config)` to create the view for one seat. The
`Observation` contains:

- The acting player's hand, with one count for each suit.
- Public cash balances for every seat.
- The player's seat number.
- The best displayed bid and ask for each suit.
- The quote owner and quote identifier for each displayed quote.
- A bounded public trade history and a mask for valid history rows.
- Time remaining in the round.

It does not contain other hands, the goal suit, the deck layout, the random key,
or the other seats' clock times. Do not pass `State` to a decentralized policy.
`State` is a privileged simulator value.

`features(obs, config)` produces a fixed-length vector with absolute seat
references. `relative_features(obs, config)` rotates cash balances, quote owners,
and trade participants around the acting seat. The acting seat then has relative
index zero. This keeps public references consistent while preserving suit names
and the private hand. The policy example uses this relative vector.

`decision_observation(state, player, config)` is a helper for host-side agents.
When `player` is the next actor, it advances the observation time to that seat's
decision opportunity. For other seats, it uses the current simulation time. The
helper does not expose another seat's private hand.

### Raw `Observation` fields

Let `N = config.num_players` and `H = config.history_size`. The native
`Observation` fields have these shapes, types, and units:

| Field | Shape | Type | Meaning and units |
| --- | --- | --- | --- |
| `hand` | `(4,)` | `int32` | Private card counts in suit order |
| `cash` | `(N,)` | `float32` | Public cash for each absolute seat, in chips |
| `player` | scalar | `int32` | Acting seat index from 0 through `N - 1` |
| `prices` | `(2, 4)` | `int32` | Best bid row 0 and best ask row 1, in chips; 0 means no quote |
| `owners` | `(2, 4)` | `int32` | Quote owner seat index; -1 means no quote |
| `quote_ids` | `(2, 4)` | `int32` | Positive quote IDs; 0 means no quote |
| `history` | `(H, 5)` | `int32` | Rows are `[time_ms, buyer, seller, suit, price]`; padded rows use -1 |
| `history_valid` | `(H,)` | `bool` | True for a retained trade row |
| `remaining_ms` | scalar | `int32` | Time remaining in the round, in milliseconds |

The native `relative_features` output and adapter `observation_array` output are
one-dimensional `float32` vectors. Their length is:

$$
D = 4 + N + N + 8 + 8(N+1) + 5H + H + 1 = 21 + 10N + 6H.
$$

Here, `D` is the feature count; `N` and `H` are defined above. With four seats
and the default history size of 16, `D = 157`. The feature fields appear in this
order: hand counts divided by 12; cash rotated around the acting seat and divided
by 350; a relative seat one-hot vector; prices divided by `max_price`; relative
quote-owner one-hot vectors, including an empty-owner category; scaled trade
history; history-valid values; and remaining time divided by `duration_ms`.
History time uses `duration_ms` as its scale. Buyer and seller indices use `N`,
suit uses 4, and price uses `max_price`. Invalid history rows become zero in the
feature vector. The relative seat one-hot vector always identifies the actor as
seat zero.

## Structured engine actions

The native engine action has four fields: `kind`, `suit`, `price`, and
`quote_id`. The action kinds are:

| Kind | Meaning | Required fields |
| --- | --- | --- |
| `WAIT` | Consume an opportunity without a quote change | None |
| `BID` | Place or replace a bid | Suit and whole-chip price |
| `ASK` | Place or replace an ask | Suit and whole-chip price |
| `BUY` | Accept the displayed ask | Suit and current ask quote ID |
| `SELL` | Accept the displayed bid | Suit and current bid quote ID |
| `CANCEL_BID` | Remove the player's displayed bid | Suit |
| `CANCEL_ASK` | Remove the player's displayed ask | Suit |

`BUY`, `SELL`, and both cancel actions do not need a price. A trade transfers one
card. It clears every quote in every suit. The exchange checks the quote ID again
when it processes a `BUY` or `SELL` action. This rejects delayed acceptances after
the quote changed or after another trade cleared the market.

The engine exposes `action_masks(obs, config)` for a factorized kind and suit
mask. `price_mask(obs, kind, suit, config)` gives the observable legal prices for
one quote action. The exchange remains the final authority. An observation
cannot reveal whether the other side owns a card to sell.

## Finite action IDs for adapters

`figgie_jax_adapters.common.action_count(config)` returns the action-space size
for `P = config.max_price`:

$$
N_{\mathrm{actions}} = 1 + 8P + 16.
$$

Here, `P` is the maximum whole-chip quote price. The first ID is `WAIT`. The
remaining IDs use this layout:

| ID range | Action |
| --- | --- |
| `0` | `WAIT` |
| `1` through `4P` | `BID`, four suits with prices 1 through `P` |
| `1 + 4P` through `8P` | `ASK`, four suits with prices 1 through `P` |
| `1 + 8P` through `4 + 8P` | `BUY`, one per suit |
| `5 + 8P` through `8 + 8P` | `SELL`, one per suit |
| `9 + 8P` through `12 + 8P` | `CANCEL_BID`, one per suit |
| `13 + 8P` through `16 + 8P` | `CANCEL_ASK`, one per suit |

For a bid, the suit-major price index is:

$$
\mathrm{id}_{\mathrm{bid}} = 1 + sP + (p - 1),
$$

where `s` is the suit index from 0 to 3, and `p` is an integer price from 1 to
`P`. For an ask, add `4P` to this ID. These ranges have no gaps or overlaps.

`legal_action_mask(obs, config)` returns a Boolean vector in this ID order. It
always enables `WAIT`. Bid and ask entries observe price-improvement rules,
inventory for asks, and the configured cash rule. Buy and sell entries require a
visible quote and use observable cash and hand information. Cancel entries
require the acting player to own the quote. A mask cannot check hidden inventory
on the other side of a trade, and the engine checks every submitted event.

`decode_action(action_id, obs, config)` converts one ID to a native `Action`.
For a buy or sell, it reads the quote ID shown in `obs`. The engine still rejects
the event if that quote is stale by the time it executes. It raises a `ValueError`
for an ID outside the configured action space.

`observation_array(obs, config)` returns the relative actor feature vector as a
NumPy `float32` array. `decision_observation(state, player, config)` creates an
observation at the correct decision time for the current actor. These host-side
helpers use NumPy conversion. Do not call them from inside a JIT-compiled engine
transition.

## Adapter limits

The action space is finite because prices are capped by `max_price`. With the
default cap of 200, it has 1,617 entries. A learner can use a mask to exclude
unavailable actions. Gymnasium and PettingZoo wrap one or more rounds and retain
the same action IDs. OpenSpiel maps the event game to its API with explicit
chance and scheduling behavior. See [the OpenSpiel adapter guide](openspiel.md)
for its support limits.
