# Game-logic audit — 2026-11-03

## Conclusion and scope

No game-logic defect was found in the inspected v0.13.0 engine or the executed
conformance cases. The core implements the published economic rules under the
documented exchange and timing assumptions. This is test evidence, not a formal
proof or a comparison with Jane Street's private server. No runtime behavior,
learning setting, reward calculation, or dependency was changed by this audit.

The audit reviewed `env.py`, `types.py`, `observations.py`, and `reference.py`.
The Python oracle uses scalar branches and lists; it does not call JAX transition
logic. The new payout expectations use Python `Fraction` arithmetic, independently
enumerated deck layouts, and explicit color partners.

## Rules checked against primary sources

The official [How to Play](https://www.figgie.com/how-to-play.html) was checked
again on 2026-11-03. The engine agrees on four/five seats, equal contributions to
a 200-chip pot, forty cards with 8/10/10/12 suit counts, the goal's color relation,
single-card transfers, quote clearing after every trade, and payout to goal-card
holders with the remaining pot split among leaders. The official
[FAQ](https://www.figgie.com/faqs.html) permits continued play with negative chips.

The sources still conflict on standard duration: How to Play says four minutes;
the FAQ says two. The default remains 240 seconds. Quote replacement, price caps,
credit enforcement, clock arbitration, and fractional-share precision remain
explicit simulation choices. See [rules and assumptions](rules.md).

## New automated checks

`tests/test_game_logic_audit.py` adds 14 parameterized test cases:

| Check | Coverage per backend | Required outcome |
| --- | --- | --- |
| Exhaustive settlement | 2,948 four-seat and 9,988 five-seat cases | Correct payout for every ordered allocation of 8 or 10 goal cards, across all 12 layouts |
| Deadline and absorbing state | All 12,936 settlement fixtures | Deadline action cannot trade; repeated calls preserve the entire state and return zero reward |
| Seeded deals | 4,096 per player count | Every layout appears; correct goal, equal starting hands, exact card totals, and correct ante |
| Adversarial traces | 128 tables, 256 submissions each | Every economic field, public event outcome, and terminal reward agrees with the Python oracle |
| Private observations | Every seat for both player counts | Other hands, deck, goal, scheduler clocks, and PRNG key do not affect actor observations, encodings, or action masks |

Adversarial configurations cover both player counts, both credit settings, and
deadline versus event-cap termination. They use a three-entry history ring,
17 ms jitter, distinct seat delays, fractional and negative initial bankrolls,
valid fills, invalid indices/kinds/prices, stale quote IDs, backward timestamps,
and calls after termination. Inputs are generated from the Python oracle's state,
not the JAX result. The JAX implementation executes the traces with `jit`, `vmap`,
and `lax.scan`. The fixed seed is 20261103; payout and privacy fixtures use 41 and 73.

Checks compare all economic state fields after every submitted event. They verify
nonnegative inventory, suit conservation, cash plus the unsettled pot, complete
public event information, and terminal profit exactly once. Event-cap truncations
must not distribute the pot or return a terminal-profit reward. The payout
fixtures isolate the distribution of the pot; the traces check ante-to-terminal
accounting. Float32 comparisons use explicit absolute tolerances: 1e-5 chips for
isolated payouts, 3e-5 for their sum, 1e-4 for trace cash/rewards, and 3e-4 for total
trace cash conservation. Discrete state and event outcomes must match exactly.

Existing tests additionally cover official payout examples, quote improvement and
cancellation, crossing quotes without automatic execution, self-trade rejection,
empty sellers, stale acceptance after replacement, chronological history wrap,
per-seat scheduling, PRNG preservation, deterministic replay, and private views.

## Results

These are historical checks from 2026-11-03, not a validation of every later
release. The targeted suite passed **103 tests on CPU** and **103 tests on CUDA**,
with no failures or skips. An isolated standalone dependency environment also
passed all **103 tests**. The same 14 new audit cases ran in each suite.
Raw logs from that audit are not distributed with this package. Run the current
checks below for evidence about the checkout in use.

## Reproduce with this package

The standalone test target runs the engine conformance tests:

```bash
docker build --target test -t figgie-engine:test .
docker run --rm --cpus 4 figgie-engine:test
```

To run only the original audit's engine test files:

```bash
docker run --rm --cpus 4 --entrypoint uv figgie-engine:test run --no-sync pytest \
  tests/test_env.py tests/test_parity.py tests/test_engine_optimization.py \
  tests/test_replay.py tests/test_game_logic_audit.py
```

The historical hardware was a Linux Ryzen 5 3600 / RTX 2070 SUPER host with
JAX/JAXlib 0.12.0.1 and Python 3.12.11. These are correctness checks, not speed
measurements. The commands above run on CPU. See [GPU setup](gpu-throughput.md)
for the separate CUDA prerequisites; CPU test results do not establish GPU
correctness for the current release.

## Limits

The exhaustive claim applies to settlement-relevant goal holdings, not all
possible action sequences or all floating-point bankrolls. Seeded deal checks
verify invariants and layout coverage; they are not a proof of statistical
uniformity. The oracle is independent code for the same written specification,
not an independent source of official rules. Observation checks detect direct
hidden-state exposure, not every possible information-theoretic side channel.
This audit covers the simulator. It does not validate a training application,
web service, or agent strength.
