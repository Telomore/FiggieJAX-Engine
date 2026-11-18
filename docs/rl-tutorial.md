# A small reinforcement-learning example

This tutorial explains `examples/train_reinforce.py`. The script uses the native
JAX engine and framework-independent helpers. It does not need Optax or an
optional Gymnasium, PettingZoo, or OpenSpiel installation. Its round
collection is a Python host loop; JAX executes the policy math, gradients, and
compiled event calls. For a fully compiled simulation collector and a custom
training-loop workflow, see [the README](../README.md#run-the-simulation-loop-fully-in-jax).
For the separate example that also compiles collection and repeated parameter
updates, see [compiled learning](compiled-learning.md).

Run a short seeded example:

```bash
uv run python examples/train_reinforce.py \
  --episodes 8 --eval-episodes 4 --duration-ms 10000 --seed 7 \
  --output results/reinforce-example.json
```

`--episodes`, `--eval-episodes`, `--duration-ms`, and `--seed` control the
training count, evaluation count, simulated round length, and random seed. The
script also records the player count, event cap, price cap, action count, feature
count, learning rate, entropy coefficient, device, update count, gradient norm,
and parameter change. Set `--max-events` to change the safety cap. If a round is
truncated, the script skips that round because it has no terminal reward.

## Policy and action mask

The learner sits in seat zero. It sees `observation_array(obs, config)`, which is
a relative encoding of its own hand and public market information. It never
receives the privileged `State`, the hidden goal, or other hands. The opponents
use the engine's fixed heuristic policy.

The policy is one linear categorical layer. If $x_t$ is the row feature vector at
decision $t$, and $W$ is the weight matrix, the unmasked action logits are:

$$
z_t = x_t W.
$$

Here, $x_t \in \mathbb{R}^{d}$, where $d$ is the observation feature
count, a positive integer. Let $K$ be the positive integer action count.
The logit row vector $z_t \in \mathbb{R}^{K}$ contains one unnormalized score
per action. Then $W \in \mathbb{R}^{d \times K}$. Invalid actions receive a large negative logit
before softmax. The resulting categorical distribution samples only actions
enabled by `legal_action_mask`.

At the default price cap, $P=200$, so $K=1+8P+16=1617$. With the default
four-player observations, this is a small single-layer policy. Its output size
grows with the configured price cap. This example uses no hidden layers, target
network, replay buffer, or optimizer package.

## Reward and update

The engine gives zero reward before settlement. At the deadline, every seat gets
its final cash minus its pre-ante cash. This includes the ante, trading transfers,
and pot share exactly once. The terminal event can happen when another player
acts. Therefore the script adds the learner's reward component after every
transition, including opponent transitions. It does not wait for a reward on the
learner's own action.

For a completed round, let $G$ be the learner's terminal chip profit, and let
$b$ be a running mean baseline. The normalized advantage is:

$$
A = \max\left(-5, \min\left(5, \frac{G-b}{S}\right)\right).
$$

Here, $G$ and $b$ are in chips, and the reward scale is $S=100$ chips.
Thus, $A$ and the clipping bounds are dimensionless. The clip operation clamps
the scaled return to the interval $[-5,5]$. These are training-example settings. They keep large single-round returns from producing
large updates. The baseline starts at $b_0=0$. After each completed training
game, it changes to $b \leftarrow 0.9b + 0.1G$. For the sampled action ID $a_t \in \{0,\ldots,K-1\}$ at learner decision $t$,
the batch policy-gradient loss is:

$$
L = -\frac{1}{M}\sum_{t=1}^{M} A_t \log \pi_W(a_t \mid x_t) -
\beta \frac{1}{M}\sum_{t=1}^{M} H\!\left(\pi_W(\cdot \mid x_t)\right).
$$

Here, $L$ is the dimensionless scalar loss. The positive integer $M$ is
the total number of recorded learner decisions from completed
training rounds. $A_t$ is the normalized episode advantage repeated for each
decision in that round. $\pi_W$ is the masked categorical policy, $H$ is
its categorical entropy computed with natural logarithms, and $\beta \geq 0$
is the dimensionless entropy coefficient. The example
collects these decisions with one fixed policy, differentiates the mean loss
with JAX, clips the gradient's global norm to 1, and applies one batch
stochastic-gradient update. The moving baseline changes during collection,
after each completed training round; the parameter update follows collection
of the full batch. If there are no recorded learner decisions, the script skips
the parameter update. Truncated
rounds do not create an invented reward target.

The update is real: weights change through a reward-weighted log-probability
gradient, with an entropy bonus. The script reports parameter change and gradient
norm. These implementation checks show that the code updated parameters. They
do not show that the policy is stronger.

## Evaluation and interpretation

The initial and final policies use the same list of evaluation keys. Those keys
come from a PRNG domain separate from the training keys. Each evaluation plays
fresh simulated games against the same fixed heuristic opponents. The JSON
output reports completed-game count, truncations, mean terminal profit,
population standard deviation, and win fraction. These are simulated outcomes
from a small evaluation sample. They are not empirical observations or evidence
of strategic improvement.

Keep the run reproducible by preserving the output JSON with the script version,
package version, seed, duration, player count, event cap, maximum quote price,
and dependency lockfile. For larger research experiments, use a separately
designed and validated training method. This example is for learning the engine
and adapter interfaces.
