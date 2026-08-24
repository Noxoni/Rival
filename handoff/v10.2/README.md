# Rival v10.2 — Ball Acquisition

Milestone 10.2 is the next prerequisite-learning stage after the v10.1 agency bootstrap.

## Decision

v10.1 proved that the scratch actor can learn a simple primitive from dense reinforcement: it learned to move quickly. It did **not** learn reliable ball acquisition. v10.2 therefore removes every direct incentive for speed and teaches exactly one next skill:

> **Reduce car-to-ball separation and make real physical contact with the ball.**

Nothing else is a training objective in this milestone.

## Frozen architecture

Do not change:

- `RivalPolicyV1`;
- `RivalObsV1` (714 floats);
- `RivalActionV1` (five continuous controller axes + joint jump/boost/handbrake categorical);
- native 120-Hz decision cadence;
- one-tick RocketSim/RLBot action-delay contract;
- canonical observation adapters;
- actor topology/export path.

v10.2 is a learning-distribution change, not an architecture experiment.

## Starting actor

Resume the **actor weights** from the final recoverable v10.1 +10h checkpoint:

`training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`

Expected actor SHA-256:

`e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`

Preserve that source checkpoint byte-for-byte.

Because the reward contract changes, initialize a fresh critic and fresh actor/critic optimizer state. Do not carry the v10.1 critic or Adam moments into v10.2. The actor weights are the retained skill prior.

## Reward authority

The v10.2 reward has only two positive learning signals:

1. **car-caused progress toward the ball** — small, signed, dense;
2. **a real new physical ball touch** — the maximum per-event reward.

There is:

- no speed reward;
- no boost reward;
- no jump/dodge reward;
- no aerial reward;
- no recovery reward;
- no ball-to-goal reward;
- no goal reward;
- no concede penalty;
- no dribble or possession reward;
- no opponent-relative tactical reward.

Goals terminate/reset the episode but have reward `0.0`. Scoring is deliberately locked until later prerequisite stages.

See `BALL_ACQUISITION_REWARD.md`.

## Curriculum authority

The initial curriculum isolates ball acquisition. The learner sees broad randomized ground acquisition tasks with increasing geometry/velocity difficulty plus a small natural-kickoff holdout.

The second car remains present so `RivalObsV1` is unchanged, but it must be non-interfering and receive no learning update in this stage. Randomize which team is the active learner.

See `BALL_ACQUISITION_CURRICULUM.md`.

## Exit criterion

v10.2 ends as soon as Rival has demonstrated **reliable first-touch acquisition**, not when a fixed step budget is exhausted. Passing requires two consecutive deterministic evaluation boundaries with high touch success across every acquisition family.

Only after this gate passes may the project move to the next prerequisite: **ground ball control / dribbling**.

See `EVALUATION_AND_EXIT_GATES.md` and `SKILL_PROGRESSION.md`.

## Production

No production promotion is authorized. Frozen Wisp remains production until a future milestone explicitly changes that decision.
