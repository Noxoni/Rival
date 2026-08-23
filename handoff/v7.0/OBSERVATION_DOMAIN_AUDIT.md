# Observation / Policy / Action / Transition Domain Audit

This audit follows the useful RLGym decomposition:

`state s -> observation O(s) -> policy pi(o) -> action function I -> action a -> transition T(s'|s,a)`

The M06 transfer failure should be localized to one or more of those boundaries rather than labeled generically as a sim-to-RLBot gap.

## A. Observation function O(s)

Compare the training `WispCompatible432RLGymV1` representation against the live RLBot Wisp-compatible observation path feature-group by feature-group.

The training documentation already admits several approximations:

- RocketSim BallPredictor vs RLBot ball-prediction slices;
- bounded box-surface landing normal vs live arena-SDF/normal behavior;
- bounded kinematic ETA vs live Wisp cached ETA;
- episodic score-difference handling;
- training previous-action plumbing.

Do not assume these are harmless because the tensor is still length 432.

### Required audit

Collect a representative corpus of live RLBot states/packets from frozen-Wisp natural matches without changing gameplay. For every state where practical:

1. compute the exact live 432 observation used by production Rival;
2. convert the same physical state into the closest supported RLGym/RocketSim representation and compute the training 432 observation;
3. compare by named feature ranges, not only whole-vector RMSE.

Record per feature group:

- mean absolute error;
- max absolute error;
- percentile error;
- sign disagreement where meaningful;
- distribution mean/std/min/max;
- correlation;
- frequency of out-of-range values;
- effect on frozen-Wisp first-90 logits/top-1 action when swapping the feature source.

At minimum isolate:

- ball position/velocity/angular velocity;
- goal-relative vectors;
- kickoff flag;
- four ball-prediction horizons;
- boost-pad availability/timers;
- close-pad relative positions;
- previous action;
- wall distances / landing normal;
- score differential;
- self car block;
- opponent car block;
- ETA-related fields;
- touch/jump/dodge/handbrake state.

### Ablation test

Where a training approximation differs materially, substitute the live-derived feature group into an otherwise training-style observation and rerun frozen-Wisp inference. Rank feature groups by how much they change:

- first-90 logits;
- top-1 action agreement;
- confidence/margin.

This identifies which observation approximations are actually policy-sensitive.

## B. Policy pi(o)

On identical live observation tensors, compare:

- original frozen TorchScript Wisp;
- zero-step trainable reconstruction;
- 20M trained actor.

The zero-step reconstruction must preserve first-90 logits/top-1 on live observations. If it does not, fix the model/export path before any further conclusion.

For the trained actor, quantify legacy-policy drift:

- top-1 agreement with Wisp;
- JS/KL divergence where stable;
- confidence/margin shift;
- disagreement by state/feature regime;
- action transition matrix.

Because M06 selected zero appended actions deterministically in RLBot, focus first on drift within legacy actions 0-89.

## C. Action function I

Prove that the same selected legacy index yields the same controller semantics across:

- production Wisp action parser;
- reconstructed/student deployment parser;
- RLGym training parser.

Test both blue/orange and X-mirroring regimes.

Then isolate action repeat / action delay:

- tick 8 / delay 7;
- tick 4 / delay 3.

Do not conflate a policy change with a different temporal action function.

## D. Transition function T(s'|s,a)

After O, pi, and I have been checked, measure short-horizon physical divergence between RocketSim and RLBot for identical observable initial states and fixed controller sequences.

Use broad natural-state samples rather than hand-authored trick scenarios.

For short horizons such as 4, 8, 16, 32, and 64 physics ticks, compare:

- car position/velocity/angular velocity/orientation;
- ball position/velocity/angular velocity;
- jump/dodge/flip availability and state;
- boost amount;
- contact/touch occurrence;
- ground/wall state where observable.

This is not a requirement for bit-identical physics. The purpose is to find whether divergence is large enough, early enough, and systematic enough to invalidate the training transfer assumptions.

## Decision output

Rank the transfer contributors with evidence:

1. observation-function mismatch;
2. policy drift;
3. temporal/action-function mismatch;
4. transition/physics mismatch;
5. interactions.

Do not resume serious training until the dominant contributor(s) have a concrete correction plan and a zero-step/frozen-Wisp RLBot control demonstrates that the corrected transfer seam preserves baseline gameplay reasonably well.
