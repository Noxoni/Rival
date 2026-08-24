# Rival v10.2 — Implementation Gates

These gates are intentionally narrow. The v9 architecture is already proven; v10.2 only needs to prove that the new prerequisite-learning distribution is implemented exactly before spending campaign experience.

## Gate 0 — Preserve authority and source

Before implementation/training:

- verify local/remote `main` includes Milestone 10.1 closeout;
- verify source actor checkpoint exists at:
  `training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`;
- verify source actor SHA-256:
  `e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`;
- preserve source checkpoint byte-for-byte;
- preserve frozen Wisp production hashes/configuration;
- confirm no stale M10/v10.1 training process is running before starting v10.2.

## Gate 1 — Actor-only skill transfer

Prove:

- `RivalPolicyV1` actor parameters load exactly from v10.1 +10;
- held-observation deterministic actor outputs match source exactly before any v10.2 update;
- critic is freshly initialized;
- actor optimizer has fresh empty Adam state before first update;
- critic optimizer has fresh empty Adam state;
- no v10.1 optimizer moment or critic parameter is accidentally resumed.

Record source and initialized hashes/state counts.

## Gate 2 — Reward truth table

Unit-test `RivalBallAcquisitionRewardV1` against explicit transitions:

- car moves toward stationary ball -> positive distance reward;
- car moves away -> negative;
- car stationary while ball moves toward car -> approximately zero car-caused distance reward;
- car stationary while ball moves away -> approximately zero car-caused distance reward;
- approach/retreat cycle does not create free positive dense return;
- dense absolute episode spend never exceeds 0.75;
- one true new touch -> exactly +1.0 touch event;
- sustained contact over many 120-Hz ticks -> one touch event;
- two genuinely separated contacts -> two +1.0 events;
- three separated contacts -> three +1.0 events;
- aerial versus ground touch has identical reward value;
- goal for -> zero outcome reward;
- goal against -> zero outcome reward;
- speed/jump/dodge/boost/controller activity alone -> zero reward.

Run cadence aggregation checks to ensure diagnostic stepping does not duplicate touch events.

## Gate 3 — Touch detector validation

The touch detector is critical. Verify on RocketSim traces, not only mocked state:

- one collision/contact is counted once;
- continuous contact is not repeated at 120 Hz;
- separation followed by contact is counted again;
- rapid but genuinely separate touches are not suppressed by an arbitrary long debounce;
- opponent/dummy contact never credits the active learner;
- reset clears all touch state;
- goal reset cannot emit a phantom touch.

Publish a compact trace with tick, contact semantics, detector state, and emitted event.

## Gate 4 — Active learner / dummy isolation

For at least 10,000 environment steps across many resets, prove:

- exactly one active learner per episode;
- active team is approximately balanced blue/orange;
- dummy controller is exactly all-zero;
- dummy does not contribute PPO observations/actions/logprobs/advantages/returns/loss rows;
- learner experience does contribute normally;
- no gradient can be attributed to dummy transitions;
- opponent portion of `RivalObsV1` remains finite/legal;
- dummy placement does not physically interfere before ordinary learner acquisition in >99.9% of sampled starts; any interference cases are identified and fixed before training.

Do not modify `RivalObsV1` to hide/remove the dummy.

## Gate 5 — Reset distribution audit

Sample at least 10,000 resets per family (50,000 total Phase A) and verify:

- finite/legal positions/velocities/orientations;
- configured family shares when mixed;
- active-team balance;
- left/right symmetry;
- intended distance ranges;
- intended ball-speed ranges;
- dummy non-interference;
- no impossible geometry inside walls/ceiling/goal structure;
- natural kickoff family is a legal ordinary kickoff geometry.

Repeat a smaller 5,000-per-family audit for Phase B widened distributions before Phase B can activate.

## Gate 6 — Deterministic evaluation authority

Before first PPO update:

- generate/freeze the 500-episode gate corpus;
- generate/freeze the disjoint >=250-episode generalization corpus;
- hash and commit the seed/state manifests (not huge raw simulator dumps);
- evaluate the exact source v10.1 +10 actor on both;
- store source metrics under `training/results/milestone10_2/`.

Every later boundary must reuse the frozen gate corpus exactly.

## Gate 7 — Throughput / trainer smoke

Because only one of two cars is trainable, run a short worker sweep or sanity check over candidates `32, 40, 48, 56, 64` if the new masking path materially changes throughput.

Select based on stable **trainable active-learner steps/sec**, not raw environment-agent rows including dummy data.

Then run one disposable PPO iteration and prove:

- CUDA path works;
- rollout log-prob replay is within existing tolerance;
- GAE uses only active-learner trajectory boundaries;
- continuous actor head has nonzero finite gradients;
- button categorical head has nonzero finite gradients;
- fresh critic updates;
- actor parameters change;
- checkpoint save/reload is exact;
- workers remain alive;
- no dummy sample reaches the PPO batch.

Discard disposable updated weights and reinitialize from the exact source actor + fresh critic/optimizers before the real campaign.

## Gate 8 — Start authority

Only after Gates 0–7 pass may the real v10.2 campaign begin.

Do not rerun the v9 architecture gates unless implementation evidence reveals a regression in a frozen contract.
