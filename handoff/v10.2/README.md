# Rival v10.2 — Progressive Prerequisite Curriculum (Stages 1–4)

Milestone 10.2 is now the **progressive skill-learning package** for Rival's scratch policy.

It starts from the locomotion/agency primitive learned in v10.1 and authorizes Codex to train four prerequisite lessons in strict order without waiting for a human between successful lessons:

1. **Ball acquisition** — get to the ball and touch it.
2. **Ground ball control / dribbling** — keep it reachable and make repeated controlled ground contacts.
3. **Aerial acquisition / air-dribble control** — reach elevated balls and sustain repeated aerial contacts.
4. **Finishing / scoring** — use learned ground/aerial control to deliberately put the ball in the target goal.

After Stage 4 passes, Codex must stop for human review. Active-opponent training and self-play are not authorized by this package.

## Governing principle

> **Learn one prerequisite, prove it deterministically, reduce/remove its direct reward, then use that capability as the starting skill for the next lesson.**

A later stage may begin only if the previous stage emits its exact success decision. A failed stage stops the entire progression; Codex may not skip ahead.

See `PROGRESSIVE_STAGE_PROTOCOL.md` and `M10_2_PROGRESSIVE_CAMPAIGN.json`.

## Frozen architecture

Stages 1–4 do not redesign the bot. Keep frozen:

- `RivalPolicyV1`;
- `RivalObsV1` (714 floats);
- `RivalActionV1` (five continuous controller axes + joint jump/boost/handbrake categorical);
- native 120-Hz policy and physics cadence;
- one-tick RocketSim/RLBot action-delay contract;
- canonical RocketSim/RLBot adapters;
- actor topology/export/live inference path.

Each isolated lesson uses one active learner plus one inert second car solely to preserve the unchanged opponent observation contract. Dummy transitions are excluded from PPO.

No Wisp/Nexto actor/trunk enters the scratch policy.

## Initial source actor

Stage 1 begins from the **actor weights** of the final recoverable v10.1 +10h checkpoint:

`training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`

Expected actor SHA-256:

`e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`

That actor is retained because v10.1 established a locomotion/speed primitive.

At every stage transition, transfer **actor weights only** from the exact passing prerequisite checkpoint and initialize:

- a fresh `RivalCriticV1`;
- fresh actor optimizer state;
- fresh critic optimizer state.

Preserve every passing prerequisite checkpoint byte-for-byte.

## Stage 1 — Ball acquisition

Question:

> Can Rival locate/reach broad reachable ground balls and physically touch them?

Reward:

- small signed car-caused distance reduction;
- `+1.0` for every genuine new physical learner touch;
- zero speed reward;
- zero goal/concede reward;
- zero future-skill rewards.

Authority:

- `BALL_ACQUISITION_REWARD.md`
- `BALL_ACQUISITION_CURRICULUM.md`
- `EVALUATION_AND_EXIT_GATES.md`
- `IMPLEMENTATION_GATES.md`
- `M10_2_CAMPAIGN.json`

Exact success decision:

`ball_acquisition_skill_passed_unlock_ground_control`

Maximum: 15 learner-simulated hours.

## Stage 2 — Ground control / dribbling

Locked until Stage 1 passes.

Question:

> After first contact, can Rival keep the ball reachable and produce repeated controlled ground touches?

The first touch becomes a small bridge; **follow-up physical touches become the maximum reward event**. Scoring is still reward-neutral.

Authority:

- `STAGE_2_GROUND_CONTROL.md`

Exact success decision:

`ground_control_skill_passed_unlock_aerial_control`

Maximum: 20 learner-simulated hours.

## Stage 3 — Aerial acquisition / air-dribble control

Locked until Stage 2 passes.

Question:

> Can Rival intentionally reach an airborne ball and sustain repeated aerial contacts?

Aerial acquisition is learned first, then air-control/repeated aerial touches. No direct reward for pressing jump/boost, and scoring remains reward-neutral.

Authority:

- `STAGE_3_AERIAL_CONTROL.md`

Exact success decision:

`aerial_control_skill_passed_unlock_finishing`

Maximum: 30 learner-simulated hours.

## Stage 4 — Finishing / scoring

Locked until Stage 3 passes.

This is the **first stage where a goal is a positive objective**.

Question:

> Can Rival use the acquisition/control skills it already owns to deliberately finish into the target goal from ground and aerial possession?

Reward hierarchy:

- small bounded control retention;
- small bounded ball-to-target-goal progress;
- correct goal `+10`;
- own goal `-10`.

Capability evaluation separately requires ground-control-qualified and aerial-control-qualified goals so accidental/passive or trivial one-touch goals cannot satisfy the final gate by themselves.

Authority:

- `STAGE_4_FINISHING.md`

Exact success decision:

`finishing_skill_passed_unlock_opponent_pressure`

Maximum: 25 learner-simulated hours.

After this decision, **stop**. Do not start Stage 5/opponent pressure or self-play automatically.

## Total unattended authority

Maximum if every stage consumes its full ceiling:

- **90 learner-simulated hours**;
- **38,880,000 active-learner 120-Hz steps**.

This is a ceiling, not a target. Every stage terminates immediately on mastery or on its documented failure/stop gate.

## Reporting

Use one umbrella report:

`docs/MILESTONE_10_2_RESULTS.md`

and compact evidence under:

`training/results/milestone10_2/`

with per-stage subdirectories/records.

Push coherent stable boundaries after preflight, each evaluation boundary, every stage transition, every failure closeout, and final Stage-4 completion.

## Production

No production promotion is authorized anywhere in Stages 1–4. Frozen Wisp remains production until a future milestone explicitly changes that decision.
