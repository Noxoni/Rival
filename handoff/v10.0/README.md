# Rival v10.0 — first serious scratch training campaign

Milestone 10 is the first sustained training campaign for the completed Milestone 09 scratch foundation.

Starting authority:

`824e328f6bbf4fe9a47b8e54706b5fcf645fd409`

M09 proved the architecture. M10 changes **experience volume**, not the architecture.

## Core decision

Resume the exact highest-step M09 Gate 13 checkpoint and train the same `RivalPolicyV1` for up to **100 additional simulated game-hours**.

Do not reset the actor, critic, optimizers, trainer counters, observation contract, action contract, reward, cadence, or curriculum merely to make M10 look like a fresh run.

The policy remains:

- `RivalObsV1` — 714 floats, canonical shared train/deploy implementation;
- `RivalActionV1` — five continuous controller axes plus the joint 8-way jump/boost/handbrake categorical;
- native 120-Hz policy cadence;
- no action lookup table;
- no RepeatAction;
- no Wisp actor/trunk;
- no mechanics macros;
- independent actor and critic.

Production remains frozen Wisp. M10 does not authorize promotion.

## Campaign budget

One simulated 1v1 game-hour is 864,000 agent-steps at 120 Hz with two agents.

M10 authorizes **86,400,000 additional agent-steps**, corresponding to 100 additional simulated game-hours, subject to normal complete-PPO-iteration alignment.

Starting M09 checkpoint state:

- cumulative agent-steps: `1,680,214`;
- simulated game-hours: `1.9446921296`.

Nominal M10 cumulative ceiling:

- cumulative agent-steps: `88,080,214`;
- cumulative simulated game-hours: `101.9446921296`.

See `MILESTONE_10_SPEC.md`, `EVALUATION_PROTOCOL.md`, and `CODEX_START_PROMPT.md`.
