# Rival v10.1 — Agency Bootstrap

## Decision

Milestone 10 demonstrated that the scratch policy/trainer is healthy but that the current learning distribution can spend large amounts of simulated time without learning useful ball interaction. The +10h fixed evaluation reported zero deterministic jumps, zero dodges, zero aerial touches, roughly 10.4 touches per 100k agent-steps, no scoring activity, and worse mean ball distance than the M09 foundation baseline, while recovery improved sharply.

v10.1 is a **temporary basic-agency bootstrap**, not a new policy architecture.

It keeps unchanged:

- `RivalPolicyV1`;
- `RivalObsV1` (714 floats);
- `RivalActionV1` (native continuous controller + joint buttons);
- native 120-Hz policy cadence;
- one-tick RLBot delay semantics;
- actor/critic architecture;
- PPO implementation and the M09-proven train/deploy contracts.

It changes only the learning distribution:

1. a temporary `RivalAgencyBootstrapRewardV1` that strongly rewards physically useful activity, ball interaction, aerial interaction, repeated possession touches, and goals;
2. a temporary broad reset curriculum that makes useful interactions frequent enough to discover;
3. shorter dead-play recycling;
4. capability-driven exit gates so the temporary shaping is removed once Rival can actually play.

## Activation boundary

Do not mutate the currently running M10 process in place.

Finish the already-near **+25 simulated-hour M10 boundary**, evaluate it with the existing fixed protocol, save/push the exact checkpoint and evidence, and stop the original M10 continuation there unless the user explicitly overrides this steering.

Then bring this package onto that completed +25 history and start v10.1 from the exact +25 checkpoint.

This preserves the clean M09 -> +5 -> +10 -> +25 learning curve and prevents an unversioned reward change in the middle of an iteration.

## Purpose

The bootstrap has one job:

> Make Rival discover that moving, controlling the car, reaching the ball, touching it repeatedly, touching it in the air, and ultimately scoring are valuable.

This is deliberately more elementary than the normal Rival reward.

The bootstrap is **not** meant to be the permanent mature reward. Once basic agency is reliable, revert toward the outcome-dominant normal reward/curriculum and introduce stronger opponent/league training.

## Anti-hacking principle

Do **not** reward raw button presses. A reward for `jump == 1`, `boost == 1`, steering magnitude, or action changes would teach input spam.

Reward the physical consequences instead:

- speed;
- useful motion toward the ball;
- actual ball contacts;
- actual airborne ball contacts;
- multiple distinct contacts in one possession chain;
- attacking ball progress;
- goals.

All dense terms remain cadence-safe at 120 Hz and all non-outcome shaping has a strict total episode budget below one goal reward.

## Files

- `AGENCY_BOOTSTRAP_REWARD.md` — exact reward contract and anti-farming rules.
- `AGENCY_BOOTSTRAP_CURRICULUM.md` — reset distribution and staged bootstrap curriculum.
- `EVALUATION_AND_EXIT_GATES.md` — metrics, stop/advance rules, and anti-reward-hacking checks.
- `M10_1_CAMPAIGN.json` — machine-readable authority.
- `CODEX_START_PROMPT.md` — execution authority.
- `VERSION.md` / `PACKAGE_MANIFEST.json` — package identity.
