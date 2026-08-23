# v10.1 Evaluation and Exit Gates

## Principle

The bootstrap is successful when Rival develops **basic agency**, not when reward/throughput looks healthy.

Training loss, collected reward, entropy, worker health and steps/sec are necessary diagnostics but are not capability evidence.

Use fixed deterministic evaluations plus curriculum-specific probes at every boundary.

## Evaluation boundaries

From the exact v10.1 start checkpoint, evaluate at approximately:

- +2.5 bootstrap simulated hours;
- +5h;
- +10h;
- +15h;
- +20h;
- +25h maximum.

Always stop on a clean PPO/checkpoint boundary near the target rather than truncating an update.

Continue recording the original M09/M10 fixed deterministic protocol so the learning curve remains comparable across reward changes. Add a separate bootstrap probe suite; do not replace the historical protocol.

## Required capability metrics

### Motion / controller effectiveness

- mean and p50/p95 car speed;
- mean planar speed;
- time share >500, >1000, >1500, >2000 uu/s;
- throttle distribution;
- boost-pressed share;
- steer/pitch/yaw/roll distribution and saturation;
- deterministic jump initiations / 100k steps;
- deterministic dodge initiations / 100k steps;
- no-button/button-combo distribution.

Control fields are diagnostics only; they are not directly rewarded.

### Ball interaction

- mean/p50 distance to ball;
- distinct logical touches / 100k agent-steps;
- raw touch records / 100k;
- aerial logical touches / 100k;
- touches by reset family;
- touch-chain starts / 100k;
- maximum touch-chain length;
- chain-length histogram;
- percentage of episodes with >=1, >=2 and >=3 self touches;
- time from reset to first touch distribution.

### Outcome

- goals for/against;
- scoring episodes / total episodes;
- easy-finish success rate;
- natural-kickoff scoring activity;
- no-touch-timeout share;
- ordinary timeout share;
- goal termination share.

A 0-0 draw against another inactive policy is explicitly **not success**.

### Reward integrity

For every component report:

- signed reward;
- absolute reward;
- fraction of episode budget consumed;
- number of budget clips;
- reward per logical touch;
- reward per simulated second;
- correlations between reward and actual capability metrics.

If total reward rises while touches/goals/motion do not, treat it as potential reward hacking and stop to inspect.

## Phase A readiness

Phase A may transition to Phase B only at an evaluation boundary after at least **2.5 bootstrap simulated hours**, and only if all of the following are true in the deterministic/bootstrap probe suite:

1. mean planar speed >= **600 uu/s**;
2. distinct logical touches >= **75 / 100k agent-steps**;
3. deterministic jump initiations >= **5 / 100k**;
4. at least one successful goal is produced in the fixed easy-finish probe set;
5. no-touch-timeout share <= **60%**;
6. no evidence that `useful_speed_rate` dominates reward while ball interaction is stagnant.

These are basic-agency thresholds, not claims of competitive skill.

If Phase A has not passed by +10 bootstrap hours, **stop and report** rather than automatically increasing the reward again.

## Phase B readiness

Phase B may transition to Phase C after at least **10 total bootstrap hours** only if:

1. distinct logical touches >= **150 / 100k**;
2. deterministic jump initiations >= **20 / 100k**;
3. deterministic dodge initiations >= **5 / 100k**;
4. aerial touches >= **3 / 100k**;
5. at least some multi-touch behavior exists: >=10 two-or-more-touch chains / 100k, or a clearly documented equivalent metric;
6. goal activity is nonzero outside the easiest finish-only probe;
7. no-touch-timeout share <= **40%**.

If not achieved by +15h, remain in Phase B only to the next boundary while reporting the specific blocker. Do not silently weaken the gate.

## Bootstrap exit gate

End v10.1 and return toward normal outcome-dominant training when two consecutive evaluation boundaries show all of:

- mean planar speed >= **700 uu/s**;
- distinct logical touches >= **150 / 100k**;
- deterministic jumps >= **20 / 100k**;
- deterministic dodges >= **10 / 100k**;
- aerial touches >= **5 / 100k**;
- repeated-touch chains are nontrivial and not contact-farming artifacts;
- nonzero goal activity in natural/full-play evaluations;
- no-touch-timeout share <= **30%**;
- mean ball distance materially better than the pre-bootstrap +25 M10 checkpoint;
- no single non-outcome reward component explains capability gains by obvious farming.

The two-boundary requirement protects against one lucky evaluation.

## Hard maximum

v10.1 authorizes at most **25 additional simulated game-hours**.

If the exit gate has not passed by then, stop with all evidence preserved. Do not simply add another 100 hours.

At that point the next intervention should be chosen from evidence and may include imitation/motor pretraining or an active external opponent curriculum.

## External opponents

Do not mix Wisp/Nexto into the bootstrap merely to make the opponent stronger. The purpose of v10.1 is basic motor/ball agency.

Once the exit gate passes, the next campaign should introduce an opponent league containing current Rival, historical Rival checkpoints, and stronger external opponents. That is a separate versioned training change.
