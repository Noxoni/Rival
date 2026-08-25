# Milestone 10.8 — Controlled PPO Credit-Assignment / GAE Experiment

## Outcome

Milestone 10.8 completed the authorized three-arm Stage-1 experiment and stopped every arm at +1 learner-simulated hour. No arm was promoted, production was not modified, and Stage 2 was not started.

The controlled evidence supports a bounded conclusion: the native-120-Hz GAE credit horizon was materially limiting Stage-1 learning, but increasing lambda was not sufficient to produce reliable ball acquisition. Arm C is the only monotonic deterministic curve and has the strongest final reacquisition result. Its lambda, `0.9983695094257663`, is the value supported for a subsequent controlled run—not for production promotion.

This is a one-seed, one-hour paired experiment. The capability remains poor in absolute terms, and the modest differences must not be described as a solved acquisition policy or proof that lambda was the sole remaining cause.

## Frozen paired initialization

All three arms were cloned from the exact M10.7 pre-PPO initialization checkpoint:

- checkpoint: `training/checkpoints/milestone10_7/stage_1/initialization/000000000`
- source actor file SHA-256: `1e58fa4f6ad107344fa7d163b53e1af2aad871ec01d16ed47776d61b20397548`
- source manifest SHA-256: `301b6f3a65cf998bdffe39fcd23c353c5187d2c760cfd74ebbec356bf5fd2824`
- paired actor-state SHA-256: `1bce479b61613b6284f94861ade03214f0d940be39dcce4499fea541178b0daf`
- paired critic-state SHA-256: `d204ecae323d911465bcd3a0f5541a9823928f470ebca2beb300f1edd0c1ab97`
- actor optimizer entries per arm at initialization: `0`
- critic optimizer entries per arm at initialization: `0`

The arm configurations are byte-equivalent after masking only `ppo.gae_lambda`. Reward, curriculum, observations, actions, policy, environment, optimizer settings, persistence, entropy schedule, PPO learning rates, clip range, batch sizes, and epochs remain frozen to M10.7.

## Physical-time GAE proof

The preflight analytically calculated each horizon and reproduced it on a 361-transition synthetic trajectory with a unit terminal reward. All measured advantages matched the analytical values within `1e-6`.

| Arm | Lambda | Gamma × lambda | Half-life | 0.25 s | 0.5 s | 1 s | 2 s | 3 s | 5 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A — control | 0.9872585449014338 | 0.9860190386615196 | 0.410255 s | 65.55% | 42.97% | 18.46% | 3.41% | 0.63% | 0.02% |
| B — physical conversion | 0.993608849045455 | 0.9923613699785113 | 0.753294 s | 79.45% | 63.12% | 39.85% | 15.88% | 6.33% | 1.00% |
| C — longer credit | 0.9983695094257663 | 0.9971160533345892 | 2.000000 s | 91.70% | 84.09% | 70.71% | 50.00% | 35.36% | 17.68% |

Gamma remained exactly `0.9987444968227265` in all arms.

## Deterministic capability

Every row is the same frozen 500-episode Stage-1 corpus. Values are shares of episodes.

| Arm | Boundary | First | Second | Third | All three | No-touch timeout |
|---|---:|---:|---:|---:|---:|---:|
| shared | init | 18.0% | 3.2% | 0.8% | 0.8% | 97.0% |
| A | +0.5h | 15.8% | 3.0% | 1.0% | 1.0% | 97.6% |
| A | +1h | 15.8% | 1.8% | 0.6% | 0.6% | 97.8% |
| B | +0.5h | 17.8% | 3.6% | 1.4% | 1.4% | 97.2% |
| B | +1h | 19.8% | 3.4% | 1.4% | 1.4% | 94.4% |
| C | +0.5h | 18.8% | 3.8% | 1.6% | 1.6% | 97.2% |
| C | +1h | 19.2% | 4.0% | 2.2% | 2.2% | 97.2% |

The deterministic first-contact curves were:

- Arm A: `18.0% → 15.8% → 15.8%` (regressed, then flat)
- Arm B: `18.0% → 17.8% → 19.8%` (highest endpoint, but non-monotonic)
- Arm C: `18.0% → 18.8% → 19.2%` (only monotonic curve)

Arm B’s first-contact endpoint leads Arm C by only three episodes (`0.6` percentage point). Arm C has the strongest second-, third-, and all-three-contact endpoint and is the only arm satisfying the monotonicity criterion. Relative to Arm A at +1h, Arm C is `+3.4` points on first contact, `+2.2` on second contact, and `+1.6` on third/all-three contact.

## Stochastic capability

| Arm | Boundary | First | Second | Third | All three | No-touch timeout |
|---|---:|---:|---:|---:|---:|---:|
| shared | init | 8.4% | 0.2% | 0.0% | 0.0% | 98.2% |
| A | +0.5h | 10.4% | 0.2% | 0.0% | 0.0% | 98.2% |
| A | +1h | 10.4% | 0.6% | 0.0% | 0.0% | 98.2% |
| B | +0.5h | 10.0% | 2.0% | 0.6% | 0.6% | 98.2% |
| B | +1h | 10.4% | 1.4% | 0.8% | 0.8% | 98.0% |
| C | +0.5h | 10.8% | 1.0% | 0.2% | 0.2% | 97.0% |
| C | +1h | 9.6% | 1.2% | 0.2% | 0.2% | 97.6% |

All three final stochastic policies remained materially worse than deterministic deployment on first contact. Longer lambda did not eliminate that gap.

## Final deterministic trajectory and action behavior

Positive failed-distance change means the car finished farther from the ball. All arms still moved substantially farther away on failed trajectories, so none is a reliable acquisition policy.

| Arm | Failed distance change | Failed alignment change | Mean throttle | Mean abs steer | Jump | Boost | Handbrake |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | +2933.40 uu | -0.5284 | +0.2708 | 0.1154 | 0.0% | 0.0% | 0.0% |
| B | +3072.35 uu | -0.7477 | +0.2015 | 0.0319 | 0.0% | 0.0% | 0.0% |
| C | +2934.24 uu | -0.7222 | +0.1878 | 0.0381 | 0.0% | 0.0% | 0.0% |

Successful deterministic timing means at +1h:

| Arm | Reset → first | First → second | Second → third |
|---|---:|---:|---:|
| A | 1.780 s | 1.512 s | 0.806 s |
| B | 2.252 s | 1.780 s | 1.089 s |
| C | 2.072 s | 2.057 s | 0.922 s |

These timing values describe the small successful subset and must not be generalized to the failed majority.

## Credit-assignment diagnostics

Mean raw advantage in successful first-contact trajectories at the +1h boundary:

| Arm | 0–0.5 s | 0.5–1 s | 1–2 s | 2–3 s |
|---|---:|---:|---:|---:|
| A | +6.267 | +2.339 | +0.775 | +0.241 |
| B | +6.749 | +4.011 | +2.102 | +0.183 |
| C | +7.765 | +6.739 | +5.469 | +2.956 |

Arm C plainly delivered meaningful positive advantage to actions two to three seconds before contact. The matched failed-timeout windows remained negative in raw advantage for all arms; Arm C’s 2–3-second failed-window mean was `-1.899`. This confirms that the implementation changed physical-time credit as intended and did not merely inflate every preceding action.

The credit windows also report raw and normalized advantage, throttle, steer magnitude, heading improvement, and car-caused distance progress for successful and failed cohorts in every PPO iteration. See the arm boundary JSON files for the complete statistics.

## Interpretation

Question A — does `0.95^(1/8)` improve over the old lambda?

Arm B finished above Arm A on deterministic first contact and all reacquisition measures, but its curve dipped at +0.5h before recovering. The evidence favors B over the control endpoint, but it does not demonstrate monotonic learning.

Question B — does the approximately two-second half-life improve further?

Arm C did not beat B’s raw first-contact endpoint, but the difference was only `0.6` point. C was the only monotonic deterministic curve, produced the strongest second/third/all-three contact result, and demonstrably propagated positive advantage into the 2–3-second pre-contact window. Under the predeclared comparison rules, C is the supported carry-forward arm.

Question C — did either arm avoid the analog-policy deterioration seen in M10.7?

Partially. Arms B and C did not show the control arm’s endpoint regression in deterministic first contact, and C improved monotonically. However, failed trajectories still ended roughly 2934–3072 uu farther from the ball and reacquisition remained at or below 4.0%. The policy is still not reliable.

Final diagnosis: physical-time GAE horizon is supported as a material limitation, not as the sole or sufficient explanation for the remaining acquisition failure. The next controlled run may carry Arm C’s lambda `0.9983695094257663`; it must not promote this checkpoint or start Stage 2. Continued investigation should still examine critic accuracy, advantage normalization, PPO update stability, actor update magnitude, and reward-component attribution.

## Evidence and checkpoints

- paired preflight: `training/results/milestone10_8/preflight.json`
- final side-by-side comparison: `training/results/milestone10_8/final_comparison.json`
- per-arm boundary/evaluation/training evidence: `training/results/milestone10_8/arms/`
- ignored final checkpoints:
  - A: `training/checkpoints/milestone10_8/arms/arm_a/boundaries/plus-001h/000432016`
  - B: `training/checkpoints/milestone10_8/arms/arm_b/boundaries/plus-001h/000432003`
  - C: `training/checkpoints/milestone10_8/arms/arm_c/boundaries/plus-001h/000432008`

M10.7 and all prior M10.x evidence/checkpoints remain preserved. Frozen production Wisp remains unchanged.
