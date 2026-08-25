# Milestone 10.9: Stage-1 PPO V2

M10.9 completed the authorized Stage-1-only experiment and stopped at +1.0
learner-simulated hour. No checkpoint was promoted and Stage 2 was not started.

## Result

PPO V2 corrected the targeted formulation mechanics, but it did **not**
materially improve deterministic native-continuous-control learning.

The deterministic first-contact curve was:

| Boundary | First | Second | Third | All three | No-touch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initialization | 18.0% | 3.2% | 0.8% | 0.8% | 97.0% |
| +0.25h | 18.0% | 4.0% | 2.4% | 2.4% | 97.2% |
| +0.5h | 18.8% | 4.4% | 1.6% | 1.6% | 97.0% |
| +1.0h | 17.8% | 4.4% | 1.0% | 1.0% | 97.0% |

The stochastic first-contact curve was 8.0%, 8.6%, 9.0%, and 10.0%.
Stochastic all-three success finished at 0.4%.

At +1h deterministic failures ended 2,866.98 uu farther from the ball and
lost 0.6593 alignment on average. Relative to M10.8 Arm C at +1h, M10.9 was
1.4 percentage points lower on first contact, 0.4 points higher on second
contact, 1.2 points lower on third/all-three success, and only 67.26 uu less
bad on failed distance change. The required monotonic capability improvement
did not occur.

## What PPO V2 proved

- Scale-only advantage normalization preserved every raw-advantage sign.
- The disposable critic held-out EV improved from -0.00324 to 0.58864 and
  held-out loss fell from 12.51270 to 2.32225 after the specified 96 updates.
- Every real campaign critic iteration improved held-out EV and loss.
- Actor and critic schedules were independently executed at 2 and 8 epochs;
  KL protection remained active at 0.015.
- AR(1) exploration used `tau=0.075 s` and exact
  `rho=0.8948393168143698` at 120 Hz.
- Same-policy and independent conditional log-probability replay passed for
  every campaign iteration. The maximum errors were 1.10e-05 and 5.72e-06.
- Successful first-contact actions retained positive raw and scaled credit
  through the 2-3 second window. Negative failed-window means remained
  negative after scale-only normalization.

These results isolate the remaining failure from the specific defects targeted
by M10.9. The next investigation should address state/value decomposition,
policy architecture, per-component advantage attribution, replay/off-policy or
alternative actor-critic methods, and PPO suitability at native 120 Hz. Reward
retuning is not supported by this result.

## Evidence

- `preflight.json`: hard preflight, paired initialization, disposable critic,
  action mapping, reward truth table, AR and likelihood proofs.
- `training_plus-000p25h.json`, `training_plus-000p5h.json`, and
  `training_plus-001h.json`: complete PPO and credit-assignment diagnostics.
- `evaluation_*`: frozen 500-episode deterministic and stochastic evaluations.
- `final_comparison.json`: full capability curve, M10.8 Arm C comparison, and
  exact experiment conclusion.
- `final_verification.json`: tests, lint, compilation, parsing, checkpoint
  parity, process cleanup, Wisp hashes, and stash preservation.

The final checkpoint remains evidence-only at
`training/checkpoints/milestone10_9/stage_1/boundaries/plus-001h/000432006`.
Checkpoints remain Git-ignored.
