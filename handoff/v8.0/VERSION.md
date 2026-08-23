# Rival handoff version 8.0

- Status: active Milestone 08 authority
- Created after completed Milestone 07 transfer diagnosis
- Required starting ancestor: `10c41f708d6e8145bf719f8f322041e7753f6c3f`
- Rejected M06 20M actor: diagnostic/reference only; do not resume as base policy
- Base policy: verified zero-step direct Wisp reconstruction with exact first-90 logits
- Production policy: unchanged frozen Wisp at tick skip 8
- Production promotion: not authorized by this milestone

## Core change from v7

v7 localized the failure. v8 implements the correction:

1. live/training Wisp observation parity;
2. exact native Wisp 8-tick temporal execution in RocketSim;
3. frozen strategic Wisp branch;
4. separate PASS-or-appended-action mechanics branch at 4 ticks;
5. PPO training limited to the mechanics branch and critic until transfer gates pass.

A later milestone may authorize constrained legacy-policy fine-tuning only after this architecture demonstrates healthy RLBot transfer.