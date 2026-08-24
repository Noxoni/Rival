# Rival handoff v10.2

- Milestone: `10.2`
- Title: `Progressive Prerequisite Curriculum — Stages 1 through 4`
- Design branch: `rival-v10.2-ball-acquisition`
- Base authority commit: `cc2d971b4990121684920f87a0ee2b87b6dc801b`
- Initial source actor checkpoint: `training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`
- Initial source actor SHA-256: `e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`
- Frozen policy: `RivalPolicyV1`
- Frozen observation: `RivalObsV1`
- Frozen action: `RivalActionV1`
- Authorized lessons: `ball_acquisition -> ground_control -> aerial_control -> finishing`
- Terminal automatic stage: `4`
- Terminal success decision: `finishing_skill_passed_unlock_opponent_pressure`
- Stage 5 opponent pressure: `not_authorized_automatic`
- Overnight wall-clock authority: `10 real hours total`
- Finalization reserve: `20 minutes`
- Total experience ceiling: `90 learner-simulated hours / 38,880,000 active-learner steps`
- Production promotion: `not_authorized`

Milestone intent:

> Retain each learned prerequisite in the actor, reset critic/optimizer state when the lesson changes, remove/reduce obsolete shaping, and advance only after deterministic mastery of the current skill.

The run stops on whichever applies first: stage mastery, documented stage failure/no-learning/exploit gate, the global 10-hour wall-clock envelope, or the relevant stage/total experience ceiling.
