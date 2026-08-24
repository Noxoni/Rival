# Rival handoff v10.2

- Milestone: `10.2`
- Title: `Ball Acquisition`
- Design branch: `rival-v10.2-ball-acquisition`
- Base authority commit: `cc2d971b4990121684920f87a0ee2b87b6dc801b`
- Source actor checkpoint: `training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`
- Source actor SHA-256: `e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`
- Frozen policy: `RivalPolicyV1`
- Frozen observation: `RivalObsV1`
- Frozen action: `RivalActionV1`
- New reward: `RivalBallAcquisitionRewardV1`
- New curriculum: `RivalBallAcquisitionCurriculumV1`
- Production promotion: `not_authorized`

Milestone intent:

> Retain the locomotion primitive learned by v10.1, remove all speed reward, and teach reliable ball acquisition through signed car-caused distance reduction plus maximum reward for every genuine new ball touch.

Next skill if passed: `ground_ball_control_dribbling`.
