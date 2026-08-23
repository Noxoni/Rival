# Rival Codex Handoff v7.0

Status: active transfer-diagnostic milestone.

Starting implementation boundary:

`652395a9f512ce835830bfc5bc3a7cb078f6105e`

Purpose:

- isolate the M06 RocketSim -> RLBot transfer failure;
- separate observation, policy, action-function/cadence, and transition-physics causes;
- test zero-step reconstructed Wisp at both tick 8 and tick 4 before blaming PPO;
- quantify 20M legacy-logit drift on live observations;
- produce the corrective architecture for the next training milestone.

No serious PPO continuation is authorized by this handoff.
