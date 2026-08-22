# Wisp v2 Third-Party Notice

Rival Milestone 01 is derived from the Wisp v2-75B Python release published at:

`https://github.com/NicEastvillage/RLBot-Wisp-v2-py`

Imported commit: `58d4ab18fd0c92529b5ae6582ecf1713a6b1887a` (`Prepare Wisp v2-75B for release`).

The upstream `LICENSE` is preserved verbatim in this directory. The upstream README is also preserved because it states an additional restriction: the software and its derivatives may not be used to cheat or otherwise break Rocket League's terms of service. Rival is for offline RLBot play only.

The initial import copied the upstream `src/` runtime except its Wisp identity TOML and logo. These upstream files were modified for Rival:

- `bot/bot.py`: unique Rival runtime class, inspectable decision seam, measurement-only metrics, and optional telemetry.
- `bot/config.py`: unique agent id and telemetry/inspection configuration.
- `bot/backend/model.py`: raw/masked policy-output seam while preserving Wisp action selection.
- `bot/backend/gamestate/phys_obj.py` and `bot/utils.py`: whitespace-only cleanup; runtime behavior is unchanged.

Rival-added modules under `bot/policy/`, `bot/analysis/`, and `bot/telemetry/` are not upstream Wisp files.

The published upstream model artifacts are included unchanged:

- `bot/models/POLICY.lt`: SHA-256 `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`
- `bot/models/SHARED_HEAD.lt`: SHA-256 `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`

No Nexto/Necto source or model artifact is incorporated into Rival.
