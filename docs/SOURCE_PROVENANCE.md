# Rival Source Provenance

## Installed RLBot v5 references

The installed BotPack root was inspected read-only on 2026-08-22:

`C:\Users\patri\AppData\Local\RLBot5\bots`

Metadata discovery selected these exact primary references:

| Reference | Installed path | Identity | Package form |
| --- | --- | --- | --- |
| Wisp v2-75B | `C:\Users\patri\AppData\Local\RLBot5\bots\bob_build_x86_64-windows\WispV2` | `eastvillage/wisp/v2-75B` | Windows executable plus TOML/loadout/logo |
| Nexto | `C:\Users\patri\AppData\Local\RLBot5\bots\bob_build_x86_64-windows\Nexto` | `rlgym/nexto` | Windows executable plus TOML/loadout/logo |

The local immutable snapshot is `.local_reference_sources/v1/`. It is Git-ignored. Its exact tracked SHA-256 record is `reference_manifests/v1/MANIFEST.json`; the snapshot verifier passed immediately after creation.

The primary executable hashes are:

- Wisp `bot.exe`: `b9dbe32bdae28c299daffcc0673f5a438d8013a553be3938dafaa112884e7184`
- Nexto `nexto.exe`: `6371a5b9dd740aea858d86532f928e0fdb13bf387d8379f76ccd679c9b33e845`

Neither executable reports a Windows file/product version. Neither installed tree exposes Python source or a standalone model artifact, so source-level work cannot be derived from the installed packages alone.

## Wisp baseline source

Rival uses the public Wisp v2 Python release source:

- Repository: `https://github.com/NicEastvillage/RLBot-Wisp-v2-py`
- Commit: `58d4ab18fd0c92529b5ae6582ecf1713a6b1887a`
- Commit subject: `Prepare Wisp v2-75B for release`
- Upstream bot identity: `eastvillage/wisp/v2-75B`

The public source identity matches the installed Wisp name and agent id. The installed config launches a packaged `bot.exe`; the public config launches `bot.py`. No claim is made that the executable can be reproduced bit-for-bit from this source commit.

The Rival baseline uses the two TorchScript artifacts published in that Wisp commit:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/models/POLICY.lt` | 7,689,613 | `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7` |
| `src/models/SHARED_HEAD.lt` | 5,995,907 | `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42` |

These model artifacts come from the public Wisp repository, not as separately extractable files from the installed BotPack executable.

Wisp's repository `LICENSE` is MIT. Its README adds that the software and derivatives may not be used to cheat or otherwise break Rocket League's terms of service. Rival is restricted to offline RLBot play, retains the upstream notice, and does not target online matchmaking or Terms-of-Service circumvention.

## Nexto references and incorporation boundary

Two upstream references were recorded for analysis only:

- RLBot v5 port: `https://github.com/VirxEC/NectoFamily` at `0bdb6b49072f6f3829319e68bd6210a0ca4b24a2`
- Original Necto/Nexto project: `https://github.com/Rolv-Arild/Necto` at `2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

The original repository carries CC BY-NC-SA 4.0. The NectoFamily port has no repository license file at the recorded commit, so redistribution permission for that port was not established. Rival therefore incorporates no Nexto/Necto source or model artifact in Milestone 01. The installed Nexto and both public repositories remain comparison/provenance references only.

## Machine-readable record

`reference_manifests/v1/UPSTREAM_SOURCES.json` records all upstream commits, licenses, baseline-selection decisions, and Wisp model hashes.
