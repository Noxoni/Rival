# Rival Source Provenance Notes

## Installed local BotPack

Primary local lookup root:

```text
C:\Users\patri\AppData\Local\RLBot5\bots
```

The installed files must be inspected by Codex on the user's machine. This handoff does not assume exact installed subfolder names.

## Public Wisp reference

Repository:

```text
https://github.com/NicEastvillage/RLBot-Wisp-v2-py
```

Relevant known metadata as of handoff:

- RLBot v5 Python implementation.
- Agent id: `eastvillage/wisp/v2-75B`.
- Public repository contains `src`, model/runtime code, RLBot bot TOML, and build metadata.
- Repository license is MIT with an additional restriction against using the software or derivatives to cheat or otherwise break Rocket League's terms of service.

This project is for offline RLBot play only.

## Public Nexto references

RLBot v5 port:

```text
https://github.com/VirxEC/NectoFamily
```

Original project:

```text
https://github.com/Rolv-Arild/Necto
```

The original project includes RLBot support and historical training code/material. The v5 port exists specifically to run Necto/Nexto under RLBot v5.

## Provenance record Codex must create

After discovery, create `docs/SOURCE_PROVENANCE.md` in the implementation repository containing:

- discovered local path for Wisp
- discovered local path for Nexto
- file hashes or manifest path
- local package/release metadata if detectable
- upstream repository URLs
- exact upstream Git commit SHAs if cloned/fetched
- which model files are used by the dev baseline
- whether those model files came from the local BotPack or upstream
- any differences found between local installed artifacts and upstream source

## Rival repository

Canonical implementation repository:

```text
https://github.com/Noxoni/Rival
```

Local installed reference trees should normally remain outside tracked source. Their manifests and provenance belong in the repository.

Any third-party source/model incorporated into Rival must retain required license notices and must be checked for redistribution permission before commit.
