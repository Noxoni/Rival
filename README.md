# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**.

The initial development baseline uses **Wisp v2-75B** and **Nexto** as the primary reference bots. The goal is not to make a beginner hand-coded bot; Rival is intended to preserve and improve on strong learned-bot behavior, including possession, flick pressure, boost denial, aerial mechanics, challenge recognition, resource feasibility, and score/clock-aware 1v1 strategy.

## Current Codex handoff

Start here:

`handoff/v1.1/CODEX_START_PROMPT.md`

Codex should read the complete `handoff/v1.1/` package before modifying implementation code.

## Local RLBot BotPack source

The user's installed RLBot v5 bots are located at:

`C:\Users\patri\AppData\Local\RLBot5\bots`

Treat the installed BotPack as read-only. The primary local references are Wisp v2-75B and Nexto.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
