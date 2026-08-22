# Rival Milestone 02 Candidate Evidence Report

> Detector findings are ranking candidates for review, not confirmed gameplay defects.

- Detector: `rival-m02-events-v1`
- Sessions: 12
- Decision records: 35230
- Candidate events detected: 1451
- Candidate events persisted: 30

## Candidate counts

| Class | Count |
| --- | ---: |
| `resource_stressed_aerial` | 281 |
| `boost_detour_possession_loss` | 300 |
| `apparent_vs_actual_challenge` | 870 |

## Sessions

| Session | Source | Opponent | Decisions | Warnings |
| --- | --- | --- | ---: | ---: |
| `rival-v2-natural-nexto-blue-20260822T172339Z-79e6298a` | natural_match | Nexto | 5468 | 0 |
| `rival-v2-natural-nexto-blue-20260822T173729Z-ec302968` | natural_match | Nexto | 5484 | 0 |
| `rival-v2-natural-nexto-orange-20260822T173013Z-345c6877` | natural_match | Nexto | 5844 | 0 |
| `rival-v2-natural-wisp-blue-20260822T174422Z-1d48f31f` | natural_match | Wisp v2-75B | 5561 | 0 |
| `rival-v2-natural-wisp-blue-20260822T175850Z-72963bbf` | natural_match | Wisp v2-75B | 5000 | 0 |
| `rival-v2-natural-wisp-orange-20260822T175114Z-7c4865d5` | natural_match | Wisp v2-75B | 6066 | 0 |
| `rival-v2-probe-fake_challenge-boost_then_brake-blue-20260822T172104Z-01611f12` | controlled_probe | Controlled probe (boost_then_brake) | 261 | 0 |
| `rival-v2-probe-fake_challenge-boost_then_veer-blue-20260822T172129Z-3d9b5777` | controlled_probe | Controlled probe (boost_then_veer) | 307 | 0 |
| `rival-v2-probe-fake_challenge-delayed_challenge-blue-20260822T172219Z-41c7e1b7` | controlled_probe | Controlled probe (delayed_challenge) | 216 | 0 |
| `rival-v2-probe-fake_challenge-jump_fake-blue-20260822T172154Z-aaa8a01a` | controlled_probe | Controlled probe (jump_fake) | 308 | 0 |
| `rival-v2-probe-fake_challenge-true_commit-blue-20260822T172029Z-b3f4c035` | controlled_probe | Controlled probe (true_commit) | 307 | 0 |
| `rival-v2-probe-resource_aerial-shadow-blue-20260822T172251Z-f938fbe6` | controlled_probe | Controlled probe (shadow) | 408 | 0 |

## Highest-ranked candidates by class

### `resource_stressed_aerial`

#### reso-6ce872d88ccc

- Session/time: `rival-v2-natural-wisp-blue-20260822T174422Z-1d48f31f` at 105.175s
- Opponent/source: Wisp v2-75B / natural_match
- Ranking score: 85.286
- Outcome: `none` next touch
- Why ranked: aerial-like action transition with elevated ball; low starting boost relative to detector reference; no later touch was observed in the bounded window; no grounded recovery was observed in the bounded window

#### reso-2b6fe4e80ec1

- Session/time: `rival-v2-natural-wisp-blue-20260822T174422Z-1d48f31f` at 125.083s
- Opponent/source: Wisp v2-75B / natural_match
- Ranking score: 83.719
- Outcome: `opponent` next touch
- Why ranked: aerial-like action transition with elevated ball; low starting boost relative to detector reference; opponent recorded the next touch

#### reso-59e994329842

- Session/time: `rival-v2-natural-wisp-orange-20260822T175114Z-7c4865d5` at 160.625s
- Opponent/source: Wisp v2-75B / natural_match
- Ranking score: 82.769
- Outcome: `opponent` next touch
- Why ranked: aerial-like action transition with elevated ball; low starting boost relative to detector reference; opponent recorded the next touch

#### reso-66d96995d99c

- Session/time: `rival-v2-natural-wisp-blue-20260822T174422Z-1d48f31f` at 102.308s
- Opponent/source: Wisp v2-75B / natural_match
- Ranking score: 82.750
- Outcome: `opponent` next touch
- Why ranked: aerial-like action transition with elevated ball; low starting boost relative to detector reference; opponent recorded the next touch

#### reso-1d60090eae56

- Session/time: `rival-v2-natural-wisp-orange-20260822T175114Z-7c4865d5` at 197.050s
- Opponent/source: Wisp v2-75B / natural_match
- Ranking score: 82.672
- Outcome: `opponent` next touch
- Why ranked: aerial-like action transition with elevated ball; low starting boost relative to detector reference; opponent recorded the next touch

### `boost_detour_possession_loss`

#### boos-161e02d9648e

- Session/time: `rival-v2-natural-nexto-blue-20260822T172339Z-79e6298a` at 77.708s
- Opponent/source: Nexto / natural_match
- Ranking score: 87.000
- Outcome: `opponent` next touch
- Why ranked: boost increased across a decision interval; distance to ball increased before the pickup; ETA possession proxy changed from favorable to unfavorable; opponent recorded the next touch

#### boos-ee643a0639a5

- Session/time: `rival-v2-natural-nexto-blue-20260822T173729Z-ec302968` at 73.100s
- Opponent/source: Nexto / natural_match
- Ranking score: 87.000
- Outcome: `opponent` next touch
- Why ranked: boost increased across a decision interval; distance to ball increased before the pickup; ETA possession proxy changed from favorable to unfavorable; opponent recorded the next touch

#### boos-6c46d0dff757

- Session/time: `rival-v2-natural-wisp-blue-20260822T174422Z-1d48f31f` at 381.817s
- Opponent/source: Wisp v2-75B / natural_match
- Ranking score: 87.000
- Outcome: `opponent` next touch
- Why ranked: boost increased across a decision interval; distance to ball increased before the pickup; ETA possession proxy changed from favorable to unfavorable; opponent recorded the next touch

#### boos-09e58e9dd11d

- Session/time: `rival-v2-natural-wisp-blue-20260822T175850Z-72963bbf` at 67.217s
- Opponent/source: Wisp v2-75B / natural_match
- Ranking score: 87.000
- Outcome: `opponent` next touch
- Why ranked: boost increased across a decision interval; distance to ball increased before the pickup; ETA possession proxy changed from favorable to unfavorable; opponent recorded the next touch

#### boos-0a02e17a02db

- Session/time: `rival-v2-natural-wisp-blue-20260822T174422Z-1d48f31f` at 272.458s
- Opponent/source: Wisp v2-75B / natural_match
- Ranking score: 86.278
- Outcome: `opponent` next touch
- Why ranked: boost increased across a decision interval; distance to ball increased before the pickup; ETA possession proxy changed from favorable to unfavorable; opponent recorded the next touch

### `apparent_vs_actual_challenge`

#### appa-2853d7379f43

- Session/time: `rival-v2-natural-nexto-blue-20260822T172339Z-79e6298a` at 90.542s
- Opponent/source: Nexto / natural_match
- Ranking score: 95.000
- Outcome: `opponent` next touch
- Why ranked: natural-match opponent closing threshold crossed; closing speed later fell below the abort reference; Rival jumped during the bounded response window

#### appa-1d8d681ee3a9

- Session/time: `rival-v2-natural-nexto-blue-20260822T172339Z-79e6298a` at 127.967s
- Opponent/source: Nexto / natural_match
- Ranking score: 95.000
- Outcome: `opponent` next touch
- Why ranked: natural-match opponent closing threshold crossed; closing speed later fell below the abort reference; Rival jumped during the bounded response window

#### appa-b91d6aa6af1e

- Session/time: `rival-v2-natural-nexto-blue-20260822T172339Z-79e6298a` at 136.033s
- Opponent/source: Nexto / natural_match
- Ranking score: 95.000
- Outcome: `opponent` next touch
- Why ranked: natural-match opponent closing threshold crossed; closing speed later fell below the abort reference; Rival jumped during the bounded response window

#### appa-f3ab975a4cb8

- Session/time: `rival-v2-natural-nexto-blue-20260822T172339Z-79e6298a` at 193.433s
- Opponent/source: Nexto / natural_match
- Ranking score: 95.000
- Outcome: `opponent` next touch
- Why ranked: natural-match opponent closing threshold crossed; closing speed later fell below the abort reference; Rival jumped during the bounded response window

#### appa-9b9f11fd7739

- Session/time: `rival-v2-natural-nexto-blue-20260822T172339Z-79e6298a` at 230.033s
- Opponent/source: Nexto / natural_match
- Ranking score: 95.000
- Outcome: `opponent` next touch
- Why ranked: natural-match opponent closing threshold crossed; closing speed later fell below the abort reference; Rival jumped during the bounded response window

## Integrity warnings

No loader integrity warnings.
