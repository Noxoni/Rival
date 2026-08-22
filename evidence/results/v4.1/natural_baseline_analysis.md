# Natural-play analysis: baseline

Analyzer `rival-m04p1-natural-v1` processed **8** full natural matches and **43,792** policy decisions.

## Aggregate context

| Metric | Result |
| --- | ---: |
| Wins / losses / ties | 4 / 4 / 0 |
| Goals for / against | 35 / 30 |
| Goal differential | +5 |
| Favorable ETA share | 0.6407 |
| Possession-loss transitions | 367 |
| Goals conceded within the consequence window after loss | 41 |
| Adjustment applied decisions | 0 |
| Applied next touch self / opponent / none | 0 / 0 / 0 |

Scores are context only; natural trajectories are not paired skill evidence.

## Ranked recurring patterns

| Pattern | Episodes | Matches | Opponent next touch | Conceded next goal | Priority |
| --- | ---: | ---: | ---: | ---: | ---: |
| `apparent_pressure_release_after_closing_abort` | 510 | 8 | 173 (0.339) | 42 (0.082) | 6.959 |
| `low_resource_aerial_commitment` | 104 | 8 | 38 (0.365) | 13 (0.125) | 4.923 |
| `boost_pickup_with_eta_possession_flip` | 66 | 8 | 15 (0.227) | 6 (0.091) | 3.539 |

The priority score combines log-scaled cross-match frequency with opponent-next-touch and near-term-concession rates. It ranks what to inspect; it is not a causal effect estimate.

## Session ledger

| # | Opponent | Rival side | Score | Decisions | Raw SHA-256 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | nexto | blue | 6-5 | 6,604 | `9734fa124d23a1e3ee681d41dc80cb102371f484bd0e7368307d291837695591` |
| 2 | wisp | orange | 5-4 | 5,402 | `a0b1773ad180eed82da49ae9795fb6e0127a952b11a60fdbcf2ecfa83686bf8b` |
| 3 | nexto | orange | 2-3 | 5,090 | `447abba2c2db15c254cb921f35ab375a149e6af5a0906cedb84a5f7309413cab` |
| 4 | wisp | blue | 8-4 | 5,607 | `0a56decd86fcb7fcb8cda01c76c1e9c6d65922ccc0a37849bd48d70dd7e10802` |
| 5 | nexto | blue | 2-3 | 5,089 | `56ccdd6beeca893215c41145009220769ac694231588fe0020f5c0e5a438a89a` |
| 6 | wisp | orange | 5-4 | 5,558 | `a1b2742bbf835bad9aa77494b5a36dcbe57987203f2aaa33268152d2d8c23a5d` |
| 7 | nexto | orange | 2-5 | 5,225 | `90ba2e821f32f9a13205453c38c2804f80d0f0f439ebab15894cdf2ad22982d9` |
| 8 | wisp | blue | 3-4 | 5,217 | `7e2c85b7e3e251b11b921551527df519fc1907d2cce33ab9a978a3ac966e5ea3` |

All raw hashes matched the compact batch manifest: **true**.
