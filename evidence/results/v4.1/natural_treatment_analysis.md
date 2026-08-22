# Natural-play analysis: treatment

Analyzer `rival-m04p1-natural-v1` processed **8** full natural matches and **43,158** policy decisions.

## Aggregate context

| Metric | Result |
| --- | ---: |
| Wins / losses / ties | 4 / 4 / 0 |
| Goals for / against | 33 / 33 |
| Goal differential | +0 |
| Favorable ETA share | 0.6243 |
| Possession-loss transitions | 365 |
| Goals conceded within the consequence window after loss | 36 |
| Adjustment applied decisions | 17 |
| Applied next touch self / opponent / none | 1 / 15 / 1 |

Scores are context only; natural trajectories are not paired skill evidence.

## Ranked recurring patterns

| Pattern | Episodes | Matches | Opponent next touch | Conceded next goal | Priority |
| --- | ---: | ---: | ---: | ---: | ---: |
| `apparent_pressure_release_after_closing_abort` | 507 | 8 | 212 (0.418) | 44 (0.087) | 7.352 |
| `low_resource_aerial_commitment` | 96 | 8 | 46 (0.479) | 12 (0.125) | 5.076 |
| `boost_pickup_with_eta_possession_flip` | 75 | 8 | 14 (0.187) | 4 (0.053) | 3.275 |

The priority score combines log-scaled cross-match frequency with opponent-next-touch and near-term-concession rates. It ranks what to inspect; it is not a causal effect estimate.

## Session ledger

| # | Opponent | Rival side | Score | Decisions | Raw SHA-256 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | nexto | blue | 5-4 | 5,590 | `78e7fab87ff600eeba9410ac5ffa5064301736db4c122c6de672d764f8772d41` |
| 2 | wisp | orange | 5-3 | 5,370 | `6160f95ad6d49b539cab29ba2ba7f50d580c0a3e0737017afb52e83eb26fcdd3` |
| 3 | nexto | orange | 4-3 | 5,353 | `952d051a127be2f84b0da10e90dc24503441eeb867cd0b731e0ae693d81f1e6d` |
| 4 | wisp | blue | 6-5 | 5,602 | `59cf0556654894bc897adeafc2de2032b2180fe8327b490bcd6a1e96f1397723` |
| 5 | nexto | blue | 5-2 | 5,252 | `ce5c8f454aee9a6de5e58b915b2451174f082dbd2c2a13b734bd31a6c63fb3d0` |
| 6 | wisp | orange | 5-3 | 5,363 | `4ab5b97e988a68e3bd6f1e067c123ed25ffaa7250c471017c9ed628aaea84da9` |
| 7 | nexto | orange | 2-3 | 5,068 | `8532289405ee47f90681ef5641ee124824f21fc4ac1cde4d1aa5b57d036e0e8d` |
| 8 | wisp | blue | 5-6 | 5,560 | `a6df6dad0d626aa6f683afa30e97ca2b793505014ab2ae5649b954fb5e93d513` |

All raw hashes matched the compact batch manifest: **true**.
