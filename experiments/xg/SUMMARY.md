# Experiment: rolling xG features (arm B, GW19 cutoffs)

Date: 2026-11-21. User-authorized local experiment after the FotMob xG source
check. Protocol identical to the primary matched baseline: arm B (chrono 80/20
early stopping, best val checkpoint), official GW19 cutoffs, seasons
2021-22..2025-26, 5 seeds, 730d decay, 4000 sims, rng 42, same evaluation code.

## Data

- Source: FotMob matchDetails API, stats-block "Expected goals (xG)" per match.
- Coverage census (not sampled): all 2,380 football-data PL matches
  2020-21..2026-27-cutoff matched 2,380/2,380 by date+score+team alias.
  Zero missing. The abandoned Bournemouth-Luton match of 2023-12-16
  (coverage "lower", no xG) is excluded - the replayed fixture is the one in
  football-data, matched with xG.
- Verified boundary: FotMob match-level xG starts 2020-21 (2019-20 and
  earlier are ratings-only, directly checked).
- pl_xg_matched.csv: season, date, home, away, xg_home, xg_away, fotmob_id.
- xG provider/model identity unverified; do not mix with another xG source.

## Fair-isolation design

xG does not exist before 2020-21, so BOTH arms train on the identical
shortened history (2020-21 onward). Comparing xG-short-history against
rich-full-history would conflate the feature with the training window.

- shortrich (control): rich features only, train 2020-21 onward.
- shortxg (test): rich + 12 rolling xG features (per team: xG for/against
  last-5, last-10, venue last-5; strictly pre-match; teams' xG windows only
  contain xG-bearing matches), same training window.
- Context only: full-history rich baseline 1.598 / 1.864 / 3.034.

## Results (5-season averages, arm B)

| arm | big6 MAE | all-20 MAE | NLL |
|-----|---------|-----------|-----|
| shortrich (control) | 1.551 | 1.830 | 3.061 |
| shortxg             | 1.514 | 1.791 | 3.068 |
| delta               | -0.037 | -0.040 | +0.007 |

Per season (shortrich -> shortxg):
| season | big6 | all20 | NLL |
|--------|------|-------|-----|
| 2122 | 0.851 -> 0.794 | 1.603 -> 1.840 | 3.089 -> 3.107 |
| 2223 | 0.977 -> 0.894 | 1.738 -> 1.666 | 2.973 -> 2.953 |
| 2324 | 1.998 -> 1.876 | 1.938 -> 1.750 | 3.308 -> 3.319 |
| 2425 | 2.070 -> 2.228 | 1.758 -> 1.709 | 3.000 -> 3.032 |
| 2526 | 1.858 -> 1.780 | 2.115 -> 1.988 | 2.936 -> 2.931 |

## Reading

- Position MAE: xG arm better in 4 of 5 seasons (all but 2122), average
  -0.04 places all-20. Scoreline NLL: worse in 3 of 5, average +0.007.
- The training-window effect is as large as the feature effect: the
  short-history CONTROL (1.551/1.830/3.061) already differs from the
  full-history baseline (1.598/1.864/3.034) - better MAE, worse NLL.
- Verdict: small, mixed, inconsistent in sign across metrics. Same reading
  as every previous feature experiment this morning: NOT an improvement
  claim on five reused seasons. Not adopted; nothing pushed anywhere.

## Artifacts

- Harness: exp_xg_gw19.py; per-season JSONs ctx_results/xg_{shortrich,shortxg}_gw19_*.json (10)
- Raw download: xg_download/pl_xg_raw.csv (2,380 rows + downloader with resume)
- Matched data: xg_download/pl_xg_matched.csv (2,380 rows)
