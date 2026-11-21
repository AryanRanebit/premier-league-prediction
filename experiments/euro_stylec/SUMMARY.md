# Experiment: European workload + lagged high-recovery features (arm B, GW19 cutoffs)

Date: 2026-11-21. Protocol identical to primary baseline: arm B (early stopping,
best val checkpoint), official GW19 cutoffs, seasons 2021-22..2025-26, 5 seeds,
4000 sims, rng 42, same split/scaler. Metrics from the same evaluation code.

## Control check

Mode `rich` (no new features) reproduces the primary baseline averages exactly:
big6 MAE 1.598, all-20 MAE 1.864, NLL 3.034. Protocol fidelity confirmed.

## Modes

- rich: control, existing feature set only.
- euro: rich + European-inclusive workload features (days since last match and
  matches in last 7/14/28 days counting played European fixtures alongside PL;
  domestic cups excluded, awarded Spurs-Rennes 2021 not counted, all-English
  ties deduped).
- styleC: rich + prior-season FotMob "possession won final 3rd per match" per
  club (lagged full prior season only, no current/future-season leakage).
- euroC: rich + both.

## Results (5-season averages, arm B)

| mode | big6 MAE | all-20 MAE | NLL | dNLL vs rich | dAll20 vs rich | dBig6 vs rich |
|------|---------|-----------|-----|--------------|----------------|----------------|
| rich   | 1.598 | 1.864 | 3.034 | - | - | - |
| euro   | 1.620 | 1.942 | 3.027 | -0.007 | +0.077 | +0.022 |
| styleC | 1.608 | 1.857 | 3.033 | -0.002 | -0.008 | +0.009 |
| euroC  | 1.545 | 1.877 | 3.026 | -0.009 | +0.013 | -0.053 |

Per-season NLL (rich / euro / styleC / euroC):

| season | rich | euro | styleC | euroC |
|--------|------|------|--------|-------|
| 2122 | 3.022 | 3.006 | 3.019 | 2.960 |
| 2223 | 2.971 | 2.969 | 2.994 | 2.990 |
| 2324 | 3.249 | 3.278 | 3.258 | 3.276 |
| 2425 | 3.006 | 2.980 | 2.986 | 2.977 |
| 2526 | 2.924 | 2.902 | 2.906 | 2.925 |

## Reading

- Deltas are small and inconsistent in sign across seasons. euro alone is worse
  on all-20 MAE (+0.077) and on 2324 NLL (+0.029). styleC is within noise on
  every metric. euroC has the best average NLL (-0.009) and big6 MAE (-0.053)
  but is slightly worse on all-20 MAE (+0.013), and its NLL edge comes mostly
  from one season (2122, -0.062).
- On 5 backtest seasons this is not an improvement claim. The honest statement:
  neither European workload nor lagged high-recovery features produce a
  consistent, material gain over the current feature set at these cutoffs; the
  combined variant is marginally ahead on NLL/big6, behind on all-20, and would
  need more held-out seasons to say anything stronger.

## Coverage and data caveats

- High-recovery coverage: 17/20 PL teams per test season have a prior-season
  FotMob value; the 3 promoted clubs each season have no prior top-flight season
  and are mean-filled (None -> column mean), as designed.
- FotMob source: https://data.fotmob.com/stats/47/season/{seasonId}/poss_won_att_3rd_team.json
  (all 20 teams, 2016-17 onward; season aggregates, not per-match).
- European fixtures: 902 played club-matches for 19 PL clubs, 2015-16..2025-26
  (ESPN core API + FotMob verification); workload counts played matches only;
  PL+Europe is not full congestion (FA Cup / League Cup excluded).
- Per-match PPDA/pressures remain unverified: Understat down (maintenance page);
  retry scheduled 2026-11-21 11:30 IST.
- Backtest re-uses known seasons; no fresh-holdout claims. Data artifacts are
  local research copies (free-to-view sources; no reuse license established).

## Artifacts

- Harness: exp_euro_stylec_gw19.py
- Per-season JSONs: ctx_results/euro_{rich,euro,styleC,euroC}_gw19_{2122..2526}.json (20 files)
- Inputs: fotmob_poss_won_final3rd_by_season.csv (220 rows, 11 seasons),
  pl_clubs_european_fixtures_2015_2027.csv (938 rows)
