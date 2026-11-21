# 2026-27 final-table forecast after 10 games per club (experimental)

Generated 2026-11-21 ~09:38 IST. Cutoff: 2026-11-08 (first 100 PL matches; every
club exactly 10 played). GW11 kicks off later today, so this is the state before it.

These are forecasts, not accuracy scores. Nothing has been adopted; the production
forecast and GitHub are unchanged.

## What was run

- New NN forecasts, two variants, identical protocol to the matched backtest arm B:
  chrono 80/20 early stopping (best validation checkpoint, max 250 epochs), 5 seeds,
  730d decay, 4000 sims, rng 42, 6+ tail from training data only.
  - `rich` = matched baseline (repo rich feature set only).
  - `euroC` = experimental candidate (rich + European-inclusive workload + lagged
    FotMob poss-won-final-3rd season aggregate).
- Training data: all PL matches 2015-16 through the cutoff (incl. the 100 played
  2026-27 matches), from the repo snapshot at commit 89c4838.
- Remaining 280 fixtures with announced dates: FotMob PL 2026-27 fixtures JSON
  fetched 2026-11-21 (https://www.fotmob.com/api/data/leagues?id=47&ccode3=IND&season=2026/2027).
  Cross-check: FotMob's 100 played fixtures match the football-data snapshot
  100/100 on home/away/date.
- Best-val epochs: rich [4,6,7,8,6], euroC [6,4,6,7,7].

## Future-feature assumptions (stated per instructions)

- Rich form features: frozen inclusive at the cutoff (repo protocol); form does
  not update with simulated results.
- euroC workload (days since last match, matches in last 14 days): computed per
  future fixture from the KNOWN CALENDAR ONLY - played PL + played Europe +
  announced future PL + announced future European league-phase dates.
  NOT counted: unscheduled European knockout rounds (draws not made, so spring
  congestion for advancing clubs is understated), domestic cups, future
  postponements. No past-14-day load is held constant; load moves with the
  announced calendar.
- High recovery: 2025-26 full-season aggregate, frozen all season (lagged by
  design); the 3 promoted clubs (Coventry, Hull, Ipswich) have no 2025-26 PL
  value and are training-mean filled. Coverage 17/20.

## New forecast: euroC (experimental)

| exp pos | club | exp pts | p(champion) | p(top4) | p(relegation) | pos 10-90% |
|---|---|---|---|---|---|---|
| 2.28 | Arsenal | 75.3 | 0.386 | 0.92 | 0.000 | 1-4 |
| 2.30 | Brighton | 74.4 | 0.348 | 0.93 | 0.000 | 1-4 |
| 2.93 | Man City | 71.9 | 0.199 | 0.85 | 0.000 | 1-5 |
| 5.36 | Bournemouth | 63.4 | 0.030 | 0.39 | 0.001 | 3-9 |
| 5.43 | Chelsea | 63.3 | 0.018 | 0.38 | 0.000 | 3-9 |
| 6.26 | Liverpool | 61.2 | 0.012 | 0.27 | 0.001 | 3-10 |
| 7.41 | Tottenham | 58.0 | 0.004 | 0.15 | 0.002 | 4-11 |
| 9.78 | Everton | 53.0 | 0.001 | 0.04 | 0.013 | 6-14 |
| 10.31 | Man United | 51.7 | 0.001 | 0.03 | 0.021 | 6-15 |
| 10.96 | Leeds | 50.4 | 0.000 | 0.02 | 0.032 | 7-16 |
| 11.07 | Nott'm Forest | 50.1 | 0.000 | 0.02 | 0.037 | 7-16 |
| 11.60 | Fulham | 49.1 | 0.001 | 0.01 | 0.049 | 7-16 |
| 12.55 | Brentford | 47.1 | 0.001 | 0.01 | 0.075 | 8-17 |
| 13.98 | Aston Villa | 44.6 | 0.000 | 0.00 | 0.166 | 9-18 |
| 14.32 | Newcastle | 43.9 | 0.000 | 0.00 | 0.189 | 10-18 |
| 15.03 | Ipswich | 42.5 | 0.000 | 0.00 | 0.243 | 11-19 |
| 16.27 | Hull | 39.5 | 0.000 | 0.00 | 0.399 | 12-20 |
| 16.35 | Sunderland | 39.6 | 0.000 | 0.00 | 0.414 | 12-20 |
| 17.04 | Coventry | 37.8 | 0.000 | 0.00 | 0.538 | 13-20 |
| 18.77 | Crystal Palace | 32.6 | 0.000 | 0.00 | 0.821 | 16-20 |

## New vs previous saved forecast (DC v3, asof 2026-11-19, 20000 sims, commit 89c4838)

Both see exactly the same played results (the 100 matches through 2026-11-08;
no PL matches Nov 9-19). All differences are model/settings, not data window.

Biggest expected-position movers, euroC vs DC v3 (negative = better than before):
- Aston Villa 8.57 -> 13.98 (+5.41), Newcastle 9.98 -> 14.32 (+4.34),
  Brentford 8.48 -> 12.55 (+4.07), Liverpool 3.32 -> 6.26 (+2.94),
  Crystal Palace 16.14 -> 18.77 (+2.63), Hull 13.84 -> 16.27 (+2.43)
- Bournemouth 9.72 -> 5.36 (-4.36), Ipswich 18.54 -> 15.03 (-3.51),
  Tottenham 10.30 -> 7.41 (-2.89), Nott'm Forest 13.51 -> 11.07 (-2.44),
  Fulham 13.89 -> 11.60 (-2.30), Brighton 4.59 -> 2.30 (-2.29)
- Title race: DC v3 had Arsenal 1.50 exp pos / 61% champion. euroC: Arsenal
  2.28 / 38.6%, Brighton 2.30 / 34.8%, Man City 2.93 / 19.9% - a genuine
  three-way race, driven by Brighton's 22 pts and league-best +19 GD after 10.

## Feature attribution (so model differences aren't credited to the new features)

euroC vs the matched rich baseline is small everywhere: max expected-position
shift 0.88 (Man United 9.43 -> 10.31), most clubs within +/-0.5. The big moves
vs the previous forecast are almost entirely model-family/settings differences
(DC ratings vs NN with form frozen at cutoff, early stopping, 4000 vs 20000
sims), not the European/high-recovery features. This matches the backtest,
where the same features were a wash-to-marginal change.

Settings differences old vs new: Dixon-Coles ratings (slow-adapting, 20000
sims, boost=1.0) vs NN rich features with form frozen at cutoff, arm-B early
stopping, 5 seeds, 4000 sims; euroC additionally adds the two feature groups.

## Caveats

- Forecasts, not accuracy claims; judgeable only once the season ends.
- Same known seasons in training; no fresh-holdout claim.
- Data artifacts remain local research copies (free-to-view, no reuse license).
- PPDA still unverified (Understat maintenance; retry due 11:30 IST today).

## Artifacts

- forecast_2627_gw10.py (harness)
- ctx_results/forecast_2627_gw10_rich.json, forecast_2627_gw10_euroC.json
  (full per-club distributions: pos_dist, exp pts, p top4/champion/relegation)
- ctx_results/forecast_2627_gw10_comparison.json (per-club old/new comparison)
- pl_2627_remaining_fixtures.csv (280 announced fixtures used for the calendar)
