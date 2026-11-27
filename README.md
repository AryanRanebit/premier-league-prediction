# premier-league-prediction

Predicting Premier League finishing positions from match data, two ways:
a classic **Dixon-Coles** Poisson model and a small **neural network** trained from
scratch (numpy only, hand-written backprop). Built for the Query Ka Scene YouTube
series. Everything is reproducible from public data (see Reproduce; about six
minutes measured on a laptop, most of it the two neural-network backtests).

## Results at a glance

Backtest: five completed seasons (2021-22 to 2025-26), each frozen after its first
100 matches - the same point of the season the current 2026-27 forecast stands at.
The model only sees what it could really have seen at the time.

| Model | Big Six mean position error | All-team mean position error | Scoreline log loss |
|---|---|---|---|
| Dixon-Coles (goals only) | **2.49** | 2.55 | **3.04** |
| Neural net (goals features) | 2.95 | 2.53 | 3.09 |
| Neural net (rich features) | 2.56 | **2.42** | 3.07 |

How to read this, honestly:

- **Scoreline log loss** (lower = better probabilistic score predictions): the
  Dixon-Coles model was lower than both neural variants in four of the five test
  seasons (both neural nets edged it in 2022-23). This is a prediction-loss
  comparison, not a calibration study.
- **Position error**: not clearly separated - five test seasons cannot establish a
  clear advantage for any model; the gaps sit well inside season-to-season
  differences within them. The simple model edges the Big Six; the rich neural net
  edges the all-team average.
- Validation curves showed the neural nets overfitting: training loss kept falling
  while validation loss bottomed within the first ~20 epochs and then rose. Both
  neural variants now train with early stopping on a held-out recent slice (the
  most recent 10% of training matches by date) and restore their best checkpoint.
  On the same five backtest seasons this cut held-out scoreline log loss from 3.63
  to 3.07 (rich) and 3.39 to 3.09 (goals), with position errors unchanged to
  slightly better (Big Six 2.75 to 2.56 rich; all-team 2.53 to 2.42).
- Some clubs are simply harder to predict: across both neural variants, Tottenham's
  average miss was 6.9 places and Chelsea's 3.5, versus 1.1 for Manchester City and
  0.8 for Liverpool. Chelsea is volatile in both directions - the models miss them
  high and low.

## Chelsea 2026-27 (forecasts as of 8 Nov 2026, 20,000 simulations)

All conditional on model assumptions:

| Model | Expected finish | Expected points | Top-4 chance | Title chance |
|---|---|---|---|---|
| Dixon-Coles | 6.2 | 60.0 | 26% | 0.1% |
| Neural net (rich) | 6.5 | 60.5 | 29% | 1.8% |
| Neural net (goals) | 5.8 | 61.6 | 39% | 0.6% |

All three say the same thing: the chasing pack, around 60 points, a top-four race
rather than a title race.

## Reproduce

```bash
pip install -r requirements.txt
./reproduce.sh        # fetch data, run both backtests + forecasts, run tests
```

Or step by step: see `reproduce.sh`. Tests: `python3 tests/test_pipeline.py`
(41 checks: data integrity, no same-day leakage, points/conservation and tie-break
invariants, reproducibility).

Data: football-data.co.uk (publicly accessible CSVs; reuse terms unresolved -
attribution given, CSVs not redistributed; see `DATA.md`).

## Method notes

- Dixon-Coles: attack/defence ratings per team, home advantage, low-score rho
  adjustment, exponential time decay (730-day half-life). Current-season weight
  boost 1.0 everywhere (identical to backtests).
- Neural net: 2 hidden layers (48, 24, ReLU), two 7-way softmax heads
  (goals 0-5/6+ per side), cross-entropy loss, bias-corrected Adam, 5-seed ensemble.
  Rolling last-5/last-10 features from strictly earlier dates only (no same-day
  leakage), preprocessing fitted on the training split only. Backtest training uses
  validation-based early stopping (most recent 10% of training matches by date held
  out, minimum 30 epochs, patience 20, best checkpoint restored).
- Season simulation: fixture probabilities computed once at the cutoff and frozen;
  6+ cells expanded from the empirical training tail. Positions decided by points,
  then goal difference, then goals for; dead-level ties fall back to alphabetical
  order (the real league uses head-to-head records) - an unquantified
  simplification.
- Scoreline log loss compares all models on identical merged 0-5/6+ cells over the
  same 1,400 held-out matches, with frozen-cutoff information for every model.
- Limitations: the neural net sees fixed rolling windows while Dixon-Coles sees
  exponentially weighted full history (a practical pipeline comparison, not an
  architecture isolation); position-error differences are within season-to-season
  variation; forecasts are conditional on model assumptions and the 8 Nov 2026
  snapshot. The frozen 2026-27 forecast and next-9 checkpoint were produced before
  early stopping was added and are intentionally unchanged.
