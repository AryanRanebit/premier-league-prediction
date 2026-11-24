# Season-end evaluation — frozen predictions and scoring protocol

Purpose: pre-registered comparison of fixed forecasts against the actual
2026-27 Premier League final table. No model changes after this point;
the rules below are fixed before the remaining results happen.

## Frozen snapshots (in this directory)

| File | Model | Data state | Sims | Role |
|---|---|---|---|---|
| `dc_forecast_full_v3.json` | Dixon-Coles (production), identical to `results/forecast_full_v3.json` | asof 2026-11-19 (= the first 100 PL matches; no PL matches were played 2026-11-08..11-19) | 20,000 | **Primary prediction** |
| `nn_forecast_rich_gw10.json` | NN, repo rich features | cutoff 2026-11-08 (same 100 matches) | 4,000, rng 42, 5 seeds | Challenger |
| `nn_forecast_euroc_gw10.json` | NN, rich + European workload + lagged high recovery | same cutoff | same | Challenger |

- There is **no xG forecast**. xG features were backtested only
  (`experiments/xg/`); do not treat the xG backtest as a season prediction.
- The points-spread / draw post-processing outputs
  (`experiments/postprocessing/`) are reference-fit **illustrations** and
  are **excluded** from this evaluation.
- Model selection caveat: the choice of DC as primary rests on comparisons
  over the same five reused validation seasons (2021-22..2025-26), where it
  had the strongest scoreline results; that is prior evidence, not a
  guarantee for 2026-27.

## Season end

The final announced Premier League matchday is **2027-05-30** (all 10
fixtures, per the announced schedule retrieved 2026-11-21). Subject to
postponements; evaluate once all 380 matches are final.

## Metrics (fixed definitions)

1. **Position MAE (primary table metric):** mean over the 20 clubs of
   |expected finish − actual finish|, using each snapshot's `exp_pos`.
   Also report the big-six subset for continuity with the backtests.
2. **Scoreline NLL (primary probability metric):** average negative
   log-probability each model assigned to the actual scoreline of the 280
   remaining fixtures (2026-11-09 onward). The frozen JSONs are table-level;
   this metric requires re-running the pinned code and data:
   - DC: `src/dc_forecast.py` at commit 89c4838 with the same data state.
   - NN: `experiments/forecast_2627_gw10/forecast_2627_gw10.py` (arm B,
     5 seeds, rng 42 — deterministic, reproduces the frozen table exactly).
3. **Secondary, report but don't optimize:** champion pick correctness
   (highest `p_champion` / `exp_pos` 1), relegation picks, mean absolute
   error of `exp_pts` vs actual points, per-rank points comparison.

Report BOTH metric families. A single "accuracy %" is not meaningful for
this kind of forecast.

## Rules

- No refitting, reseeding, or feature changes between now and scoring.
- If a snapshot must be regenerated for metric 2, the pinned code/data and
  the frozen table JSON are the check: regeneration must reproduce the
  frozen `exp_pos` values exactly, otherwise the run is invalid.
- Actual final table source: football-data.co.uk E0_2627 CSV (same source
  as training data), cross-checked against the official PL table. Note any
  points deductions when comparing `exp_pts`.
