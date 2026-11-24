#!/usr/bin/env bash
# Reproduce every published number from raw data. Takes ~5 minutes.
set -e
python3 scripts/fetch_data.py          # 1. download CSVs (see DATA.md)
python3 src/epl_model.py backtest > results/backtest.jsonl   # 2. DC baseline backtest
python3 src/dc_nll.py                  # 3. DC scoreline NLL -> results/dc_nll.json
python3 src/dc_forecast.py             # 4. DC current forecast -> results/forecast_full_v3.json
python3 src/epl_nn.py rich             # 5. NN backtests -> results/nn_backtest_*_v3.json
python3 src/epl_nn.py goals
python3 src/nn_forecast.py rich        # 6. NN current forecasts -> results/nn_forecast_*_v3.json
python3 src/nn_forecast.py goals
python3 tests/test_pipeline.py         # 7. sanity checks (~15s)
