"""The ONE file an experiment may change: neural-net settings for the backtest.

evaluate.py (locked) reads CFG. Change values here, then run
`python3 autoresearch_pl/run_experiment.py "<what you tried>"`.
"""
CFG = {
    'rich': True,
    'h1': 24,
    'h2': 12,
    'lr': 0.005,
    'l2': 0.003,
    'bs': 256,
    'val_frac': 0.1,
    'patience': 20,
    'min_epochs': 30,
    'epochs': 250,
    'n_seeds': 5,
    'half_life': 730.0,
}
