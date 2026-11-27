#!/usr/bin/env python3
"""LOCKED evaluation for the autoresearch-style NN experiments. Experiments must not edit this file.

Metric: mean scoreline NLL (lower is better) of the NN on the matches after the
100-match cutoff of each test season, with information frozen at the cutoff.
This is the same quantity as score_nll in src/epl_nn.py. The two-head model,
features, cutoffs and early-stopping mechanics come from src/epl_nn.py unchanged.

Selection uses TUNE seasons only (2021-22 to 2024-25). 2025-26 is reported
separately. It was already used in earlier reports and experiments, so it is NOT
a clean holdout. The only untouched test is the frozen 2026-27 checkpoint protocol
under experiments/ (not scored here).
"""
import math, os, sys, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import epl_nn as E

TUNE = ["2122", "2223", "2324", "2425"]
SEEN = ["2526"]

def season_nll(s, cfg, ALL, hist_map, X_all, YH_all, YA_all):
    sm = sorted([m for m in ALL if m["season"] == s], key=lambda x: x["date"])
    cut = sm[:100]; cutoff = cut[-1]["date"]
    ok = lambda m: m["date"] <= cutoff and not (m["season"] == s and m not in cut)
    idx = [i for i, m in enumerate(ALL) if ok(m)]
    train = [ALL[i] for i in idx]
    Xtr, YH, YA = X_all[idx], YH_all[idx], YA_all[idx]
    w = np.array([0.5 ** (max((cutoff - m["date"]).days, 0) / cfg["half_life"]) for m in train])
    rest = sm[100:]
    pairs = sorted({(m["home"], m["away"]) for m in rest})
    Xf = np.array([E.features_for(hist_map, h, cutoff, "home", cfg["rich"], inclusive=True) +
                   E.features_for(hist_map, a, cutoff, "away", cfg["rich"], inclusive=True)
                   for h, a in pairs], dtype=object)
    Xtr_s, Xf_s, _, _ = E.fill_and_scale(Xtr, Xf)
    E.Net = functools.partial(_Net, h1=cfg["h1"], h2=cfg["h2"])
    dates = [m["date"] for m in train]
    nets = [E.train_net(Xtr_s, YH, YA, w, seed=k, epochs=cfg["epochs"], bs=cfg["bs"], lr=cfg["lr"],
                        l2=cfg["l2"], val_frac=cfg["val_frac"], min_epochs=cfg["min_epochs"],
                        patience=cfg["patience"], dates=dates)[0] for k in range(cfg["n_seeds"])]
    pmf = dict(zip(pairs, E.predict_pmf(nets, Xf_s)))
    ll = [-math.log(max(pmf[(m["home"], m["away"])][min(m["hg"], 6), min(m["ag"], 6)], 1e-9)) for m in rest]
    return sum(ll) / len(ll)

_Net = E.Net

def evaluate(cfg):
    E.Net = _Net
    ALL = E.load_all()
    hist_map = E.build_history(ALL)
    X_all, YH_all, YA_all, _ = E.make_xy(ALL, hist_map, cfg["rich"])
    out = {s: season_nll(s, cfg, ALL, hist_map, X_all, YH_all, YA_all) for s in TUNE + SEEN}
    out["tune_mean"] = float(np.mean([out[s] for s in TUNE]))
    out["seen_2526"] = out[SEEN[0]]
    out["five_mean"] = float(np.mean([out[s] for s in TUNE + SEEN]))
    return out

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from candidate import CFG
    t = time.time(); r = evaluate(CFG)
    print({k: round(v, 4) for k, v in r.items()}, f"{time.time()-t:.0f}s")
