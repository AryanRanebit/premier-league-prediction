#!/usr/bin/env python3
"""Recency-weighting experiment: training-match decay half-life, GW19 official
cutoffs, ARM B only (chrono 80/20, best-val checkpoint), rich control features
(no European/high-recovery additions - isolates the weighting effect).

Pre-set grid before any 2026-27 inspection: 730d (current control), 540, 365,
240, 120. Selection criterion: backtest averages only, never where it places
specific clubs in the live forecast.

Same protocol as exp_euro_stylec_gw19.py: official GW19 cutoffs, 5 seeds,
4000 sims, rng 42, frozen-cutoff remainder NLL. The 730d arm re-runs here as
an in-harness control (expected to reproduce rich 1.598/1.864/3.034).

Usage: python3 exp_halflife_gw19.py HALF_LIFE_DAYS [SEASON...]
"""
import os, sys, json
import numpy as np
from datetime import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "euro_stylec")))
import exp_context_gw100 as C
import epl_nn as N
import exp_style_gw19 as S
import exp_euro_stylec_gw19 as E

OUT = C.OUT
OFFICIAL_GW19_END = S.OFFICIAL_GW19_END
N_SEEDS = 5
EPOCHS = 250

def run_season(s, hl, hist_map, X_all, YH_all, YA_all, idx_of):
    ALL = C.ALL
    sm = sorted([m for m in ALL if m["season"] == s], key=lambda x: x["date"])
    cd = dt.strptime(OFFICIAL_GW19_END[s], "%Y-%m-%d")
    cut = [m for m in sm if m["date"] <= cd]
    rest = [m for m in sm if m["date"] > cd]
    train = [m for m in ALL if m["date"] <= cd]
    tra, val = S.chrono_split(train)
    wa = np.array([0.5 ** (max((cd - m["date"]).days, 0) / hl) for m in tra])
    iia = [idx_of[id(m)] for m in tra]; iib = [idx_of[id(m)] for m in val]
    Xa, YHa, YAa = X_all[iia], YH_all[iia], YA_all[iia]
    Xb, YHb, YAb = X_all[iib], YH_all[iib], YA_all[iib]
    rem = N.remaining_fixtures(cut)
    Xf = np.array([E.fixture_x(hist_map, h, a, cd, s, "rich") for h, a in rem], dtype=object)
    Xa_s, Xb_s, meansB, stdsB = S.safe_fill_and_scale(Xa, Xb)
    Xf_sB = (np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]
                       for r in Xf], dtype=float) - meansB) / stdsB
    def scaled_feat(h, a):
        r = E.fixture_x(hist_map, h, a, cd, s, "rich")
        A = np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]], dtype=float)
        return (A - meansB) / stdsB
    netsB, best_epochs = [], []
    for sd in range(N_SEEDS):
        _, best = S.train_with_val(Xa_s, YHa, YAa, wa, Xb_s, YHb, YAb, sd, epochs=EPOCHS)
        netsB.append(S.net_from_state(Xa_s.shape[1], sd, best["state"]))
        best_epochs.append(best["epoch"])
    resB = S.evaluate(netsB, Xf_sB, rem, cut, train, rest, sm, scaled_feat)
    rec = {"season": s, "half_life_days": hl, "cutoff": str(cd.date()), "arm": "B_bestckpt80_only",
           "features": "rich control (no euro/highrec)", "best_val_epochs": best_epochs,
           "arm_B_bestckpt80": resB}
    json.dump(rec, open(os.path.join(OUT, f"halflife{hl}_gw19_{s}.json"), "w"), indent=1)
    print(f"hl{hl} {s}: big6 {resB['mae_big6']:.3f} | all {resB['mae_all']:.3f} | nll {resB['score_nll']:.3f} | epochs {best_epochs}", flush=True)

if __name__ == "__main__":
    hl = float(sys.argv[1])
    seasons = sys.argv[2:] or list(OFFICIAL_GW19_END)
    hist_map = N.build_history(C.ALL)
    X_all, YH_all, YA_all = E.make_xy(C.ALL, hist_map, "rich")
    idx_of = {id(m): i for i, m in enumerate(C.ALL)}
    for s in seasons:
        run_season(s, hl, hist_map, X_all, YH_all, YA_all, idx_of)
