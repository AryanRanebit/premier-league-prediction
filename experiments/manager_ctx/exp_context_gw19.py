#!/usr/bin/env python3
"""Context features x early stopping, at the OFFICIAL GW19 cutoffs used by
exp_earlystop_gw19.py. Arms per season:
  A epoch250   : scaler on full train, 5 seeds x 250 epochs, final weights (repo recipe)
  B bestckpt80 : chronological 80/20 split, best-validation checkpoint per seed
Same cutoffs, seeds, sims (4000, rng 42), 730d decay, frozen-cutoff remainder NLL as
the attached earlystop bundle. Feature set = rich + manager/congestion ctx features.
Compare against attached results/earlystop_gw19_rich.json (same arms, rich features).

Usage: python3 exp_context_gw19.py SEASON [SEASON...]   (appends per-season JSON)
"""
import os, sys, json, math
import numpy as np
from datetime import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "euro_stylec")))
import exp_context_gw100 as C   # reuses ALL, ctx_feats, make_xy_ctx, fixture_x
import epl_nn as N

OUT = C.OUT
OFFICIAL_GW19_END = {"2122": "2021-12-27", "2223": "2023-01-05", "2324": "2023-12-28",
                     "2425": "2025-01-01", "2526": "2026-01-01"}
BIG6 = N.BIG6
EPOCHS = 250
N_SEEDS = 5
PARAMS = ["W1", "b1", "W2", "b2", "Wh", "bh", "Wa", "ba"]

def plain_nll(net, X, YH, YA):
    n = len(X)
    _, _, lh, la = net.forward(X)
    ph, pa = N.softmax(lh), N.softmax(la)
    return float(-(np.log(np.clip(ph[np.arange(n), YH], 1e-9, None))).sum() / n
                 -(np.log(np.clip(pa[np.arange(n), YA], 1e-9, None))).sum() / n)

def chrono_split(train):
    tr = sorted(train, key=lambda m: m["date"])
    dates = sorted({m["date"] for m in tr})
    target = int(len(tr) * 0.8)
    acc, split_date = 0, dates[0]
    for d in dates:
        cnt = sum(1 for m in tr if m["date"] == d)
        if acc + cnt >= target:
            split_date = d
            break
        acc += cnt
    a = [m for m in tr if m["date"] < split_date]
    b = [m for m in tr if m["date"] >= split_date]
    return a, b

def train_with_val(Xa, YHa, YAa, wa, Xb, YHb, YAb, seed, epochs=EPOCHS):
    net = N.Net(Xa.shape[1], seed=seed)
    m = {p: np.zeros_like(getattr(net, p)) for p in PARAMS}
    v = {p: np.zeros_like(getattr(net, p)) for p in PARAMS}
    rng = np.random.default_rng(seed + 1000)
    n = len(Xa); t = 0
    best = {"val": float("inf"), "epoch": -1, "state": None}
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, 256):
            idx = perm[s:s + 256]
            xb, yh, ya, wb = Xa[idx], YHa[idx], YAa[idx], wa[idx]
            z1, z2, lh, la = net.forward(xb)
            ph, pa = N.softmax(lh), N.softmax(la)
            bs_w = wb.sum() + 1e-9
            dh = ph; dh[np.arange(len(idx)), yh] -= 1; dh = (dh * wb[:, None]) / bs_w
            da = pa; da[np.arange(len(idx)), ya] -= 1; da = (da * wb[:, None]) / bs_w
            g = {}
            g["Wh"] = z2.T @ dh + 2 * 1e-4 * net.Wh; g["bh"] = dh.sum(0)
            g["Wa"] = z2.T @ da + 2 * 1e-4 * net.Wa; g["ba"] = da.sum(0)
            dz2 = dh @ net.Wh.T + da @ net.Wa.T
            dz2[z2 <= 0] = 0
            g["W2"] = z1.T @ dz2 + 2 * 1e-4 * net.W2; g["b2"] = dz2.sum(0)
            dz1 = dz2 @ net.W2.T
            dz1[z1 <= 0] = 0
            g["W1"] = xb.T @ dz1 + 2 * 1e-4 * net.W1; g["b1"] = dz1.sum(0)
            t += 1
            bc1 = 1 - 0.9 ** t
            bc2 = 1 - 0.999 ** t
            for p in PARAMS:
                m[p] = 0.9 * m[p] + 0.1 * g[p]
                v[p] = 0.999 * v[p] + 0.001 * g[p] ** 2
                setattr(net, p, getattr(net, p) - 5e-3 * (m[p] / bc1) / (np.sqrt(v[p] / bc2) + 1e-8))
        vl = plain_nll(net, Xb, YHb, YAb)
        if vl < best["val"]:
            best = {"val": vl, "epoch": ep, "state": {p: getattr(net, p).copy() for p in PARAMS}}
    return net, best

def net_from_state(d, seed, state):
    net = N.Net(d, seed=seed)
    for p in PARAMS:
        setattr(net, p, state[p])
    return net

def evaluate(nets, Xf_s, rem, cut, train, rest, full_season, feat_row_scaled):
    pmfs = N.predict_pmf(nets, Xf_s)
    tk, tp = N.tail_dist(train)
    teams, fin_pos, _ = N.simulate_nn(pmfs, rem, cut, 4000, np.random.default_rng(42), tk, tp)
    actual = N.final_table(full_season)
    exp_pos = {t: float(fin_pos[:, i].mean()) for i, t in enumerate(teams)}
    big6 = float(np.mean([abs(exp_pos[t] - actual[t]) for t in BIG6 if t in exp_pos]))
    allm = float(np.mean([abs(exp_pos[t] - actual[t]) for t in teams]))
    cache = {}
    ll = 0.0
    for m in rest:
        k = (m["home"], m["away"])
        if k not in cache:
            cache[k] = N.predict_pmf(nets, feat_row_scaled(*k))[0]
        ll += -math.log(max(cache[k][min(m["hg"], 6), min(m["ag"], 6)], 1e-9))
    return {"mae_big6": big6, "mae_all": allm, "score_nll": ll / len(rest)}

def scaled_row_factory(means, stds, hist_map, cd):
    def feat(h, a):
        r = C.fixture_x(hist_map, h, a, cd, True)
        A = np.array([[means[c] if r[c] is None else r[c] for c in range(len(means))]], dtype=float)
        return (A - means) / stds
    return feat

def run_season(s, hist_map, X_all, YH_all, YA_all, idx_of):
    ALL = C.ALL
    sm = sorted([m for m in ALL if m["season"] == s], key=lambda x: x["date"])
    cd = dt.strptime(OFFICIAL_GW19_END[s], "%Y-%m-%d")
    cut = [m for m in sm if m["date"] <= cd]
    rest = [m for m in sm if m["date"] > cd]
    train = [m for m in ALL if m["date"] <= cd]
    w = np.array([0.5 ** (max((cd - m["date"]).days, 0) / N.HALF_LIFE) for m in train])
    ii = [idx_of[id(m)] for m in train]
    Xtr, YHtr, YAtr = X_all[ii], YH_all[ii], YA_all[ii]
    rem = N.remaining_fixtures(cut)
    Xf = np.array([C.fixture_x(hist_map, h, a, cd, True) for h, a in rem], dtype=object)

    # arm A: full-train scaler, 250 epochs
    Xtr_s, Xf_s, meansA, stdsA = N.fill_and_scale(Xtr, Xf)
    netsA = [N.train_net(Xtr_s, YHtr, YAtr, w, seed=sd, epochs=EPOCHS)[0] for sd in range(N_SEEDS)]
    resA = evaluate(netsA, Xf_s, rem, cut, train, rest, sm, scaled_row_factory(meansA, stdsA, hist_map, cd))

    # arm B: chrono 80/20, best-val checkpoint
    tra, val = chrono_split(train)
    iia = [idx_of[id(m)] for m in tra]; iib = [idx_of[id(m)] for m in val]
    Xa, YHa, YAa = X_all[iia], YH_all[iia], YA_all[iia]
    Xb, YHb, YAb = X_all[iib], YH_all[iib], YA_all[iib]
    wa = np.array([0.5 ** (max((cd - m["date"]).days, 0) / N.HALF_LIFE) for m in tra])
    Xa_s, Xb_s, meansB, stdsB = N.fill_and_scale(Xa, Xb)
    Xf_sB = (np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]
                       for r in Xf], dtype=float) - meansB) / stdsB
    netsB, best_epochs = [], []
    for sd in range(N_SEEDS):
        _, best = train_with_val(Xa_s, YHa, YAa, wa, Xb_s, YHb, YAb, sd)
        netsB.append(net_from_state(Xa_s.shape[1], sd, best["state"]))
        best_epochs.append(best["epoch"])
    resB = evaluate(netsB, Xf_sB, rem, cut, train, rest, sm, scaled_row_factory(meansB, stdsB, hist_map, cd))

    rec = {"season": s, "variant": "ctx", "cutoff": str(cd.date()),
           "matches_in_cut": len(cut), "n_eval": len(rest),
           "best_val_epochs": best_epochs,
           "arm_A_epoch250": resA, "arm_B_bestckpt80": resB}
    path = os.path.join(OUT, f"ctx_gw19_{s}.json")
    json.dump(rec, open(path, "w"), indent=1)
    print(f"ctx {s}: best epochs {best_epochs} | big6 A {resA['mae_big6']:.2f} B {resB['mae_big6']:.2f} | "
          f"all A {resA['mae_all']:.2f} B {resB['mae_all']:.2f} | "
          f"nll A {resA['score_nll']:.3f} B {resB['score_nll']:.3f}", flush=True)

if __name__ == "__main__":
    seasons = sys.argv[1:] or list(OFFICIAL_GW19_END)
    hist_map = N.build_history(C.ALL)
    X_all, YH_all, YA_all = C.make_xy_ctx(C.ALL, hist_map, True)
    idx_of = {id(m): i for i, m in enumerate(C.ALL)}
    for s in seasons:
        run_season(s, hist_map, X_all, YH_all, YA_all, idx_of)
