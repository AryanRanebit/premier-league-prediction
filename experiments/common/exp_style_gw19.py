#!/usr/bin/env python3
"""Tactical-style features experiment, GW19 official cutoffs, arms A (250 epochs) and
B (best-validation checkpoint, the PRIMARY arm since 250-epoch overfit is known).

Feature modes (added to the repo's rich feature set):
  styleA : prior-PL-season possession% per team (SEASON-AGGREGATE PRIOR STYLE).
           Source: FBref squad possession tables 2020-21..2025-26 (fetched 21 Nov
           2026, raw markdown in bundle fb_<season>.md, parsed to
           fbref_possession.json). For a match in season N the value is the FULL
           season N-1 aggregate - completed before season N starts, so causal.
           No current-season aggregate is ever used. Newly promoted teams (not in
           prior PL season) get None -> training-mean fill (existing mechanism).
  styleB : PRE-MATCH ROLLING STYLE proxies from football-data match stats only:
           rolling-10 shots share, SOT share, corners share (for / (for+against)).
           Dominance/style proxies derived from data the model already sees as
           raw counts; labeled proxies, not measured possession/pressing.
  styleAB: both sets.

Same protocol as exp_earlystop_gw19.py / exp_context_gw19.py: official GW19
cutoffs, 5 seeds, 730d decay, 4000 sims, rng 42, frozen-cutoff remainder NLL.

Usage: python3 exp_style_gw19.py MODE SEASON [SEASON...]
"""
import os, sys, json, math, bisect
import numpy as np
from datetime import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_context_gw100 as C   # ALL matches, stint/event infrastructure
import epl_nn as N

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = C.OUT
_pp = os.path.join(HERE, "fbref_possession.json")
POSS = json.load(open(_pp)) if os.path.exists(_pp) else {}
if not POSS:
    print("WARNING: fbref_possession.json missing - styleA modes need it (compiled FBref squad possession, see experiments/README.md).", flush=True)

OFFICIAL_GW19_END = {"2122": "2021-12-27", "2223": "2023-01-05", "2324": "2023-12-28",
                     "2425": "2025-01-01", "2526": "2026-01-01"}
BIG6 = N.BIG6
EPOCHS = 250
N_SEEDS = 5
PARAMS = ["W1", "b1", "W2", "b2", "Wh", "bh", "Wa", "ba"]

def prior_poss(team, season):
    return POSS.get(str(int(season) - 101), {}).get(team)

def rolling_share(rows, fkey, akey, n=10):
    r = rows[-n:]
    if not r: return None
    f = sum(v[fkey] for v in r); a = sum(v[akey] for v in r)
    return f / (f + a) if (f + a) > 0 else None

def style_feats(hist_map, team, d, season, venue, mode, inclusive=False):
    rows = [v for (dd, v) in hist_map.get(team, []) if dd <= d] if inclusive \
        else [v for (dd, v) in hist_map.get(team, []) if dd < d]
    out = []
    if mode in ("styleA", "styleAB"):
        out.append(prior_poss(team, season))
    if mode in ("styleB", "styleAB"):
        out += [rolling_share(rows, "sf", "sa"),
                rolling_share(rows, "stf", "sta"),
                rolling_share(rows, "cf", "ca")]
    return out

def make_xy_style(matches, hist_map, mode):
    X, YH, YA = [], [], []
    for m in matches:
        fh = N.features_for(hist_map, m["home"], m["date"], "home", True) + \
             style_feats(hist_map, m["home"], m["date"], m["season"], "home", mode)
        fa = N.features_for(hist_map, m["away"], m["date"], "away", True) + \
             style_feats(hist_map, m["away"], m["date"], m["season"], "away", mode)
        X.append(fh + fa)
        YH.append(min(m["hg"], 6)); YA.append(min(m["ag"], 6))
    return np.array(X, dtype=object), np.array(YH), np.array(YA)

def fixture_x_style(hist_map, h, a, cutoff, season, mode):
    fh = N.features_for(hist_map, h, cutoff, "home", True, inclusive=True) + \
         style_feats(hist_map, h, cutoff, season, "home", mode, inclusive=True)
    fa = N.features_for(hist_map, a, cutoff, "away", True, inclusive=True) + \
         style_feats(hist_map, a, cutoff, season, "away", mode, inclusive=True)
    return fh + fa

# ---- training helpers identical to exp_context_gw19.py ----
def safe_fill_and_scale(Xtr, Xte):
    """fill_and_scale, but a column with zero non-None training values gets
    mean 0 / std 1 (neutral fill) instead of NaN. Guard for partial-coverage
    features (possession starts at 2018-19 matches; an 80% chrono split can
    leave a train part whose possession column is entirely None)."""
    a, b, means, stds = N.fill_and_scale(Xtr, Xte)
    bad = ~np.isfinite(means) | ~np.isfinite(stds) | (stds <= 0)
    if bad.any():
        means[bad] = 0.0; stds[bad] = 1.0
        a = np.where(np.isnan(a), 0.0, a); b = np.where(np.isnan(b), 0.0, b)
    return a, b, means, stds

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

def run_season(s, mode, hist_map, X_all, YH_all, YA_all, idx_of):
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
    Xf = np.array([fixture_x_style(hist_map, h, a, cd, s, mode) for h, a in rem], dtype=object)

    def scaled(means, stds):
        def feat(h, a):
            r = fixture_x_style(hist_map, h, a, cd, s, mode)
            A = np.array([[means[c] if r[c] is None else r[c] for c in range(len(means))]], dtype=float)
            return (A - means) / stds
        return feat

    # arm A: full-train scaler, 250 epochs
    Xtr_s, Xf_s, meansA, stdsA = safe_fill_and_scale(Xtr, Xf)
    netsA = [N.train_net(Xtr_s, YHtr, YAtr, w, seed=sd, epochs=EPOCHS)[0] for sd in range(N_SEEDS)]
    resA = evaluate(netsA, Xf_s, rem, cut, train, rest, sm, scaled(meansA, stdsA))

    # arm B: chrono 80/20, best-val checkpoint (PRIMARY)
    tra, val = chrono_split(train)
    iia = [idx_of[id(m)] for m in tra]; iib = [idx_of[id(m)] for m in val]
    Xa, YHa, YAa = X_all[iia], YH_all[iia], YA_all[iia]
    Xb, YHb, YAb = X_all[iib], YH_all[iib], YA_all[iib]
    wa = np.array([0.5 ** (max((cd - m["date"]).days, 0) / N.HALF_LIFE) for m in tra])
    Xa_s, Xb_s, meansB, stdsB = safe_fill_and_scale(Xa, Xb)
    Xf_sB = (np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]
                       for r in Xf], dtype=float) - meansB) / stdsB
    netsB, best_epochs = [], []
    for sd in range(N_SEEDS):
        _, best = train_with_val(Xa_s, YHa, YAa, wa, Xb_s, YHb, YAb, sd)
        netsB.append(net_from_state(Xa_s.shape[1], sd, best["state"]))
        best_epochs.append(best["epoch"])
    resB = evaluate(netsB, Xf_sB, rem, cut, train, rest, sm, scaled(meansB, stdsB))

    # coverage audit: how many of the 20 teams had prior-season possession
    teams_s = sorted({m["home"] for m in sm} | {m["away"] for m in sm})
    cov = sum(1 for t in teams_s if prior_poss(t, s) is not None)
    rec = {"season": s, "mode": mode, "cutoff": str(cd.date()),
           "possession_coverage": f"{cov}/20",
           "best_val_epochs": best_epochs,
           "arm_A_epoch250": resA, "arm_B_bestckpt80": resB}
    json.dump(rec, open(os.path.join(OUT, f"{mode}_gw19_{s}.json"), "w"), indent=1)
    print(f"{mode} {s}: poss-cov {cov}/20 | big6 A {resA['mae_big6']:.2f} B {resB['mae_big6']:.2f} | "
          f"all A {resA['mae_all']:.2f} B {resB['mae_all']:.2f} | "
          f"nll A {resA['score_nll']:.3f} B {resB['score_nll']:.3f}", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1]
    seasons = sys.argv[2:] or list(OFFICIAL_GW19_END)
    hist_map = N.build_history(C.ALL)
    X_all, YH_all, YA_all = make_xy_style(C.ALL, hist_map, mode)
    idx_of = {id(m): i for i, m in enumerate(C.ALL)}
    for s in seasons:
        run_season(s, mode, hist_map, X_all, YH_all, YA_all, idx_of)
