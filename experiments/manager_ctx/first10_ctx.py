#!/usr/bin/env python3
"""First-10-matchweeks 2026-27 preseason-frozen test, NN rich + ctx features.
Same protocol as backtest_first10.py 'nn rich' arm: training strictly before the
Aug 21 opener, fixture features frozen at the cutoff, 5 seeds x 250 epochs
(original recipe), 4000 sims, seed 42, tail from training data only.
Compare against attached first10 bundle nn_rich_first10.json.
"""
import os, sys, json, math
import numpy as np

REPO = os.environ.get("EPL_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("EPL_DATA", os.path.join(REPO, "data"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "euro_stylec")))
import epl_model as D
import epl_nn as N
import exp_context_gw100 as C

OUT = os.path.join(C.OUT)
CUR = sorted([m for m in D.ALL if m["season"] == "2627"], key=lambda x: x["date"])
SEASON_START = CUR[0]["date"]
CUTOFF = SEASON_START
FIXTURES = [(m["home"], m["away"]) for m in CUR]
TEAMS = sorted({t for h, a in FIXTURES for t in (h, a)})

def metrics(pmfs):
    n = len(CUR); acc = 0; ll1x2 = 0.0; llsc = 0.0; exact = 0; brier = 0.0
    for m, p in zip(CUR, pmfs):
        phw = float(sum(p[x, y] for x in range(7) for y in range(7) if x > y))
        pdw = float(sum(p[x, y] for x in range(7) for y in range(7) if x == y))
        paw = float(sum(p[x, y] for x in range(7) for y in range(7) if x < y))
        probs = np.array([phw, pdw, paw]); probs /= probs.sum()
        actual = 0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2)
        acc += int(probs.argmax() == actual)
        ll1x2 += -math.log(max(probs[actual], 1e-12))
        onehot = np.zeros(3); onehot[actual] = 1
        brier += float(((probs - onehot) ** 2).sum())
        pb = p[min(m["hg"], 6), min(m["ag"], 6)]
        llsc += -math.log(max(pb, 1e-12))
        exact += int(np.unravel_index(p.argmax(), p.shape) == (m["hg"], m["ag"]))
    return {"n": n, "acc_1x2": acc / n, "logloss_1x2": ll1x2 / n,
            "score_nll": llsc / n, "exact_score_hits": exact, "brier": brier / n}

def sim_segment(pmfs7, n_sims=4000, seed=42, tail=None):
    rng = np.random.default_rng(seed)
    nT = len(TEAMS); t_idx = {t: i for i, t in enumerate(TEAMS)}
    grid7 = [(x, y) for x in range(7) for y in range(7)]
    P = np.zeros((n_sims, nT)); G = np.zeros((n_sims, nT)); F = np.zeros((n_sims, nT))
    for k, (h, a) in enumerate(FIXTURES):
        p = pmfs7[k].ravel(); p = p / p.sum()
        draws = rng.choice(len(grid7), size=n_sims, p=p)
        hg = np.array([grid7[d][0] for d in draws]); ag = np.array([grid7[d][1] for d in draws])
        if tail is not None:
            tk, tp = tail
            mh = hg == 6; ma = ag == 6
            if mh.any(): hg[mh] = rng.choice(tk, size=mh.sum(), p=tp)
            if ma.any(): ag[ma] = rng.choice(tk, size=ma.sum(), p=tp)
        ih, ia = t_idx[h], t_idx[a]
        P[:, ih] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        P[:, ia] += np.where(ag > hg, 3, np.where(ag == hg, 1, 0))
        G[:, ih] += hg - ag; G[:, ia] += ag - hg
        F[:, ih] += hg; F[:, ia] += ag
    fin_pos = np.zeros((n_sims, nT), dtype=int)
    for s in range(n_sims):
        order = sorted(range(nT), key=lambda t: (-P[s, t], -G[s, t], -F[s, t]))
        for pos, t in enumerate(order, 1):
            fin_pos[s, t] = pos
    exp_pos = {t: float(fin_pos[:, i].mean()) for i, t in enumerate(TEAMS)}
    exp_pts = {t: float(P[:, i].mean()) for i, t in enumerate(TEAMS)}
    return exp_pos, exp_pts

def actual_state():
    pts = {t: 0 for t in TEAMS}; gd = {t: 0 for t in TEAMS}; gf = {t: 0 for t in TEAMS}
    for m in CUR:
        if m["hg"] > m["ag"]: pts[m["home"]] += 3
        elif m["hg"] < m["ag"]: pts[m["away"]] += 3
        else: pts[m["home"]] += 1; pts[m["away"]] += 1
        gd[m["home"]] += m["hg"] - m["ag"]; gd[m["away"]] += m["ag"] - m["hg"]
        gf[m["home"]] += m["hg"]; gf[m["away"]] += m["ag"]
    order = sorted(TEAMS, key=lambda t: (-pts[t], -gd[t], -gf[t]))
    pos = {t: i + 1 for i, t in enumerate(order)}
    return pos, pts

ACT_POS, ACT_PTS = actual_state()

ALLN = C.ALL  # same object as N.load_all()
hist_map = N.build_history(ALLN)
train_idx = [i for i, m in enumerate(ALLN) if m["date"] < CUTOFF]
X_all, YH, YA = C.make_xy_ctx(ALLN, hist_map, True)
Xtr, YHtr, YAtr = X_all[train_idx], YH[train_idx], YA[train_idx]
w = np.array([0.5 ** (max((CUTOFF - ALLN[i]["date"]).days, 0) / N.HALF_LIFE) for i in train_idx])
Xf = np.array([C.fixture_x(hist_map, h, a, CUTOFF, True) for h, a in FIXTURES], dtype=object)
Xtr_s, Xf_s, _, _ = N.fill_and_scale(Xtr, Xf)
train_matches = [ALLN[i] for i in train_idx]
tk, tp = N.tail_dist(train_matches)
nets = []
for sd in range(5):
    net, curve = N.train_net(Xtr_s, YHtr, YAtr, w, seed=sd, epochs=250)
    nets.append(net)
    print(f"seed {sd} final train loss {curve[-1]}", flush=True)
pmfs = N.predict_pmf(nets, Xf_s)
exp_pos, exp_pts = sim_segment(pmfs, tail=(tk, tp))
res = {"metrics": metrics(pmfs),
       "table": {"pos_mae": float(np.mean([abs(exp_pos[t] - ACT_POS[t]) for t in TEAMS])),
                 "pts_mae": float(np.mean([abs(exp_pts[t] - ACT_PTS[t]) for t in TEAMS])),
                 "per_club": {t: {"exp_pos": round(exp_pos[t], 1), "act_pos": ACT_POS[t],
                                  "exp_pts": round(exp_pts[t], 1), "act_pts": ACT_PTS[t]} for t in TEAMS}}}
json.dump(res, open(os.path.join(OUT, "nn_ctx_first10.json"), "w"), indent=1)
print("acc", res["metrics"]["acc_1x2"], "nll", round(res["metrics"]["score_nll"], 3),
          "pos_mae", res["table"]["pos_mae"], "pts_mae", res["table"]["pts_mae"])
