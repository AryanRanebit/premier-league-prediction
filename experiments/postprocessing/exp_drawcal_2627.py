#!/usr/bin/env python3
"""Experimental draw-rate calibration of the 2627 GW10 forecast (euroC).

POST-PROCESSING, not a retrained model: the neural nets and their predicted
score matrices are unchanged. Each remaining fixture's 7x7 score matrix has its
draw cells (x == y) multiplied by a single global factor LAMBDA and the matrix
renormalized; the season is then re-simulated (4000 sims, rng 42, same seeds).

LAMBDA is solved so the EXPECTED full-season draw rate matches the 2025-26
actual rate (104/380 = 27.4%) - a reference fully known at the cutoff, no
future leakage. Match-consistent: simulated matches still produce valid tables;
no per-club adjustments of any kind.

Assumption (stated): the 2025-26 draw rate persists. Not accuracy-validated:
calibrating draws on a validation backtest would be the follow-up before any
production use.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "euro_stylec")))
import exp_context_gw100 as C
import epl_nn as N
import exp_style_gw19 as S
import exp_euro_stylec_gw19 as E
import forecast_2627_gw10 as F

ALL = C.ALL
CUR = F.CUR
CUTOFF = F.CUTOFF
REM = F.REM
TARGET_DRAW_RATE = 104 / 380  # 2025-26 actual, known at cutoff
ACT_DRAWS = sum(1 for m in CUR if m["hg"] == m["ag"])  # 27 in first 100

# --- identical euroC training as forecast_2627_gw10.py / analyze_2627_points.py ---
hist_map = N.build_history(ALL)
train = list(ALL)
X_all, YH_all, YA_all = E.make_xy(train, hist_map, "euroC")
tra, val = S.chrono_split(train)
idx_of = {id(m): i for i, m in enumerate(train)}
iia = [idx_of[id(m)] for m in tra]; iib = [idx_of[id(m)] for m in val]
Xa, YHa, YAa = X_all[iia], YH_all[iia], YA_all[iia]
Xb, YHb, YAb = X_all[iib], YH_all[iib], YA_all[iib]
wa = np.array([0.5 ** (max((CUTOFF - m["date"]).days, 0) / N.HALF_LIFE) for m in tra])
Xa_s, Xb_s, meansB, stdsB = S.safe_fill_and_scale(Xa, Xb)
Xf = np.array([F.fixture_x(hist_map, h, a, fdate.date(), "euroC") for fdate, h, a in REM], dtype=object)
Xf_s = (np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]
                  for r in Xf], dtype=float) - meansB) / stdsB
nets = []
for sd in range(F.N_SEEDS):
    _, best = S.train_with_val(Xa_s, YHa, YAa, wa, Xb_s, YHb, YAb, sd, epochs=F.EPOCHS)
    nets.append(S.net_from_state(Xa_s.shape[1], sd, best["state"]))
pmfs = N.predict_pmf(nets, Xf_s)
tk, tp = N.tail_dist(train)
rem_pairs = [(h, a) for _, h, a in REM]

def exp_draws(pmfs_list):
    return sum(float(sum(p[x, x] for x in range(7))) for p in pmfs_list)

def calibrated(lam):
    out = []
    for p in pmfs:
        q = p.copy()
        for x in range(7):
            q[x, x] *= lam
        out.append(q / q.sum())
    return out

# solve lambda for target season draw rate
target_remaining = TARGET_DRAW_RATE * 380 - ACT_DRAWS
lo, hi = 0.5, 3.0
for _ in range(60):
    mid = (lo + hi) / 2
    if exp_draws(calibrated(mid)) < target_remaining:
        lo = mid
    else:
        hi = mid
LAM = (lo + hi) / 2
print(f"lambda = {LAM:.4f} (target {target_remaining:.1f} expected draws in remaining 280)")

def sim_summary(pmfs_list):
    teams, fin_pos, fin_pts = N.simulate_nn(pmfs_list, rem_pairs, CUR, F.N_SIMS, np.random.default_rng(42), tk, tp)
    sorted_pts = np.sort(fin_pts, axis=1)[:, ::-1]
    club = {t: {"exp_pos": float(fin_pos[:, i].mean()), "exp_pts": float(fin_pts[:, i].mean())}
            for i, t in enumerate(teams)}
    return {
        "rank_mean_pts": [round(float(x), 1) for x in sorted_pts.mean(axis=0)],
        "champion_pts_mean": round(float(sorted_pts[:, 0].mean()), 1),
        "total_pts": round(float(fin_pts.sum(axis=1).mean()), 0),
        "spread_std": round(float(np.std(sorted_pts.mean(axis=0))), 1),
        "club": club,
    }

before = sim_summary(pmfs)
after = sim_summary(calibrated(LAM))
actual2526 = json.load(open(os.path.join(C.OUT, "forecast_2627_points_check.json")))["actual_2526"]

print("\nrank mean pts (model before -> after | 2526 actual):")
for r in (0, 3, 16, 19):
    print(f"  rank{r+1}: {before['rank_mean_pts'][r]} -> {after['rank_mean_pts'][r]} | {actual2526['rank_pts'][r]}")
print(f"champion mean: {before['champion_pts_mean']} -> {after['champion_pts_mean']} | actual 85")
print(f"total pts: {before['total_pts']} -> {after['total_pts']} | actual {actual2526['total_pts']}")
print(f"spread std: {before['spread_std']} -> {after['spread_std']} | actual 15.4")
print(f"draws after: {(ACT_DRAWS + exp_draws(calibrated(LAM)))/380:.1%} (target 27.4%)")

order = sorted(before["club"], key=lambda t: before["club"][t]["exp_pos"])
print("\nper-club (before pos/pts -> after pos/pts):")
for t in order:
    b, a = before["club"][t], after["club"][t]
    print(f"  {t:16s} {b['exp_pos']:5.2f} {b['exp_pts']:5.1f} -> {a['exp_pos']:5.2f} {a['exp_pts']:5.1f}")

json.dump({"lambda": LAM, "method": "draw cells of each remaining-fixture score matrix x lambda, renormalized; lambda solved so expected season draw rate = 2025-26 actual 27.4% (104/380); post-processing only, nets unchanged; 4000 sims rng 42",
           "target_draw_rate": TARGET_DRAW_RATE, "act_draws_first100": ACT_DRAWS,
           "before": before, "after": after},
          open(os.path.join(C.OUT, "forecast_2627_drawcal.json"), "w"), indent=1)
