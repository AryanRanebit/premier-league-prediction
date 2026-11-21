#!/usr/bin/env python3
"""Experimental SPREAD calibration of the 2627 GW10 forecast (euroC), FIVE-SEASON
reference (2021-22..2025-26 completed seasons) instead of 2025-26 alone.
Post-processing only; nets unchanged.

Reference pooling (stated, not silent): the per-season actual final-table
points std is computed for each of the 5 seasons and AVERAGED (17.97);
the draw target is the average draw rate across the same 5 seasons (23.9%).
The spread solve targets the model's mean WITHIN-SIMULATION season spread
(the direct analogue of a within-season actual table spread) - NOT the std of
per-rank mean points, which is a different, smaller quantity and is reported
separately for comparison with the one-season variant.

Step 1 - strength stretch: each remaining fixture's score matrix is reweighted
toward larger goal supremacy using the model's OWN club strengths (base exp
points per club). For fixture (h,a) with expected goals (muh, mua) and relative
strengths (sh, sa):
    muh' = muh * exp(G * (sh - sa));  mua' = mua * exp(-G * (sh - sa))
    p'(x,y) ∝ p(x,y) * Pois(x;muh')Pois(y;mua') / [Pois(x;muh)Pois(y;mua)]
One global G, solved so the average within-season spread (std of the 20 clubs'
points per simulated season) matches the 2025-26 actual spread (15.4).
No per-club adjustments; the same G applies to every fixture.

Step 2 - draw calibration (as exp_drawcal_2627.py) re-solved on the stretched
matrices so the expected season draw rate still equals 27.4% (104/380).

References are fully known at the cutoff (2025-26 final table). Assumption:
2025-26 spread and draw rate are representative targets. Not accuracy-validated;
simulated tables remain match-consistent.
"""
import os, sys, json, math
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
ACT_DRAWS = sum(1 for m in CUR if m["hg"] == m["ag"])
TARGET_DRAW_RATE = 454 / 1900  # mean draw rate 2021-22..2025-26 (88+87+82+93+104)/1900
TARGET_SPREAD = 17.97          # mean within-season points std, same 5 seasons
# per-season: 2122 18.86 | 2223 17.73 | 2324 19.72 | 2425 18.11 | 2526 15.43

# --- identical euroC training ---
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

def sim(pmfs_list, seed=42):
    teams, fin_pos, fin_pts = N.simulate_nn(pmfs_list, rem_pairs, CUR, F.N_SIMS, np.random.default_rng(seed), tk, tp)
    return teams, fin_pos, fin_pts

def draw_calibrate(pmfs_list):
    def exp_draws(pl):
        return sum(float(sum(p[x, x] for x in range(7))) for p in pl)
    target = TARGET_DRAW_RATE * 380 - ACT_DRAWS
    lo, hi = 0.5, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        out = []
        for p in pmfs_list:
            q = p.copy()
            for x in range(7):
                q[x, x] *= mid
            out.append(q / q.sum())
        if exp_draws(out) < target:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    out = []
    for p in pmfs_list:
        q = p.copy()
        for x in range(7):
            q[x, x] *= lam
        out.append(q / q.sum())
    return out, lam

# base strengths from unadjusted sim
teams0, _, fin_pts0 = sim(pmfs)
base_pts = {t: float(fin_pts0[:, i].mean()) for i, t in enumerate(teams0)}
lg_mean = float(np.mean(list(base_pts.values())))
strength = {t: (base_pts[t] - lg_mean) / lg_mean for t in teams0}

pois = lambda k, mu: math.exp(-mu) * mu ** k / math.factorial(k)

def stretch(G):
    out = []
    for p, (h, a) in zip(pmfs, rem_pairs):
        muh = sum(float(p[x, y]) * x for x in range(7) for y in range(7))
        mua = sum(float(p[x, y]) * y for x in range(7) for y in range(7))
        d = strength[h] - strength[a]
        mh, ma = muh * math.exp(G * d), mua * math.exp(-G * d)
        q = np.zeros_like(p, dtype=float)
        for x in range(7):
            for y in range(7):
                q[x, y] = float(p[x, y]) * (pois(x, mh) / pois(x, muh)) * (pois(y, ma) / pois(y, mua))
        out.append(q / q.sum())
    return out

def combined_of(G):
    stretched = stretch(G)
    fin, lam = draw_calibrate(stretched)
    _, _, fp = sim(fin)
    sorted_pts = np.sort(fp, axis=1)[:, ::-1]
    rank_spread = float(np.std(sorted_pts.mean(axis=0)))
    within = float(np.std(fp, axis=1).mean())
    return fin, lam, rank_spread, within

# solve G so the mean WITHIN-SIM season spread matches the 5-season mean
# within-season actual spread, with draw recalibration inside the loop
lo, hi = 0.0, 10.0
final, lam2, rsp, wsp = None, None, None, None
for _ in range(12):
    mid = (lo + hi) / 2
    final, lam2, rsp, wsp = combined_of(mid)
    if wsp < TARGET_SPREAD:
        lo = mid
    else:
        hi = mid
G = (lo + hi) / 2
final, lam2, rsp, wsp = combined_of(G)
print(f"G = {G:.3f} -> within-sim spread {wsp:.2f} (target {TARGET_SPREAD}), rank-mean spread {rsp:.2f}", flush=True)
print(f"draw lambda re-solved on stretched matrices: {lam2:.4f}")
stretched = stretch(G)

def summary(pmfs_list):
    teams, fin_pos, fin_pts = sim(pmfs_list)
    sorted_pts = np.sort(fin_pts, axis=1)[:, ::-1]
    return {
        "rank_mean_pts": [round(float(x), 1) for x in sorted_pts.mean(axis=0)],
        "champion_pts_mean": round(float(sorted_pts[:, 0].mean()), 1),
        "total_pts": round(float(fin_pts.sum(axis=1).mean()), 0),
        "within_spread": round(float(np.std(fin_pts, axis=1).mean()), 1),
        "rank_spread": round(float(np.std(sorted_pts.mean(axis=0))), 1),
        "club": {t: {"exp_pos": round(float(fin_pos[:, i].mean()), 2),
                     "exp_pts": round(float(fin_pts[:, i].mean()), 1)} for i, t in enumerate(teams)},
    }

def draw_rate(pmfs_list):
    return (ACT_DRAWS + sum(float(sum(p[x, x] for x in range(7))) for p in pmfs_list)) / 380

before = summary(pmfs)                       # unadjusted
draw_only = summary(draw_calibrate(pmfs)[0]) # draw-calibrated only (5-season rate)
combined = summary(final)                    # stretch + draw recalibration (5-season)

print(f"draw rates: base {draw_rate(pmfs):.1%} | combined {draw_rate(final):.1%} (target 23.9% 5-season mean)")
for name, s in (("before", before), ("draw-only", draw_only), ("combined-5season", combined)):
    print(f"{name:16s} champ {s['champion_pts_mean']} | r4 {s['rank_mean_pts'][3]} | r17 {s['rank_mean_pts'][16]} | r20 {s['rank_mean_pts'][19]} | total {s['total_pts']} | within-spread {s['within_spread']} | rank-spread {s['rank_spread']}")
print("5-season mean: champ 88.4 | r4 68.8 | r17 37.8 | r20 19.0 | total 1049.2 | within-spread 17.97 | draws 23.9%")
print("2526 alone:    champ 85   | r4 65   | r17 41   | r20 20   | total 1036 | within-spread 15.43 | draws 27.4%")
print("\ncombined per-club (base pos/pts -> combined pos/pts):")
for t in sorted(before["club"], key=lambda t: combined["club"][t]["exp_pos"]):
    b, a = before["club"][t], combined["club"][t]
    print(f"  {t:16s} {b['exp_pos']:5.2f} {b['exp_pts']:5.1f} -> {a['exp_pos']:5.2f} {a['exp_pts']:5.1f}")

json.dump({"method": "strength stretch (one global G on goal supremacy, model's own base strengths) solved to the 5-season (2021-22..2025-26) mean within-season points spread 17.97, then draw-cell factor re-solved to the 5-season mean draw rate 23.9%; spread solve targets mean within-simulation season spread (NOT per-rank-mean std); post-processing, nets unchanged, no per-club edits, 4000 sims rng 42; reference fit to historical seasons, not an out-of-sample test",
           "G": G, "lambda": lam2,
           "targets": {"spread_within_season_mean": TARGET_SPREAD, "draw_rate": TARGET_DRAW_RATE},
           "reference_by_season": {"2122": {"spread": 18.86, "draws": 88}, "2223": {"spread": 17.73, "draws": 87}, "2324": {"spread": 19.72, "draws": 82}, "2425": {"spread": 18.11, "draws": 93}, "2526": {"spread": 15.43, "draws": 104}},
           "before": before, "draw_only": draw_only, "combined": combined},
          open(os.path.join(C.OUT, "forecast_2627_spreadcal5.json"), "w"), indent=1)
