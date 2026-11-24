#!/usr/bin/env python3
"""Points-distribution check for the 2627 GW10 forecast (euroC), against the
2025-26 actual table. Re-runs the identical seeded forecast (same protocol as
forecast_2627_gw10.py) and captures raw simulation arrays.

Distinguishes:
- per-CLUB mean points (compressed by finishing-position uncertainty)
- per-RANK mean points across sims (whoever finishes k-th)
- champion-points distribution
Plus draw rate, total goals, total points: model vs 2025-26 actual.
"""
import os, sys, json, csv, bisect
import numpy as np
from datetime import datetime as dt

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

# --- identical euroC training/sims as forecast_2627_gw10.py ---
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
teams, fin_pos, fin_pts = N.simulate_nn(pmfs, rem_pairs, CUR, F.N_SIMS, np.random.default_rng(42), tk, tp)
t_idx = {t: i for i, t in enumerate(teams)}

# --- model: per-rank mean points across sims ---
n_sims = fin_pts.shape[0]
sorted_pts = np.sort(fin_pts, axis=1)[:, ::-1]  # per sim, rank 1..20 points
rank_mean = sorted_pts.mean(axis=0)
champ = sorted_pts[:, 0]
print("MODEL per-rank mean points (sim): rank1 {:.1f} rank4 {:.1f} rank17 {:.1f} rank20 {:.1f}".format(
    rank_mean[0], rank_mean[3], rank_mean[16], rank_mean[19]))
print("MODEL champion pts: mean {:.1f} median {:.1f} p10 {:.0f} p90 {:.0f}".format(
    champ.mean(), np.median(champ), np.percentile(champ, 10), np.percentile(champ, 90)))
print("MODEL total points per season: mean {:.0f}".format(fin_pts.sum(axis=1).mean() + sum(
    3 if m['hg'] != m['ag'] else 2 for m in CUR)))

# --- model: draw rate and goals ---
# expected draws/goals over remaining fixtures from pmfs + actual first-100
exp_draws = sum(float(sum(p[x, x] for x in range(7))) for p in pmfs)
act_draws = sum(1 for m in CUR if m['hg'] == m['ag'])
grid7 = [(x, y) for x in range(7) for y in range(7)]
exp_goals = sum(float(sum(p[x, y] * (x + y) for x in range(7) for y in range(7))) for p in pmfs)
act_goals = sum(m['hg'] + m['ag'] for m in CUR)
print("MODEL draws: {:.0f} exp remaining + {} actual = {:.0f}/380 = {:.1%}".format(
    exp_draws, act_draws, exp_draws + act_draws, (exp_draws + act_draws) / 380))
print("MODEL goals: {:.0f} exp remaining + {} actual = {:.0f} = {:.2f}/game".format(
    exp_goals, act_goals, exp_goals + act_goals, (exp_goals + act_goals) / 380))

# --- actual 2025-26 table ---
prev = sorted([m for m in ALL if m["season"] == "2526"], key=lambda x: x["date"])
pts = {}; gd = {}; gf = {}
for m in prev:
    for t in (m['home'], m['away']): pts.setdefault(t, 0)
    if m['hg'] > m['ag']: pts[m['home']] += 3
    elif m['hg'] < m['ag']: pts[m['away']] += 3
    else: pts[m['home']] += 1; pts[m['away']] += 1
ordered = sorted(pts, key=lambda t: -pts[t])
act_rank_pts = sorted(pts.values(), reverse=True)
d2526 = sum(1 for m in prev if m['hg'] == m['ag'])
g2526 = sum(m['hg'] + m['ag'] for m in prev)
print("ACTUAL 2526: rank1 {} rank4 {} rank17 {} rank20 {} | total pts {} | draws {}/380 = {:.1%} | goals {} = {:.2f}/game".format(
    act_rank_pts[0], act_rank_pts[3], act_rank_pts[16], act_rank_pts[19],
    sum(pts.values()), d2526, d2526 / 380, g2526, g2526 / 380))
print("ACTUAL 2526 champion:", ordered[0], act_rank_pts[0], "| relegated:", ordered[-3:])

# --- compression comparison: per-club mean vs per-rank mean ---
club_mean = sorted([fin_pts[:, t_idx[t]].mean() for t in teams], reverse=True)
print("MODEL per-CLUB mean points sorted: top1 {:.1f} top4 {:.1f} | spread std {:.1f}".format(
    club_mean[0], club_mean[3], np.std(club_mean)))
print("MODEL per-RANK mean points: top1 {:.1f} top4 {:.1f} | spread std {:.1f}".format(
    rank_mean[0], rank_mean[3], np.std(rank_mean)))
print("ACTUAL 2526 per-rank points: top1 {} top4 {} | spread std {:.1f}".format(
    act_rank_pts[0], act_rank_pts[3], np.std(act_rank_pts)))

# save
out = {
    "model_rank_mean_pts": [round(float(x), 1) for x in rank_mean],
    "model_champion_pts": {"mean": round(float(champ.mean()), 1), "median": float(np.median(champ)),
                           "p10": float(np.percentile(champ, 10)), "p90": float(np.percentile(champ, 90))},
    "model_total_pts": round(float(fin_pts.sum(axis=1).mean() + sum(3 if m['hg'] != m['ag'] else 2 for m in CUR)), 0),
    "model_draws_pct": round(float((exp_draws + act_draws) / 380), 3),
    "model_goals_per_game": round(float((exp_goals + act_goals) / 380), 2),
    "actual_2526": {"rank_pts": act_rank_pts, "total_pts": sum(pts.values()),
                    "draws_pct": round(d2526 / 380, 3), "goals_per_game": round(g2526 / 380, 2),
                    "champion": ordered[0], "relegated": ordered[-3:]},
}
json.dump(out, open(os.path.join(C.OUT, "forecast_2627_points_check.json"), "w"), indent=1)
