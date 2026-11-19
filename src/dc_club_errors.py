#!/usr/bin/env python3
"""Per-club mean abs position error for the DC baseline, v2-matched protocol.
Same fits/sims as epl_model.backtest(); only adds per-club reporting."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("EPL_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(OUT, exist_ok=True)
import epl_model as D

out = {}
for s in ["2122", "2223", "2324", "2425", "2526"]:
    sm = sorted([m for m in D.ALL if m["season"] == s], key=lambda x: x["date"])
    cut = sm[:100]
    cutoff = cut[-1]["date"]
    hist = [m for m in D.ALL if m["date"] <= cutoff and not (m["season"] == s and m not in cut)]
    model = D.fit_dc(hist, cutoff, current_season=s, current_boost=1.0)
    teams, fin_pos, fin_pts = D.simulate(model, cut, n_sims=4000, rng=np.random.default_rng(42))
    pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
    for m in sm:
        if m["hg"] > m["ag"]: pts[m["home"]] += 3
        elif m["hg"] < m["ag"]: pts[m["away"]] += 3
        else: pts[m["home"]] += 1; pts[m["away"]] += 1
        gd[m["home"]] += m["hg"] - m["ag"]; gd[m["away"]] += m["ag"] - m["hg"]
        gf[m["home"]] += m["hg"]; gf[m["away"]] += m["ag"]
    actual = {t: i + 1 for i, t in enumerate(sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t])))}
    exp_pos = {t: float(fin_pos[:, i].mean()) for i, t in enumerate(teams)}
    out[s] = {t: {"exp": round(exp_pos[t], 2), "act": actual[t]} for t in D.BIG6}
    print(s, {t: (out[s][t]["exp"], out[s][t]["act"]) for t in D.BIG6}, flush=True)
json.dump(out, open(os.path.join(OUT, "dc_club_errors_v3.json"), "w"), indent=1)
agg = {t: [] for t in D.BIG6}
for s, clubs in out.items():
    for t, v in clubs.items():
        agg[t].append(abs(v["exp"] - v["act"]))
print("\nDC per-club mean abs error (5 seasons):")
for t in D.BIG6:
    print(f"  {t:12s} {np.mean(agg[t]):.2f}  {[round(e,1) for e in agg[t]]}")
