#!/usr/bin/env python3
"""Current-season DC forecast, boost=1.0 (matches backtest protocol).
Regenerates forecast_full_v3.json: same schema as forecast_full.json.
v3: fixed simulate() away-points bug."""
import os, sys, json
from datetime import datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("EPL_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(OUT, exist_ok=True)
import epl_model as D

now = datetime(2026, 11, 19)
hist = [m for m in D.ALL if m["date"] <= now]
cur = [m for m in hist if m["season"] == "2627"]
model = D.fit_dc(hist, now, current_season="2627", current_boost=1.0)
teams, fin_pos, fin_pts = D.simulate(model, cur, n_sims=20000)
summ = D.summarize(teams, fin_pos, fin_pts)
ratings = {t: {"att": float(model["att"][model["idx"][t]]), "def": float(model["def"][model["idx"][t]])} for t in teams}
json.dump({"asof": str(now.date()), "sims": 20000, "protocol": "boost=1.0, half_life=730d, v3 fixed away-points",
           "home_adv": float(model["home"]), "rho": float(model["rho"]),
           "ratings": ratings, "forecast": summ},
          open(os.path.join(OUT, "forecast_full_v3.json"), "w"), indent=1)
for t in D.BIG6:
    c = summ[t]
    top = sorted(c["pos_dist"].items(), key=lambda kv: -kv[1])[:3]
    print(f"{t:12s} exp_pos={c['exp_pos']:.2f} exp_pts={c['exp_pts']:.1f} top4={c['p_top4']:.1%} champ={c['p_champion']:.2%} modal={top}")
