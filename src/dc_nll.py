#!/usr/bin/env python3
"""DC scoreline NLL on the 280 post-cutoff matches per backtest season.

Scoring convention identical to the NN benchmark (epl_nn.py): goals binned
0,1,2,3,4,5,6+ (7 buckets per side); NLL = mean -log P(actual bucket pair).
DC pmf is the full 0..10 grid from match_pmf merged into the 7x7 buckets.
DC is inherently frozen at cutoff: ratings fitted on history <= cutoff,
no updates during evaluation - the same information state as the NN's
frozen-cutoff simulation. Run: python3 dc_nll.py  (writes dc_nll.json)
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("EPL_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(OUT, exist_ok=True)
import epl_model as D

def merged_pmf(model, home, away):
    """DC 11x11 pmf merged into 7x7 (0-5, 6+)."""
    pmf, lam, mu = D.match_pmf(model, home, away)
    g = pmf.reshape(D.MAXG + 1, D.MAXG + 1)
    out = np.zeros((7, 7))
    for x in range(7):
        xs = range(x, x + 1) if x < 6 else range(6, D.MAXG + 1)
        for y in range(7):
            ys = range(y, y + 1) if y < 6 else range(6, D.MAXG + 1)
            out[x, y] = sum(g[a, b] for a in xs for b in ys)
    return out

def bucket(v):
    return min(v, 6)

res = {}
tot_ll, tot_n = 0.0, 0
for s in ["2122", "2223", "2324", "2425", "2526"]:
    sm = sorted([m for m in D.ALL if m["season"] == s], key=lambda x: x["date"])
    cut = sm[:100]
    cutoff = cut[-1]["date"]
    hist = [m for m in D.ALL if m["date"] <= cutoff and not (m["season"] == s and m not in cut)]
    model = D.fit_dc(hist, cutoff, current_season=s, current_boost=1.0)
    cache = {}
    ll, n = 0.0, 0
    for m in sm[100:]:
        key = (m["home"], m["away"])
        if key not in cache:
            cache[key] = merged_pmf(model, m["home"], m["away"])
        p = cache[key][bucket(m["hg"]), bucket(m["ag"])]
        ll += -math.log(max(p, 1e-12)); n += 1
    res[s] = {"cutoff": str(cutoff.date()), "n_matches": n, "scoreline_nll": round(ll / n, 4)}
    tot_ll += ll; tot_n += n
    print(f"{s}: n={n} scoreline_nll={ll/n:.4f}")
res["mean"] = round(tot_ll / tot_n, 4)
print(f"MEAN over {tot_n} matches: {tot_ll/tot_n:.4f}")
json.dump(res, open(os.path.join(OUT, "dc_nll.json"), "w"), indent=1)
