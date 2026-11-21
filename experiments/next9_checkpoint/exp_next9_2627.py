#!/usr/bin/env python3
"""Frozen THREE-MODEL predictions for each club's NEXT 9 fixtures (games 11-19
per club), 2026-27 PL. User-requested checkpoint comparison.

Models:
  dc        - production Dixon-Coles snapshot: ratings/home_adv/rho loaded from
              the repo's frozen results/forecast_full_v3.json (fit asof
              2026-11-19 on its OWN protocol: full history, 730d half-life,
              boost=1.0; that asof equals the same first 100 played matches).
              11x11 score grid (goals 0-10).
  nn_rich   - NN, repo rich features, trained 2020-21..cutoff (the matched
              short-history control from the xG test).
  nn_xg     - NN, rich + rolling xG features, SAME training window (the xG arm).
              Both NN arms: arm B chrono 80/20 early stop, 5 seeds, 730d decay;
              PMF = average over the 5 seed nets; 7x7 grid (0..6, 6 = 6+).
Histories are NOT identical across model families: DC uses its own full-history
fit; the two NN arms share the 2020-21+ window by design.

Fixture set: for each club its next 9 ANNOUNCED fixtures after the 2026-11-08
cutoff (union = 90 matches), from pl_2627_remaining_fixtures.csv (FotMob
announced schedule, cross-checked). Verified 2026-11-21T05:54Z: all 90
pre-match (earliest KO 2026-11-21T12:30Z). Postponements may change actual
dates; evaluation happens when every club has actually PLAYED 19, not on a
fixed round date.

Output: full frozen per-match score matrices (this is the record of record -
do not "recompute later and call it the prediction"), 1X2 probabilities,
expected goals, per-club expected points over its 9 matches.
"""
import os, sys, json, csv, math
import numpy as np
from datetime import datetime as dt
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "common")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "euro_stylec")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "xg")))
REPO = os.environ.get("EPL_REPO", os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, os.path.join(REPO, "src"))
import exp_context_gw100 as C
import epl_nn as N
import exp_style_gw19 as S
import exp_xg_gw19 as X
import epl_model as D

CUTOFF = dt(2026, 11, 8)
XG_START = X.XG_START
N_SEEDS = 5

# ---- fixture set: each club's next 9 announced ----
_rf = os.path.abspath(os.path.join(HERE, "..", "forecast_2627_gw10", "pl_2627_remaining_fixtures.csv"))
rows = list(csv.DictReader(open(_rf)))
byclub = defaultdict(list)
for r in rows:
    byclub[r["home"]].append(r); byclub[r["away"]].append(r)
sel = {}
for t, ms in byclub.items():
    ms.sort(key=lambda r: r["date"])
    assert len(ms) >= 9, t
    for r in ms[:9]:
        sel[(r["date"], r["home"], r["away"])] = True
FIX = sorted(sel)
assert len(FIX) == 90, len(FIX)

# ---- DC from the frozen production snapshot ----
snap = json.load(open(os.path.join(REPO, "results", "forecast_full_v3.json")))
teams = sorted(snap["ratings"])
model = {"idx": {t: i for i, t in enumerate(teams)},
         "att": np.array([snap["ratings"][t]["att"] for t in teams]),
         "def": np.array([snap["ratings"][t]["def"] for t in teams]),
         "home": snap["home_adv"], "rho": snap["rho"]}

def dc_pmf(h, a):
    pmf, lam, mu = D.match_pmf(model, h, a)
    return pmf.reshape(11, 11), lam, mu

# ---- NN arms ----
def train_arm(mode):
    hist_map = N.build_history(C.ALL)
    X_all, YH_all, YA_all = X.make_xy(C.ALL, hist_map, mode)
    idx_of = {id(m): i for i, m in enumerate(C.ALL)}
    train = [m for m in C.ALL if m["date"] <= CUTOFF and m["date"] >= XG_START]
    tra, val = S.chrono_split(train)
    wa = np.array([0.5 ** (max((CUTOFF - m["date"]).days, 0) / N.HALF_LIFE) for m in tra])
    iia = [idx_of[id(m)] for m in tra]; iib = [idx_of[id(m)] for m in val]
    Xa, YHa, YAa = X_all[iia], YH_all[iia], YA_all[iia]
    Xb, YHb, YAb = X_all[iib], YH_all[iib], YA_all[iib]
    Xa_s, Xb_s, meansB, stdsB = S.safe_fill_and_scale(Xa, Xb)
    Xf = np.array([X.fixture_x(hist_map, h, a, CUTOFF, mode) for _, h, a in FIX], dtype=object)
    Xf_s = (np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]
                      for r in Xf], dtype=float) - meansB) / stdsB
    nets, epochs = [], []
    for sd in range(N_SEEDS):
        _, best = S.train_with_val(Xa_s, YHa, YAa, wa, Xb_s, YHb, YAb, sd, epochs=250)
        nets.append(S.net_from_state(Xa_s.shape[1], sd, best["state"]))
        epochs.append(best["epoch"])
    pmfs = N.predict_pmf(nets, Xf_s)  # average over seeds, 7x7 each
    return pmfs, epochs, len(train)

out = {"generated_utc": "2026-11-21T05:54Z", "cutoff": str(CUTOFF.date()),
       "played_matches_at_cutoff": 100, "fixture_rule": "each club's next 9 announced fixtures (union 90)",
       "prematch_verified_utc": "2026-11-21T05:54Z (earliest KO 2026-11-21T12:30Z)",
       "models": {}, "fixtures": []}

grids = {"dc": {}, "nn_rich": {}, "nn_xg": {}}
lams = {"dc": {}}
for i, (d, h, a) in enumerate(FIX):
    p, lam, mu = dc_pmf(h, a)
    grids["dc"][i] = p; lams["dc"][i] = (lam, mu)

for mode, key in (("shortrich", "nn_rich"), ("shortxg", "nn_xg")):
    pmfs, epochs, ntrain = train_arm(mode)
    out["models"][key] = {"train_from": str(XG_START.date()), "train_n": ntrain,
                          "best_val_epochs": epochs, "seeds": N_SEEDS,
                          "features": "repo rich" if key == "nn_rich" else "repo rich + 12 rolling xG (last5/last10/venue5 for/against)"}
    for i, p in enumerate(pmfs):
        grids[key][i] = np.array(p, dtype=float)

out["models"]["dc"] = {"source": "results/forecast_full_v3.json (frozen production snapshot)",
                       "protocol": snap["protocol"], "asof": snap["asof"], "grid": "11x11 goals 0-10"}

def stats_1x2(p):
    ph = sum(p[x, y] for x in range(p.shape[0]) for y in range(p.shape[1]) if x > y)
    pd = sum(p[x, x] for x in range(min(p.shape)))
    pa = 1 - ph - pd
    eh = sum(p[x, y] * x for x in range(p.shape[0]) for y in range(p.shape[1]))
    ea = sum(p[x, y] * y for x in range(p.shape[0]) for y in range(p.shape[1]))
    return ph, pd, pa, eh, ea

exp_pts = {k: defaultdict(float) for k in grids}
for i, (d, h, a) in enumerate(FIX):
    rec = {"i": i, "date": d, "home": h, "away": a, "pred": {}}
    for k in grids:
        p = grids[k][i]
        ph, pd, pa, eh, ea = stats_1x2(p)
        flat = [(float(p[x, y]), x, y) for x in range(p.shape[0]) for y in range(p.shape[1])]
        flat.sort(reverse=True)
        rec["pred"][k] = {"p_home": ph, "p_draw": pd, "p_away": pa,
                          "exp_goals_home": eh, "exp_goals_away": ea,
                          "top3": [[x, y, pr] for pr, x, y in flat[:3]],
                          "grid": f"{p.shape[0]}x{p.shape[1]}",
                          "pmf": [[float(v) for v in row] for row in p.tolist()]}
        if k == "dc":
            # common 7x7 event space for cross-model scoreline NLL: bin 6 = 6+
            p7 = np.zeros((7, 7))
            for x in range(11):
                for y in range(11):
                    p7[min(x, 6), min(y, 6)] += p[x, y]
            rec["pred"]["dc_binned7x7"] = {"grid": "7x7 (6 = 6+), aggregated from dc 11x11",
                                            "pmf": [[float(v) for v in row] for row in p7.tolist()]}
        exp_pts[k][h] += 3 * ph + pd
        exp_pts[k][a] += 3 * pa + pd
    out["fixtures"].append(rec)

out["expected_points_next9"] = {k: {t: exp_pts[k][t] for t in sorted(exp_pts[k])} for k in grids}
out["protocol_version"] = "v2"
out["protocol_changes_v2"] = [
    "full-precision probabilities (v1 rounded to 5 decimals, which produced exact zeros in tiny DC cells; v1 preserved alongside, superseded for scoring)",
    "dc_binned7x7 added per fixture: DC 11x11 aggregated onto the common 7x7 event space (bin 6 = 6+ goals) so scoreline NLL compares all three models on identical events",
    "all model cells are strictly positive (DC: exp-link Poisson; NN: softmax); if a future regeneration ever yielded a zero cell, scoring clips at eps=1e-9 before log, uniformly for all models (predeclared, not post-hoc)",
    "input pinning: SHA256 of every input file in INPUT_HASHES.txt; live football-data CSVs update in-season, so reproduction uses the pinned private snapshot",
    "postponement rule: score the fixed 90 frozen fixtures as they complete and report coverage; no substitution",
]
out["fixture_source"] = "FotMob announced schedule, https://www.fotmob.com/api/data/leagues?id=47&ccode3=IND&season=2026/2027 (fetched 2026-11-21)"
out["results_source"] = "football-data.co.uk E0 CSVs, https://www.football-data.co.uk/mmz4281/<season>/E0.csv (snapshot hashes in INPUT_HASHES.txt)"
json.dump(out, open(os.path.join(HERE, "next9_2627_predictions_v2.json"), "w"), indent=1)

clubs = sorted(byclub)
print(f"fixtures {len(out['fixtures'])} | dc lam sample {lams['dc'][0]}")
print(f"{'club':16s} {'DC':>6s} {'NN':>6s} {'NNxG':>6s}")
for t in clubs:
    print(f"{t:16s} {exp_pts['dc'][t]:6.2f} {exp_pts['nn_rich'][t]:6.2f} {exp_pts['nn_xg'][t]:6.2f}")
