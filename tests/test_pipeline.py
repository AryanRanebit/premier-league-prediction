#!/usr/bin/env python3
"""Fast pipeline checks (<2 min). Run from repo root:
    EPL_DATA=data python3 tests/test_pipeline.py
Uses the downloaded CSVs (scripts/fetch_data.py first)."""
import os, sys, math
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
os.environ.setdefault("EPL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import epl_model as D
import epl_nn as N

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        sys.exit(1)

# 1. data integrity
completed = ["1516","1617","1718","1819","1920","2021","2122","2223","2324","2425","2526"]
for s in completed:
    rows = D.load_season(s)
    check(f"season {s}: 380 matches, 20 teams", len(rows) == 380 and
          len({m['home'] for m in rows} | {m['away'] for m in rows}) == 20)
cur = D.load_season("2627")
check("current season: >=100 matches played", len(cur) >= 100)

# 2. DC fit sanity (one backtest cutoff)
s = "2425"
sm = sorted([m for m in D.ALL if m["season"] == s], key=lambda x: x["date"])
cut = sm[:100]; cutoff = cut[-1]["date"]
hist = [m for m in D.ALL if m["date"] <= cutoff and not (m["season"] == s and m not in cut)]
model = D.fit_dc(hist, cutoff, current_season=s, current_boost=1.0)
check("DC fit converged", model["success"])
check("DC home advantage plausible (0.2-0.4)", 0.2 < model["home"] < 0.4)
check("DC rho plausible (-0.3..0.1)", -0.3 < model["rho"] < 0.1)
pmf, lam, mu = D.match_pmf(model, "Arsenal", "Chelsea")
check("DC pmf sums to 1", abs(pmf.sum() - 1) < 1e-9 and lam > 0 and mu > 0)

# 3. NN smoke: trains, loss falls, pmfs valid, features frozen-reproducible
ALL = N.load_all()
hist_nn = N.build_history(ALL)
train = [m for m in ALL if m["date"] <= cutoff]
X, YH, YA, _ = N.make_xy(train, hist_nn, False)
Xs, _, _, _ = N.fill_and_scale(X[:1200], X[:1])
w = np.ones(1200)
net, curve = N.train_net(Xs[:1200], YH[:1200], YA[:1200], w, seed=0, epochs=15, bs=256)
check("NN training loss decreases", curve[-1][1] < curve[0][1])
p = N.predict_pmf([net], Xs[:3])
check("NN pmfs sum to 1 (both heads)", np.allclose(p[0].reshape(7,7).sum(), 1, atol=1e-6))
f1 = N.features_for(hist_nn, "Chelsea", cutoff, "home", True, inclusive=True)
f2 = N.features_for(hist_nn, "Chelsea", cutoff, "home", True, inclusive=True)
check("frozen features reproducible", f1 == f2)
# 3b. same-day leakage: injecting a fake same-day result must not change features
import copy
from collections import Counter
dates = Counter(m["date"] for m in ALL)
multi_date = next(d for d, c in dates.items() if c > 1)
team = next(m["home"] for m in ALL if m["date"] == multi_date)
hist2 = copy.deepcopy(hist_nn)
fake_view = {"date": multi_date, "gf": 9, "ga": 0, "sf": 30, "sa": 1,
             "stf": 15, "sta": 0, "cf": 12, "ca": 0, "venue": "home"}
hist2[team].append((multi_date, fake_view))
check("same-day result excluded from features (per-match)",
      N.features_for(hist_nn, team, multi_date, "home", True) ==
      N.features_for(hist2, team, multi_date, "home", True))
check("same-day result included when frozen at cutoff (inclusive)",
      N.features_for(hist_nn, team, multi_date, "home", True, inclusive=True) !=
      N.features_for(hist2, team, multi_date, "home", True, inclusive=True))

# 4. simulation smoke + points conservation (a match awards 3 total if decisive, 2 if drawn)
tk, tp = N.tail_dist(train)
check("tail distribution sums to 1", abs(sum(tp) - 1) < 1e-9)
rem_sm = N.remaining_fixtures(cut)
init_pts = sum(3 if m["hg"] != m["ag"] else 2 for m in cut)
pmfs_sm = [p[0] for _ in rem_sm]
teams, fin_pos, fin_pts = N.simulate_nn(pmfs_sm, rem_sm, cut, 300, np.random.default_rng(0), tk, tp)
tot = fin_pts.sum(axis=1)
check("NN sim: 2..3 points awarded per match",
      tot.min() >= init_pts + 2 * len(rem_sm) - 1e-9 and tot.max() <= init_pts + 3 * len(rem_sm) + 1e-9)
check("NN sim positions within 1..20", fin_pos.min() >= 1 and fin_pos.max() <= 20)
teams_d, fin_pos_d, fin_pts_d = D.simulate(model, cut, n_sims=300, rng=np.random.default_rng(0))
tot_d = fin_pts_d.sum(axis=1)
check("DC sim: 2..3 points awarded per match",
      tot_d.min() >= init_pts + 2 * len(rem_sm) - 1e-9 and tot_d.max() <= init_pts + 3 * len(rem_sm) + 1e-9)


# 5. semantic simulation invariants
def mini_cut():
    """Two played matches so both teams exist with known records."""
    d = cutoff
    return [{"season": "2425", "date": d, "home": "Arsenal", "away": "Chelsea", "hg": 1, "ag": 0,
             "hs": 0, "as_": 0, "hst": 0, "ast": 0, "hc": 0, "ac": 0}]

def pmf_cell(x, y):
    g = np.zeros((7, 7)); g[x, y] = 1.0
    return g

# 5a. W/D/L points: home win 3/0, draw 1/1, away win 0/3
for (x, y, eh, ea) in [(2, 0, 3, 0), (1, 1, 1, 1), (0, 2, 0, 3)]:
    cutm = mini_cut()
    pmfs = [pmf_cell(x, y)]
    teams, fp, fpts = N.simulate_nn(pmfs, [("Chelsea", "Arsenal")], cutm, 50,
                                    np.random.default_rng(1), tk, tp)
    ia, ic = teams.index("Arsenal"), teams.index("Chelsea")
    check(f"points {x}-{y}: fixture-home +{eh} fixture-away +{ea}",
          np.all(fpts[:, ic] == 0 + eh) and np.all(fpts[:, ia] == 3 + ea))

# 5b. goals for/against accounting on real season data: every goal scored is conceded
# by someone, so league-wide GF == GA, GD sums to 0, and points equal 3*decisive + 2*drawn
def league_accounts(matches):
    gf = {}; ga = {}; w = {}; d = {}; l = {}; pts = {}
    for m in matches:
        for t, f, a in ((m["home"], m["hg"], m["ag"]), (m["away"], m["ag"], m["hg"])):
            gf[t] = gf.get(t, 0) + f; ga[t] = ga.get(t, 0) + a
            r = "W" if f > a else ("D" if f == a else "L")
            w[t] = w.get(t, 0) + (r == "W"); d[t] = d.get(t, 0) + (r == "D"); l[t] = l.get(t, 0) + (r == "L")
            pts[t] = pts.get(t, 0) + (3 if r == "W" else 1 if r == "D" else 0)
    return gf, ga, w, d, l, pts
gf, ga, w, d, l, pts = league_accounts(sm)
check("GF == GA league-wide (full season)", sum(gf.values()) == sum(ga.values()))
check("GD sums to zero league-wide", sum(gf[t] - ga[t] for t in gf) == 0)
check("points == 3*W + 1*D for every team", all(pts[t] == 3*w[t] + d[t] for t in pts))
check("38 games per team (W+D+L)", all(w[t] + d[t] + l[t] == 38 for t in pts))
teams, fp, fpts = N.simulate_nn(pmfs_sm, rem_sm, cut, 100, np.random.default_rng(0), tk, tp)
check("positions are a permutation of 1..20 in every sim",
      all(sorted(fp[s]) == list(range(1, 21)) for s in range(100)))

# 5c. 38 games per team over the full season (played + remaining fixtures)
played_count = {t: 0 for t in {m['home'] for m in sm} | {m['away'] for m in sm}}
for m in cut:
    played_count[m["home"]] += 1; played_count[m["away"]] += 1
rem_count = {t: 0 for t in played_count}
for (h, a) in rem_sm:
    rem_count[h] += 1; rem_count[a] += 1
check("every team plays exactly 38 games (100 played + remaining fixtures)",
      all(played_count[t] + rem_count[t] == 38 for t in played_count))
home_ct = {t: 0 for t in played_count}
for m in cut: home_ct[m["home"]] += 1
for (h, a) in rem_sm: home_ct[h] += 1
check("every team plays exactly 19 home / 19 away", all(home_ct[t] == 19 for t in home_ct))

# 5d. rank/tie rules: points, then GD, then GF
d0 = cutoff
def fake(h, a, hg, ag):
    return {"season": "2425", "date": d0, "home": h, "away": a, "hg": hg, "ag": ag,
            "hs": 0, "as_": 0, "hst": 0, "ast": 0, "hc": 0, "ac": 0}
syn = [fake("Arsenal", "Chelsea", 2, 0),   # Arsenal 3pts GD+2
       fake("Liverpool", "Chelsea", 1, 0), # Liverpool 3pts GD+1 GF1
       fake("Man City", "Chelsea", 2, 1)]  # City 3pts GD+1 GF2
tab = N.final_table(syn)
check("tie-breaks: points > GD > GF",
      tab["Arsenal"] == 1 and tab["Man City"] == 2 and tab["Liverpool"] == 3 and tab["Chelsea"] == 4)

# 5e. tail application, real behavioral assertions: a forced 6+ cell must expand via the
# empirical tail to a concrete scoreline with the right result. Baselines from mini_cut:
# Arsenal 3 pts (beat Chelsea 1-0), Chelsea 0.
pmfs6 = [pmf_cell(6, 0)]
teams6, fp6, fpts6 = N.simulate_nn(pmfs6, [("Arsenal", "Chelsea")], mini_cut(), 200,
                                   np.random.default_rng(2), tk, tp)
ia6, ic6 = teams6.index("Arsenal"), teams6.index("Chelsea")
check("forced home-6+ cell: home always wins (+3)", np.all(fpts6[:, ia6] == 6) and np.all(fpts6[:, ic6] == 0))
# 6+/6+ expands each side independently, so a degenerate single-value tail must keep
# it a draw; the multi-value empirical tail may split (both behaviors are by design)
pmfs7 = [pmf_cell(6, 6)]
teams7, fp7, fpts7 = N.simulate_nn(pmfs7, [("Arsenal", "Chelsea")], mini_cut(), 200,
                                   np.random.default_rng(2), [8], np.array([1.0]))
check("6+-6+ with degenerate tail stays a draw (+1 each)",
      np.all(fpts7[:, teams7.index("Arsenal")] == 4) and np.all(fpts7[:, teams7.index("Chelsea")] == 1))
draws = np.random.default_rng(0).choice(tk, size=1000, p=tp)
check("tail samples within [6,10]", draws.min() >= 6 and draws.max() <= 10)
check("tail probabilities sum to 1", abs(float(np.sum(tp)) - 1) < 1e-9)

# 5f. home/away symmetry: each pair plays exactly twice over the season; if one leg is
# played, the remaining leg is the reverse fixture
played_pairs = {}
for m in cut:
    played_pairs.setdefault(frozenset((m["home"], m["away"])), []).append((m["home"], m["away"]))
rem_pairs = {}
for (h, a) in rem_sm:
    rem_pairs.setdefault(frozenset((h, a)), []).append((h, a))
ok = True
for pair in set(played_pairs) | set(rem_pairs):
    p, r = played_pairs.get(pair, []), rem_pairs.get(pair, [])
    if len(p) + len(r) != 2: ok = False
    if len(p) == 1 and len(r) == 1 and p[0] != (r[0][1], r[0][0]): ok = False
    if len(p) == 2 and p[0] != (p[1][1], p[1][0]): ok = False
check("pair legs: two per season, reverse orientations", ok)

print("\nAll checks passed.")
