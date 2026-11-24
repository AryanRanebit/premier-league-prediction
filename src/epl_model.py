#!/usr/bin/env python3
"""Dixon-Coles style model for PL 2026-27 Big Six finish prediction.

Data: football-data.co.uk E0.csv per season (free, personal use).
Model: independent Poisson home/away goals with DC low-score adjustment,
attack/defence ratings per team, home advantage, exponential time decay.
Validation: backtest on completed seasons truncated at the same point
(100 matches played) as the current season.
"""
import csv, math, os, sys, json
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

DATA = os.environ.get("EPL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
SEASONS = ["1516","1617","1718","1819","1920","2021","2122","2223","2324","2425","2526","2627"]
BIG6 = ["Arsenal","Chelsea","Liverpool","Man City","Man United","Tottenham"]

def parse_date(s):
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(s)

def load_season(code):
    path = os.path.join(DATA, f"E0_{code}.csv")
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("HomeTeam") or not r.get("FTHG"):
                continue
            rows.append({
                "season": code,
                "date": parse_date(r["Date"]),
                "home": r["HomeTeam"].strip(),
                "away": r["AwayTeam"].strip(),
                "hg": int(r["FTHG"]),
                "ag": int(r["FTAG"]),
            })
    rows.sort(key=lambda x: x["date"])
    return rows

ALL = []
for s in SEASONS:
    ALL.extend(load_season(s))

def season_of(m): return m["season"]

def fit_dc(matches, asof, half_life_days=730.0, current_season=None, current_boost=1.0):
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    # weights
    w = []
    for m in matches:
        age = max((asof - m["date"]).days, 0)
        wt = 0.5 ** (age / half_life_days)
        if current_season and m["season"] == current_season:
            wt *= current_boost
        w.append(wt)
    w = np.array(w)
    hg = np.array([m["hg"] for m in matches], dtype=float)
    ag = np.array([m["ag"] for m in matches], dtype=float)
    hi = np.array([idx[m["home"]] for m in matches])
    ai = np.array([idx[m["away"]] for m in matches])

    def tau(x, y, lam, mu, rho):
        if x == 0 and y == 0: return 1 - lam * mu * rho
        if x == 0 and y == 1: return 1 + lam * rho
        if x == 1 and y == 0: return 1 + mu * rho
        if x == 1 and y == 1: return 1 - rho
        return 1.0

    def negll(p):
        att = p[:n]; dfn = p[n:2*n]; home = p[2*n]; rho = p[2*n+1]
        lam = np.exp(home + att[hi] - dfn[ai])
        mu = np.exp(att[ai] - dfn[hi])
        # Poisson log pmf
        ll = w * (hi*0 + (-lam + hg*np.log(lam) - np.vectorize(math.lgamma)(hg+1))
                  + (-mu + ag*np.log(mu) - np.vectorize(math.lgamma)(ag+1)))
        taus = np.array([tau(x, y, l, m, rho) for x, y, l, m in zip(hg, ag, lam, mu)])
        taus = np.clip(taus, 1e-6, None)
        ll = ll + w * np.log(taus)
        return -ll.sum()

    x0 = np.zeros(2*n + 2); x0[2*n] = 0.25; x0[2*n+1] = 0.0
    cons = ({"type": "eq", "fun": lambda p: p[:n].sum()},
            {"type": "eq", "fun": lambda p: p[n:2*n].sum()})
    res = minimize(negll, x0, method="SLSQP", constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    p = res.x
    return {"teams": teams, "idx": idx, "att": p[:n], "def": p[n:2*n],
            "home": p[2*n], "rho": p[2*n+1], "success": res.success}

MAXG = 10
def match_pmf(model, home, away):
    i, j = model["idx"][home], model["idx"][away]
    lam = math.exp(model["home"] + model["att"][i] - model["def"][j])
    mu = math.exp(model["att"][j] - model["def"][i])
    g = np.arange(MAXG+1)
    ph = np.exp(-lam) * lam**g / np.vectorize(math.factorial)(g)
    pa = np.exp(-mu) * mu**g / np.vectorize(math.factorial)(g)
    pmf = np.outer(ph, pa)
    rho = model["rho"]
    def tau(x, y):
        if x == 0 and y == 0: return 1 - lam * mu * rho
        if x == 0 and y == 1: return 1 + lam * rho
        if x == 1 and y == 0: return 1 + mu * rho
        if x == 1 and y == 1: return 1 - rho
        return 1.0
    for (x, y) in ((0,0),(0,1),(1,0),(1,1)):
        pmf[x, y] *= max(tau(x, y), 1e-9)
    pmf = np.clip(pmf, 0, None)
    pmf /= pmf.sum()
    return pmf.ravel(), lam, mu

GRID = np.array([(x, y) for x in range(MAXG+1) for y in range(MAXG+1)])

def remaining_fixtures(season_matches):
    played = {(m["home"], m["away"]) for m in season_matches}
    teams = sorted({m["home"] for m in season_matches} | {m["away"] for m in season_matches})
    rem = [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]
    return rem

def simulate(model, season_matches, n_sims=20000, rng=None):
    rng = rng or np.random.default_rng(7)
    teams = sorted({m["home"] for m in season_matches} | {m["away"] for m in season_matches})
    rem = remaining_fixtures(season_matches)
    # current points/GD/GF
    pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
    for m in season_matches:
        if m["hg"] > m["ag"]: pts[m["home"]] += 3
        elif m["hg"] < m["ag"]: pts[m["away"]] += 3
        else: pts[m["home"]] += 1; pts[m["away"]] += 1
        gd[m["home"]] += m["hg"] - m["ag"]; gd[m["away"]] += m["ag"] - m["hg"]
        gf[m["home"]] += m["hg"]; gf[m["away"]] += m["ag"]
    pmfs = []
    for (h, a) in rem:
        pmf, lam, mu = match_pmf(model, h, a)
        pmfs.append(pmf)
    pmfs = np.array(pmfs)
    t_idx = {t: i for i, t in enumerate(teams)}
    nT = len(teams)
    fin_pos = np.zeros((n_sims, nT), dtype=int)
    fin_pts = np.zeros((n_sims, nT))
    P = np.tile(np.array([pts[t] for t in teams]), (n_sims, 1)).astype(float)
    G = np.tile(np.array([gd[t] for t in teams]), (n_sims, 1)).astype(float)
    F = np.tile(np.array([gf[t] for t in teams]), (n_sims, 1)).astype(float)
    for k, (h, a) in enumerate(rem):
        draws = np.array([rng.choice(len(GRID), p=pmfs[k]) for _ in range(n_sims)])
        hg = GRID[draws][:, 0]; ag = GRID[draws][:, 1]
        ih, ia = t_idx[h], t_idx[a]
        P[:, ih] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        P[:, ia] += np.where(ag > hg, 3, np.where(ag == hg, 1, 0))
        G[:, ih] += hg - ag; G[:, ia] += ag - hg
        F[:, ih] += hg; F[:, ia] += ag
    for s in range(n_sims):
        order = sorted(range(nT), key=lambda t: (-P[s, t], -G[s, t], -F[s, t]))
        for pos, t in enumerate(order, 1):
            fin_pos[s, t] = pos
            fin_pts[s, t] = P[s, t]
    return teams, fin_pos, fin_pts

def summarize(teams, fin_pos, fin_pts, focus=None):
    out = {}
    n_sims, _ = fin_pos.shape
    for i, t in enumerate(teams):
        if focus and t not in focus:
            continue
        pos = fin_pos[:, i]
        out[t] = {
            "exp_pos": float(pos.mean()),
            "exp_pts": float(fin_pts[:, i].mean()),
            "pos_dist": {int(p): float((pos == p).mean()) for p in range(1, 21)},
            "p_top4": float((pos <= 4).mean()),
            "p_champion": float((pos == 1).mean()),
        }
    return out

# ---------- Backtest ----------
def backtest():
    results = []
    for s in ["2122","2223","2324","2425","2526"]:
        sm = [m for m in ALL if m["season"] == s]
        sm.sort(key=lambda x: x["date"])
        cut = sm[:100]  # same depth as current season (100 matches)
        cutoff_date = cut[-1]["date"]
        hist = [m for m in ALL if m["date"] <= cutoff_date and not (m["season"] == s and m not in cut)]
        model = fit_dc(hist, cutoff_date, current_season=s, current_boost=1.0)
        teams, fin_pos, fin_pts = simulate(model, cut, n_sims=4000,
                                           rng=np.random.default_rng(42))
        # actual final table
        pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
        for m in sm:
            if m["hg"] > m["ag"]: pts[m["home"]] += 3
            elif m["hg"] < m["ag"]: pts[m["away"]] += 3
            else: pts[m["home"]] += 1; pts[m["away"]] += 1
            gd[m["home"]] += m["hg"] - m["ag"]; gd[m["away"]] += m["ag"] - m["hg"]
            gf[m["home"]] += m["hg"]; gf[m["away"]] += m["ag"]
        actual = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t]))
        act_pos = {t: i+1 for i, t in enumerate(actual)}
        exp_pos = {t: float(fin_pos[:, i].mean()) for i, t in enumerate(teams)}
        mae_all = np.mean([abs(exp_pos[t] - act_pos[t]) for t in teams])
        mae_b6 = np.mean([abs(exp_pos[t] - act_pos[t]) for t in BIG6 if t in teams])
        champ_pred = max(teams, key=lambda t: float((fin_pos[:, teams.index(t)] == 1).mean()))
        top4_pred = set(sorted(teams, key=lambda t: exp_pos[t])[:4])
        top4_act = set(actual[:4])
        results.append({
            "season": s, "cutoff": str(cutoff_date.date()),
            "mae_all": round(float(mae_all), 2),
            "mae_big6": round(float(mae_b6), 2),
            "champ_pred": champ_pred, "champ_actual": actual[0],
            "champ_hit": champ_pred == actual[0],
            "champ_in_pred_top3": actual[0] in sorted(teams, key=lambda t: exp_pos[t])[:3],
            "top4_overlap": len(top4_pred & top4_act),
        })
    return results

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "current"
    if mode == "backtest":
        for r in backtest():
            print(json.dumps(r))
    else:
        now = datetime(2026, 11, 19)
        hist = [m for m in ALL if m["date"] <= now]
        cur = [m for m in hist if m["season"] == "2627"]
        print(f"current season matches: {len(cur)}", file=sys.stderr)
        model = fit_dc(hist, now, current_season="2627", current_boost=1.0)
        print("home_adv", round(model["home"], 3), "rho", round(model["rho"], 3), file=sys.stderr)
        teams, fin_pos, fin_pts = simulate(model, cur, n_sims=20000)
        summ = summarize(teams, fin_pos, fin_pts, focus=set(BIG6))
        ratings = sorted(((t, model["att"][model["idx"][t]], model["def"][model["idx"][t]]) for t in teams),
                         key=lambda x: -x[1])
        print("RATINGS (attack, defence):")
        for t, a, d in ratings:
            print(f"  {t:15s} att={a:+.3f} def={d:+.3f}")
        print("\nBIG SIX FORECAST:")
        for t in BIG6:
            s = summ[t]
            dist = ", ".join(f"{p}:{v:.0%}" for p, v in sorted(s["pos_dist"].items()) if v >= 0.01)
            print(f"{t:12s} exp_pos={s['exp_pos']:.1f} exp_pts={s['exp_pts']:.1f} "
                  f"P(champ)={s['p_champion']:.1%} P(top4)={s['p_top4']:.1%}\n    {dist}")

def backtest_detail():
    print("season | team | pred_pos | act_pos | |err|")
    for s in ["2122","2223","2324","2425","2526"]:
        sm = sorted([m for m in ALL if m["season"] == s], key=lambda x: x["date"])
        cut = sm[:100]
        cutoff_date = cut[-1]["date"]
        hist = [m for m in ALL if m["date"] <= cutoff_date and not (m["season"] == s and m not in cut)]
        model = fit_dc(hist, cutoff_date, current_season=s, current_boost=1.0)
        assert model["success"], f"fit failed {s}"
        teams, fin_pos, fin_pts = simulate(model, cut, n_sims=4000, rng=np.random.default_rng(42))
        pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
        for m in sm:
            if m["hg"] > m["ag"]: pts[m["home"]] += 3
            elif m["hg"] < m["ag"]: pts[m["away"]] += 3
            else: pts[m["home"]] += 1; pts[m["away"]] += 1
            gd[m["home"]] += m["hg"] - m["ag"]; gd[m["away"]] += m["ag"] - m["hg"]
            gf[m["home"]] += m["hg"]; gf[m["away"]] += m["ag"]
        actual = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t]))
        act_pos = {t: i+1 for i, t in enumerate(actual)}
        exp_pos = {t: float(fin_pos[:, i].mean()) for i, t in enumerate(teams)}
        for t in BIG6:
            print(f"{s} | {t:12s} | {exp_pos[t]:5.1f} | {act_pos[t]:2d} | {abs(exp_pos[t]-act_pos[t]):4.1f}")
