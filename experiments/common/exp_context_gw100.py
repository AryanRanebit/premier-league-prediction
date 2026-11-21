#!/usr/bin/env python3
"""Experimental context features: manager stints + fixture congestion, added to the
rich NN feature set. Same backtest protocol as repo src/epl_nn.py run_backtest:
5 seasons (2122..2526), 100-match cutoff, 5 seeds x 250 epochs (ORIGINAL recipe,
no early stopping, so the feature effect is isolated), 4000 sims, rng seed 42,
730-day decay, frozen-cutoff remainder NLL.

Arms:
  rich : repo features unchanged (re-run through this harness as the control)
  ctx  : rich + 7 context features per team (manager x5, congestion x2)

Manager data: Wikipedia 'List of Premier League managers' (fetched 21 Nov 2026,
saved as wiki_mgrs.html, parsed to manager_stints.csv). 521 dated stints;
strict interval containment covers 8560/8560 team-matches 2015-16..2026-27.
Features are causal: computed strictly before each match date (fixtures at the
cutoff use inclusive-as-of-cutoff, matching the repo harness semantics).
Left-censoring caveat: managerial history before 2015-16 is outside the data
window, so 'prior PL matches/ppg' counts only 2015-16 onward.

No invented values: matches with no prior managerial history yield None, which
the existing fill_and_scale (training column means) handles, same as promoted
clubs' missing form history.
"""
import os, sys, json, math, csv, bisect
from datetime import datetime, date, timedelta
import numpy as np

REPO = os.environ.get("EPL_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("EPL_DATA", os.path.join(REPO, "data"))
import epl_nn as N

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ctx_results")
os.makedirs(OUT, exist_ok=True)

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
BIG6 = N.BIG6
EPOCHS = 250
N_SEEDS = 5
N_SIMS = 4000

# ---------------- manager stint table ----------------
def load_stints(path):
    stints = []
    for r in csv.DictReader(open(path)):
        stints.append({
            "club": r["club"], "name": r["name"],
            "from": date.fromisoformat(r["from"]),
            "until": date.fromisoformat(r["until"]) if r["until"] else None,
        })
    return stints

_sp = os.path.join(HERE, "manager_stints.csv")
if not os.path.exists(_sp):
    _sp = os.path.abspath(os.path.join(HERE, "..", "manager_ctx", "manager_stints.csv"))
STINTS = load_stints(_sp)
BY_CLUB = {}
for s in STINTS:
    BY_CLUB.setdefault(s["club"], []).append(s)
for c in BY_CLUB:
    BY_CLUB[c].sort(key=lambda s: s["from"])

def stint_on(club, d):
    """Stint containing date d; on boundary overlap prefer the latest-appointed."""
    best = None
    for s in BY_CLUB.get(club, []):
        if s["from"] <= d and (s["until"] is None or d <= s["until"]):
            if best is None or s["from"] > best["from"]:
                best = s
    if best is None:  # fallback (unused: coverage verified 8560/8560 strict)
        prev = [s for s in BY_CLUB.get(club, []) if s["from"] <= d]
        best = prev[-1] if prev else None
    return best

# ---------------- event indexes (built once from ALL matches) ----------------
ALL = N.load_all()

def pts_for(m, t):
    if m["hg"] == m["ag"]: return 1
    home_win = m["hg"] > m["ag"]
    return 3 if (home_win == (m["home"] == t)) else 0

TEAM_EVENTS = {}   # team -> (sorted dates, [(date, manager, pts)])
MGR_EVENTS = {}    # manager name -> (sorted dates, [(date, pts)])
for m in ALL:
    for t in (m["home"], m["away"]):
        s = stint_on(t, m["date"].date())
        assert s is not None, f"no stint for {t} on {m['date']}"
        p = pts_for(m, t)
        TEAM_EVENTS.setdefault(t, []).append((m["date"].date(), s["name"], p))
        MGR_EVENTS.setdefault(s["name"], []).append((m["date"].date(), p))
for t in TEAM_EVENTS:
    TEAM_EVENTS[t].sort(key=lambda e: e[0])
for g in MGR_EVENTS:
    MGR_EVENTS[g].sort(key=lambda e: e[0])
TEAM_DATES = {t: [e[0] for e in ev] for t, ev in TEAM_EVENTS.items()}
MGR_DATES = {g: [e[0] for e in ev] for g, ev in MGR_EVENTS.items()}

def mgr_career_before(name, d, inclusive=False):
    """(matches, ppg) managed in window strictly before d (or <= d if inclusive)."""
    ev = MGR_EVENTS.get(name, [])
    ds = MGR_DATES.get(name, [])
    i = bisect.bisect_right(ds, d) if inclusive else bisect.bisect_left(ds, d)
    sel = ev[:i]
    n = len(sel)
    return n, (sum(p for _, p in sel) / n if n else None)

def ctx_feats(team, d, inclusive=False):
    """7 context features for `team` as of date d (None where no history)."""
    s = stint_on(team, d)
    tenure = (d - s["from"]).days
    ds = TEAM_DATES.get(team, [])
    ev = TEAM_EVENTS.get(team, [])
    i = bisect.bisect_right(ds, d) if inclusive else bisect.bisect_left(ds, d)
    prev = ev[:i]
    stint_matches = sum(1 for (md, _, _) in prev if md >= s["from"])
    prior_pl, _ = mgr_career_before(s["name"], s["from"])
    _, ppg = mgr_career_before(s["name"], d, inclusive=inclusive)
    days_since = (d - prev[-1][0]).days if prev else None
    m14 = sum(1 for (md, _, _) in prev if (d - md).days <= 14 and (d - md).days > 0)
    return [math.log1p(tenure), math.log1p(stint_matches), float(tenure <= 90),
            math.log1p(prior_pl), ppg, days_since, float(m14)]

# ---------------- feature matrix builders ----------------
def make_xy_ctx(matches, hist_map, use_ctx):
    X, YH, YA = [], [], []
    for m in matches:
        fh = N.features_for(hist_map, m["home"], m["date"], "home", True)
        fa = N.features_for(hist_map, m["away"], m["date"], "away", True)
        if use_ctx:
            fh = fh + ctx_feats(m["home"], m["date"].date())
            fa = fa + ctx_feats(m["away"], m["date"].date())
        X.append(fh + fa)
        YH.append(min(m["hg"], 6)); YA.append(min(m["ag"], 6))
    return np.array(X, dtype=object), np.array(YH), np.array(YA)

def fixture_x(hist_map, h, a, cutoff, use_ctx):
    fh = N.features_for(hist_map, h, cutoff, "home", True, inclusive=True)
    fa = N.features_for(hist_map, a, cutoff, "away", True, inclusive=True)
    if use_ctx:
        fh = fh + ctx_feats(h, cutoff.date(), inclusive=True)
        fa = fa + ctx_feats(a, cutoff.date(), inclusive=True)
    return fh + fa

# ---------------- backtest (mirrors epl_nn.run_backtest) ----------------
def run_backtest(use_ctx):
    hist_map = N.build_history(ALL)
    X_all, YH_all, YA_all = make_xy_ctx(ALL, hist_map, use_ctx)
    results = []
    for s in SEASONS:
        sm = sorted([m for m in ALL if m["season"] == s], key=lambda x: x["date"])
        cut = sm[:100]
        cutoff = cut[-1]["date"]
        train = [m for m in ALL if m["date"] <= cutoff and not (m["season"] == s and m not in cut)]
        idx_train = [i for i, m in enumerate(ALL) if m["date"] <= cutoff and not (m["season"] == s and m not in cut)]
        Xtr, YHtr, YAtr = X_all[idx_train], YH_all[idx_train], YA_all[idx_train]
        w = np.array([0.5 ** (max((cutoff - m["date"]).days, 0) / N.HALF_LIFE) for m in train])
        rem = N.remaining_fixtures(cut)
        Xf = np.array([fixture_x(hist_map, h, a, cutoff, use_ctx) for (h, a) in rem], dtype=object)
        Xtr_s, Xf_s, _, _ = N.fill_and_scale(Xtr, Xf)
        nets, curves = [], None
        for seed in range(N_SEEDS):
            net, curve = N.train_net(Xtr_s, YHtr, YAtr, w, seed=seed, epochs=EPOCHS)
            nets.append(net); curves = curve
        pmfs = N.predict_pmf(nets, Xf_s)
        tk, tp = N.tail_dist(train)
        teams, fin_pos, fin_pts = N.simulate_nn(pmfs, rem, cut, N_SIMS, np.random.default_rng(42), tk, tp)
        actual = N.final_table(sm)
        exp_pos = {t: float(fin_pos[:, i].mean()) for i, t in enumerate(teams)}
        big6_mae = float(np.mean([abs(exp_pos[t] - actual[t]) for t in BIG6]))
        all_mae = float(np.mean([abs(exp_pos[t] - actual[t]) for t in teams]))
        rest = sm[100:]
        fr_cache = {}
        ll_sum = 0.0; ll_n = 0
        for m in rest:
            key = (m["home"], m["away"])
            if key not in fr_cache:
                _, xr, _, _ = N.fill_and_scale(Xtr, np.array([fixture_x(hist_map, m["home"], m["away"], cutoff, use_ctx)], dtype=object))
                fr_cache[key] = N.predict_pmf(nets, xr)[0]
            pmf = fr_cache[key]
            p = pmf[min(m["hg"], 6), min(m["ag"], 6)]
            ll_sum += -math.log(max(p, 1e-9)); ll_n += 1
        per_club = {t: {"exp": exp_pos[t], "act": actual[t]} for t in BIG6 if t in exp_pos}
        results.append({"season": s, "cutoff": str(cutoff.date()),
                        "per_club": per_club, "big6_mae": big6_mae, "all_mae": all_mae,
                        "score_nll": ll_sum / max(ll_n, 1),
                        "chelsea_exp": exp_pos.get("Chelsea"), "chelsea_actual": actual.get("Chelsea"),
                        "loss_curve": curves})
        print(f"  {s}: big6_mae={big6_mae:.2f} all_mae={all_mae:.2f} nll={ll_sum/max(ll_n,1):.3f}", flush=True)
    return results

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ctx"
    use_ctx = mode == "ctx"
    # feature variation diagnostic on test seasons
    if use_ctx:
        n_new = n_tot = 0
        for m in ALL:
            if m["season"] in SEASONS:
                n_tot += 2
                for t in (m["home"], m["away"]):
                    if ctx_feats(t, m["date"].date())[2] == 1.0: n_new += 1
        print(f"new-manager(<=90d) share of team-matches in test seasons: {n_new}/{n_tot} = {n_new/n_tot:.3f}")
    res = run_backtest(use_ctx)
    json.dump(res, open(os.path.join(OUT, f"nn_backtest_{'ctx' if use_ctx else 'richrerun'}.json"), "w"), indent=1)
    print(f"ARM {mode}: mean big6={np.mean([r['big6_mae'] for r in res]).round(3)} "
          f"mean all={np.mean([r['all_mae'] for r in res]).round(3)} "
          f"mean nll={np.mean([r['score_nll'] for r in res]).round(4)}")
