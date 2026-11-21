#!/usr/bin/env python3
"""2026-27 final-table forecast after the first 100 matches (every club exactly
10 played, last played 2026-11-08; football-data E0_2627 snapshot cross-checked
1:1 against FotMob on 2026-11-21: 100 played / 280 announced).

Modes:
  rich  : matched NN baseline (repo rich feature set only)
  euroC : experimental variant from the matched GW19 backtest (rich + European
          workload + lagged FotMob poss-won-final-3rd season aggregate)

Protocol (identical to the backtest arm B): chrono 80/20 split, best-validation
checkpoint (max 250 epochs), 5 seeds, 730d half-life decay, 4000 sims, rng 42,
6+ tail from training data only. Training: all PL matches 2015-16..cutoff.

FUTURE-FEATURE ASSUMPTIONS (stated, not hidden):
- repo rich form features: frozen inclusive at the cutoff (repo protocol);
  form does not update as simulated results come in.
- euroC workload features (days since last match, matches in last 14 days):
  computed per future fixture date from the KNOWN CALENDAR ONLY: played PL +
  played Europe (<= cutoff) + announced future PL (FotMob, 2026-11-21) +
  announced future Europe (league-phase dates). NOT counted: unscheduled
  European knockout rounds (draws not made; understates spring congestion for
  clubs that advance), domestic cups, future postponements.
- styleC high recovery: 2025-26 full-season aggregate, frozen all season
  (lagged by design); promoted clubs -> None -> training-mean fill.

Usage: python3 forecast_2627_gw10.py {rich|euroC}
"""
import os, sys, json, math, csv, bisect
import numpy as np
from datetime import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "euro_stylec")))
import exp_context_gw100 as C
import epl_nn as N
import exp_style_gw19 as S
import exp_euro_stylec_gw19 as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = C.OUT
N_SEEDS = 5
EPOCHS = 250
N_SIMS = 4000

ALL = C.ALL
CUR = sorted([m for m in ALL if m["season"] == "2627"], key=lambda x: x["date"])
assert len(CUR) == 100, f"expected 100 played 2627 matches, got {len(CUR)}"
from collections import Counter
_pc = Counter()
for m in CUR:
    _pc[m["home"]] += 1; _pc[m["away"]] += 1
assert set(_pc.values()) == {10}, f"not all clubs at 10 played: {_pc}"
CUTOFF = CUR[-1]["date"]

# remaining fixtures with announced dates (FotMob, fetched 2026-11-21)
REM = [(dt.strptime(r["date"], "%Y-%m-%d"), r["home"], r["away"])
       for r in csv.DictReader(open(os.path.join(HERE, "pl_2627_remaining_fixtures.csv")))]
REM.sort(key=lambda r: r[0])
assert len(REM) == 280

# known-calendar event dates per club for schedule-aware workload:
# played PL + played Europe (<= cutoff) + announced future PL + announced future Europe
KNOWN_EVENTS = {}
for t, dates in C.TEAM_DATES.items():
    KNOWN_EVENTS.setdefault(t, []).extend(dates)
for t, dates in E.EURO_EVENTS.items():
    KNOWN_EVENTS.setdefault(t, []).extend(dates)
for d, h, a in REM:
    KNOWN_EVENTS.setdefault(h, []).append(d.date())
    KNOWN_EVENTS.setdefault(a, []).append(d.date())
for r in csv.DictReader(open(os.path.join(HERE, "pl_clubs_european_fixtures_2015_2027.csv") if os.path.exists(os.path.join(HERE, "pl_clubs_european_fixtures_2015_2027.csv")) else os.path.abspath(os.path.join(HERE, "..", "euro_stylec", "pl_clubs_european_fixtures_2015_2027.csv")))):
    if r["status"] != "scheduled":
        continue
    d = dt.strptime(r["date"], "%Y-%m-%d").date()
    for club in r["pl_club"].split("/"):
        KNOWN_EVENTS.setdefault(club, []).append(d)
for t in KNOWN_EVENTS:
    KNOWN_EVENTS[t] = sorted(set(KNOWN_EVENTS[t]))

def workload_future(team, d):
    """days_since, matches-in-14d as of date d over the known calendar."""
    ds = KNOWN_EVENTS.get(team, [])
    i = bisect.bisect_left(ds, d)
    prev = ds[:i]
    days_since = (d - prev[-1]).days if prev else None
    m14 = sum(1 for md in prev if 0 < (d - md).days <= 14)
    return [days_since, float(m14)]

def fixture_x(hist_map, h, a, fdate, mode):
    fh = N.features_for(hist_map, h, CUTOFF, "home", True, inclusive=True)
    fa = N.features_for(hist_map, a, CUTOFF, "away", True, inclusive=True)
    if mode == "euroC":
        fh = fh + workload_future(h, fdate) + [E.prior_highrec(h, "2627")]
        fa = fa + workload_future(a, fdate) + [E.prior_highrec(a, "2627")]
    return fh + fa

def main(mode):
    assert mode in ("rich", "euroC")
    hist_map = N.build_history(ALL)
    train = list(ALL)  # dataset ends at the cutoff by construction
    assert all(m["date"] <= CUTOFF for m in train)
    X_all, YH_all, YA_all = E.make_xy(train, hist_map, "euroC" if mode == "euroC" else "rich")
    tra, val = S.chrono_split(train)
    idx_of = {id(m): i for i, m in enumerate(train)}
    iia = [idx_of[id(m)] for m in tra]; iib = [idx_of[id(m)] for m in val]
    Xa, YHa, YAa = X_all[iia], YH_all[iia], YA_all[iia]
    Xb, YHb, YAb = X_all[iib], YH_all[iib], YA_all[iib]
    wa = np.array([0.5 ** (max((CUTOFF - m["date"]).days, 0) / N.HALF_LIFE) for m in tra])
    Xa_s, Xb_s, meansB, stdsB = S.safe_fill_and_scale(Xa, Xb)
    Xf = np.array([fixture_x(hist_map, h, a, fdate.date(), mode) for fdate, h, a in REM], dtype=object)
    Xf_s = (np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]
                      for r in Xf], dtype=float) - meansB) / stdsB
    nets, best_epochs = [], []
    for sd in range(N_SEEDS):
        _, best = S.train_with_val(Xa_s, YHa, YAa, wa, Xb_s, YHb, YAb, sd, epochs=EPOCHS)
        nets.append(S.net_from_state(Xa_s.shape[1], sd, best["state"]))
        best_epochs.append(best["epoch"])
    pmfs = N.predict_pmf(nets, Xf_s)
    tk, tp = N.tail_dist(train)
    rem_pairs = [(h, a) for _, h, a in REM]
    teams, fin_pos, fin_pts = N.simulate_nn(pmfs, rem_pairs, CUR, N_SIMS, np.random.default_rng(42), tk, tp)
    table = {}
    for i, t in enumerate(teams):
        col = fin_pos[:, i]
        dist = {str(p): float((col == p).mean()) for p in range(1, 21)}
        table[t] = {
            "exp_pos": float(col.mean()), "sd_pos": float(col.std()),
            "pos_p10": float(np.percentile(col, 10)), "pos_p90": float(np.percentile(col, 90)),
            "exp_pts": float(fin_pts[:, i].mean()),
            "p_champion": float((col == 1).mean()), "p_top4": float((col <= 4).mean()),
            "p_releg": float((col >= 18).mean()), "pos_dist": dist,
        }
    rec = {"mode": mode, "season": "2627", "cutoff": str(CUTOFF.date()),
           "played_matches": len(CUR), "remaining_fixtures": len(REM),
           "protocol": "arm B best-val-checkpoint (chrono 80/20), 5 seeds, 4000 sims, rng 42, 730d decay",
           "best_val_epochs": best_epochs,
           "future_feature_assumptions": "rich form frozen at cutoff; workload from known announced calendar only (no unscheduled Europe knockouts, no domestic cups, no postponements); highrec = 2025-26 aggregate frozen, promoted clubs mean-filled",
           "table": table}
    json.dump(rec, open(os.path.join(OUT, f"forecast_2627_gw10_{mode}.json"), "w"), indent=1)
    order = sorted(teams, key=lambda t: table[t]["exp_pos"])
    for t in order:
        r = table[t]
        print(f"{r['exp_pos']:5.2f} {t:16s} pts {r['exp_pts']:5.1f} top4 {r['p_top4']:.2f} champ {r['p_champion']:.3f} rel {r['p_releg']:.3f}")
    print("epochs:", best_epochs, flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
