#!/usr/bin/env python3
"""European-workload + FotMob high-recovery features, GW19 official cutoffs,
ARM B ONLY (chrono 80/20, best-validation checkpoint - the PRIMARY arm; the
250-epoch arm is known to overfit, see prior bundles). Rich control is re-run
through this same harness so the comparison is matched by construction.

Modes (features added per team to the repo's rich feature set):
  rich   : control, no extra features
  euro   : 2 workload features computed over PL + UCL/UEL/UECL matches:
           days since previous match, matches in previous 14 days.
           European rows: pl_clubs_european_fixtures_2015_2027.csv,
           status == 'played' ONLY (the 2021-12-09 Spurs-Rennes walkover is
           excluded; scheduled future fixtures excluded). All-English ties
           count once per club (each club physically played the match).
           Domestic cups NOT included - still not full congestion.
  styleC : 1 feature: prior-PL-season "possession won final 3rd per match"
           (FotMob/Opta high recoveries, SEASON-AGGREGATE PRIOR STYLE).
           Source: fotmob_poss_won_final3rd_by_season.csv (fetched 21 Nov
           2026 from data.fotmob.com). For a match in season N the value is
           the FULL season N-1 aggregate - completed before season N starts.
           No current-season aggregate is ever used. Promoted clubs (not in
           prior PL season) -> None -> training-mean fill. Feature exists for
           matches from 2017-18 onward (FotMob coverage starts 2016-17);
           earlier training matches are mean-filled.
  euroC  : euro + styleC together.

Same protocol as exp_style_gw19.py arm B: official GW19 cutoffs, 5 seeds,
730d decay, 4000 sims, rng 42, frozen-cutoff remainder NLL.

Usage: python3 exp_euro_stylec_gw19.py MODE [SEASON...]
"""
import os, sys, json, math, csv, bisect
import numpy as np
from datetime import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "euro_stylec")))
import exp_context_gw100 as C   # ALL matches, stint/event infrastructure
import epl_nn as N
import exp_style_gw19 as S      # shared train/eval helpers (arm B path)

OUT = C.OUT
OFFICIAL_GW19_END = S.OFFICIAL_GW19_END
BIG6 = N.BIG6
N_SEEDS = 5
EPOCHS = 250

# ---------------- FotMob high-recovery (season aggregate, lagged) ----------------
FM2FD = {'AFC Bournemouth':'Bournemouth','Brighton & Hove Albion':'Brighton',
 'Leicester City':'Leicester','Manchester City':'Man City','Manchester United':'Man United',
 'Newcastle United':'Newcastle','Nottingham Forest':"Nott'm Forest",'Tottenham Hotspur':'Tottenham',
 'West Ham United':'West Ham','Wolverhampton Wanderers':'Wolves','Leeds United':'Leeds',
 'West Bromwich Albion':'West Brom','Sheffield United':'Sheffield United',
 'Huddersfield Town':'Huddersfield','Cardiff City':'Cardiff','Norwich City':'Norwich',
 'Stoke City':'Stoke','Swansea City':'Swansea','Luton Town':'Luton','Ipswich Town':'Ipswich',
 'Sunderland':'Sunderland','Hull City':'Hull','Coventry City':'Coventry','Middlesbrough':'Middlesbrough',
 'Watford':'Watford','Fulham':'Fulham','Burnley':'Burnley','Everton':'Everton',
 'Liverpool':'Liverpool','Arsenal':'Arsenal','Chelsea':'Chelsea','Southampton':'Southampton','Brentford':'Brentford',
'Aston Villa':'Aston Villa','Crystal Palace':'Crystal Palace'}

HIGHREC = {}  # season label '2016-17' -> {fd team: value}
_p = os.path.join(HERE, 'fotmob_poss_won_final3rd_by_season.csv')
if not os.path.exists(_p):
    print("WARNING: fotmob_poss_won_final3rd_by_season.csv missing - styleC/euroC modes need it; "
          "vendor data not redistributed, retrieve per experiments/README.md (data.fotmob.com "
          "poss_won_att_3rd_team endpoint, seasons 2016-17..2026-27).", flush=True)
for r in csv.DictReader(open(_p)) if os.path.exists(_p) else []:
    t = FM2FD.get(r['team'])
    if t and r['poss_won_final_3rd_per_match']:
        HIGHREC.setdefault(r['season'], {})[t] = float(r['poss_won_final_3rd_per_match'])

def prior_highrec(team, season_code):
    y1 = 2000 + int(season_code[:2]) - 1
    prev = f"{y1}-{str(y1+1)[2:]}"
    return HIGHREC.get(prev, {}).get(team)

# ---------------- European-inclusive workload events ----------------
EURO_EVENTS = {}  # fd club -> [date,...] (played European matches only)
n_euro = 0
for r in csv.DictReader(open(os.path.join(HERE, 'pl_clubs_european_fixtures_2015_2027.csv'))):
    if r['status'] != 'played':
        continue
    d = dt.strptime(r['date'], '%Y-%m-%d').date()
    for club in r['pl_club'].split('/'):
        EURO_EVENTS.setdefault(club, []).append(d)
        n_euro += 1
for c in EURO_EVENTS:
    EURO_EVENTS[c] = sorted(set(EURO_EVENTS[c]))
print(f"euro events loaded: {n_euro} club-matches for {len(EURO_EVENTS)} clubs", flush=True)

def workload_feats(team, d, inclusive=False):
    """days_since, matches-in-14d over PL + played European fixtures."""
    ds = list(C.TEAM_DATES.get(team, []))
    es = EURO_EVENTS.get(team, [])
    # merge sorted date lists
    all_ds = sorted(ds + es)
    i = bisect.bisect_right(all_ds, d) if inclusive else bisect.bisect_left(all_ds, d)
    prev = all_ds[:i]
    days_since = (d - prev[-1]).days if prev else None
    m14 = sum(1 for md in prev if 0 < (d - md).days <= 14)
    return [days_since, float(m14)]

# ---------------- feature assembly ----------------
def extra_feats(team, d, season, mode, inclusive=False):
    out = []
    if mode in ("euro", "euroC"):
        out += workload_feats(team, d, inclusive=inclusive)
    if mode in ("styleC", "euroC"):
        out.append(prior_highrec(team, season))
    return out

def make_xy(matches, hist_map, mode):
    X, YH, YA = [], [], []
    for m in matches:
        fh = N.features_for(hist_map, m["home"], m["date"], "home", True) + \
             extra_feats(m["home"], m["date"].date(), m["season"], mode)
        fa = N.features_for(hist_map, m["away"], m["date"], "away", True) + \
             extra_feats(m["away"], m["date"].date(), m["season"], mode)
        X.append(fh + fa)
        YH.append(min(m["hg"], 6)); YA.append(min(m["ag"], 6))
    return np.array(X, dtype=object), np.array(YH), np.array(YA)

def fixture_x(hist_map, h, a, cutoff, season, mode):
    fh = N.features_for(hist_map, h, cutoff, "home", True, inclusive=True) + \
         extra_feats(h, cutoff.date(), season, mode, inclusive=True)
    fa = N.features_for(hist_map, a, cutoff, "away", True, inclusive=True) + \
         extra_feats(a, cutoff.date(), season, mode, inclusive=True)
    return fh + fa

def run_season(s, mode, hist_map, X_all, YH_all, YA_all, idx_of):
    ALL = C.ALL
    sm = sorted([m for m in ALL if m["season"] == s], key=lambda x: x["date"])
    cd = dt.strptime(OFFICIAL_GW19_END[s], "%Y-%m-%d")
    cut = [m for m in sm if m["date"] <= cd]
    rest = [m for m in sm if m["date"] > cd]
    train = [m for m in ALL if m["date"] <= cd]
    tra, val = S.chrono_split(train)
    wa = np.array([0.5 ** (max((cd - m["date"]).days, 0) / N.HALF_LIFE) for m in tra])
    iia = [idx_of[id(m)] for m in tra]; iib = [idx_of[id(m)] for m in val]
    Xa, YHa, YAa = X_all[iia], YH_all[iia], YA_all[iia]
    Xb, YHb, YAb = X_all[iib], YH_all[iib], YA_all[iib]
    rem = N.remaining_fixtures(cut)
    Xf = np.array([fixture_x(hist_map, h, a, cd, s, mode) for h, a in rem], dtype=object)

    Xa_s, Xb_s, meansB, stdsB = S.safe_fill_and_scale(Xa, Xb)
    Xf_sB = (np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]
                       for r in Xf], dtype=float) - meansB) / stdsB

    def scaled_feat(h, a):
        r = fixture_x(hist_map, h, a, cd, s, mode)
        A = np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]], dtype=float)
        return (A - meansB) / stdsB

    netsB, best_epochs = [], []
    for sd in range(N_SEEDS):
        _, best = S.train_with_val(Xa_s, YHa, YAa, wa, Xb_s, YHb, YAb, sd, epochs=EPOCHS)
        netsB.append(S.net_from_state(Xa_s.shape[1], sd, best["state"]))
        best_epochs.append(best["epoch"])
    resB = S.evaluate(netsB, Xf_sB, rem, cut, train, rest, sm, scaled_feat)

    teams_s = sorted({m["home"] for m in sm} | {m["away"] for m in sm})
    cov = sum(1 for t in teams_s if prior_highrec(t, s) is not None)
    rec = {"season": s, "mode": mode, "cutoff": str(cd.date()), "arm": "B_bestckpt80_only",
           "highrec_coverage": f"{cov}/20" if mode in ("styleC", "euroC") else None,
           "best_val_epochs": best_epochs, "arm_B_bestckpt80": resB}
    json.dump(rec, open(os.path.join(OUT, f"euro_{mode}_gw19_{s}.json"), "w"), indent=1)
    print(f"{mode} {s}: highrec-cov {cov}/20 | big6 B {resB['mae_big6']:.3f} | "
          f"all B {resB['mae_all']:.3f} | nll B {resB['score_nll']:.3f} | epochs {best_epochs}", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1]
    assert mode in ("rich", "euro", "styleC", "euroC")
    seasons = sys.argv[2:] or list(OFFICIAL_GW19_END)
    hist_map = N.build_history(C.ALL)
    X_all, YH_all, YA_all = make_xy(C.ALL, hist_map, mode)
    idx_of = {id(m): i for i, m in enumerate(C.ALL)}
    for s in seasons:
        run_season(s, mode, hist_map, X_all, YH_all, YA_all, idx_of)
