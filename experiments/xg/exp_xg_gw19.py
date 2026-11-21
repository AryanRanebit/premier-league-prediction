#!/usr/bin/env python3
"""xG feature experiment, GW19 official cutoffs, ARM B ONLY (matched protocol).

Question: do causal rolling expected-goals features add skill on top of the
rich feature set? Fair-isolation design: xG (FotMob stats-block match xG) only
exists 2020-21 onward (100% census: 2380/2380 football-data matches matched,
see pl_xg_matched.csv), so BOTH arms train on the IDENTICAL shortened history
(2020-21 onward). The full-history rich baseline (1.598/1.864/3.034) is NOT
the control; it is reported separately as context.

Modes:
  shortrich : rich features only, training from 2020-21.
  shortxg   : rich + rolling xG features (per team: xG for/against last-5,
              last-10, venue last-5; strictly pre-match; matches without xG
              are simply absent from the xG window), training from 2020-21.

Same arm-B protocol: chrono 80/20 early stopping, 5 seeds, 730d decay,
4000 sims, rng 42, frozen-cutoff remainder NLL. Same five reused validation
seasons - no fresh-holdout claim.

Usage: python3 exp_xg_gw19.py MODE [SEASON...]
"""
import os, sys, json, csv, bisect
import numpy as np
from datetime import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "euro_stylec")))
import exp_context_gw100 as C
import epl_nn as N
import exp_style_gw19 as S

OUT = C.OUT
OFFICIAL_GW19_END = S.OFFICIAL_GW19_END
N_SEEDS = 5
EPOCHS = 250
XG_START = dt(2020, 8, 1)

# ---- match-level xG keyed by (season, date, home, away) ----
XG = {}
_xg_path = os.path.join(HERE, "data", "pl_xg_matched.csv")
if not os.path.exists(_xg_path):
    raise SystemExit("Missing data/pl_xg_matched.csv - vendor xG not redistributed; regenerate locally: "
                     "python3 download_xg.py 2020/2021 2021/2022 2022/2023 2023/2024 2024/2025 2025/2026 2026/2027 "
                     "then python3 match_xg.py (see experiments/README.md).")
for r in csv.DictReader(open(_xg_path)):
    d = dt.strptime(r["date"], "%d/%m/%Y")
    XG[(r["season"], d.date(), r["home"], r["away"])] = (float(r["xg_home"]), float(r["xg_away"]))

# ---- per-team xG history: team -> sorted [(date, venue, xgf, xga)] ----
XGH = {}
for m in C.ALL:
    key = (m["season"], m["date"].date(), m["home"], m["away"])
    if key not in XG:
        continue
    xh, xa = XG[key]
    XGH.setdefault(m["home"], []).append((m["date"], "home", xh, xa))
    XGH.setdefault(m["away"], []).append((m["date"], "away", xa, xh))
for t in XGH:
    XGH[t].sort()

def xg_feats(team, d, venue, inclusive=False):
    rows = XGH.get(team, [])
    dates = [r[0] for r in rows]
    i = bisect.bisect_right(dates, d) if inclusive else bisect.bisect_left(dates, d)
    rows = rows[:i]
    f = []
    for n in (5, 10):
        w = rows[-n:]
        f.append(sum(r[2] for r in w) / len(w) if w else None)
        f.append(sum(r[3] for r in w) / len(w) if w else None)
    vrows = [r for r in rows if r[1] == venue][-5:]
    f.append(sum(r[2] for r in vrows) / len(vrows) if vrows else None)
    f.append(sum(r[3] for r in vrows) / len(vrows) if vrows else None)
    return f

def make_xy(matches, hist_map, mode):
    X, YH, YA = [], [], []
    for m in matches:
        fh = N.features_for(hist_map, m["home"], m["date"], "home", True)
        fa = N.features_for(hist_map, m["away"], m["date"], "away", True)
        if mode == "shortxg":
            fh += xg_feats(m["home"], m["date"], "home")
            fa += xg_feats(m["away"], m["date"], "away")
        X.append(fh + fa)
        YH.append(min(m["hg"], 6)); YA.append(min(m["ag"], 6))
    return np.array(X, dtype=object), np.array(YH), np.array(YA)

def fixture_x(hist_map, h, a, cutoff, mode):
    fh = N.features_for(hist_map, h, cutoff, "home", True, inclusive=True)
    fa = N.features_for(hist_map, a, cutoff, "away", True, inclusive=True)
    if mode == "shortxg":
        fh += xg_feats(h, cutoff, "home", inclusive=True)
        fa += xg_feats(a, cutoff, "away", inclusive=True)
    return fh + fa

def run_season(s, mode, hist_map, X_all, YH_all, YA_all, idx_of):
    ALL = C.ALL
    sm = sorted([m for m in ALL if m["season"] == s], key=lambda x: x["date"])
    cd = dt.strptime(OFFICIAL_GW19_END[s], "%Y-%m-%d")
    cut = [m for m in sm if m["date"] <= cd]
    rest = [m for m in sm if m["date"] > cd]
    train = [m for m in ALL if m["date"] <= cd and m["date"] >= XG_START]
    tra, val = S.chrono_split(train)
    wa = np.array([0.5 ** (max((cd - m["date"]).days, 0) / N.HALF_LIFE) for m in tra])
    iia = [idx_of[id(m)] for m in tra]; iib = [idx_of[id(m)] for m in val]
    Xa, YHa, YAa = X_all[iia], YH_all[iia], YA_all[iia]
    Xb, YHb, YAb = X_all[iib], YH_all[iib], YA_all[iib]
    rem = N.remaining_fixtures(cut)
    Xf = np.array([fixture_x(hist_map, h, a, cd, mode) for h, a in rem], dtype=object)

    Xa_s, Xb_s, meansB, stdsB = S.safe_fill_and_scale(Xa, Xb)
    Xf_sB = (np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]
                       for r in Xf], dtype=float) - meansB) / stdsB

    def scaled_feat(h, a):
        r = fixture_x(hist_map, h, a, cd, mode)
        A = np.array([[meansB[c] if r[c] is None else r[c] for c in range(len(meansB))]], dtype=float)
        return (A - meansB) / stdsB

    netsB, best_epochs = [], []
    for sd in range(N_SEEDS):
        _, best = S.train_with_val(Xa_s, YHa, YAa, wa, Xb_s, YHb, YAb, sd, epochs=EPOCHS)
        netsB.append(S.net_from_state(Xa_s.shape[1], sd, best["state"]))
        best_epochs.append(best["epoch"])
    resB = S.evaluate(netsB, Xf_sB, rem, cut, train, rest, sm, scaled_feat)

    rec = {"season": s, "mode": mode, "cutoff": str(cd.date()), "arm": "B_bestckpt80_only",
           "train_from": str(XG_START.date()), "train_n": len(train),
           "best_val_epochs": best_epochs, "arm_B_bestckpt80": resB}
    json.dump(rec, open(os.path.join(OUT, f"xg_{mode}_gw19_{s}.json"), "w"), indent=1)
    print(f"{mode} {s}: train {len(train)} | big6 {resB['mae_big6']:.3f} | all20 {resB['mae_all']:.3f} | "
          f"nll {resB['score_nll']:.3f} | epochs {best_epochs}", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1]
    assert mode in ("shortrich", "shortxg")
    seasons = sys.argv[2:] or list(OFFICIAL_GW19_END)
    hist_map = N.build_history(C.ALL)
    X_all, YH_all, YA_all = make_xy(C.ALL, hist_map, mode)
    idx_of = {id(m): i for i, m in enumerate(C.ALL)}
    for s in seasons:
        run_season(s, mode, hist_map, X_all, YH_all, YA_all, idx_of)
