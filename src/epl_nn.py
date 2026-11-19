import os
#!/usr/bin/env python3
"""v2: strictly-earlier-date features, bias-corrected Adam, frozen-cutoff NLL.
Neural-network challenger for PL match prediction (Query Ka Scene Chelsea video).

Small deep MLP (2 hidden layers) in numpy, two categorical goal heads (0..5, 6+),
cross-entropy loss, time-decay sample weights (same 730d half-life as DC baseline).
Features: rolling pre-match form only (strictly prior matches, cross-season carry-over).
6+ tail: empirical P(g | g>=6) from training history, sampled at simulation time.
Backtest: same 5 seasons, same 100-match cutoffs as epl_model.py.
"""
import csv, math, os, sys, json
import numpy as np
from datetime import datetime

OUT = os.environ.get("EPL_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("EPL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
SEASONS = ["1516","1617","1718","1819","1920","2021","2122","2223","2324","2425","2526","2627"]
TEST_SEASONS = ["2122","2223","2324","2425","2526"]
BIG6 = ["Arsenal","Chelsea","Liverpool","Man City","Man United","Tottenham"]
HALF_LIFE = 730.0

def parse_date(s):
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try: return datetime.strptime(s, fmt)
        except ValueError: pass
    raise ValueError(s)

COLS = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC"]
def load_all():
    rows = []
    for s in SEASONS:
        with open(os.path.join(DATA, f"E0_{s}.csv"), encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if not r.get("HomeTeam") or not r.get("FTHG"): continue
                rows.append({
                    "season": s, "date": parse_date(r["Date"]),
                    "home": r["HomeTeam"].strip(), "away": r["AwayTeam"].strip(),
                    "hg": int(r["FTHG"]), "ag": int(r["FTAG"]),
                    "hs": int(r["HS"]), "as_": int(r["AS"]),
                    "hst": int(r["HST"]), "ast": int(r["AST"]),
                    "hc": int(r["HC"]), "ac": int(r["AC"]),
                })
    rows.sort(key=lambda x: (x["date"], x["season"]))
    return rows

# ---------- rolling pre-match features ----------
# per team stats windows
STAT = ["gf","ga","sf","sa","stf","sta","cf","ca"]  # for/against: goals, shots, SOT, corners

def team_view(m, side):
    """Return this match from side's perspective: stats for/against."""
    if side == "home":
        return {"date": m["date"], "gf": m["hg"], "ga": m["ag"], "sf": m["hs"], "sa": m["as_"],
                "stf": m["hst"], "sta": m["ast"], "cf": m["hc"], "ca": m["ac"]}
    return {"date": m["date"], "gf": m["ag"], "ga": m["hg"], "sf": m["as_"], "sa": m["hs"],
            "stf": m["ast"], "sta": m["hst"], "cf": m["ac"], "ca": m["hc"]}

def build_history(matches):
    """Per team: chronologically ordered list of (date, view)."""
    hist = {}
    for i, m in enumerate(matches):
        for side, t in (("home", m["home"]), ("away", m["away"])):
            v = team_view(m, "home" if side == "home" else "away")
            v["venue"] = side
            hist.setdefault(t, []).append((m["date"], v))
    return hist

def avg(rows, key, n):
    rows = rows[-n:]
    if not rows: return None
    return sum(r[key] for r in rows) / len(rows)

def features_for(hist, team, before_date, venue, rich, inclusive=False):
    """Rolling features from team history strictly before before_date (no same-day
    leakage: same-day matches are excluded for per-match features). For the frozen
    cutoff state pass inclusive=True to include the cutoff date's played matches."""
    rows = [v for (d, v) in hist.get(team, []) if d <= before_date] if inclusive \
        else [v for (d, v) in hist.get(team, []) if d < before_date]
    f = []
    for n in (5, 10):
        for k in (["gf","ga"] if not rich else STAT):
            f.append(avg(rows, k, n))
    vrows = [v for v in rows if v["venue"] == venue]
    f.append(avg(vrows, "gf", 5)); f.append(avg(vrows, "ga", 5))
    f.append(min(len(rows), 30) / 30.0)  # sample-size indicator (0=newly promoted)
    return f  # None entries = no history yet

def make_xy(matches, hist, rich):
    X, YH, YA, W_dates = [], [], [], []
    for i, m in enumerate(matches):
        fh = features_for(hist, m["home"], m["date"], "home", rich)
        fa = features_for(hist, m["away"], m["date"], "away", rich)
        X.append(fh + fa)
        YH.append(min(m["hg"], 6)); YA.append(min(m["ag"], 6))
        W_dates.append(m["date"])
    return np.array(X, dtype=object), np.array(YH), np.array(YA), W_dates

def fill_and_scale(Xtr, Xte):
    """Fill None with training column means, then standardize on training stats."""
    tr = Xtr.astype(float) if Xtr.dtype != object else None
    ncol = len(Xtr[0])
    means = np.zeros(ncol); stds = np.ones(ncol)
    for c in range(ncol):
        col = np.array([r[c] for r in Xtr if r[c] is not None], dtype=float)
        means[c] = col.mean(); stds[c] = col.std() if col.std() > 1e-9 else 1.0
    def conv(X):
        A = np.array([[means[c] if r[c] is None else r[c] for c in range(ncol)] for r in X], dtype=float)
        return (A - means) / stds
    return conv(Xtr), conv(Xte), means, stds

# ---------- tiny deep net ----------
class Net:
    def __init__(self, d, h1=48, h2=24, k=7, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, math.sqrt(2/d), (d, h1)); self.b1 = np.zeros(h1)
        self.W2 = r.normal(0, math.sqrt(2/h1), (h1, h2)); self.b2 = np.zeros(h2)
        self.Wh = r.normal(0, math.sqrt(2/h2), (h2, k)); self.bh = np.zeros(k)
        self.Wa = r.normal(0, math.sqrt(2/h2), (h2, k)); self.ba = np.zeros(k)

    def forward(self, X):
        z1 = np.maximum(X @ self.W1 + self.b1, 0)
        z2 = np.maximum(z1 @ self.W2 + self.b2, 0)
        return z1, z2, z2 @ self.Wh + self.bh, z2 @ self.Wa + self.ba

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def train_net(X, YH, YA, w, seed=0, epochs=250, bs=256, lr=5e-3, l2=1e-4):
    net = Net(X.shape[1], seed=seed)
    params = ["W1","b1","W2","b2","Wh","bh","Wa","ba"]
    m = {p: np.zeros_like(getattr(net, p)) for p in params}
    v = {p: np.zeros_like(getattr(net, p)) for p in params}
    rng = np.random.default_rng(seed + 1000)
    n = len(X); t = 0
    curve = []
    wsum = w.sum()
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            xb, yh, ya, wb = X[idx], YH[idx], YA[idx], w[idx]
            z1, z2, lh, la = net.forward(xb)
            ph, pa = softmax(lh), softmax(la)
            bs_w = wb.sum() + 1e-9
            loss = -(wb * np.log(np.clip(ph[np.arange(len(idx)), yh], 1e-9, None))).sum() / bs_w \
                   -(wb * np.log(np.clip(pa[np.arange(len(idx)), ya], 1e-9, None))).sum() / bs_w
            reg = l2 * sum((getattr(net, p)**2).sum() for p in ["W1","W2","Wh","Wa"])
            # grads
            dh = ph; dh[np.arange(len(idx)), yh] -= 1; dh = (dh * wb[:, None]) / bs_w
            da = pa; da[np.arange(len(idx)), ya] -= 1; da = (da * wb[:, None]) / bs_w
            g = {}
            g["Wh"] = z2.T @ dh + 2*l2*net.Wh; g["bh"] = dh.sum(0)
            g["Wa"] = z2.T @ da + 2*l2*net.Wa; g["ba"] = da.sum(0)
            dz2 = dh @ net.Wh.T + da @ net.Wa.T
            dz2[z2 <= 0] = 0
            g["W2"] = z1.T @ dz2 + 2*l2*net.W2; g["b2"] = dz2.sum(0)
            dz1 = dz2 @ net.W2.T
            dz1[z1 <= 0] = 0
            g["W1"] = xb.T @ dz1 + 2*l2*net.W1; g["b1"] = dz1.sum(0)
            t += 1
            bc1 = 1 - 0.9 ** t
            bc2 = 1 - 0.999 ** t
            for p in params:
                m[p] = 0.9*m[p] + 0.1*g[p]
                v[p] = 0.999*v[p] + 0.001*g[p]**2
                mhat = m[p] / bc1
                vhat = v[p] / bc2
                setattr(net, p, getattr(net, p) - lr * mhat/(np.sqrt(vhat) + 1e-8))
        if ep % 10 == 0 or ep == epochs-1:
            _, _, lh, la = net.forward(X)
            ph, pa = softmax(lh), softmax(la)
            l = -(w * np.log(np.clip(ph[np.arange(n), YH], 1e-9, None))).sum()/wsum \
                -(w * np.log(np.clip(pa[np.arange(n), YA], 1e-9, None))).sum()/wsum
            curve.append((ep, round(float(l), 4)))
    return net, curve

def predict_pmf(nets, X):
    """Average softmax heads over ensemble -> per-match 7x7 scoreline grid."""
    out = []
    for x in X:
        ph = np.zeros(7); pa = np.zeros(7)
        for net in nets:
            _, _, lh, la = net.forward(x[None, :])
            ph += softmax(lh)[0]; pa += softmax(la)[0]
        ph /= len(nets); pa /= len(nets)
        out.append(np.outer(ph, pa))
    return out

# ---------- 6+ tail ----------
def tail_dist(matches):
    """Empirical P(goals=k | goals>=6), k=6..10+, home and away merged."""
    big = [m["hg"] for m in matches if m["hg"] >= 6] + [m["ag"] for m in matches if m["ag"] >= 6]
    cnt = {}
    for g in big: cnt[min(g, 10)] = cnt.get(min(g, 10), 0) + 1
    tot = sum(cnt.values()) or 1
    ks = sorted(cnt)
    return ks, np.array([cnt[k]/tot for k in ks])

def simulate_nn(pmfs, rem, cut_matches, n_sims, rng, tail_ks, tail_p):
    """Same machinery as epl_model.simulate: points, GD, GF ordering."""
    teams = sorted({m["home"] for m in cut_matches} | {m["away"] for m in cut_matches})
    pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
    for m in cut_matches:
        if m["hg"] > m["ag"]: pts[m["home"]] += 3
        elif m["hg"] < m["ag"]: pts[m["away"]] += 3
        else: pts[m["home"]] += 1; pts[m["away"]] += 1
        gd[m["home"]] += m["hg"]-m["ag"]; gd[m["away"]] += m["ag"]-m["hg"]
        gf[m["home"]] += m["hg"]; gf[m["away"]] += m["ag"]
    t_idx = {t: i for i, t in enumerate(teams)}
    nT = len(teams)
    grid7 = [(x, y) for x in range(7) for y in range(7)]
    P = np.tile(np.array([pts[t] for t in teams]), (n_sims, 1)).astype(float)
    G = np.tile(np.array([gd[t] for t in teams]), (n_sims, 1)).astype(float)
    F = np.tile(np.array([gf[t] for t in teams]), (n_sims, 1)).astype(float)
    for k, (h, a) in enumerate(rem):
        p = pmfs[k].ravel(); p = p / p.sum()
        draws = rng.choice(len(grid7), size=n_sims, p=p)
        hg = np.array([grid7[d][0] for d in draws]); ag = np.array([grid7[d][1] for d in draws])
        # expand 6+ via empirical tail
        mh = hg == 6; ma = ag == 6
        if mh.any(): hg = hg.copy(); hg[mh] = rng.choice(tail_ks, size=mh.sum(), p=tail_p)
        if ma.any(): ag = ag.copy(); ag[ma] = rng.choice(tail_ks, size=ma.sum(), p=tail_p)
        ih, ia = t_idx[h], t_idx[a]
        P[:, ih] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        P[:, ia] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        G[:, ih] += hg - ag; G[:, ia] += ag - hg
        F[:, ih] += hg; F[:, ia] += ag
    fin_pos = np.zeros((n_sims, nT), dtype=int); fin_pts = np.zeros((n_sims, nT))
    for s in range(n_sims):
        order = sorted(range(nT), key=lambda t: (-P[s, t], -G[s, t], -F[s, t]))
        for pos, t in enumerate(order, 1):
            fin_pos[s, t] = pos; fin_pts[s, t] = P[s, t]
    return teams, fin_pos, fin_pts

def remaining_fixtures(season_matches):
    played = {(m["home"], m["away"]) for m in season_matches}
    teams = sorted({m["home"] for m in season_matches} | {m["away"] for m in season_matches})
    return [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]

def final_table(sm):
    pts = {}; gd = {}; gf = {}
    for m in sm:
        pts.setdefault(m["home"], 0); pts.setdefault(m["away"], 0)
        gd.setdefault(m["home"], 0); gd.setdefault(m["away"], 0)
        gf.setdefault(m["home"], 0); gf.setdefault(m["away"], 0)
        if m["hg"] > m["ag"]: pts[m["home"]] += 3
        elif m["hg"] < m["ag"]: pts[m["away"]] += 3
        else: pts[m["home"]] += 1; pts[m["away"]] += 1
        gd[m["home"]] += m["hg"]-m["ag"]; gd[m["away"]] += m["ag"]-m["hg"]
        gf[m["home"]] += m["hg"]; gf[m["away"]] += m["ag"]
    order = sorted(pts, key=lambda t: (-pts[t], -gd[t], -gf[t]))
    return {t: i+1 for i, t in enumerate(order)}

def run_backtest(rich, n_sims=4000, epochs=250, n_seeds=5):
    ALL = load_all()
    hist_map = build_history(ALL)
    results = []
    for s in TEST_SEASONS:
        sm = sorted([m for m in ALL if m["season"] == s], key=lambda x: x["date"])
        cut = sm[:100]
        cutoff = cut[-1]["date"]
        train = [m for m in ALL if m["date"] <= cutoff and not (m["season"] == s and m not in cut)]
        idx_train = [i for i, m in enumerate(ALL) if m["date"] <= cutoff and not (m["season"] == s and m not in cut)]
        # features from global history (only entries < match index used, so no leakage)
        X_all, YH_all, YA_all, _ = make_xy(ALL, hist_map, rich)
        Xtr, YHtr, YAtr = X_all[idx_train], YH_all[idx_train], YA_all[idx_train]
        w = np.array([0.5 ** (max((cutoff - m["date"]).days, 0) / HALF_LIFE) for m in train])
        # predict pmfs for remaining fixtures of season s
        rem = remaining_fixtures(cut)
        # find fixture feature rows: build pseudo-matches at cutoff for rem fixtures
        # feature index: use index of the LAST match before cutoff + fixtures have no row;
        # build features by appending placeholder entries
        Xf = []
        for (h, a) in rem:
            fh = features_for(hist_map, h, cutoff, "home", rich, inclusive=True)
            fa = features_for(hist_map, a, cutoff, "away", rich, inclusive=True)
            Xf.append(fh + fa)
        Xf = np.array(Xf, dtype=object)
        Xtr_s, Xf_s, _, _ = fill_and_scale(Xtr, Xf)
        nets, curves = [], None
        for seed in range(n_seeds):
            net, curve = train_net(Xtr_s, YHtr, YAtr, w, seed=seed, epochs=epochs)
            nets.append(net); curves = curve
        pmfs = predict_pmf(nets, Xf_s)
        tk, tp = tail_dist(train)
        teams, fin_pos, fin_pts = simulate_nn(pmfs, rem, cut, n_sims, np.random.default_rng(42), tk, tp)
        actual = final_table(sm)
        exp_pos = {t: float(fin_pos[:, i].mean()) for i, t in enumerate(teams)}
        big6_mae = float(np.mean([abs(exp_pos[t] - actual[t]) for t in BIG6]))
        all_mae = float(np.mean([abs(exp_pos[t] - actual[t]) for t in teams]))
        # log-loss on played remainder (evaluation metric, scoreline NLL)
        rest = sm[100:]
        # NLL with the SAME frozen-cutoff information the simulation uses (matched comparison)
        fr_cache = {}
        ll_sum = 0.0; ll_n = 0
        for m in rest:
            key = (m["home"], m["away"])
            if key not in fr_cache:
                fh = features_for(hist_map, m["home"], cutoff, "home", rich, inclusive=True)
                fa = features_for(hist_map, m["away"], cutoff, "away", rich, inclusive=True)
                _, xr, _, _ = fill_and_scale(Xtr, np.array([fh + fa], dtype=object))
                fr_cache[key] = predict_pmf(nets, xr)[0]
            pmf = fr_cache[key]
            p = pmf[min(m["hg"], 6), min(m["ag"], 6)]
            ll_sum += -math.log(max(p, 1e-9)); ll_n += 1
        per_club = {t: {"exp": exp_pos[t], "act": actual[t]} for t in BIG6 if t in exp_pos}
        results.append({"season": s, "cutoff": str(cutoff.date()),
                        "per_club": per_club,
                        "big6_mae": big6_mae, "all_mae": all_mae,
                        "score_nll": ll_sum / max(ll_n, 1),
                        "chelsea_exp": exp_pos.get("Chelsea"), "chelsea_actual": actual.get("Chelsea"),
                        "loss_curve": curves})
        print(f"{s}: big6_mae={big6_mae:.2f} all_mae={all_mae:.2f} nll={ll_sum/max(ll_n,1):.3f} "
              f"CHE exp={exp_pos.get('Chelsea'):.1f} act={actual.get('Chelsea')}", flush=True)
    return results

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "rich"
    rich = mode == "rich"
    res = run_backtest(rich)
    tag = "rich" if rich else "goals"
    json.dump(res, open(os.path.join(OUT, f"nn_backtest_{tag}_v3.json"), "w"), indent=1)
    print("mean big6:", np.mean([r["big6_mae"] for r in res]).round(2),
          "mean all:", np.mean([r["all_mae"] for r in res]).round(2),
          "mean nll:", np.mean([r["score_nll"] for r in res]).round(3))
