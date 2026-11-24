import os
import sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("EPL_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(OUT, exist_ok=True)
import epl_nn as N

rich = sys.argv[1] == "rich"
ALL = N.load_all()
hist = N.build_history(ALL)
cur = sorted([m for m in ALL if m["season"] == "2627"], key=lambda x: x["date"])
cutoff = cur[-1]["date"]  # 8 Nov 2026
train = [m for m in ALL if m["date"] <= cutoff]
idx_train = [i for i, m in enumerate(ALL) if m["date"] <= cutoff]
X_all, YH_all, YA_all, _ = N.make_xy(ALL, hist, rich)
Xtr, YHtr, YAtr = X_all[idx_train], YH_all[idx_train], YA_all[idx_train]
w = np.array([0.5 ** (max((cutoff - m["date"]).days, 0) / N.HALF_LIFE) for m in train])
rem = N.remaining_fixtures(cur)
last_i = max(idx_train)
Xf = []
for (h, a) in rem:
    fh = N.features_for(hist, h, cutoff, "home", rich, inclusive=True)
    fa = N.features_for(hist, a, cutoff, "away", rich, inclusive=True)
    Xf.append(fh + fa)
Xtr_s, Xf_s, _, _ = N.fill_and_scale(Xtr, np.array(Xf, dtype=object))
nets = []
curve = None
for seed in range(5):
    net, curve = N.train_net(Xtr_s, YHtr, YAtr, w, seed=seed, epochs=250)
    nets.append(net)
pmfs = N.predict_pmf(nets, Xf_s)
tk, tp = N.tail_dist(train)
teams, fin_pos, fin_pts = N.simulate_nn(pmfs, rem, cur, 20000, np.random.default_rng(7), tk, tp)
out = {}
for i, t in enumerate(teams):
    pos = fin_pos[:, i]
    out[t] = {"exp_pos": float(pos.mean()), "exp_pts": float(fin_pts[:, i].mean()),
              "p_top4": float((pos <= 4).mean()), "p_champion": float((pos == 1).mean()),
              "pos_dist": {int(p): float((pos == p).mean()) for p in range(1, 21)}}
tag = "rich" if rich else "goals"
json.dump({"forecast": out, "loss_curve": curve, "cutoff": str(cutoff.date())},
          open(os.path.join(OUT, f"nn_forecast_{tag}_v3.json"), "w"), indent=1)
c = out["Chelsea"]
print(f"[{tag}] cutoff {cutoff.date()} | CHELSEA exp_pos={c['exp_pos']:.2f} exp_pts={c['exp_pts']:.1f} "
      f"top4={c['p_top4']:.1%} title={c['p_champion']:.2%}")
print("loss curve:", curve[:3], "...", curve[-2:])
