#!/usr/bin/env python3
"""One experiment = one candidate.py change -> locked evaluation -> keep or discard -> log row.

  python3 autoresearch_pl/run_experiment.py "short description" key=value [key=value ...]

Applies the overrides to the current best CFG in candidate.py, evaluates it, and
appends a row to results.tsv. Keep rule (tune seasons only): tune-mean NLL improves
by at least MARGIN and at most one tune season gets worse. A discarded change is
reverted, so candidate.py always holds the best kept config. 2025-26 is logged,
never used for the decision.
"""
import ast, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as EV
MARGIN = 0.005
LOG = os.path.join(HERE, "results.tsv")
HEAD = "id\tcommit\ttune_mean\tseen_2526\tfive_mean\tworse_tune_seasons\tstatus\tseconds\tdescription\tconfig\n"

def write_cfg(cfg):
    src = open(os.path.join(HERE, "candidate.py")).read().split("CFG = ")[0]
    open(os.path.join(HERE, "candidate.py"), "w").write(src + "CFG = {\n" + "".join(f"    {k!r}: {v!r},\n" for k,v in cfg.items()) + "}\n")

def rows():
    if not os.path.exists(LOG): return []
    return [l.rstrip("\n").split("\t") for l in open(LOG).read().splitlines()[1:]]

if __name__ == "__main__":
    desc, sets = sys.argv[1], sys.argv[2:]
    from candidate import CFG
    cfg = dict(CFG)
    for kv in sets:
        k, v = kv.split("=", 1)
        if k not in cfg: sys.exit(f"unknown key {k}")
        cfg[k] = ast.literal_eval(v)
    t = time.time(); r = EV.evaluate(cfg); sec = time.time() - t
    prev = rows(); kept = [x for x in prev if x[6] in ("baseline", "keep")]
    if not prev:
        status, worse = "baseline", 0
    else:
        best = kept[-1]; best_per = json.loads(best[9])["_per"]
        worse = sum(r[s] > best_per[s] + 1e-9 for s in EV.TUNE)
        better = float(best[2]) - r["tune_mean"] >= MARGIN and worse <= 1
        status = "keep" if better else "discard"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    rec = dict(cfg, _per={s: r[s] for s in EV.TUNE + EV.SEEN})
    if not os.path.exists(LOG): open(LOG, "w").write(HEAD)
    open(LOG, "a").write("\t".join([str(len(prev)), commit, f"{r['tune_mean']:.4f}", f"{r['seen_2526']:.4f}",
        f"{r['five_mean']:.4f}", str(worse), status, f"{sec:.0f}", desc, json.dumps(rec)]) + "\n")
    if status in ("baseline", "keep"): write_cfg(cfg)
    print(status, f"tune_mean={r['tune_mean']:.4f} seen_2526={r['seen_2526']:.4f} worse_tune_seasons={worse}")
