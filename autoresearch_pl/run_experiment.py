#!/usr/bin/env python3
"""One experiment = one candidate.py change -> locked evaluation -> keep or discard -> log row.

  python3 autoresearch_pl/run_experiment.py "short description" key=value [key=value ...]

This is a plain config runner. It is NOT an autonomous LLM proposer: a person or
agent chooses each change, one command per experiment. The 10 experiments in
results.tsv were run by hand in a fixed list.

Applies the overrides to the current best CFG in candidate.py (whitelisted keys,
type and range checked), evaluates it, and appends a row to results.tsv. Keep rule
(tune seasons only): tune-mean NLL improves by at least MARGIN and at most MAX_WORSE
tune seasons get worse. A discarded or crashed run leaves candidate.py untouched, so
it always holds the best kept config. 2025-26 is logged, never used for the decision.
The locked files (evaluate.py, src/epl_nn.py) are hash-checked against locked.sha256.
"""
import ast, hashlib, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MARGIN = 0.005
MAX_WORSE = 1
TUNE = ["2122", "2223", "2324", "2425"]
SEEN = ["2526"]
SPEC = {  # key: (type, lo, hi) allowed range, inclusive
    "rich": (bool, None, None), "h1": (int, 2, 512), "h2": (int, 2, 512),
    "lr": (float, 1e-5, 1.0), "l2": (float, 0.0, 1.0), "bs": (int, 8, 4096),
    "val_frac": (float, 0.02, 0.5), "patience": (int, 1, 200), "min_epochs": (int, 1, 250),
    "epochs": (int, 10, 500), "n_seeds": (int, 1, 30), "half_life": (float, 30.0, 5000.0),
}
LOCKED = ["autoresearch_pl/evaluate.py", "src/epl_nn.py"]
HEAD = "id\tgit\tcfg_sha\ttune_mean\tseen_2526\tfive_mean\tworse_tune_seasons\tstatus\tseconds\tdescription\tconfig\n"

def apply_overrides(cfg, sets):
    cfg = dict(cfg)
    for kv in sets:
        if "=" not in kv: raise ValueError(f"expected key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if k not in SPEC: raise ValueError(f"unknown key {k!r}; allowed: {sorted(SPEC)}")
        t, lo, hi = SPEC[k]
        val = ast.literal_eval(v)
        if t is bool:
            if not isinstance(val, bool): raise ValueError(f"{k} must be True/False")
        else:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or (t is int and not isinstance(val, int)):
                raise ValueError(f"{k} must be {t.__name__}")
            val = t(val)
            if not (lo <= val <= hi): raise ValueError(f"{k}={val} outside [{lo}, {hi}]")
        cfg[k] = val
    if cfg["min_epochs"] > cfg["epochs"]: raise ValueError("min_epochs > epochs")
    return cfg

def decide(best_mean, best_per, new_mean, new_per, margin=MARGIN, max_worse=MAX_WORSE):
    """Returns (status, worse_count). Uses TUNE seasons only: SEEN seasons never enter."""
    worse = sum(new_per[s] > best_per[s] + 1e-9 for s in TUNE)
    ok = (best_mean - new_mean) >= margin - 1e-12 and worse <= max_worse
    return ("keep" if ok else "discard"), worse

def cfg_sha(cfg): return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

def write_cfg(path, cfg):
    src = open(path).read().split("CFG = ")[0]
    open(path, "w").write(src + "CFG = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in cfg.items()) + "}\n")

def read_log(path):
    if not os.path.exists(path): return []
    return [l.split("\t") for l in open(path).read().splitlines()[1:]]

def check_locked(root):
    want = dict(l.split()[::-1] for l in open(os.path.join(root, "autoresearch_pl", "locked.sha256")).read().splitlines())
    for f in LOCKED:
        got = hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()
        if want.get(f) != got: raise RuntimeError(f"locked file changed: {f}")

def git_id(root):
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip() or "nogit"
    d = subprocess.run(["git", "status", "--porcelain", "--", "src", "autoresearch_pl/evaluate.py"], capture_output=True, text=True, cwd=root).stdout.strip()
    return r + ("-dirty" if d else "")

def run(desc, sets, evaluate_fn, cand_path, log_path, git="nogit"):
    ns = {}; exec(open(cand_path).read(), ns)
    cfg = apply_overrides(ns["CFG"], sets)
    prev = read_log(log_path)
    kept = [x for x in prev if x[7] in ("baseline", "keep")]
    t = time.time()
    try:
        r = evaluate_fn(cfg)
    except Exception as e:
        status, r, worse = "crash", None, 0
        err = f"{type(e).__name__}: {e}"[:200].replace("\t", " ").replace("\n", " ")
    else:
        if not kept: status, worse = "baseline", 0
        else:
            best_per = json.loads(kept[-1][10])["_per"]
            status, worse = decide(float(kept[-1][3]), best_per, r["tune_mean"], r)
    sec = time.time() - t
    per = {s: r[s] for s in TUNE + SEEN} if r else {}
    rec = json.dumps(dict(cfg, _per=per) if r else dict(cfg, _error=err), sort_keys=False)
    cells = [str(len(prev)), git, cfg_sha(cfg)] + ([f"{r['tune_mean']:.4f}", f"{r['seen_2526']:.4f}", f"{r['five_mean']:.4f}"] if r else ["nan"] * 3)
    cells += [str(worse), status, f"{sec:.0f}", desc.replace("\t", " "), rec]
    if not os.path.exists(log_path): open(log_path, "w").write(HEAD)
    open(log_path, "a").write("\t".join(cells) + "\n")
    if status in ("baseline", "keep"): write_cfg(cand_path, cfg)
    return status, r, worse

if __name__ == "__main__":
    root = os.path.abspath(os.path.join(HERE, ".."))
    check_locked(root)
    import evaluate as EV
    assert EV.TUNE == TUNE and EV.SEEN == SEEN
    st, r, worse = run(sys.argv[1], sys.argv[2:], EV.evaluate, os.path.join(HERE, "candidate.py"),
                       os.path.join(HERE, "results.tsv"), git_id(root))
    print(st, f"tune_mean={r['tune_mean']:.4f} seen_2526={r['seen_2526']:.4f} worse_tune_seasons={worse}" if r else "(run failed, see log)")
