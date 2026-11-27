#!/usr/bin/env python3
"""Tests for autoresearch_pl/run_experiment.py (runner logic uses a stub evaluator, no data needed)
plus one baseline-reproduction check that needs data/ (skipped if absent)."""
import json, os, shutil, sys, tempfile
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "autoresearch_pl"))
import run_experiment as R

n = 0
def ok(name, cond):
    global n
    assert cond, "FAIL " + name
    n += 1; print("PASS", name)

def raises(fn, *a):
    try: fn(*a)
    except Exception: return True
    return False

BEST = {"2122": 3.0, "2223": 3.0, "2324": 3.2, "2425": 3.1, "2526": 2.9}
def per(delta, **over):
    d = {s: BEST[s] - delta for s in BEST}; d.update(over); return d
mean = lambda d: sum(d[s] for s in R.TUNE) / 4
bm = mean(BEST)

# keep/discard threshold
ok("threshold: improvement exactly at margin is kept", R.decide(bm, BEST, bm - R.MARGIN, per(R.MARGIN))[0] == "keep")
ok("threshold: improvement just below margin is discarded", R.decide(bm, BEST, bm - 0.0049, per(0.0049))[0] == "discard")
ok("threshold: worse overall is discarded", R.decide(bm, BEST, bm + 0.01, per(-0.01))[0] == "discard")
# allowed worse-season count
n1 = per(0.03, **{"2122": BEST["2122"] + 0.001})
n2 = per(0.03, **{"2122": BEST["2122"] + 0.001, "2223": BEST["2223"] + 0.001})
ok("one worse tune season is allowed", R.decide(bm, BEST, mean(n1), n1) == ("keep", 1))
ok("two worse tune seasons are discarded", R.decide(bm, BEST, mean(n2), n2) == ("discard", 2))
# 2025-26 never decides
bad_seen = per(0.03, **{"2526": 9.9}); good_seen = per(0.0, **{"2526": 0.1})
ok("a much worse 2025-26 does not block a keep", R.decide(bm, BEST, mean(bad_seen), bad_seen)[0] == "keep")
ok("a much better 2025-26 does not rescue a discard", R.decide(bm, BEST, mean(good_seen), good_seen)[0] == "discard")
mix = per(0.03, **{"2526": 9.9, "2122": BEST["2122"] + 0.001})
ok("2025-26 is not counted as a worse season", R.decide(bm, BEST, mean(mix), mix) == ("keep", 1))
ok("tune seasons exclude 2025-26", "2526" not in R.TUNE and R.SEEN == ["2526"])
# override validation (config whitelist)
base = {"rich": True, "h1": 48, "h2": 24, "lr": 0.005, "l2": 1e-4, "bs": 256, "val_frac": 0.1,
        "patience": 20, "min_epochs": 30, "epochs": 250, "n_seeds": 5, "half_life": 730.0}
ok("override applies and keeps other keys", R.apply_overrides(base, ["l2=1e-3", "h1=32"])["h1"] == 32)
ok("unknown key rejected", raises(R.apply_overrides, base, ["data_dir=/tmp"]))
ok("wrong type rejected", raises(R.apply_overrides, base, ["h1=2.5"]) and raises(R.apply_overrides, base, ["rich=1"]))
ok("out-of-range rejected", raises(R.apply_overrides, base, ["val_frac=0.9"]) and raises(R.apply_overrides, base, ["n_seeds=500"]))
ok("min_epochs > epochs rejected", raises(R.apply_overrides, base, ["epochs=20"]))
ok("int accepted for float key", R.apply_overrides(base, ["lr=1"])["lr"] == 1.0)

# run(): baseline / keep / discard / crash with a stub evaluator
tmp = tempfile.mkdtemp()
cand, log = os.path.join(tmp, "candidate.py"), os.path.join(tmp, "results.tsv")
def write_cand():
    open(cand, "w").write('"""doc"""\nCFG = {\n' + "".join(f"    {k!r}: {v!r},\n" for k, v in base.items()) + "}\n")
write_cand()
def stub(shift, **over):
    def f(cfg):
        d = {s: BEST[s] - shift for s in BEST}; d.update(over)
        d["tune_mean"] = mean(d); d["seen_2526"] = d["2526"]; d["five_mean"] = sum(d[s] for s in BEST) / 5
        return d
    return f
def cur(): ns = {}; exec(open(cand).read(), ns); return ns["CFG"]
st, _, _ = R.run("baseline", [], stub(0), cand, log, "abc1234")
ok("first run is the baseline", st == "baseline")
st, _, _ = R.run("improve", ["h1=32"], stub(0.02), cand, log, "abc1234")
ok("improvement is kept and candidate.py updated", st == "keep" and cur()["h1"] == 32)
before = open(cand).read()
st, _, _ = R.run("no gain", ["h1=64"], stub(0.02), cand, log, "abc1234")
ok("no-gain run is discarded and candidate.py is restored", st == "discard" and open(cand).read() == before)
def boom(cfg): raise RuntimeError("oom\tbad\nline")
st, r, _ = R.run("crash", ["h1=96"], boom, cand, log, "abc1234")
rows = R.read_log(log)
ok("failed run is logged as crash with error and candidate.py unchanged",
   st == "crash" and r is None and rows[-1][7] == "crash" and "RuntimeError" in rows[-1][10] and open(cand).read() == before)
st, _, _ = R.run("after crash", ["h1=16"], stub(0.05), cand, log, "abc1234")
ok("crash does not break comparison against the last kept run", st == "keep" and cur()["h1"] == 16)
ok("rejected override writes no log row", (raises(R.run, "bad", ["nope=1"], stub(0), cand, log) and len(R.read_log(log)) == len(rows) + 1))
rows = R.read_log(log)
ok("every log row has the full 11 columns", all(len(x) == 11 for x in rows) and len(rows) == 5)
ok("each row logs the exact config and its hash", all(json.loads(x[10]).get("h1") and R.cfg_sha({k: v for k, v in json.loads(x[10]).items() if not k.startswith("_")}) == x[2] for x in rows if x[7] != "crash"))
ok("config hash is deterministic and sensitive", R.cfg_sha(base) == R.cfg_sha(dict(base)) and R.cfg_sha(base) != R.cfg_sha(dict(base, l2=0.1)))
shutil.rmtree(tmp)

# locked-file enforcement
root = tempfile.mkdtemp()
for f in R.LOCKED: os.makedirs(os.path.dirname(os.path.join(root, f)), exist_ok=True); open(os.path.join(root, f), "w").write("x")
import hashlib
open(os.path.join(root, "autoresearch_pl", "locked.sha256"), "w").write("".join(f"{hashlib.sha256(b'x').hexdigest()}  {f}\n" for f in R.LOCKED))
R.check_locked(root)
open(os.path.join(root, "src", "epl_nn.py"), "w").write("tampered")
ok("tampering with a locked file is caught", raises(R.check_locked, root))
shutil.rmtree(root)
ok("repo's locked hashes match the files", R.check_locked(os.path.abspath(ROOT)) is None)

# real baseline reproduction (needs data)
if os.path.exists(os.path.join(ROOT, "data", "E0_2526.csv")):
    import evaluate as EV
    r = EV.evaluate(base)
    exp = {"2122": 3.0163, "2223": 2.9904, "2324": 3.2262, "2425": 3.1485, "2526": 2.9509}
    ok("baseline reproduces main's per-season NLL", all(abs(r[s] - exp[s]) < 5e-4 for s in exp))
    ok("evaluate.py agrees with runner on seasons", EV.TUNE == R.TUNE and EV.SEEN == R.SEEN)
else:
    print("SKIP baseline reproduction (no data/)")
print(f"\nAll {n} autoresearch checks passed.")
