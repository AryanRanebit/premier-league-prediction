#!/usr/bin/env python3
"""Download FotMob match-level xG (stats block) for PL seasons 2020-21..2026-27.
Resume-safe: skips match ids already in the output CSV. Usage: download_xg.py SEASON [SEASON...]"""
import json, urllib.request, urllib.parse, sys, os, csv, time
from concurrent.futures import ThreadPoolExecutor

UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl_xg_raw.csv")

def get(u, tries=4):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))

def one(mid):
    md = get(f"https://www.fotmob.com/api/data/matchDetails?matchId={mid}")
    g = md.get("general") or {}
    cov = g.get("coverageLevel")
    xg = None
    for grp in (((md.get("content") or {}).get("stats") or {}).get("Periods", {}).get("All", {}).get("stats")) or []:
        for it in grp.get("stats") or []:
            if it.get("key") == "expected_goals":
                xg = it.get("stats")
    status = md.get("header", {}).get("status", {})
    return {"fotmob_id": mid,
            "utc": g.get("matchTimeUTCDate", ""),
            "home": (g.get("homeTeam") or {}).get("name"),
            "away": (g.get("awayTeam") or {}).get("name"),
            "score": status.get("scoreStr", ""),
            "coverage": cov,
            "xg_home": xg[0] if xg else None,
            "xg_away": xg[1] if xg else None}

def main():
    seasons = sys.argv[1:]
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add(r["fotmob_id"])
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = None
    for s in seasons:
        enc = urllib.parse.quote(s, safe="")
        d = get(f"https://www.fotmob.com/api/data/leagues?id=47&ccode3=IND&season={enc}")
        matches = [m for m in d["fixtures"]["allMatches"] if m.get("status", {}).get("finished")]
        ids = [str(m["id"]) for m in matches if str(m["id"]) not in done]
        print(f"{s}: {len(matches)} finished, {len(ids)} to fetch", flush=True)
        rows = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            for i, r in enumerate(ex.map(one, ids)):
                r["season"] = s
                rows.append(r)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(ids)}", flush=True)
        for r in rows:
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r.keys()))
                if new:
                    w.writeheader()
            w.writerow(r)
        f.flush()
        bad = [r for r in rows if r["xg_home"] is None]
        print(f"{s}: fetched {len(rows)}, missing xG: {len(bad)}", flush=True)
    f.close()

main()
