#!/usr/bin/env python3
"""Match downloaded FotMob xG rows (data/pl_xg_raw.csv, from download_xg.py)
to football-data E0 CSV rows by date (+/-1 day) + score + team alias.
Writes data/pl_xg_matched.csv. Requires the football-data CSVs (see DATA.md).
"""
import os, csv, sys
from datetime import datetime as dt, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("EPL_REPO", os.path.abspath(os.path.join(HERE, "..", "..")))
ALIAS = {'AFC Bournemouth':'Bournemouth','Brighton & Hove Albion':'Brighton','Coventry City':'Coventry',
 'Hull City':'Hull','Ipswich Town':'Ipswich','Leeds United':'Leeds','Leicester City':'Leicester',
 'Luton Town':'Luton','Manchester City':'Man City','Manchester United':'Man United',
 'Newcastle United':'Newcastle','Norwich City':'Norwich','Nottingham Forest':"Nott'm Forest",
 'Tottenham Hotspur':'Tottenham','West Bromwich Albion':'West Brom','West Ham United':'West Ham',
 'Wolverhampton Wanderers':'Wolves'}
norm = lambda n: ALIAS.get(n, n)
SEAS = {'2020/2021':'2021','2021/2022':'2122','2022/2023':'2223','2023/2024':'2324',
        '2024/2025':'2425','2025/2026':'2526','2026/2027':'2627'}

fd = {}
for fs, s in SEAS.items():
    fd[s] = list(csv.DictReader(open(os.path.join(REPO, "data", f"E0_{s}.csv"), encoding="utf-8-sig")))
fm = [r for r in csv.DictReader(open(os.path.join(HERE, "pl_xg_raw.csv"))) if r["xg_home"]]

out, unmatched = [], []
for r in fm:
    d = dt.fromisoformat(r["utc"].replace("Z", "+00:00")).date()
    h, a = norm(r["home"]), norm(r["away"])
    sh, sa = [int(x) for x in r["score"].split("-")]
    hit = None
    for cand in fd[SEAS[r["season"]]]:
        cd = dt.strptime(cand["Date"], "%d/%m/%Y").date()
        if abs((cd - d).days) <= 1 and cand["HomeTeam"] == h and cand["AwayTeam"] == a \
           and int(cand["FTHG"]) == sh and int(cand["FTAG"]) == sa:
            hit = cand; break
    if hit is None:
        unmatched.append(r)
    else:
        out.append({"season": SEAS[r["season"]], "date": hit["Date"], "home": h, "away": a,
                    "xg_home": r["xg_home"], "xg_away": r["xg_away"], "fotmob_id": r["fotmob_id"]})
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
w = csv.DictWriter(open(os.path.join(HERE, "data", "pl_xg_matched.csv"), "w", newline=""),
                   fieldnames=["season","date","home","away","xg_home","xg_away","fotmob_id"])
w.writeheader(); w.writerows(out)
print(f"matched {len(out)}/{len(fm)}; unmatched {len(unmatched)}")
if unmatched:
    sys.exit("unmatched rows - inspect before proceeding")
