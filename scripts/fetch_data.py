#!/usr/bin/env python3
"""Download Premier League match data (E0.csv) from football-data.co.uk.

Data source: football-data.co.uk (publicly accessible CSVs; reuse terms unresolved, see DATA.md).
This repo does not redistribute the CSVs; run this script to fetch them.
Season codes: 1516 = 2015-16, ..., 2627 = 2026-27.
"""
import os, sys, time, urllib.request

SEASONS = ["1516","1617","1718","1819","1920","2021","2122","2223","2324","2425","2526","2627"]
URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def main():
    os.makedirs(DEST, exist_ok=True)
    for code in SEASONS:
        path = os.path.join(DEST, f"E0_{code}.csv")
        if os.path.exists(path) and "--force" not in sys.argv:
            print(f"skip E0_{code}.csv (exists; use --force to re-download)")
            continue
        req = urllib.request.Request(URL.format(code=code), headers={"User-Agent": "pl-prediction-research/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
        print(f"fetched E0_{code}.csv")
        time.sleep(1)  # be polite to the host
    print("done. Data (c) football-data.co.uk - attribution required, see DATA.md")

if __name__ == "__main__":
    main()
